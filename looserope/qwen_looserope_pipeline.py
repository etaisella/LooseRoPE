"""
QwenLooseRoPEPipeline - extends QwenImageEditPipeline with LooseRoPE attention manipulation
and VLM verdict support.

Mirrors LooseRoPEPipeline's interface so that inference.py can switch between
Kontext and Qwen backends with minimal code changes.
"""

import os
import copy
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable, Tuple

from PIL import Image
from diffusers import QwenImageEditPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_dimensions, calculate_shift, retrieve_timesteps
from .qwen_attention_processor import LooseRoPEQwenAttnProcessor
from .looserope_vlm_mixin import LooseRoPEVLMMixin, vlm_exceeds_4b, vlm_should_swap_for_verdict

try:
    from diffusers.pipelines.qwenimage import QwenImagePipelineOutput
except ImportError:
    QwenImagePipelineOutput = None


class QwenLooseRoPEPipeline(LooseRoPEVLMMixin, QwenImageEditPipeline):
    """
    QwenImageEditPipeline + LooseRoPE attention manipulation + VLM verdict.

    Provides the same API surface as LooseRoPEPipeline:
      - set_attn_processor_to_looserope(save_folder, attention_config)
      - get_text_offsets(prompt)
      - _initialize_vlm / _load_vlm_context_examples  (via mixin)
      - VLM verdict loop inside __call__
    """

    def set_attn_processor_to_looserope(self, save_folder: str, attention_config):
        self.attention_config = attention_config
        self.save_folder = save_folder
        self.output_folder = os.path.dirname(save_folder)

        cfg_off = getattr(attention_config, "perform_offloading", False)
        self.perform_offloading = vlm_should_swap_for_verdict(attention_config)
        if self.perform_offloading and not cfg_off and vlm_exceeds_4b(getattr(attention_config, "vlm_model_size", "4B")):
            print("[CONFIG] perform_offloading auto-enabled (VLM > 4B)")
        print(f"[CONFIG] perform_offloading effective: {self.perform_offloading}")

        if hasattr(self, 'vlm_enabled') and self.vlm_enabled and self.perform_offloading:
            self.vlm_model.to("cpu")
            torch.cuda.empty_cache()

        pos_embed = self.transformer.pos_embed

        layer_num = 0
        for block in self.transformer.transformer_blocks:
            processor = LooseRoPEQwenAttnProcessor()
            processor.set_attention_config(save_folder, layer_num, attention_config, pos_embed)
            block.attn.set_processor(processor)
            layer_num += 1

        if attention_config.save_x0_predictions and attention_config.x0_prediction_steps:
            x0_prediction_dir = os.path.join(os.path.dirname(save_folder), "x0_predictions")
            self.set_x0_prediction_dir(x0_prediction_dir)
            self.x0_prediction_steps = attention_config.x0_prediction_steps
        else:
            self.x0_prediction_steps = []
            if hasattr(self, "x0_prediction_dir"):
                delattr(self, "x0_prediction_dir")

        print(f"[QWEN LOOSEROPE] Set LooseRoPEQwenAttnProcessor on {layer_num} transformer blocks")

    def get_text_offsets(self, prompt: str) -> Dict[str, List[int]]:
        word2idx: Dict[str, List[int]] = {}

        curr_offset = 0
        for word in prompt.split(" "):
            word_ids = self.tokenizer(word, add_special_tokens=False).input_ids
            n_tokens = len(word_ids)
            word2idx[word] = list(range(curr_offset, curr_offset + n_tokens))
            curr_offset += n_tokens

        return word2idx

    def get_x0_prediction(self, scheduler, noise_pred, latents, height, width):
        sigma_idx = scheduler.step_index
        sigma = scheduler.sigmas[sigma_idx]
        x0 = latents - sigma * noise_pred
        x0 = self._unpack_latents(x0, height, width, self.vae_scale_factor)
        x0 = x0.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(x0.device, x0.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            x0.device, x0.dtype
        )
        x0 = x0 / latents_std + latents_mean
        image = self.vae.decode(x0, return_dict=False)[0][:, :, 0]
        image = self.image_processor.postprocess(image, output_type="pil")
        return image

    def set_metrics_arguments(self, original_latent_dir, input_latent_dir, trimap, crop_mask):
        self.original_latent_dir = original_latent_dir
        self.input_latent_dir = input_latent_dir
        self.trimap = trimap
        self.crop_mask = crop_mask

    def set_temp_latent_dir(self, temp_latent_dir):
        self.temp_latent_dir = temp_latent_dir
        os.makedirs(self.temp_latent_dir, exist_ok=True)

    def set_x0_prediction_dir(self, x0_prediction_dir):
        self.x0_prediction_dir = x0_prediction_dir
        os.makedirs(self.x0_prediction_dir, exist_ok=True)

    @torch.no_grad()
    def __call__(
        self,
        image=None,
        prompt=None,
        negative_prompt=None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas=None,
        guidance_scale=None,
        num_images_per_prompt: int = 1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        """
        Overrides QwenImageEditPipeline.__call__ to add the VLM verdict
        retry loop, matching LooseRoPEPipeline's behavior.
        """
        _t_call_start = time.time()
        self.timing_data = {
            'timestep_times': [], 'vlm_times': [], 'x0_prediction_times': [],
            'vlm_tries': 0,
        }

        # ---------- setup (mirrors parent) ----------
        image_size = image[0].size if isinstance(image, list) else image.size
        calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        self.check_inputs(
            prompt, height, width, negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask, negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            image = self.image_processor.resize(image, calculated_height, calculated_width)
            prompt_image = image
            self.input_image_pil = image if isinstance(image, Image.Image) else image[0]
            image = self.image_processor.preprocess(image, calculated_height, calculated_width)
            image = image.unsqueeze(2)
        else:
            prompt_image = image
            self.input_image_pil = None

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        _t_enc_start = time.time()
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=prompt_image, prompt=prompt,
            prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_embeds_mask,
            device=device, num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=prompt_image, prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds, prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device, num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        self.timing_data['prompt_encoding'] = time.time() - _t_enc_start

        _t_latent_start = time.time()
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            image, batch_size * num_images_per_prompt, num_channels_latents,
            height, width, prompt_embeds.dtype, device, generator, latents,
        )
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
            ]
        ] * batch_size
        self.timing_data['latent_preparation'] = time.time() - _t_latent_start

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        # ---------- denoising with VLM verdict ----------
        _t_denoise_start = time.time()
        vlm_verdict = False
        initial_latents = latents.clone()
        initial_scheduler = copy.deepcopy(self.scheduler)

        vlm_tries = 0
        vlm_max_tries = getattr(self.attention_config, 'vlm_max_tries', 4) if hasattr(self, 'attention_config') else 4
        last_step = False

        while not last_step:
            self.timing_data['timestep_times'] = []
            self.timing_data['x0_prediction_times'] = []
            self.scheduler = copy.deepcopy(initial_scheduler)
            self.scheduler.set_begin_index(0)
            self._current_timestep = None
            latents = initial_latents.clone()

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if i == num_inference_steps - 1:
                        last_step = True
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    _t_step_start = time.time()

                    latent_model_input = latents
                    if image_latents is not None:
                        latent_model_input = torch.cat([latents, image_latents], dim=1)

                    timestep_val = t.expand(latents.shape[0]).to(latents.dtype)

                    _t_transformer_start = time.time()
                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep_val / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            encoder_hidden_states=prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_pred[:, : latents.size(1)]
                    _t_transformer_elapsed = time.time() - _t_transformer_start

                    _t_neg_cfg = 0.0
                    if do_true_cfg:
                        _t_neg_start = time.time()
                        with self.transformer.cache_context("uncond"):
                            neg_noise_pred = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep_val / 1000,
                                guidance=guidance,
                                encoder_hidden_states_mask=negative_prompt_embeds_mask,
                                encoder_hidden_states=negative_prompt_embeds,
                                img_shapes=img_shapes,
                                txt_seq_lens=negative_txt_seq_lens,
                                attention_kwargs=self.attention_kwargs,
                                return_dict=False,
                            )[0]
                        neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                        _t_neg_cfg = time.time() - _t_neg_start

                        comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                        noise_pred = comb_pred * (cond_norm / noise_norm)

                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if latents.dtype != latents_dtype:
                        latents = latents.to(latents_dtype)

                    # --- x0 predictions at configured steps ---
                    if hasattr(self, 'x0_prediction_dir') and hasattr(self, 'x0_prediction_steps'):
                        if i in self.x0_prediction_steps:
                            _t_x0_start = time.time()
                            x0_image = self.get_x0_prediction(self.scheduler, noise_pred, latents, height, width)
                            x0_image[0].save(os.path.join(self.x0_prediction_dir, f"x0_step_{i:04d}_vlm_try_{vlm_tries}.png"))
                            self.timing_data['x0_prediction_times'].append({'step': i, 'time': time.time() - _t_x0_start})

                    # --- VLM verdict ---
                    vlm_timestep = self.attention_config.vlm_verdict_timestep if hasattr(self, 'attention_config') else 2
                    _t_vlm_total = 0.0

                    if getattr(self, 'vlm_enabled', False) and i == vlm_timestep and vlm_tries < vlm_max_tries:
                        print(f"\n{'='*80}")
                        print(f"[PIPELINE] Timestep {i} - triggering VLM verdict")
                        print(f"{'='*80}\n")

                        if not (hasattr(self, 'x0_prediction_steps') and i in self.x0_prediction_steps):
                            _t_x0_vlm_start = time.time()
                            x0_image = self.get_x0_prediction(self.scheduler, noise_pred, latents, height, width)
                            self.timing_data['x0_prediction_times'].append({
                                'step': i, 'time': time.time() - _t_x0_vlm_start, 'note': 'for_vlm'
                            })

                            if hasattr(self, 'x0_prediction_dir'):
                                x0_image[0].save(os.path.join(self.x0_prediction_dir, f"x0_step_{i:04d}_vlm_try_{vlm_tries}.png"))

                        _t_offload_start = time.time()
                        if self.perform_offloading:
                            self.transformer.to("cpu")
                            self.vae.to("cpu")
                            self.text_encoder.to("cpu")
                            torch.cuda.empty_cache()
                        _t_offload_to_cpu = time.time() - _t_offload_start

                        from .looserope_pipeline import get_vlm_verdict
                        vlm_start_time = time.time()
                        _ac = self.attention_config if hasattr(self, "attention_config") else None
                        _us = getattr(_ac, "use_simplified_instruction", False) if _ac is not None else False
                        _max_tok = None
                        if _ac is not None:
                            _max_tok = (
                                getattr(_ac, "vlm_max_new_tokens_simplified", 512)
                                if _us
                                else getattr(_ac, "vlm_max_new_tokens", 1024)
                            )
                        verdict = get_vlm_verdict(
                            self.vlm_model, self.vlm_processor,
                            self.input_image_pil, x0_image[0],
                            getattr(self, "vlm_context_success_input", None),
                            getattr(self, "vlm_context_success_x0", None),
                            getattr(self, "vlm_context_success2_input", None),
                            getattr(self, "vlm_context_success2_x0", None),
                            getattr(self, "vlm_context_neglect_input", None),
                            getattr(self, "vlm_context_neglect_x0", None),
                            getattr(self, "vlm_context_neglect2_input", None),
                            getattr(self, "vlm_context_neglect2_x0", None),
                            getattr(self, "vlm_context_suppression_input", None),
                            getattr(self, "vlm_context_suppression_x0", None),
                            getattr(self, "vlm_context_suppression2_input", None),
                            getattr(self, "vlm_context_suppression2_x0", None),
                            perform_offloading=self.perform_offloading,
                            use_simplified_instruction=_us,
                            crop_mask_np=getattr(self, "crop_mask", None),
                            example_folder=getattr(_ac, "example_folder", None) if _ac is not None else None,
                            max_new_tokens=_max_tok,
                        )
                        vlm_elapsed_time = time.time() - vlm_start_time
                        torch.cuda.empty_cache()

                        _t_reload_start = time.time()
                        if self.perform_offloading:
                            self.transformer.to("cuda")
                            self.vae.to("cuda")
                            self.text_encoder.to("cuda")
                            torch.cuda.empty_cache()
                        _t_reload_elapsed = time.time() - _t_reload_start
                        _t_vlm_offload = _t_offload_to_cpu + _t_reload_elapsed

                        self._set_params_by_verdict(verdict)
                        self._log_vlm_verdict(verdict, vlm_tries, i)
                        _t_vlm_total = vlm_elapsed_time + _t_vlm_offload
                        self.timing_data['vlm_times'].append({
                            'try': vlm_tries, 'step': i,
                            'inference_time': vlm_elapsed_time,
                            'offload_time': _t_vlm_offload,
                            'total_time': _t_vlm_total,
                            'verdict': verdict['classification'],
                        })
                        vlm_verdict = verdict["classification"].lower()
                        if vlm_verdict != "success" and vlm_tries < vlm_max_tries:
                            vlm_tries += 1
                            self.timing_data['timestep_times'].append({
                                'step': i, 'total': time.time() - _t_step_start,
                                'transformer_forward': _t_transformer_elapsed,
                                'neg_cfg_forward': _t_neg_cfg, 'vlm_total': _t_vlm_total,
                            })
                            print(f"[PIPELINE] VLM verdict not success, retrying ({vlm_tries}/{vlm_max_tries})")
                            break

                    _t_step_elapsed = time.time() - _t_step_start
                    self.timing_data['timestep_times'].append({
                        'step': i, 'total': _t_step_elapsed,
                        'transformer_forward': _t_transformer_elapsed,
                        'neg_cfg_forward': _t_neg_cfg, 'vlm_total': _t_vlm_total,
                    })

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        self._current_timestep = None
        self.timing_data['denoising_loop'] = time.time() - _t_denoise_start

        # ---------- decode ----------
        _t_decode_start = time.time()
        if output_type == "latent":
            final_image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            final_image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            final_image = self.image_processor.postprocess(final_image, output_type=output_type)
        self.timing_data['vae_decode'] = time.time() - _t_decode_start

        self._reset_all_vlm_modified_params()

        try:
            self._save_all_locality_scores()
        except Exception as e:
            print(f"[LOCALITY] Exception: {e}")

        self.maybe_free_model_hooks()

        self.timing_data['total_call'] = time.time() - _t_call_start
        self.timing_data['vlm_tries'] = vlm_tries

        if not return_dict:
            return (final_image,)

        if QwenImagePipelineOutput is not None:
            return QwenImagePipelineOutput(images=final_image)
        return type('Output', (), {'images': final_image})()
