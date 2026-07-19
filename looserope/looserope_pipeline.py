#from utils import get_naive_mask
from diffusers import FluxKontextPipeline
from typing import Any, Callable, Dict, List, Optional, Union
import os
import time
import torch
from .attention_processor import LooseRoPEFluxAttnProcessor, OUT_IMAGE_ATTN_OFFSET, ATTN_WIDTH, ATTN_HEIGHT
import numpy as np
import wandb
import logging
import cv2
import importlib
import copy
import shutil
import tempfile

from PIL import Image
from .looserope_vlm_mixin import qwen_vl_hf_repo_id, vlm_exceeds_4b, vlm_should_swap_for_verdict
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.pipelines.flux.pipeline_flux_kontext import (
    EXAMPLE_DOC_STRING,
    calculate_shift,
    PREFERRED_KONTEXT_RESOLUTIONS,
    retrieve_timesteps,
)
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

logger = logging.get_logger(__name__)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

class LooseRoPEPipeline(FluxKontextPipeline):
    def set_blending_args(self, blend_start_step, blend_end_step):
        self.blend_start_step = blend_start_step
        self.blend_end_step = blend_end_step

    def set_temp_latent_dir(self, temp_latent_dir):
        self.temp_latent_dir = temp_latent_dir
        os.makedirs(self.temp_latent_dir, exist_ok=True)

    def set_x0_prediction_dir(self, x0_prediction_dir):
        self.x0_prediction_dir = x0_prediction_dir
        os.makedirs(self.x0_prediction_dir, exist_ok=True)

    def set_blend_masks(self, masks):
        self.blend_masks = masks

    def set_metrics_arguments(self, original_latent_dir, input_latent_dir, trimap, crop_mask):
        self.original_latent_dir = original_latent_dir
        self.input_latent_dir = input_latent_dir
        self.trimap = trimap
        self.crop_mask = crop_mask

    def set_attn_processor_to_looserope(self, save_folder, attention_config):
        # Store attention config for later access (e.g., VLM verdict)
        self.attention_config = attention_config

        # Store save_folder and output_folder for VLM verdict logging
        self.save_folder = save_folder
        self.output_folder = os.path.dirname(save_folder)

        cfg_off = getattr(attention_config, "perform_offloading", False)
        self.perform_offloading = vlm_should_swap_for_verdict(attention_config)
        if self.perform_offloading and not cfg_off and vlm_exceeds_4b(getattr(attention_config, "vlm_model_size", "4B")):
            print("[CONFIG] perform_offloading auto-enabled (VLM > 4B)")
        print(f"[CONFIG] perform_offloading effective: {self.perform_offloading}")

        # Apply initial offloading if VLM is enabled and offloading is active
        if hasattr(self, 'vlm_enabled') and self.vlm_enabled and self.perform_offloading:
            print(f"[CONFIG] Offloading VLM to CPU to save GPU memory...")
            self.vlm_model.to("cpu")
            torch.cuda.empty_cache()
            print(f"[CONFIG] ✓ VLM offloaded to CPU")

        layer_num = 0
        for block in self.transformer.transformer_blocks:
            block.attn.set_processor(LooseRoPEFluxAttnProcessor())
            block.attn.processor.set_attention_config(save_folder, layer_num, attention_config)
            layer_num += 1

        for block in self.transformer.single_transformer_blocks:
            block.attn.set_processor(LooseRoPEFluxAttnProcessor())
            block.attn.processor.set_attention_config(save_folder, layer_num, attention_config)
            layer_num += 1

        # Set up x0 prediction saving only when explicitly enabled in the config.
        if attention_config.save_x0_predictions and attention_config.x0_prediction_steps:
            x0_prediction_dir = os.path.join(os.path.dirname(save_folder), "x0_predictions")
            self.set_x0_prediction_dir(x0_prediction_dir)
            self.x0_prediction_steps = attention_config.x0_prediction_steps
        else:
            self.x0_prediction_steps = []
            if hasattr(self, "x0_prediction_dir"):
                delattr(self, "x0_prediction_dir")

    def _initialize_vlm(self, model_size="4B"):
        """
        Load VLM model and processor for verdict generation.
        Handles graceful degradation if dependencies are missing.

        Args:
            model_size: Qwen3-VL variant tag, e.g. "4B", "8B" (Instruct) or
                "8B-Thinking" / "30B-A3B-Thinking" (Thinking checkpoints).
        """
        print(f"\n{'='*80}")
        print(f"[VLM INIT] Starting VLM initialization with model size: {model_size}")
        print(f"{'='*80}\n")

        try:
            # Check if required packages are available
            print("[VLM INIT] Importing transformers and qwen_vl_utils...")
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            print("[VLM INIT] ✓ Imports successful")

            # Detect flash_attention_2 availability
            print("[VLM INIT] Detecting flash_attention_2 availability...")
            _has_fa2 = False
            try:
                _has_fa2 = importlib.util.find_spec("flash_attn") is not None
            except Exception:
                _has_fa2 = False

            _attn_impl = "flash_attention_2" if _has_fa2 else "eager"
            print(f"[VLM INIT] Using attention implementation: {_attn_impl}")

            repo_id = qwen_vl_hf_repo_id(model_size)
            _devmap = "cpu" if vlm_exceeds_4b(model_size) else "auto"
            print(f"[VLM INIT] Loading VLM model: {repo_id} (device_map={_devmap})...")
            logger.info(f"Loading VLM model: {repo_id} with {_attn_impl} attention, device_map={_devmap}")

            # Load VLM model
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                repo_id,
                torch_dtype=torch.bfloat16,
                attn_implementation=_attn_impl,
                device_map=_devmap,
            )
            print(f"[VLM INIT] ✓ VLM model loaded successfully")
            print(f"[VLM INIT] Model device: {self.vlm_model.device}")
            if vlm_exceeds_4b(model_size):
                torch.cuda.empty_cache()
                print("[VLM INIT] VLM >4B: kept on CPU until verdict (frees GPU for diffusion)")

            # Load processor
            print(f"[VLM INIT] Loading VLM processor...")
            self.vlm_processor = AutoProcessor.from_pretrained(repo_id)
            print(f"[VLM INIT] ✓ VLM processor loaded successfully")

            self.vlm_enabled = True
            logger.info("VLM model loaded successfully")

            # Store offloading preference (will be set from config later)
            self.perform_offloading = False

            print(f"\n[VLM INIT] ✓✓✓ VLM initialization COMPLETE ✓✓✓\n")

        except ImportError as e:
            print(f"[VLM INIT] ✗ Import error: {e}")
            logger.warning(f"VLM dependencies not available: {e}. VLM verdict will be disabled.")
            self.vlm_enabled = False
            print(f"[VLM INIT] VLM verdict DISABLED\n")
        except Exception as e:
            print(f"[VLM INIT] ✗ Error loading VLM: {e}")
            logger.error(f"Error loading VLM model: {e}. VLM verdict will be disabled.")
            self.vlm_enabled = False
            print(f"[VLM INIT] VLM verdict DISABLED\n")

    def _log_vlm_verdict(self, verdict, vlm_try_number, timestep):
        """
        Log VLM verdict to a text file in the output folder.

        Args:
            verdict: dict with 'classification' and 'reasoning' keys
            vlm_try_number: current VLM try number
            timestep: the timestep at which the verdict was generated
        """
        if not hasattr(self, 'output_folder') or self.output_folder is None:
            logger.warning("Output folder not set, skipping VLM verdict logging")
            return

        log_file_path = os.path.join(self.output_folder, "vlm_verdicts.txt")

        try:
            # Get current values from layer 0's attention processor
            attn_config = self.transformer.transformer_blocks[0].attn.processor.attention_config
            current_saliency_boost = attn_config.cont_saliency_boost
            current_attn_factor_low = attn_config.cont_in_img_attn_factor_low
            current_shrink_factor_low = attn_config.cont_shrink_factor_low
            current_attn_factors = attn_config.cont_in_img_attn_factors if attn_config.cont_mode else None
            current_shrink_factors = attn_config.cont_shrink_factors if attn_config.cont_mode else None

            # Create timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Prepare log entry
            saliency_boost_str = f"{current_saliency_boost:.4f}"
            attn_factor_low_str = f"{current_attn_factor_low:.4f}"
            shrink_factor_low_str = f"{current_shrink_factor_low:.4f}"
            attn_factors_str = ", ".join([f"{f:.4f}" for f in current_attn_factors]) if current_attn_factors is not None else "N/A"
            shrink_factors_str = ", ".join([f"{f:.4f}" for f in current_shrink_factors]) if current_shrink_factors is not None else "N/A"

            log_entry = f"""
{'='*80}
Timestamp: {timestamp}
VLM Try: {vlm_try_number}
Timestep: {timestep}
Classification: {verdict['classification']}
Reasoning: {verdict['reasoning']}
Current Saliency Boost: {saliency_boost_str}
Current Attn Factor Low: {attn_factor_low_str}
Current Attn Factors: [{attn_factors_str}]
Current Shrink Factor Low: {shrink_factor_low_str}
Current Shrink Factors: [{shrink_factors_str}]
{'='*80}

"""

            # Start a new log for the first verdict and append any retries.
            mode = 'w' if vlm_try_number == 0 else 'a'
            with open(log_file_path, mode, encoding='utf-8') as f:
                f.write(log_entry)

            print(f"[VLM LOG] ✓ Verdict logged to: {log_file_path} (saliency_boost={saliency_boost_str}, attn_factor_low={attn_factor_low_str}, shrink_factor_low={shrink_factor_low_str})")

        except Exception as e:
            logger.error(f"Error logging VLM verdict to file: {e}")
            print(f"[VLM LOG] ✗ Failed to log verdict: {e}")

    def _load_vlm_context_examples(self, context_folder=None, use_simplified_instruction=False):
        """
        Load context examples (success/neglect) for VLM few-shot classification.

        Args:
            context_folder: Optional folder containing context examples.
                          If None, uses default folder: vlm_in_context_inputs
            use_simplified_instruction: If True, few-shot images are not loaded.
        """
        print(f"\n{'='*80}")
        print(f"[VLM CONTEXT] Loading VLM context examples...")
        print(f"{'='*80}\n")

        if not self.vlm_enabled:
            print(f"[VLM CONTEXT] ✗ VLM not enabled, skipping context example loading")
            logger.warning("VLM not enabled, skipping context example loading")
            return

        if use_simplified_instruction:
            print("[VLM CONTEXT] use_simplified_instruction: skipping few-shot example images")
            print(f"{'='*80}\n")
            return

        print(f"[VLM CONTEXT] VLM is enabled, proceeding with context loading")

        # Get timestep from attention config (default to 2)
        timestep = 2
        if hasattr(self, 'attention_config') and self.attention_config is not None:
            timestep = self.attention_config.vlm_verdict_timestep
        print(f"[VLM CONTEXT] Using timestep: {timestep}")

        try:
            if context_folder is None:
                print(f"[VLM CONTEXT] Using default context folder")
                # Use new default folder with consolidated context images
                context_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vlm_in_context_inputs")

                print(f"[VLM CONTEXT] Context folder: {context_folder}")
                print(f"[VLM CONTEXT] Timestep: {timestep}")

                # Success example paths (using new naming convention)
                success_input_path = os.path.join(context_folder, "success_input.png")
                success_x0_path = os.path.join(context_folder, f"success_x0_ts{timestep}.png")

                # Second success example paths
                success2_input_path = os.path.join(context_folder, "success2_input.png")
                success2_x0_path = os.path.join(context_folder, f"success2_x0_ts{timestep}.png")

                # Neglect example paths (using new naming convention)
                neglect_input_path = os.path.join(context_folder, "neglect_input.png")
                neglect_x0_path = os.path.join(context_folder, f"neglect_x0_ts{timestep}.png")

                # Second neglect example paths
                neglect2_input_path = os.path.join(context_folder, "neglect2_input.png")
                neglect2_x0_path = os.path.join(context_folder, f"neglect2_x0_ts{timestep}.png")

                # Suppression example paths (using new naming convention)
                suppression_input_path = os.path.join(context_folder, "suppression_input.png")
                suppression_x0_path = os.path.join(context_folder, f"suppression_x0_ts{timestep}.png")

                # Second suppression example paths
                suppression2_input_path = os.path.join(context_folder, "suppression2_input.png")
                suppression2_x0_path = os.path.join(context_folder, f"suppression2_x0_ts{timestep}.png")
            else:
                print(f"[VLM CONTEXT] Using custom context folder: {context_folder}")
                # Use provided context folder with new naming convention
                success_input_path = os.path.join(context_folder, "success_input.png")
                success_x0_path = os.path.join(context_folder, f"success_x0_ts{timestep}.png")
                success2_input_path = os.path.join(context_folder, "success2_input.png")
                success2_x0_path = os.path.join(context_folder, f"success2_x0_ts{timestep}.png")
                neglect_input_path = os.path.join(context_folder, "neglect_input.png")
                neglect_x0_path = os.path.join(context_folder, f"neglect_x0_ts{timestep}.png")
                neglect2_input_path = os.path.join(context_folder, "neglect2_input.png")
                neglect2_x0_path = os.path.join(context_folder, f"neglect2_x0_ts{timestep}.png")
                suppression_input_path = os.path.join(context_folder, "suppression_input.png")
                suppression_x0_path = os.path.join(context_folder, f"suppression_x0_ts{timestep}.png")
                suppression2_input_path = os.path.join(context_folder, "suppression2_input.png")
                suppression2_x0_path = os.path.join(context_folder, f"suppression2_x0_ts{timestep}.png")

            # Load images and resize to 512x512
            print(f"[VLM CONTEXT] Loading success input from: {success_input_path}")
            self.vlm_context_success_input = Image.open(success_input_path).convert("RGB").resize((512, 512))
            print(f"[VLM CONTEXT] ✓ Success input loaded and resized ({self.vlm_context_success_input.size})")

            print(f"[VLM CONTEXT] Loading success x0 from: {success_x0_path}")
            self.vlm_context_success_x0 = Image.open(success_x0_path).convert("RGB").resize((512, 512))
            print(f"[VLM CONTEXT] ✓ Success x0 loaded and resized ({self.vlm_context_success_x0.size})")

            print(f"[VLM CONTEXT] Loading success2 input from: {success2_input_path}")
            self.vlm_context_success2_input = Image.open(success2_input_path).convert("RGB").resize((512, 512))
            print(f"[VLM CONTEXT] ✓ Success2 input loaded and resized ({self.vlm_context_success2_input.size})")

            print(f"[VLM CONTEXT] Loading success2 x0 from: {success2_x0_path}")
            self.vlm_context_success2_x0 = Image.open(success2_x0_path).convert("RGB").resize((512, 512))
            print(f"[VLM CONTEXT] ✓ Success2 x0 loaded and resized ({self.vlm_context_success2_x0.size})")

            print(f"[VLM CONTEXT] Loading neglect input from: {neglect_input_path}")
            self.vlm_context_neglect_input = Image.open(neglect_input_path).convert("RGB").resize((512, 512))
            print(f"[VLM CONTEXT] ✓ Neglect input loaded and resized ({self.vlm_context_neglect_input.size})")

            print(f"[VLM CONTEXT] Loading neglect x0 from: {neglect_x0_path}")
            self.vlm_context_neglect_x0 = Image.open(neglect_x0_path).convert("RGB").resize((512, 512))
            print(f"[VLM CONTEXT] ✓ Neglect x0 loaded and resized ({self.vlm_context_neglect_x0.size})")

            print(f"[VLM CONTEXT] Loading neglect2 input from: {neglect2_input_path}")
            self.vlm_context_neglect2_input = Image.open(neglect2_input_path).convert("RGB").resize((512, 512))
            print(f"[VLM CONTEXT] ✓ Neglect2 input loaded and resized ({self.vlm_context_neglect2_input.size})")

            print(f"[VLM CONTEXT] Loading neglect2 x0 from: {neglect2_x0_path}")
            self.vlm_context_neglect2_x0 = Image.open(neglect2_x0_path).convert("RGB").resize((512, 512))
            print(f"[VLM CONTEXT] ✓ Neglect2 x0 loaded and resized ({self.vlm_context_neglect2_x0.size})")

            print(f"[VLM CONTEXT] Loading suppression input from: {suppression_input_path}")
            self.vlm_context_suppression_input = Image.open(suppression_input_path).convert("RGB").resize((512, 512))
            print(f"[VLM CONTEXT] ✓ Suppression input loaded and resized ({self.vlm_context_suppression_input.size})")

            print(f"[VLM CONTEXT] Loading suppression x0 from: {suppression_x0_path}")
            self.vlm_context_suppression_x0 = Image.open(suppression_x0_path).convert("RGB").resize((512, 512))
            print(f"[VLM CONTEXT] ✓ Suppression x0 loaded and resized ({self.vlm_context_suppression_x0.size})")

            print(f"[VLM CONTEXT] Loading suppression2 input from: {suppression2_input_path}")
            self.vlm_context_suppression2_input = Image.open(suppression2_input_path).convert("RGB").resize((512, 512))
            print(f"[VLM CONTEXT] ✓ Suppression2 input loaded and resized ({self.vlm_context_suppression2_input.size})")

            print(f"[VLM CONTEXT] Loading suppression2 x0 from: {suppression2_x0_path}")
            self.vlm_context_suppression2_x0 = Image.open(suppression2_x0_path).convert("RGB").resize((512, 512))
            print(f"[VLM CONTEXT] ✓ Suppression2 x0 loaded and resized ({self.vlm_context_suppression2_x0.size})")

            logger.info(f"Loaded VLM context examples from: {context_folder} (timestep={timestep})")
            print(f"\n[VLM CONTEXT] ✓✓✓ All context examples loaded successfully (timestep={timestep}) ✓✓✓\n")

        except Exception as e:
            print(f"[VLM CONTEXT] ✗ Error loading context examples: {e}")
            logger.error(f"Error loading VLM context examples: {e}. VLM verdict will be disabled.")
            self.vlm_enabled = False
            print(f"[VLM CONTEXT] VLM verdict DISABLED\n")

    def _reset_all_vlm_modified_params(self):
        """Reset all VLM-modified parameters across all attention processors."""
        for block in self.transformer.transformer_blocks:
            if hasattr(block.attn, 'processor'):
                if hasattr(block.attn.processor, 'attention_config'):
                    block.attn.processor.attention_config.reset_all_vlm_modified_params()

        for block in self.transformer.single_transformer_blocks:
            if hasattr(block.attn, 'processor'):
                if hasattr(block.attn.processor, 'attention_config'):
                    block.attn.processor.attention_config.reset_all_vlm_modified_params()

        print(f"[CONFIG RESET] ✓ All VLM-modified parameters reset to original values")

    def _save_all_locality_scores(self):
        """Collect and save locality scores from all attention processors."""
        print(f"[LOCALITY DEBUG] _save_all_locality_scores called in pipeline")
        print(f"[LOCALITY DEBUG] Checking transformer_blocks: {len(self.transformer.transformer_blocks)}")
        print(f"[LOCALITY DEBUG] Checking single_transformer_blocks: {len(self.transformer.single_transformer_blocks)}")

        # Check if any processor has locality scores to save
        processor_with_scores = None
        total_scores = 0
        for i, block in enumerate(self.transformer.transformer_blocks):
            if hasattr(block.attn, 'processor'):
                if hasattr(block.attn.processor, 'locality_scores'):
                    num_scores = len(block.attn.processor.locality_scores)
                    if num_scores > 0:
                        print(f"[LOCALITY DEBUG] Block {i} has {num_scores} locality scores")
                        processor_with_scores = block.attn.processor
                        total_scores += num_scores
                        break

        if processor_with_scores is None:
            # Also check single transformer blocks
            for i, block in enumerate(self.transformer.single_transformer_blocks):
                if hasattr(block.attn, 'processor'):
                    if hasattr(block.attn.processor, 'locality_scores'):
                        num_scores = len(block.attn.processor.locality_scores)
                        if num_scores > 0:
                            print(f"[LOCALITY DEBUG] Single block {i} has {num_scores} locality scores")
                            processor_with_scores = block.attn.processor
                            total_scores += num_scores
                            break

        if processor_with_scores is None:
            print(f"[LOCALITY DEBUG] No locality scores found in any processor, returning early")
            return  # No locality scores to save

        print(f"[LOCALITY DEBUG] Total locality scores found: {total_scores}")

        # Call save_locality_scores on the processor that has the scores
        processor_with_scores.save_locality_scores()

        print(f"[LOCALITY] ✓ Locality scores saved")

    def _set_params_by_verdict(self, verdict):
        """
        Modify attention configuration based on VLM verdict.

        Args:
            verdict: dict with 'classification' and 'reasoning' keys
        """
        print(f"\n{'='*80}")
        print(f"VLM Verdict: {verdict['classification']}")
        print(f"Reasoning: {verdict['reasoning']}")
        print(f"{'='*80}\n")

        # Log to wandb if available
        if hasattr(self, 'attention_config') and self.attention_config.use_wandb:
            try:
                wandb.log({
                    "vlm_classification": verdict['classification'],
                    "vlm_reasoning": verdict['reasoning']
                }, step=2)
            except Exception as e:
                logger.warning(f"Failed to log VLM verdict to wandb: {e}")

        # Get the config file path based on verdict
        classification = verdict['classification'].lower()

        boost_addition = 0.0
        attn_low_addition = 0.0
        shrink_low_addition = 0.0
        if classification == 'success':
            return
        elif classification == 'neglect':
            boost_addition = getattr(self.attention_config, 'vlm_boost_neglect', -0.08)
            attn_low_addition = getattr(self.attention_config, 'vlm_attn_low_neglect', 0.0)
            shrink_low_addition = getattr(self.attention_config, 'vlm_shrink_low_neglect', 0.0)
        elif classification == 'suppression':
            boost_addition = getattr(self.attention_config, 'vlm_boost_suppression', 0.1)
            attn_low_addition = getattr(self.attention_config, 'vlm_attn_low_suppression', 0.0)
            shrink_low_addition = getattr(self.attention_config, 'vlm_shrink_low_suppression', 0.0)
        else:
            print(f"[CONFIG UPDATE] Unknown verdict '{verdict['classification']}', keeping current config and setting verdict to SUCCESS")
            verdict['classification'] = 'success'
            return

        layer_num = 0
        for block in self.transformer.transformer_blocks:
            if hasattr(block.attn, 'processor'):
                if hasattr(block.attn.processor, 'attention_config'):
                    if layer_num == 0:
                        old_boost = block.attn.processor.attention_config.cont_saliency_boost
                        old_attn_low = block.attn.processor.attention_config.cont_in_img_attn_factor_low
                        old_shrink_low = block.attn.processor.attention_config.cont_shrink_factor_low

                        block.attn.processor.attention_config.cont_saliency_boost += boost_addition
                        block.attn.processor.attention_config.cont_in_img_attn_factor_low += attn_low_addition
                        block.attn.processor.attention_config.cont_shrink_factor_low += shrink_low_addition

                        # Regenerate the factors with the new low values
                        block.attn.processor.attention_config.update_cont_in_img_attn_factors()
                        block.attn.processor.attention_config.update_cont_shrink_factors()

                        new_boost = block.attn.processor.attention_config.cont_saliency_boost
                        new_attn_low = block.attn.processor.attention_config.cont_in_img_attn_factor_low
                        new_shrink_low = block.attn.processor.attention_config.cont_shrink_factor_low

                        print(f"[CONFIG UPDATE] Saliency Boost: {old_boost:.4f} -> {new_boost:.4f} (Δ={boost_addition:+.4f})")
                        print(f"[CONFIG UPDATE] Attn Factor Low: {old_attn_low:.4f} -> {new_attn_low:.4f} (Δ={attn_low_addition:+.4f})")
                        print(f"[CONFIG UPDATE] Shrink Factor Low: {old_shrink_low:.4f} -> {new_shrink_low:.4f} (Δ={shrink_low_addition:+.4f})")
                    block.attn.processor.curr_step = 0
            layer_num += 1

        for block in self.transformer.single_transformer_blocks:
            if hasattr(block.attn, 'processor'):
                if hasattr(block.attn.processor, 'attention_config'):
                    block.attn.processor.curr_step = 0
            layer_num += 1

        print(f"[CONFIG UPDATE] ✓ All {layer_num} processors updated")
        print(f"\n{'='*80}")
        print(f"[CONFIG UPDATE] ✓✓✓ Config update COMPLETE ✓✓✓")
        print(f"{'='*80}\n")

    def get_text_offsets(self, prompt):
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        word2idx = {}
        curr_offset = 0
        text_input_ids = text_input_ids[0].tolist()
        for word in prompt.split(" "):
            word2idx[word] = []
            word_id = self.tokenizer_2(
                word,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).input_ids
            word_id = word_id[0].tolist()
            j = 0
            for i in range(len(word_id)):
                if word_id[i] == 0 or word_id[i] == 1:
                    break
                curr_id = text_input_ids.pop(0)
                if word_id[i] == curr_id:
                    word2idx[word].append(i + curr_offset)
                else:
                    raise ValueError(f"Token mismatch")
                j += 1
            curr_offset += j
        return word2idx

    def get_x0_prediction(self, scheduler, noise_pred, latents, height, width):
        sigma_idx = scheduler.step_index
        sigma = scheduler.sigmas[sigma_idx]
        x0 = latents - sigma * noise_pred
        x0 = self._unpack_latents(x0, height, width, self.vae_scale_factor)
        x0 = (x0 / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(x0, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil")
        return image

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        max_area: int = 1024**2,
        _auto_resize: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guidance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with prompt at the expense of lower image quality.

                Guidance-distilled models approximates true classifier-free guidance for `guidance_scale` > 1. Refer to
                the [paper](https://huggingface.co/papers/2210.03142) to learn more.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512):
                Maximum sequence length to use with the `prompt`.
            max_area (`int`, defaults to `1024 ** 2`):
                The maximum area of the generated image in pixels. The height and width will be adjusted to fit this
                area while maintaining the aspect ratio.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        self.timing_data = {
            'timestep_times': [],
            'vlm_times': [],
            'vlm_offload_times': [],
            'x0_prediction_times': [],
        }
        _t_call_start = time.time()

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_height, original_width = height, width
        aspect_ratio = width / height
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        if height != original_height or width != original_width:
            logger.warning(
                f"Generation `height` and `width` have been adjusted to {height} and {width} to fit the model requirements."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        _t_encode_start = time.time()
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
        self.timing_data['prompt_encoding'] = time.time() - _t_encode_start

        # 3. Preprocess image
        _t_preprocess_start = time.time()
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            img = image[0] if isinstance(image, list) else image

            # Cache the PIL image for VLM verdict (before preprocessing converts to tensor)
            self.input_image_pil = img
            print(f"[PIPELINE] ✓ Cached input image for VLM (size: {img.size})")
            image_height, image_width = self.image_processor.get_default_height_width(img)
            aspect_ratio = image_width / image_height
            if _auto_resize:
                # Kontext is trained on specific resolutions, using one of them is recommended
                _, image_width, image_height = min(
                    (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
                )
            image_width = image_width // multiple_of * multiple_of
            image_height = image_height // multiple_of * multiple_of
            image = self.image_processor.resize(image, image_height, image_width)
            image = self.image_processor.preprocess(image, image_height, image_width)

        self.timing_data['image_preprocessing'] = time.time() - _t_preprocess_start

        # 4. Prepare latent variables
        _t_latent_prep_start = time.time()
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents, latent_ids, image_ids = self.prepare_latents(
            image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        if image_ids is not None:
            latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension

        self.timing_data['latent_preparation'] = time.time() - _t_latent_prep_start

        # 5. Prepare timesteps
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
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        _t_denoise_start = time.time()
        vlm_verdict = False
        initial_latents = latents.clone()
        initial_scheduler = copy.deepcopy(self.scheduler)

        vlm_tries = 0
        vlm_max_tries = getattr(self.attention_config, 'vlm_max_tries', 4)
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
                    if image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                    latent_model_input = latents
                    if image_latents is not None:
                        latent_model_input = torch.cat([latents, image_latents], dim=1)
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    _t_transformer_start = time.time()
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    _t_transformer_elapsed = time.time() - _t_transformer_start
                    noise_pred = noise_pred[:, : latents.size(1)]

                    _t_neg_cfg = 0.0
                    if do_true_cfg:
                        if negative_image_embeds is not None:
                            self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                        _t_neg_start = time.time()
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        _t_neg_cfg = time.time() - _t_neg_start
                        neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                        noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    # Generate x0 prediction if requested for this step
                    if hasattr(self, 'x0_prediction_dir') and hasattr(self, 'x0_prediction_steps'):
                        if i in self.x0_prediction_steps:
                            _t_x0_start = time.time()
                            x0_image = self.get_x0_prediction(self.scheduler, noise_pred, latents, height, width)
                            x0_image[0].save(os.path.join(self.x0_prediction_dir, f"x0_step_{i:04d}_vlm_try_{vlm_tries}.png"))
                            self.timing_data['x0_prediction_times'].append({'step': i, 'time': time.time() - _t_x0_start})

                    #########################################################
                    ####                   VLM VERDICT                   ####
                    #########################################################

                    # VLM verdict at configured timestep
                    vlm_timestep = self.attention_config.vlm_verdict_timestep if hasattr(self, 'attention_config') else 2
                    _t_vlm_total = 0.0
                    _t_vlm_offload = 0.0
                    if hasattr(self, 'vlm_enabled') and self.vlm_enabled and i == vlm_timestep and vlm_tries < vlm_max_tries:
                        print(f"\n{'='*80}")
                        print(f"[PIPELINE] Timestep {i} reached - triggering VLM verdict")
                        print(f"{'='*80}\n")

                        # Generate x0 prediction if not already done for this step
                        if not (hasattr(self, 'x0_prediction_steps') and i in self.x0_prediction_steps):
                            print(f"[PIPELINE] Generating x0 prediction for VLM verdict...")
                            _t_x0_vlm_start = time.time()
                            x0_image = self.get_x0_prediction(self.scheduler, noise_pred, latents, height, width)
                            self.timing_data['x0_prediction_times'].append({'step': i, 'time': time.time() - _t_x0_vlm_start, 'note': 'for_vlm'})
                            print(f"[PIPELINE] ✓ X0 prediction generated")
                        else:
                            print(f"[PIPELINE] Using already generated x0 prediction")

                        # Conditionally offload FLUX components to CPU based on perform_offloading flag
                        _t_offload_start = time.time()
                        if self.perform_offloading:
                            print(f"[PIPELINE] Offloading enabled - moving FLUX components to CPU...")
                            print(f"[PIPELINE] Offloading FLUX transformer to CPU...")
                            self.transformer.to("cpu")
                            print(f"[PIPELINE] Offloading VAE to CPU...")
                            self.vae.to("cpu")
                            print(f"[PIPELINE] Offloading text encoders to CPU...")
                            self.text_encoder.to("cpu")
                            if hasattr(self, 'text_encoder_2'):
                                self.text_encoder_2.to("cpu")
                            torch.cuda.empty_cache()
                            print(f"[PIPELINE] ✓ All FLUX components offloaded, GPU memory freed")
                        _t_offload_to_cpu = time.time() - _t_offload_start

                        # Get VLM verdict (it will handle its own GPU placement based on perform_offloading)
                        print(f"[PIPELINE] Calling get_vlm_verdict...")
                        logger.info(f"Running VLM verdict at timestep {i}")

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
                            self.input_image_pil,
                            x0_image[0],
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
                        print(f"[PIPELINE] ✓ VLM verdict received (took {vlm_elapsed_time:.2f} seconds)")

                        # Conditionally move FLUX components back to GPU
                        _t_reload_start = time.time()
                        if self.perform_offloading:
                            print(f"[PIPELINE] Moving FLUX components back to GPU...")
                            self.transformer.to("cuda")
                            self.vae.to("cuda")
                            self.text_encoder.to("cuda")
                            if hasattr(self, 'text_encoder_2'):
                                self.text_encoder_2.to("cuda")
                            torch.cuda.empty_cache()
                            print(f"[PIPELINE] ✓ All FLUX components back on GPU")
                        _t_reload_elapsed = time.time() - _t_reload_start
                        _t_vlm_offload = _t_offload_to_cpu + _t_reload_elapsed

                        # Update config based on verdict
                        print(f"[PIPELINE] Updating config based on verdict...")
                        self._set_params_by_verdict(verdict)
                        print(f"[PIPELINE] ✓ Config update complete\n")

                        # Log the verdict to file (after params update to capture new saliency_boost)
                        self._log_vlm_verdict(verdict, vlm_tries, i)
                        _t_vlm_total = vlm_elapsed_time + _t_vlm_offload
                        self.timing_data['vlm_times'].append({
                            'try': vlm_tries,
                            'step': i,
                            'inference_time': vlm_elapsed_time,
                            'offload_time': _t_vlm_offload,
                            'total_time': _t_vlm_total,
                            'verdict': verdict['classification'],
                        })
                        vlm_verdict = verdict["classification"].lower()
                        if vlm_verdict != "success" and vlm_tries < vlm_max_tries:
                            vlm_tries += 1
                            _t_step_elapsed = time.time() - _t_step_start
                            self.timing_data['timestep_times'].append({
                                'step': i,
                                'total': _t_step_elapsed,
                                'transformer_forward': _t_transformer_elapsed,
                                'neg_cfg_forward': _t_neg_cfg,
                                'vlm_total': _t_vlm_total,
                            })
                            print(f"[PIPELINE] VLM verdict not success, trying again... ({vlm_tries}/{vlm_max_tries})")
                            break

                    _t_step_elapsed = time.time() - _t_step_start
                    self.timing_data['timestep_times'].append({
                        'step': i,
                        'total': _t_step_elapsed,
                        'transformer_forward': _t_transformer_elapsed,
                        'neg_cfg_forward': _t_neg_cfg,
                        'vlm_total': _t_vlm_total,
                    })

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

        self._current_timestep = None
        self.timing_data['denoising_loop'] = time.time() - _t_denoise_start

        _t_decode_start = time.time()
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        self.timing_data['vae_decode'] = time.time() - _t_decode_start

        # Reset all VLM-modified parameters to their original values
        self._reset_all_vlm_modified_params()

        # Save attention locality scores if recording was enabled
        print(f"[LOCALITY DEBUG] About to call _save_all_locality_scores()")
        try:
            self._save_all_locality_scores()
            print(f"[LOCALITY DEBUG] Finished _save_all_locality_scores()")
        except Exception as e:
            print(f"[LOCALITY DEBUG] Exception in _save_all_locality_scores(): {e}")
            import traceback
            traceback.print_exc()

        # Offload all models
        self.maybe_free_model_hooks()

        self.timing_data['total_call'] = time.time() - _t_call_start
        self.timing_data['vlm_tries'] = vlm_tries

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

def calculate_edge_gradient(latents, crop_mask):
    """
    Calculate the average gradient magnitude along the edges of the crop mask.

    Args:
        latents: torch.Tensor of shape (4096, 64) - flattened latent representation
        crop_mask: numpy.ndarray of shape (64, 64) - binary crop mask

    Returns:
        float: average gradient magnitude along mask edges
    """
    # Reshape latents to spatial dimensions (64, 64, 64 channels)
    latents_spatial = latents.reshape(64, 64, 64).float()

    # Convert to numpy for cv2 operations
    latents_np = latents_spatial.detach().cpu().numpy()

    # Find edges of the crop mask using Canny edge detection
    crop_mask_uint8 = (crop_mask * 255).astype(np.uint8)
    edges = cv2.Canny(crop_mask_uint8, 50, 150)
    edge_coords = np.where(edges > 0)

    if len(edge_coords[0]) == 0:
        return 0.0  # No edges found

    # Calculate gradients for each channel using Sobel operators
    gradients = []
    for channel in range(64):
        channel_data = latents_np[:, :, channel]

        # Calculate gradients in x and y directions
        grad_x = cv2.Sobel(channel_data, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel_data, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Extract gradients at edge locations
        edge_gradients = grad_magnitude[edge_coords]
        gradients.extend(edge_gradients)

    # Return average gradient magnitude along edges
    return np.mean(gradients) if gradients else 0.0

def callback_store_latents(pipe, step, timestep, callback_kwargs):
    """
    Callback to print the current step and save the latents to temp_latent_dir.
    """
    latents = callback_kwargs.get("latents", None)
    if latents is not None:
        # Save as .pt file
        save_path = os.path.join(pipe.temp_latent_dir, f"latents_step_{step:04d}.pt")
        torch.save(latents.detach().cpu(), save_path)
    return callback_kwargs

def callback_blend_latents(pipe, step, timestep, callback_kwargs):
    """
    Callback to blend the reconstruction latents with the ones from the inference step.
    """
    if step < pipe.blend_start_step or step > pipe.blend_end_step:
        return callback_kwargs
    curr_phase = step // 7

    latents = callback_kwargs.get("latents", None)
    recon_latent = torch.load(os.path.join(pipe.temp_latent_dir, f"latents_step_{step:04d}.pt"))
    recon_latent = recon_latent.to(latents.device)

    # load mask and convert to torch tensor
    mask = pipe.blend_masks[curr_phase]
    mask = torch.from_numpy(mask).to(latents.device).reshape(1, 64 * 64, 1)

    # perform blending
    #original_latents_shape = latents.shape
    #latents = latents.reshape(latents.shape[0], 64, 64, 64)
    #recon_latent = recon_latent.reshape(recon_latent.shape[0], 64, 64, 64).to(latents.device)
    blended_latents = mask * latents + (1 - mask) * recon_latent
    #blended_latents = blended_latents.reshape(original_latents_shape)
    callback_kwargs["latents"] = blended_latents

    return callback_kwargs

def callback_calculate_metrics(pipe, step, timestep, callback_kwargs):
    """
    Callback to calculate the metrics.
    """
    latents = callback_kwargs.get("latents", None).squeeze()
    recon_latent = torch.load(os.path.join(pipe.input_latent_dir, f"latents_step_{step:04d}.pt")).squeeze().to(latents.device)
    trimap_high_idxs = (torch.Tensor(pipe.trimap).flatten() == 3).nonzero().squeeze()
    crop_mask_idxs = (torch.Tensor(pipe.crop_mask).flatten() == 1).nonzero().squeeze()
    crop_recon_diff = torch.norm(latents[crop_mask_idxs] - recon_latent[crop_mask_idxs], p=2).mean().item()
    trimap_recon_diff = torch.norm(latents[trimap_high_idxs] - recon_latent[trimap_high_idxs], p=2).mean().item()

    # Calculate average gradient along crop mask edges
    edge_gradient = calculate_edge_gradient(latents, pipe.crop_mask)

    wandb.log({
        "crop_recon_diff": crop_recon_diff,
        "trimap_recon_diff": trimap_recon_diff,
        "crop_edge_gradient": edge_gradient,
    }, step=step)
    return callback_kwargs

def parse_vlm_output(output_text):
    """
    Parse VLM output to extract classification and reasoning.

    Args:
        output_text: String output from VLM model

    Returns:
        dict with 'classification' (str) and 'reasoning' (str) keys
    """
    lines = output_text.split('\n')
    classification = 'Unknown'
    reasoning = ''

    for line in lines:
        if 'Classification:' in line:
            # Extract classification (one of: Success / Neglect / Suppression)
            classification = line.split('Classification:')[-1].strip()
            # Clean up any extra text
            for class_name in ['Success', 'Neglect', 'Suppression']:
                if class_name.lower() in classification.lower():
                    classification = class_name
                    break
        elif 'Reasoning:' in line:
            # Extract reasoning
            reasoning = line.split('Reasoning:')[-1].strip()

    # If reasoning is on subsequent lines, capture it
    if not reasoning and 'Reasoning:' in output_text:
        reasoning_start = output_text.index('Reasoning:') + len('Reasoning:')
        reasoning = output_text[reasoning_start:].strip()

    return {
        'classification': classification,
        'reasoning': reasoning if reasoning else 'No reasoning provided'
    }


def vlm_bbox_crop_from_mask(pil_img: Image.Image, crop_mask_np: np.ndarray) -> Image.Image:
    """Tight bbox around mask>0; upsamples the mask to the PIL image size when needed."""
    img_w, img_h = pil_img.size
    arr = np.asarray(crop_mask_np)
    if arr.ndim > 2:
        arr = arr[:, :, 0]
    mh, mw = arr.shape[:2]
    if (mh, mw) != (img_h, img_w):
        bin_mask = (arr > 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(bin_mask).resize((img_w, img_h), resample=Image.NEAREST)
        crop_mask_full = np.array(mask_pil)
    else:
        crop_mask_full = arr
    ys, xs = np.where(crop_mask_full > 0)
    if len(ys) == 0:
        return pil_img.copy()
    return pil_img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def get_vlm_verdict(vlm_model, vlm_processor, input_image, x0_prediction,
                   context_success_input, context_success_x0,
                   context_success2_input, context_success2_x0,
                   context_neglect_input, context_neglect_x0,
                   context_neglect2_input, context_neglect2_x0,
                   context_suppression_input, context_suppression_x0,
                   context_suppression2_input, context_suppression2_x0,
                   perform_offloading=False,
                   use_simplified_instruction=False,
                   crop_mask_np=None,
                   example_folder=None,
                   max_new_tokens: Optional[int] = None):
    """
    Run VLM classification on input and x0 prediction pair.

    Args:
        vlm_model: Loaded Qwen3VL model
        vlm_processor: Loaded Qwen3VL processor
        input_image: PIL Image of the input (crop)
        x0_prediction: PIL Image of x0 prediction
        context_success_input: PIL Image of success example input
        context_success_x0: PIL Image of success example x0
        context_success2_input: PIL Image of second success example input
        context_success2_x0: PIL Image of second success example x0
        context_neglect_input: PIL Image of neglect example input
        context_neglect_x0: PIL Image of neglect example x0
        context_neglect2_input: PIL Image of second neglect example input
        context_neglect2_x0: PIL Image of second neglect example x0
        context_suppression_input: PIL Image of suppression example input
        context_suppression_x0: PIL Image of suppression example x0
        context_suppression2_input: PIL Image of second suppression example input
        context_suppression2_x0: PIL Image of second suppression example x0
        perform_offloading: Whether to move VLM to GPU/CPU for inference (default: False)
        use_simplified_instruction: Few-shot-free prompt with three views (object crop, full input, x0).
        crop_mask_np: Optional ``crop_mask.npy`` array; else loaded from ``example_folder``.
        example_folder: Directory containing ``crop_mask.npy`` when ``crop_mask_np`` is omitted.
        max_new_tokens: Generation cap; if None, uses 512 (simplified) or 1024 (few-shot).

    Returns:
        dict with 'classification' and 'reasoning' keys
    """
    print(f"\n{'='*80}")
    print(f"[VLM VERDICT] Starting VLM verdict generation")
    print(f"[VLM VERDICT] perform_offloading: {perform_offloading}")
    print(f"[VLM VERDICT] use_simplified_instruction: {use_simplified_instruction}")
    print(f"{'='*80}\n")

    if max_new_tokens is None:
        max_new_tokens = 512 if use_simplified_instruction else 1024
    print(f"[VLM VERDICT] max_new_tokens: {max_new_tokens}")

    if use_simplified_instruction:
        mask = crop_mask_np
        if mask is None and example_folder:
            _mp = os.path.join(example_folder, "crop_mask.npy")
            if os.path.isfile(_mp):
                mask = np.load(_mp)
        if mask is None:
            raise ValueError(
                "use_simplified_instruction requires a crop mask: set pipe.crop_mask, pass crop_mask_np, "
                "or set attention_config.example_folder to a folder containing crop_mask.npy"
            )
        input_image = input_image.resize((512, 512))
        x0_prediction = x0_prediction.resize((512, 512))
        print(f"[VLM VERDICT] Resized images to 512x512 (simplified mode)")
        object_tight = vlm_bbox_crop_from_mask(input_image, mask)
        print(f"[VLM VERDICT] Tight crop (paste region only) for image 1: {object_tight.size}")

        _instr = (
            "You will see three images in this order:\n"
            "1) **Pasted region (tight crop)** — paste region from the edit input (object + immediate paste backdrop).\n"
            "2) **Full edit input** — object composited onto the target scene.\n"
            "3) **Early prediction** — the diffusion model's early x0 estimate.\n\n"
            "**Apply this decision order using image 3 (prediction), compared to images 1–2. Stop at the first rule that applies.**\n\n"
            "**Step A — Suppression (check first):** In the prediction, is the main object largely missing, invisible, "
            "or are **significant portions** of it gone compared to image 1? If **yes** → **Suppression**.\n\n"
            "**Step B — Suppression (still):** If Step A was no: are **key parts** of the object missing or clearly "
            "gone (e.g. major structure absent)? If **yes** → **Suppression**.\n\n"
            "**Step C — Neglect:** Only if neither A nor B: inspect the pasted region background. If the "
            "object's **original paste background** is still **clearly** visible as that same backdrop "
            "(for example a solid paste color, a rectangular cut-out edge, or any old crop texture), choose "
            "**Neglect**. Also choose **Neglect** if the old background has merely been replaced by an "
            "unmotivated filler such as a black/white/flat region that does **not** match the surrounding "
            "target scene. **Light blur, softness, or haze is OK** only when the region still reads as the "
            "target scene. If the pasted-region background is clearly old crop background or arbitrary "
            "filler → **Neglect**.\n\n"
            "**Step D — Success:** If you did not choose Suppression or Neglect: the pasted region's old "
            "background has disappeared and been replaced by the scene background that belongs there, "
            "consistent with the target scene around it (it may be a bit blurry or hazy), and the object "
            "remains adequately present → **Success**.\n\n"
            "In **Reasoning**, briefly show you applied A→B→C→D in order (which step decided).\n\n"
            "Be concise. Use exactly this output format:\n"
            "Reasoning: [2–5 short sentences]\n"
            "Classification: [Success / Neglect / Suppression]\n"
        )

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a vision assistant in an image editing pipeline that blends pasted objects into "
                            "a scene. Classify an early diffusion prediction using the **ordered** rules in the user "
                            "message: prefer **Suppression** when the object is missing or key parts are gone; "
                            "prefer **Neglect** only when the **original paste background** is still **clearly** "
                            "visible; do **not** call Neglect for mild haze or blur alone. Otherwise **Success**."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _instr},
                    {"type": "text", "text": "Image 1 — Pasted region (tight crop from mask):"},
                    {"type": "image", "image": object_tight},
                    {"type": "text", "text": "Image 2 — Full edit input (composite on scene):"},
                    {"type": "image", "image": input_image},
                    {"type": "text", "text": "Image 3 — Early diffusion prediction (x0 estimate):"},
                    {"type": "image", "image": x0_prediction},
                ],
            },
        ]
        print(f"[VLM VERDICT] ✓ Messages created (simplified: 3 images)")
    else:
        # Resize query images to 512x512
        input_image = input_image.resize((512, 512))
        x0_prediction = x0_prediction.resize((512, 512))
        print(f"[VLM VERDICT] Resized images to 512x512")

        print(f"[VLM VERDICT] Creating messages for VLM...")

        print(f"[VLM VERDICT] Input image size: {input_image.size}")
        print(f"[VLM VERDICT] X0 prediction size: {x0_prediction.size}")

        # Create messages following the exact structure from vlm.ipynb Cell 19
        messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a visual analyst. Your task is to classify a diffusion model's x0 prediction. You will be given a 'Crop' (the original input) and a 'Prediction' (the x0 estimate). You must classify the blend."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
Crop Background: The background area of the cropped part in the 'Crop' image.
Target Scene: The native scene background that surrounds the pasted crop and should continue behind it.

You must classify using these definitions:

Success: The object from the 'Crop' is in the 'Prediction'. The Crop Background has disappeared and has been replaced by the Target Scene that belongs behind the object. It is not enough for the crop background to become black, white, a flat color, or any other arbitrary filler unless that filler genuinely matches the surrounding Target Scene. It's ok if the object is slightly blurred but not if it is transparent to the point where you see background through it or if distinct parts of it are missing (in comparison to the original Crop).
Neglect: The object is present but the crop still reads as pasted because the original Crop Background remains visible, a rectangular/cutout edge remains visible, or the background behind the object has been replaced by an unrelated filler that does not match the Target Scene.
Look at all cropped areas - failure in at least one of them is sufficient to claim neglect.
Suppression: The object from the 'Crop' is MISSING in the 'Prediction'. If distinct parts of the object are missing, it should also count as suppression.
---
"""
                },
                {
                    "type": "text", "text": "Here is Example 1 (A 'Success'):"
                },
                {
                    "type": "text", "text": "Example 1 'Crop':"
                },
                {
                    "type": "image", "image": context_success_input
                },
                {
                    "type": "text", "text": "Example 1 'Prediction':"
                },
                {
                    "type": "image", "image": context_success_x0
                },
                {
                    "type": "text", "text": "Reasoning: The bulldozer's background is blended into the target scene, and the bulldozer itself is clearly visible. Therefore, the result is SUCCESS.\nClassification: Success"
                },
                {
                    "type": "text", "text": "\n---\nHere is Example 2 (Another 'Success'):"
                },
                {
                    "type": "text", "text": "Example 2 'Crop':"
                },
                {
                    "type": "image", "image": context_success2_input
                },
                {
                    "type": "text", "text": "Example 2 'Prediction':"
                },
                {
                    "type": "image", "image": context_success2_x0
                },
                {
                    "type": "text", "text": "Reasoning: While slightly blurry, the bulldozer is clearly visible while its white background has been removed, indicating success.\nClassification: Success"
                },
                {
                    "type": "text", "text": "\n---\nHere is Example 3 (A 'Neglect'):"
                },
                {
                    "type": "text", "text": "Example 3 'Crop':"
                },
                {
                    "type": "image", "image": context_neglect_input
                },
                {
                    "type": "text", "text": "Example 3 'Prediction':"
                },
                {
                    "type": "image", "image": context_neglect_x0
                },
                {
                    "type": "text", "text": "Reasoning: While the bowling balls background is faded and the ball is clearly visible (which would count as success) the other cropped object's background (the bowling pins) is still distinct and so the overall result is NEGLECT.\nClassification: Neglect"
                },
                {
                    "type": "text", "text": "\n---\nHere is Example 4 (Another 'Neglect'):"
                },
                {
                    "type": "text", "text": "Example 4 'Crop':"
                },
                {
                    "type": "image", "image": context_neglect2_input
                },
                {
                    "type": "text", "text": "Example 4 'Prediction':"
                },
                {
                    "type": "image", "image": context_neglect2_x0
                },
                {
                    "type": "text", "text": "Reasoning: The sphere from the crop is clearly visible, however its background (in beige color) while slightly blurred is still visible around the sphere. So it should count as neglect.\nClassification: Neglect"
                },
                {
                    "type": "text", "text": "\n---\nHere is Example 5 (A 'Suppression'):"
                },
                {
                    "type": "text", "text": "Example 5 'Crop':"
                },
                {
                    "type": "image", "image": context_suppression_input
                },
                {
                    "type": "text", "text": "Example 5 'Prediction':"
                },
                {
                    "type": "image", "image": context_suppression_x0
                },
                {
                    "type": "text", "text": "Reasoning: The third eye is completely missing in the prediction. This indicates SUPPRESSION.\nClassification: Suppression"
                },
                {
                    "type": "text", "text": "\n---\nHere is Example 6 (Another 'Suppression'):"
                },
                {
                    "type": "text", "text": "Example 6 'Crop':"
                },
                {
                    "type": "image", "image": context_suppression2_input
                },
                {
                    "type": "text", "text": "Example 6 'Prediction':"
                },
                {
                    "type": "image", "image": context_suppression2_x0
                },
                {
                    "type": "text", "text": "Reasoning: In comparison to the original image where the full skull is visible in the crop, in this image the top and top left parts of the skull are completely missing, which indicates suppression.\nClassification: Suppression"
                },
                {
                    "type": "text", "text": "\n---\nNow, analyze this new pair:"
                },
                {
                    "type": "text", "text": "Query 'Crop':"
                },
                {
                    "type": "image", "image": input_image
                },
                {
                    "type": "text", "text": "Query 'Prediction':"
                },
                {
                    "type": "image", "image": x0_prediction
                },
                {
                    "type":"text",
                    "text": """
Task:
Based on the definitions and examples, classify this new pair.

Output:
Reasoning: [1 or 2 sentences comparing the 'Query Crop' and 'Query Prediction'. Refer to whether the cropped object is present and whether the crop background was replaced by the surrounding target scene rather than by old crop background or arbitrary filler.]
Classification: [Success / Neglect / Suppression]

"""
                }
            ]
        }
    ]
        print(f"[VLM VERDICT] ✓ Messages created (14 images total: 12 context + 2 query)")

    # Conditionally move VLM to GPU for inference based on perform_offloading flag
    original_device = vlm_model.device
    if perform_offloading:
        print(f"[VLM VERDICT] Offloading enabled - moving VLM model to GPU for inference...")
        vlm_model.to("cuda")
        print(f"[VLM VERDICT] ✓ VLM model on GPU")
    else:
        print(f"[VLM VERDICT] Offloading disabled - VLM already on GPU (device: {vlm_model.device})")

    try:
        # Prepare inputs for VLM
        print(f"[VLM VERDICT] Applying chat template and tokenizing...")
        inputs = vlm_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(vlm_model.device)
        print(f"[VLM VERDICT] ✓ Inputs prepared and moved to device: {vlm_model.device}")

        # Run inference
        print(f"[VLM VERDICT] Running VLM inference (max_new_tokens={max_new_tokens})...")
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens)
        print(f"[VLM VERDICT] ✓ Inference complete")

        print(f"[VLM VERDICT] Decoding output...")
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(f"[VLM VERDICT] ✓ Output decoded")
        print(f"[VLM VERDICT] Raw output: {output_text[0]}")

        # Parse output
        print(f"[VLM VERDICT] Parsing verdict...")
        verdict = parse_vlm_output(output_text[0])
        print(f"[VLM VERDICT] ✓ Verdict parsed: {verdict['classification']}")

    finally:
        # Conditionally move VLM back to CPU based on perform_offloading flag
        if perform_offloading:
            print(f"[VLM VERDICT] Offloading enabled - moving VLM model back to CPU...")
            vlm_model.to("cpu")
            torch.cuda.empty_cache()
            print(f"[VLM VERDICT] ✓ VLM offloaded to CPU, GPU cache cleared")
        else:
            print(f"[VLM VERDICT] Offloading disabled - VLM remains on GPU")

    return verdict
