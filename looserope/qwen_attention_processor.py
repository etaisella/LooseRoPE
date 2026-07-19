"""
Qwen-specific LooseRoPE attention processor.

Mirrors the attention manipulation from LooseRoPEFluxAttnProcessor but adapted for
Qwen's double-stream architecture and complex-valued RoPE.
"""

import math
import os
import torch
import numpy as np
from typing import Optional, Dict, Any, List
from .qwen_positional_embedding import set_custom_vid_freqs_qwen
from .attention_processor import AttentionConfig

try:
    from diffusers.models.transformers.transformer_qwenimage import (
        apply_rotary_emb_qwen,
    )
except ImportError:
    apply_rotary_emb_qwen = None


class LooseRoPEQwenAttnProcessor:
    """
    LooseRoPE attention processor for Qwen's double-stream architecture.

    Applies the same attention manipulations as LooseRoPEFluxAttnProcessor:
    - Custom RoPE with shrink_factor for the crop region (cont_mode)
    - Saliency-based per-bin re-attention
    - Pre-softmax masking of in-image attention
    - Text emphasizing mask
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("LooseRoPEQwenAttnProcessor requires PyTorch 2.0+")
        self.save_folder = None
        self.curr_step = 0
        self._layout_initialized = False

    def set_attention_config(
        self,
        save_folder: str,
        layer_num: int,
        attention_config: "AttentionConfig",
        pos_embed,
    ):
        self.save_folder = save_folder
        self.layer_num = layer_num
        self.attention_config = attention_config
        self.pos_embed = pos_embed

    def _init_layout(self, seq_img: int, seq_txt: int):
        """Lazily initialize sequence layout on the first forward call."""
        self.txt_seq_len = seq_txt
        self.out_img_seq_len = seq_img // 2
        self.in_img_seq_len = seq_img // 2
        self.attn_height = int(math.isqrt(self.out_img_seq_len))
        self.attn_width = self.attn_height
        assert self.attn_height * self.attn_width == self.out_img_seq_len, (
            f"Expected square spatial grid, got {self.out_img_seq_len} tokens"
        )
        self.out_image_offset = seq_txt
        self.in_image_offset = seq_txt + self.out_img_seq_len
        self.total_seq_len = seq_txt + seq_img
        self._layout_initialized = True
        self._custom_rotary_embs_built = False
        if self.layer_num == 0:
            print(f"[QWEN LAYOUT] txt={self.txt_seq_len}, out_img={self.out_img_seq_len}, "
                  f"in_img={self.in_img_seq_len}, grid={self.attn_height}x{self.attn_width}")

    def _build_custom_rotary_embs(self, annealing_factor: float = 1.0):
        """Build custom RoPE embeddings with shrink_factor for cont_mode."""
        config = self.attention_config
        if not config.cont_mode:
            return

        self.custom_rotary_embs = []
        shrink_factors = config.cont_shrink_factors * annealing_factor
        shrink_factors = np.clip(shrink_factors, 0.0, 1.0)

        frame = 1
        height = self.attn_height
        width = self.attn_width

        for i in range(config.cont_N):
            custom_vid_freqs = set_custom_vid_freqs_qwen(
                self.pos_embed,
                frame, height, width,
                shrink_factor_h=shrink_factors[i],
                shrink_factor_w=shrink_factors[i],
                start_h=min(config.start_h, height - 1),
                start_w=min(config.start_w, width - 1),
                end_h=min(config.end_h, height - 1),
                end_w=min(config.end_w, width - 1),
            )
            standard_vid_freqs = self.pos_embed._compute_video_freqs(frame, height, width, 1)
            custom_vid_freqs = custom_vid_freqs.to(standard_vid_freqs.device)
            combined_vid_freqs = torch.cat([custom_vid_freqs, standard_vid_freqs], dim=0)
            self.custom_rotary_embs.append(combined_vid_freqs)

        self._custom_rotary_embs_built = True

    def _build_masks(self):
        """Build unified attention mask and text emphasizing mask for Qwen's layout."""
        config = self.attention_config
        crop_mask_np = config.per_query_mask
        crop_mask_flat = torch.Tensor(crop_mask_np).flatten()

        unified_mask = torch.ones(1, self.total_seq_len, dtype=torch.bfloat16)
        in_image_attn_mask_w_val = torch.ones(self.in_img_seq_len, dtype=torch.float32)

        in_image_attn_mask_np = np.ones((self.attn_height, self.attn_width), dtype=np.float64)
        if hasattr(config, '_loaded_in_image_attn_mask'):
            in_image_attn_mask_np = config._loaded_in_image_attn_mask
        mask_flat = torch.Tensor(in_image_attn_mask_np).flatten()
        in_image_attn_mask_w_val[mask_flat != 0] = config.in_image_mask_val
        unified_mask[0, self.in_image_offset:self.in_image_offset + self.in_img_seq_len] = in_image_attn_mask_w_val.to(torch.bfloat16)
        self._attn_mask = unified_mask

        text_emph = torch.ones(self.total_seq_len, self.total_seq_len, dtype=torch.bfloat16)
        text_emph_mask_w_val = torch.ones(self.in_img_seq_len, dtype=torch.float32)

        text_emph_mask_np = np.ones((self.attn_height, self.attn_width), dtype=np.float64)
        if hasattr(config, '_loaded_text_emphasizing_mask'):
            text_emph_mask_np = config._loaded_text_emphasizing_mask
        te_flat = torch.Tensor(text_emph_mask_np).flatten()
        text_emph_mask_w_val[te_flat != 0] = config.text_emphasizing_mask_val
        text_emph[:self.txt_seq_len, self.in_image_offset:self.in_image_offset + self.in_img_seq_len] = text_emph_mask_w_val.to(torch.bfloat16)
        self._text_emphasizing_mask = text_emph

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("LooseRoPEQwenAttnProcessor requires encoder_hidden_states")

        seq_txt = encoder_hidden_states.shape[1]
        seq_img = hidden_states.shape[1]

        if not self._layout_initialized:
            self._init_layout(seq_img, seq_txt)

        if not self._custom_rotary_embs_built:
            self._build_custom_rotary_embs()

        config = self.attention_config
        curr_step = self.curr_step

        # Compute QKV
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        use_custom_rope = (
            config.cont_mode
            and config.should_use_custom_pos_embed(curr_step, self.layer_num)
            and hasattr(self, 'custom_rotary_embs')
            and len(self.custom_rotary_embs) > 0
        )

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            txt_query_rot = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key_rot = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
            img_query_std = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key_std = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
        else:
            txt_query_rot = txt_query
            txt_key_rot = txt_key
            img_query_std = img_query
            img_key_std = img_key

        if use_custom_rope and image_rotary_emb is not None:
            custom_rotated_img_querys = []
            custom_rotated_img_keys = []
            for i in range(config.cont_N):
                custom_freqs = self.custom_rotary_embs[i].to(img_query.device)
                custom_img_q = apply_rotary_emb_qwen(img_query, custom_freqs, use_real=False)
                custom_img_k = apply_rotary_emb_qwen(img_key, custom_freqs, use_real=False)
                custom_rotated_img_querys.append(custom_img_q)
                custom_rotated_img_keys.append(custom_img_k)

        # Joint attention: [text, image]
        joint_query = torch.cat([txt_query_rot, img_query_std], dim=1)
        joint_key = torch.cat([txt_key_rot, img_key_std], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # B, S, H, D -> B, H, S, D
        joint_query = joint_query.transpose(1, 2)
        joint_key = joint_key.transpose(1, 2)
        joint_value = joint_value.transpose(1, 2)

        L, S = joint_query.size(-2), joint_key.size(-2)
        scale_factor = 1 / math.sqrt(joint_query.size(-1))

        attn_weight = joint_query @ joint_key.transpose(-2, -1) * scale_factor

        # Crop mask indices in joint attention space
        mask_idxs = torch.Tensor(config.per_query_mask).flatten().nonzero().squeeze() + self.in_image_offset

        # Apply cont_mode attention manipulation (saliency-based per-bin re-attention)
        if use_custom_rope and image_rotary_emb is not None:
            cuttoffs = config.cont_cutoffs
            phase = 0
            on_phase_change = False
            for ci, cutoff in enumerate(cuttoffs):
                if curr_step > cutoff:
                    phase = ci + 1
                if curr_step == cutoff:
                    on_phase_change = True

            if on_phase_change and curr_step != 0:
                self._build_custom_rotary_embs(
                    annealing_factor=config.cont_annealing_factor ** (phase + 1)
                )

            diffs_from_one = 1.0 - config.cont_in_img_attn_factors
            if config.anneal_below_zero:
                af = config.cont_annealing_factor ** (phase + 1)
                diffs_from_one = np.where(diffs_from_one < 0, diffs_from_one * af, diffs_from_one)
            shrink_factors = diffs_from_one * (config.cont_step_shrink ** phase)
            cont_in_img_attn_factors = 1.0 - shrink_factors

            saliency = config.saliency * config.cont_saliency_boost
            saliency = np.clip(saliency, 0.0, 1.0)

            per_query_mask = config.per_query_mask
            filtered_saliency = np.copy(saliency)
            filtered_saliency[per_query_mask == 0] = -1

            for i in range(config.cont_N):
                epsilon = 1e-1 * (i == (config.cont_N - 1))
                saliency_flat = torch.Tensor(filtered_saliency).flatten()
                lower = i * (1.0 / config.cont_N)
                upper = (i + 1) * (1.0 / config.cont_N)
                curr_idxs = ((saliency_flat >= lower) & (saliency_flat < (upper + epsilon))).nonzero().squeeze()
                if curr_idxs.dim() == 0 or curr_idxs.numel() == 0:
                    continue

                curr_joint_idxs = curr_idxs + self.out_image_offset

                custom_q = custom_rotated_img_querys[i].transpose(1, 2)
                custom_k = custom_rotated_img_keys[i].transpose(1, 2)

                custom_joint_q = torch.cat([txt_query_rot.transpose(1, 2), custom_q], dim=2)
                custom_joint_k = torch.cat([txt_key_rot.transpose(1, 2), custom_k], dim=2)

                indexed_q = custom_joint_q[:, :, curr_joint_idxs, :]
                curr_attn = indexed_q @ custom_joint_k.transpose(-2, -1) * scale_factor
                curr_attn[:, :, :, mask_idxs] = curr_attn[:, :, :, mask_idxs] * cont_in_img_attn_factors[i]

                attn_weight[:, :, curr_joint_idxs, self.in_image_offset:] = curr_attn[:, :, :, self.in_image_offset:]

        # Pre-softmax masking
        if config.should_apply_masking(curr_step, self.layer_num):
            if not hasattr(self, '_attn_mask'):
                self._build_masks()
            attn_weight = attn_weight * self._attn_mask.to(attn_weight.device).detach()

        attn_weight = torch.softmax(attn_weight, dim=-1)

        # Post-softmax text emphasizing
        if config.should_use_text_emphasizing(curr_step, self.layer_num):
            if not hasattr(self, '_text_emphasizing_mask'):
                self._build_masks()
            attn_weight = attn_weight * self._text_emphasizing_mask.to(attn_weight.device).detach()

        out = attn_weight @ joint_value
        out = out.transpose(1, 2)  # B, H, S, D -> B, S, H, D
        out = out.flatten(2, 3)
        out = out.to(joint_query.dtype)

        txt_attn_output = out[:, :seq_txt, :]
        img_attn_output = out[:, seq_txt:, :]

        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output)

        self.curr_step += 1

        return img_attn_output, txt_attn_output

    def save_locality_scores(self):
        pass
