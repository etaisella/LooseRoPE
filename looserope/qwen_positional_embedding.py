"""
Qwen-specific positional embedding manipulation for the LooseRoPE method.

Mirrors the logic from positional_embedding.py (Flux-specific) but adapted for
Qwen's RoPE format: complex-valued, 3-axis (frame, height, width), with scale_rope.
"""

import torch
import numpy as np
from typing import List, Tuple


def set_custom_vid_freqs_qwen(
    pos_embed,
    frame: int,
    height: int,
    width: int,
    shrink_factor_h: float = 1.0,
    shrink_factor_w: float = 1.0,
    start_h: int = 0,
    start_w: int = 0,
    end_h: int = 63,
    end_w: int = 63,
) -> torch.Tensor:
    """
    Build custom video (image) RoPE frequencies with shrink_factor applied to the
    crop region, using Qwen's complex-valued RoPE.

    This is the Qwen equivalent of set_custom_img_ids + FluxPosEmbed.
    Instead of building float IDs and feeding them to a PosEmbed module,
    we build custom fractional position grids and compute the complex RoPE
    frequencies directly using Qwen's rope_params approach.

    Args:
        pos_embed: The QwenEmbedRope module from the transformer.
        frame, height, width: Spatial dimensions of the latent (after packing).
        shrink_factor_h, shrink_factor_w: Shrink factors for the crop region.
        start_h, start_w, end_h, end_w: Crop region boundaries.

    Returns:
        vid_freqs: Complex-valued RoPE tensor of shape (frame*height*width, D).
    """
    axes_dim = pos_embed.axes_dim
    theta = pos_embed.theta

    frame_indices = torch.arange(frame, dtype=torch.float32)

    h_ids = _build_shrunk_1d_ids(height, shrink_factor_h, start_h, end_h)
    w_ids = _build_shrunk_1d_ids(width, shrink_factor_w, start_w, end_w)

    if pos_embed.scale_rope:
        h_ids = h_ids - height / 2.0
        w_ids = w_ids - width / 2.0

    freqs_frame = _rope_params(frame_indices, axes_dim[0], theta)
    freqs_height = _rope_params(h_ids, axes_dim[1], theta)
    freqs_width = _rope_params(w_ids, axes_dim[2], theta)

    freqs_frame = freqs_frame[:frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
    freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
    freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)

    freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1)
    freqs = freqs.reshape(frame * height * width, -1)
    return freqs


def _build_shrunk_1d_ids(size: int, shrink_factor: float, start: int, end: int) -> torch.Tensor:
    """
    Build 1D position indices with the crop region shrunk by shrink_factor.
    Same logic as set_custom_img_ids from positional_embedding.py, but for a single axis.
    """
    ids_pre = torch.arange(start, dtype=torch.float32)

    ids_mid = torch.arange(end - start + 1, dtype=torch.float32) * shrink_factor
    if start > 0 and len(ids_pre) > 0:
        ids_mid = ids_mid + ids_pre[-1] + shrink_factor

    ids_post = torch.arange(size - (end + 1), dtype=torch.float32)
    if len(ids_mid) > 0:
        ids_post = ids_post + ids_mid[-1] + shrink_factor

    return torch.cat([ids_pre, ids_mid, ids_post], dim=0)


def _rope_params(index: torch.Tensor, dim: int, theta: float = 10000) -> torch.Tensor:
    """
    Compute complex-valued RoPE frequencies for given (possibly fractional) position indices.
    Mirrors QwenEmbedRope.rope_params but accepts float indices.
    """
    assert dim % 2 == 0
    freqs = torch.outer(
        index.float(),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float32) / dim),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def build_custom_image_rotary_emb_qwen(
    pos_embed,
    img_shapes: List[Tuple[int, int, int]],
    txt_seq_lens: List[int],
    shrink_factor_h: float = 1.0,
    shrink_factor_w: float = 1.0,
    start_h: int = 0,
    start_w: int = 0,
    end_h: int = 63,
    end_w: int = 63,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a complete image_rotary_emb tuple (vid_freqs, txt_freqs) with custom
    shrink_factor applied to the output-image region.

    The input-image region gets standard (unshrunk) frequencies since the
    LooseRoPE method only manipulates the output image's spatial perception.

    Args:
        pos_embed: The QwenEmbedRope module.
        img_shapes: List of (frame, height, width) tuples per batch element.
        txt_seq_lens: Text sequence lengths per batch element.
        shrink_factor_h/w: Shrink factors for the crop region.
        start_h/w, end_h/w: Crop region boundaries in the output-image grid.
        device: Target device.

    Returns:
        (vid_freqs, txt_freqs) tuple matching the format expected by the transformer.
    """
    if device is not None and pos_embed.pos_freqs.device != device:
        pos_embed.pos_freqs = pos_embed.pos_freqs.to(device)
        pos_embed.neg_freqs = pos_embed.neg_freqs.to(device)

    fhw_list = img_shapes
    if isinstance(fhw_list, list) and not isinstance(fhw_list[0], (list, tuple)):
        fhw_list = [fhw_list]

    vid_freqs_parts = []
    max_vid_index = 0

    for idx, fhw in enumerate(fhw_list):
        frame, height, width = fhw
        if idx == 0:
            custom_freqs = set_custom_vid_freqs_qwen(
                pos_embed, frame, height, width,
                shrink_factor_h, shrink_factor_w,
                start_h, start_w, end_h, end_w,
            )
            vid_freqs_parts.append(custom_freqs.to(device))
        else:
            standard_freqs = pos_embed._compute_video_freqs(frame, height, width, idx)
            vid_freqs_parts.append(standard_freqs.to(device))

        if pos_embed.scale_rope:
            max_vid_index = max(height // 2, width // 2, max_vid_index)
        else:
            max_vid_index = max(height, width, max_vid_index)

    max_len = max(txt_seq_lens)
    txt_freqs = pos_embed.pos_freqs[max_vid_index: max_vid_index + max_len, ...].to(device)
    vid_freqs = torch.cat(vid_freqs_parts, dim=0).to(device)

    return vid_freqs, txt_freqs
