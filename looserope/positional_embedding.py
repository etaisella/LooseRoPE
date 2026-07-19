import torch
import torch.nn as nn
from .attention_processor import ATTN_HEIGHT, ATTN_WIDTH
import numpy as np
from typing import List, Union

def set_custom_img_ids(shrink_factor_h=1.0, shrink_factor_w=1.0, start_h=0, start_w=0, end_h=63, end_w=63):
    img_ids = torch.zeros(ATTN_HEIGHT, ATTN_WIDTH, 3)
    # set pre
    h_ids_pre = torch.arange(start_h)
    w_ids_pre = torch.arange(start_w)
    # set mid
    h_ids_mid = (torch.arange(end_h-start_h+1) * shrink_factor_h)
    if start_h > 0:
        h_ids_mid = h_ids_mid + h_ids_pre[-1] + shrink_factor_h
    w_ids_mid = (torch.arange(end_w-start_w+1) * shrink_factor_w)
    if start_w > 0:
        w_ids_mid = w_ids_mid + w_ids_pre[-1] + shrink_factor_w
    # set post
    h_ids_post = torch.arange(ATTN_HEIGHT-(end_h+1)) + h_ids_mid[-1] + shrink_factor_h
    w_ids_post = torch.arange(ATTN_WIDTH-(end_w+1)) + w_ids_mid[-1] + shrink_factor_w
    h_ids = torch.cat([h_ids_pre, h_ids_mid, h_ids_post], dim=0)
    w_ids = torch.cat([w_ids_pre, w_ids_mid, w_ids_post], dim=0)
    img_ids[..., 1] = img_ids[..., 1] + h_ids[:, None]
    img_ids[..., 2] = img_ids[..., 2] + w_ids[None, :]
    img_ids = img_ids.reshape(ATTN_HEIGHT * ATTN_WIDTH, 3)
    img_ids = img_ids.to(torch.bfloat16)
    return img_ids

def get_1d_rotary_pos_embed_custom(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
    modify_freq_start=0,
    modify_freq_end=0,
    modify_freq_value=1.0
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )  # [D/2]

    # modify frequencies in specified range by multiplying with modify_freq_value
    freqs[modify_freq_start:modify_freq_end] = freqs[modify_freq_start:modify_freq_end] * modify_freq_value

    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis

class LooseRoPEFluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int], modify_freq_start=0, modify_freq_end=0, modify_freq_value=1.0):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.modify_freq_start = modify_freq_start
        self.modify_freq_end = modify_freq_end
        self.modify_freq_value = modify_freq_value

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        print(f"LooseRoPEFluxPosEmbed forward called with ids: {ids.shape}")
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        modify_freq_start = 0
        modify_freq_end = 0
        modify_freq_value = 1.0
        for i in range(n_axes):
            if i > 0:
                modify_freq_start = self.modify_freq_start
                modify_freq_end = self.modify_freq_end
                modify_freq_value = self.modify_freq_value
            cos, sin = get_1d_rotary_pos_embed_custom(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
                modify_freq_start=modify_freq_start,
                modify_freq_end=modify_freq_end,
                modify_freq_value=modify_freq_value
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin