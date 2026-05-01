"""PyTorch FlexAttention fp16 masked baseline.

This uses FlexAttention's ``BlockMask`` path.  The mask is cached by tensor
identity/shape/block size so repeated decode queries can reuse the BlockMask.
"""

from __future__ import annotations

import math

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    HAS_FLEX_ATTENTION = True
except Exception:  # pragma: no cover
    HAS_FLEX_ATTENTION = False
    create_block_mask = None
    flex_attention = None


_FLEX_COMPILED = None
_MASK_CACHE: dict[tuple, object] = {}


def _compiled_flex():
    global _FLEX_COMPILED
    if _FLEX_COMPILED is None:
        _FLEX_COMPILED = torch.compile(flex_attention, dynamic=False)
    return _FLEX_COMPILED


def sdpa_flex_attention_fp16(
    q: torch.Tensor,
    keys_f16: torch.Tensor,
    values_f16: torch.Tensor,
    mask_i8: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
    *,
    block_size: tuple[int, int] = (16, 128),
    num_warps: int = 8,
    prescale_qk: bool = False,
) -> torch.Tensor:
    if not HAS_FLEX_ATTENTION:
        raise RuntimeError("FlexAttention unavailable in this PyTorch build")
    if q.dtype != torch.float16 or keys_f16.dtype != torch.float16 or values_f16.dtype != torch.float16:
        raise TypeError("sdpa_flex_attention_fp16 expects fp16 q/keys/values")
    if mask_i8.dtype != torch.int8:
        raise TypeError("sdpa_flex_attention_fp16 expects int8 mask")
    h_q, d = q.shape
    h_kv, n_ctx, d_k = keys_f16.shape
    d_v = int(values_f16.shape[-1])
    if d != d_k or values_f16.shape[:2] != (h_kv, n_ctx):
        raise ValueError("incompatible Q/K/V shapes")
    if mask_i8.shape != (h_q, n_ctx):
        raise ValueError("mask_i8 must be (H_q, N)")
    if q_head_to_kv is not None and h_q % h_kv != 0:
        raise ValueError("only grouped GQA supported")
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    mask_bool = mask_i8.bool()
    cache_key = (
        mask_i8.data_ptr(),
        mask_i8._version,
        h_q,
        n_ctx,
        tuple(block_size),
        int(num_warps),
        bool(prescale_qk),
        q.device.index,
    )
    block_mask = _MASK_CACHE.get(cache_key)
    if block_mask is None:
        def mask_mod(_b, h, _q_idx, kv_idx):
            return mask_bool[h, kv_idx]

        block_mask = create_block_mask(
            mask_mod,
            B=1,
            H=h_q,
            Q_LEN=1,
            KV_LEN=n_ctx,
            device=q.device,
            BLOCK_SIZE=block_size,
            _compile=True,
        )
        _MASK_CACHE.clear()
        _MASK_CACHE[cache_key] = block_mask

    q4 = q.view(1, h_q, 1, d)
    k4 = keys_f16.view(1, h_kv, n_ctx, d)
    v4 = values_f16.view(1, h_kv, n_ctx, d_v)
    out = _compiled_flex()(
        q4,
        k4,
        v4,
        block_mask=block_mask,
        scale=float(scale),
        enable_gqa=(h_q != h_kv),
        kernel_options={
            "BLOCK_M": int(block_size[0]),
            "BLOCK_N": int(block_size[1]),
            "ROWS_GUARANTEED_SAFE": True,
            "num_warps": int(num_warps),
            "PRESCALE_QK": bool(prescale_qk),
        },
    )
    return out.view(h_q, d_v)
