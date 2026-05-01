"""v1.1 sparse masked fp16 decode-SDPA CUDA baseline.

CUDA compaction path:
1. Build compact live-key ids from mask.
2. Run attention over live ids only.

This is intended for scattered low-density masks.  It is slower for all-true
masks because compaction adds overhead.
"""

from __future__ import annotations

import torch

from ._sdpa_cuda_atomic_fp16 import _load_ext


_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}


def sdpa_cuda_sparse_v1_1_fp16(
    q: torch.Tensor,
    keys_f16: torch.Tensor,
    values_f16: torch.Tensor,
    mask_i8: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
    *,
    num_splits: int = 32,
) -> torch.Tensor:
    del q_head_to_kv
    if q.dtype != torch.float16 or keys_f16.dtype != torch.float16 or values_f16.dtype != torch.float16:
        raise TypeError("sdpa_cuda_sparse_v1_1_fp16 expects fp16 q/keys/values")
    if mask_i8.dtype != torch.int8:
        raise TypeError("sdpa_cuda_sparse_v1_1_fp16 expects int8 mask")
    h_q, d = q.shape
    h_kv, n_ctx, d_k = keys_f16.shape
    d_v = int(values_f16.shape[-1])
    if d != 128 or d_k != 128 or d_v != 128:
        raise ValueError("sdpa_cuda_sparse_v1_1_fp16 currently specializes D=Dv=128")
    if h_q % h_kv != 0:
        raise ValueError("sdpa_cuda_sparse_v1_1_fp16 expects grouped GQA")
    if mask_i8.shape != (h_q, n_ctx):
        raise ValueError("mask_i8 must be (H_q, N)")
    if scale is None:
        scale = d ** -0.5

    splits = int(num_splits)
    cols = (n_ctx + splits - 1) // splits
    key = (q.device.index, h_q, h_kv, n_ctx, d, d_v, splits)
    ws = _CACHE.get(key)
    if ws is None:
        ws = {
            "indices": torch.empty(h_q, n_ctx, device=q.device, dtype=torch.int32),
            "counts": torch.empty(h_q, device=q.device, dtype=torch.int32),
            "partial_m": torch.empty(h_q, splits, device=q.device, dtype=torch.float32),
            "partial_l": torch.empty(h_q, splits, device=q.device, dtype=torch.float32),
            "partial_o": torch.empty(h_q, splits, d_v, device=q.device, dtype=torch.float32),
            "scores": torch.empty(h_q, splits, cols, device=q.device, dtype=torch.float32),
            "counters": torch.empty(h_q, device=q.device, dtype=torch.int32),
            "out": torch.empty(h_q, d_v, device=q.device, dtype=torch.float16),
        }
        _CACHE.clear()
        _CACHE[key] = ws

    ext = _load_ext()
    return ext.forward_sparse_v1(
        q.contiguous(),
        keys_f16.contiguous(),
        values_f16.contiguous(),
        mask_i8.contiguous(),
        ws["indices"],
        ws["counts"],
        ws["partial_m"],
        ws["partial_l"],
        ws["partial_o"],
        ws["scores"],
        ws["counters"],
        ws["out"],
        float(scale),
        splits,
    )
