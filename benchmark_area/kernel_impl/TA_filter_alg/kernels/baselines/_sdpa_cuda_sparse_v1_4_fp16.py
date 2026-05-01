from __future__ import annotations
import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

_EXT = None
_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}

def _load_ext():
    global _EXT
    if _EXT is not None: return _EXT
    base = Path(__file__).resolve().parent
    _EXT = load(
        name="ta_sdpa_cuda_sparse_v1_4_fp16",
        sources=[str(base / "sdpa_cuda_sparse_v1_4_kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _EXT

def sdpa_cuda_sparse_v1_4_fp16(
    q: torch.Tensor,
    keys_f16: torch.Tensor,
    values_f16: torch.Tensor,
    mask_i8: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
    *,
    num_splits: int = 34,
) -> torch.Tensor:
    h_q, d = q.shape
    h_kv, n_ctx, _ = keys_f16.shape
    if scale is None: scale = d ** -0.5
    key = (q.device.index, h_q, h_kv, n_ctx, d, int(num_splits))
    ws = _CACHE.get(key)
    cols = (n_ctx + int(num_splits) - 1) // int(num_splits)
    if ws is None:
        ws = {
            "partial_m": torch.empty(h_q, num_splits, device=q.device, dtype=torch.float32),
            "partial_l": torch.empty(h_q, num_splits, device=q.device, dtype=torch.float32),
            "partial_o": torch.empty(h_q, num_splits, d, device=q.device, dtype=torch.float32),
            "scores": torch.empty(h_q, num_splits, cols, device=q.device, dtype=torch.float32),
            "counters": torch.empty(h_q, device=q.device, dtype=torch.int32),
            "out": torch.empty(h_q, d, device=q.device, dtype=torch.float16),
        }
        _CACHE[key] = ws
    ext = _load_ext()
    return ext.forward(
        q.contiguous(), keys_f16.contiguous(), values_f16.contiguous(), mask_i8.contiguous(),
        ws["partial_m"], ws["partial_l"], ws["partial_o"], ws["scores"], ws["counters"], ws["out"],
        float(scale), int(num_splits)
    )
