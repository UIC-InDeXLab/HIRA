"""v1.6 sparse masked fp16 decode-SDPA CUDA baseline.

Atomics-free ordered compaction:
1. Count active keys per fixed mask block.
2. Prefix block counts per head.
3. Fill compact live-key ids in key order without global atomicAdd.
4. Run split decode attention over live ids only.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT = None
_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    base = Path(__file__).resolve().parent
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ.get("PATH", "")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    _EXT = load(
        name="ta_sdpa_cuda_sparse_v1_6_fp16",
        sources=[
            str(base / "sdpa_cuda_sparse_v1_6.cpp"),
            str(base / "sdpa_cuda_sparse_v1_6_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _EXT


def sdpa_cuda_sparse_v1_6_fp16(
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
        raise TypeError("sdpa_cuda_sparse_v1_6_fp16 expects fp16 q/keys/values")
    if mask_i8.dtype != torch.int8:
        raise TypeError("sdpa_cuda_sparse_v1_6_fp16 expects int8 mask")
    h_q, d = q.shape
    h_kv, n_ctx, d_k = keys_f16.shape
    d_v = int(values_f16.shape[-1])
    if d != 128 or d_k != 128 or d_v != 128:
        raise ValueError("sdpa_cuda_sparse_v1_6_fp16 currently specializes D=Dv=128")
    if h_q % h_kv != 0:
        raise ValueError("sdpa_cuda_sparse_v1_6_fp16 expects grouped GQA")
    if mask_i8.shape != (h_q, n_ctx):
        raise ValueError("mask_i8 must be (H_q, N)")
    if scale is None:
        scale = d ** -0.5

    splits = int(num_splits)
    num_blocks = (n_ctx + 255) // 256
    key = (q.device.index, h_q, h_kv, n_ctx, d, d_v, splits)
    ws = _CACHE.get(key)
    if ws is None:
        ws = {
            "indices": torch.empty(h_q, n_ctx, device=q.device, dtype=torch.int32),
            "block_counts": torch.empty(h_q, num_blocks, device=q.device, dtype=torch.int32),
            "block_offsets": torch.empty(h_q, num_blocks, device=q.device, dtype=torch.int32),
            "counts": torch.empty(h_q, device=q.device, dtype=torch.int32),
            "partial_m": torch.empty(h_q, splits, device=q.device, dtype=torch.float32),
            "partial_l": torch.empty(h_q, splits, device=q.device, dtype=torch.float32),
            "partial_o": torch.empty(h_q, splits, d_v, device=q.device, dtype=torch.float32),
            "counters": torch.empty(h_q, device=q.device, dtype=torch.int32),
            "out": torch.empty(h_q, d_v, device=q.device, dtype=torch.float16),
        }
        _CACHE.clear()
        _CACHE[key] = ws

    ext = _load_ext()
    return ext.forward(
        q.contiguous(),
        keys_f16.contiguous(),
        values_f16.contiguous(),
        mask_i8.contiguous(),
        ws["indices"],
        ws["block_counts"],
        ws["block_offsets"],
        ws["counts"],
        ws["partial_m"],
        ws["partial_l"],
        ws["partial_o"],
        ws["counters"],
        ws["out"],
        float(scale),
        splits,
    )
