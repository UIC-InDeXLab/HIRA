"""Triton centroid score kernel for TA-filter v2 attention."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _ta_centroid_scores_kernel(
        Q_ptr,             # (H_q, D) fp16
        Centers_ptr,       # (S, H_kv, K, MAX_W) fp16
        DimOffsets_ptr,    # (S,) int32
        DimWidths_ptr,     # (S,) int32
        Scores_ptr,        # (H_q, S, K) fp32
        K_CLUSTERS,
        D: tl.constexpr,
        S_SUB: tl.constexpr,
        H_KV: tl.constexpr,
        GROUPS: tl.constexpr,
        MAX_W: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        hq = tl.program_id(0)
        s_idx = tl.program_id(1)
        k_block = tl.program_id(2)
        kvh = hq // GROUPS

        k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k_valid = k_offsets < K_CLUSTERS
        w = tl.arange(0, MAX_W)
        dim_off = tl.load(DimOffsets_ptr + s_idx).to(tl.int32)
        dim_w = tl.load(DimWidths_ptr + s_idx).to(tl.int32)
        w_valid = w < dim_w

        q_vals = tl.load(
            Q_ptr + hq * D + dim_off + w,
            mask=w_valid,
            other=0.0,
        )
        centers = tl.load(
            Centers_ptr
            + (((s_idx * H_KV + kvh) * K_CLUSTERS + k_offsets[:, None]) * MAX_W)
            + w[None, :],
            mask=k_valid[:, None] & w_valid[None, :],
            other=0.0,
        )
        scores = tl.sum(centers.to(tl.float32) * q_vals[None, :].to(tl.float32), axis=1)
        tl.store(
            Scores_ptr + (hq * S_SUB + s_idx) * K_CLUSTERS + k_offsets,
            scores,
            mask=k_valid,
        )


def run_ta_centroid_scores(
    q_f16: torch.Tensor,
    centers_padded_f16: torch.Tensor,
    dim_offsets_i32: torch.Tensor,
    dim_widths_i32: torch.Tensor,
    scores_f32: torch.Tensor,
    *,
    groups: int,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    h_q, d = q_f16.shape
    s_sub, h_kv, k_clusters, max_w = centers_padded_f16.shape
    grid = (h_q, s_sub, triton.cdiv(k_clusters, block_k))
    _ta_centroid_scores_kernel[grid](
        q_f16,
        centers_padded_f16,
        dim_offsets_i32,
        dim_widths_i32,
        scores_f32,
        k_clusters,
        D=int(d),
        S_SUB=int(s_sub),
        H_KV=int(h_kv),
        GROUPS=int(groups),
        MAX_W=int(max_w),
        BLOCK_K=int(block_k),
        num_warps=num_warps,
        num_stages=num_stages,
    )
