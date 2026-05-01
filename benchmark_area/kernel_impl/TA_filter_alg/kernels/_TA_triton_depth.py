"""Triton stop-depth helper for TA-filter sorted centroid scores."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


if HAS_TRITON:

    @triton.jit
    def _ta_stop_depth_kernel(
        SortedScores_ptr,  # (H_q, S, K) fp32
        Threshold_ptr,     # (H_q,) fp32
        Depth_ptr,         # (H_q,) int32
        K_CLUSTERS,
        S_SUB: tl.constexpr,
        K_BLOCK: tl.constexpr,
    ):
        hq = tl.program_id(0)
        offs = tl.arange(0, K_BLOCK)
        valid = offs < K_CLUSTERS
        row_sum = tl.zeros([K_BLOCK], dtype=tl.float32)
        for s_idx in tl.static_range(S_SUB):
            vals = tl.load(
                SortedScores_ptr + (hq * S_SUB + s_idx) * K_CLUSTERS + offs,
                mask=valid,
                other=0.0,
            )
            row_sum += vals
        th = tl.load(Threshold_ptr + hq)
        below = (row_sum < th) & valid
        first = tl.min(tl.where(below, offs, K_CLUSTERS), axis=0)
        depth = tl.where(first < K_CLUSTERS, first + 1, K_CLUSTERS)
        tl.store(Depth_ptr + hq, depth)


def run_ta_stop_depth(
    sorted_scores_f32: torch.Tensor,
    threshold_f32: torch.Tensor,
    depth_i32: torch.Tensor,
) -> None:
    h_q, s_sub, k_clusters = sorted_scores_f32.shape
    k_block = _next_pow2(k_clusters)
    _ta_stop_depth_kernel[(h_q,)](
        sorted_scores_f32,
        threshold_f32,
        depth_i32,
        k_clusters,
        S_SUB=int(s_sub),
        K_BLOCK=int(k_block),
        num_warps=8,
        num_stages=3,
    )
