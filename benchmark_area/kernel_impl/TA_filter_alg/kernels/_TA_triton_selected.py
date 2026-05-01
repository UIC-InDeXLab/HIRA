"""Build selected[h, s, c] = 1 iff cluster c has rank < depth[h] in subspace s.

Used by v6 inline-mask attention: instead of materialising an (H_q, N_pad)
candidate mask, we keep a small (H_q, S, K) int8 ``selected`` lookup that the
attention kernel reads inline.
"""

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
    def _ta_build_selected_kernel(
        Order_ptr,           # (H_q, S, K) int32 -- cluster ids, descending order
        SortedScores_ptr,    # (H_q, S, K) fp32 -- sorted desc
        Threshold_ptr,       # (H_q,) fp32
        Selected_ptr,        # (H_q, S, K) int8 -- output
        K_CLUSTERS,
        S_SUB: tl.constexpr,
        K_BLOCK: tl.constexpr,    # next pow2 of K_CLUSTERS for depth scan
    ):
        hq = tl.program_id(0)
        s_idx = tl.program_id(1)

        # Compute depth per h_q (shared across s) by summing sorted_scores across S.
        offs = tl.arange(0, K_BLOCK)
        valid = offs < K_CLUSTERS
        row_sum = tl.zeros([K_BLOCK], dtype=tl.float32)
        for s in tl.static_range(S_SUB):
            v = tl.load(
                SortedScores_ptr + (hq * S_SUB + s) * K_CLUSTERS + offs,
                mask=valid,
                other=0.0,
            )
            row_sum += v
        th = tl.load(Threshold_ptr + hq).to(tl.float32)
        below = (row_sum < th) & valid
        first = tl.min(tl.where(below, offs, K_CLUSTERS), axis=0)
        depth = tl.where(first < K_CLUSTERS, first + 1, K_CLUSTERS)

        # Zero-init selected for this (hq, s_idx) row, then stamp 1 at top-depth slots.
        # First clear:
        tl.store(
            Selected_ptr + (hq * S_SUB + s_idx) * K_CLUSTERS + offs,
            tl.zeros([K_BLOCK], dtype=tl.int8),
            mask=valid,
        )
        # Then for ranks < depth, scatter 1 at order[hq, s_idx, rank].
        in_top = offs < depth
        clusters = tl.load(
            Order_ptr + (hq * S_SUB + s_idx) * K_CLUSTERS + offs,
            mask=in_top,
            other=0,
        ).to(tl.int32)
        cluster_valid = in_top & (clusters >= 0) & (clusters < K_CLUSTERS)
        tl.store(
            Selected_ptr + (hq * S_SUB + s_idx) * K_CLUSTERS + clusters,
            tl.full([K_BLOCK], 1, dtype=tl.int8),
            mask=cluster_valid,
        )


    @triton.jit
    def _ta_build_selected_topk_kernel(
        TopOrder_ptr,        # (H_q, S, L) int64/int32 cluster ids, descending score
        TopScores_ptr,       # (H_q, S, L) fp16/fp32 scores, descending
        Threshold_ptr,       # (H_q,) fp32
        Selected_ptr,        # (H_q, S, K) int8 -- output
        K_CLUSTERS,
        TOPK,
        S_SUB: tl.constexpr,
        K_BLOCK: tl.constexpr,    # next pow2 of K_CLUSTERS for clearing
        TOPK_BLOCK: tl.constexpr, # next pow2 of TOPK for depth scan/scatter
    ):
        hq = tl.program_id(0)
        s_idx = tl.program_id(1)

        # Clear the full selected row because downstream kernels index by the
        # original cluster id space, not by the compact top-k position.
        k_offs = tl.arange(0, K_BLOCK)
        k_valid = k_offs < K_CLUSTERS
        tl.store(
            Selected_ptr + (hq * S_SUB + s_idx) * K_CLUSTERS + k_offs,
            tl.zeros([K_BLOCK], dtype=tl.int8),
            mask=k_valid,
        )

        # Compute the TA stop depth inside the top-k window.
        top_offs = tl.arange(0, TOPK_BLOCK)
        top_valid = top_offs < TOPK
        row_sum = tl.zeros([TOPK_BLOCK], dtype=tl.float32)
        for s in tl.static_range(S_SUB):
            v = tl.load(
                TopScores_ptr + (hq * S_SUB + s) * TOPK + top_offs,
                mask=top_valid,
                other=0.0,
            )
            row_sum += v
        th = tl.load(Threshold_ptr + hq).to(tl.float32)
        below = (row_sum < th) & top_valid
        first = tl.min(tl.where(below, top_offs, TOPK), axis=0)
        depth = tl.where(first < TOPK, first + 1, TOPK)

        # Scatter selected bits for ranks inside the top-k stop depth.
        in_top = top_offs < depth
        clusters = tl.load(
            TopOrder_ptr + (hq * S_SUB + s_idx) * TOPK + top_offs,
            mask=in_top,
            other=0,
        ).to(tl.int32)
        cluster_valid = in_top & (clusters >= 0) & (clusters < K_CLUSTERS)
        tl.store(
            Selected_ptr + (hq * S_SUB + s_idx) * K_CLUSTERS + clusters,
            tl.full([TOPK_BLOCK], 1, dtype=tl.int8),
            mask=cluster_valid,
        )


    @triton.jit
    def _ta_build_selected_fixed_topk_kernel(
        TopOrder_ptr,        # (H_q, S, L) int64/int32 cluster ids
        Selected_ptr,        # (H_q, S, K) int8 -- output
        K_CLUSTERS,
        TOPK,
        S_SUB: tl.constexpr,
        K_BLOCK: tl.constexpr,    # next pow2 of K_CLUSTERS for clearing
        TOPK_BLOCK: tl.constexpr, # next pow2 of TOPK for scatter
    ):
        hq = tl.program_id(0)
        s_idx = tl.program_id(1)

        k_offs = tl.arange(0, K_BLOCK)
        k_valid = k_offs < K_CLUSTERS
        tl.store(
            Selected_ptr + (hq * S_SUB + s_idx) * K_CLUSTERS + k_offs,
            tl.zeros([K_BLOCK], dtype=tl.int8),
            mask=k_valid,
        )

        top_offs = tl.arange(0, TOPK_BLOCK)
        top_valid = top_offs < TOPK
        clusters = tl.load(
            TopOrder_ptr + (hq * S_SUB + s_idx) * TOPK + top_offs,
            mask=top_valid,
            other=0,
        ).to(tl.int32)
        cluster_valid = top_valid & (clusters >= 0) & (clusters < K_CLUSTERS)
        tl.store(
            Selected_ptr + (hq * S_SUB + s_idx) * K_CLUSTERS + clusters,
            tl.full([TOPK_BLOCK], 1, dtype=tl.int8),
            mask=cluster_valid,
        )


def run_ta_build_selected(
    order_i32: torch.Tensor,         # (H_q, S, K) int32
    sorted_scores_f32: torch.Tensor, # (H_q, S, K) fp32
    threshold_f32: torch.Tensor,     # (H_q,) fp32
    selected_i8: torch.Tensor,       # (H_q, S, K) int8 (overwritten)
    *,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, k_clusters = order_i32.shape
    k_block = _next_pow2(int(k_clusters))
    grid = (int(h_q), int(s_sub))
    _ta_build_selected_kernel[grid](
        order_i32,
        sorted_scores_f32,
        threshold_f32,
        selected_i8,
        int(k_clusters),
        S_SUB=int(s_sub),
        K_BLOCK=int(k_block),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_build_selected_topk(
    top_order_i32: torch.Tensor,     # (H_q, S, L) int32/int64
    top_scores_f32: torch.Tensor,    # (H_q, S, L) fp16/fp32
    threshold_f32: torch.Tensor,     # (H_q,) fp32
    selected_i8: torch.Tensor,       # (H_q, S, K) int8 (overwritten)
    *,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, topk = top_order_i32.shape
    k_clusters = int(selected_i8.shape[-1])
    grid = (int(h_q), int(s_sub))
    _ta_build_selected_topk_kernel[grid](
        top_order_i32,
        top_scores_f32,
        threshold_f32,
        selected_i8,
        k_clusters,
        int(topk),
        S_SUB=int(s_sub),
        K_BLOCK=_next_pow2(k_clusters),
        TOPK_BLOCK=_next_pow2(int(topk)),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_build_selected_fixed_topk(
    top_order_i32: torch.Tensor,     # (H_q, S, L) int32/int64
    selected_i8: torch.Tensor,       # (H_q, S, K) int8 (overwritten)
    *,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, topk = top_order_i32.shape
    k_clusters = int(selected_i8.shape[-1])
    grid = (int(h_q), int(s_sub))
    _ta_build_selected_fixed_topk_kernel[grid](
        top_order_i32,
        selected_i8,
        k_clusters,
        int(topk),
        S_SUB=int(s_sub),
        K_BLOCK=_next_pow2(k_clusters),
        TOPK_BLOCK=_next_pow2(int(topk)),
        num_warps=num_warps,
        num_stages=num_stages,
    )
