"""Small mask-construction kernels for TA-filter v4 attention."""

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
    def _ta_mark_candidates_from_children_kernel(
        Order_ptr,       # (H_q, S, K) int64/int32
        Depth_ptr,       # (H_q,) int32
        Children_ptr,    # (S, H_kv, K, BF) int32
        CandMask_ptr,    # (H_q, N_pad) int8
        N_PAD,
        K_CLUSTERS,
        S_SUB: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        H_KV: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        CANDS: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        hq = tl.program_id(0)
        split = tl.program_id(1)
        kvh = hq // GROUPS

        row_inner = tl.arange(0, BLOCK_ROWS)
        child_inner = tl.arange(0, BF)
        depth = tl.load(Depth_ptr + hq).to(tl.int32)

        rows_per_split = (K_CLUSTERS + NUM_SPLITS - 1) // NUM_SPLITS
        r_start = split * rows_per_split
        r_stop = tl.minimum(r_start + rows_per_split, K_CLUSTERS)
        r_stop = tl.minimum(r_stop, depth)

        for r_chunk in range(r_start, r_stop, BLOCK_ROWS):
            rows = r_chunk + row_inner
            row_valid = rows < r_stop
            rows_safe = tl.where(row_valid, rows, 0)

            for s_idx in tl.static_range(S_SUB):
                clusters = tl.load(
                    Order_ptr + (hq * S_SUB + s_idx) * K_CLUSTERS + rows_safe,
                    mask=row_valid,
                    other=0,
                ).to(tl.int32)
                key_ids = tl.load(
                    Children_ptr
                    + ((((s_idx * H_KV + kvh) * K_CLUSTERS) + clusters[:, None]) * BF)
                    + child_inner[None, :],
                    mask=row_valid[:, None],
                    other=-1,
                ).to(tl.int32)
                key_flat = tl.reshape(key_ids, [CANDS])
                key_valid = (key_flat >= 0) & (key_flat < N_PAD)
                tl.store(
                    CandMask_ptr + hq * N_PAD + key_flat,
                    tl.full([CANDS], 1, dtype=tl.int8),
                    mask=key_valid,
                )


    @triton.jit
    def _ta_mark_candidates_stamp_from_children_kernel(
        Order_ptr,       # (H_q, S, K) int64/int32
        Depth_ptr,       # (H_q,) int32
        Children_ptr,    # (S, H_kv, K, BF) int32
        CandMask_ptr,    # (H_q, N_pad) int8 stamps
        Stamp_ptr,       # scalar int8
        N_PAD,
        K_CLUSTERS,
        S_SUB: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        H_KV: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        CANDS: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        hq = tl.program_id(0)
        split = tl.program_id(1)
        kvh = hq // GROUPS
        stamp = tl.load(Stamp_ptr).to(tl.int8)

        row_inner = tl.arange(0, BLOCK_ROWS)
        child_inner = tl.arange(0, BF)
        depth = tl.load(Depth_ptr + hq).to(tl.int32)

        rows_per_split = (K_CLUSTERS + NUM_SPLITS - 1) // NUM_SPLITS
        r_start = split * rows_per_split
        r_stop = tl.minimum(r_start + rows_per_split, K_CLUSTERS)
        r_stop = tl.minimum(r_stop, depth)

        for r_chunk in range(r_start, r_stop, BLOCK_ROWS):
            rows = r_chunk + row_inner
            row_valid = rows < r_stop
            rows_safe = tl.where(row_valid, rows, 0)

            for s_idx in tl.static_range(S_SUB):
                clusters = tl.load(
                    Order_ptr + (hq * S_SUB + s_idx) * K_CLUSTERS + rows_safe,
                    mask=row_valid,
                    other=0,
                ).to(tl.int32)
                key_ids = tl.load(
                    Children_ptr
                    + ((((s_idx * H_KV + kvh) * K_CLUSTERS) + clusters[:, None]) * BF)
                    + child_inner[None, :],
                    mask=row_valid[:, None],
                    other=-1,
                ).to(tl.int32)
                key_flat = tl.reshape(key_ids, [CANDS])
                key_valid = (key_flat >= 0) & (key_flat < N_PAD)
                tl.store(
                    CandMask_ptr + hq * N_PAD + key_flat,
                    stamp + tl.zeros([CANDS], dtype=tl.int8),
                    mask=key_valid,
                )


    @triton.jit
    def _ta_mark_candidates_prefix_from_children_kernel(
        Order_ptr,       # (H_q, S, ORDER_K) int64/int32
        Depth_ptr,       # (H_q,) int32, <= ORDER_K
        Children_ptr,    # (S, H_kv, CHILD_K, BF) int32
        CandMask_ptr,    # (H_q, N_pad) int8
        N_PAD,
        ORDER_K,
        CHILD_K,
        S_SUB: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        H_KV: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        CANDS: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        hq = tl.program_id(0)
        split = tl.program_id(1)
        kvh = hq // GROUPS

        row_inner = tl.arange(0, BLOCK_ROWS)
        child_inner = tl.arange(0, BF)
        depth = tl.load(Depth_ptr + hq).to(tl.int32)

        rows_per_split = (ORDER_K + NUM_SPLITS - 1) // NUM_SPLITS
        r_start = split * rows_per_split
        r_stop = tl.minimum(r_start + rows_per_split, ORDER_K)
        r_stop = tl.minimum(r_stop, depth)

        for r_chunk in range(r_start, r_stop, BLOCK_ROWS):
            rows = r_chunk + row_inner
            row_valid = rows < r_stop
            rows_safe = tl.where(row_valid, rows, 0)

            for s_idx in tl.static_range(S_SUB):
                clusters = tl.load(
                    Order_ptr + (hq * S_SUB + s_idx) * ORDER_K + rows_safe,
                    mask=row_valid,
                    other=0,
                ).to(tl.int32)
                cl_valid = row_valid & (clusters >= 0) & (clusters < CHILD_K)
                clusters_safe = tl.where(cl_valid, clusters, 0)
                key_ids = tl.load(
                    Children_ptr
                    + ((((s_idx * H_KV + kvh) * CHILD_K) + clusters_safe[:, None]) * BF)
                    + child_inner[None, :],
                    mask=cl_valid[:, None],
                    other=-1,
                ).to(tl.int32)
                key_flat = tl.reshape(key_ids, [CANDS])
                key_valid = (key_flat >= 0) & (key_flat < N_PAD)
                tl.store(
                    CandMask_ptr + hq * N_PAD + key_flat,
                    tl.full([CANDS], 1, dtype=tl.int8),
                    mask=key_valid,
                )


    @triton.jit
    def _ta_mark_selected_clusters_kernel(
        Order_ptr,       # (H_q, S, K) int64/int32
        Depth_ptr,       # (H_q,) int32
        Selected_ptr,    # (H_q, S, K) int8
        K_CLUSTERS,
        S_SUB: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        hq = tl.program_id(0)
        s_idx = tl.program_id(1)
        block = tl.program_id(2)

        rows = block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        depth = tl.load(Depth_ptr + hq).to(tl.int32)
        row_valid = rows < tl.minimum(depth, K_CLUSTERS)
        clusters = tl.load(
            Order_ptr + (hq * S_SUB + s_idx) * K_CLUSTERS + rows,
            mask=row_valid,
            other=0,
        ).to(tl.int32)
        tl.store(
            Selected_ptr + (hq * S_SUB + s_idx) * K_CLUSTERS + clusters,
            tl.full([BLOCK_ROWS], 1, dtype=tl.int8),
            mask=row_valid,
        )


def run_ta_mark_candidates_from_children(
    order: torch.Tensor,
    depth_i32: torch.Tensor,
    children_i32: torch.Tensor,
    cand_mask_i8: torch.Tensor,
    *,
    n_pad: int,
    groups: int,
    block_rows: int,
    num_splits: int,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    s_sub = int(children_i32.shape[0])
    h_kv = int(children_i32.shape[1])
    k_clusters = int(children_i32.shape[2])
    bf = int(children_i32.shape[3])
    cands = int(block_rows) * bf
    grid = (int(order.shape[0]), int(num_splits))
    _ta_mark_candidates_from_children_kernel[grid](
        order,
        depth_i32,
        children_i32,
        cand_mask_i8,
        n_pad,
        k_clusters,
        S_SUB=s_sub,
        BF=bf,
        GROUPS=int(groups),
        H_KV=h_kv,
        BLOCK_ROWS=int(block_rows),
        CANDS=cands,
        NUM_SPLITS=int(num_splits),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_mark_candidates_stamp_from_children(
    order: torch.Tensor,
    depth_i32: torch.Tensor,
    children_i32: torch.Tensor,
    cand_mask_i8: torch.Tensor,
    stamp_i8: torch.Tensor,
    *,
    n_pad: int,
    groups: int,
    block_rows: int,
    num_splits: int,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    s_sub = int(children_i32.shape[0])
    h_kv = int(children_i32.shape[1])
    k_clusters = int(children_i32.shape[2])
    bf = int(children_i32.shape[3])
    cands = int(block_rows) * bf
    grid = (int(order.shape[0]), int(num_splits))
    _ta_mark_candidates_stamp_from_children_kernel[grid](
        order,
        depth_i32,
        children_i32,
        cand_mask_i8,
        stamp_i8,
        n_pad,
        k_clusters,
        S_SUB=s_sub,
        BF=bf,
        GROUPS=int(groups),
        H_KV=h_kv,
        BLOCK_ROWS=int(block_rows),
        CANDS=cands,
        NUM_SPLITS=int(num_splits),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_mark_candidates_prefix_from_children(
    order_prefix: torch.Tensor,
    depth_i32: torch.Tensor,
    children_i32: torch.Tensor,
    cand_mask_i8: torch.Tensor,
    *,
    n_pad: int,
    groups: int,
    block_rows: int,
    num_splits: int,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    s_sub = int(children_i32.shape[0])
    h_kv = int(children_i32.shape[1])
    child_k = int(children_i32.shape[2])
    bf = int(children_i32.shape[3])
    order_k = int(order_prefix.shape[-1])
    cands = int(block_rows) * bf
    grid = (int(order_prefix.shape[0]), int(num_splits))
    _ta_mark_candidates_prefix_from_children_kernel[grid](
        order_prefix,
        depth_i32,
        children_i32,
        cand_mask_i8,
        n_pad,
        order_k,
        child_k,
        S_SUB=s_sub,
        BF=bf,
        GROUPS=int(groups),
        H_KV=h_kv,
        BLOCK_ROWS=int(block_rows),
        CANDS=cands,
        NUM_SPLITS=int(num_splits),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_mark_selected_clusters(
    order: torch.Tensor,
    depth_i32: torch.Tensor,
    selected_i8: torch.Tensor,
    *,
    block_rows: int,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    h_q, s_sub, k_clusters = selected_i8.shape
    grid = (h_q, s_sub, triton.cdiv(k_clusters, block_rows))
    _ta_mark_selected_clusters_kernel[grid](
        order,
        depth_i32,
        selected_i8,
        k_clusters,
        S_SUB=s_sub,
        BLOCK_ROWS=int(block_rows),
        num_warps=num_warps,
        num_stages=num_stages,
    )
