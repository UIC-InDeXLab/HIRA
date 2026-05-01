"""TA-filter candidate compaction kernels for v2 attention."""

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
    def _ta_compact_candidates_kernel(
        Order_ptr,       # (H_q, S, K) int32
        Depth_ptr,       # (H_q,) int32
        Children_ptr,    # (S, H_kv, K, BF) int32
        Visited_ptr,     # (H_q, N_pad + 1) int32
        Counts_ptr,      # (H_q,) int32
        CandIds_ptr,     # (H_q, N_pad) int32
        N_PAD,
        VISITED_STRIDE,
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
                key_safe = tl.where(key_valid, key_flat, N_PAD)

                old = tl.atomic_cas(
                    Visited_ptr + hq * VISITED_STRIDE + key_safe,
                    tl.full([CANDS], 0, dtype=tl.int32),
                    tl.full([CANDS], 1, dtype=tl.int32),
                    sem="relaxed",
                )
                is_new = key_valid & (old == 0)
                count_ptrs = Counts_ptr + hq + tl.zeros([CANDS], dtype=tl.int32)
                slots = tl.atomic_add(
                    count_ptrs,
                    tl.full([CANDS], 1, dtype=tl.int32),
                    mask=is_new,
                    sem="relaxed",
                )
                tl.store(
                    CandIds_ptr + hq * N_PAD + slots,
                    key_flat,
                    mask=is_new & (slots < N_PAD),
                )


    @triton.jit
    def _ta_compact_candidates_stamp_kernel(
        Order_ptr,       # (H_q, S, K) int32/int64
        Depth_ptr,       # (H_q,) int32
        Children_ptr,    # (S, H_kv, K, BF) int32
        Visited_ptr,     # (H_q, N_pad) int32 stamps
        Counts_ptr,      # (H_q,) int32
        CandIds_ptr,     # (H_q, N_pad) int32
        Stamp_ptr,       # scalar int32
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
        stamp = tl.load(Stamp_ptr).to(tl.int32)

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
                key_safe = tl.where(key_valid, key_flat, 0)

                stamp_vec = stamp + tl.zeros([CANDS], dtype=tl.int32)
                old = tl.atomic_xchg(
                    Visited_ptr + hq * N_PAD + key_safe,
                    stamp_vec,
                    mask=key_valid,
                    sem="relaxed",
                )
                is_new = key_valid & (old != stamp_vec)
                count_ptrs = Counts_ptr + hq + tl.zeros([CANDS], dtype=tl.int32)
                slots = tl.atomic_add(
                    count_ptrs,
                    tl.full([CANDS], 1, dtype=tl.int32),
                    mask=is_new,
                    sem="relaxed",
                )
                tl.store(
                    CandIds_ptr + hq * N_PAD + slots,
                    key_flat,
                    mask=is_new & (slots < N_PAD),
                )


def run_ta_compact_candidates(
    order_i32: torch.Tensor,
    depth_i32: torch.Tensor,
    children_i32: torch.Tensor,
    visited_i32: torch.Tensor,
    counts_i32: torch.Tensor,
    cand_ids_i32: torch.Tensor,
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
    grid = (int(order_i32.shape[0]), int(num_splits))
    _ta_compact_candidates_kernel[grid](
        order_i32,
        depth_i32,
        children_i32,
        visited_i32,
        counts_i32,
        cand_ids_i32,
        n_pad,
        int(visited_i32.shape[1]),
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


def run_ta_compact_candidates_stamp(
    order_i32: torch.Tensor,
    depth_i32: torch.Tensor,
    children_i32: torch.Tensor,
    visited_i32: torch.Tensor,
    counts_i32: torch.Tensor,
    cand_ids_i32: torch.Tensor,
    stamp_i32: torch.Tensor,
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
    grid = (int(order_i32.shape[0]), int(num_splits))
    _ta_compact_candidates_stamp_kernel[grid](
        order_i32,
        depth_i32,
        children_i32,
        visited_i32,
        counts_i32,
        cand_ids_i32,
        stamp_i32,
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
