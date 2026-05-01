"""v13 Triton helpers for parent-child mask attention."""

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


def _next_pow2_min16(x: int) -> int:
    return max(_next_pow2(x), 16)


if HAS_TRITON:

    @triton.jit
    def _ta_v13_depth_from_sorted_kernel(
        SortedScores_ptr,  # (H_q, 4, K) fp16/fp32
        Threshold_ptr,     # (H_q,) fp32
        Depth_ptr,         # (H_q,) int32
        K_CLUSTERS,
        K_BLOCK: tl.constexpr,
    ):
        hq = tl.program_id(0)
        offs = tl.arange(0, K_BLOCK)
        valid = offs < K_CLUSTERS

        row_sum = tl.zeros([K_BLOCK], dtype=tl.float32)
        for s_idx in tl.static_range(4):
            vals = tl.load(
                SortedScores_ptr + (hq * 4 + s_idx) * K_CLUSTERS + offs,
                mask=valid,
                other=0.0,
            )
            row_sum += vals
        th = tl.load(Threshold_ptr + hq).to(tl.float32)
        below = (row_sum < th) & valid
        first = tl.min(tl.where(below, offs, K_CLUSTERS), axis=0)
        depth = tl.where(first < K_CLUSTERS, first + 1, K_CLUSTERS)
        tl.store(Depth_ptr + hq, depth)


    @triton.jit
    def _ta_v13_depth_count_stamp_kernel(
        SortedScores_ptr,  # (H_q, 4, K) fp16/fp32
        Threshold_ptr,     # (H_q,) fp32
        Depth_ptr,         # (H_q,) int32
        Counts_ptr,        # (H_q,) int32
        Stamp_ptr,         # scalar int32
        K_CLUSTERS,
        K_BLOCK: tl.constexpr,
    ):
        hq = tl.program_id(0)
        offs = tl.arange(0, K_BLOCK)
        valid = offs < K_CLUSTERS

        row_sum = tl.zeros([K_BLOCK], dtype=tl.float32)
        for s_idx in tl.static_range(4):
            vals = tl.load(
                SortedScores_ptr + (hq * 4 + s_idx) * K_CLUSTERS + offs,
                mask=valid,
                other=0.0,
            )
            row_sum += vals
        th = tl.load(Threshold_ptr + hq).to(tl.float32)
        below = (row_sum < th) & valid
        first = tl.min(tl.where(below, offs, K_CLUSTERS), axis=0)
        depth = tl.where(first < K_CLUSTERS, first + 1, K_CLUSTERS)
        tl.store(Depth_ptr + hq, depth)
        tl.store(Counts_ptr + hq, tl.full((), 0, dtype=tl.int32))
        if hq == 0:
            stamp = tl.load(Stamp_ptr).to(tl.int32) + 1
            tl.store(Stamp_ptr, stamp)


    @triton.jit
    def _ta_v13_depth_count_stamp_rows_kernel(
        SortedScores_ptr,  # (H_q, 4, ROW_STRIDE) fp16/fp32
        Threshold_ptr,     # (H_q,) fp32
        Depth_ptr,         # (H_q,) int32
        Counts_ptr,        # (H_q,) int32
        Stamp_ptr,         # scalar int32
        ROW_STRIDE,
        ROWS,
        ROWS_BLOCK: tl.constexpr,
    ):
        hq = tl.program_id(0)
        offs = tl.arange(0, ROWS_BLOCK)
        valid = offs < ROWS

        row_sum = tl.zeros([ROWS_BLOCK], dtype=tl.float32)
        for s_idx in tl.static_range(4):
            vals = tl.load(
                SortedScores_ptr + (hq * 4 + s_idx) * ROW_STRIDE + offs,
                mask=valid,
                other=0.0,
            )
            row_sum += vals
        th = tl.load(Threshold_ptr + hq).to(tl.float32)
        below = (row_sum < th) & valid
        first = tl.min(tl.where(below, offs, ROWS), axis=0)
        depth = tl.where(first < ROWS, first + 1, ROWS)
        tl.store(Depth_ptr + hq, depth)
        tl.store(Counts_ptr + hq, tl.full((), 0, dtype=tl.int32))
        if hq == 0:
            stamp = tl.load(Stamp_ptr).to(tl.int32) + 1
            tl.store(Stamp_ptr, stamp)


    @triton.jit
    def _ta_v13_clear_mask_kernel(
        Mask_ptr,  # (H_q, N_pad) int8
        N_PAD,
        BLOCK_N: tl.constexpr,
    ):
        hq = tl.program_id(0)
        block = tl.program_id(1)
        offs = block * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = offs < N_PAD
        tl.store(Mask_ptr + hq * N_PAD + offs, tl.zeros([BLOCK_N], dtype=tl.int8), mask=valid)


    @triton.jit
    def _ta_v13_mark_mask_kernel(
        Order_ptr,       # (H_q, 4, K) int64/int32
        Depth_ptr,       # (H_q,) int32
        Children_ptr,    # (4, H_kv, K * bf) int32, parent-contiguous
        Mask_ptr,        # (H_q, N_pad) int8
        K_CLUSTERS,
        KBF,
        N_PAD,
        H_KV: tl.constexpr,
        GROUPS: tl.constexpr,
        BF: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        CANDS: tl.constexpr,
    ):
        hq = tl.program_id(0)
        s_idx = tl.program_id(1)
        row_block = tl.program_id(2)
        kvh = hq // GROUPS

        row_inner = tl.arange(0, BLOCK_ROWS)
        child_inner = tl.arange(0, BF)
        rows = row_block * BLOCK_ROWS + row_inner
        depth = tl.load(Depth_ptr + hq).to(tl.int32)
        row_valid = rows < depth

        clusters = tl.load(
            Order_ptr + (hq * 4 + s_idx) * K_CLUSTERS + rows,
            mask=row_valid,
            other=0,
        ).to(tl.int32)
        cluster_valid = row_valid & (clusters >= 0) & (clusters < K_CLUSTERS)
        clusters_safe = tl.where(cluster_valid, clusters, 0)

        key_ids = tl.load(
            Children_ptr
            + (s_idx * H_KV + kvh) * KBF
            + clusters_safe[:, None] * BF
            + child_inner[None, :],
            mask=cluster_valid[:, None],
            other=-1,
        ).to(tl.int32)

        flat = tl.reshape(key_ids, [CANDS])
        live = (flat >= 0) & (flat < N_PAD)
        tl.store(
            Mask_ptr + hq * N_PAD + flat,
            tl.full([CANDS], 1, dtype=tl.int8),
            mask=live,
        )


    @triton.jit
    def _ta_v13_mark_stamp_kernel(
        Order_ptr,       # (H_q, 4, K) int64/int32
        Depth_ptr,       # (H_q,) int32
        Children_ptr,    # (4, H_kv, K * bf) int32
        Visited_ptr,     # (H_q, N_pad) int32 stamps
        Stamp_ptr,       # scalar int32
        K_CLUSTERS,
        KBF,
        N_PAD,
        H_KV: tl.constexpr,
        GROUPS: tl.constexpr,
        BF: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        CANDS: tl.constexpr,
    ):
        hq = tl.program_id(0)
        s_idx = tl.program_id(1)
        row_block = tl.program_id(2)
        kvh = hq // GROUPS
        stamp = tl.load(Stamp_ptr).to(tl.int32)

        row_inner = tl.arange(0, BLOCK_ROWS)
        child_inner = tl.arange(0, BF)
        rows = row_block * BLOCK_ROWS + row_inner
        depth = tl.load(Depth_ptr + hq).to(tl.int32)
        row_valid = rows < depth

        clusters = tl.load(
            Order_ptr + (hq * 4 + s_idx) * K_CLUSTERS + rows,
            mask=row_valid,
            other=0,
        ).to(tl.int32)
        cluster_valid = row_valid & (clusters >= 0) & (clusters < K_CLUSTERS)
        clusters_safe = tl.where(cluster_valid, clusters, 0)

        key_ids = tl.load(
            Children_ptr
            + (s_idx * H_KV + kvh) * KBF
            + clusters_safe[:, None] * BF
            + child_inner[None, :],
            mask=cluster_valid[:, None],
            other=-1,
        ).to(tl.int32)

        flat = tl.reshape(key_ids, [CANDS])
        live = (flat >= 0) & (flat < N_PAD)
        tl.store(
            Visited_ptr + hq * N_PAD + flat,
            stamp + tl.zeros([CANDS], dtype=tl.int32),
            mask=live,
        )


    @triton.jit
    def _ta_v13_mark_stamp_rs_kernel(
        Order_ptr,       # (H_q, 4, ROW_STRIDE) int64/int32
        Depth_ptr,       # (H_q,) int32
        Children_ptr,    # (4, H_kv, K * bf) int32
        Visited_ptr,     # (H_q, N_pad) int32 stamps
        Stamp_ptr,       # scalar int32
        K_CLUSTERS,
        KBF,
        N_PAD,
        ROW_STRIDE,
        H_KV: tl.constexpr,
        GROUPS: tl.constexpr,
        BF: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        CANDS: tl.constexpr,
    ):
        hq = tl.program_id(0)
        s_idx = tl.program_id(1)
        row_block = tl.program_id(2)
        kvh = hq // GROUPS
        stamp = tl.load(Stamp_ptr).to(tl.int32)

        row_inner = tl.arange(0, BLOCK_ROWS)
        child_inner = tl.arange(0, BF)
        rows = row_block * BLOCK_ROWS + row_inner
        depth = tl.load(Depth_ptr + hq).to(tl.int32)
        row_valid = rows < depth

        clusters = tl.load(
            Order_ptr + (hq * 4 + s_idx) * ROW_STRIDE + rows,
            mask=row_valid,
            other=0,
        ).to(tl.int32)
        cluster_valid = row_valid & (clusters >= 0) & (clusters < K_CLUSTERS)
        clusters_safe = tl.where(cluster_valid, clusters, 0)

        key_ids = tl.load(
            Children_ptr
            + (s_idx * H_KV + kvh) * KBF
            + clusters_safe[:, None] * BF
            + child_inner[None, :],
            mask=cluster_valid[:, None],
            other=-1,
        ).to(tl.int32)

        flat = tl.reshape(key_ids, [CANDS])
        live = (flat >= 0) & (flat < N_PAD)
        tl.store(
            Visited_ptr + hq * N_PAD + flat,
            stamp + tl.zeros([CANDS], dtype=tl.int32),
            mask=live,
        )


    @triton.jit
    def _ta_v13_pack_candidates_stamp_kernel(
        Order_ptr,       # (H_q, 4, K) int64/int32
        Depth_ptr,       # (H_q,) int32
        Children_ptr,    # (4, H_kv, K * bf) int32
        Visited_ptr,     # (H_q, N_pad) int32 stamps
        Counts_ptr,      # (H_q,) int32
        CandIds_ptr,     # (H_q, N_pad) int32
        Stamp_ptr,       # scalar int32
        K_CLUSTERS,
        KBF,
        N_PAD,
        H_KV: tl.constexpr,
        GROUPS: tl.constexpr,
        BF: tl.constexpr,
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
        stamp = tl.load(Stamp_ptr).to(tl.int32)

        rows_per_split = (depth + NUM_SPLITS - 1) // NUM_SPLITS
        r_start = split * rows_per_split
        r_stop = tl.minimum(r_start + rows_per_split, depth)

        for r_chunk in range(r_start, r_stop, BLOCK_ROWS):
            rows = r_chunk + row_inner
            row_valid = rows < r_stop
            rows_safe = tl.where(row_valid, rows, 0)

            for s_idx in tl.static_range(4):
                clusters = tl.load(
                    Order_ptr + (hq * 4 + s_idx) * K_CLUSTERS + rows_safe,
                    mask=row_valid,
                    other=0,
                ).to(tl.int32)
                cluster_valid = row_valid & (clusters >= 0) & (clusters < K_CLUSTERS)
                clusters_safe = tl.where(cluster_valid, clusters, 0)
                key_ids = tl.load(
                    Children_ptr
                    + (s_idx * H_KV + kvh) * KBF
                    + clusters_safe[:, None] * BF
                    + child_inner[None, :],
                    mask=cluster_valid[:, None],
                    other=-1,
                ).to(tl.int32)

                flat = tl.reshape(key_ids, [CANDS])
                live = (flat >= 0) & (flat < N_PAD)
                key_safe = tl.where(live, flat, 0)
                stamp_vec = stamp + tl.zeros([CANDS], dtype=tl.int32)
                old = tl.atomic_xchg(
                    Visited_ptr + hq * N_PAD + key_safe,
                    stamp_vec,
                    mask=live,
                    sem="relaxed",
                )
                is_new = live & (old != stamp_vec)
                slots = tl.atomic_add(
                    Counts_ptr + hq + tl.zeros([CANDS], dtype=tl.int32),
                    tl.full([CANDS], 1, dtype=tl.int32),
                    mask=is_new,
                    sem="relaxed",
                )
                tl.store(
                    CandIds_ptr + hq * N_PAD + slots,
                    flat,
                    mask=is_new & (slots < N_PAD),
                )


    @triton.jit
    def _ta_v13_masked_key_attn_kernel(
        Q_ptr,           # (H_q, D) fp16
        Mask_ptr,        # (H_q, N_pad) int8
        KeysT_ptr,       # (H_kv, D, N_pad) fp16
        Values_ptr,      # (H_kv, N_pad, D_v) fp16
        BufKeysT_ptr,    # (H_kv, D, L_buf) fp16
        BufValues_ptr,   # (H_kv, L_buf, D_v) fp16
        BufInvalid_ptr,  # (H_kv, L_buf) int8
        M_out_ptr,       # (H_q, NUM_SPLITS) fp32
        L_out_ptr,       # (H_q, NUM_SPLITS) fp32
        O_out_ptr,       # (H_q, NUM_SPLITS, D_v) fp32
        N_PAD,
        SCALE_LOG2E: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        H_KV: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        HAS_BUFFER: tl.constexpr,
        L_BUF_MAX: tl.constexpr,
        BUF_COLS_PER_SPLIT: tl.constexpr,
    ):
        hq = tl.program_id(0)
        split = tl.program_id(1)
        kvh = hq // GROUPS

        d_range = tl.arange(0, D)
        dv_range = tl.arange(0, D_V)
        n_inner = tl.arange(0, BLOCK_N)

        q_vals = tl.load(Q_ptr + hq * D + d_range)

        m = tl.full((), -1.0e30, dtype=tl.float32)
        l_acc = tl.full((), 0.0, dtype=tl.float32)
        o_acc = tl.zeros([D_V], dtype=tl.float32)

        rows_per_split = (N_PAD + NUM_SPLITS - 1) // NUM_SPLITS
        n_start = split * rows_per_split
        n_stop = tl.minimum(n_start + rows_per_split, N_PAD)

        for n_chunk in range(n_start, n_stop, BLOCK_N):
            offs = n_chunk + n_inner
            valid = offs < n_stop
            mask_vals = tl.load(
                Mask_ptr + hq * N_PAD + offs,
                mask=valid,
                other=0,
            )
            live = valid & (mask_vals != 0)

            if tl.max(live.to(tl.int32), axis=0) != 0:
                keys = tl.load(
                    KeysT_ptr + (kvh * D + d_range[:, None]) * N_PAD + offs[None, :],
                    mask=live[None, :],
                    other=0.0,
                )
                raw = tl.dot(q_vals[None, :], keys)
                raw = tl.reshape(raw, [BLOCK_N]).to(tl.float32)
                scaled = tl.where(live, raw * SCALE_LOG2E, -1.0e30)

                chunk_max = tl.max(scaled, axis=0)
                m_new = tl.maximum(m, chunk_max)
                alpha = tl.exp2(m - m_new)
                pvals = tl.exp2(scaled - m_new)
                pvals = tl.where(live, pvals, 0.0)
                l_acc = alpha * l_acc + tl.sum(pvals, axis=0)

                values = tl.load(
                    Values_ptr + (kvh * N_PAD + offs[:, None]) * D_V + dv_range[None, :],
                    mask=live[:, None],
                    other=0.0,
                )
                o_acc = alpha * o_acc + tl.reshape(
                    tl.dot(pvals[None, :].to(tl.float16), values),
                    [D_V],
                )
                m = m_new

        if HAS_BUFFER:
            buf_start = split * BUF_COLS_PER_SPLIT
            buf_end = tl.minimum(buf_start + BUF_COLS_PER_SPLIT, L_BUF_MAX)
            b_inner = tl.arange(0, BUF_COLS_PER_SPLIT)
            buf_idx = buf_start + b_inner
            buf_valid = buf_idx < buf_end
            buf_safe = tl.where(buf_valid, buf_idx, 0)
            buf_inv = tl.load(
                BufInvalid_ptr + kvh * L_BUF_MAX + buf_safe,
                mask=buf_valid,
                other=1,
            )
            buf_live = buf_valid & (buf_inv == 0)

            if tl.max(buf_live.to(tl.int32), axis=0) != 0:
                buf_keys = tl.load(
                    BufKeysT_ptr
                    + (kvh * D + d_range[:, None]) * L_BUF_MAX
                    + buf_safe[None, :],
                    mask=buf_live[None, :],
                    other=0.0,
                )
                buf_raw = tl.dot(q_vals[None, :], buf_keys)
                buf_raw = tl.reshape(buf_raw, [BUF_COLS_PER_SPLIT]).to(tl.float32)
                buf_scaled = tl.where(buf_live, buf_raw * SCALE_LOG2E, -1.0e30)

                buf_max = tl.max(buf_scaled, axis=0)
                m_new = tl.maximum(m, buf_max)
                alpha = tl.exp2(m - m_new)
                bp = tl.exp2(buf_scaled - m_new)
                bp = tl.where(buf_live, bp, 0.0)
                l_acc = alpha * l_acc + tl.sum(bp, axis=0)

                buf_values = tl.load(
                    BufValues_ptr
                    + (kvh * L_BUF_MAX + buf_safe[:, None]) * D_V
                    + dv_range[None, :],
                    mask=buf_live[:, None],
                    other=0.0,
                )
                o_acc = alpha * o_acc + tl.reshape(
                    tl.dot(bp[None, :].to(tl.float16), buf_values),
                    [D_V],
                )
                m = m_new

        tl.store(M_out_ptr + hq * NUM_SPLITS + split, m)
        tl.store(L_out_ptr + hq * NUM_SPLITS + split, l_acc)
        tl.store(O_out_ptr + (hq * NUM_SPLITS + split) * D_V + dv_range, o_acc)


    @triton.jit
    def _ta_v13_stamp_key_attn_kernel(
        Q_ptr,           # (H_q, D) fp16
        Visited_ptr,     # (H_q, N_pad) int32
        Stamp_ptr,       # scalar int32
        KeysT_ptr,       # (H_kv, D, N_pad) fp16
        Values_ptr,      # (H_kv, N_pad, D_v) fp16
        BufKeysT_ptr,
        BufValues_ptr,
        BufInvalid_ptr,
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        N_PAD,
        SCALE_LOG2E: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        H_KV: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        HAS_BUFFER: tl.constexpr,
        L_BUF_MAX: tl.constexpr,
        BUF_COLS_PER_SPLIT: tl.constexpr,
    ):
        hq = tl.program_id(0)
        split = tl.program_id(1)
        kvh = hq // GROUPS

        d_range = tl.arange(0, D)
        dv_range = tl.arange(0, D_V)
        n_inner = tl.arange(0, BLOCK_N)
        stamp = tl.load(Stamp_ptr).to(tl.int32)

        q_vals = tl.load(Q_ptr + hq * D + d_range)
        m = tl.full((), -1.0e30, dtype=tl.float32)
        l_acc = tl.full((), 0.0, dtype=tl.float32)
        o_acc = tl.zeros([D_V], dtype=tl.float32)

        rows_per_split = (N_PAD + NUM_SPLITS - 1) // NUM_SPLITS
        n_start = split * rows_per_split
        n_stop = tl.minimum(n_start + rows_per_split, N_PAD)

        for n_chunk in range(n_start, n_stop, BLOCK_N):
            offs = n_chunk + n_inner
            valid = offs < n_stop
            stamps = tl.load(
                Visited_ptr + hq * N_PAD + offs,
                mask=valid,
                other=0,
            )
            live = valid & (stamps == stamp)

            if tl.max(live.to(tl.int32), axis=0) != 0:
                keys = tl.load(
                    KeysT_ptr + (kvh * D + d_range[:, None]) * N_PAD + offs[None, :],
                    mask=live[None, :],
                    other=0.0,
                )
                raw = tl.dot(q_vals[None, :], keys)
                raw = tl.reshape(raw, [BLOCK_N]).to(tl.float32)
                scaled = tl.where(live, raw * SCALE_LOG2E, -1.0e30)

                chunk_max = tl.max(scaled, axis=0)
                m_new = tl.maximum(m, chunk_max)
                alpha = tl.exp2(m - m_new)
                pvals = tl.exp2(scaled - m_new)
                pvals = tl.where(live, pvals, 0.0)
                l_acc = alpha * l_acc + tl.sum(pvals, axis=0)

                values = tl.load(
                    Values_ptr + (kvh * N_PAD + offs[:, None]) * D_V + dv_range[None, :],
                    mask=live[:, None],
                    other=0.0,
                )
                o_acc = alpha * o_acc + tl.reshape(
                    tl.dot(pvals[None, :].to(tl.float16), values),
                    [D_V],
                )
                m = m_new

        if HAS_BUFFER:
            buf_start = split * BUF_COLS_PER_SPLIT
            buf_end = tl.minimum(buf_start + BUF_COLS_PER_SPLIT, L_BUF_MAX)
            b_inner = tl.arange(0, BUF_COLS_PER_SPLIT)
            buf_idx = buf_start + b_inner
            buf_valid = buf_idx < buf_end
            buf_safe = tl.where(buf_valid, buf_idx, 0)
            buf_inv = tl.load(
                BufInvalid_ptr + kvh * L_BUF_MAX + buf_safe,
                mask=buf_valid,
                other=1,
            )
            buf_live = buf_valid & (buf_inv == 0)

            if tl.max(buf_live.to(tl.int32), axis=0) != 0:
                buf_keys = tl.load(
                    BufKeysT_ptr
                    + (kvh * D + d_range[:, None]) * L_BUF_MAX
                    + buf_safe[None, :],
                    mask=buf_live[None, :],
                    other=0.0,
                )
                buf_raw = tl.dot(q_vals[None, :], buf_keys)
                buf_raw = tl.reshape(buf_raw, [BUF_COLS_PER_SPLIT]).to(tl.float32)
                buf_scaled = tl.where(buf_live, buf_raw * SCALE_LOG2E, -1.0e30)

                buf_max = tl.max(buf_scaled, axis=0)
                m_new = tl.maximum(m, buf_max)
                alpha = tl.exp2(m - m_new)
                bp = tl.exp2(buf_scaled - m_new)
                bp = tl.where(buf_live, bp, 0.0)
                l_acc = alpha * l_acc + tl.sum(bp, axis=0)

                buf_values = tl.load(
                    BufValues_ptr
                    + (kvh * L_BUF_MAX + buf_safe[:, None]) * D_V
                    + dv_range[None, :],
                    mask=buf_live[:, None],
                    other=0.0,
                )
                o_acc = alpha * o_acc + tl.reshape(
                    tl.dot(bp[None, :].to(tl.float16), buf_values),
                    [D_V],
                )
                m = m_new

        tl.store(M_out_ptr + hq * NUM_SPLITS + split, m)
        tl.store(L_out_ptr + hq * NUM_SPLITS + split, l_acc)
        tl.store(O_out_ptr + (hq * NUM_SPLITS + split) * D_V + dv_range, o_acc)


def run_ta_v13_depth_from_sorted(
    sorted_scores: torch.Tensor,
    threshold_f32: torch.Tensor,
    depth_i32: torch.Tensor,
    *,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, k_clusters = sorted_scores.shape
    if int(s_sub) != 4:
        raise ValueError(f"v13 depth requires S=4, got {s_sub}")
    _ta_v13_depth_from_sorted_kernel[(int(h_q),)](
        sorted_scores,
        threshold_f32,
        depth_i32,
        int(k_clusters),
        K_BLOCK=_next_pow2(int(k_clusters)),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v13_depth_count_stamp(
    sorted_scores: torch.Tensor,
    threshold_f32: torch.Tensor,
    depth_i32: torch.Tensor,
    counts_i32: torch.Tensor,
    stamp_i32: torch.Tensor,
    *,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, k_clusters = sorted_scores.shape
    if int(s_sub) != 4:
        raise ValueError(f"v13 depth/stamp requires S=4, got {s_sub}")
    _ta_v13_depth_count_stamp_kernel[(int(h_q),)](
        sorted_scores,
        threshold_f32,
        depth_i32,
        counts_i32,
        stamp_i32,
        int(k_clusters),
        K_BLOCK=_next_pow2(int(k_clusters)),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v13_depth_count_stamp_rows(
    sorted_scores: torch.Tensor,
    threshold_f32: torch.Tensor,
    depth_i32: torch.Tensor,
    counts_i32: torch.Tensor,
    stamp_i32: torch.Tensor,
    *,
    rows: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, row_stride = sorted_scores.shape
    if int(s_sub) != 4:
        raise ValueError(f"v13 depth/stamp rows requires S=4, got {s_sub}")
    n_rows = int(row_stride if rows is None else rows)
    _ta_v13_depth_count_stamp_rows_kernel[(int(h_q),)](
        sorted_scores,
        threshold_f32,
        depth_i32,
        counts_i32,
        stamp_i32,
        int(row_stride),
        int(n_rows),
        ROWS_BLOCK=_next_pow2(int(n_rows)),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v13_clear_mask(
    mask_i8: torch.Tensor,
    *,
    block_n: int = 256,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, n_pad = mask_i8.shape
    grid = (int(h_q), (int(n_pad) + int(block_n) - 1) // int(block_n))
    _ta_v13_clear_mask_kernel[grid](
        mask_i8,
        int(n_pad),
        BLOCK_N=int(block_n),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v13_mark_stamp_rs(
    order_i64: torch.Tensor,
    depth_i32: torch.Tensor,
    parent_children_i32: torch.Tensor,
    visited_i32: torch.Tensor,
    stamp_i32: torch.Tensor,
    *,
    groups: int,
    block_rows: int = 64,
    rows_limit: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, row_stride = order_i64.shape
    s_pc, h_kv, kbf = parent_children_i32.shape
    if int(s_sub) != 4 or int(s_pc) != 4:
        raise ValueError(f"v13 mark stamp rs requires S=4, got order S={s_sub}, children S={s_pc}")
    bf = 4
    k_clusters = int(kbf) // bf
    n_pad = int(visited_i32.shape[-1])
    n_rows = int(row_stride if rows_limit is None else min(int(rows_limit), int(row_stride)))
    grid = (int(h_q), 4, (n_rows + int(block_rows) - 1) // int(block_rows))
    _ta_v13_mark_stamp_rs_kernel[grid](
        order_i64,
        depth_i32,
        parent_children_i32,
        visited_i32,
        stamp_i32,
        int(k_clusters),
        int(kbf),
        int(n_pad),
        int(row_stride),
        H_KV=int(h_kv),
        GROUPS=int(groups),
        BF=4,
        BLOCK_ROWS=int(block_rows),
        CANDS=int(block_rows) * 4,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v13_pack_candidates_stamp(
    order_i64: torch.Tensor,
    depth_i32: torch.Tensor,
    parent_children_i32: torch.Tensor,
    visited_i32: torch.Tensor,
    counts_i32: torch.Tensor,
    cand_ids_i32: torch.Tensor,
    stamp_i32: torch.Tensor,
    *,
    groups: int,
    block_rows: int,
    num_splits: int,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    h_q, s_sub, k_clusters = order_i64.shape
    s_pc, h_kv, kbf = parent_children_i32.shape
    if int(s_sub) != 4 or int(s_pc) != 4:
        raise ValueError(f"v13 pack requires S=4, got order S={s_sub}, children S={s_pc}")
    bf = int(kbf) // int(k_clusters)
    if bf != 4:
        raise ValueError(f"v13 pack requires bf=4, got bf={bf}")
    n_pad = int(visited_i32.shape[-1])
    _ta_v13_pack_candidates_stamp_kernel[(int(h_q), int(num_splits))](
        order_i64,
        depth_i32,
        parent_children_i32,
        visited_i32,
        counts_i32,
        cand_ids_i32,
        stamp_i32,
        int(k_clusters),
        int(kbf),
        int(n_pad),
        H_KV=int(h_kv),
        GROUPS=int(groups),
        BF=4,
        BLOCK_ROWS=int(block_rows),
        CANDS=int(block_rows) * 4,
        NUM_SPLITS=int(num_splits),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v13_mark_mask(
    order_i64: torch.Tensor,
    depth_i32: torch.Tensor,
    parent_children_i32: torch.Tensor,
    mask_i8: torch.Tensor,
    *,
    groups: int,
    block_rows: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, k_clusters = order_i64.shape
    s_pc, h_kv, kbf = parent_children_i32.shape
    if int(s_sub) != 4 or int(s_pc) != 4:
        raise ValueError(f"v13 mark mask requires S=4, got order S={s_sub}, children S={s_pc}")
    bf = int(kbf) // int(k_clusters)
    if bf != 4:
        raise ValueError(f"v13 mark mask requires bf=4, got bf={bf}")
    n_pad = int(mask_i8.shape[-1])
    grid = (int(h_q), 4, (int(k_clusters) + int(block_rows) - 1) // int(block_rows))
    _ta_v13_mark_mask_kernel[grid](
        order_i64,
        depth_i32,
        parent_children_i32,
        mask_i8,
        int(k_clusters),
        int(kbf),
        int(n_pad),
        H_KV=int(h_kv),
        GROUPS=int(groups),
        BF=4,
        BLOCK_ROWS=int(block_rows),
        CANDS=int(block_rows) * 4,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v13_mark_stamp(
    order_i64: torch.Tensor,
    depth_i32: torch.Tensor,
    parent_children_i32: torch.Tensor,
    visited_i32: torch.Tensor,
    stamp_i32: torch.Tensor,
    *,
    groups: int,
    block_rows: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, k_clusters = order_i64.shape
    s_pc, h_kv, kbf = parent_children_i32.shape
    if int(s_sub) != 4 or int(s_pc) != 4:
        raise ValueError(f"v13 mark stamp requires S=4, got order S={s_sub}, children S={s_pc}")
    bf = int(kbf) // int(k_clusters)
    if bf != 4:
        raise ValueError(f"v13 mark stamp requires bf=4, got bf={bf}")
    n_pad = int(visited_i32.shape[-1])
    grid = (int(h_q), 4, (int(k_clusters) + int(block_rows) - 1) // int(block_rows))
    _ta_v13_mark_stamp_kernel[grid](
        order_i64,
        depth_i32,
        parent_children_i32,
        visited_i32,
        stamp_i32,
        int(k_clusters),
        int(kbf),
        int(n_pad),
        H_KV=int(h_kv),
        GROUPS=int(groups),
        BF=4,
        BLOCK_ROWS=int(block_rows),
        CANDS=int(block_rows) * 4,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v13_masked_key_attn(
    q: torch.Tensor,
    mask_i8: torch.Tensor,
    keys_t_f16: torch.Tensor,
    values_f16: torch.Tensor,
    buf_keys_t_f16: torch.Tensor | None,
    buf_values_f16: torch.Tensor | None,
    buf_invalid_i8: torch.Tensor | None,
    *,
    scale_log2e: float,
    groups: int,
    block_n: int,
    num_splits: int,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    d = int(q.shape[-1])
    h_kv, _d, n_pad = keys_t_f16.shape
    d_v = int(values_f16.shape[-1])
    has_buffer = buf_keys_t_f16 is not None and int(buf_keys_t_f16.shape[-1]) > 0
    if has_buffer:
        l_buf_max = int(buf_keys_t_f16.shape[-1])
        buf_cols_per_split = _next_pow2_min16(
            max(1, (l_buf_max + int(num_splits) - 1) // int(num_splits))
        )
        bk = buf_keys_t_f16
        bv = buf_values_f16
        bi = buf_invalid_i8
    else:
        l_buf_max = 16
        buf_cols_per_split = 16
        bk = q
        bv = q
        bi = mask_i8

    _ta_v13_masked_key_attn_kernel[(int(q.shape[0]), int(num_splits))](
        q,
        mask_i8,
        keys_t_f16,
        values_f16,
        bk,
        bv,
        bi,
        out_m,
        out_l,
        out_o,
        int(n_pad),
        SCALE_LOG2E=float(scale_log2e),
        D=d,
        D_V=d_v,
        H_KV=int(h_kv),
        GROUPS=int(groups),
        BLOCK_N=int(block_n),
        NUM_SPLITS=int(num_splits),
        HAS_BUFFER=bool(has_buffer),
        L_BUF_MAX=int(l_buf_max),
        BUF_COLS_PER_SPLIT=int(buf_cols_per_split),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v13_stamp_key_attn(
    q: torch.Tensor,
    visited_i32: torch.Tensor,
    stamp_i32: torch.Tensor,
    keys_t_f16: torch.Tensor,
    values_f16: torch.Tensor,
    buf_keys_t_f16: torch.Tensor | None,
    buf_values_f16: torch.Tensor | None,
    buf_invalid_i8: torch.Tensor | None,
    *,
    scale_log2e: float,
    groups: int,
    block_n: int,
    num_splits: int,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    d = int(q.shape[-1])
    h_kv, _d, n_pad = keys_t_f16.shape
    d_v = int(values_f16.shape[-1])
    has_buffer = buf_keys_t_f16 is not None and int(buf_keys_t_f16.shape[-1]) > 0
    if has_buffer:
        l_buf_max = int(buf_keys_t_f16.shape[-1])
        buf_cols_per_split = _next_pow2_min16(
            max(1, (l_buf_max + int(num_splits) - 1) // int(num_splits))
        )
        bk = buf_keys_t_f16
        bv = buf_values_f16
        bi = buf_invalid_i8
    else:
        l_buf_max = 16
        buf_cols_per_split = 16
        bk = q
        bv = q
        bi = visited_i32

    _ta_v13_stamp_key_attn_kernel[(int(q.shape[0]), int(num_splits))](
        q,
        visited_i32,
        stamp_i32,
        keys_t_f16,
        values_f16,
        bk,
        bv,
        bi,
        out_m,
        out_l,
        out_o,
        int(n_pad),
        SCALE_LOG2E=float(scale_log2e),
        D=d,
        D_V=d_v,
        H_KV=int(h_kv),
        GROUPS=int(groups),
        BLOCK_N=int(block_n),
        NUM_SPLITS=int(num_splits),
        HAS_BUFFER=bool(has_buffer),
        L_BUF_MAX=int(l_buf_max),
        BUF_COLS_PER_SPLIT=int(buf_cols_per_split),
        num_warps=num_warps,
        num_stages=num_stages,
    )

