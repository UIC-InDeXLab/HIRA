"""v9.1 Triton kernels: fixed top-k survivor compaction and survivor attention.

These kernels intentionally do not reuse the older compact-candidate or
compact-attention implementations.  The attention kernel consumes only the
exact survivor list produced by the compaction kernel, so it is not a full
``N_pad`` scan.
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


def _next_pow2_min16(x: int) -> int:
    return max(_next_pow2(x), 16)


if HAS_TRITON:

    @triton.jit
    def _ta_v9_1_copy_q_th_kernel(
        QIn_ptr,
        ThIn_ptr,
        QOut_ptr,
        ThOut_ptr,
        H_Q,
        D,
        Q_BLOCK: tl.constexpr,
        TH_BLOCK: tl.constexpr,
    ):
        q_offs = tl.arange(0, Q_BLOCK)
        q_valid = q_offs < (H_Q * D)
        q = tl.load(QIn_ptr + q_offs, mask=q_valid, other=0.0)
        tl.store(QOut_ptr + q_offs, q, mask=q_valid)

        th_offs = tl.arange(0, TH_BLOCK)
        th_valid = th_offs < H_Q
        th = tl.load(ThIn_ptr + th_offs, mask=th_valid, other=0.0)
        tl.store(ThOut_ptr + th_offs, th, mask=th_valid)


    @triton.jit
    def _ta_v9_1_scatter_selected_kernel(
        TopOrder_ptr,        # (H_q, S, TOPK) int64/int32
        Selected_ptr,        # (H_q, S, K) int8
        K_CLUSTERS,
        TOPK,
        S_SUB: tl.constexpr,
        K_BLOCK: tl.constexpr,
        TOPK_BLOCK: tl.constexpr,
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

        t_offs = tl.arange(0, TOPK_BLOCK)
        t_valid = t_offs < TOPK
        clusters = tl.load(
            TopOrder_ptr + (hq * S_SUB + s_idx) * TOPK + t_offs,
            mask=t_valid,
            other=0,
        ).to(tl.int32)
        c_valid = t_valid & (clusters >= 0) & (clusters < K_CLUSTERS)
        tl.store(
            Selected_ptr + (hq * S_SUB + s_idx) * K_CLUSTERS + clusters,
            tl.full([TOPK_BLOCK], 1, dtype=tl.int8),
            mask=c_valid,
        )


    @triton.jit
    def _ta_v9_1_build_survivors_kernel(
        Q_ptr,                # (H_q, D) fp16
        TopOrder_ptr,         # (H_q, S, TOPK) int64/int32
        Selected_ptr,         # (H_q, S, K) int8
        Assigns_ptr,          # (S, H_kv, N_pad) int32
        Children_ptr,         # (S, H_kv, K, BF) int32
        InvalidMask_ptr,      # (H_kv, N_pad) int8
        KeysT_ptr,            # (H_kv, D, N_pad) fp16
        Threshold_ptr,        # (H_q,) fp32
        SurvivorIds_ptr,      # (H_q, MAX_SURVIVORS) int32
        SurvivorScores_ptr,   # (H_q, MAX_SURVIVORS) fp32 raw q.k
        Counts_ptr,           # (H_q,) int32
        TOPK,
        K_CLUSTERS,
        N_PAD,
        MAX_SURVIVORS,
        ASSIGN_STRIDE_S,
        SELECTED_STRIDE_HQ,
        SELECTED_STRIDE_S,
        S_SUB: tl.constexpr,
        H_KV: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        D: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        CANDS: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        hq = tl.program_id(0)
        s_idx = tl.program_id(1)
        split = tl.program_id(2)
        kvh = hq // GROUPS

        rows_per_split = (TOPK + NUM_SPLITS - 1) // NUM_SPLITS
        r_start = split * rows_per_split
        r_stop = tl.minimum(r_start + rows_per_split, TOPK)

        row_inner = tl.arange(0, BLOCK_ROWS)
        child_inner = tl.arange(0, BF)
        d_range = tl.arange(0, D)
        q_vals = tl.load(Q_ptr + hq * D + d_range)
        threshold = tl.load(Threshold_ptr + hq).to(tl.float32)

        for r_chunk in range(r_start, r_stop, BLOCK_ROWS):
            rows = r_chunk + row_inner
            row_valid = rows < r_stop
            rows_safe = tl.where(row_valid, rows, 0)

            clusters = tl.load(
                TopOrder_ptr + (hq * S_SUB + s_idx) * TOPK + rows_safe,
                mask=row_valid,
                other=0,
            ).to(tl.int32)
            cluster_valid = row_valid & (clusters >= 0) & (clusters < K_CLUSTERS)
            clusters_safe = tl.where(cluster_valid, clusters, 0)

            key_ids = tl.load(
                Children_ptr
                + ((((s_idx * H_KV + kvh) * K_CLUSTERS + clusters_safe[:, None]) * BF)
                + child_inner[None, :]),
                mask=cluster_valid[:, None],
                other=-1,
            ).to(tl.int32)
            key_flat = tl.reshape(key_ids, [CANDS])
            key_valid = (key_flat >= 0) & (key_flat < N_PAD)
            key_safe = tl.where(key_valid, key_flat, 0)

            inv = tl.load(
                InvalidMask_ptr + kvh * N_PAD + key_safe,
                mask=key_valid,
                other=1,
            )
            owned = key_valid & (inv == 0)

            # First-selected-subspace ownership removes duplicates across
            # subspaces without a global visited table or atomics on key ids.
            for p in tl.static_range(S_SUB):
                parent_p = tl.load(
                    Assigns_ptr + p * ASSIGN_STRIDE_S + kvh * N_PAD + key_safe,
                    mask=owned,
                    other=0,
                ).to(tl.int32)
                prev_sel = tl.load(
                    Selected_ptr
                    + hq * SELECTED_STRIDE_HQ
                    + p * SELECTED_STRIDE_S
                    + parent_p,
                    mask=owned,
                    other=0,
                )
                earlier_selected = (p < s_idx) & (prev_sel != 0)
                owned = owned & ~earlier_selected

            if tl.max(owned.to(tl.int32), axis=0) != 0:
                keys_tile = tl.load(
                    KeysT_ptr
                    + (kvh * D + d_range[:, None]) * N_PAD
                    + key_safe[None, :],
                    mask=owned[None, :],
                    other=0.0,
                )
                raw = tl.dot(q_vals[None, :], keys_tile)
                raw = tl.reshape(raw, [CANDS]).to(tl.float32)
                survive = owned & (raw >= threshold)

                slots = tl.atomic_add(
                    Counts_ptr + hq + tl.zeros([CANDS], dtype=tl.int32),
                    tl.full([CANDS], 1, dtype=tl.int32),
                    mask=survive,
                    sem="relaxed",
                )
                slots_ok = survive & (slots < MAX_SURVIVORS)
                tl.store(
                    SurvivorIds_ptr + hq * MAX_SURVIVORS + slots,
                    key_flat,
                    mask=slots_ok,
                )
                tl.store(
                    SurvivorScores_ptr + hq * MAX_SURVIVORS + slots,
                    raw,
                    mask=slots_ok,
                )


    @triton.jit
    def _ta_v9_1_attn_kernel(
        Q_ptr,                # (H_q, D) fp16, used for buffer only
        Values_ptr,           # (H_kv, N_pad, D_v) fp16
        SurvivorIds_ptr,      # (H_q, MAX_SURVIVORS) int32
        SurvivorScores_ptr,   # (H_q, MAX_SURVIVORS) fp32
        Counts_ptr,           # (H_q,) int32
        BufKeysT_ptr,         # (H_kv, D, L_buf) fp16
        BufValues_ptr,        # (H_kv, L_buf, D_v) fp16
        BufInvalid_ptr,       # (H_kv, L_buf) int8
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        N_PAD,
        MAX_SURVIVORS,
        SCALE_LOG2E: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
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

        m = tl.full((), -1.0e30, dtype=tl.float32)
        l_acc = tl.full((), 0.0, dtype=tl.float32)
        o_acc = tl.zeros([D_V], dtype=tl.float32)

        count = tl.minimum(tl.load(Counts_ptr + hq).to(tl.int32), MAX_SURVIVORS)
        cols_per_split = (count + NUM_SPLITS - 1) // NUM_SPLITS
        c_start = split * cols_per_split
        c_end = tl.minimum(c_start + cols_per_split, count)

        for c_chunk in range(c_start, c_end, BLOCK_N):
            c_idx = c_chunk + n_inner
            c_valid = c_idx < c_end
            c_safe = tl.where(c_valid, c_idx, 0)

            key_ids = tl.load(
                SurvivorIds_ptr + hq * MAX_SURVIVORS + c_safe,
                mask=c_valid,
                other=0,
            ).to(tl.int32)
            raw = tl.load(
                SurvivorScores_ptr + hq * MAX_SURVIVORS + c_safe,
                mask=c_valid,
                other=-1.0e30,
            ).to(tl.float32)
            scaled = tl.where(c_valid, raw * SCALE_LOG2E, -1.0e30)

            chunk_max = tl.max(scaled, axis=0)
            m_new = tl.maximum(m, chunk_max)
            alpha = tl.exp2(m - m_new)
            p = tl.exp2(scaled - m_new)
            p = tl.where(c_valid, p, 0.0)
            l_acc = alpha * l_acc + tl.sum(p, axis=0)

            v_tile = tl.load(
                Values_ptr
                + (kvh * N_PAD + key_ids[:, None]) * D_V
                + dv_range[None, :],
                mask=c_valid[:, None],
                other=0.0,
            )
            o_acc = alpha * o_acc + tl.reshape(
                tl.dot(p[None, :].to(tl.float16), v_tile),
                [D_V],
            )
            m = m_new

        if HAS_BUFFER:
            q_vals = tl.load(Q_ptr + hq * D + d_range)
            b_start = split * BUF_COLS_PER_SPLIT
            b_end = tl.minimum(b_start + BUF_COLS_PER_SPLIT, L_BUF_MAX)
            b_inner = tl.arange(0, BUF_COLS_PER_SPLIT)
            b_idx = b_start + b_inner
            b_valid = b_idx < b_end
            b_safe = tl.where(b_valid, b_idx, 0)
            b_inv = tl.load(
                BufInvalid_ptr + kvh * L_BUF_MAX + b_safe,
                mask=b_valid,
                other=1,
            )
            b_live = b_valid & (b_inv == 0)

            if tl.max(b_live.to(tl.int32), axis=0) != 0:
                b_keys = tl.load(
                    BufKeysT_ptr
                    + (kvh * D + d_range[:, None]) * L_BUF_MAX
                    + b_safe[None, :],
                    mask=b_live[None, :],
                    other=0.0,
                )
                b_raw = tl.dot(q_vals[None, :], b_keys)
                b_raw = tl.reshape(b_raw, [BUF_COLS_PER_SPLIT]).to(tl.float32)
                b_scaled = tl.where(b_live, b_raw * SCALE_LOG2E, -1.0e30)

                b_max = tl.max(b_scaled, axis=0)
                m_new = tl.maximum(m, b_max)
                alpha = tl.exp2(m - m_new)
                bp = tl.exp2(b_scaled - m_new)
                bp = tl.where(b_live, bp, 0.0)
                l_acc = alpha * l_acc + tl.sum(bp, axis=0)

                b_values = tl.load(
                    BufValues_ptr
                    + (kvh * L_BUF_MAX + b_safe[:, None]) * D_V
                    + dv_range[None, :],
                    mask=b_live[:, None],
                    other=0.0,
                )
                o_acc = alpha * o_acc + tl.reshape(
                    tl.dot(bp[None, :].to(tl.float16), b_values),
                    [D_V],
                )
                m = m_new

        tl.store(M_out_ptr + hq * NUM_SPLITS + split, m)
        tl.store(L_out_ptr + hq * NUM_SPLITS + split, l_acc)
        tl.store(O_out_ptr + (hq * NUM_SPLITS + split) * D_V + dv_range, o_acc)


    @triton.jit
    def _ta_v9_1_attn_direct_kernel(
        Q_ptr,                # (H_q, D) fp16, used for buffer only
        Values_ptr,           # (H_kv, N_pad, D_v) fp16
        SurvivorIds_ptr,      # (H_q, MAX_SURVIVORS) int32
        SurvivorScores_ptr,   # (H_q, MAX_SURVIVORS) fp32
        Counts_ptr,           # (H_q,) int32
        BufKeysT_ptr,         # (H_kv, D, L_buf) fp16
        BufValues_ptr,        # (H_kv, L_buf, D_v) fp16
        BufInvalid_ptr,       # (H_kv, L_buf) int8
        Out_ptr,              # (H_q, D_v) fp32
        N_PAD,
        MAX_SURVIVORS,
        SCALE_LOG2E: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_BUFFER: tl.constexpr,
        L_BUF_MAX: tl.constexpr,
        BUF_BLOCK: tl.constexpr,
    ):
        hq = tl.program_id(0)
        kvh = hq // GROUPS

        d_range = tl.arange(0, D)
        dv_range = tl.arange(0, D_V)
        n_inner = tl.arange(0, BLOCK_N)

        m = tl.full((), -1.0e30, dtype=tl.float32)
        l_acc = tl.full((), 0.0, dtype=tl.float32)
        o_acc = tl.zeros([D_V], dtype=tl.float32)

        count = tl.minimum(tl.load(Counts_ptr + hq).to(tl.int32), MAX_SURVIVORS)
        for c_chunk in range(0, count, BLOCK_N):
            c_idx = c_chunk + n_inner
            c_valid = c_idx < count
            c_safe = tl.where(c_valid, c_idx, 0)

            key_ids = tl.load(
                SurvivorIds_ptr + hq * MAX_SURVIVORS + c_safe,
                mask=c_valid,
                other=0,
            ).to(tl.int32)
            raw = tl.load(
                SurvivorScores_ptr + hq * MAX_SURVIVORS + c_safe,
                mask=c_valid,
                other=-1.0e30,
            ).to(tl.float32)
            scaled = tl.where(c_valid, raw * SCALE_LOG2E, -1.0e30)

            chunk_max = tl.max(scaled, axis=0)
            m_new = tl.maximum(m, chunk_max)
            alpha = tl.exp2(m - m_new)
            p = tl.exp2(scaled - m_new)
            p = tl.where(c_valid, p, 0.0)
            l_acc = alpha * l_acc + tl.sum(p, axis=0)

            v_tile = tl.load(
                Values_ptr
                + (kvh * N_PAD + key_ids[:, None]) * D_V
                + dv_range[None, :],
                mask=c_valid[:, None],
                other=0.0,
            )
            o_acc = alpha * o_acc + tl.reshape(
                tl.dot(p[None, :].to(tl.float16), v_tile),
                [D_V],
            )
            m = m_new

        if HAS_BUFFER:
            q_vals = tl.load(Q_ptr + hq * D + d_range)
            b_inner = tl.arange(0, BUF_BLOCK)
            for b_chunk in range(0, L_BUF_MAX, BUF_BLOCK):
                b_idx = b_chunk + b_inner
                b_valid = b_idx < L_BUF_MAX
                b_safe = tl.where(b_valid, b_idx, 0)
                b_inv = tl.load(
                    BufInvalid_ptr + kvh * L_BUF_MAX + b_safe,
                    mask=b_valid,
                    other=1,
                )
                b_live = b_valid & (b_inv == 0)

                if tl.max(b_live.to(tl.int32), axis=0) != 0:
                    b_keys = tl.load(
                        BufKeysT_ptr
                        + (kvh * D + d_range[:, None]) * L_BUF_MAX
                        + b_safe[None, :],
                        mask=b_live[None, :],
                        other=0.0,
                    )
                    b_raw = tl.dot(q_vals[None, :], b_keys)
                    b_raw = tl.reshape(b_raw, [BUF_BLOCK]).to(tl.float32)
                    b_scaled = tl.where(b_live, b_raw * SCALE_LOG2E, -1.0e30)

                    b_max = tl.max(b_scaled, axis=0)
                    m_new = tl.maximum(m, b_max)
                    alpha = tl.exp2(m - m_new)
                    bp = tl.exp2(b_scaled - m_new)
                    bp = tl.where(b_live, bp, 0.0)
                    l_acc = alpha * l_acc + tl.sum(bp, axis=0)

                    b_values = tl.load(
                        BufValues_ptr
                        + (kvh * L_BUF_MAX + b_safe[:, None]) * D_V
                        + dv_range[None, :],
                        mask=b_live[:, None],
                        other=0.0,
                    )
                    o_acc = alpha * o_acc + tl.reshape(
                        tl.dot(bp[None, :].to(tl.float16), b_values),
                        [D_V],
                    )
                    m = m_new

        l_safe = tl.where(l_acc > 0.0, l_acc, 1.0)
        tl.store(Out_ptr + hq * D_V + dv_range, o_acc / l_safe)


def run_ta_v9_1_scatter_selected(
    top_order_i32: torch.Tensor,
    selected_i8: torch.Tensor,
    *,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, topk = top_order_i32.shape
    k_clusters = int(selected_i8.shape[-1])
    _ta_v9_1_scatter_selected_kernel[(int(h_q), int(s_sub))](
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


def run_ta_v9_1_copy_q_th(
    q_in: torch.Tensor,
    th_in: torch.Tensor,
    q_out: torch.Tensor,
    th_out: torch.Tensor,
) -> None:
    h_q, d = q_in.shape
    _ta_v9_1_copy_q_th_kernel[(1,)](
        q_in,
        th_in,
        q_out,
        th_out,
        int(h_q),
        int(d),
        Q_BLOCK=_next_pow2(int(h_q) * int(d)),
        TH_BLOCK=_next_pow2_min16(int(h_q)),
        num_warps=4,
        num_stages=1,
    )


def run_ta_v9_1_build_survivors(
    q: torch.Tensor,
    top_order_i32: torch.Tensor,
    selected_i8: torch.Tensor,
    assigns_i32: torch.Tensor,
    children_i32: torch.Tensor,
    invalid_mask_i8: torch.Tensor,
    keys_t_f16: torch.Tensor,
    threshold_f32: torch.Tensor,
    survivor_ids_i32: torch.Tensor,
    survivor_scores_f32: torch.Tensor,
    counts_i32: torch.Tensor,
    *,
    groups: int,
    block_rows: int = 8,
    num_splits: int = 16,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    h_q, s_sub, topk = top_order_i32.shape
    _s, h_kv, n_pad = assigns_i32.shape
    _sc, _hc, k_clusters, bf = children_i32.shape
    d = int(q.shape[-1])
    cands = int(block_rows) * int(bf)
    _ta_v9_1_build_survivors_kernel[(int(h_q), int(s_sub), int(num_splits))](
        q,
        top_order_i32,
        selected_i8,
        assigns_i32,
        children_i32,
        invalid_mask_i8,
        keys_t_f16,
        threshold_f32,
        survivor_ids_i32,
        survivor_scores_f32,
        counts_i32,
        int(topk),
        int(k_clusters),
        int(n_pad),
        int(survivor_ids_i32.shape[-1]),
        int(h_kv) * int(n_pad),
        int(s_sub) * int(k_clusters),
        int(k_clusters),
        S_SUB=int(s_sub),
        H_KV=int(h_kv),
        BF=int(bf),
        GROUPS=int(groups),
        D=d,
        BLOCK_ROWS=int(block_rows),
        CANDS=cands,
        NUM_SPLITS=int(num_splits),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v9_1_attn(
    q: torch.Tensor,
    values_f16: torch.Tensor,
    survivor_ids_i32: torch.Tensor,
    survivor_scores_f32: torch.Tensor,
    counts_i32: torch.Tensor,
    buf_keys_t_f16: torch.Tensor | None,
    buf_values_f16: torch.Tensor | None,
    buf_invalid_i8: torch.Tensor | None,
    *,
    n_pad: int,
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
        bi = counts_i32

    _ta_v9_1_attn_kernel[(int(q.shape[0]), int(num_splits))](
        q,
        values_f16,
        survivor_ids_i32,
        survivor_scores_f32,
        counts_i32,
        bk,
        bv,
        bi,
        out_m,
        out_l,
        out_o,
        int(n_pad),
        int(survivor_ids_i32.shape[-1]),
        SCALE_LOG2E=float(scale_log2e),
        D=d,
        D_V=d_v,
        GROUPS=int(groups),
        BLOCK_N=int(block_n),
        NUM_SPLITS=int(num_splits),
        HAS_BUFFER=bool(has_buffer),
        L_BUF_MAX=int(l_buf_max),
        BUF_COLS_PER_SPLIT=int(buf_cols_per_split),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v9_1_attn_direct(
    q: torch.Tensor,
    values_f16: torch.Tensor,
    survivor_ids_i32: torch.Tensor,
    survivor_scores_f32: torch.Tensor,
    counts_i32: torch.Tensor,
    buf_keys_t_f16: torch.Tensor | None,
    buf_values_f16: torch.Tensor | None,
    buf_invalid_i8: torch.Tensor | None,
    out: torch.Tensor,
    *,
    n_pad: int,
    scale_log2e: float,
    groups: int,
    block_n: int,
    buf_block: int,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    d = int(q.shape[-1])
    d_v = int(values_f16.shape[-1])
    has_buffer = buf_keys_t_f16 is not None and int(buf_keys_t_f16.shape[-1]) > 0
    if has_buffer:
        l_buf_max = int(buf_keys_t_f16.shape[-1])
        bk = buf_keys_t_f16
        bv = buf_values_f16
        bi = buf_invalid_i8
    else:
        l_buf_max = 16
        bk = q
        bv = q
        bi = counts_i32
    _ta_v9_1_attn_direct_kernel[(int(q.shape[0]),)](
        q,
        values_f16,
        survivor_ids_i32,
        survivor_scores_f32,
        counts_i32,
        bk,
        bv,
        bi,
        out,
        int(n_pad),
        int(survivor_ids_i32.shape[-1]),
        SCALE_LOG2E=float(scale_log2e),
        D=d,
        D_V=d_v,
        GROUPS=int(groups),
        BLOCK_N=int(block_n),
        HAS_BUFFER=bool(has_buffer),
        L_BUF_MAX=int(l_buf_max),
        BUF_BLOCK=int(buf_block),
        num_warps=num_warps,
        num_stages=num_stages,
    )
