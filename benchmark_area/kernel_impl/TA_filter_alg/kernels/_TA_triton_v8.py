"""v8.0 Triton kernels: cluster-iteration TA-filter attention.

Two kernels:

  1. ``_ta_v8_build_cand_kernel`` — fuses two outputs from
     ``selected[H_q, S, K]`` (int8) and the existing index layout:
       * ``cand_b[H_q, K, BF]``  int8 — per-head per-(s=0 cluster, child-pos)
         candidate flag.  Cluster-aligned layout means the attention kernel
         loads cand_b with a single contiguous (groups, BF) tile per cluster.
       * ``cluster_active[H_kv, K]`` int8 — OR over heads-in-group AND the
         BF children → cluster has ≥1 candidate for any head in the group.

     Per (kvh, c) it loads BF child key indices, then for each child does
     ``OR_s selected[h_q, s, assigns[s, kvh, child]]`` for every head in the
     KV group.  Padded children (``children == -1``) are masked out and
     padded keys (``invalid_mask``) are filtered.

  2. ``_ta_v8_attn_kernel`` — split-K parent-iteration attention.  Pattern
     mirrors v5.14: each (kvh, split) program walks parents in the s=0
     partition in chunks of ``PARENTS_PER_PROG``, skipping chunks whose
     ``cluster_active`` is uniformly zero.  Live chunks gather a
     (D, PARENTS_PER_PROG*BF) cluster-contiguous keys tile from the s=0
     slice of ``cluster_keys_t_f16``, perform ``q @ keys``, mask by
     ``cand_b`` and the per-head threshold T, do online softmax, then
     accumulate ``p @ V`` from ``cluster_values_f16``.  Phase-2 buffer
     pass is identical to v6.0 so results merge cleanly through the
     same reduce kernel.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


def _next_pow2_min16(x: int) -> int:
    p = 16
    while p < x:
        p *= 2
    return p


if HAS_TRITON:

    @triton.jit
    def _ta_v8_build_cand_kernel(
        Selected_ptr,        # (H_q, S, K) int8
        Assigns_ptr,         # (S, H_kv, N_pad) int32 -- caller passes int32 view
        Children_ptr,        # (S, H_kv, K, BF) int32 -- s=0 slice base passed in
        InvalidMask_ptr,     # (H_kv, N_pad) int8
        CandB_ptr,           # (H_q, K, BF) int8 -- output
        ClusterActive_ptr,   # (H_kv, K) int8 -- output
        K_CLUSTERS,
        N_PAD,
        ASSIGN_STRIDE_S,     # = H_kv * N_pad
        SELECTED_STRIDE_HQ,  # = S * K
        SELECTED_STRIDE_S,   # = K
        S_SUB: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        KCHUNK: tl.constexpr,
    ):
        kvh = tl.program_id(0)
        c_chunk = tl.program_id(1)

        c_range = c_chunk * KCHUNK + tl.arange(0, KCHUNK)
        c_valid = c_range < K_CLUSTERS
        c_safe = tl.where(c_valid, c_range, 0)

        b_range = tl.arange(0, BF)
        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        hq_vec = kvh * GROUPS + g_range

        # Load child keys for this cluster chunk: (KCHUNK, BF) int32.
        child_keys = tl.load(
            Children_ptr
            + (kvh * K_CLUSTERS + c_safe[:, None]) * BF
            + b_range[None, :],
            mask=c_valid[:, None],
            other=-1,
        ).to(tl.int32)
        valid_child = (child_keys >= 0) & c_valid[:, None]
        safe_child = tl.where(valid_child, child_keys, 0)

        # Filter out padded keys (invalid_mask).
        inv = tl.load(
            InvalidMask_ptr + kvh * N_PAD + safe_child,
            mask=valid_child,
            other=1,
        )
        valid_child = valid_child & (inv == 0)

        # Accumulator: (GROUPS_POW, KCHUNK, BF) int8 cand bits.
        cand = tl.zeros([GROUPS_POW, KCHUNK, BF], dtype=tl.int8)
        for s in tl.static_range(S_SUB):
            parent_s = tl.load(
                Assigns_ptr + s * ASSIGN_STRIDE_S + kvh * N_PAD + safe_child,
                mask=valid_child,
                other=0,
            ).to(tl.int32)
            sel = tl.load(
                Selected_ptr
                + hq_vec[:, None, None] * SELECTED_STRIDE_HQ
                + s * SELECTED_STRIDE_S
                + parent_s[None, :, :],
                mask=g_valid[:, None, None] & valid_child[None, :, :],
                other=0,
            )
            cand = cand | sel

        # Mask out padded children (defensively).
        cand = tl.where(g_valid[:, None, None] & valid_child[None, :, :], cand, tl.zeros_like(cand))

        # Store cand_b[h_q, c, b] for each head in group.
        tl.store(
            CandB_ptr
            + hq_vec[:, None, None] * (K_CLUSTERS * BF)
            + c_safe[None, :, None] * BF
            + b_range[None, None, :],
            cand,
            mask=g_valid[:, None, None] & c_valid[None, :, None],
        )

        # Cluster-active: OR over heads-in-group AND BF children.
        active_per_c = tl.max(cand.to(tl.int32), axis=2)        # (GROUPS_POW, KCHUNK)
        active_per_c = tl.max(active_per_c, axis=0)             # (KCHUNK,)
        tl.store(
            ClusterActive_ptr + kvh * K_CLUSTERS + c_safe,
            active_per_c.to(tl.int8),
            mask=c_valid,
        )


def run_ta_v8_build_cand(
    selected_i8: torch.Tensor,        # (H_q, S, K) int8
    assigns_i32: torch.Tensor,        # (S, H_kv, N_pad) int32
    children_s0_i32: torch.Tensor,    # (H_kv, K, BF) int32 -- s=0 slice contiguous
    invalid_mask_i8: torch.Tensor,    # (H_kv, N_pad) int8
    cand_b_i8: torch.Tensor,          # (H_q, K, BF) int8 (overwritten)
    cluster_active_i8: torch.Tensor,  # (H_kv, K) int8 (overwritten)
    *,
    groups: int,
    kchunk: int = 8,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, k_clusters = selected_i8.shape
    _s, h_kv, n_pad = assigns_i32.shape
    _, _, bf = children_s0_i32.shape

    groups_pow = max(_next_pow2_min16(groups) // 1, 4)
    if groups_pow < 4:
        groups_pow = 4
    # min 16 for tl loads; but for OR over a small dim we just need pow2. 4 is fine.
    p = 1
    while p < groups:
        p *= 2
    groups_pow = max(p, 4)

    grid = (int(h_kv), (int(k_clusters) + kchunk - 1) // kchunk)
    _ta_v8_build_cand_kernel[grid](
        selected_i8,
        assigns_i32,
        children_s0_i32,
        invalid_mask_i8,
        cand_b_i8,
        cluster_active_i8,
        int(k_clusters),
        int(n_pad),
        int(h_kv) * int(n_pad),
        int(s_sub) * int(k_clusters),
        int(k_clusters),
        S_SUB=int(s_sub),
        BF=int(bf),
        GROUPS=int(groups),
        GROUPS_POW=int(groups_pow),
        KCHUNK=int(kchunk),
        num_warps=num_warps,
        num_stages=num_stages,
    )


if HAS_TRITON:

    @triton.jit
    def _ta_v8_attn_kernel(
        Q_ptr,                  # (H_q, D) fp16
        ClusterKeysT_ptr,       # (H_kv, K, D, BF) fp16  -- s=0 slice base
        ClusterValues_ptr,      # (H_kv, K, BF, D_v) fp16 -- s=0 slice base
        CandB_ptr,              # (H_q, K, BF) int8
        ClusterActive_ptr,      # (H_kv, K) int8
        Threshold_ptr,          # (H_q,) fp32
        BufKeysT_ptr,           # (H_kv, D, L_buf) fp16
        BufValues_ptr,          # (H_kv, L_buf, D_v) fp16
        BufInvalid_ptr,         # (H_kv, L_buf) int8
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        K_CLUSTERS,
        SCALE_LOG2E: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        PARENTS_PER_PROG: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        HAS_BUFFER: tl.constexpr,
        L_BUF_MAX: tl.constexpr,
        BUF_COLS_PER_SPLIT: tl.constexpr,
    ):
        kvh = tl.program_id(0)
        split = tl.program_id(1)

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        hq_vec = kvh * GROUPS + g_range

        d_range = tl.arange(0, D)
        dv_range = tl.arange(0, D_V)

        q_f16 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        th_raw = tl.load(
            Threshold_ptr + hq_vec, mask=g_valid, other=float("inf"),
        ).to(tl.float32)

        m = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
        l_acc = tl.zeros([GROUPS_POW], dtype=tl.float32)
        o_acc = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

        parents_per_split = (K_CLUSTERS + NUM_SPLITS - 1) // NUM_SPLITS
        p_start = split * parents_per_split
        p_end = tl.minimum(p_start + parents_per_split, K_CLUSTERS)

        cols = tl.arange(0, PARENTS_PER_PROG * BF)
        parent_rel = cols // BF
        child_rel = cols % BF
        parent_rel_p = tl.arange(0, PARENTS_PER_PROG)

        for p_chunk_start in range(p_start, p_end, PARENTS_PER_PROG):
            parent_idx_p = p_chunk_start + parent_rel_p
            col_valid_p = parent_idx_p < p_end
            parent_idx_p_safe = tl.where(col_valid_p, parent_idx_p, 0)

            active = tl.load(
                ClusterActive_ptr + kvh * K_CLUSTERS + parent_idx_p_safe,
                mask=col_valid_p,
                other=0,
            )
            any_active = tl.max(active.to(tl.int32))
            if any_active != 0:
                parent_idx = p_chunk_start + parent_rel
                col_valid = parent_idx < p_end
                parent_idx_safe = tl.where(col_valid, parent_idx, 0)

                cand_p = tl.load(
                    CandB_ptr
                    + hq_vec[:, None] * (K_CLUSTERS * BF)
                    + parent_idx_safe[None, :] * BF
                    + child_rel[None, :],
                    mask=g_valid[:, None] & col_valid[None, :],
                    other=0,
                )
                survive_pre = (cand_p != 0) & g_valid[:, None] & col_valid[None, :]

                live_cols = tl.max(survive_pre.to(tl.int32), axis=0) != 0
                if tl.max(live_cols.to(tl.int32), axis=0) != 0:
                    keys_tile = tl.load(
                        ClusterKeysT_ptr
                        + (kvh * K_CLUSTERS + parent_idx_safe[None, :]) * (D * BF)
                        + d_range[:, None] * BF
                        + child_rel[None, :],
                        mask=live_cols[None, :],
                        other=0.0,
                    )
                    raw_scores = tl.dot(q_f16, keys_tile)
                    pass_t = raw_scores >= th_raw[:, None]
                    survive = survive_pre & pass_t
                    scaled = raw_scores * SCALE_LOG2E
                    scaled = tl.where(survive, scaled, -1.0e30)

                    chunk_max = tl.max(scaled, axis=1)
                    m_new = tl.maximum(m, chunk_max)
                    alpha = tl.exp2(m - m_new)
                    p = tl.exp2(scaled - m_new[:, None])
                    p = tl.where(survive, p, 0.0)
                    l_acc = alpha * l_acc + tl.sum(p, axis=1)

                    v_tile = tl.load(
                        ClusterValues_ptr
                        + (kvh * K_CLUSTERS + parent_idx_safe[:, None]) * (BF * D_V)
                        + child_rel[:, None] * D_V
                        + dv_range[None, :],
                        mask=live_cols[:, None],
                        other=0.0,
                    )
                    o_acc = alpha[:, None] * o_acc + tl.dot(p.to(tl.float16), v_tile)
                    m = m_new

        if HAS_BUFFER:
            buf_start = split * BUF_COLS_PER_SPLIT
            buf_end = tl.minimum(buf_start + BUF_COLS_PER_SPLIT, L_BUF_MAX)
            b_inner = tl.arange(0, BUF_COLS_PER_SPLIT)
            buf_idx = buf_start + b_inner
            buf_valid_col = buf_idx < buf_end
            buf_idx_safe = tl.where(buf_valid_col, buf_idx, 0)
            buf_inv = tl.load(
                BufInvalid_ptr + kvh * L_BUF_MAX + buf_idx_safe,
                mask=buf_valid_col,
                other=1,
            )
            buf_live = buf_valid_col & (buf_inv == 0)
            buf_survive = g_valid[:, None] & buf_live[None, :]

            any_buf = tl.max(buf_live.to(tl.int32))
            if any_buf != 0:
                buf_keys_tile = tl.load(
                    BufKeysT_ptr
                    + (kvh * D + d_range[:, None]) * L_BUF_MAX
                    + buf_idx_safe[None, :],
                    mask=buf_live[None, :],
                    other=0.0,
                )
                buf_raw = tl.dot(q_f16, buf_keys_tile)
                buf_scaled = buf_raw * SCALE_LOG2E
                buf_scaled = tl.where(buf_survive, buf_scaled, -1.0e30)

                buf_chunk_max = tl.max(buf_scaled, axis=1)
                m_new = tl.maximum(m, buf_chunk_max)
                alpha = tl.exp2(m - m_new)
                bp = tl.exp2(buf_scaled - m_new[:, None])
                bp = tl.where(buf_survive, bp, 0.0)
                l_acc = alpha * l_acc + tl.sum(bp, axis=1)

                buf_v_tile = tl.load(
                    BufValues_ptr
                    + (kvh * L_BUF_MAX + buf_idx_safe[:, None]) * D_V
                    + dv_range[None, :],
                    mask=buf_live[:, None],
                    other=0.0,
                )
                o_acc = alpha[:, None] * o_acc + tl.dot(bp.to(tl.float16), buf_v_tile)
                m = m_new

        tl.store(M_out_ptr + hq_vec * NUM_SPLITS + split, m, mask=g_valid)
        tl.store(L_out_ptr + hq_vec * NUM_SPLITS + split, l_acc, mask=g_valid)
        tl.store(
            O_out_ptr
            + (hq_vec[:, None] * NUM_SPLITS + split) * D_V
            + dv_range[None, :],
            o_acc,
            mask=g_valid[:, None],
        )


def run_ta_v8_attn(
    q: torch.Tensor,
    cluster_keys_t_s0_f16: torch.Tensor,    # (H_kv, K, D, BF) fp16
    cluster_values_s0_f16: torch.Tensor,    # (H_kv, K, BF, D_v) fp16
    cand_b_i8: torch.Tensor,                # (H_q, K, BF) int8
    cluster_active_i8: torch.Tensor,        # (H_kv, K) int8
    threshold_f32: torch.Tensor,            # (H_q,) fp32
    buf_keys_t_f16: torch.Tensor | None,
    buf_values_f16: torch.Tensor | None,
    buf_invalid_i8: torch.Tensor | None,
    h_kv_eff: int,
    k_clusters: int,
    scale_log2e: float,
    groups: int,
    groups_pow: int,
    parents_per_prog: int,
    num_splits: int,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    d = int(q.shape[1])
    d_v = int(cluster_values_s0_f16.shape[-1])
    bf = int(cluster_values_s0_f16.shape[-2])
    has_buffer = buf_keys_t_f16 is not None and int(buf_keys_t_f16.shape[-1]) > 0
    if has_buffer:
        l_buf_max = int(buf_keys_t_f16.shape[-1])
        buf_cols_per_split = _next_pow2_min16(
            max(1, (l_buf_max + num_splits - 1) // num_splits)
        )
        buf_keys_arg = buf_keys_t_f16
        buf_values_arg = buf_values_f16
        buf_invalid_arg = buf_invalid_i8
    else:
        l_buf_max = 16
        buf_cols_per_split = 16
        buf_keys_arg = q
        buf_values_arg = q
        buf_invalid_arg = cluster_active_i8  # any non-null int8

    grid = (h_kv_eff, num_splits)
    _ta_v8_attn_kernel[grid](
        q,
        cluster_keys_t_s0_f16,
        cluster_values_s0_f16,
        cand_b_i8,
        cluster_active_i8,
        threshold_f32,
        buf_keys_arg,
        buf_values_arg,
        buf_invalid_arg,
        out_m,
        out_l,
        out_o,
        int(k_clusters),
        SCALE_LOG2E=float(scale_log2e),
        D=int(d),
        D_V=int(d_v),
        BF=int(bf),
        GROUPS=int(groups),
        GROUPS_POW=int(groups_pow),
        PARENTS_PER_PROG=int(parents_per_prog),
        NUM_SPLITS=int(num_splits),
        HAS_BUFFER=bool(has_buffer),
        L_BUF_MAX=int(l_buf_max),
        BUF_COLS_PER_SPLIT=int(buf_cols_per_split),
        num_warps=num_warps,
        num_stages=num_stages,
    )
