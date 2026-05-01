"""v8.1 Triton kernels: per-kvh compact gather TA-filter attention.

Diagnostic on the bench input shows that with depth ~140 of K=2052 clusters
per (h, s), the per-key candidate fraction is ~39% per head and ~66% per
kvh — but the cluster-aligned skip predicate of v8.0 fires for ~96% of
clusters, so it doesn't actually cut key bandwidth.

v8.1 instead builds, per kvh, a **compact ordered list** of candidate key
indices (those flagged in any head of the kvh group).  The attention
kernel then iterates only over that list, gathering keys/values by
absolute index.  Bandwidth scales with `cand_kvh` density (~66%) instead
of with N_pad (100%).

Two kernels:

  1. ``_ta_v8_1_build_cand_compact_kernel`` — single grid (H_kv,) launch.
     Each program loads one full row of N_pad candidate bits, computes
     ``cand_b[h_q, n] = OR_s selected[h_q, s, assigns[s, kvh, n]]`` for
     every head in the kvh group (storing per-head ``cand_b`` for later
     fine masking), reduces to per-kvh union ``cand_kvh``, then performs
     an in-block ``tl.cumsum`` and scatters compact indices into
     ``cand_idx[kvh, ...]`` with the total count in ``cand_count[kvh]``.

  2. ``_ta_v8_1_attn_kernel`` — split-list flash attention.  Each
     (kvh, split) program walks its slice of ``cand_idx`` in chunks of
     ``BLOCK_N``, gathers keys (by index) into a (D, BLOCK_N) tile,
     applies the per-head ``cand_b`` mask + threshold T, runs an online
     softmax update, gathers values, and writes per-split (m, l, o).
     Phase-2 buffer pass identical to v6.0 / v8.0.
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


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


if HAS_TRITON:

    @triton.jit
    def _ta_v8_1_build_cand_compact_kernel(
        Selected_ptr,        # (H_q, S, K) int8
        Assigns_ptr,         # (S, H_kv, N_pad) int32
        InvalidMask_ptr,     # (H_kv, N_pad) int8
        CandB_ptr,           # (H_q, N_pad) int8 -- output
        CandIdx_ptr,         # (H_kv, MAX_CAND) int32 -- output
        CandCount_ptr,       # (H_kv,) int32 -- output
        N_PAD,
        ASSIGN_STRIDE_S,     # = H_kv * N_pad
        SELECTED_STRIDE_HQ,  # = S * K
        SELECTED_STRIDE_S,   # = K
        MAX_CAND,
        S_SUB: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        kvh = tl.program_id(0)

        offs = tl.arange(0, BLOCK_N)
        valid = offs < N_PAD

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        hq_vec = kvh * GROUPS + g_range

        inv = tl.load(
            InvalidMask_ptr + kvh * N_PAD + offs, mask=valid, other=1
        )
        valid_n = valid & (inv == 0)

        cand_b_grp = tl.zeros([GROUPS_POW, BLOCK_N], dtype=tl.int8)
        for s in tl.static_range(S_SUB):
            parents_s = tl.load(
                Assigns_ptr + s * ASSIGN_STRIDE_S + kvh * N_PAD + offs,
                mask=valid_n,
                other=0,
            ).to(tl.int32)
            sel = tl.load(
                Selected_ptr
                + hq_vec[:, None] * SELECTED_STRIDE_HQ
                + s * SELECTED_STRIDE_S
                + parents_s[None, :],
                mask=g_valid[:, None] & valid_n[None, :],
                other=0,
            )
            cand_b_grp = cand_b_grp | sel

        cand_b_grp = tl.where(
            g_valid[:, None] & valid_n[None, :],
            cand_b_grp,
            tl.zeros_like(cand_b_grp),
        )
        tl.store(
            CandB_ptr + hq_vec[:, None] * N_PAD + offs[None, :],
            cand_b_grp,
            mask=g_valid[:, None] & valid[None, :],
        )

        # Per-kvh union: OR over heads in group, per column.
        cand_kvh_lane = tl.max(cand_b_grp.to(tl.int32), axis=0)  # (BLOCK_N,)
        is_cand = (cand_kvh_lane != 0).to(tl.int32)
        is_cand = tl.where(valid, is_cand, tl.zeros_like(is_cand))

        # Exclusive scan via inclusive cumsum - is_cand.
        cum = tl.cumsum(is_cand, axis=0)
        pos = cum - is_cand

        # Scatter the compact index list.
        tl.store(
            CandIdx_ptr + kvh * MAX_CAND + pos,
            offs.to(tl.int32),
            mask=(is_cand != 0),
        )

        total = tl.sum(is_cand, axis=0)
        # All lanes write the same scalar; harmless redundancy.
        tl.store(CandCount_ptr + kvh, total)


def run_ta_v8_1_build_cand_compact(
    selected_i8: torch.Tensor,        # (H_q, S, K) int8
    assigns_i32: torch.Tensor,        # (S, H_kv, N_pad) int32
    invalid_mask_i8: torch.Tensor,    # (H_kv, N_pad) int8
    cand_b_i8: torch.Tensor,          # (H_q, N_pad) int8 (overwritten)
    cand_idx_i32: torch.Tensor,       # (H_kv, MAX_CAND) int32 (partially overwritten)
    cand_count_i32: torch.Tensor,     # (H_kv,) int32 (overwritten)
    *,
    groups: int,
    num_warps: int = 8,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, k_clusters = selected_i8.shape
    _s, h_kv, n_pad = assigns_i32.shape
    max_cand = int(cand_idx_i32.shape[-1])
    block_n = _next_pow2(int(n_pad))
    if block_n < 16:
        block_n = 16
    p = 1
    while p < groups:
        p *= 2
    groups_pow = max(p, 4)

    grid = (int(h_kv),)
    _ta_v8_1_build_cand_compact_kernel[grid](
        selected_i8,
        assigns_i32,
        invalid_mask_i8,
        cand_b_i8,
        cand_idx_i32,
        cand_count_i32,
        int(n_pad),
        int(h_kv) * int(n_pad),
        int(s_sub) * int(k_clusters),
        int(k_clusters),
        int(max_cand),
        S_SUB=int(s_sub),
        GROUPS=int(groups),
        GROUPS_POW=int(groups_pow),
        BLOCK_N=int(block_n),
        num_warps=num_warps,
        num_stages=num_stages,
    )


if HAS_TRITON:

    @triton.jit
    def _ta_v8_1_attn_kernel(
        Q_ptr,                  # (H_q, D) fp16
        KeysT_ptr,              # (H_kv, D, N_pad) fp16
        Values_ptr,             # (H_kv, N_pad, D_v) fp16
        CandIdx_ptr,            # (H_kv, MAX_CAND) int32
        CandCount_ptr,          # (H_kv,) int32
        CandB_ptr,              # (H_q, N_pad) int8
        Threshold_ptr,          # (H_q,) fp32
        BufKeysT_ptr,
        BufValues_ptr,
        BufInvalid_ptr,
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        N_PAD,
        MAX_CAND,
        SCALE_LOG2E: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        BLOCK_N: tl.constexpr,
        N_PER_SPLIT: tl.constexpr,
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

        cand_count = tl.load(CandCount_ptr + kvh).to(tl.int32)
        n_start = split * N_PER_SPLIT
        n_end = tl.minimum(n_start + N_PER_SPLIT, cand_count)

        n_inner = tl.arange(0, BLOCK_N)
        for n_chunk in range(n_start, n_end, BLOCK_N):
            pos = n_chunk + n_inner
            pos_valid = pos < n_end
            pos_safe = tl.where(pos_valid, pos, 0)

            n_idx = tl.load(
                CandIdx_ptr + kvh * MAX_CAND + pos_safe,
                mask=pos_valid,
                other=0,
            ).to(tl.int32)

            cand = tl.load(
                CandB_ptr + hq_vec[:, None] * N_PAD + n_idx[None, :],
                mask=g_valid[:, None] & pos_valid[None, :],
                other=0,
            )
            survive_pre = (cand != 0) & g_valid[:, None] & pos_valid[None, :]

            keys_tile = tl.load(
                KeysT_ptr
                + (kvh * D + d_range[:, None]) * N_PAD
                + n_idx[None, :],
                mask=pos_valid[None, :],
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
                Values_ptr
                + (kvh * N_PAD + n_idx[:, None]) * D_V
                + dv_range[None, :],
                mask=pos_valid[:, None],
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


def run_ta_v8_1_attn(
    q: torch.Tensor,
    keys_t_f16: torch.Tensor,
    values_f16: torch.Tensor,
    cand_idx_i32: torch.Tensor,
    cand_count_i32: torch.Tensor,
    cand_b_i8: torch.Tensor,
    threshold_f32: torch.Tensor,
    buf_keys_t_f16: torch.Tensor | None,
    buf_values_f16: torch.Tensor | None,
    buf_invalid_i8: torch.Tensor | None,
    h_kv_eff: int,
    n_pad: int,
    scale_log2e: float,
    groups: int,
    groups_pow: int,
    block_n: int,
    num_splits: int,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    d = int(q.shape[1])
    d_v = int(values_f16.shape[-1])
    max_cand = int(cand_idx_i32.shape[-1])
    n_per_split = (max_cand + num_splits - 1) // num_splits
    # Round n_per_split up to a multiple of block_n so the inner range is safe.
    if n_per_split % block_n != 0:
        n_per_split = ((n_per_split + block_n - 1) // block_n) * block_n

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
        buf_invalid_arg = cand_b_i8

    grid = (h_kv_eff, num_splits)
    _ta_v8_1_attn_kernel[grid](
        q,
        keys_t_f16,
        values_f16,
        cand_idx_i32,
        cand_count_i32,
        cand_b_i8,
        threshold_f32,
        buf_keys_arg,
        buf_values_arg,
        buf_invalid_arg,
        out_m,
        out_l,
        out_o,
        int(n_pad),
        int(max_cand),
        SCALE_LOG2E=float(scale_log2e),
        D=int(d),
        D_V=int(d_v),
        GROUPS=int(groups),
        GROUPS_POW=int(groups_pow),
        BLOCK_N=int(block_n),
        N_PER_SPLIT=int(n_per_split),
        NUM_SPLITS=int(num_splits),
        HAS_BUFFER=bool(has_buffer),
        L_BUF_MAX=int(l_buf_max),
        BUF_COLS_PER_SPLIT=int(buf_cols_per_split),
        num_warps=num_warps,
        num_stages=num_stages,
    )
