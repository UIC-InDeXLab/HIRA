"""v8.2 Triton kernels: precomputed cand_b + cand_kvh, v6.0-pattern attention.

Diagnostic: at the bench's depth profile, cluster-aligned skip (v8.0) and
per-kvh compact gather (v8.1) don't help — candidate density is ~66% per
kvh and chunk-skip almost never fires.  v8.2 sticks with the v6.0 split-N
attention shape, but moves the OR-over-S out of the inner loop into a
parallel preprocessing kernel, then uses ``cand_kvh`` for cheap chunk
skipping and ``cand_b`` for per-head masking.

Two kernels:

  1. ``_ta_v8_2_build_cand_kernel`` — grid (H_kv, ceil(N_pad / CHUNK_N)).
     Each program does CHUNK_N keys.  Per key: load assigns for S
     subspaces, lookup selected for each head in the kvh group, OR them.
     Writes ``cand_b[H_q, N_pad]`` int8 and the per-kvh union
     ``cand_kvh[H_kv, N_pad]`` int8.  Lots of program parallelism →
     saturates SMs.

  2. ``_ta_v8_2_attn_kernel`` — split-N attention identical in shape to
     v6.0 but reads precomputed ``cand_kvh`` for the chunk-level skip
     check (single int8 per col, not 8 int32 + 8 int8 indexed) and
     ``cand_b`` for the per-head mask used inside the live branch.  Phase-2
     buffer pass identical to v6.0 / v8.0.
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
    def _ta_v8_2_build_cand_kernel(
        Selected_ptr,        # (H_q, S, K) int8
        Assigns_ptr,         # (S, H_kv, N_pad) int32
        InvalidMask_ptr,     # (H_kv, N_pad) int8
        CandB_ptr,           # (H_q, N_pad) int8 -- output
        CandKvh_ptr,         # (H_kv, N_pad) int8 -- output
        N_PAD,
        ASSIGN_STRIDE_S,     # = H_kv * N_pad
        SELECTED_STRIDE_HQ,  # = S * K
        SELECTED_STRIDE_S,   # = K
        S_SUB: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        CHUNK_N: tl.constexpr,
    ):
        kvh = tl.program_id(0)
        chunk = tl.program_id(1)

        offs = chunk * CHUNK_N + tl.arange(0, CHUNK_N)
        valid = offs < N_PAD
        offs_safe = tl.where(valid, offs, 0)

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        hq_vec = kvh * GROUPS + g_range

        inv = tl.load(
            InvalidMask_ptr + kvh * N_PAD + offs_safe, mask=valid, other=1
        )
        valid_n = valid & (inv == 0)

        cand = tl.zeros([GROUPS_POW, CHUNK_N], dtype=tl.int8)
        for s in tl.static_range(S_SUB):
            parents_s = tl.load(
                Assigns_ptr + s * ASSIGN_STRIDE_S + kvh * N_PAD + offs_safe,
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
            cand = cand | sel

        cand = tl.where(
            g_valid[:, None] & valid_n[None, :], cand, tl.zeros_like(cand)
        )

        tl.store(
            CandB_ptr + hq_vec[:, None] * N_PAD + offs_safe[None, :],
            cand,
            mask=g_valid[:, None] & valid[None, :],
        )

        kvh_lane = tl.max(cand.to(tl.int32), axis=0).to(tl.int8)
        tl.store(
            CandKvh_ptr + kvh * N_PAD + offs_safe,
            kvh_lane,
            mask=valid,
        )


def run_ta_v8_2_build_cand(
    selected_i8: torch.Tensor,        # (H_q, S, K) int8
    assigns_i32: torch.Tensor,        # (S, H_kv, N_pad) int32
    invalid_mask_i8: torch.Tensor,    # (H_kv, N_pad) int8
    cand_b_i8: torch.Tensor,          # (H_q, N_pad) int8 (overwritten)
    cand_kvh_i8: torch.Tensor,        # (H_kv, N_pad) int8 (overwritten)
    *,
    groups: int,
    chunk_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, k_clusters = selected_i8.shape
    _s, h_kv, n_pad = assigns_i32.shape
    p = 1
    while p < groups:
        p *= 2
    groups_pow = max(p, 4)

    grid = (int(h_kv), (int(n_pad) + chunk_n - 1) // chunk_n)
    _ta_v8_2_build_cand_kernel[grid](
        selected_i8,
        assigns_i32,
        invalid_mask_i8,
        cand_b_i8,
        cand_kvh_i8,
        int(n_pad),
        int(h_kv) * int(n_pad),
        int(s_sub) * int(k_clusters),
        int(k_clusters),
        S_SUB=int(s_sub),
        GROUPS=int(groups),
        GROUPS_POW=int(groups_pow),
        CHUNK_N=int(chunk_n),
        num_warps=num_warps,
        num_stages=num_stages,
    )


if HAS_TRITON:

    @triton.jit
    def _ta_v8_2_attn_kernel(
        Q_ptr,                  # (H_q, D) fp16
        KeysT_ptr,              # (H_kv, D, N_pad) fp16
        Values_ptr,             # (H_kv, N_pad, D_v) fp16
        CandB_ptr,              # (H_q, N_pad) int8
        CandKvh_ptr,            # (H_kv, N_pad) int8
        InvalidMask_ptr,        # (H_kv, N_pad) int8
        Threshold_ptr,          # (H_q,) fp32
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
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        BLOCK_N: tl.constexpr,
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

        cols_per_split = (N_PAD + NUM_SPLITS - 1) // NUM_SPLITS
        n_start = split * cols_per_split
        n_end = tl.minimum(n_start + cols_per_split, N_PAD)
        n_inner = tl.arange(0, BLOCK_N)

        for n_chunk in range(n_start, n_end, BLOCK_N):
            n_idx = n_chunk + n_inner
            n_valid = n_idx < n_end
            n_idx_safe = tl.where(n_valid, n_idx, 0)

            cand_kvh = tl.load(
                CandKvh_ptr + kvh * N_PAD + n_idx_safe, mask=n_valid, other=0
            )
            inv = tl.load(
                InvalidMask_ptr + kvh * N_PAD + n_idx_safe, mask=n_valid, other=1
            )
            valid_n = n_valid & (inv == 0) & (cand_kvh != 0)

            any_live = tl.max(valid_n.to(tl.int32))
            if any_live != 0:
                cand = tl.load(
                    CandB_ptr + hq_vec[:, None] * N_PAD + n_idx_safe[None, :],
                    mask=g_valid[:, None] & valid_n[None, :],
                    other=0,
                )
                survive_pre = (cand != 0) & g_valid[:, None] & valid_n[None, :]

                keys_tile = tl.load(
                    KeysT_ptr
                    + (kvh * D + d_range[:, None]) * N_PAD
                    + n_idx_safe[None, :],
                    mask=valid_n[None, :],
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
                    + (kvh * N_PAD + n_idx_safe[:, None]) * D_V
                    + dv_range[None, :],
                    mask=valid_n[:, None],
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


def run_ta_v8_2_attn(
    q: torch.Tensor,
    keys_t_f16: torch.Tensor,
    values_f16: torch.Tensor,
    cand_b_i8: torch.Tensor,
    cand_kvh_i8: torch.Tensor,
    invalid_mask_i8: torch.Tensor,
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
        buf_invalid_arg = invalid_mask_i8

    grid = (h_kv_eff, num_splits)
    _ta_v8_2_attn_kernel[grid](
        q,
        keys_t_f16,
        values_f16,
        cand_b_i8,
        cand_kvh_i8,
        invalid_mask_i8,
        threshold_f32,
        buf_keys_arg,
        buf_values_arg,
        buf_invalid_arg,
        out_m,
        out_l,
        out_o,
        int(n_pad),
        SCALE_LOG2E=float(scale_log2e),
        D=int(d),
        D_V=int(d_v),
        GROUPS=int(groups),
        GROUPS_POW=int(groups_pow),
        BLOCK_N=int(block_n),
        NUM_SPLITS=int(num_splits),
        HAS_BUFFER=bool(has_buffer),
        L_BUF_MAX=int(l_buf_max),
        BUF_COLS_PER_SPLIT=int(buf_cols_per_split),
        num_warps=num_warps,
        num_stages=num_stages,
    )
