"""Inline-mask split-N attention for TA_attention_v6 series.

Skips the materialised (H_q, N_pad) cand_mask entirely.  Instead the kernel
loads the small ``selected[h_q, s, c]`` (int8, H_q*S*K bytes -- ~400 KB at
S=8/K=2 K) and the per-key parent assignments ``assigns[s, kvh, n]`` to
evaluate the OR-over-subspaces candidate predicate inline per chunk.

Compared with the existing ``_TA_triton_attn_inline``:
* Uses int8 (rather than int32) for the OR accumulator -- cheaper register use.
* Threshold is materialised in log2 space once.
* Buffer phase uses the same chunk loop instead of a separate buffer-cols
  scheme so we share q/keys staging and reduce launch parameters.
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
    def _ta_attn_v6_kernel(
        Q_ptr,                  # (H_q, D) fp16
        KeysT_ptr,              # (H_kv_eff, D, N_pad) fp16
        Values_ptr,             # (H_kv_eff, N_pad, D_v) fp16
        Selected_ptr,           # (H_q, S, K) int8 -- 1 if cluster in top-depth
        Assigns_ptr,            # (S, H_kv_eff, N_pad) int32
        InvalidMask_ptr,        # (H_kv_eff, N_pad) int8
        Threshold_ptr,          # (H_q,) fp32
        BufKeysT_ptr,           # (H_kv_eff, D, L_buf) fp16
        BufValues_ptr,          # (H_kv_eff, L_buf, D_v) fp16
        BufInvalid_ptr,         # (H_kv_eff, L_buf) int8
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        N_PAD,
        K_CLUSTERS,
        ASSIGN_STRIDE_S,        # = H_kv_eff * N_pad
        SELECTED_STRIDE_HQ,     # = S * K_CLUSTERS
        SELECTED_STRIDE_S,      # = K_CLUSTERS
        SCALE_LOG2E: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        S_SUB: tl.constexpr,
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

            inv = tl.load(
                InvalidMask_ptr + kvh * N_PAD + n_idx_safe,
                mask=n_valid,
                other=1,
            )
            valid_n = n_valid & (inv == 0)

            # Inline OR-over-subspaces: for each n in chunk, look up
            # parents[s, kvh, n] in S subspaces, fetch selected[h, s, parent],
            # OR the result.
            cand = tl.zeros([GROUPS_POW, BLOCK_N], dtype=tl.int8)
            for s_idx in tl.static_range(S_SUB):
                parents_s = tl.load(
                    Assigns_ptr + s_idx * ASSIGN_STRIDE_S
                    + kvh * N_PAD + n_idx_safe,
                    mask=valid_n,
                    other=0,
                ).to(tl.int32)
                sel = tl.load(
                    Selected_ptr
                    + hq_vec[:, None] * SELECTED_STRIDE_HQ
                    + s_idx * SELECTED_STRIDE_S
                    + parents_s[None, :],
                    mask=g_valid[:, None] & valid_n[None, :],
                    other=0,
                )
                cand = cand | sel

            cand_b = cand != 0
            survive_pre = g_valid[:, None] & valid_n[None, :] & cand_b

            any_live = tl.max(survive_pre.to(tl.int32))
            if any_live != 0:
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


def run_ta_attn_v6(
    q: torch.Tensor,
    keys_t_f16: torch.Tensor,
    values_f16: torch.Tensor,
    selected_i8: torch.Tensor,
    assigns_i32: torch.Tensor,
    invalid_mask_i8: torch.Tensor,
    threshold_f32: torch.Tensor,
    buf_keys_t_f16: torch.Tensor | None,
    buf_values_f16: torch.Tensor | None,
    buf_invalid_i8: torch.Tensor | None,
    h_kv_eff: int,
    n_pad: int,
    s_sub: int,
    k_clusters: int,
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
    _ta_attn_v6_kernel[grid](
        q,
        keys_t_f16,
        values_f16,
        selected_i8,
        assigns_i32,
        invalid_mask_i8,
        threshold_f32,
        buf_keys_arg,
        buf_values_arg,
        buf_invalid_arg,
        out_m,
        out_l,
        out_o,
        n_pad,
        k_clusters,
        h_kv_eff * n_pad,
        s_sub * k_clusters,
        k_clusters,
        SCALE_LOG2E=float(scale_log2e),
        D=d,
        D_V=d_v,
        S_SUB=s_sub,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        BLOCK_N=block_n,
        NUM_SPLITS=num_splits,
        HAS_BUFFER=bool(has_buffer),
        L_BUF_MAX=l_buf_max,
        BUF_COLS_PER_SPLIT=buf_cols_per_split,
        num_warps=num_warps,
        num_stages=num_stages,
    )
