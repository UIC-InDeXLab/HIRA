"""Attention over compact TA-filter candidate id lists."""

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
    def _ta_attn_compact_kernel(
        Q_ptr,
        KeysT_ptr,          # (H_kv, D, N_pad) fp16
        Values_ptr,         # (H_kv, N_pad, D_v) fp16
        CandIds_ptr,        # (H_q, N_pad) int32
        Counts_ptr,         # (H_q,) int32
        Threshold_ptr,      # (H_q,) fp32
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

        q_f16 = tl.load(Q_ptr + hq * D + d_range)
        threshold = tl.load(Threshold_ptr + hq).to(tl.float32)
        count = tl.load(Counts_ptr + hq).to(tl.int32)

        m = tl.full((), -1.0e30, dtype=tl.float32)
        l_acc = tl.full((), 0.0, dtype=tl.float32)
        o_acc = tl.zeros([D_V], dtype=tl.float32)

        cols_per_split = (count + NUM_SPLITS - 1) // NUM_SPLITS
        c_start = split * cols_per_split
        c_end = tl.minimum(c_start + cols_per_split, count)

        for c_chunk in range(c_start, c_end, BLOCK_N):
            c_idx = c_chunk + n_inner
            c_valid = c_idx < c_end
            c_safe = tl.where(c_valid, c_idx, 0)
            key_ids = tl.load(
                CandIds_ptr + hq * N_PAD + c_safe,
                mask=c_valid,
                other=0,
            ).to(tl.int32)
            key_valid = c_valid & (key_ids >= 0) & (key_ids < N_PAD)
            key_safe = tl.where(key_valid, key_ids, 0)

            if tl.max(key_valid.to(tl.int32), axis=0) != 0:
                keys_tile = tl.load(
                    KeysT_ptr + (kvh * D + d_range[:, None]) * N_PAD + key_safe[None, :],
                    mask=key_valid[None, :],
                    other=0.0,
                )
                raw = tl.dot(q_f16[None, :], keys_tile)
                raw = tl.reshape(raw, [BLOCK_N])
                survive = key_valid & (raw >= threshold)
                scaled = tl.where(survive, raw * SCALE_LOG2E, -1.0e30)

                chunk_max = tl.max(scaled, axis=0)
                m_new = tl.maximum(m, chunk_max)
                alpha = tl.exp2(m - m_new)
                p = tl.exp2(scaled - m_new)
                p = tl.where(survive, p, 0.0)
                l_acc = alpha * l_acc + tl.sum(p, axis=0)

                v_tile = tl.load(
                    Values_ptr + (kvh * N_PAD + key_safe[:, None]) * D_V + dv_range[None, :],
                    mask=survive[:, None],
                    other=0.0,
                )
                po = tl.dot(p[None, :].to(tl.float16), v_tile)
                o_acc = alpha * o_acc + tl.reshape(po, [D_V])
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

            if tl.max(buf_live.to(tl.int32), axis=0) != 0:
                buf_keys_tile = tl.load(
                    BufKeysT_ptr + (kvh * D + d_range[:, None]) * L_BUF_MAX + buf_idx_safe[None, :],
                    mask=buf_live[None, :],
                    other=0.0,
                )
                buf_raw = tl.dot(q_f16[None, :], buf_keys_tile)
                buf_raw = tl.reshape(buf_raw, [BUF_COLS_PER_SPLIT])
                buf_scaled = tl.where(buf_live, buf_raw * SCALE_LOG2E, -1.0e30)

                buf_chunk_max = tl.max(buf_scaled, axis=0)
                m_new = tl.maximum(m, buf_chunk_max)
                alpha = tl.exp2(m - m_new)
                bp = tl.exp2(buf_scaled - m_new)
                bp = tl.where(buf_live, bp, 0.0)
                l_acc = alpha * l_acc + tl.sum(bp, axis=0)

                buf_v_tile = tl.load(
                    BufValues_ptr
                    + (kvh * L_BUF_MAX + buf_idx_safe[:, None]) * D_V
                    + dv_range[None, :],
                    mask=buf_live[:, None],
                    other=0.0,
                )
                bo = tl.dot(bp[None, :].to(tl.float16), buf_v_tile)
                o_acc = alpha * o_acc + tl.reshape(bo, [D_V])
                m = m_new

        tl.store(M_out_ptr + hq * NUM_SPLITS + split, m)
        tl.store(L_out_ptr + hq * NUM_SPLITS + split, l_acc)
        tl.store(O_out_ptr + (hq * NUM_SPLITS + split) * D_V + dv_range, o_acc)


def run_ta_attn_compact(
    q: torch.Tensor,
    keys_t_f16: torch.Tensor,
    values_f16: torch.Tensor,
    cand_ids_i32: torch.Tensor,
    counts_i32: torch.Tensor,
    threshold_f32: torch.Tensor,
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
    d = int(q.shape[1])
    d_v = int(values_f16.shape[-1])
    has_buffer = buf_keys_t_f16 is not None and int(buf_keys_t_f16.shape[-1]) > 0
    if has_buffer:
        l_buf_max = int(buf_keys_t_f16.shape[-1])
        buf_cols_per_split = _next_pow2_min16(max(1, (l_buf_max + num_splits - 1) // num_splits))
        buf_keys_arg = buf_keys_t_f16
        buf_values_arg = buf_values_f16
        buf_invalid_arg = buf_invalid_i8
    else:
        l_buf_max = 16
        buf_cols_per_split = 16
        buf_keys_arg = q
        buf_values_arg = q
        buf_invalid_arg = counts_i32

    grid = (int(q.shape[0]), int(num_splits))
    _ta_attn_compact_kernel[grid](
        q,
        keys_t_f16,
        values_f16,
        cand_ids_i32,
        counts_i32,
        threshold_f32,
        buf_keys_arg,
        buf_values_arg,
        buf_invalid_arg,
        out_m,
        out_l,
        out_o,
        n_pad,
        SCALE_LOG2E=float(scale_log2e),
        D=d,
        D_V=d_v,
        GROUPS=int(groups),
        BLOCK_N=int(block_n),
        NUM_SPLITS=int(num_splits),
        HAS_BUFFER=bool(has_buffer),
        L_BUF_MAX=l_buf_max,
        BUF_COLS_PER_SPLIT=buf_cols_per_split,
        num_warps=num_warps,
        num_stages=num_stages,
    )
