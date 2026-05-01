"""Triton helpers for v13.4 arithmetic parent->child attention."""

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
    def _ta_v13_4_parent_block_attn_kernel(
        Q_ptr,           # (H_q, D) fp16
        ParentMask_ptr,  # (H_q, 4, K) int8
        KeysT_ptr,       # (H_kv, D, N_pad) fp16
        Values_ptr,      # (H_kv, N_pad, D_v) fp16
        BufKeysT_ptr,
        BufValues_ptr,
        BufInvalid_ptr,
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        K_CLUSTERS,
        N_REAL,
        N_PAD,
        SCALE_LOG2E: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        H_KV: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_P: tl.constexpr,
        CANDS: tl.constexpr,
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
        p_inner = tl.arange(0, BLOCK_P)
        child_inner = tl.arange(0, 4)

        q_vals = tl.load(Q_ptr + hq * D + d_range)
        m = tl.full((), -1.0e30, dtype=tl.float32)
        l_acc = tl.full((), 0.0, dtype=tl.float32)
        o_acc = tl.zeros([D_V], dtype=tl.float32)

        rows_per_split = (K_CLUSTERS + NUM_SPLITS - 1) // NUM_SPLITS
        p_start = split * rows_per_split
        p_stop = tl.minimum(p_start + rows_per_split, K_CLUSTERS)

        for p_chunk in range(p_start, p_stop, BLOCK_P):
            parents = p_chunk + p_inner
            parent_valid = parents < p_stop
            parents_safe = tl.where(parent_valid, parents, 0)

            selected = tl.zeros([BLOCK_P], dtype=tl.int32)
            for s_idx in tl.static_range(4):
                vals = tl.load(
                    ParentMask_ptr + (hq * 4 + s_idx) * K_CLUSTERS + parents_safe,
                    mask=parent_valid,
                    other=0,
                )
                selected += vals.to(tl.int32)
            parent_live = parent_valid & (selected != 0)

            offs_2d = parents[:, None] * 4 + child_inner[None, :]
            live_2d = parent_live[:, None] & (offs_2d < N_REAL)
            offs = tl.reshape(offs_2d, [CANDS])
            live = tl.reshape(live_2d, [CANDS])
            offs_safe = tl.where(live, offs, 0)

            if tl.max(live.to(tl.int32), axis=0) != 0:
                keys = tl.load(
                    KeysT_ptr + (kvh * D + d_range[:, None]) * N_PAD + offs_safe[None, :],
                    mask=live[None, :],
                    other=0.0,
                )
                raw = tl.dot(q_vals[None, :], keys)
                raw = tl.reshape(raw, [CANDS]).to(tl.float32)
                scaled = tl.where(live, raw * SCALE_LOG2E, -1.0e30)

                chunk_max = tl.max(scaled, axis=0)
                m_new = tl.maximum(m, chunk_max)
                alpha = tl.exp2(m - m_new)
                pvals = tl.exp2(scaled - m_new)
                pvals = tl.where(live, pvals, 0.0)
                l_acc = alpha * l_acc + tl.sum(pvals, axis=0)

                values = tl.load(
                    Values_ptr + (kvh * N_PAD + offs_safe[:, None]) * D_V + dv_range[None, :],
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


def run_ta_v13_4_parent_block_attn(
    q: torch.Tensor,
    parent_mask_i8: torch.Tensor,
    keys_t_f16: torch.Tensor,
    values_f16: torch.Tensor,
    buf_keys_t_f16: torch.Tensor | None,
    buf_values_f16: torch.Tensor | None,
    buf_invalid_i8: torch.Tensor | None,
    *,
    n_real: int,
    scale_log2e: float,
    groups: int,
    block_p: int,
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
    h_q, s_sub, k_clusters = parent_mask_i8.shape
    if int(s_sub) != 4:
        raise ValueError(f"v13.4 parent-block attention requires S=4, got {s_sub}")
    if int(h_q) != int(q.shape[0]):
        raise ValueError(f"parent mask H mismatch: mask={h_q}, q={q.shape[0]}")

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
        bi = parent_mask_i8

    _ta_v13_4_parent_block_attn_kernel[(int(q.shape[0]), int(num_splits))](
        q,
        parent_mask_i8,
        keys_t_f16,
        values_f16,
        bk,
        bv,
        bi,
        out_m,
        out_l,
        out_o,
        int(k_clusters),
        int(n_real),
        int(n_pad),
        SCALE_LOG2E=float(scale_log2e),
        D=d,
        D_V=d_v,
        H_KV=int(h_kv),
        GROUPS=int(groups),
        BLOCK_P=int(block_p),
        CANDS=int(block_p) * 4,
        NUM_SPLITS=int(num_splits),
        HAS_BUFFER=bool(has_buffer),
        L_BUF_MAX=int(l_buf_max),
        BUF_COLS_PER_SPLIT=int(buf_cols_per_split),
        num_warps=num_warps,
        num_stages=num_stages,
    )
