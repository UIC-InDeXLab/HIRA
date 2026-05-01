"""Split-N flash-style Triton attention with a precomputed (H_q, N_pad) mask.

Used by ``TA_attention_v_1_1``: the candidate-set mask (which keys are in the
TA-filter sweep up to depth L*) is computed in torch and passed in as a single
(H_q, N_pad) boolean tensor.  The kernel:

  1. Tiles N into ``num_splits`` programs per kv-head; inside each program we
     iterate over chunks of size ``BLOCK_N``.
  2. For each chunk, loads keys (D x BLOCK_N), evaluates
        survive = cand_mask & ~invalid & (raw_score >= T)
     on raw (pre-scale) scores, then runs an online-softmax update in
     log2-score space (scale * log2(e) factor folded into SCALE_LOG2E so we use
     ``tl.exp2`` exclusively).
  3. After the index pass, the SAME program absorbs its slice of the buffer
     keys/values (always survive — buffer keys are not pruned).
  4. A small reduce kernel merges per-split (m, l, o) into the final output.

The pre-mask path is a sound reference for the fully-fused inline-mask kernel
(v1.2) — it isolates the work the inline kernel must replicate per chunk.
"""

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
    def _ta_attn_premask_kernel(
        Q_ptr,
        KeysT_ptr,             # (H_kv_eff, D, N_pad)
        Values_ptr,            # (H_kv_eff, N_pad, D_v)
        CandMask_ptr,          # (H_q, N_pad) int8 (0/1)
        CandStamp_ptr,         # scalar int8 when USE_CAND_STAMP
        InvalidMask_ptr,       # (H_kv_eff, N_pad) int8
        Threshold_ptr,         # (H_q,) fp32 -- raw threshold T (in score space)
        BufKeysT_ptr,          # (H_kv_eff, D, L_buf_max) fp16
        BufValues_ptr,         # (H_kv_eff, L_buf_max, D_v) fp16
        BufInvalid_ptr,        # (H_kv_eff, L_buf_max) int8
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        N_PAD,
        SCALE: tl.constexpr,           # raw scale factor (e.g. 1/sqrt(D))
        SCALE_LOG2E: tl.constexpr,     # SCALE * LOG2E (for tl.exp2)
        D: tl.constexpr,
        D_V: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        USE_CAND_STAMP: tl.constexpr,
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

        # Load per-program q (groups, D) and per-head threshold.
        q_f16 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        th_raw = tl.load(
            Threshold_ptr + hq_vec, mask=g_valid, other=float("inf"),
        ).to(tl.float32)
        # Compare in log2-score space: scaled_score = score * SCALE * LOG2E.
        th_log2 = th_raw * SCALE_LOG2E
        cand_stamp = tl.load(CandStamp_ptr).to(tl.int8)

        m = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
        l_acc = tl.zeros([GROUPS_POW], dtype=tl.float32)
        o_acc = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

        # ── Phase 1: indexed keys (sweep N within this split) ──
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

            # Per (group, n) candidate mask.
            cand = tl.load(
                CandMask_ptr + hq_vec[:, None] * N_PAD + n_idx_safe[None, :],
                mask=g_valid[:, None] & n_valid[None, :],
                other=0,
            )
            if USE_CAND_STAMP:
                cand_b = cand == cand_stamp
            else:
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
                # Raw score (no scale yet) for T comparison.
                raw_scores = tl.dot(q_f16, keys_tile)
                # Apply T-filter on raw scores.
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

        # ── Phase 2: buffer keys (always survive their validity check, no T) ──
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

        tl.store(
            M_out_ptr + hq_vec * NUM_SPLITS + split,
            m,
            mask=g_valid,
        )
        tl.store(
            L_out_ptr + hq_vec * NUM_SPLITS + split,
            l_acc,
            mask=g_valid,
        )
        tl.store(
            O_out_ptr
            + (hq_vec[:, None] * NUM_SPLITS + split) * D_V
            + dv_range[None, :],
            o_acc,
            mask=g_valid[:, None],
        )


def _next_pow2_min16(x: int) -> int:
    p = 16
    while p < x:
        p *= 2
    return p


def run_ta_attn_premask(
    q: torch.Tensor,
    keys_t_f16: torch.Tensor,
    values_f16: torch.Tensor,
    cand_mask_i8: torch.Tensor,
    invalid_mask_i8: torch.Tensor,
    threshold_f32: torch.Tensor,
    buf_keys_t_f16: torch.Tensor | None,
    buf_values_f16: torch.Tensor | None,
    buf_invalid_i8: torch.Tensor | None,
    h_kv_eff: int,
    n_pad: int,
    scale: float,
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
    cand_stamp_i8: torch.Tensor | None = None,
) -> None:
    d = int(q.shape[1])
    d_v = int(values_f16.shape[-1])
    has_buffer = buf_keys_t_f16 is not None and int(buf_keys_t_f16.shape[-1]) > 0
    use_cand_stamp = cand_stamp_i8 is not None
    cand_stamp_arg = cand_stamp_i8 if use_cand_stamp else invalid_mask_i8
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
        # Provide dummy non-null pointers; with HAS_BUFFER=False they aren't read.
        buf_keys_arg = q
        buf_values_arg = q
        buf_invalid_arg = invalid_mask_i8
    grid = (h_kv_eff, num_splits)
    _ta_attn_premask_kernel[grid](
        q,
        keys_t_f16,
        values_f16,
        cand_mask_i8,
        cand_stamp_arg,
        invalid_mask_i8,
        threshold_f32,
        buf_keys_arg,
        buf_values_arg,
        buf_invalid_arg,
        out_m,
        out_l,
        out_o,
        n_pad,
        SCALE=float(scale),
        SCALE_LOG2E=float(scale_log2e),
        D=d,
        D_V=d_v,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        BLOCK_N=block_n,
        NUM_SPLITS=num_splits,
        USE_CAND_STAMP=bool(use_cand_stamp),
        HAS_BUFFER=bool(has_buffer),
        L_BUF_MAX=l_buf_max,
        BUF_COLS_PER_SPLIT=buf_cols_per_split,
        num_warps=num_warps,
        num_stages=num_stages,
    )
