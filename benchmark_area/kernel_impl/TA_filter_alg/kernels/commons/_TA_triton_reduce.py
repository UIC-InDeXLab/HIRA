"""Reduce kernel for TA-filter attention split-N partials (exp2 space)."""

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
    def _ta_reduce_kernel(
        M_ptr,            # (H_q, NUM_SPLITS) fp32
        L_ptr,            # (H_q, NUM_SPLITS) fp32
        O_ptr,            # (H_q, NUM_SPLITS, D_V) fp32
        Out_ptr,          # (H_q, D_V) fp32
        NUM_SPLITS: tl.constexpr,
        D_V: tl.constexpr,
        SPLITS_POW: tl.constexpr,
    ):
        hq = tl.program_id(0)
        s_range = tl.arange(0, SPLITS_POW)
        s_valid = s_range < NUM_SPLITS
        dv = tl.arange(0, D_V)

        m = tl.load(M_ptr + hq * NUM_SPLITS + s_range, mask=s_valid, other=-1.0e30)
        l_ = tl.load(L_ptr + hq * NUM_SPLITS + s_range, mask=s_valid, other=0.0)
        m_global = tl.max(m, axis=0)
        alpha = tl.exp2(m - m_global)
        l_sum = tl.sum(alpha * l_, axis=0)

        o = tl.load(
            O_ptr + (hq * NUM_SPLITS + s_range[:, None]) * D_V + dv[None, :],
            mask=s_valid[:, None],
            other=0.0,
        )
        o_sum = tl.sum(alpha[:, None] * o, axis=0)

        l_safe = tl.where(l_sum > 0.0, l_sum, 1.0)
        out = o_sum / l_safe
        tl.store(Out_ptr + hq * D_V + dv, out)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def run_ta_reduce(
    m_idx: torch.Tensor,
    l_idx: torch.Tensor,
    o_idx: torch.Tensor,
    out: torch.Tensor,
) -> None:
    h_q, num_splits = m_idx.shape
    d_v = int(o_idx.shape[-1])
    splits_pow = max(_next_pow2(num_splits), 1)
    _ta_reduce_kernel[(h_q,)](
        m_idx, l_idx, o_idx, out,
        NUM_SPLITS=num_splits,
        D_V=d_v,
        SPLITS_POW=splits_pow,
    )
