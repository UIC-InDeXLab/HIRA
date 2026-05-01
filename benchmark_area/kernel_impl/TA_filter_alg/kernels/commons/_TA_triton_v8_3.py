"""v8.3 helper kernels: fused q/th copy.

Tiny Triton kernel used to stage the user's query and threshold tensors
into a graph-stable workspace before replaying the captured pipeline.
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
    def _ta_v8_3_copy_q_th_kernel(
        QIn_ptr,        # (H_q, D) fp16
        ThIn_ptr,       # (H_q,) fp32
        QOut_ptr,       # (H_q, D) fp16
        ThOut_ptr,      # (H_q,) fp32
        H_Q,
        D,
        Q_BLOCK: tl.constexpr,    # next pow2 >= H_q * D
        TH_BLOCK: tl.constexpr,   # next pow2 >= H_q
    ):
        # Single program: copy both buffers.  H_q*D <= 24*128 = 3072,
        # H_q <= 32, easily fits in one block at moderate warps.
        q_offs = tl.arange(0, Q_BLOCK)
        q_valid = q_offs < (H_Q * D)
        q = tl.load(QIn_ptr + q_offs, mask=q_valid, other=0.0)
        tl.store(QOut_ptr + q_offs, q, mask=q_valid)

        th_offs = tl.arange(0, TH_BLOCK)
        th_valid = th_offs < H_Q
        th = tl.load(ThIn_ptr + th_offs, mask=th_valid, other=0.0)
        tl.store(ThOut_ptr + th_offs, th, mask=th_valid)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def run_ta_v8_3_copy_q_th(
    q_in: torch.Tensor,         # (H_q, D) fp16
    th_in: torch.Tensor,        # (H_q,) fp32
    q_out: torch.Tensor,        # (H_q, D) fp16
    th_out: torch.Tensor,       # (H_q,) fp32
) -> None:
    h_q, d = q_in.shape
    q_block = _next_pow2(int(h_q) * int(d))
    th_block = _next_pow2(int(h_q))
    if th_block < 16:
        th_block = 16
    _ta_v8_3_copy_q_th_kernel[(1,)](
        q_in,
        th_in,
        q_out,
        th_out,
        int(h_q),
        int(d),
        Q_BLOCK=int(q_block),
        TH_BLOCK=int(th_block),
        num_warps=4,
        num_stages=1,
    )
