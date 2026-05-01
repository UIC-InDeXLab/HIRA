"""Masked fp16 SDPA baseline for one decode query.

Computes softmax(QK^T * scale) @ V for fp16 Q/K/V and an int8 key mask.
The benchmark uses an all-true mask to measure attention-only overhead.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


_LOG2E = 1.4426950408889634
_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _groups_from_map(q_head_to_kv: torch.Tensor | None, h_q: int, h_kv: int) -> int | None:
    if q_head_to_kv is None:
        return 1 if h_q == h_kv else None
    if h_q % h_kv != 0:
        return None
    groups = h_q // h_kv
    expected = torch.arange(h_q, device=q_head_to_kv.device, dtype=q_head_to_kv.dtype) // groups
    if not torch.equal(q_head_to_kv, expected):
        return None
    return groups


if HAS_TRITON:

    @triton.jit
    def _sdpa_mask_split_kernel(
        Q_ptr,          # (H_q, D) fp16
        KeysT_ptr,      # (H_kv, D, N) fp16
        Values_ptr,     # (H_kv, N, D_v) fp16
        Mask_ptr,       # (H_q, N) int8
        M_ptr,          # (H_q, NUM_SPLITS) fp16
        L_ptr,          # (H_q, NUM_SPLITS) fp16
        O_ptr,          # (H_q, NUM_SPLITS, D_v) fp16
        N_CTX,
        SCALE_LOG2E: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        kvh = tl.program_id(0)
        split = tl.program_id(1)

        g = tl.arange(0, GROUPS_POW)
        g_valid = g < GROUPS
        hq = kvh * GROUPS + g
        d = tl.arange(0, D)
        dv = tl.arange(0, D_V)
        n_inner = tl.arange(0, BLOCK_N)

        q = tl.load(
            Q_ptr + hq[:, None] * D + d[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )

        m = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
        l_acc = tl.zeros([GROUPS_POW], dtype=tl.float32)
        o_acc = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

        cols_per_split = (N_CTX + NUM_SPLITS - 1) // NUM_SPLITS
        n_start = split * cols_per_split
        n_end = tl.minimum(n_start + cols_per_split, N_CTX)

        for n0 in range(n_start, n_end, BLOCK_N):
            n = n0 + n_inner
            n_valid = n < n_end
            n_safe = tl.where(n_valid, n, 0)
            mask = tl.load(
                Mask_ptr + hq[:, None] * N_CTX + n_safe[None, :],
                mask=g_valid[:, None] & n_valid[None, :],
                other=0,
            ) != 0
            live = g_valid[:, None] & n_valid[None, :] & mask

            if tl.max(live.to(tl.int32)) != 0:
                k = tl.load(
                    KeysT_ptr + (kvh * D + d[:, None]) * N_CTX + n_safe[None, :],
                    mask=n_valid[None, :],
                    other=0.0,
                )
                raw = tl.dot(q, k)
                scores = tl.where(live, raw * SCALE_LOG2E, -1.0e30)

                m_new = tl.maximum(m, tl.max(scores, axis=1))
                alpha = tl.exp2(m - m_new)
                p = tl.exp2(scores - m_new[:, None])
                p = tl.where(live, p, 0.0)
                l_acc = alpha * l_acc + tl.sum(p, axis=1)

                v = tl.load(
                    Values_ptr + (kvh * N_CTX + n_safe[:, None]) * D_V + dv[None, :],
                    mask=n_valid[:, None],
                    other=0.0,
                )
                o_acc = alpha[:, None] * o_acc + tl.dot(p.to(tl.float16), v)
                m = m_new

        tl.store(M_ptr + hq * NUM_SPLITS + split, m, mask=g_valid)
        tl.store(L_ptr + hq * NUM_SPLITS + split, l_acc, mask=g_valid)
        tl.store(
            O_ptr + (hq[:, None] * NUM_SPLITS + split) * D_V + dv[None, :],
            o_acc,
            mask=g_valid[:, None],
        )

    @triton.jit
    def _sdpa_mask_reduce_kernel(
        M_ptr,
        L_ptr,
        O_ptr,
        Out_ptr,
        NUM_SPLITS: tl.constexpr,
        D_V: tl.constexpr,
        SPLITS_POW: tl.constexpr,
    ):
        hq = tl.program_id(0)
        s = tl.arange(0, SPLITS_POW)
        s_valid = s < NUM_SPLITS
        dv = tl.arange(0, D_V)

        m = tl.load(M_ptr + hq * NUM_SPLITS + s, mask=s_valid, other=-65000.0)
        l = tl.load(L_ptr + hq * NUM_SPLITS + s, mask=s_valid, other=0.0)
        m_global = tl.max(m, axis=0)
        alpha = tl.exp2((m - m_global).to(tl.float32)).to(tl.float16)
        l_sum = tl.sum(alpha * l, axis=0)

        o = tl.load(
            O_ptr + (hq * NUM_SPLITS + s[:, None]) * D_V + dv[None, :],
            mask=s_valid[:, None],
            other=0.0,
        )
        o_sum = tl.sum(alpha[:, None] * o, axis=0)
        out = o_sum / tl.where(l_sum > 0.0, l_sum, 1.0)
        tl.store(Out_ptr + hq * D_V + dv, out)


def sdpa_triton_fp16(
    q: torch.Tensor,
    keys_t_f16: torch.Tensor,
    values_f16: torch.Tensor,
    mask_i8: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
    *,
    block_n: int = 128,
    num_splits: int = 16,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("sdpa_triton_fp16 requires Triton")
    if q.dtype != torch.float16 or keys_t_f16.dtype != torch.float16 or values_f16.dtype != torch.float16:
        raise TypeError("sdpa_triton_fp16 expects fp16 q/keys/values")
    if not (q.is_cuda and keys_t_f16.is_cuda and values_f16.is_cuda and mask_i8.is_cuda):
        raise RuntimeError("sdpa_triton_fp16 expects CUDA tensors")
    if keys_t_f16.ndim != 3:
        raise ValueError("keys_t_f16 must be (H_kv, D, N)")

    h_q, d = q.shape
    h_kv, d_k, n_ctx = keys_t_f16.shape
    d_v = int(values_f16.shape[-1])
    if d != d_k or values_f16.shape[:2] != (h_kv, n_ctx):
        raise ValueError("incompatible Q/K/V shapes")
    groups = _groups_from_map(q_head_to_kv, h_q, h_kv)
    if groups is None:
        raise ValueError("only identity/grouped GQA q_head_to_kv supported")
    if d & (d - 1) or d_v & (d_v - 1):
        raise ValueError("D and D_v must be powers of two for this baseline")
    if mask_i8.shape != (h_q, n_ctx):
        raise ValueError("mask_i8 must be (H_q, N)")

    q_f16 = q if q.is_contiguous() else q.contiguous()
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    groups_pow = _next_pow2(groups)
    key = (
        q.device.index,
        h_q,
        h_kv,
        n_ctx,
        d,
        d_v,
        groups,
        num_splits,
    )
    ws = _CACHE.get(key)
    if ws is None:
        ws = {
            "m": torch.empty(h_q, num_splits, device=q.device, dtype=torch.float16),
            "l": torch.empty(h_q, num_splits, device=q.device, dtype=torch.float16),
            "o": torch.empty(h_q, num_splits, d_v, device=q.device, dtype=torch.float16),
            "out": torch.empty(h_q, d_v, device=q.device, dtype=torch.float16),
        }
        _CACHE.clear()
        _CACHE[key] = ws

    _sdpa_mask_split_kernel[(h_kv, num_splits)](
        q_f16,
        keys_t_f16,
        values_f16,
        mask_i8,
        ws["m"],
        ws["l"],
        ws["o"],
        n_ctx,
        SCALE_LOG2E=float(scale) * _LOG2E,
        D=d,
        D_V=d_v,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        BLOCK_N=block_n,
        NUM_SPLITS=num_splits,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _sdpa_mask_reduce_kernel[(h_q,)](
        ws["m"],
        ws["l"],
        ws["o"],
        ws["out"],
        NUM_SPLITS=num_splits,
        D_V=d_v,
        SPLITS_POW=_next_pow2(num_splits),
    )
    return ws["out"]
