from __future__ import annotations
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _sparse_decode_attn_fwd_kernel(
    Q, K, V, Mask, Out,
    stride_qh, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_mh, stride_mn,
    stride_oh, stride_od,
    H_q, H_kv, N, D: tl.constexpr,
    sm_scale,
    BLOCK_N: tl.constexpr
):
    hq_idx = tl.program_id(0)
    hkv_idx = hq_idx // (H_q // H_kv)
    
    # Load Q
    offs_d = tl.arange(0, D)
    q_ptrs = Q + hq_idx * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs)
    
    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    
    offs_n = tl.arange(0, BLOCK_N)
    
    for start_n in range(0, N, BLOCK_N):
        offs = start_n + offs_n
        mask_ptrs = Mask + hq_idx * stride_mh + offs * stride_mn
        # Load mask
        mask_val = tl.load(mask_ptrs, mask=offs < N, other=0)
        
        if tl.sum(mask_val) > 0:
            k_ptrs = K + hkv_idx * stride_kh + offs[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=offs[:, None] < N, other=0.0)
            
            qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
            # Apply mask
            qk = tl.where((mask_val > 0) & (offs < N), qk, -float('inf'))
            
            m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
            p = tl.exp(qk - m_ij)
            
            l_ij = tl.sum(p, axis=0)
            alpha = tl.exp(m_i - m_ij)
            
            v_ptrs = V + hkv_idx * stride_vh + offs[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=offs[:, None] < N, other=0.0)
            
            # v is (BLOCK_N, D), p is (BLOCK_N,)
            p_v = tl.sum(p[:, None] * v, axis=0)
            
            acc = acc * alpha + p_v
            l_i = l_i * alpha + l_ij
            m_i = m_ij

    acc = acc / l_i
    out_ptrs = Out + hq_idx * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty))


def sdpa_cuda_sparse_v1_0_fp16(
    q: torch.Tensor,
    keys_f16: torch.Tensor,
    values_f16: torch.Tensor,
    mask_i8: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    h_q, d = q.shape
    h_kv, n_ctx, d_k = keys_f16.shape
    if scale is None:
        scale = d ** -0.5
        
    out = torch.empty((h_q, d), device=q.device, dtype=q.dtype)
    
    grid = (h_q,)
    
    _sparse_decode_attn_fwd_kernel[grid](
        q, keys_f16, values_f16, mask_i8, out,
        q.stride(0), q.stride(1),
        keys_f16.stride(0), keys_f16.stride(1), keys_f16.stride(2),
        values_f16.stride(0), values_f16.stride(1), values_f16.stride(2),
        mask_i8.stride(0), mask_i8.stride(1),
        out.stride(0), out.stride(1),
        h_q, h_kv, n_ctx, d,
        scale,
        BLOCK_N=128
    )
    return out
