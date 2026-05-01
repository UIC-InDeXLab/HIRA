import torch
import triton
import triton.language as tl
import math

@triton.jit
def _gather_sparse_kv_kernel(
    K, V, Mask, SurvK, SurvV,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_mh, stride_mn,
    stride_skh, stride_skn, stride_skd,
    stride_svh, stride_svn, stride_svd,
    H_q, H_kv, N, D: tl.constexpr,
    MAX_K: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    hq_idx = tl.program_id(0)
    hkv_idx = hq_idx // (H_q // H_kv)
    
    # Each head has a separate counter
    curr_k = 0
    
    offs_d = tl.arange(0, BLOCK_D)
    
    # We loop over N and collect if mask is True
    # This is slightly serial per head but parallel across heads.
    # To make it faster, we could use a prefix sum, but for start let's do this.
    for n in range(0, N):
        m_val = tl.load(Mask + hq_idx * stride_mh + n * stride_mn)
        if m_val != 0:
            if curr_k < MAX_K:
                # Load from K
                k_ptr = K + hkv_idx * stride_kh + n * stride_kn + offs_d * stride_kd
                k_val = tl.load(k_ptr)
                # Store to SurvK
                sk_ptr = SurvK + hq_idx * stride_skh + curr_k * stride_skn + offs_d * stride_skd
                tl.store(sk_ptr, k_val)
                
                # Load from V
                v_ptr = V + hkv_idx * stride_vh + n * stride_vn + offs_d * stride_vd
                v_val = tl.load(v_ptr)
                # Store to SurvV
                sv_ptr = SurvV + hq_idx * stride_svh + curr_k * stride_svn + offs_d * stride_svd
                tl.store(sv_ptr, v_val)
                
                curr_k += 1

def gather_sparse_kv(k, v, mask_i8, max_k):
    h_q = mask_i8.shape[0]
    h_kv, n, d = k.shape
    
    surv_k = torch.zeros(h_q, max_k, d, device=k.device, dtype=k.dtype)
    surv_v = torch.zeros(h_q, max_k, d, device=k.device, dtype=k.dtype)
    
    grid = (h_q,)
    _gather_sparse_kv_kernel[grid](
        k, v, mask_i8, surv_k, surv_v,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        mask_i8.stride(0), mask_i8.stride(1),
        surv_k.stride(0), surv_k.stride(1), surv_k.stride(2),
        surv_v.stride(0), surv_v.stride(1), surv_v.stride(2),
        h_q, h_kv, n, d,
        MAX_K=max_k,
        BLOCK_D=d
    )
    return surv_k, surv_v

def sdpa_cuda_sparse_v1_3_fp16(
    q: torch.Tensor,
    keys_f16: torch.Tensor,
    values_f16: torch.Tensor,
    mask_i8: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    # v1.3: Triton Gather + Padded Dense SDPA
    h_q, d = q.shape
    h_kv, n_ctx, _ = keys_f16.shape
    if scale is None: scale = d ** -0.5
    
    counts = mask_i8.sum(dim=-1)
    max_k = counts.max().item()
    if max_k == 0: return torch.zeros_like(q)
    
    # Power of 2 max_k for better tiling? 
    # Not strictly needed but helps some kernels.
    
    surv_k, surv_v = gather_sparse_kv(keys_f16, values_f16, mask_i8, max_k)
            
    out = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0).unsqueeze(2),
        surv_k.unsqueeze(0),
        surv_v.unsqueeze(0),
        is_causal=False
    )
    return out.squeeze(0).squeeze(1)
