#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kTile = 32;

__device__ __forceinline__ float block_reduce_max(float v) {
  __shared__ float smem[kThreads];
  int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int stride = kThreads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }
  return smem[0];
}

__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float smem[kThreads];
  int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int stride = kThreads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  return smem[0];
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

// v1.20: single fused kernel, in-block ballot-based mask compaction (no atomics,
// no global index buffer). Compaction does one warp scanning tiles of 32 keys;
// QK + softmax + V loop iterates over compacted live list.
__global__ void sdpa_cuda_sparse_v1_20_kernel(
    const half* __restrict__ q,
    const half* __restrict__ keys,
    const half* __restrict__ values,
    const int8_t* __restrict__ mask,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_o,
    int* __restrict__ counters,
    half* __restrict__ out,
    int Hq,
    int Hkv,
    int N,
    int num_splits,
    int cols_per_split,
    float scale_log2e) {
  int hq = blockIdx.x;
  int split = blockIdx.y;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int kvh = hq / (Hq / Hkv);
  int n_start = split * cols_per_split;
  int n_end = min(n_start + cols_per_split, N);
  int span = max(0, n_end - n_start);

  const half* qh = q + hq * 128;
  const half* kh_base = keys + kvh * N * 128;
  const half* vh_base = values + kvh * N * 128;
  const int8_t* mask_h = mask + hq * N + n_start;

  extern __shared__ unsigned char smem_raw[];
  int* s_live = reinterpret_cast<int*>(smem_raw);
  float* s_scores = reinterpret_cast<float*>(s_live + cols_per_split);
  int* s_total = reinterpret_cast<int*>(s_scores + cols_per_split);

  // Phase 1: warp-0 scans tiles of 32 keys, ballot-compacts into s_live.
  if (warp == 0) {
    int total = 0;
    int num_tiles = (span + kTile - 1) / kTile;
    for (int t = 0; t < num_tiles; ++t) {
      int idx = t * kTile + lane;
      bool active = (idx < span) && (mask_h[idx] != 0);
      unsigned bits = __ballot_sync(0xffffffff, active);
      if (active) {
        unsigned lower = (lane == 0) ? 0u : ((1u << lane) - 1u);
        int rank = __popc(bits & lower);
        s_live[total + rank] = idx;
      }
      total += __popc(bits);
    }
    if (lane == 0) {
      *s_total = total;
    }
  }
  __syncthreads();
  int total_live = *s_total;

  // Phase 2: QK scoring, warp per live key.
  const half2* qh2 = reinterpret_cast<const half2*>(qh);
  float local_m = -CUDART_INF_F;
  for (int i = warp; i < total_live; i += kWarps) {
    int n = n_start + s_live[i];
    const half2* kh2 = reinterpret_cast<const half2*>(kh_base + n * 128);
    float part = 0.0f;
#pragma unroll
    for (int d = lane; d < 64; d += 32) {
      half2 prod = __hmul2(qh2[d], kh2[d]);
      float2 f = __half22float2(prod);
      part += f.x + f.y;
    }
    float s = warp_reduce_sum(part) * scale_log2e;
    if (lane == 0) {
      s_scores[i] = s;
      local_m = fmaxf(local_m, s);
    }
  }
  float m = block_reduce_max(local_m);

  // Phase 3: exp normalization.
  float local_l = 0.0f;
  if (m != -CUDART_INF_F) {
    for (int i = tid; i < total_live; i += blockDim.x) {
      float p = exp2f(s_scores[i] - m);
      s_scores[i] = p;
      local_l += p;
    }
  }
  float l = block_reduce_sum(local_l);

  // Phase 4: V accumulation.
  float* po = partial_o + (hq * num_splits + split) * 128;
  if (m != -CUDART_INF_F) {
    for (int dv = tid; dv < 128; dv += blockDim.x) {
      float acc = 0.0f;
      for (int i = 0; i < total_live; ++i) {
        int n = n_start + s_live[i];
        acc += s_scores[i] * __half2float(vh_base[n * 128 + dv]);
      }
      po[dv] = acc;
    }
  } else {
    for (int dv = tid; dv < 128; dv += blockDim.x) {
      po[dv] = 0.0f;
    }
  }

  if (tid == 0) {
    partial_m[hq * num_splits + split] = m;
    partial_l[hq * num_splits + split] = l;
  }

  // Use cuda::atomic_ref with release/acquire for explicit ordering instead of
  // __threadfence(). Cheaper on modern hardware while still correct.
  __syncthreads();

  __shared__ bool is_last;
  if (tid == 0) {
    int old;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;"
                 : "=r"(old) : "l"(counters + hq) : "memory");
    is_last = (old == num_splits - 1);
  }
  __syncthreads();
  if (!is_last) {
    return;
  }

  // Phase 5: cross-split combine. Acquire fence to pair with the .release
  // atomicAdd above so partial_m/l/o writes from sibling blocks are visible.
  asm volatile("fence.acquire.gpu;" ::: "memory");

  float m_global = -CUDART_INF_F;
  for (int s = tid; s < num_splits; s += blockDim.x) {
    m_global = fmaxf(m_global, partial_m[hq * num_splits + s]);
  }
  m_global = block_reduce_max(m_global);

  float l_global_local = 0.0f;
  for (int s = tid; s < num_splits; s += blockDim.x) {
    float alpha = exp2f(partial_m[hq * num_splits + s] - m_global);
    l_global_local += alpha * partial_l[hq * num_splits + s];
  }
  float l_global = block_reduce_sum(l_global_local);
  float inv_l = l_global > 0.0f ? 1.0f / l_global : 0.0f;

  for (int dv = tid; dv < 128; dv += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
      float alpha = exp2f(partial_m[hq * num_splits + s] - m_global);
      acc += alpha * partial_o[(hq * num_splits + s) * 128 + dv];
    }
    out[hq * 128 + dv] = __float2half(acc * inv_l);
  }
  if (tid == 0) {
    counters[hq] = 0;  // self-reset for next call; eliminates per-call cudaMemsetAsync
  }
}

}  // namespace

void sdpa_cuda_sparse_v1_20_launch(
    torch::Tensor q,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor mask,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_o,
    torch::Tensor counters,
    torch::Tensor out,
    double scale,
    int64_t num_splits) {
  TORCH_CHECK(q.is_cuda() && keys.is_cuda() && values.is_cuda() && mask.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16, "q must be fp16");
  TORCH_CHECK(keys.scalar_type() == torch::kFloat16, "keys must be fp16");
  TORCH_CHECK(values.scalar_type() == torch::kFloat16, "values must be fp16");
  TORCH_CHECK(mask.scalar_type() == torch::kInt8, "mask must be int8");
  TORCH_CHECK(q.size(1) == 128 && keys.size(2) == 128 && values.size(2) == 128, "D/Dv must be 128");

  int Hq = static_cast<int>(q.size(0));
  int Hkv = static_cast<int>(keys.size(0));
  int N = static_cast<int>(keys.size(1));
  int splits = static_cast<int>(num_splits);
  int cols = (N + splits - 1) / splits;
  auto stream = at::cuda::getCurrentCUDAStream();
  // Counter is zero-initialized once at workspace creation; kernel last-block
  // resets it to 0 after combine, so no per-call cudaMemsetAsync is needed.

  dim3 grid(Hq, splits);
  size_t shared_bytes =
      static_cast<size_t>(cols) * sizeof(int) +     // s_live
      static_cast<size_t>(cols) * sizeof(float) +   // s_scores
      sizeof(int);                                  // s_total
  sdpa_cuda_sparse_v1_20_kernel<<<grid, kThreads, shared_bytes, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(keys.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(values.data_ptr<at::Half>()),
      reinterpret_cast<const int8_t*>(mask.data_ptr<int8_t>()),
      partial_m.data_ptr<float>(),
      partial_l.data_ptr<float>(),
      partial_o.data_ptr<float>(),
      counters.data_ptr<int>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      Hq,
      Hkv,
      N,
      splits,
      cols,
      static_cast<float>(scale * 1.4426950408889634));
}
