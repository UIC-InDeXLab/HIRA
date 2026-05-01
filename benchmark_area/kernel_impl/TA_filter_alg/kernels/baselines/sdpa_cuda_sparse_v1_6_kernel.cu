#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kMaskBlock = 256;

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

__device__ __forceinline__ int block_reduce_sum_int(int v) {
  __shared__ int smem[kThreads];
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

__global__ void count_mask_blocks_kernel(
    const int8_t* __restrict__ mask,
    int* __restrict__ block_counts,
    int Hq,
    int N,
    int num_blocks) {
  int hq = blockIdx.x;
  int bid = blockIdx.y;
  int tid = threadIdx.x;
  int n = bid * kMaskBlock + tid;
  int active = (hq < Hq && n < N && mask[hq * N + n] != 0) ? 1 : 0;
  int count = block_reduce_sum_int(active);
  if (tid == 0) {
    block_counts[hq * num_blocks + bid] = count;
  }
}

__global__ void prefix_mask_blocks_kernel(
    const int* __restrict__ block_counts,
    int* __restrict__ block_offsets,
    int* __restrict__ counts,
    int num_blocks) {
  int hq = blockIdx.x;
  if (threadIdx.x == 0) {
    int total = 0;
    for (int b = 0; b < num_blocks; ++b) {
      int idx = hq * num_blocks + b;
      block_offsets[idx] = total;
      total += block_counts[idx];
    }
    counts[hq] = total;
  }
}

__global__ void fill_sparse_indices_ordered_kernel(
    const int8_t* __restrict__ mask,
    const int* __restrict__ block_offsets,
    int* __restrict__ indices,
    int Hq,
    int N,
    int num_blocks) {
  int hq = blockIdx.x;
  int bid = blockIdx.y;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int n = bid * kMaskBlock + tid;
  bool active = hq < Hq && n < N && mask[hq * N + n] != 0;

  __shared__ int warp_counts[kWarps];
  __shared__ int warp_offsets[kWarps];
  unsigned bits = __ballot_sync(0xffffffff, active);
  if (lane == 31) {
    warp_counts[warp] = __popc(bits);
  }
  __syncthreads();
  if (tid < kWarps) {
    int off = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      if (w < tid) {
        off += warp_counts[w];
      }
    }
    warp_offsets[tid] = off;
  }
  __syncthreads();

  if (active) {
    unsigned lower = lane == 0 ? 0u : ((1u << lane) - 1u);
    int local_rank = warp_offsets[warp] + __popc(bits & lower);
    int dst = block_offsets[hq * num_blocks + bid] + local_rank;
    indices[hq * N + dst] = n;
  }
}

__global__ void sdpa_cuda_sparse_v1_6_attention_kernel(
    const half* __restrict__ q,
    const half* __restrict__ keys,
    const half* __restrict__ values,
    const int* __restrict__ indices,
    const int* __restrict__ counts,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_o,
    int* __restrict__ counters,
    half* __restrict__ out,
    int Hq,
    int Hkv,
    int N,
    int num_splits,
    float scale_log2e) {
  int hq = blockIdx.x;
  int split = blockIdx.y;
  int tid = threadIdx.x;
  int groups = Hq / Hkv;
  int kvh = hq / groups;
  int live_n = counts[hq];
  int live_cols = (live_n + num_splits - 1) / num_splits;
  int live_start = split * live_cols;
  int live_end = min(live_start + live_cols, live_n);
  int span = max(0, live_end - live_start);

  const half* qh = q + hq * 128;
  const half* kh_base = keys + kvh * N * 128;
  const half* vh_base = values + kvh * N * 128;
  const int* idx_h = indices + hq * N;
  extern __shared__ float s_scores[];

  float local_m = -CUDART_INF_F;
  int lane = tid & 31;
  int warp = tid >> 5;
  for (int i = warp; i < span; i += kWarps) {
    int n = idx_h[live_start + i];
    const half* kh = kh_base + n * 128;
    float part = 0.0f;
#pragma unroll
    for (int d = lane; d < 128; d += 32) {
      part += __half2float(qh[d]) * __half2float(kh[d]);
    }
    float s = warp_reduce_sum(part);
    if (lane == 0) {
      s *= scale_log2e;
      s_scores[i] = s;
      local_m = fmaxf(local_m, s);
    }
  }
  float m = block_reduce_max(local_m);

  float local_l = 0.0f;
  if (m != -CUDART_INF_F) {
    for (int i = tid; i < span; i += blockDim.x) {
      float p = exp2f(s_scores[i] - m);
      s_scores[i] = p;
      local_l += p;
    }
  }
  float l = block_reduce_sum(local_l);

  float* po = partial_o + (hq * num_splits + split) * 128;
  for (int dv = tid; dv < 128; dv += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < span; ++i) {
      int n = idx_h[live_start + i];
      acc += s_scores[i] * __half2float(vh_base[n * 128 + dv]);
    }
    po[dv] = acc;
  }
  if (tid == 0) {
    partial_m[hq * num_splits + split] = m;
    partial_l[hq * num_splits + split] = l;
  }

  __threadfence();
  __syncthreads();

  __shared__ bool is_last;
  if (tid == 0) {
    int old = atomicAdd(counters + hq, 1);
    is_last = (old == num_splits - 1);
  }
  __syncthreads();
  if (!is_last) {
    return;
  }

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
}

}  // namespace

void sdpa_cuda_sparse_v1_6_launch(
    torch::Tensor q,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor mask,
    torch::Tensor indices,
    torch::Tensor block_counts,
    torch::Tensor block_offsets,
    torch::Tensor counts,
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
  TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(block_counts.scalar_type() == torch::kInt32, "block_counts must be int32");
  TORCH_CHECK(block_offsets.scalar_type() == torch::kInt32, "block_offsets must be int32");
  TORCH_CHECK(counts.scalar_type() == torch::kInt32, "counts must be int32");
  TORCH_CHECK(q.size(1) == 128 && keys.size(2) == 128 && values.size(2) == 128, "D/Dv must be 128");

  int Hq = static_cast<int>(q.size(0));
  int Hkv = static_cast<int>(keys.size(0));
  int N = static_cast<int>(keys.size(1));
  int splits = static_cast<int>(num_splits);
  int num_blocks = (N + kMaskBlock - 1) / kMaskBlock;
  int cols = (N + splits - 1) / splits;
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(counters.data_ptr<int>(), 0, Hq * sizeof(int), stream);

  dim3 mask_grid(Hq, num_blocks);
  count_mask_blocks_kernel<<<mask_grid, kThreads, 0, stream>>>(
      reinterpret_cast<const int8_t*>(mask.data_ptr<int8_t>()),
      block_counts.data_ptr<int>(),
      Hq,
      N,
      num_blocks);
  prefix_mask_blocks_kernel<<<Hq, 1, 0, stream>>>(
      block_counts.data_ptr<int>(),
      block_offsets.data_ptr<int>(),
      counts.data_ptr<int>(),
      num_blocks);
  fill_sparse_indices_ordered_kernel<<<mask_grid, kThreads, 0, stream>>>(
      reinterpret_cast<const int8_t*>(mask.data_ptr<int8_t>()),
      block_offsets.data_ptr<int>(),
      indices.data_ptr<int>(),
      Hq,
      N,
      num_blocks);

  dim3 attn_grid(Hq, splits);
  size_t shared_bytes = static_cast<size_t>(cols) * sizeof(float);
  sdpa_cuda_sparse_v1_6_attention_kernel<<<attn_grid, kThreads, shared_bytes, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(keys.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(values.data_ptr<at::Half>()),
      indices.data_ptr<int>(),
      counts.data_ptr<int>(),
      partial_m.data_ptr<float>(),
      partial_l.data_ptr<float>(),
      partial_o.data_ptr<float>(),
      counters.data_ptr<int>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      Hq,
      Hkv,
      N,
      splits,
      static_cast<float>(scale * 1.4426950408889634));
}
