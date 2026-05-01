#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kMaxColsPerSplit = 1024;

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

__global__ void sdpa_cuda_sparse_v1_1_kernel(
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
  int groups = Hq / Hkv;
  int kvh = hq / groups;
  int n_start = split * cols_per_split;
  int n_end = min(n_start + cols_per_split, N);
  int span = max(0, n_end - n_start);

  const half* qh = q + hq * 128;
  const half* kh_base = keys + kvh * N * 128;
  const half* vh_base = values + kvh * N * 128;
  const int8_t* mask_h = mask + hq * N;

  __shared__ float s_scores[kMaxColsPerSplit];

  float local_m = -CUDART_INF_F;
  int lane = tid & 31;
  int warp = tid >> 5;
  
  for (int i = warp; i < span; i += kWarps) {
    int n = n_start + i;
    float part = 0.0f;
    if (mask_h[n] != 0) {
      const half* kh = kh_base + n * 128;
#pragma unroll
      for (int d = lane; d < 128; d += 32) {
        part += __half2float(qh[d]) * __half2float(kh[d]);
      }
    } else {
      part = -CUDART_INF_F;
    }
    float s = warp_reduce_sum(part);
    if (lane == 0) {
      if (mask_h[n] == 0) {
        s = -CUDART_INF_F;
      } else {
        s *= scale_log2e;
      }
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
      int n = n_start + i;
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
}

torch::Tensor forward(
    torch::Tensor q,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor mask,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_o,
    torch::Tensor counters,
    torch::Tensor out,
    float scale,
    int num_splits) {
  int Hq = q.size(0);
  int Hkv = keys.size(0);
  int N = keys.size(1);
  int cols_per_split = (N + num_splits - 1) / num_splits;
  counters.zero_();
  dim3 grid(Hq, num_splits);
  dim3 block(256);
  float scale_log2e = scale * M_LOG2E;
  sdpa_cuda_sparse_v1_1_kernel<<<grid, block>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(keys.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(values.data_ptr<at::Half>()),
      mask.data_ptr<int8_t>(),
      partial_m.data_ptr<float>(),
      partial_l.data_ptr<float>(),
      partial_o.data_ptr<float>(),
      counters.data_ptr<int>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      Hq, Hkv, N, num_splits, cols_per_split, scale_log2e);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "v1.1 sparse forward");
}
