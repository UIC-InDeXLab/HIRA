from __future__ import annotations
import torch
import time
import math
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.baselines._sdpa_cuda_atomic_fp16 import sdpa_cuda_atomic_fp16
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.baselines._sdpa_cuda_sparse_v1_1_fp16 import sdpa_cuda_sparse_v1_1_fp16
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.baselines._sdpa_cuda_sparse_v1_3_fp16 import sdpa_cuda_sparse_v1_3_fp16
from hira.benchmark_area.quick_pruning.pruning_bench_utils import CaptureState

def benchmark_real_data(capture_path: str, true_fraction: float = 0.2):
    cap = CaptureState.load(capture_path)
    layer = 15
    q_all, k_cpu, v_cpu = cap.to_layer_tensors(layer)
    k = k_cpu.cuda().half()
    v = v_cpu.cuda().half()
    h_q = q_all.shape[0]
    n_ctx = k.shape[1]
    d = q_all.shape[2]
    scale = 1.0 / math.sqrt(d)
    
    t_idx = q_all.shape[1] // 2
    q = q_all[:, t_idx, :].cuda().half() 
    
    mask = torch.rand(h_q, n_ctx, device="cuda") < true_fraction
    mask_i8 = mask.to(torch.int8)
    
    iters = 100
    def bench_fn(fn):
        for _ in range(10): _ = fn(q, k, v, mask_i8, scale=scale)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters): _ = fn(q, k, v, mask_i8, scale=scale)
        torch.cuda.synchronize()
        return (time.time() - start) * 1000 / iters

    ms_atomic = bench_fn(sdpa_cuda_atomic_fp16)
    ms_v1_1 = bench_fn(sdpa_cuda_sparse_v1_1_fp16)
    ms_v1_3 = bench_fn(sdpa_cuda_sparse_v1_3_fp16)
    
    # Dense Baseline (all True)
    mask_all = torch.ones(h_q, n_ctx, device="cuda", dtype=torch.int8)
    ms_dense = bench_fn(sdpa_cuda_atomic_fp16) # atomic on all True
    
    print(f"Capture: {capture_path}")
    print(f"SeqLen: {n_ctx}, Heads: {h_q}, Fraction: {true_fraction*100:.1f}%")
    print(f"  sdpa_cuda_atomic_fp16: {ms_atomic:.4f} ms")
    print(f"  sdpa_cuda_sparse_v1_1: {ms_v1_1:.4f} ms (CUDA Optimized Masked)")
    print(f"  sdpa_cuda_sparse_v1_3: {ms_v1_3:.4f} ms (Triton Gather + Dense SDPA)")
    print(f"  sdpa_dense (atomic):   {ms_dense:.4f} ms")

if __name__ == "__main__":
    path = "benchmark_area/quick_pruning/capture_qkv_8000_meta-llama_Llama-3.2-3B-Instruct.pt"
    benchmark_real_data(path, true_fraction=0.2)
    benchmark_real_data(path, true_fraction=0.05)
