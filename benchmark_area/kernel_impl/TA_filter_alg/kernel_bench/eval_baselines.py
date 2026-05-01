import torch
import time
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.baselines._sdpa_cuda_atomic_fp16 import sdpa_cuda_atomic_fp16
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.baselines._sdpa_cuda_sparse_v1_0_fp16 import sdpa_cuda_sparse_v1_0_fp16

def benchmark_baselines(heads=32, seq_len=8192, head_dim=128, true_fraction=0.2):
    torch.manual_seed(42)
    q = torch.randn(heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(1, seq_len, head_dim, device="cuda", dtype=torch.float16).expand(heads, seq_len, head_dim).contiguous()
    v = torch.randn(1, seq_len, head_dim, device="cuda", dtype=torch.float16).expand(heads, seq_len, head_dim).contiguous()
    
    mask = torch.rand(heads, seq_len, device="cuda") < true_fraction
    mask_i8 = mask.to(torch.int8)
    
    iters = 100
    
    def bench_fn(fn, name):
        # Warmup
        for _ in range(10):
            _ = fn(q, k, v, mask_i8)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = fn(q, k, v, mask_i8)
        torch.cuda.synchronize()
        end = time.time()
        return (end - start) * 1000 / iters

    ms_atomic = bench_fn(sdpa_cuda_atomic_fp16, "atomic")
    ms_v1_0 = bench_fn(sdpa_cuda_sparse_v1_0_fp16, "v1.0")
    
    print(f"SeqLen {seq_len}, {true_fraction*100:.0f}% True:")
    print(f"  atomic: {ms_atomic:.4f} ms")
    print(f"  v1_0  : {ms_v1_0:.4f} ms")

if __name__ == "__main__":
    benchmark_baselines(true_fraction=1.0)
    benchmark_baselines(true_fraction=0.2)
