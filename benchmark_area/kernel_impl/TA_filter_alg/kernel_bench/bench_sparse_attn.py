"""Benchmark masked attention baselines across sparse top-k mask fractions.

Example:
    ~/venv/bin/python benchmark_area/kernel_impl/TA_filter_alg/kernel_bench/bench_sparse_attention_baselines.py \
        --input-qkv benchmark_area/quick_pruning/capture_qkv_8000_meta-llama_Llama-3.2-3B-Instruct.pt \
        --fractions 1.0 0.2 0.1 0.05 --n-queries 20 --iters 10

Masks are built before timing.  For each query/head, the mask keeps the top-k
keys by exact full dot product, where k = fraction * N.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.baselines._sdpa_cuda_atomic_fp16 import (
    sdpa_cuda_atomic_fp16,
)
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.baselines._sdpa_cuda_sparse_v1_0_fp16 import (
    sdpa_cuda_sparse_v1_0_fp16,
)
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.baselines._sdpa_cuda_sparse_v1_1_fp16 import (
    sdpa_cuda_sparse_v1_1_fp16,
)
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.baselines._sdpa_flex_attention_fp16 import (
    sdpa_flex_attention_fp16,
)
from hira.benchmark_area.quick_pruning.pruning_bench_utils import CaptureState, _q_to_kv_map


def time_call(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def topk_dot_mask(
    q: torch.Tensor,
    keys: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    fraction: float,
) -> torch.Tensor:
    h_q = int(q.shape[0])
    n_ctx = int(keys.shape[1])
    if fraction >= 1.0:
        return torch.ones(h_q, n_ctx, device=q.device, dtype=torch.int8)
    k_eff = max(1, min(n_ctx, int(round(float(fraction) * n_ctx))))
    keys_eff = keys if q_head_to_kv is None else keys.index_select(0, q_head_to_kv)
    scores = torch.einsum("hd,hnd->hn", q.float(), keys_eff.float())
    top_idx = torch.topk(scores, k_eff, dim=-1).indices
    mask = torch.zeros(h_q, n_ctx, device=q.device, dtype=torch.int8)
    mask.scatter_(1, top_idx, 1)
    return mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", "--input", dest="input_qkv", type=Path, required=True)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--fractions", nargs="+", type=float, default=[1.0, 0.2, 0.1, 0.05])
    p.add_argument("--n-queries", type=int, default=50)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--skip-flex", action="store_true")
    args = p.parse_args()

    cap = CaptureState.load(args.input_qkv)
    qcpu, kcpu, vcpu = cap.to_layer_tensors(args.layer)
    if vcpu is None:
        raise RuntimeError("Captured values required.")

    keys = kcpu.to(device="cuda", dtype=torch.float32).half().contiguous()
    values = vcpu.to(device="cuda", dtype=torch.float32).half().contiguous()
    h_q = int(qcpu.shape[0])
    h_kv, n_ctx, d = keys.shape
    q_head_to_kv = _q_to_kv_map(h_q, h_kv, "cuda") if h_q != h_kv else None
    scale = 1.0 / math.sqrt(d)

    total_q = qcpu.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride))[
        : args.n_queries
    ]
    queries: list[torch.Tensor] = []
    queries_f32: list[torch.Tensor] = []
    for qi in q_indices:
        q = qcpu[:, qi, :].to(device="cuda", dtype=torch.float32)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        queries_f32.append(q.contiguous())
        queries.append(q.half().contiguous())

    baselines = [
        ("cuda_atomic", sdpa_cuda_atomic_fp16, {}),
        ("cuda_sparse_v1_0", sdpa_cuda_sparse_v1_0_fp16, {}),
        ("cuda_sparse_v1_1", sdpa_cuda_sparse_v1_1_fp16, {}),
    ]
    if not args.skip_flex:
        baselines.extend(
            [
                (
                    "flex_16x128_w8",
                    sdpa_flex_attention_fp16,
                    {"block_size": (16, 128), "num_warps": 8, "prescale_qk": False},
                ),
                (
                    "flex_16x64_w4",
                    sdpa_flex_attention_fp16,
                    {"block_size": (16, 64), "num_warps": 4, "prescale_qk": False},
                ),
            ]
        )

    print(f"Hq={h_q} Hkv={h_kv} N={n_ctx} D={d} queries={len(queries)}")
    for frac in args.fractions:
        masks = [topk_dot_mask(q, keys, q_head_to_kv, frac) for q in queries_f32]
        torch.cuda.synchronize()
        actual = sum(float(m.float().mean().item()) for m in masks) / len(masks)
        print(f"\nfraction={frac:g} actual={actual:.4f}")
        for name, fn, kwargs in baselines:
            def run() -> None:
                for q, mask in zip(queries, masks):
                    fn(q, keys, values, mask, q_head_to_kv, scale, **kwargs)

            try:
                ms = time_call(run, args.iters, args.warmup) / len(queries)
                print(f"  {name:<20s} {ms:9.6f} ms/query")
            except Exception as exc:
                torch.cuda.synchronize()
                print(f"  {name:<20s} skipped: {type(exc).__name__}: {str(exc).splitlines()[0]}")


if __name__ == "__main__":
    main()
