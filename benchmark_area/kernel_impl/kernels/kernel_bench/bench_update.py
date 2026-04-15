"""Micro-benchmark: compare all update_v* kernels (full + inc) on synthetic keys.

Usage:
    python -m hira.benchmark_area.kernel_impl.kernels.kernel_bench.bench_update
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.kernels import update_kernels
from hira.benchmark_area.kernel_impl.kernels.build_v1_0 import build as build_v1


def time_call(fn, iters=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--B", type=int, default=256, help="buffer size")
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", type=int, default=4)
    p.add_argument("--refine-iter", type=int, default=5)
    p.add_argument("--iters", type=int, default=3)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    torch.manual_seed(0)
    keys = torch.randn(args.H, args.N, args.D, device="cuda", dtype=torch.float32)
    buffer = torch.randn(args.H, args.B, args.D, device="cuda", dtype=torch.float32)

    base_state = build_v1(keys, args.bf, args.S, args.refine_iter)

    print(f"update micro-bench: H={args.H} N={args.N} B={args.B} D={args.D} bf={args.bf} S={args.S}")
    print("-" * 70)

    results = []
    for name, info in sorted(update_kernels().items()):
        for mode in ("full", "inc"):
            fn = info.fn
            def call(fn=fn, mode=mode):
                fn(base_state, keys, buffer, args.bf, args.S, args.refine_iter, mode)
            ms = time_call(call, iters=args.iters, warmup=1)
            label = f"{name} ({info.version}) {mode}"
            results.append((label, ms))
            print(f"  {name:<18s} {info.version:<6s} {mode:<5s}  {ms:8.2f} ms")

    print("-" * 70)
    best = min(results, key=lambda r: r[1])
    print(f"Fastest: {best[0]} at {best[1]:.2f} ms")


if __name__ == "__main__":
    main()
