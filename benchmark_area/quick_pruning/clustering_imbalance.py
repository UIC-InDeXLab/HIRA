#!/usr/bin/env python3
"""
Compare how balanced the clustering outputs are in benchmark_area/quick_pruning.

For each clustering method, this script captures keys from the same model/prompt
pipeline as comparison.py, runs clustering, and reports cluster-size imbalance
statistics only.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clusterings import CLUSTERING_METHODS
from comparison import DEVICE, DTYPE, LAYER_IDX, MODEL_NAME, PROMPT, _select_methods
from pruning_bench_utils import _capture_qkv


def summarize_cluster_imbalance(assign: torch.Tensor, K: int) -> dict[str, float]:
    """Summarize cluster sizes across all heads."""
    all_sizes: list[int] = []

    for head_idx in range(assign.shape[0]):
        counts = torch.bincount(assign[head_idx], minlength=K).to(torch.float32).cpu()
        all_sizes.extend(int(v) for v in counts.tolist())

    size_tensor = torch.tensor(all_sizes, dtype=torch.float32)

    return {
        "cluster_size_mean": float(size_tensor.mean().item()),
        "cluster_size_min": float(size_tensor.min().item()),
        "cluster_size_max": float(size_tensor.max().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf", type=int, default=4, help="Branching factor")
    parser.add_argument("--n-tokens", type=int, default=2000, help="Tokens to capture")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--clusterings", type=str, default="all", help='Comma-separated clustering names or "all"')
    parser.add_argument(
        "--sort-by",
        type=str,
        default="cluster_size_max",
        help="Metric to sort by: cluster_size_min, cluster_size_max, or cluster_size_mean.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible comparisons")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    clustering_methods = _select_methods(CLUSTERING_METHODS, args.clusterings, "clustering")

    print(f"Capturing {args.n_tokens} tokens from {args.model} ...")
    t0 = time.perf_counter()
    capture = _capture_qkv(
        model_name=args.model,
        prompt_text=PROMPT,
        n=args.n_tokens,
        device=DEVICE,
        torch_dtype=DTYPE,
        show_progress=True,
    )
    print(f"Capture done in {time.perf_counter() - t0:.1f}s\n")

    layer_ids = capture.layer_ids()
    layer = LAYER_IDX if LAYER_IDX in layer_ids else layer_ids[len(layer_ids) // 2]
    _, keys_cpu, _ = capture.to_layer_tensors(layer)
    keys = keys_cpu.to(device=DEVICE, dtype=torch.float32)

    H_kv, N, D = keys.shape
    K = max(1, math.ceil(N / args.bf))

    print(f"Layer {layer}: H_kv={H_kv}, N={N}, D={D}")
    print(f"K={K} parents (bf={args.bf})")
    print("=" * 120)

    results = []

    for clust_name, clust_fn in clustering_methods.items():
        print(f"\nClustering: {clust_name} ...")
        t0 = time.perf_counter()
        assign, centers = clust_fn(keys, args.bf)
        clust_time = (time.perf_counter() - t0) * 1000

        info = summarize_cluster_imbalance(assign, centers.shape[1])
        results.append(
            {
                "clustering": clust_name,
                **info,
            }
        )
        print(
            f"  min={info['cluster_size_min']:.0f}  "
            f"max={info['cluster_size_max']:.0f}  "
            f"mean={info['cluster_size_mean']:.2f}"
        )

    if not results:
        print("No clustering methods selected.")
        return

    if args.sort_by not in results[0]:
        available = ", ".join(results[0].keys())
        raise ValueError(f"Unknown sort metric '{args.sort_by}'. Available: {available}")

    results.sort(key=lambda row: row[args.sort_by])

    print("\n" + "=" * 120)
    print(
        f"{'CLUSTERING':<22s} {'MIN':>6s} {'MAX':>6s} {'MEAN':>8s}"
    )
    print("-" * 120)
    for row in results:
        print(
            f"{row['clustering']:<22s} "
            f"{float(row['cluster_size_min']):>6.0f} "
            f"{float(row['cluster_size_max']):>6.0f} "
            f"{float(row['cluster_size_mean']):>8.2f}"
        )

    print("=" * 120)
    best = results[0]
    print(
        f"\nMost balanced by {args.sort_by}: {best['clustering']} "
        f"(min={best['cluster_size_min']:.0f}, max={best['cluster_size_max']:.0f}, "
        f"mean={best['cluster_size_mean']:.2f})"
    )


if __name__ == "__main__":
    main()
