#!/usr/bin/env python3
"""
Compare clustering + enclosing methods for halfspace pruning.

For each (clustering_method, enclosing_method) pair, measures what fraction
of children must be scanned when using the parent-level gate to prune.

Usage:
    python method_comparison_bench.py [--bf 16] [--n-tokens 2000] [--n-queries 50]
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

from pruning_bench_utils import _capture_qkv, _q_to_kv_map
from clusterings import CLUSTERING_METHODS
from enclosings import ENCLOSING_METHODS

# ── Model / capture settings ──
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LAYER_IDX = 15
DEVICE = "cuda"
DTYPE = torch.float32

PROMPT = (
    "Solve the following problem step by step, showing all intermediate "
    "reasoning, calculations, and verification.\n\n"
    "A research lab is designing a distributed computing cluster. They have "
    "a budget for 120 machines. Each machine can be configured as a CPU node "
    "(32 cores, 128 GB RAM, $5000), a GPU node (8 cores, 64 GB RAM, 4×A100 "
    "GPUs, $35000), or a storage node (8 cores, 256 GB RAM, 100 TB disk, "
    "$12000). The workload consists of three phases that repeat in a cycle:\n\n"
    "Phase 1 (Training): Requires at least 200 A100 GPUs running in parallel. "
    "Each training job needs 4 GPUs and 48 GB RAM. Communication overhead "
    "between nodes adds 12% latency per additional node beyond the first. "
    "Calculate the optimal GPU node count to minimize total training time for "
    "a 500-epoch run where each epoch takes 45 minutes on a single 4-GPU node.\n\n"
    "Phase 2 (Data Processing): Must process 50 PB of raw data. Each CPU core "
    "can process 2 TB/hour. Storage nodes can serve data at 20 GB/s each but "
    "need 3 replicas for fault tolerance. Calculate the minimum storage and "
    "CPU nodes needed to finish processing within 72 hours.\n\n"
    "Phase 3 (Inference): Must serve 10,000 requests/second with p99 latency "
    "under 100ms. Each GPU can handle 150 requests/second. Each CPU core can "
    "handle 8 requests/second as fallback. The system must maintain 99.99% "
    "uptime, requiring N+2 redundancy.\n\n"
    "Determine the optimal allocation of the 120 machines across all three "
    "node types. Then analyze: What happens if the budget increases by 20%? "
    "What if training data doubles? What if inference load triples? For each "
    "scenario, re-derive the full allocation from scratch, show the math, "
    "compare trade-offs, and explain your reasoning at every step. Finally, "
    "prove mathematically that your allocation is Pareto-optimal across the "
    "three phases, or explain why no single allocation can be."
)


# =====================================================================
#  BENCHMARK CORE
# =====================================================================


def topk_threshold(q_normal, keys, k=20):
    """Ground-truth top-k threshold over all keys."""
    H_kv, N, D = keys.shape
    qg = q_normal.view(H_kv, -1, D)
    w = qg @ keys.transpose(-2, -1)
    w = w.reshape(q_normal.shape[0], -1)
    k = min(k, w.shape[-1])
    th, _ = w.topk(k, dim=-1)
    return th[:, -1]


def measure_scanned_fraction(gate_fn, queries, keys, q_indices, q_head_to_kv, K, bf, topk):
    """Run queries through the gate and measure scanned fraction + search time."""
    H_kv, N, D = keys.shape
    fracs = []
    search_times = []

    for qi in q_indices:
        q = queries[:, qi, :]
        q_norm = q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q_normal = q / q_norm

        # Expand to query heads via q_head_to_kv
        q_kv = q_normal[q_head_to_kv] if q_head_to_kv is not None else q_normal

        th = topk_threshold(q_kv, keys, k=topk)

        # Gate: (H_q, K) bool — timed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        parent_pass = gate_fn(q_kv, th)
        torch.cuda.synchronize()
        search_times.append(time.perf_counter() - t0)

        scanned = parent_pass.sum(dim=1).float() * bf
        frac = scanned / max(1, N)
        fracs.append(frac.mean().item())

    mean_frac = sum(fracs) / len(fracs) if fracs else 1.0
    mean_search_ms = (sum(search_times) / len(search_times)) * 1000 if search_times else 0.0
    return mean_frac, mean_search_ms


def format_speedup(ratio: float) -> str:
    """Format analytical speedup relative to full scan."""
    return f"{1 / ratio:.2f}x"


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf", type=int, default=4, help="Branching factor")
    parser.add_argument("--n-tokens", type=int, default=2000, help="Tokens to capture")
    parser.add_argument("--n-queries", type=int, default=30, help="Number of queries to evaluate")
    parser.add_argument("--topk", type=int, default=20, help="Top-k for threshold")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

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
    queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer)

    keys = keys_cpu.to(device=DEVICE, dtype=torch.float32)
    queries = queries_cpu.to(device=DEVICE, dtype=torch.float32)
    H_kv, N, D = keys.shape
    H_q = queries.shape[0]

    q_head_to_kv = _q_to_kv_map(H_q, H_kv, DEVICE) if H_q != H_kv else None
    K = max(1, math.ceil(N / args.bf))

    # Query indices: sample from end of sequence
    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride))
    q_indices = q_indices[: args.n_queries]

    print(f"Layer {layer}: H_kv={H_kv}, H_q={H_q}, N={N}, D={D}")
    print(f"K={K} parents (bf={args.bf}), {len(q_indices)} queries, topk={args.topk}")
    print("=" * 90)

    results = []

    for clust_name, clust_fn in CLUSTERING_METHODS.items():
        print(f"\nClustering: {clust_name} ...")
        t0 = time.perf_counter()
        assign, centers = clust_fn(keys, args.bf)
        clust_time = time.perf_counter() - t0

        # Expand centers/assign to query heads if needed
        if q_head_to_kv is not None:
            assign_q = assign[q_head_to_kv]
            centers_q = centers[q_head_to_kv]
            keys_q = keys[q_head_to_kv]
        else:
            assign_q = assign
            centers_q = centers
            keys_q = keys

        for enc_name, enc_fn in ENCLOSING_METHODS.items():
            t1 = time.perf_counter()
            gate_fn, enc_info = enc_fn(keys_q, assign_q, centers_q, K, args.bf)
            enc_time = time.perf_counter() - t1

            frac, search_ms = measure_scanned_fraction(
                gate_fn, queries, keys_q, q_indices, None, K, args.bf, args.topk
            )

            results.append({
                "clustering": clust_name,
                "enclosing": enc_name,
                "scanned_frac": frac,
                "clust_ms": clust_time * 1000,
                "enc_ms": enc_time * 1000,
                "search_ms": search_ms,
                **{f"enc_{k}": v for k, v in enc_info.items()},
            })

            pruning = 1.0 - frac
            print(
                f"  {enc_name:<20s}  scanned={frac:.4f}  pruned={pruning:.4f}  "
                f"search={search_ms:.3f}ms  "
                f"clust={clust_time*1000:.1f}ms  enc={enc_time*1000:.1f}ms  "
                + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in enc_info.items())
            )

    # ── Gate cost per enclosing method (in dot-product equivalents) ──
    # A dot product of D-dim vectors costs 2D FLOPs.
    # gate_g = gate_FLOPs_per_cluster / (2*D)
    GATE_COST_DP = {
        "ball_centroid": 1.0,       # einsum (2D) + add + cmp
        "min_enclosing_ball": 1.0,  # same gate as ball
        "aabb": 1.5,                # 2 muls + max + sum (3D)
        "cone": 1.5,                # einsum + trig (~2D+20)
        "hybrid": 3.5,              # ball + AABB + cone
        "ellipsoid": 2.5,           # einsum + scaled norm (5D)
        "split_aabb": 3.0,          # 2x AABB
        "split_hybrid": 6.5,        # split_aabb + ball + ellipsoid
        "split_full_hybrid": 9.1,   # split_aabb + ball + AABB + cone + ellipsoid
        "hybrid_plus": 9.5,         # ball + AABB + cone + ellipsoid + centerline
        "quad_aabb": 6.0,           # 4x AABB
        "bisect_aabb": 3.5,         # 2x AABB + ball
        "slab_bundle": 2.0,         # projection + ball
        "pca_obb": 1.5,             # rotated AABB
        "topk_aabb_residual": 3.0,  # partial AABB + residual
        "centerline": 3.0,          # einsum + proj + residual
        "span_ball": 1.0,           # einsum + add (same as ball)
    }

    # ── Summary table ──
    print("\n" + "=" * 120)
    print(
        f"{'CLUSTERING':<22s} {'ENCLOSING':<22s} {'SCANNED':>8s} {'PRUNED':>8s} "
        f"{'SEARCH_ms':>10s} {'BUILD_ms':>9s} {'g':>5s} {'RATIO':>7s} {'SPEEDUP':>8s}"
    )
    print("-" * 120)

    results.sort(key=lambda r: r["scanned_frac"])
    for r in results:
        build_ms = r["clust_ms"] + r["enc_ms"]
        pruned = 1.0 - r["scanned_frac"]
        g = GATE_COST_DP.get(r["enclosing"], 2.0)
        ratio = g / args.bf + (1.0 - pruned)  # g/bf + (1-p)  where p=pruned
        print(
            f"{r['clustering']:<22s} {r['enclosing']:<22s} "
            f"{r['scanned_frac']:>8.4f} {pruned:>8.4f} "
            f"{r['search_ms']:>10.3f} {build_ms:>9.1f} "
            f"{g:>5.1f} {ratio:>7.3f} {format_speedup(ratio):>8s}"
        )

    print("=" * 120)
    best = results[0]
    print(
        f"\nBest pruning: {best['clustering']} + {best['enclosing']} "
        f"-> scanned={best['scanned_frac']:.4f} (pruned {1-best['scanned_frac']:.4f})"
    )
    # Find best analytical speedup
    best_speedup = min(results, key=lambda r:
        GATE_COST_DP.get(r["enclosing"], 2.0) / args.bf + r["scanned_frac"])
    g_best = GATE_COST_DP.get(best_speedup["enclosing"], 2.0)
    ratio_best = g_best / args.bf + best_speedup["scanned_frac"]
    print(
        f"Best speedup: {best_speedup['clustering']} + {best_speedup['enclosing']} "
        f"-> ratio={ratio_best:.3f} ({format_speedup(ratio_best)})"
    )


if __name__ == "__main__":
    main()
