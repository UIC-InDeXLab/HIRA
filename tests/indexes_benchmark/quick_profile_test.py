#!/usr/bin/env python3
"""
Quick profiling test to compare indexed vs brute-force search.

This script provides a focused analysis on a single configuration
to understand where the indexed search spends time and how it compares
to brute-force.
"""

import math
import torch
import numpy as np
import time
from hira.index import KMeansIndex, KMeansIndexConfig
from hira.search import HalfspaceSearcher


def generate_uniform_data(num_keys: int, dim: int, seed: int = 42) -> torch.Tensor:
    """Generate uniformly distributed random vectors."""
    torch.manual_seed(seed)
    return torch.randn(num_keys, dim)


def generate_gaussian_mixture(
    num_keys: int, dim: int, num_clusters: int = 10, seed: int = 42
) -> torch.Tensor:
    """Generate data from a mixture of Gaussians."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate cluster centers
    centers = torch.randn(num_clusters, dim) * 5

    # Assign each key to a cluster
    cluster_assignments = np.random.choice(num_clusters, size=num_keys)

    # Generate keys around cluster centers
    keys = []
    for i in range(num_keys):
        cluster_id = cluster_assignments[i]
        center = centers[cluster_id]
        noise = torch.randn(dim) * 0.5  # Small variance around centers
        keys.append(center + noise)

    return torch.stack(keys)


def generate_zipf_data(
    num_keys: int, dim: int, alpha: float = 1.5, num_modes: int = 20, seed: int = 42
) -> torch.Tensor:
    """Generate data following a Zipf-like distribution with multiple modes."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate mode centers
    modes = torch.randn(num_modes, dim) * 3

    # Generate Zipf distribution for mode frequencies
    # More keys will be assigned to earlier modes
    frequencies = np.random.zipf(alpha, size=num_keys)
    frequencies = np.clip(frequencies, 1, num_modes)
    mode_assignments = frequencies - 1

    # Generate keys around modes with varying spread
    keys = []
    for i in range(num_keys):
        mode_id = int(mode_assignments[i])
        center = modes[mode_id]
        # Earlier modes have tighter clustering
        variance = 0.2 + (mode_id / num_modes) * 2.0
        noise = torch.randn(dim) * variance
        keys.append(center + noise)

    return torch.stack(keys)


def main():
    print("=" * 80)
    print("QUICK PROFILING TEST")
    print("=" * 80)

    # Configuration
    num_keys = 100000
    dim = 128
    # Use branching factor that works better with the data size
    # For 100k keys, branching_factor=100 gives 2 levels (100 -> 10000 clusters)
    branching_factor = 10
    num_levels = math.ceil(math.log(num_keys) / math.log(branching_factor))

    # ===== CHOOSE DATA DISTRIBUTION HERE =====
    # Options:
    #   "uniform"           - Random uniform vectors (no structure)
    #   "gaussian_mixture"  - Mixture of 10 Gaussian clusters (moderate structure)
    #   "zipf"              - Zipf distribution with 20 modes (heavy structure)
    data_distribution = "uniform"
    # ==========================================

    print(f"\nConfiguration:")
    print(f"  Keys: {num_keys}")
    print(f"  Dimension: {dim}")
    print(f"  Index: {num_levels} levels, branching factor {branching_factor}")
    print(f"  Data distribution: {data_distribution}")

    # Generate data based on selected distribution
    print(f"\nGenerating {data_distribution} data...")
    torch.manual_seed(42)

    if data_distribution == "uniform":
        keys = generate_uniform_data(num_keys, dim)
    elif data_distribution == "gaussian_mixture":
        keys = generate_gaussian_mixture(num_keys, dim, num_clusters=10)
    elif data_distribution == "zipf":
        keys = generate_zipf_data(num_keys, dim, alpha=1.5, num_modes=20)
    else:
        raise ValueError(f"Unknown distribution: {data_distribution}")

    query = torch.randn(dim)
    query = torch.nn.functional.normalize(query, dim=0)  # Normalize query for search

    # Build index
    print(f"\nBuilding index...")
    config = KMeansIndexConfig(
        num_levels=num_levels,
        branching_factor=branching_factor,
        max_iterations=100,
        device="cpu",
        verbose=False,
    )
    index_builder = KMeansIndex(config)
    index = index_builder.build(keys=keys, device=torch.device("cpu"))
    print(f"  Index built with {index.num_levels()} levels")

    # Test with different thresholds
    # Compute score distribution to select meaningful thresholds
    # IMPORTANT: Use ALL keys to compute correct threshold
    all_scores = torch.matmul(keys, query)
    percentiles = [0.9]
    thresholds = torch.quantile(all_scores, torch.tensor(percentiles)).tolist()

    print(f"\nTesting with {len(thresholds)} thresholds...")

    for i, threshold in enumerate(thresholds):
        print(f"\n{'='*80}")
        print(f"TEST {i+1}/{len(thresholds)}: Threshold = {threshold:.4f}")
        print(f"{'='*80}")

        # INDEXED SEARCH with profiling
        print(f"\n--- INDEXED SEARCH ---")
        searcher = HalfspaceSearcher(enable_profiling=True)

        # Warm-up
        _ = searcher.search(query=query, threshold=threshold, index=index, keys=keys)

        # Actual timed run
        searcher = HalfspaceSearcher(enable_profiling=True)
        start = time.perf_counter()
        result_indexed = searcher.search(
            query=query, threshold=threshold, index=index, keys=keys
        )
        indexed_time = time.perf_counter() - start

        searcher.print_stats()

        # BRUTE-FORCE SEARCH
        print(f"\n--- BRUTE-FORCE SEARCH ---")

        # Warm-up
        _ = torch.matmul(keys, query)

        # Actual timed run
        start = time.perf_counter()
        scores = torch.matmul(keys, query)
        matmul_time = time.perf_counter() - start

        start = time.perf_counter()
        result_bf = (scores >= threshold).nonzero(as_tuple=True)[0]
        filter_time = time.perf_counter() - start

        bf_time = matmul_time + filter_time

        print(f"Matrix multiplication: {matmul_time*1000:.3f} ms")
        print(f"Filtering: {filter_time*1000:.3f} ms")
        print(f"Total: {bf_time*1000:.3f} ms")
        print(f"Distance computations: {len(keys)}")
        print(f"Results: {len(result_bf)}")

        # COMPARISON
        print(f"\n--- COMPARISON ---")
        speedup = bf_time / indexed_time
        dist_reduction = 100 * (
            1 - searcher.stats["num_distance_computations"] / len(keys)
        )

        # Compute precision and recall
        result_indexed_set = set(result_indexed.cpu().tolist())
        result_bf_set = set(result_bf.cpu().tolist())

        true_positives = len(result_indexed_set & result_bf_set)
        false_positives = len(result_indexed_set - result_bf_set)
        false_negatives = len(result_bf_set - result_indexed_set)

        precision = (
            true_positives / len(result_indexed_set)
            if len(result_indexed_set) > 0
            else 0.0
        )
        recall = true_positives / len(result_bf_set) if len(result_bf_set) > 0 else 1.0

        print(f"Indexed time: {indexed_time*1000:.3f} ms")
        print(f"Brute-force time: {bf_time*1000:.3f} ms")
        print(f"Speedup: {speedup:.2f}x {'✓ FASTER' if speedup > 1 else '✗ SLOWER'}")
        print(f"Distance computation reduction: {dist_reduction:.1f}%")
        print(f"\nAccuracy:")
        print(
            f"  Precision: {precision:.4f} ({true_positives}/{len(result_indexed_set)})"
        )
        print(f"  Recall: {recall:.4f} ({true_positives}/{len(result_bf_set)})")
        print(
            f"  Results match: {torch.equal(torch.sort(result_indexed)[0], torch.sort(result_bf)[0])}"
        )

        # Analysis
        if speedup < 1:
            print(f"\n--- WHY IS IT SLOWER? ---")
            overhead = indexed_time - bf_time
            print(f"Total overhead: {overhead*1000:.3f} ms")
            print(f"\nBreakdown of indexed search time:")
            print(
                f"  Centroid scoring: {searcher.stats['time_centroid_scoring']*1000:.3f} ms ({100*searcher.stats['time_centroid_scoring']/indexed_time:.1f}%)"
            )
            print(
                f"  Radius checking: {searcher.stats['time_radius_checking']*1000:.3f} ms ({100*searcher.stats['time_radius_checking']/indexed_time:.1f}%)"
            )
            print(
                f"  Data movement: {searcher.stats['time_data_movement']*1000:.3f} ms ({100*searcher.stats['time_data_movement']/indexed_time:.1f}%)"
            )
            print(
                f"  Exact filtering: {searcher.stats['time_exact_filtering']*1000:.3f} ms ({100*searcher.stats['time_exact_filtering']/indexed_time:.1f}%)"
            )

    print(f"\n{'='*80}")
    print("PROFILING COMPLETE")
    print(f"{'='*80}")
    print(f"\nTo visualize PyTorch profiling:")
    print(f"  1. Uncomment the PyTorch profiler section in test_index_performance.py")
    print(f"  2. Open chrome://tracing in Chrome browser")
    print(f"  3. Load the generated .json trace files")


if __name__ == "__main__":
    main()
