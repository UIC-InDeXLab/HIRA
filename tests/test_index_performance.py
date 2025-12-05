"""
Comprehensive performance test for hierarchical indices.

This test evaluates KMeansIndex and RandomizedClustering across:
- Different synthetic data distributions (uniform, Gaussian mixtures, Zipf)
- Various hierarchy configurations (levels, branching factors)
- Different search scenarios (thresholds, result set sizes)
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple
import json
import argparse
from tqdm import tqdm

from hira.index import (
    KMeansIndex,
    KMeansIndexConfig,
    RandomizedClustering,
    RandomizedClusteringConfig,
)
from hira.search import HalfspaceSearcher


# Global percentile configuration for threshold selection
PERCENTILES = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]


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


def build_index_with_timing(
    index_class, config, keys: torch.Tensor, device: torch.device
) -> Tuple[any, float, Dict]:
    """Build an index and measure timing and statistics."""
    index_obj = index_class(config)

    start_time = time.time()
    index = index_obj.build(keys=keys, device=device)
    build_time = time.time() - start_time

    # Gather statistics
    stats = {
        "num_keys": index.num_keys,
        "head_dim": index.head_dim,
        "num_levels": index.num_levels(),
        "build_time_sec": build_time,
        "levels": [],
    }

    total_clusters = 0
    for level_idx in range(index.num_levels()):
        level = index.get_level(level_idx)
        num_clusters = level.num_clusters()
        total_clusters += num_clusters

        level_stats = {
            "level_idx": level_idx,
            "num_clusters": num_clusters,
            "avg_radius": level.cluster_radii.mean().item(),
            "min_radius": level.cluster_radii.min().item(),
            "max_radius": level.cluster_radii.max().item(),
            "std_radius": level.cluster_radii.std().item(),
        }
        stats["levels"].append(level_stats)

    stats["total_clusters"] = total_clusters
    memory_usage = index.total_memory_usage()
    stats["memory_mb"] = memory_usage["total_mb"]

    return index, build_time, stats


def perform_search_benchmark(
    index,
    keys: torch.Tensor,
    num_queries: int = 100,
    thresholds: List[float] = None,
    seed: int = 42,
) -> Dict:
    """Benchmark search performance with multiple queries and thresholds."""
    torch.manual_seed(seed)

    if thresholds is None:
        # Auto-select thresholds based on data statistics
        sample_scores = []
        for _ in range(min(10, num_queries)):
            q = torch.randn(keys.shape[1])
            scores = torch.matmul(keys, q)
            sample_scores.append(scores)

        all_scores = torch.cat(sample_scores)
        percentiles = torch.quantile(all_scores, torch.tensor(PERCENTILES))
        thresholds = percentiles.tolist()

    searcher = HalfspaceSearcher()
    results = {
        "num_queries": num_queries,
        "thresholds": thresholds,
        "threshold_results": [],
    }

    for threshold in thresholds:
        threshold_stats = {
            "threshold": threshold,
            "search_times_ms": [],
            "num_results": [],
            "precision": [],
            "recall": [],
        }

        for query_idx in tqdm(
            range(num_queries), desc=f"  Threshold {threshold:.2f}", leave=False
        ):
            # Generate random query
            query = torch.randn(keys.shape[1], device=keys.device)
            query = torch.nn.functional.normalize(query, dim=0)

            # Measure search time
            start_time = time.time()
            candidate_indices = searcher.search(
                query=query, threshold=threshold, index=index, keys=keys
            )
            search_time = (time.time() - start_time) * 1000  # Convert to ms

            # Compute ground truth for precision/recall
            true_scores = torch.matmul(keys, query)
            true_positives = (true_scores >= threshold).nonzero(as_tuple=True)[0]

            # Calculate metrics
            num_candidates = len(candidate_indices)
            num_true = len(true_positives)

            if num_candidates > 0:
                # Check precision: how many candidates are actually above threshold
                candidate_scores = true_scores[candidate_indices]
                correct_candidates = (candidate_scores >= threshold).sum().item()
                precision = correct_candidates / num_candidates
            else:
                precision = 0.0 if num_true > 0 else 1.0

            # Calculate recall: how many true positives were found
            if num_true > 0:
                found_true = len(
                    set(candidate_indices.tolist()) & set(true_positives.tolist())
                )
                recall = found_true / num_true
            else:
                recall = 1.0 if num_candidates == 0 else 0.0

            threshold_stats["search_times_ms"].append(search_time)
            threshold_stats["num_results"].append(num_candidates)
            threshold_stats["precision"].append(precision)
            threshold_stats["recall"].append(recall)

        # Compute summary statistics
        threshold_stats["avg_search_time_ms"] = np.mean(
            threshold_stats["search_times_ms"]
        )
        threshold_stats["std_search_time_ms"] = np.std(
            threshold_stats["search_times_ms"]
        )
        threshold_stats["avg_num_results"] = np.mean(threshold_stats["num_results"])
        threshold_stats["avg_precision"] = np.mean(threshold_stats["precision"])
        threshold_stats["avg_recall"] = np.mean(threshold_stats["recall"])

        # Remove per-query details to save space
        del threshold_stats["search_times_ms"]
        del threshold_stats["num_results"]
        del threshold_stats["precision"]
        del threshold_stats["recall"]

        results["threshold_results"].append(threshold_stats)

    return results


def perform_bruteforce_search(
    keys: torch.Tensor,
    num_queries: int = 100,
    thresholds: List[float] = None,
    seed: int = 42,
) -> Dict:
    """Benchmark brute-force search (no index) for comparison."""
    torch.manual_seed(seed)

    if thresholds is None:
        # Auto-select thresholds based on data statistics
        sample_scores = []
        for _ in range(min(10, num_queries)):
            q = torch.randn(keys.shape[1])
            scores = torch.matmul(keys, q)
            sample_scores.append(scores)

        all_scores = torch.cat(sample_scores)
        percentiles = torch.quantile(all_scores, torch.tensor(PERCENTILES))
        thresholds = percentiles.tolist()

    results = {
        "num_queries": num_queries,
        "thresholds": thresholds,
        "threshold_results": [],
    }

    for threshold in thresholds:
        threshold_stats = {
            "threshold": threshold,
            "search_times_ms": [],
            "num_results": [],
        }

        for query_idx in tqdm(
            range(num_queries), desc=f"  Threshold {threshold:.2f}", leave=False
        ):
            # Generate random query
            query = torch.randn(keys.shape[1], device=keys.device)
            query = torch.nn.functional.normalize(query, dim=0)

            # Measure brute-force search time
            start_time = time.time()
            scores = torch.matmul(keys, query)
            result_indices = (scores >= threshold).nonzero(as_tuple=True)[0]
            search_time = (time.time() - start_time) * 1000  # Convert to ms

            threshold_stats["search_times_ms"].append(search_time)
            threshold_stats["num_results"].append(len(result_indices))

        # Compute summary statistics
        threshold_stats["avg_search_time_ms"] = np.mean(
            threshold_stats["search_times_ms"]
        )
        threshold_stats["std_search_time_ms"] = np.std(
            threshold_stats["search_times_ms"]
        )
        threshold_stats["avg_num_results"] = np.mean(threshold_stats["num_results"])

        # Remove per-query details to save space
        del threshold_stats["search_times_ms"]
        del threshold_stats["num_results"]

        results["threshold_results"].append(threshold_stats)

    return results


def test_configuration(
    data_dist: str,
    index_type: str,
    num_keys: int,
    dim: int,
    num_levels: int,
    branching_factor: int,
    num_queries: int = 50,
    device_str: str = "cpu",
) -> Dict:
    """Test a specific configuration."""
    device = torch.device(device_str)

    print(f"\n{'='*80}")
    print(f"Testing: {index_type} | {data_dist} | keys={num_keys}, dim={dim}")
    print(f"         levels={num_levels}, branch={branching_factor}")
    print(f"{'='*80}")

    # Generate data
    print(f"Generating {data_dist} data...")
    if data_dist == "uniform":
        keys = generate_uniform_data(num_keys, dim)
    elif data_dist == "gaussian_mixture":
        keys = generate_gaussian_mixture(num_keys, dim, num_clusters=20)
    elif data_dist == "zipf":
        keys = generate_zipf_data(num_keys, dim, alpha=1.5, num_modes=30)
    else:
        raise ValueError(f"Unknown distribution: {data_dist}")

    keys = keys.to(device)

    # Build index
    print(f"Building {index_type} index...")
    if index_type == "kmeans":
        config = KMeansIndexConfig(
            num_levels=num_levels,
            branching_factor=branching_factor,
            max_iterations=25,
            device=device_str,
            verbose=False,
        )
        index, build_time, index_stats = build_index_with_timing(
            KMeansIndex, config, keys, device
        )
    elif index_type == "randomized":
        config = RandomizedClusteringConfig(
            num_levels=num_levels,
            branching_factor=branching_factor,
            random_seed=42,
            device=device_str,
        )
        index, build_time, index_stats = build_index_with_timing(
            RandomizedClustering, config, keys, device
        )
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    print(f"  Build time: {build_time:.3f}s")
    print(
        f"  Levels: {index_stats['num_levels']}, Total clusters: {index_stats['total_clusters']}"
    )
    print(f"  Memory: {index_stats['memory_mb']:.2f} MB")

    # Print level details
    for level_stats in index_stats["levels"]:
        print(
            f"    Level {level_stats['level_idx']}: "
            f"{level_stats['num_clusters']} clusters, "
            f"avg_radius={level_stats['avg_radius']:.3f}"
        )

    # Benchmark search
    print(f"Benchmarking indexed search with {num_queries} queries...")
    search_results = perform_search_benchmark(
        index=index, keys=keys, num_queries=num_queries
    )

    print(f"  Indexed search results:")
    for thresh_result in search_results["threshold_results"]:
        print(
            f"    Threshold {thresh_result['threshold']:.2f}: "
            f"avg_time={thresh_result['avg_search_time_ms']:.3f}ms, "
            f"avg_results={thresh_result['avg_num_results']:.1f}, "
            f"precision={thresh_result['avg_precision']:.3f}, "
            f"recall={thresh_result['avg_recall']:.3f}"
        )

    # Benchmark brute-force search
    print(f"Benchmarking brute-force search with {num_queries} queries...")
    bruteforce_results = perform_bruteforce_search(
        keys=keys, num_queries=num_queries, thresholds=search_results["thresholds"]
    )

    print(f"  Brute-force search results:")
    for thresh_result in bruteforce_results["threshold_results"]:
        print(
            f"    Threshold {thresh_result['threshold']:.2f}: "
            f"avg_time={thresh_result['avg_search_time_ms']:.3f}ms, "
            f"avg_results={thresh_result['avg_num_results']:.1f}"
        )

    # Combine results
    result = {
        "configuration": {
            "data_dist": data_dist,
            "index_type": index_type,
            "num_keys": num_keys,
            "dim": dim,
            "num_levels": num_levels,
            "branching_factor": branching_factor,
            "device": device_str,
        },
        "index_stats": index_stats,
        "search_results": search_results,
        "bruteforce_results": bruteforce_results,
    }

    return result


import math


def main():
    """Run comprehensive performance tests."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Performance testing for hierarchical indices"
    )
    parser.add_argument(
        "-n",
        "--num_keys",
        type=int,
        default=10000,
        help="Number of keys in the dataset (default: 10000)",
    )
    args = parser.parse_args()

    num_keys = args.num_keys

    print("=" * 80)
    print("HIERARCHICAL INDEX PERFORMANCE EVALUATION")
    print("=" * 80)
    print(f"Dataset size: {num_keys} keys")
    print("=" * 80)

    # Test configurations
    data_distributions = ["uniform", "gaussian_mixture", "zipf"]
    index_types = ["kmeans", "randomized"]

    # Base configuration
    dim = 128
    num_queries = 50

    # Test different hierarchy configurations
    hierarchy_configs = [
        # {"num_levels": math.ceil(math.log(num_keys, 4)), "branching_factor": 4},
        {"num_levels": math.ceil(math.log(num_keys, 16)), "branching_factor": 16},
        {"num_levels": math.ceil(math.log(num_keys, 32)), "branching_factor": 32},
        {"num_levels": math.ceil(math.log(num_keys, 64)), "branching_factor": 64},
    ]

    all_results = []

    # Run tests
    for data_dist in data_distributions:
        for index_type in index_types:
            for config in hierarchy_configs:
                try:
                    result = test_configuration(
                        data_dist=data_dist,
                        index_type=index_type,
                        num_keys=num_keys,
                        dim=dim,
                        num_levels=config["num_levels"],
                        branching_factor=config["branching_factor"],
                        num_queries=num_queries,
                        device_str="cpu",
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback

                    traceback.print_exc()

    # Save results to JSON
    output_file = f"./performance_results_{num_keys}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Total tests completed: {len(all_results)}")
    print(f"{'='*80}")

    # Generate summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    # Compare KMeans vs Randomized for each distribution
    for data_dist in data_distributions:
        print(f"\n{data_dist.upper()} Distribution:")
        print("-" * 80)

        # Filter results for this distribution
        dist_results = [
            r for r in all_results if r["configuration"]["data_dist"] == data_dist
        ]

        for config in hierarchy_configs:
            kmeans_result = next(
                (
                    r
                    for r in dist_results
                    if r["configuration"]["index_type"] == "kmeans"
                    and r["configuration"]["num_levels"] == config["num_levels"]
                    and r["configuration"]["branching_factor"]
                    == config["branching_factor"]
                ),
                None,
            )

            random_result = next(
                (
                    r
                    for r in dist_results
                    if r["configuration"]["index_type"] == "randomized"
                    and r["configuration"]["num_levels"] == config["num_levels"]
                    and r["configuration"]["branching_factor"]
                    == config["branching_factor"]
                ),
                None,
            )

            if kmeans_result and random_result:
                print(
                    f"\n  Config: levels={config['num_levels']}, branch={config['branching_factor']}"
                )

                km_build = kmeans_result["index_stats"]["build_time_sec"]
                rc_build = random_result["index_stats"]["build_time_sec"]
                print(
                    f"    Build time:    KMeans={km_build:.3f}s, RC={rc_build:.3f}s "
                    f"(speedup: {km_build/rc_build:.2f}x)"
                )

                km_clusters = kmeans_result["index_stats"]["total_clusters"]
                rc_clusters = random_result["index_stats"]["total_clusters"]
                print(f"    Total clusters: KMeans={km_clusters}, RC={rc_clusters}")

                # Compare search performance (use first threshold as representative)
                if (
                    kmeans_result["search_results"]["threshold_results"]
                    and random_result["search_results"]["threshold_results"]
                ):
                    km_search = kmeans_result["search_results"]["threshold_results"][0]
                    rc_search = random_result["search_results"]["threshold_results"][0]

                    km_time = km_search["avg_search_time_ms"]
                    rc_time = rc_search["avg_search_time_ms"]
                    print(
                        f"    Search time:   KMeans={km_time:.3f}ms, RC={rc_time:.3f}ms "
                        f"(speedup: {km_time/rc_time:.2f}x)"
                    )

                    print(
                        f"    Recall:        KMeans={km_search['avg_recall']:.3f}, "
                        f"RC={rc_search['avg_recall']:.3f}"
                    )

                    # Add brute-force comparison
                    if (
                        kmeans_result.get("bruteforce_results")
                        and kmeans_result["bruteforce_results"]["threshold_results"]
                    ):
                        bf_search = kmeans_result["bruteforce_results"][
                            "threshold_results"
                        ][0]
                        bf_time = bf_search["avg_search_time_ms"]
                        print(
                            f"    Brute-force:   {bf_time:.3f}ms "
                            f"(KMeans speedup: {bf_time/km_time:.2f}x, RC speedup: {bf_time/rc_time:.2f}x)"
                        )


if __name__ == "__main__":
    main()
