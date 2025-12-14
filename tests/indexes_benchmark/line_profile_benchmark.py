#!/usr/bin/env python3
"""
Line-by-line profiling of the range search implementation.

Install line_profiler first:
    pip install line_profiler

Run with:
    kernprof -l -v line_profile_benchmark.py

This will show line-by-line timing for functions decorated with @profile.
To profile specific functions, add the @profile decorator to them in the source code.
"""

import argparse
import math
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hira.index import KMeansIndex, KMeansIndexConfig, Index
from hira.search import HalfspaceSearcher

# Force single-threaded execution
torch.set_num_threads(1)


def generate_uniform_data(num_keys: int, dim: int, seed: int = 42) -> torch.Tensor:
    """Generate uniformly distributed random vectors (Gaussian distribution)."""
    torch.manual_seed(seed)
    return torch.randn(num_keys, dim)


def generate_uniform_sphere_data(
    num_keys: int, dim: int, seed: int = 42
) -> torch.Tensor:
    """Generate vectors uniformly distributed on the unit sphere."""
    torch.manual_seed(seed)
    keys = torch.randn(num_keys, dim)
    # Normalize each vector to lie on the unit sphere
    keys = keys / torch.norm(keys, dim=1, keepdim=True)
    return keys


def generate_anisotropic_gaussian_data(
    num_keys: int, dim: int, seed: int = 42
) -> torch.Tensor:
    """Generate anisotropic Gaussian data with varying variances per dimension."""
    torch.manual_seed(seed)
    # Create diagonal covariance with exponentially decaying variances
    variances = torch.exp(-torch.arange(dim, dtype=torch.float32) / (dim / 4))
    keys = torch.randn(num_keys, dim) * variances.sqrt()
    return keys


def generate_mixture_of_gaussians_data(
    num_keys: int, dim: int, num_gaussians: int = 5, seed: int = 42
) -> torch.Tensor:
    """Generate mixture of Gaussians with random means and covariances."""
    torch.manual_seed(seed)

    # Generate random means for each Gaussian
    means = torch.randn(num_gaussians, dim) * 2

    # Generate random covariance scales for each Gaussian
    scales = torch.rand(num_gaussians, dim) * 0.5 + 0.3

    # Assign each point to a random Gaussian component
    assignments = torch.randint(0, num_gaussians, (num_keys,))

    # Generate points from their assigned Gaussian
    keys = torch.zeros(num_keys, dim)
    for i in range(num_gaussians):
        mask = assignments == i
        num_in_component = mask.sum().item()
        if num_in_component > 0:
            keys[mask] = means[i] + torch.randn(num_in_component, dim) * scales[i]

    return keys


def generate_zipf_data(
    num_keys: int, dim: int, num_clusters: int = 10, s: float = 1.5, seed: int = 42
) -> torch.Tensor:
    """Generate data where cluster sizes follow a Zipf distribution.

    Args:
        num_keys: Total number of keys to generate
        dim: Dimension of vectors
        num_clusters: Number of clusters
        s: Zipf parameter (larger s = more skewed distribution)
        seed: Random seed

    Returns:
        Tensor of shape (num_keys, dim)
    """
    torch.manual_seed(seed)

    # Generate Zipf probabilities for cluster sizes
    # P(k) ~ 1 / k^s
    ranks = torch.arange(1, num_clusters + 1, dtype=torch.float32)
    probabilities = 1.0 / (ranks**s)
    probabilities = probabilities / probabilities.sum()

    # Assign number of points to each cluster
    cluster_sizes = (probabilities * num_keys).long()
    # Adjust to ensure we have exactly num_keys points
    diff = num_keys - cluster_sizes.sum().item()
    cluster_sizes[0] += diff

    # Generate random cluster centers
    cluster_centers = torch.randn(num_clusters, dim) * 2

    # Generate random covariance scales for each cluster
    scales = torch.rand(num_clusters, dim) * 0.5 + 0.3

    # Generate points from each cluster
    keys = torch.zeros(num_keys, dim)
    current_idx = 0
    for i in range(num_clusters):
        size = cluster_sizes[i].item()
        if size > 0:
            end_idx = current_idx + size
            keys[current_idx:end_idx] = (
                cluster_centers[i] + torch.randn(size, dim) * scales[i]
            )
            current_idx = end_idx

    return keys


def generate_data(
    num_keys: int, dim: int, distribution: str = "uniform", seed: int = 42
) -> torch.Tensor:
    """Generate test data based on distribution type."""
    if distribution == "uniform":
        return generate_uniform_data(num_keys, dim, seed)
    elif distribution == "uniform_sphere":
        return generate_uniform_sphere_data(num_keys, dim, seed)
    elif distribution == "anisotropic_gaussian":
        return generate_anisotropic_gaussian_data(num_keys, dim, seed)
    elif distribution == "mixture_of_gaussians":
        return generate_mixture_of_gaussians_data(num_keys, dim, seed=seed)
    elif distribution == "zipf":
        return generate_zipf_data(num_keys, dim, seed=seed)
    else:
        raise ValueError(
            f"Unknown distribution: {distribution}. Choose from: uniform, uniform_sphere, anisotropic_gaussian, mixture_of_gaussians, zipf"
        )


@profile
def brute_force_search(index: "Index", query: torch.Tensor, threshold: float):
    """Brute-force search."""
    query_norm = query / torch.norm(query, p=2)
    scores = torch.matmul(index.keys, query_norm)
    result = (scores >= threshold).nonzero(as_tuple=True)[0]
    return result


@profile
def profile_indexed_search(index, query, threshold):
    """Profile the indexed search."""
    searcher = HalfspaceSearcher(enable_profiling=True)
    result = searcher.search(query, threshold, index)
    return result, searcher


def profile_brute_force(index, query, threshold):
    """Profile brute-force search."""
    return brute_force_search(index, query, threshold)


DEFAULT_NUM_KEYS = 100000
DEFAULT_DIM = 128
DEFAULT_BRANCHING_FACTOR = 32
DEFAULT_LEVELS = None  # Auto-calculate
DEFAULT_ITERATIONS = 1
DEFAULT_DISTRIBUTION = "uniform"
DEFAULT_DEVICE = "cpu"
DEFAULT_TARGET_RESULTS = 10


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Line-by-line profiling of the range search implementation."
    )
    parser.add_argument(
        "--num-keys",
        type=int,
        default=DEFAULT_NUM_KEYS,
        help="Number of keys to generate (default: 100000)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_DIM,
        help="Dimension of vectors (default: 128)",
    )
    parser.add_argument(
        "--branching-factor",
        type=int,
        default=DEFAULT_BRANCHING_FACTOR,
        help="Branching factor for index (default: 32)",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=DEFAULT_LEVELS,
        help="Number of levels in index (default: auto-calculated based on num_keys and branching_factor)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Max iterations for KMeans clustering (default: 1)",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default=DEFAULT_DISTRIBUTION,
        choices=[
            "uniform",
            "uniform_sphere",
            "anisotropic_gaussian",
            "mixture_of_gaussians",
            "zipf",
        ],
        help="Data distribution type (default: uniform)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help="Device to use for computation (default: cpu)",
    )
    parser.add_argument(
        "--target-results",
        type=int,
        default=DEFAULT_TARGET_RESULTS,
        help="Target number of results to find (default: 10)",
    )
    return parser.parse_args()


def main():
    """Main profiling function."""
    args = parse_args()

    print("=" * 80)
    print("LINE-BY-LINE PROFILING")
    print("=" * 80)
    print()

    # ===== CONFIGURATION =====
    num_keys = args.num_keys
    dim = args.dim
    branching_factor = args.branching_factor
    num_levels = (
        args.num_levels
        if args.num_levels is not None
        else math.ceil(math.log(num_keys) / math.log(branching_factor))
    )
    max_iterations = args.max_iterations
    data_distribution = args.distribution
    device = args.device
    target_results = args.target_results
    # =========================

    print(f"Configuration:")
    print(f"  Keys: {num_keys}")
    print(f"  Dimension: {dim}")
    print(f"  Levels: {num_levels}")
    print(f"  Branching factor: {branching_factor}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Data distribution: {data_distribution}")
    print(f"  Device: {device}")
    print(f"  Target results: {target_results}")
    print()

    print("Generating data...")
    keys = generate_data(num_keys, dim, distribution=data_distribution)

    print("Building index...")
    config = KMeansIndexConfig(
        num_levels=num_levels,
        branching_factor=branching_factor,
        max_iterations=max_iterations,
        device=device,
    )
    index = KMeansIndex(config)
    index.build(keys)

    print("Creating query...")
    query = torch.randn(dim)

    # Find threshold for target number of results
    query_norm = query / torch.norm(query, p=2)
    all_scores = torch.matmul(keys, query_norm)
    # Sort and find threshold that gives exactly the target number of results
    sorted_scores, _ = torch.sort(all_scores, descending=True)
    threshold = sorted_scores[min(target_results, len(sorted_scores) - 1)].item()
    expected_count = (all_scores >= threshold).sum().item()

    print(
        f"\nRunning profiled searches (threshold={threshold:.4f}, expected ~{expected_count} results)..."
    )

    # Profile indexed search
    print("  Indexed search...")
    result_indexed, searcher = profile_indexed_search(index, query, threshold)
    print(f"    Found {len(result_indexed)} results")

    # Profile brute-force search
    print("  Brute-force search...")
    result_bf = profile_brute_force(index, query, threshold)
    print(f"    Found {len(result_bf)} results")

    print("STATS")
    searcher.print_stats()

    print("\nDone!")


if __name__ == "__main__":
    main()
