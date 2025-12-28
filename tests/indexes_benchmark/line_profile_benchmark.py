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
import numpy as np
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
    num_keys: int, dim: int, num_gaussians: int = 10, seed: int = 42
) -> torch.Tensor:
    """Generate mixture of Gaussians using fitted parameters from real data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Component parameters from GMM fitted to real KV cache data
    gmm_weights = np.array(
        [0.1255, 0.2067, 0.0046, 0.1678, 0.0027, 0.0928, 0.0143, 0.3021, 0.0258, 0.0576]
    )
    gmm_weights = (
        gmm_weights / gmm_weights.sum()
    )  # Normalize to ensure sum is exactly 1.0
    gmm_means = [
        1.3819,
        -0.9039,
        -12.4851,
        0.6147,
        10.9554,
        -1.9572,
        4.2475,
        -0.0570,
        -3.3843,
        2.4689,
    ]
    gmm_stds = [
        0.3879,
        0.3699,
        2.1610,
        0.3008,
        1.9482,
        0.5171,
        1.0753,
        0.2514,
        1.0471,
        0.5737,
    ]

    # Sample component assignments based on weights
    component_assignments = np.random.choice(
        num_gaussians, size=num_keys, p=gmm_weights
    )

    # Generate points from each component
    keys = torch.zeros(num_keys, dim)
    for i in range(num_gaussians):
        # Find indices assigned to this component
        component_mask = component_assignments == i
        n_points_in_component = component_mask.sum()

        if n_points_in_component > 0:
            # Generate points from this Gaussian component
            # Use the fitted mean and std for each dimension
            component_points = (
                torch.randn(n_points_in_component, dim) * gmm_stds[i] + gmm_means[i]
            )
            keys[component_mask] = component_points

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


def generate_real_data(
    num_keys: int, dim: int, real_data_path: str, seed: int = 42
) -> torch.Tensor:
    """Load real KV cache data from NPZ file.

    Args:
        num_keys: Number of keys to use (will subsample if needed)
        dim: Expected dimension (for validation)
        real_data_path: Path to .npz file containing 'keys' array
        seed: Random seed for subsampling

    Returns:
        Tensor of shape (num_keys, dim) containing real KV cache keys
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"  Loading real data from: {real_data_path}")
    data = np.load(real_data_path)
    keys = torch.from_numpy(data["keys"]).float()

    # Subsample if needed
    if len(keys) > num_keys:
        indices = torch.randperm(len(keys))[:num_keys]
        keys = keys[indices]

    print(f"  Loaded {len(keys)} real keys (dimension={keys.shape[1]})")

    return keys


def generate_data(
    num_keys: int,
    dim: int,
    distribution: str = "uniform",
    seed: int = 42,
    real_data_path: str = None,
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
    elif distribution == "real":
        if real_data_path is None:
            raise ValueError("real_data_path must be provided for 'real' distribution")
        return generate_real_data(num_keys, dim, real_data_path, seed=seed)
    else:
        raise ValueError(
            f"Unknown distribution: {distribution}. Choose from: uniform, uniform_sphere, anisotropic_gaussian, mixture_of_gaussians, zipf, real"
        )


@profile
def brute_force_search(
    index: "Index", query: torch.Tensor, threshold: float, num_runs: int = 1
):
    """Brute-force search."""
    for _ in range(num_runs):
        query_norm = query / torch.norm(query, p=2)
        scores = torch.matmul(index.keys, query_norm)
        result = (scores >= threshold).nonzero(as_tuple=True)[0]
    return result


@profile
def profile_indexed_search(index, query, threshold, num_runs: int = 1):
    """Profile the indexed search."""
    for _ in range(num_runs):
        searcher = HalfspaceSearcher(
            enable_profiling=True
        )  # make True if you want stats
        result = searcher.search(query, threshold, index)
    return result, searcher


def profile_brute_force(index, query, threshold, num_runs: int = 1):
    """Profile brute-force search."""
    return brute_force_search(index, query, threshold, num_runs=num_runs)


DEFAULT_NUM_KEYS = 100000
DEFAULT_DIM = 128
DEFAULT_BRANCHING_FACTOR = 32
DEFAULT_LEVELS = None  # Auto-calculate
DEFAULT_ITERATIONS = 1
DEFAULT_DISTRIBUTION = "uniform"
DEFAULT_DEVICE = "cpu"
DEFAULT_TARGET_RESULTS = 10
DEFAULT_NUM_RUNS = 1


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
            "real",
        ],
        help="Data distribution type (default: uniform)",
    )
    parser.add_argument(
        "--real-data-path",
        type=str,
        default="/home/mohsen/kvcache/hira/tests/kv_sampling/kv_data/kv_data_Meta-Llama-3-8B-Instruct_layer31_20251219_005742.npz",
        help="Path to .npz file containing real KV cache data (required when --distribution=real)",
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
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help="Number of runs to average for profiling (default: 1)",
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
    num_runs = args.num_runs
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
    keys = generate_data(
        num_keys,
        dim,
        distribution=data_distribution,
        real_data_path=args.real_data_path,
    )
    keys = keys.to(device)

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
    query = torch.randn(dim).to(device)

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
    result_indexed, searcher = profile_indexed_search(index, query, threshold, num_runs)
    print(f"    Found {len(result_indexed)} results")

    # Profile brute-force search
    print("  Brute-force search...")
    result_bf = profile_brute_force(index, query, threshold, num_runs)
    print(f"    Found {len(result_bf)} results")

    print("STATS")
    searcher.print_stats()

    print("\nDone!")


if __name__ == "__main__":
    main()
