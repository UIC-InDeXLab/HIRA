import torch
import numpy as np
from typing import Tuple, Optional


def generate_uniform_data(num_keys: int, dim: int, seed: int = 42) -> torch.Tensor:
    """Generate uniformly distributed random vectors (Gaussian distribution)."""
    torch.manual_seed(seed)
    return torch.randn(num_keys, dim)


def generate_mixture_of_gaussians_data(
    num_keys: int, dim: int, num_gaussians: int = 10, seed: int = 42
) -> torch.Tensor:
    """Generate mixture of Gaussians with random centers and spreads."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate random cluster centers
    cluster_centers = torch.randn(num_gaussians, dim) * 3

    # Generate random standard deviations for each cluster (between 0.3 and 2.0)
    cluster_stds = torch.rand(num_gaussians) * 1.7 + 0.3

    # Assign keys uniformly to clusters
    cluster_assignments = torch.randint(0, num_gaussians, (num_keys,))

    # Generate points from each cluster
    keys = torch.zeros(num_keys, dim)
    for i in range(num_gaussians):
        cluster_mask = cluster_assignments == i
        n_points_in_cluster = cluster_mask.sum().item()

        if n_points_in_cluster > 0:
            # Generate points around this cluster center
            cluster_points = (
                torch.randn(n_points_in_cluster, dim) * cluster_stds[i]
                + cluster_centers[i]
            )
            keys[cluster_mask] = cluster_points

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
    ranks = torch.arange(1, num_clusters + 1, dtype=torch.float32)
    probabilities = 1.0 / (ranks**s)
    probabilities = probabilities / probabilities.sum()

    # Assign number of points to each cluster
    cluster_sizes = (probabilities * num_keys).long()
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

    # print(f"  Loading real data from: {real_data_path}")
    data = np.load(real_data_path)
    keys = torch.from_numpy(data["keys"]).float()

    # Subsample if needed
    if len(keys) > num_keys:
        indices = torch.randperm(len(keys))[:num_keys]
        keys = keys[indices]

    print(f"  Loaded {len(keys)} real keys (dimension={keys.shape[1]})")

    return keys


def generate_parent_child_structure(
    num_keys: int,
    dim: int,
    branching_factor: int,
    distribution: str = "uniform",
    seed: int = 42,
    real_data_path: Optional[
        str
    ] = "/home/mohsen/kvcache/hira/tests/kv_sampling/kv_data/kv_data_Meta-Llama-3-8B-Instruct_layer31_20251219_005742.npz",
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Calculate number of parents
    num_parents = (num_keys + branching_factor - 1) // branching_factor
    num_keys_padded = num_parents * branching_factor

    # Step 1: Generate initial keys using specified distribution
    if distribution == "uniform":
        all_keys = generate_uniform_data(num_keys, dim, seed)
    elif distribution == "mixture_of_gaussians":
        all_keys = generate_mixture_of_gaussians_data(num_keys, dim, seed=seed)
    elif distribution == "zipf":
        all_keys = generate_zipf_data(num_keys, dim, seed=seed)
    elif distribution == "real":
        if real_data_path is None:
            raise ValueError("real_data_path must be provided for 'real' distribution")
        all_keys = generate_real_data(num_keys, dim, real_data_path, seed=seed)
    else:
        raise ValueError(
            f"Unknown distribution: {distribution}. "
            f"Choose from: uniform, mixture_of_gaussians, zipf, real"
        )

    # Step 2: Sample parents randomly from keys
    parent_indices = torch.randperm(num_keys)[:num_parents]
    parents = all_keys[parent_indices].clone()

    # Step 3: Assign each key to its closest available parent
    # We'll use a greedy assignment strategy

    # Compute all pairwise distances (keys to parents)
    # Shape: (num_keys, num_parents)
    distances = torch.cdist(all_keys, parents, p=2)

    # Sort parents by distance for each key (closest first)
    # Shape: (num_keys, num_parents)
    sorted_parent_indices = distances.argsort(dim=1)

    # Track how many children each parent has
    parent_child_counts = torch.zeros(num_parents, dtype=torch.long)

    # Track assignments: which parent each key is assigned to
    key_assignments = torch.full((num_keys,), -1, dtype=torch.long)

    # Greedy assignment: assign each key to its nearest available parent
    for key_idx in range(num_keys):
        # Try parents in order of proximity
        for parent_rank in range(num_parents):
            parent_idx = sorted_parent_indices[key_idx, parent_rank].item()

            # If this parent has room, assign key to it
            if parent_child_counts[parent_idx] < branching_factor:
                key_assignments[key_idx] = parent_idx
                parent_child_counts[parent_idx] += 1
                break

    # Step 4: Reorder keys so they're grouped by parent
    # Initialize output keys tensor with padding
    K = torch.zeros(num_keys_padded, dim, dtype=all_keys.dtype)

    # Group keys by their assigned parent
    for parent_idx in range(num_parents):
        # Find all keys assigned to this parent
        child_mask = key_assignments == parent_idx
        child_keys = all_keys[child_mask]
        num_children = child_keys.shape[0]

        # Place children in their designated slots
        start_idx = parent_idx * branching_factor
        K[start_idx : start_idx + num_children] = child_keys
        # Remaining slots (start_idx + num_children : start_idx + branching_factor)
        # are already zero-padded

    # Step 5: Calculate radii for each parent
    # Radius = distance from centroid to furthest child
    R = torch.zeros(num_parents, dtype=all_keys.dtype)

    for parent_idx in range(num_parents):
        start_idx = parent_idx * branching_factor
        end_idx = start_idx + branching_factor
        children = K[start_idx:end_idx]

        # Filter out zero-padded entries
        # A child is valid if it's not all zeros
        valid_mask = children.abs().sum(dim=1) > 0
        valid_children = children[valid_mask]

        if valid_children.shape[0] > 0:
            # Compute centroid
            centroid = valid_children.mean(dim=0)

            # Compute distances from centroid to all valid children
            distances_to_centroid = torch.norm(valid_children - centroid, dim=1)

            # Radius is the maximum distance
            R[parent_idx] = distances_to_centroid.max()
        else:
            # No valid children (shouldn't happen, but handle gracefully)
            R[parent_idx] = 0.0

    # Move tensors to specified device
    K = K.to(device)
    P = parents.to(device)
    R = R.to(device)

    return K, P, R


def compute_halfspace_statistics(
    K: torch.Tensor,
    P: torch.Tensor,
    R: torch.Tensor,
    q: torch.Tensor,
    t: float,
    branching_factor: int = 256,
) -> dict:
    num_parents = P.shape[0]
    num_keys = K.shape[0]

    # Step 1: Compute parent scores
    parent_scores = (P @ q) + R  # Shape: (num_parents,)

    # Step 2: Determine which parents pass
    parents_passing_mask = parent_scores > t
    num_parents_passing = parents_passing_mask.sum().item()

    # Step 3: Compute scores for all children
    all_child_scores = K @ q  # Shape: (num_keys,)

    # Step 4: For passing parents, collect their children's scores
    passing_children_mask = torch.zeros(num_keys, dtype=torch.bool, device=K.device)

    for parent_idx in range(num_parents):
        if parents_passing_mask[parent_idx]:
            start_idx = parent_idx * branching_factor
            end_idx = start_idx + branching_factor
            passing_children_mask[start_idx:end_idx] = True

    # Get scores of children from passing parents
    passing_children_scores = all_child_scores[passing_children_mask]
    num_children_evaluated = passing_children_scores.shape[0]

    # Compute statistics
    stats = {
        "parents_passing": num_parents_passing,
        "parents_total": num_parents,
        "parents_pass_rate": (
            num_parents_passing / num_parents if num_parents > 0 else 0.0
        ),
        "children_evaluated": num_children_evaluated,
        "children_total": num_keys,
        "children_pass_rate": (
            num_children_evaluated / num_keys if num_keys > 0 else 0.0
        ),
    }

    return stats


def test_generate_level():
    """Test the generate_parent_child_structure function."""
    print("Testing parent-child structure generation...\n")

    # Test parameters
    num_keys = 1000
    dim = 128
    branching_factor = 256

    distributions = ["uniform", "mixture_of_gaussians", "zipf"]

    for dist in distributions:
        print(f"Testing distribution: {dist}")
        K, P, R = generate_parent_child_structure(
            num_keys=num_keys,
            dim=dim,
            branching_factor=branching_factor,
            distribution=dist,
            seed=42,
        )

        num_parents = P.shape[0]
        num_keys_padded = K.shape[0]

        print(f"  Generated {num_parents} parents")
        print(f"  Generated {num_keys_padded} keys (padded)")
        print(f"  K shape: {K.shape}")
        print(f"  P shape: {P.shape}")
        print(f"  R shape: {R.shape}")
        print(f"  R min/max: {R.min():.4f} / {R.max():.4f}")

        # Verify structure
        assert K.shape == (num_parents * branching_factor, dim)
        assert P.shape == (num_parents, dim)
        assert R.shape == (num_parents,)

        # Verify children assignment
        for i in range(min(3, num_parents)):  # Check first 3 parents
            start_idx = i * branching_factor
            end_idx = start_idx + branching_factor
            children = K[start_idx:end_idx]

            # Check that non-zero children are within radius
            valid_mask = children.abs().sum(dim=1) > 0
            valid_children = children[valid_mask]

            if valid_children.shape[0] > 0:
                centroid = valid_children.mean(dim=0)
                max_dist = torch.norm(valid_children - centroid, dim=1).max()
                print(
                    f"  Parent {i}: {valid_children.shape[0]} children, "
                    f"radius={R[i]:.4f}, max_dist={max_dist:.4f}"
                )
                assert torch.allclose(R[i], max_dist, atol=1e-5)

        print()

    # Test halfspace statistics
    print("\nTesting halfspace statistics...\n")

    # Generate a test dataset
    K, P, R = generate_parent_child_structure(
        num_keys=1024,
        dim=128,
        branching_factor=256,
        distribution="mixture_of_gaussians",
        seed=42,
    )

    # Generate a random query
    q = torch.randn(128)
    q = q / q.norm()  # Normalize

    # Test with different thresholds
    parent_scores = (P @ q) + R
    thresholds = [
        ("low (5%)", parent_scores.quantile(0.05).item()),
        ("median (50%)", parent_scores.median().item()),
        ("high (95%)", parent_scores.quantile(0.95).item()),
    ]

    for name, threshold in thresholds:
        stats = compute_halfspace_statistics(
            K, P, R, q, threshold, branching_factor=256
        )
        print(f"Threshold: {name} = {threshold:.4f}")
        print(
            f"  Parents passing: {stats['parents_passing']}/{stats['parents_total']} ({stats['parents_pass_rate']*100:.1f}%)"
        )
        print(
            f"  Children evaluated: {stats['children_evaluated']}/{stats['children_total']} ({stats['children_pass_rate']*100:.1f}%)"
        )
        print(f"  Avg child score (passing parents): {stats['children_avg_score']:.4f}")
        print(f"  Avg child score (all): {stats['all_children_avg_score']:.4f}")
        print()


if __name__ == "__main__":
    test_generate_level()
