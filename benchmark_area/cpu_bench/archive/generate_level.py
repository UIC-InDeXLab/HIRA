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
    ] = "../kv_sampling/kv_data/kv_data_Meta-Llama-3-8B-Instruct_layer31_20251219_005742.npz",
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


def generate_three_level_structure(
    num_keys: int,
    dim: int,
    branching_factor: int,
    distribution: str = "uniform",
    seed: int = 42,
    real_data_path: Optional[
        str
    ] = "../../tests/kv_sampling/kv_data/kv_data_Meta-Llama-3-8B-Instruct_layer31_20251219_005742.npz",
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a three-level hierarchical structure: K -> P1 -> P2.

    The structure maintains the following indexing constraints:
    - K[p1_idx * branch:(p1_idx+1) * branch] gives the children of P1[p1_idx]
    - P1[p2_idx * branch:(p2_idx+1) * branch] gives the children of P2[p2_idx]

    Args:
        num_keys: Number of leaf keys to generate
        dim: Dimension of vectors
        branching_factor: Number of children per parent
        distribution: Type of data distribution ("uniform", "mixture_of_gaussians", "zipf", "real")
        seed: Random seed
        real_data_path: Path to real data (required if distribution="real")
        device: Device to place tensors on

    Returns:
        Tuple of (K, P1, R1, P2, R2) where:
        - K: Keys tensor of shape (num_keys_padded, dim)
        - P1: Level 1 parents of shape (num_p1, dim)
        - R1: Radii for P1 of shape (num_p1,)
        - P2: Level 2 parents of shape (num_p2, dim)
        - R2: Radii for P2 of shape (num_p2,)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    # Step 2: Build first level (K -> P1)
    num_p1 = (num_keys + branching_factor - 1) // branching_factor
    num_keys_padded = num_p1 * branching_factor

    # Sample P1 parents randomly from keys
    p1_indices = torch.randperm(num_keys)[:num_p1]
    P1_initial = all_keys[p1_indices].clone()

    # Compute distances from keys to P1 parents
    distances_k_to_p1 = torch.cdist(all_keys, P1_initial, p=2)
    sorted_p1_indices = distances_k_to_p1.argsort(dim=1)

    # Track assignments
    p1_child_counts = torch.zeros(num_p1, dtype=torch.long)
    key_assignments = torch.full((num_keys,), -1, dtype=torch.long)

    # Assign each key to its nearest available P1 parent
    for key_idx in range(num_keys):
        for p1_rank in range(num_p1):
            p1_idx = sorted_p1_indices[key_idx, p1_rank].item()
            if p1_child_counts[p1_idx] < branching_factor:
                key_assignments[key_idx] = p1_idx
                p1_child_counts[p1_idx] += 1
                break

    # Reorder keys by their P1 parent
    K = torch.zeros(num_keys_padded, dim, dtype=all_keys.dtype)
    for p1_idx in range(num_p1):
        child_mask = key_assignments == p1_idx
        child_keys = all_keys[child_mask]
        num_children = child_keys.shape[0]
        start_idx = p1_idx * branching_factor
        K[start_idx : start_idx + num_children] = child_keys

    # Compute radii R1 for P1 parents
    R1 = torch.zeros(num_p1, dtype=all_keys.dtype)
    P1 = torch.zeros(num_p1, dim, dtype=all_keys.dtype)

    for p1_idx in range(num_p1):
        start_idx = p1_idx * branching_factor
        end_idx = start_idx + branching_factor
        children = K[start_idx:end_idx]

        # Filter out zero-padded entries
        valid_mask = children.abs().sum(dim=1) > 0
        valid_children = children[valid_mask]

        if valid_children.shape[0] > 0:
            # Compute centroid as the parent
            centroid = valid_children.mean(dim=0)
            P1[p1_idx] = centroid

            # Compute radius as max distance from centroid
            distances_to_centroid = torch.norm(valid_children - centroid, dim=1)
            R1[p1_idx] = distances_to_centroid.max()
        else:
            P1[p1_idx] = 0.0
            R1[p1_idx] = 0.0

    # Step 3: Build second level (P1 -> P2)
    num_p2 = (num_p1 + branching_factor - 1) // branching_factor
    num_p1_padded = num_p2 * branching_factor

    # If P1 needs padding, pad it
    if num_p1_padded > num_p1:
        P1_padded = torch.zeros(num_p1_padded, dim, dtype=P1.dtype)
        P1_padded[:num_p1] = P1
        R1_padded = torch.zeros(num_p1_padded, dtype=R1.dtype)
        R1_padded[:num_p1] = R1
    else:
        P1_padded = P1
        R1_padded = R1

    # Sample P2 parents randomly from P1
    p2_indices = torch.randperm(num_p1)[:num_p2]
    P2_initial = P1[p2_indices].clone()

    # Compute distances from P1 to P2 parents
    distances_p1_to_p2 = torch.cdist(P1, P2_initial, p=2)
    sorted_p2_indices = distances_p1_to_p2.argsort(dim=1)

    # Track assignments
    p2_child_counts = torch.zeros(num_p2, dtype=torch.long)
    p1_assignments = torch.full((num_p1,), -1, dtype=torch.long)

    # Assign each P1 to its nearest available P2 parent
    for p1_idx in range(num_p1):
        for p2_rank in range(num_p2):
            p2_idx = sorted_p2_indices[p1_idx, p2_rank].item()
            if p2_child_counts[p2_idx] < branching_factor:
                p1_assignments[p1_idx] = p2_idx
                p2_child_counts[p2_idx] += 1
                break

    # Reorder P1 by their P2 parent
    P1_reordered = torch.zeros(num_p1_padded, dim, dtype=P1.dtype)
    R1_reordered = torch.zeros(num_p1_padded, dtype=R1.dtype)

    for p2_idx in range(num_p2):
        child_mask = p1_assignments == p2_idx
        child_p1 = P1[child_mask]
        child_r1 = R1[child_mask]
        num_children = child_p1.shape[0]
        start_idx = p2_idx * branching_factor
        P1_reordered[start_idx : start_idx + num_children] = child_p1
        R1_reordered[start_idx : start_idx + num_children] = child_r1

    # Now we need to reorder K to match the reordering of P1
    # Create a mapping from old P1 indices to new P1 indices
    old_to_new_p1 = torch.full((num_p1,), -1, dtype=torch.long)
    for p2_idx in range(num_p2):
        child_mask = p1_assignments == p2_idx
        old_indices = torch.where(child_mask)[0]
        for i, old_idx in enumerate(old_indices):
            new_idx = p2_idx * branching_factor + i
            old_to_new_p1[old_idx] = new_idx

    # Reorder K based on the new P1 ordering
    # Note: K_reordered needs to accommodate the reordered P1 structure
    K_reordered_size = num_p1_padded * branching_factor
    K_reordered = torch.zeros(K_reordered_size, dim, dtype=K.dtype)
    for old_p1_idx in range(num_p1):
        new_p1_idx = old_to_new_p1[old_p1_idx]
        if new_p1_idx >= 0:  # Valid assignment
            old_start = old_p1_idx * branching_factor
            old_end = old_start + branching_factor
            new_start = new_p1_idx * branching_factor
            new_end = new_start + branching_factor
            # Only copy if within bounds
            if old_end <= K.shape[0] and new_end <= K_reordered.shape[0]:
                K_reordered[new_start:new_end] = K[old_start:old_end]

    # Compute radii R2 for P2 parents
    R2 = torch.zeros(num_p2, dtype=P1.dtype)
    P2 = torch.zeros(num_p2, dim, dtype=P1.dtype)

    for p2_idx in range(num_p2):
        start_idx = p2_idx * branching_factor
        end_idx = start_idx + branching_factor
        children = P1_reordered[start_idx:end_idx]

        # Filter out zero-padded entries
        valid_mask = children.abs().sum(dim=1) > 0
        valid_children = children[valid_mask]

        if valid_children.shape[0] > 0:
            # Compute centroid as the parent
            centroid = valid_children.mean(dim=0)
            P2[p2_idx] = centroid

            # Compute radius as max distance from centroid
            distances_to_centroid = torch.norm(valid_children - centroid, dim=1)
            R2[p2_idx] = distances_to_centroid.max()
        else:
            P2[p2_idx] = 0.0
            R2[p2_idx] = 0.0

    # Move tensors to specified device
    K_reordered = K_reordered.to(device)
    P1_reordered = P1_reordered.to(device)
    R1_reordered = R1_reordered.to(device)
    P2 = P2.to(device)
    R2 = R2.to(device)

    return K_reordered, P1_reordered, R1_reordered, P2, R2


def test_three_level_structure():
    """Test the three-level hierarchical structure."""
    print("Testing three-level structure generation...\n")

    # Test parameters
    num_keys = 2048
    dim = 128
    branching_factor = 16  # Using smaller branching factor for testing

    distributions = ["uniform", "mixture_of_gaussians"]

    for dist in distributions:
        print(f"Testing distribution: {dist}")
        K, P1, R1, P2, R2 = generate_three_level_structure(
            num_keys=num_keys,
            dim=dim,
            branching_factor=branching_factor,
            distribution=dist,
            seed=42,
        )

        num_p1 = P1.shape[0]
        num_p2 = P2.shape[0]
        num_keys_padded = K.shape[0]

        print(f"  K shape: {K.shape} (padded from {num_keys} keys)")
        print(f"  P1 shape: {P1.shape}")
        print(f"  R1 shape: {R1.shape}, range: [{R1.min():.4f}, {R1.max():.4f}]")
        print(f"  P2 shape: {P2.shape}")
        print(f"  R2 shape: {R2.shape}, range: [{R2.min():.4f}, {R2.max():.4f}]")

        # Verify the hierarchical structure
        expected_num_p1 = (num_keys + branching_factor - 1) // branching_factor
        expected_num_p2 = (expected_num_p1 + branching_factor - 1) // branching_factor
        expected_keys_padded = expected_num_p1 * branching_factor
        expected_p1_padded = expected_num_p2 * branching_factor

        assert (
            K.shape[0] == expected_keys_padded
        ), f"K shape mismatch: {K.shape[0]} != {expected_keys_padded}"
        assert (
            P1.shape[0] == expected_p1_padded
        ), f"P1 shape mismatch: {P1.shape[0]} != {expected_p1_padded}"
        assert (
            P2.shape[0] == expected_num_p2
        ), f"P2 shape mismatch: {P2.shape[0]} != {expected_num_p2}"

        print(f"  ✓ Structure sizes correct")

        # Test indexing constraint 1: K[p1_idx * branch:(p1_idx+1) * branch] gives children of P1[p1_idx]
        print(f"\n  Testing indexing constraint 1 (K -> P1):")
        for p1_idx in range(min(3, num_p1)):
            start_idx = p1_idx * branching_factor
            end_idx = start_idx + branching_factor
            children = K[start_idx:end_idx]

            # Filter valid children
            valid_mask = children.abs().sum(dim=1) > 0
            valid_children = children[valid_mask]

            if valid_children.shape[0] > 0:
                parent = P1[p1_idx]
                radius = R1[p1_idx]

                # Check that all valid children are within the radius
                distances = torch.norm(valid_children - parent, dim=1)
                max_dist = distances.max()

                print(
                    f"    P1[{p1_idx}]: {valid_children.shape[0]} children, "
                    f"radius={radius:.4f}, max_dist={max_dist:.4f}"
                )

                # Allow small numerical tolerance
                assert (
                    max_dist <= radius + 1e-4
                ), f"Children exceed radius: {max_dist:.6f} > {radius:.6f}"

        print(f"  ✓ Indexing constraint 1 verified")

        # Test indexing constraint 2: P1[p2_idx * branch:(p2_idx+1) * branch] gives children of P2[p2_idx]
        print(f"\n  Testing indexing constraint 2 (P1 -> P2):")
        for p2_idx in range(min(3, num_p2)):
            start_idx = p2_idx * branching_factor
            end_idx = start_idx + branching_factor
            children = P1[start_idx:end_idx]

            # Filter valid children
            valid_mask = children.abs().sum(dim=1) > 0
            valid_children = children[valid_mask]

            if valid_children.shape[0] > 0:
                parent = P2[p2_idx]
                radius = R2[p2_idx]

                # Check that all valid children are within the radius
                distances = torch.norm(valid_children - parent, dim=1)
                max_dist = distances.max()

                print(
                    f"    P2[{p2_idx}]: {valid_children.shape[0]} children, "
                    f"radius={radius:.4f}, max_dist={max_dist:.4f}"
                )

                # Allow small numerical tolerance
                assert (
                    max_dist <= radius + 1e-4
                ), f"Children exceed radius: {max_dist:.6f} > {radius:.6f}"

        print(f"  ✓ Indexing constraint 2 verified")
        print()

    print("✓ All three-level structure tests passed!\n")


def test_three_level_structure_edge_cases():
    """Test three-level structure with num_keys that don't divide evenly by 8 or 64."""
    print("Testing three-level structure edge cases (non-divisible num_keys)...\n")

    # Test parameters
    dim = 128
    branching_factor = 8  # Using branching factor of 8 to match the requirement
    distribution = "uniform"

    # Test cases where num_keys doesn't divide evenly by 8 or 64
    test_cases = [
        # Cases that don't divide by 8
        {"num_keys": 5, "name": "5 keys (< 8)"},
        {"num_keys": 10, "name": "10 keys (not divisible by 8)"},
        {"num_keys": 23, "name": "23 keys (not divisible by 8)"},
        {"num_keys": 31, "name": "31 keys (not divisible by 8)"},
        # Cases that divide by 8 but not by 64
        {"num_keys": 16, "name": "16 keys (= 2*8, not divisible by 64)"},
        {"num_keys": 24, "name": "24 keys (= 3*8, not divisible by 64)"},
        {"num_keys": 40, "name": "40 keys (= 5*8, not divisible by 64)"},
        {"num_keys": 56, "name": "56 keys (= 7*8, not divisible by 64)"},
        # Cases slightly above 64
        {"num_keys": 65, "name": "65 keys (64 + 1)"},
        {"num_keys": 70, "name": "70 keys (not divisible by 8 or 64)"},
        {"num_keys": 100, "name": "100 keys (not divisible by 8 or 64)"},
        # Cases with larger numbers
        {"num_keys": 500, "name": "500 keys (not divisible by 8 or 64)"},
        {"num_keys": 1000, "name": "1000 keys (not divisible by 8 or 64)"},
    ]

    for test_case in test_cases:
        num_keys = test_case["num_keys"]
        name = test_case["name"]

        print(f"Testing: {name}")

        K, P1, R1, P2, R2 = generate_three_level_structure(
            num_keys=num_keys,
            dim=dim,
            branching_factor=branching_factor,
            distribution=distribution,
            seed=42,
        )

        num_p1 = P1.shape[0]
        num_p2 = P2.shape[0]
        num_keys_padded = K.shape[0]

        # Calculate expected dimensions
        # Note: The function pads K to accommodate P1 padding, so:
        # num_p1 = ceil(num_keys / branching_factor)
        # num_p2 = ceil(num_p1 / branching_factor)
        # P1 is padded to num_p2 * branching_factor
        # K is padded to (P1_padded_size) * branching_factor
        expected_num_p1 = (num_keys + branching_factor - 1) // branching_factor
        expected_num_p2 = (expected_num_p1 + branching_factor - 1) // branching_factor
        expected_p1_padded = expected_num_p2 * branching_factor
        expected_keys_padded = (
            expected_p1_padded * branching_factor
        )  # K is sized based on P1 padding

        print(
            f"  num_keys: {num_keys} -> K padded to: {num_keys_padded} (expected: {expected_keys_padded})"
        )
        print(
            f"  num_p1: {expected_num_p1} -> P1 padded to: {num_p1} (expected: {expected_p1_padded})"
        )
        print(f"  num_p2: {num_p2} (expected: {expected_num_p2})")

        # Verify the hierarchical structure dimensions
        assert (
            K.shape[0] == expected_keys_padded
        ), f"K shape mismatch: {K.shape[0]} != {expected_keys_padded}"
        assert (
            P1.shape[0] == expected_p1_padded
        ), f"P1 shape mismatch: {P1.shape[0]} != {expected_p1_padded}"
        assert (
            P2.shape[0] == expected_num_p2
        ), f"P2 shape mismatch: {P2.shape[0]} != {expected_num_p2}"

        # Verify that K has correct dimension
        assert K.shape[1] == dim, f"K dimension mismatch: {K.shape[1]} != {dim}"
        assert P1.shape[1] == dim, f"P1 dimension mismatch: {P1.shape[1]} != {dim}"
        assert P2.shape[1] == dim, f"P2 dimension mismatch: {P2.shape[1]} != {dim}"

        # Verify radii are non-negative
        assert (R1 >= 0).all(), "R1 contains negative values"
        assert (R2 >= 0).all(), "R2 contains negative values"

        # Count non-zero (valid) keys in K
        valid_keys_mask = K.abs().sum(dim=1) > 0
        num_valid_keys = valid_keys_mask.sum().item()

        # We should have at least num_keys valid entries (might have more due to padding)
        assert (
            num_valid_keys >= num_keys
        ), f"Not enough valid keys: {num_valid_keys} < {num_keys}"

        print(f"  Valid keys in K: {num_valid_keys} (original: {num_keys})")

        # Test indexing constraint 1: verify children are within parent radius
        for p1_idx in range(min(3, num_p1)):
            start_idx = p1_idx * branching_factor
            end_idx = start_idx + branching_factor
            children = K[start_idx:end_idx]

            # Filter valid children
            valid_mask = children.abs().sum(dim=1) > 0
            valid_children = children[valid_mask]

            if valid_children.shape[0] > 0:
                parent = P1[p1_idx]
                radius = R1[p1_idx]

                # Check that all valid children are within the radius
                distances = torch.norm(valid_children - parent, dim=1)
                max_dist = distances.max()

                # Allow small numerical tolerance
                assert (
                    max_dist <= radius + 1e-4
                ), f"Level 1: Children exceed radius: {max_dist:.6f} > {radius:.6f}"

        # Test indexing constraint 2: verify P1 children are within P2 parent radius
        for p2_idx in range(min(3, num_p2)):
            start_idx = p2_idx * branching_factor
            end_idx = start_idx + branching_factor
            children = P1[start_idx:end_idx]

            # Filter valid children
            valid_mask = children.abs().sum(dim=1) > 0
            valid_children = children[valid_mask]

            if valid_children.shape[0] > 0:
                parent = P2[p2_idx]
                radius = R2[p2_idx]

                # Check that all valid children are within the radius
                distances = torch.norm(valid_children - parent, dim=1)
                max_dist = distances.max()

                # Allow small numerical tolerance
                assert (
                    max_dist <= radius + 1e-4
                ), f"Level 2: Children exceed radius: {max_dist:.6f} > {radius:.6f}"

        print(f"  ✓ All constraints verified\n")

    print("✓ All edge case tests passed!\n")


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
        print()


if __name__ == "__main__":
    # Test the two-level structure
    test_generate_level()

    print("\n" + "=" * 80 + "\n")

    # Test the three-level structure
    test_three_level_structure()

    print("\n" + "=" * 80 + "\n")

    # Test the three-level structure with edge cases
    test_three_level_structure_edge_cases()
