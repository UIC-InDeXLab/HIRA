import pytest
import torch
import torch.nn.functional as F
from hira.index.indexer import CPUIndexer
from hira.index.searcher import CPUSearcher


class TestIndexConstruction:
    """Test that index is constructed correctly from random synthetic points."""

    def test_index_basic_structure(self):
        """Test basic index structure after construction."""
        num_keys = 1000
        head_dim = 64

        # Generate random synthetic points
        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        # Build index
        config = {
            "num_levels": 3,
            "branching_factor": 10,
            "max_iterations": 25,
        }
        index = CPUIndexer(**config)
        index.build(keys)

        # Verify basic structure
        assert index.num_keys == num_keys
        assert index.dim == head_dim
        assert len(index.levels) == 3
        assert index.keys is not None
        assert index.keys.shape == (num_keys, head_dim)

    def test_level_0_contains_all_keys(self):
        """Test that level 0 contains all original keys."""
        num_keys = 500
        head_dim = 32

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = {
            "num_levels": 3,
            "branching_factor": 8,
            "max_iterations": 1,
        }
        index = CPUIndexer(**config)
        index.build(keys)

        level_0 = index.levels[0]

        # Level 0 should have all keys
        assert level_0.size == num_keys
        assert len(level_0.ball_centers) == num_keys

        # All indices should be present
        # expected_indices = set(range(num_keys))
        # actual_indices = set(level_0.ball_centers.tolist())
        assert level_0.ball_centers.equal(keys)

    def test_hierarchical_size_reduction(self):
        """Test that each level reduces size by approximately branching_factor."""
        num_keys = 10000
        head_dim = 64
        branching_factor = 10

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = {"num_levels": 4, "branching_factor": branching_factor}
        index = CPUIndexer(**config)
        index.build(keys)

        # Check each level reduces in size
        for i in range(len(index.levels) - 1):
            current_size = index.levels[i].size
            next_size = index.levels[i + 1].size

            # Next level should be roughly current_size / branching_factor
            # Allow some tolerance for clustering variance
            expected_ratio = branching_factor
            actual_ratio = current_size / max(next_size, 1)

            assert next_size <= current_size
            # Ratio should be within reasonable bounds (allow 50% variance)
            assert actual_ratio >= expected_ratio * 0.5
            assert actual_ratio <= expected_ratio * 2.0

    def test_cluster_assignments_validity(self):
        """Test that cluster assignments are valid and consistent."""
        num_keys = 2000
        head_dim = 64

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = {"num_levels": 3, "branching_factor": 10}
        index = CPUIndexer(**config)
        index.build(keys)

        # Check assignments for each level except the last
        for level_idx in range(len(index.levels) - 1):
            level = index.levels[level_idx]

            # child2parent should exist for all levels except the last
            assert level.child2parent is not None

            # child2parent should have same length as number of keys in level
            assert len(level.child2parent) == level.size

            # All parent indices should be valid
            next_level = index.levels[level_idx + 1]
            valid_parent_indices = set(range(next_level.size))

            for child_idx, parent_idx in enumerate(level.child2parent.tolist()):
                # Parent index should be valid (0 to next_level.size - 1)
                assert (
                    parent_idx >= 0
                ), f"Invalid parent index {parent_idx} at child {child_idx}"
                assert (
                    parent_idx < next_level.size
                ), f"Parent index {parent_idx} out of bounds (max: {next_level.size - 1})"

            # Verify that each parent in next level has at least one child
            parent_counts = torch.bincount(
                level.child2parent, minlength=next_level.size
            )
            assert torch.all(
                parent_counts > 0
            ), "Some parents in next level have no children"

    def test_cluster_radii_properties(self):
        """Test that cluster radii properly bound their points."""
        num_keys = 1000
        head_dim = 64

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = {"num_levels": 3, "branching_factor": 10}
        index = CPUIndexer(**config)
        index.build(keys)

        # Check each level except level 0 (which has zero radii)
        for level_idx in range(1, len(index.levels)):
            level = index.levels[level_idx]

            # All radii should be non-negative
            assert torch.all(level.ball_radii >= 0)

            # Centers should have the correct dimension
            assert level.ball_centers.shape == (level.size, head_dim)

            # For each cluster in this level, verify the radius bounds all its children
            # The child2parent mapping is stored in the PREVIOUS level
            prev_level = index.levels[level_idx - 1]

            if prev_level.child2parent is not None:
                # For each parent cluster in the current level
                for parent_idx in range(level.size):
                    # Find all children that belong to this parent
                    child_mask = prev_level.child2parent == parent_idx
                    child_indices = torch.nonzero(child_mask, as_tuple=True)[0]

                    if len(child_indices) > 0:
                        center = level.ball_centers[parent_idx]
                        radius = level.ball_radii[parent_idx]

                        # Get the actual child points from the previous level
                        child_points = prev_level.ball_centers[child_indices]
                        # child_points = index.keys[child_points_ptrs]

                        # All points should be within radius of center
                        distances = torch.norm(
                            child_points - center.unsqueeze(0), dim=1
                        )
                        max_distance = torch.max(distances) if len(distances) > 0 else 0

                        # Radius should be at least as large as max distance
                        # (allow small numerical tolerance)
                        assert (
                            radius >= max_distance - 1e-5
                        ), f"Cluster {parent_idx}: radius {radius:.6f} < max_distance {max_distance:.6f}"


class TestRangeSearchEmpty:
    """Test range search queries that should return no results."""

    def test_empty_result_orthogonal_query(self):
        """Test query orthogonal to all keys returns empty result."""
        num_keys = 1000
        head_dim = 64

        torch.manual_seed(42)
        # Create keys in a specific subspace
        keys = torch.zeros(num_keys, head_dim)
        keys[:, : head_dim // 2] = torch.randn(num_keys, head_dim // 2)

        # Build index
        config = {"num_levels": 3, "branching_factor": 10}
        index = CPUIndexer(**config)
        index.build(keys)

        # Create query orthogonal to keys (in the other half of dimensions)
        query = torch.zeros(head_dim)
        query[head_dim // 2 :] = torch.randn(head_dim // 2)
        query = F.normalize(query, p=2, dim=0)

        threshold = 0.01  # Very small positive threshold

        searcher = CPUSearcher(numba=True)
        result = searcher.search(query, threshold, index)

        # Verify ground truth is empty
        scores = torch.matmul(keys, query)
        ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

        # Both should be empty
        assert len(result) == 0 or len(result) == len(ground_truth)
        assert len(ground_truth) == 0

    def test_empty_result_high_threshold(self):
        """Test very high threshold returns no results."""
        num_keys = 1000
        head_dim = 64

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = {"num_levels": 3, "branching_factor": 10}
        index = CPUIndexer(**config)
        index.build(keys)

        query = torch.randn(head_dim)
        query = F.normalize(query, p=2, dim=0)

        # Very high threshold
        threshold = 1000.0

        searcher = CPUSearcher(numba=True)
        result = searcher.search(query, threshold, index)

        # Ground truth
        scores = torch.matmul(keys, query)
        ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

        assert len(result) == len(ground_truth)
        assert len(ground_truth) == 0

    def test_empty_result_opposite_direction(self):
        """Test query opposite to all keys returns empty result."""
        num_keys = 500
        head_dim = 32

        torch.manual_seed(42)
        # Create keys pointing in positive direction
        keys = torch.abs(torch.randn(num_keys, head_dim))

        config = {"num_levels": 2, "branching_factor": 10}
        index = CPUIndexer(**config)
        index.build(keys)

        # Query in negative direction
        query = -torch.abs(torch.randn(head_dim))
        query = F.normalize(query, p=2, dim=0)

        threshold = 0.0

        searcher = CPUSearcher(numba=True)
        result = searcher.search(query, threshold, index)

        scores = torch.matmul(keys, query)
        ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

        # Both should be empty or very small
        assert len(result) == len(ground_truth)


class TestRangeSearchManyResults:
    """Test range search queries that return many results."""

    def test_many_results_low_threshold(self):
        """Test low threshold returns many results with 99%+ recall."""
        num_keys = 2000
        head_dim = 64

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = {"num_levels": 3, "branching_factor": 10}
        index = CPUIndexer(**config)
        index.build(keys)

        query = torch.randn(head_dim)
        query = F.normalize(query, p=2, dim=0)

        threshold = -5.0  # Low threshold should match many keys

        searcher = CPUSearcher(numba=True)
        result = searcher.search(query, threshold, index)

        # Ground truth
        scores = torch.matmul(keys, query)
        ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

        # Verify recall >= 99%
        result_set = set(result.tolist())
        ground_truth_set = set(ground_truth.tolist())

        # Calculate recall
        if len(ground_truth_set) > 0:
            recall = len(result_set & ground_truth_set) / len(ground_truth_set)
            assert recall >= 0.99, f"Recall {recall*100:.2f}% is below 99%"

        # No false positives
        if len(result_set) > 0:
            precision = len(result_set & ground_truth_set) / len(result_set)
            assert (
                precision == 1.0
            ), f"Found {len(result_set - ground_truth_set)} false positives"

        # Should have many results
        assert len(ground_truth) > num_keys * 0.3

    def test_many_results_aligned_query(self):
        """Test query aligned with cluster returns many results with 99%+ recall."""
        num_keys = 1000
        head_dim = 64

        torch.manual_seed(42)
        # Create keys with some correlation
        base = torch.randn(head_dim)

        keys = torch.randn(num_keys, head_dim)
        # Add bias toward base direction
        keys = keys + 2.0 * base.unsqueeze(0)

        config = {"num_levels": 3, "branching_factor": 10}
        index = CPUIndexer(**config)
        index.build(keys)

        # Query in the base direction
        query = base + 0.1 * torch.randn(head_dim)
        query = F.normalize(query, p=2, dim=0)

        threshold = 0.3

        searcher = CPUSearcher(numba=True)
        result = searcher.search(query, threshold, index)

        # Ground truth
        scores = torch.matmul(keys, query)
        ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

        # Verify recall >= 99%
        result_set = set(result.tolist())
        ground_truth_set = set(ground_truth.tolist())

        if len(ground_truth_set) > 0:
            recall = len(result_set & ground_truth_set) / len(ground_truth_set)
            assert recall >= 0.99, f"Recall {recall*100:.2f}% is below 99%"

        # No false positives
        if len(result_set) > 0:
            precision = len(result_set & ground_truth_set) / len(result_set)
            assert (
                precision == 1.0
            ), f"Found {len(result_set - ground_truth_set)} false positives"

        # Should have many results
        assert len(ground_truth) > num_keys * 0.2

    def test_all_keys_returned(self):
        """Test very low threshold returns all keys with 99%+ recall."""
        num_keys = 500
        head_dim = 32

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = {"num_levels": 2, "branching_factor": 10}
        index = CPUIndexer(**config)
        index.build(keys)

        query = torch.randn(head_dim)
        query = F.normalize(query, p=2, dim=0)

        threshold = -100.0  # Should match all vectors

        searcher = CPUSearcher(numba=True)
        result = searcher.search(query, threshold, index)

        # Ground truth
        scores = torch.matmul(keys, query)
        ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

        # Should return all keys
        assert len(ground_truth) == num_keys

        # Verify recall >= 99%
        result_set = set(result.tolist())
        ground_truth_set = set(ground_truth.tolist())

        recall = len(result_set & ground_truth_set) / len(ground_truth_set)
        assert recall >= 0.99, f"Recall {recall*100:.2f}% is below 99%"

        # No false positives
        if len(result_set) > 0:
            precision = len(result_set & ground_truth_set) / len(result_set)
            assert precision == 1.0, f"Found false positives"


class TestRangeSearchRecall:
    """Test that recall is at least 99% for various query scenarios."""

    @pytest.mark.parametrize("num_keys", [100, 500, 1000, 5000])
    @pytest.mark.parametrize("threshold", [-0.8, -0.5, 0.0, 0.3, 0.5, 0.7])
    def test_perfect_recall_random_queries(self, num_keys, threshold):
        """Test 99%+ recall for random queries with various thresholds."""
        head_dim = 64

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = {"num_levels": 3, "branching_factor": 10}
        index = CPUIndexer(**config)
        index.build(keys)

        # Test multiple random queries
        num_queries = 5
        for seed in range(num_queries):
            torch.manual_seed(seed)
            query = torch.randn(head_dim)
            query = F.normalize(query, p=2, dim=0)

            searcher = CPUSearcher(numba=True)
            result = searcher.search(query, threshold, index)

            # Ground truth
            scores = torch.matmul(keys, query)
            ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

            # Verify recall >= 99%
            result_set = set(result.tolist())
            ground_truth_set = set(ground_truth.tolist())

            # Calculate recall
            if len(ground_truth_set) > 0:
                recall = len(result_set & ground_truth_set) / len(ground_truth_set)
                assert (
                    recall >= 0.99
                ), f"Recall {recall*100:.2f}% is below 99% for threshold={threshold}, num_keys={num_keys}"

            # No false positives
            if len(result_set) > 0:
                precision = len(result_set & ground_truth_set) / len(result_set)
                assert (
                    precision == 1.0
                ), f"Precision {precision*100:.2f}% for threshold={threshold}, num_keys={num_keys}"

    def test_recall_with_different_dimensions(self):
        """Test recall >= 99% with different head dimensions."""
        num_keys = 1000

        for head_dim in [16, 32, 64, 128]:
            torch.manual_seed(42)
            keys = torch.randn(num_keys, head_dim)

            config = {"num_levels": 3, "branching_factor": 10}
            index = CPUIndexer(**config)
            index.build(keys)

            query = torch.randn(head_dim)
            query = F.normalize(query, p=2, dim=0)
            threshold = 0.5

            searcher = CPUSearcher(numba=True)
            result = searcher.search(query, threshold, index)

            # Ground truth
            scores = torch.matmul(keys, query)
            ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

            # Verify recall >= 99%
            result_set = set(result.tolist())
            ground_truth_set = set(ground_truth.tolist())

            if len(ground_truth_set) > 0:
                recall = len(result_set & ground_truth_set) / len(ground_truth_set)
                assert (
                    recall >= 0.99
                ), f"Recall {recall*100:.2f}% is below 99% for head_dim={head_dim}"

            # No false positives
            if len(result_set) > 0:
                precision = len(result_set & ground_truth_set) / len(result_set)
                assert (
                    precision == 1.0
                ), f"Found false positives for head_dim={head_dim}"

    def test_recall_with_different_branching_factors(self):
        """Test 99%+ recall with different branching factors."""
        num_keys = 2000
        head_dim = 64

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        query = torch.randn(head_dim)
        query = F.normalize(query, p=2, dim=0)
        threshold = 0.4

        for branching_factor in [5, 10, 20, 50]:
            config = {"num_levels": 3, "branching_factor": branching_factor}
            index = CPUIndexer(**config)
            index.build(keys)

            searcher = CPUSearcher(numba=True)
            result = searcher.search(query, threshold, index)

            # Ground truth
            scores = torch.matmul(keys, query)
            ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

            # Verify recall >= 99%
            result_set = set(result.tolist())
            ground_truth_set = set(ground_truth.tolist())

            if len(ground_truth_set) > 0:
                recall = len(result_set & ground_truth_set) / len(ground_truth_set)
                assert (
                    recall >= 0.99
                ), f"Recall {recall*100:.2f}% is below 99% for branching_factor={branching_factor}"

            # No false positives
            if len(result_set) > 0:
                precision = len(result_set & ground_truth_set) / len(result_set)
                assert (
                    precision == 1.0
                ), f"Found false positives for branching_factor={branching_factor}"

    def test_recall_boundary_cases(self):
        """Test 99%+ recall at threshold boundaries."""
        num_keys = 1000
        head_dim = 64

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = {"num_levels": 3, "branching_factor": 10}
        index = CPUIndexer(**config)
        index.build(keys)

        query = torch.randn(head_dim)
        query = F.normalize(query, p=2, dim=0)

        # Compute actual score distribution
        scores = torch.matmul(keys, query)

        # Test at specific score values (boundary cases)
        test_thresholds = [
            scores.min().item(),
            scores.median().item(),
            scores.max().item() - 0.01,
        ]

        for threshold in test_thresholds:
            searcher = CPUSearcher(numba=True)
            result = searcher.search(query, threshold, index)

            ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

            result_set = set(result.tolist())
            ground_truth_set = set(ground_truth.tolist())

            if len(ground_truth_set) > 0:
                recall = len(result_set & ground_truth_set) / len(ground_truth_set)
                assert (
                    recall >= 0.99
                ), f"Recall {recall*100:.2f}% is below 99% at boundary threshold={threshold:.4f}"

            # No false positives
            if len(result_set) > 0:
                precision = len(result_set & ground_truth_set) / len(result_set)
                assert (
                    precision == 1.0
                ), f"Found false positives at threshold={threshold:.4f}"


class TestRangeSearchEdgeCases:
    """Test edge cases in range search."""

    def test_very_small_dataset(self):
        """Test with very small dataset (< branching factor)."""
        num_keys = 5
        head_dim = 32

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = {"num_levels": 2, "branching_factor": 10}
        index = CPUIndexer(**config)
        index.build(keys)

        query = torch.randn(head_dim)
        query = F.normalize(query, p=2, dim=0)
        threshold = 0.0

        searcher = CPUSearcher(numba=True)
        result = searcher.search(query, threshold, index)

        scores = torch.matmul(keys, query)
        ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

        result_set = set(result.tolist())
        ground_truth_set = set(ground_truth.tolist())

        if len(ground_truth_set) > 0:
            recall = len(result_set & ground_truth_set) / len(ground_truth_set)
            assert recall >= 0.99, f"Recall {recall*100:.2f}% is below 99%"

        # No false positives
        if len(result_set) > 0:
            precision = len(result_set & ground_truth_set) / len(result_set)
            assert precision == 1.0, f"Found false positives"

    def test_threshold_exactly_on_score(self):
        """Test threshold exactly equal to some key scores."""
        num_keys = 1000
        head_dim = 64

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = {"num_levels": 3, "branching_factor": 10}
        index = CPUIndexer(**config)
        index.build(keys)

        query = torch.randn(head_dim)
        query = F.normalize(query, p=2, dim=0)

        scores = torch.matmul(keys, query)
        # Use an actual score value as threshold
        threshold = scores[100].item()

        searcher = CPUSearcher(numba=False)
        result = searcher.search(query, threshold, index)

        ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

        result_set = set(result.tolist())
        ground_truth_set = set(ground_truth.tolist())

        # Key at index 100 should be included (>= not just >)
        assert 100 in result_set

        # Verify recall >= 99%
        if len(ground_truth_set) > 0:
            recall = len(result_set & ground_truth_set) / len(ground_truth_set)
            assert recall >= 0.99, f"Recall {recall*100:.2f}% is below 99%"

        # No false positives
        if len(result_set) > 0:
            precision = len(result_set & ground_truth_set) / len(result_set)
            assert precision == 1.0, f"Found false positives"

    def test_unnormalized_query_handling(self):
        """Test that search properly normalizes queries."""
        num_keys = 500
        head_dim = 64

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = {"num_levels": 2, "branching_factor": 10}
        index = CPUIndexer(**config)
        index.build(keys)

        # Create unnormalized query
        query_unnormalized = torch.randn(head_dim) * 10.0
        query_normalized = F.normalize(query_unnormalized, p=2, dim=0)

        threshold = 0.5

        searcher = CPUSearcher(numba=True)
        result = searcher.search(query_unnormalized, threshold, index)

        # Ground truth using normalized query
        scores = torch.matmul(keys, query_normalized)
        ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

        result_set = set(result.tolist())
        ground_truth_set = set(ground_truth.tolist())

        # Should handle normalization correctly with recall >= 99%
        if len(ground_truth_set) > 0:
            recall = len(result_set & ground_truth_set) / len(ground_truth_set)
            assert recall >= 0.99, f"Recall {recall*100:.2f}% is below 99%"

        # No false positives
        if len(result_set) > 0:
            precision = len(result_set & ground_truth_set) / len(result_set)
            assert precision == 1.0, f"Found false positives"


def brute_force_halfspace(keys: torch.Tensor, q: torch.Tensor, threshold: float):
    q = q / q.norm(p=2)
    scores = keys @ q
    return (scores >= threshold).nonzero(as_tuple=True)[0]


@pytest.fixture(scope="module")
def device():
    # Keep tests deterministic and simple: CPU
    return torch.device("cpu")


def make_config(device="cpu", num_levels=4, branching_factor=4, max_iterations=20):
    return {
        "max_iterations": max_iterations,
        # "device": device,
        "num_levels": num_levels,
        "branching_factor": branching_factor,
    }


def make_keys(n=200, d=32, seed=0, device=torch.device("cpu")):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    # Optional: normalize keys if your app assumes that
    # x = x / x.norm(dim=1, keepdim=True)
    return x.to(device)


def test_build_basic_invariants(device):
    keys = make_keys(n=257, d=24, seed=1, device=device)
    cfg = make_config(
        device=device, num_levels=5, branching_factor=4, max_iterations=10
    )
    index = CPUIndexer(**cfg).build(keys)

    assert index.keys is not None
    assert index.num_keys == keys.shape[0]
    assert index.dim == keys.shape[1]
    assert len(index.levels) >= 1

    # Level 0: balls correspond to keys
    L0 = index.levels[0]
    assert L0.level_idx == 0
    assert L0.ball_centers.shape == keys.shape
    assert L0.ball_radii.shape == (keys.shape[0],)
    assert torch.allclose(L0.ball_centers, keys)

    # Root should exist if num_levels>1 AND branching allows it; otherwise it might stop early.
    # But levels should be ordered by level_idx
    for i, L in enumerate(index.levels):
        assert L.level_idx == i
        assert L.size == L.ball_centers.shape[0]
        assert L.ball_radii.shape[0] == L.size


def test_num_parents_matches_parent_size(device):
    keys = make_keys(n=123, d=20, seed=3, device=device)
    cfg = make_config(
        device=device, num_levels=5, branching_factor=3, max_iterations=10
    )
    index = CPUIndexer(**cfg).build(keys)

    for i in range(len(index.levels) - 1):
        child = index.levels[i]
        parent = index.levels[i + 1]
        assert child.num_parents == parent.size


def test_search_matches_bruteforce_random_queries(device):
    keys = make_keys(n=400, d=32, seed=4, device=device)
    cfg = make_config(
        device=device, num_levels=6, branching_factor=4, max_iterations=20
    )
    index = CPUIndexer(**cfg).build(keys)
    searcher = CPUSearcher(profiling=True, numba=True)

    g = torch.Generator(device="cpu")
    g.manual_seed(0)

    for t in [0.0, 0.2, 0.5]:
        for _ in range(10):
            q = torch.randn(keys.shape[1], generator=g, device=device)
            # threshold in dot space; pick something reasonable
            threshold = t

            got = searcher.search(q, threshold, index)  # should return indices
            exp = brute_force_halfspace(keys, q, threshold)

            got_sorted = torch.sort(got).values
            exp_sorted = torch.sort(exp).values

            # Correctness: hierarchical pruning should not drop true positives
            assert torch.equal(
                got_sorted, exp_sorted
            ), f"Mismatch at threshold={threshold}"


def test_search_extreme_thresholds(device):
    keys = make_keys(n=200, d=16, seed=5, device=device)
    cfg = make_config(
        device=device, num_levels=5, branching_factor=4, max_iterations=10
    )
    index = CPUIndexer(**cfg).build(keys)
    searcher = CPUSearcher(profiling=False, numba=True)

    q = torch.randn(keys.shape[1], device=device)

    # Very high threshold -> usually empty
    got = searcher.search(q, threshold=10.0, indexer=index)
    assert got.numel() == 0

    # Very low threshold -> all keys qualify
    got = searcher.search(q, threshold=-10.0, indexer=index)
    assert got.numel() == keys.shape[0]


def test_search_one_level_fallback(device):
    """
    Covers the case len(index.levels)==1 (no hierarchy).
    This requires your search() to handle that case explicitly.
    """
    keys = make_keys(n=50, d=8, seed=6, device=device)
    cfg = make_config(device=device, num_levels=1, branching_factor=4, max_iterations=5)
    index = CPUIndexer(**cfg).build(keys)
    assert len(index.levels) == 1

    searcher = CPUSearcher(profiling=True, numba=True)
    q = torch.randn(keys.shape[1], device=device)
    thr = 0.1

    got = searcher.search(q, thr, index)
    exp = brute_force_halfspace(keys, q, thr)

    assert torch.equal(torch.sort(got).values, torch.sort(exp).values)


def test_search_two_levels_only(device):
    """
    Covers len(index.levels)==2 (level0 + root/level1) where manual L1->L0 expansion must work.
    """
    keys = make_keys(n=80, d=12, seed=7, device=device)
    cfg = make_config(
        device=device, num_levels=2, branching_factor=4, max_iterations=10
    )
    index = CPUIndexer(**cfg).build(keys)
    assert len(index.levels) == 2

    searcher = CPUSearcher(profiling=False, numba=True)
    q = torch.randn(keys.shape[1], device=device)
    thr = 0.0

    got = searcher.search(q, thr, index)
    exp = brute_force_halfspace(keys, q, thr)

    assert torch.equal(torch.sort(got).values, torch.sort(exp).values)


def test_refined_radii_upper_bound_property(device):
    """
    Sanity: refined parent radius should upper bound all children balls:
      For each child c assigned to parent p:
         ||center_c - center_p|| + radius_c <= radius_p
    This checks your triangle-inequality refinement logic.
    """
    keys = make_keys(n=300, d=10, seed=8, device=device)
    cfg = make_config(
        device=device, num_levels=5, branching_factor=5, max_iterations=15
    )
    index = CPUIndexer(**cfg).build(keys)

    for i in range(len(index.levels) - 1):
        child = index.levels[i]
        parent = index.levels[i + 1]

        # Skip if mapping isn't present (shouldn't happen except root)
        assert child.child2parent is not None
        c2p = child.child2parent

        child_centers = child.ball_centers
        child_radii = child.ball_radii
        parent_centers = parent.ball_centers
        parent_radii = parent.ball_radii

        # compute for all children at once:
        # dist(child_center, parent_center[child2parent])
        dist = torch.norm(child_centers - parent_centers[c2p], dim=1)
        lhs = dist + child_radii
        rhs = parent_radii[c2p]

        # Allow tiny floating error
        assert torch.all(lhs <= rhs + 1e-5).item()


class TestIndexerUpdate:
    def test_update_from_empty_one_level(self, device):
        """Starting from an empty index (no hierarchy), update should append keys and search should match brute force."""
        d = 16
        cfg = make_config(
            device=device, num_levels=1, branching_factor=4, max_iterations=5
        )

        empty = torch.empty((0, d), device=device)
        index = CPUIndexer(**cfg).build(empty)
        assert index.num_keys == 0
        assert len(index.levels) == 1

        new1 = make_keys(n=50, d=d, seed=10, device=device)
        index.update(new1)

        assert index.keys is not None
        assert index.keys.shape == new1.shape
        assert torch.allclose(index.keys, new1)
        L0 = index.levels[0]
        assert L0.size == new1.shape[0]
        assert L0.child2parent is None
        assert L0.ball_radii.shape == (new1.shape[0],)
        assert torch.all(L0.ball_radii == 0)

        # Second update should still work
        new2 = make_keys(n=30, d=d, seed=11, device=device)
        index.update(new2)
        all_keys = torch.cat([new1, new2], dim=0)
        assert index.keys.shape == all_keys.shape
        assert torch.allclose(index.keys, all_keys)

        searcher = CPUSearcher(numba=True)
        g = torch.Generator(device="cpu")
        g.manual_seed(123)
        for thr in [0.0, 0.2, 0.5]:
            for _ in range(5):
                q = torch.randn(d, generator=g, device=device)
                got = searcher.search(q, thr, index)
                exp = brute_force_halfspace(all_keys, q, thr)
                assert torch.equal(torch.sort(got).values, torch.sort(exp).values)

    def test_update_two_level_assigns_parent_and_updates_radii(self, device):
        """With 2 levels, update() must (1) append keys, (2) append assignments, (3) expand parent radii as needed."""
        n0, d = 128, 24
        cfg = make_config(
            device=device, num_levels=2, branching_factor=4, max_iterations=10
        )
        keys0 = make_keys(n=n0, d=d, seed=20, device=device)
        index = CPUIndexer(**cfg).build(keys0)
        assert len(index.levels) == 2

        L0 = index.levels[0]
        L1 = index.levels[1]
        assert L0.child2parent is not None
        assert L0.parent_mask_buf is not None
        assert L0.parent_mask_buf.shape == (L1.size,)

        radii_before = L1.ball_radii.clone()

        new = make_keys(n=37, d=d, seed=21, device=device)
        index.update(new)

        # Base storage
        assert index.num_keys == n0 + new.shape[0]
        assert index.keys.shape == (n0 + new.shape[0], d)
        assert L0.size == n0 + new.shape[0]
        assert L0.ball_centers.shape == (n0 + new.shape[0], d)
        assert L0.child2parent.shape == (n0 + new.shape[0],)

        # Parent ids valid
        new_assign = L0.child2parent[-new.shape[0] :]
        assert torch.all((new_assign >= 0) & (new_assign < L1.size)).item()

        # Parent radii monotonic non-decreasing
        assert torch.all(L1.ball_radii >= radii_before - 1e-6).item()

        # Each new key should be within assigned parent's radius
        parent_centers = L1.ball_centers
        dist = torch.norm(new - parent_centers[new_assign], dim=1)
        assert torch.all(dist <= L1.ball_radii[new_assign] + 1e-5).item()

    def test_update_windowed_matches_bruteforce_on_final_keys(self, device):
        """Windowed updates should not crash and should preserve search correctness on the final key set (2-level index)."""
        N, d = 500, 32
        n0 = 200
        w = 37
        cfg = make_config(
            device=device, num_levels=2, branching_factor=4, max_iterations=15
        )

        all_keys = make_keys(n=N, d=d, seed=30, device=device)
        keys0 = all_keys[:n0]
        tail = all_keys[n0:]

        updated = CPUIndexer(**cfg).build(keys0)
        assert len(updated.levels) == 2

        # Apply windowed updates
        start = 0
        while start < tail.shape[0]:
            chunk = tail[start : start + w]
            updated.update(chunk)
            start += w

        # Sanity vs a fresh build from all keys (structure may differ, keys must match)
        full = CPUIndexer(**cfg).build(all_keys)
        assert full.keys is not None
        assert updated.keys is not None
        assert torch.allclose(updated.keys, full.keys)
        assert updated.num_keys == N

        L0u = updated.levels[0]
        L1u = updated.levels[1]
        assert L0u.child2parent is not None
        assert L0u.child2parent.shape == (N,)
        assert torch.all((L0u.child2parent >= 0) & (L0u.child2parent < L1u.size)).item()
        assert torch.all(L1u.ball_radii >= 0).item()

        # Search correctness on the final keys
        searcher = CPUSearcher(numba=True)
        g = torch.Generator(device="cpu")
        g.manual_seed(999)
        for thr in [0.0, 0.2, 0.5]:
            for _ in range(10):
                q = torch.randn(d, generator=g, device=device)
                got = searcher.search(q, thr, updated)
                exp = brute_force_halfspace(all_keys, q, thr)
                assert torch.equal(torch.sort(got).values, torch.sort(exp).values)

    def test_update_with_auto_balance_matches_full_build(self, device):
        """update() with auto-balance should preserve correctness and behave like a fresh build on final keys.

        We compare both indices against brute force on the final key set, and also assert
        the two indices return identical results for a set of random queries.
        """
        bf = 4
        d = 16
        n0 = 64
        chunk = 20
        num_updates = 4

        cfg_bal = make_config(
            device=device,
            num_levels=3,
            branching_factor=bf,
            max_iterations=10,
        )
        cfg_bal["balance_every"] = 1  # balance on every update call

        keys0 = make_keys(n=n0, d=d, seed=1234, device=device)
        updated = CPUIndexer(**cfg_bal).build(keys0)
        assert len(updated.levels) == 3

        L0 = updated.levels[0]
        L1 = updated.levels[1]
        assert L0.child2parent is not None
        l1_size_before = L1.size

        # Make updates heavily concentrate on a single parent to force splitting.
        target_parent = int(L0.child2parent[0].item())
        center = L1.ball_centers[target_parent]

        g = torch.Generator(device="cpu")
        g.manual_seed(2026)
        new_chunks = []
        for _ in range(num_updates):
            new = center + 1e-2 * torch.randn(chunk, d, generator=g, device=device)
            new_chunks.append(new)
            updated.update(new)

        all_new = torch.cat(new_chunks, dim=0)
        all_keys = torch.cat([keys0, all_new], dim=0)

        assert updated.keys is not None
        assert torch.allclose(updated.keys, all_keys)

        # Balance should have happened and created at least one extra parent.
        assert updated.levels[1].size > l1_size_before

        # Fresh build baseline on the final keys
        cfg_full = make_config(
            device=device,
            num_levels=3,
            branching_factor=bf,
            max_iterations=10,
        )
        full = CPUIndexer(**cfg_full).build(all_keys)

        # Compare behavior on random queries
        searcher = CPUSearcher(numba=True)
        gq = torch.Generator(device="cpu")
        gq.manual_seed(999)
        for thr in [-0.2, 0.0, 0.2]:
            for _ in range(20):
                q = torch.randn(d, generator=gq, device=device)
                exp = brute_force_halfspace(all_keys, q, thr)

                got_updated = searcher.search(q, thr, updated)
                got_full = searcher.search(q, thr, full)

                assert torch.equal(
                    torch.sort(got_updated).values, torch.sort(exp).values
                )
                assert torch.equal(torch.sort(got_full).values, torch.sort(exp).values)
                assert torch.equal(
                    torch.sort(got_updated).values, torch.sort(got_full).values
                )

        # Invariants: for each level, parent radius must upper-bound child distances.
        for i in range(len(updated.levels) - 1):
            child = updated.levels[i]
            parent = updated.levels[i + 1]
            assert child.child2parent is not None
            c2p = child.child2parent
            dist = torch.norm(child.ball_centers - parent.ball_centers[c2p], dim=1)
            lhs = dist + child.ball_radii
            rhs = parent.ball_radii[c2p]
            assert torch.all(lhs <= rhs + 1e-5).item()

    def test_balance_every_n_keys_triggers_after_multiple_updates(self, device):
        """balance_every counts *keys*, so balance should trigger after enough updates accumulate."""
        bf = 4
        d = 16
        n0 = 64
        chunk = 20

        cfg = make_config(
            device=device,
            num_levels=3,
            branching_factor=bf,
            max_iterations=10,
        )
        cfg["balance_every"] = 2 * chunk  # trigger after two updates worth of keys

        keys0 = make_keys(n=n0, d=d, seed=555, device=device)
        index = CPUIndexer(**cfg).build(keys0)
        L0 = index.levels[0]
        L1 = index.levels[1]
        assert L0.child2parent is not None

        target_parent = int(L0.child2parent[0].item())
        center = L1.ball_centers[target_parent]

        l1_size_before = L1.size

        g = torch.Generator(device="cpu")
        g.manual_seed(777)

        # First update: should NOT trigger balance yet
        index.update(center + 1e-2 * torch.randn(chunk, d, generator=g, device=device))
        assert index.levels[1].size == l1_size_before

        # Second update: should trigger balance
        index.update(center + 1e-2 * torch.randn(chunk, d, generator=g, device=device))
        assert index.levels[1].size > l1_size_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
