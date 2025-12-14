"""
Comprehensive tests for hierarchical range search with halfspace queries.

Test scenarios:
1. Index construction validation for random synthetic points
2. Structural correctness: levels, assignments, parent-child relationships
3. Range search with empty results (query with no points inside)
4. Range search with many results (query with lots of points inside)
5. 99%+ recall verification for all query types
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from hira.index import KMeansIndex
from hira.index.config import KMeansIndexConfig
from hira.search import HalfspaceSearcher


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
        config = KMeansIndexConfig(
            num_levels=3, branching_factor=10, max_iterations=25, device="cpu"
        )
        index = KMeansIndex(config)
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

        config = KMeansIndexConfig(num_levels=3, branching_factor=8)
        index = KMeansIndex(config)
        index.build(keys)

        level_0 = index.levels[0]

        # Level 0 should have all keys
        assert level_0.size == num_keys
        assert len(level_0.key_ptrs) == num_keys

        # All indices should be present
        expected_indices = set(range(num_keys))
        actual_indices = set(level_0.key_ptrs.tolist())
        assert actual_indices == expected_indices

    def test_hierarchical_size_reduction(self):
        """Test that each level reduces size by approximately branching_factor."""
        num_keys = 10000
        head_dim = 64
        branching_factor = 10

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = KMeansIndexConfig(num_levels=4, branching_factor=branching_factor)
        index = KMeansIndex(config)
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

        config = KMeansIndexConfig(num_levels=3, branching_factor=10)
        index = KMeansIndex(config)
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
                assert parent_idx >= 0, f"Invalid parent index {parent_idx} at child {child_idx}"
                assert parent_idx < next_level.size, f"Parent index {parent_idx} out of bounds (max: {next_level.size - 1})"
                
            # Verify that each parent in next level has at least one child
            parent_counts = torch.bincount(level.child2parent, minlength=next_level.size)
            assert torch.all(parent_counts > 0), "Some parents in next level have no children"

    def test_cluster_radii_properties(self):
        """Test that cluster radii properly bound their points."""
        num_keys = 1000
        head_dim = 64

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = KMeansIndexConfig(num_levels=3, branching_factor=10)
        index = KMeansIndex(config)
        index.build(keys)

        # Check each level except level 0 (which has zero radii)
        for level_idx in range(1, len(index.levels)):
            level = index.levels[level_idx]

            # All radii should be non-negative
            assert torch.all(level.key_radii >= 0)

            # Centers should have the correct dimension
            assert level.key_centers.shape == (level.size, head_dim)

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
                        center = level.key_centers[parent_idx]
                        radius = level.key_radii[parent_idx]

                        # Get the actual child points from the previous level
                        child_points_ptrs = prev_level.key_ptrs[child_indices]
                        child_points = index.keys[child_points_ptrs]

                        # All points should be within radius of center
                        distances = torch.norm(
                            child_points - center.unsqueeze(0), dim=1
                        )
                        max_distance = torch.max(distances) if len(distances) > 0 else 0

                        # Radius should be at least as large as max distance
                        # (allow small numerical tolerance)
                        assert radius >= max_distance - 1e-5, \
                            f"Cluster {parent_idx}: radius {radius:.6f} < max_distance {max_distance:.6f}"


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
        config = KMeansIndexConfig(num_levels=3, branching_factor=10)
        index = KMeansIndex(config)
        index.build(keys)

        # Create query orthogonal to keys (in the other half of dimensions)
        query = torch.zeros(head_dim)
        query[head_dim // 2 :] = torch.randn(head_dim // 2)
        query = F.normalize(query, p=2, dim=0)

        threshold = 0.01  # Very small positive threshold

        searcher = HalfspaceSearcher()
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

        config = KMeansIndexConfig(num_levels=3, branching_factor=10)
        index = KMeansIndex(config)
        index.build(keys)

        query = torch.randn(head_dim)
        query = F.normalize(query, p=2, dim=0)

        # Very high threshold
        threshold = 1000.0

        searcher = HalfspaceSearcher()
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

        config = KMeansIndexConfig(num_levels=2, branching_factor=10)
        index = KMeansIndex(config)
        index.build(keys)

        # Query in negative direction
        query = -torch.abs(torch.randn(head_dim))
        query = F.normalize(query, p=2, dim=0)

        threshold = 0.0

        searcher = HalfspaceSearcher()
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

        config = KMeansIndexConfig(num_levels=3, branching_factor=10)
        index = KMeansIndex(config)
        index.build(keys)

        query = torch.randn(head_dim)
        query = F.normalize(query, p=2, dim=0)

        threshold = -5.0  # Low threshold should match many keys

        searcher = HalfspaceSearcher()
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
            assert precision == 1.0, f"Found {len(result_set - ground_truth_set)} false positives"

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

        config = KMeansIndexConfig(num_levels=3, branching_factor=10)
        index = KMeansIndex(config)
        index.build(keys)

        # Query in the base direction
        query = base + 0.1 * torch.randn(head_dim)
        query = F.normalize(query, p=2, dim=0)

        threshold = 0.3

        searcher = HalfspaceSearcher()
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
            assert precision == 1.0, f"Found {len(result_set - ground_truth_set)} false positives"

        # Should have many results
        assert len(ground_truth) > num_keys * 0.2

    def test_all_keys_returned(self):
        """Test very low threshold returns all keys with 99%+ recall."""
        num_keys = 500
        head_dim = 32

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = KMeansIndexConfig(num_levels=2, branching_factor=10)
        index = KMeansIndex(config)
        index.build(keys)

        query = torch.randn(head_dim)
        query = F.normalize(query, p=2, dim=0)

        threshold = -100.0  # Should match all vectors

        searcher = HalfspaceSearcher()
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

        config = KMeansIndexConfig(num_levels=3, branching_factor=10)
        index = KMeansIndex(config)
        index.build(keys)

        # Test multiple random queries
        num_queries = 5
        for seed in range(num_queries):
            torch.manual_seed(seed)
            query = torch.randn(head_dim)
            query = F.normalize(query, p=2, dim=0)

            searcher = HalfspaceSearcher()
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

            config = KMeansIndexConfig(num_levels=3, branching_factor=10)
            index = KMeansIndex(config)
            index.build(keys)

            query = torch.randn(head_dim)
            query = F.normalize(query, p=2, dim=0)
            threshold = 0.5

            searcher = HalfspaceSearcher()
            result = searcher.search(query, threshold, index)

            # Ground truth
            scores = torch.matmul(keys, query)
            ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

            # Verify recall >= 99%
            result_set = set(result.tolist())
            ground_truth_set = set(ground_truth.tolist())

            if len(ground_truth_set) > 0:
                recall = len(result_set & ground_truth_set) / len(ground_truth_set)
                assert recall >= 0.99, f"Recall {recall*100:.2f}% is below 99% for head_dim={head_dim}"
            
            # No false positives
            if len(result_set) > 0:
                precision = len(result_set & ground_truth_set) / len(result_set)
                assert precision == 1.0, f"Found false positives for head_dim={head_dim}"

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
            config = KMeansIndexConfig(num_levels=3, branching_factor=branching_factor)
            index = KMeansIndex(config)
            index.build(keys)

            searcher = HalfspaceSearcher()
            result = searcher.search(query, threshold, index)

            # Ground truth
            scores = torch.matmul(keys, query)
            ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

            # Verify recall >= 99%
            result_set = set(result.tolist())
            ground_truth_set = set(ground_truth.tolist())

            if len(ground_truth_set) > 0:
                recall = len(result_set & ground_truth_set) / len(ground_truth_set)
                assert recall >= 0.99, f"Recall {recall*100:.2f}% is below 99% for branching_factor={branching_factor}"
            
            # No false positives
            if len(result_set) > 0:
                precision = len(result_set & ground_truth_set) / len(result_set)
                assert precision == 1.0, f"Found false positives for branching_factor={branching_factor}"

    def test_recall_boundary_cases(self):
        """Test 99%+ recall at threshold boundaries."""
        num_keys = 1000
        head_dim = 64

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = KMeansIndexConfig(num_levels=3, branching_factor=10)
        index = KMeansIndex(config)
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
            searcher = HalfspaceSearcher()
            result = searcher.search(query, threshold, index)

            ground_truth = torch.nonzero(scores >= threshold, as_tuple=True)[0]

            result_set = set(result.tolist())
            ground_truth_set = set(ground_truth.tolist())

            if len(ground_truth_set) > 0:
                recall = len(result_set & ground_truth_set) / len(ground_truth_set)
                assert recall >= 0.99, f"Recall {recall*100:.2f}% is below 99% at boundary threshold={threshold:.4f}"
            
            # No false positives
            if len(result_set) > 0:
                precision = len(result_set & ground_truth_set) / len(result_set)
                assert precision == 1.0, f"Found false positives at threshold={threshold:.4f}"


class TestRangeSearchEdgeCases:
    """Test edge cases in range search."""

    # def test_single_key(self):
    #     """Test search with only one key."""
    #     head_dim = 64

    #     key = torch.randn(1, head_dim)

    #     config = KMeansIndexConfig(num_levels=1, branching_factor=10)
    #     index = KMeansIndex(config)
    #     index.build(key)

    #     query = torch.randn(head_dim)
    #     query = F.normalize(query, p=2, dim=0)

    #     score = torch.matmul(key[0], query).item()

    #     # Threshold below score should return the key
    #     searcher = HalfspaceSearcher()
    #     result = searcher.search(query, score - 0.1, index)
    #     assert len(result) == 1

    #     # Threshold above score should return nothing
    #     result = searcher.search(query, score + 0.1, index)
    #     assert len(result) == 0

    def test_very_small_dataset(self):
        """Test with very small dataset (< branching factor)."""
        num_keys = 5
        head_dim = 32

        torch.manual_seed(42)
        keys = torch.randn(num_keys, head_dim)

        config = KMeansIndexConfig(num_levels=2, branching_factor=10)
        index = KMeansIndex(config)
        index.build(keys)

        query = torch.randn(head_dim)
        query = F.normalize(query, p=2, dim=0)
        threshold = 0.0

        searcher = HalfspaceSearcher()
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

        config = KMeansIndexConfig(num_levels=3, branching_factor=10)
        index = KMeansIndex(config)
        index.build(keys)

        query = torch.randn(head_dim)
        query = F.normalize(query, p=2, dim=0)

        scores = torch.matmul(keys, query)
        # Use an actual score value as threshold
        threshold = scores[100].item()

        searcher = HalfspaceSearcher()
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

        config = KMeansIndexConfig(num_levels=2, branching_factor=10)
        index = KMeansIndex(config)
        index.build(keys)

        # Create unnormalized query
        query_unnormalized = torch.randn(head_dim) * 10.0
        query_normalized = F.normalize(query_unnormalized, p=2, dim=0)

        threshold = 0.5

        searcher = HalfspaceSearcher()
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
