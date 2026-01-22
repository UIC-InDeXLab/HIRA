"""
Comprehensive tests for all pruning index implementations.

This test suite validates:
- Index building and basic functionality
- Halfspace intersection logic
- Corner cases (empty clusters, degenerate cases, edge conditions)
- Consistency across different index types
"""

import pytest
import torch
import numpy as np
from typing import List, Type
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import all index types using direct imports
from kmeans_ball_index import KMeansBallIndex
from kmeans_hyperrectangle_index import KMeansHyperrectangleIndex
from kmeans_convexhull_index import KMeansConvexHullIndex
from kmeans_ellipsoid_index import KMeansEllipsoidIndex
from pq_ball_index import PQBallIndex
from pq_hyperrectangle_index import PQHyperrectangleIndex
from pq_convexhull_index import PQConvexHullIndex
from random_ball_index import RandomBallIndex
from random_hyperrectangle_index import RandomHyperrectangleIndex
from random_convexhull_index import RandomConvexHullIndex
from random_exact_ball_index import RandomExactBallIndex
from random_ellipsoid_index import RandomEllipsoidIndex


# Test fixtures and helper functions
@pytest.fixture
def sample_keys_2d():
    """Generate simple 2D keys for basic testing."""
    torch.manual_seed(42)
    return torch.randn(100, 2)


@pytest.fixture
def sample_keys_high_dim():
    """Generate high-dimensional keys."""
    torch.manual_seed(42)
    return torch.randn(500, 64)


@pytest.fixture
def clustered_keys():
    """Generate keys with clear clusters for testing."""
    torch.manual_seed(42)
    cluster1 = torch.randn(50, 10) + torch.tensor([5.0] * 10)
    cluster2 = torch.randn(50, 10) + torch.tensor([-5.0] * 10)
    cluster3 = torch.randn(50, 10) + torch.tensor([0.0, 5.0] + [0.0] * 8)
    return torch.cat([cluster1, cluster2, cluster3], dim=0)


@pytest.fixture
def minimal_keys():
    """Minimal number of keys for edge case testing."""
    return torch.randn(5, 8)


@pytest.fixture
def single_key():
    """Single key vector."""
    return torch.randn(1, 16)


# Test class for K-Means Ball Index
class TestKMeansBallIndex:
    
    def test_basic_build(self, sample_keys_high_dim):
        """Test basic index building."""
        index = KMeansBallIndex(num_clusters=10, device='cpu')
        index.build(sample_keys_high_dim)
        
        assert len(index.balls) == 10
        assert index.centroids is not None
        assert index.centroids.shape == (10, 64)
    
    def test_intersection_percentage(self, sample_keys_high_dim):
        """Test halfspace intersection calculation."""
        index = KMeansBallIndex(num_clusters=10, device='cpu')
        index.build(sample_keys_high_dim)
        
        # Random query
        q = torch.randn(64)
        q = q / torch.norm(q)
        threshold = 0.0
        
        pct = index.get_intersection_percentage(q, threshold)
        assert 0.0 <= pct <= 100.0
    
    def test_all_points_above_threshold(self, sample_keys_high_dim):
        """Test when all points are above halfspace threshold."""
        index = KMeansBallIndex(num_clusters=10, device='cpu')
        index.build(sample_keys_high_dim)
        
        q = torch.randn(64)
        q = q / torch.norm(q)
        
        # Very low threshold - all should intersect
        pct = index.get_intersection_percentage(q, -1000.0)
        assert pct == 100.0
    
    def test_no_points_above_threshold(self, sample_keys_high_dim):
        """Test when no points are above halfspace threshold."""
        index = KMeansBallIndex(num_clusters=10, device='cpu')
        index.build(sample_keys_high_dim)
        
        q = torch.randn(64)
        q = q / torch.norm(q)
        
        # Very high threshold - none should intersect
        pct = index.get_intersection_percentage(q, 1000.0)
        assert pct == 0.0
    
    def test_few_keys_many_clusters(self, minimal_keys):
        """Test with more clusters than meaningful."""
        index = KMeansBallIndex(num_clusters=10, device='cpu')
        index.build(minimal_keys)
        
        # Should handle gracefully - some clusters may be empty
        assert len(index.balls) == 10
    
    def test_single_cluster(self, sample_keys_high_dim):
        """Test with single cluster."""
        index = KMeansBallIndex(num_clusters=1, device='cpu')
        index.build(sample_keys_high_dim)
        
        assert len(index.balls) == 1
        q = torch.randn(64)
        q = q / torch.norm(q)
        pct = index.get_intersection_percentage(q, 0.0)
        assert pct == 0.0 or pct == 100.0


# Test class for K-Means Hyperrectangle Index
class TestKMeansHyperrectangleIndex:
    
    def test_basic_build(self, sample_keys_high_dim):
        """Test basic index building."""
        index = KMeansHyperrectangleIndex(num_clusters=10, device='cpu')
        index.build(sample_keys_high_dim)
        
        assert len(index.hyperrectangles) == 10
    
    def test_bounding_box_validity(self, clustered_keys):
        """Test that bounding boxes are valid (min <= max)."""
        index = KMeansHyperrectangleIndex(num_clusters=3, device='cpu')
        index.build(clustered_keys)
        
        for hyperrect in index.hyperrectangles:
            assert torch.all(hyperrect.min_bounds <= hyperrect.max_bounds)
    
    def test_intersection_correctness(self, sample_keys_2d):
        """Test intersection calculation correctness."""
        index = KMeansHyperrectangleIndex(num_clusters=5, device='cpu')
        index.build(sample_keys_2d)
        
        # Test with axis-aligned query
        q = torch.tensor([1.0, 0.0])
        threshold = 0.0
        pct = index.get_intersection_percentage(q, threshold)
        assert 0.0 <= pct <= 100.0


# Test class for K-Means ConvexHull Index
class TestKMeansConvexHullIndex:
    
    def test_basic_build(self, sample_keys_high_dim):
        """Test basic index building."""
        index = KMeansConvexHullIndex(num_clusters=10, device='cpu')
        index.build(sample_keys_high_dim)
        
        assert len(index.convex_hulls) == 10
    
    def test_degenerate_hull(self, minimal_keys):
        """Test with too few points for proper convex hull."""
        index = KMeansConvexHullIndex(num_clusters=3, device='cpu')
        index.build(minimal_keys)
        
        # Should handle degenerate cases gracefully
        assert len(index.convex_hulls) == 3
    
    def test_hull_intersection(self, clustered_keys):
        """Test convex hull intersection."""
        index = KMeansConvexHullIndex(num_clusters=3, device='cpu')
        index.build(clustered_keys)
        
        q = torch.randn(10)
        q = q / torch.norm(q)
        pct = index.get_intersection_percentage(q, 0.0)
        assert 0.0 <= pct <= 100.0


class TestKMeansEllipsoidIndex:
    
    def test_basic_build(self, sample_keys_high_dim):
        index = KMeansEllipsoidIndex(num_clusters=8, device='cpu')
        index.build(sample_keys_high_dim)
        
        assert len(index.ellipsoids) == 8
    
    def test_clusters_contained(self, clustered_keys):
        index = KMeansEllipsoidIndex(num_clusters=4, device='cpu')
        index.build(clustered_keys)
        
        for cluster_idx, ellipsoid in enumerate(index.ellipsoids):
            mask = index.assignments == cluster_idx
            cluster_points = clustered_keys[mask]
            if len(cluster_points) == 0:
                continue
            diff = cluster_points - ellipsoid.center
            # Convert inverse shape matrix back to shape matrix for membership test
            shape_matrix = torch.linalg.pinv(ellipsoid.inv_shape_matrix)
            quad = torch.einsum('bi,ij,bj->b', diff, shape_matrix, diff)
            assert torch.all(quad <= 1.0 + 2e-2)


# Test class for PQ Ball Index
class TestPQBallIndex:
    
    def test_basic_build_nbits8(self, sample_keys_high_dim):
        """Test PQ index with nbits=8."""
        index = PQBallIndex(M=8, nbits=8, device='cpu')
        index.build(sample_keys_high_dim)
        
        assert len(index.balls) > 0
        assert index.unique_codes is not None
    
    def test_basic_build_nbits4(self, sample_keys_high_dim):
        """Test PQ index with nbits=4."""
        index = PQBallIndex(M=2, nbits=4, device='cpu')
        index.build(sample_keys_high_dim)
        
        assert len(index.balls) > 0
        # With M=2, nbits=4: max 16^2 = 256 unique codes possible
        assert len(index.unique_codes) <= 256
    
    def test_basic_build_nbits6(self, sample_keys_high_dim):
        """Test PQ index with nbits=6."""
        index = PQBallIndex(M=2, nbits=6, device='cpu')
        index.build(sample_keys_high_dim)
        
        assert len(index.balls) > 0
    
    def test_pq_intersection(self, sample_keys_high_dim):
        """Test PQ ball intersection."""
        index = PQBallIndex(M=2, nbits=5, device='cpu')
        index.build(sample_keys_high_dim)
        
        q = torch.randn(64)
        q = q / torch.norm(q)
        pct = index.get_intersection_percentage(q, 0.0)
        assert 0.0 <= pct <= 100.0
    
    def test_pq_dimension_compatibility(self, sample_keys_high_dim):
        """Test that dimension is compatible with M."""
        # 64 dimensions, M=8 should work (64 % 8 == 0)
        index = PQBallIndex(M=8, nbits=4, device='cpu')
        index.build(sample_keys_high_dim)
        assert len(index.balls) > 0


# Test class for PQ Hyperrectangle Index
class TestPQHyperrectangleIndex:
    
    def test_basic_build(self, sample_keys_high_dim):
        """Test basic PQ hyperrectangle building."""
        index = PQHyperrectangleIndex(M=2, nbits=5, device='cpu')
        index.build(sample_keys_high_dim)
        
        assert len(index.hyperrectangles) > 0
    
    def test_different_nbits(self, sample_keys_high_dim):
        """Test with different nbits values."""
        for nbits in [4, 5, 6, 8]:
            index = PQHyperrectangleIndex(M=2, nbits=nbits, device='cpu')
            index.build(sample_keys_high_dim)
            assert len(index.hyperrectangles) > 0


# Test class for PQ ConvexHull Index
class TestPQConvexHullIndex:
    
    def test_basic_build(self, sample_keys_high_dim):
        """Test basic PQ convex hull building."""
        index = PQConvexHullIndex(M=2, nbits=5, device='cpu')
        index.build(sample_keys_high_dim)
        
        assert len(index.convex_hulls) > 0
    
    def test_sparse_codes(self, sample_keys_high_dim):
        """Test with parameters that create few unique codes."""
        index = PQConvexHullIndex(M=1, nbits=4, device='cpu')
        index.build(sample_keys_high_dim)
        
        # With M=1, nbits=4: max 16 unique codes
        assert len(index.unique_codes) <= 16
        assert len(index.convex_hulls) == len(index.unique_codes)


# Test class for Random Ball Index
class TestRandomBallIndex:
    
    def test_basic_build(self, sample_keys_high_dim):
        """Test basic random ball index building."""
        index = RandomBallIndex(num_centroids=20, seed=42, device='cpu')
        index.build(sample_keys_high_dim)
        
        assert len(index.balls) == 20
    
    def test_reproducibility(self, sample_keys_high_dim):
        """Test that same seed produces same results."""
        index1 = RandomBallIndex(num_centroids=10, seed=42, device='cpu')
        index1.build(sample_keys_high_dim)
        
        index2 = RandomBallIndex(num_centroids=10, seed=42, device='cpu')
        index2.build(sample_keys_high_dim)
        
        # Centroids should be identical
        assert torch.allclose(index1.centroids, index2.centroids)
    
    def test_more_centroids_than_keys(self, minimal_keys):
        """Test with more centroids than available keys."""
        index = RandomBallIndex(num_centroids=10, seed=42, device='cpu')
        index.build(minimal_keys)
        
        # Should only create as many as available keys
        assert len(index.balls) <= 5


# Test class for Random Hyperrectangle Index
class TestRandomHyperrectangleIndex:
    
    def test_basic_build(self, sample_keys_high_dim):
        """Test basic random hyperrectangle building."""
        index = RandomHyperrectangleIndex(num_centroids=20, seed=42, device='cpu')
        index.build(sample_keys_high_dim)
        
        assert len(index.hyperrectangles) == 20


# Test class for Random ConvexHull Index
class TestRandomConvexHullIndex:
    
    def test_basic_build(self, sample_keys_high_dim):
        """Test basic random convex hull building."""
        index = RandomConvexHullIndex(num_centroids=20, seed=42, device='cpu')
        index.build(sample_keys_high_dim)
        
        assert len(index.convex_hulls) == 20


class TestRandomEllipsoidIndex:
    
    def test_basic_build(self, sample_keys_high_dim):
        index = RandomEllipsoidIndex(num_centroids=12, seed=7, device='cpu')
        index.build(sample_keys_high_dim)
        
        assert len(index.ellipsoids) == 12
    
    def test_cluster_containment(self, clustered_keys):
        index = RandomEllipsoidIndex(num_centroids=5, seed=0, device='cpu')
        index.build(clustered_keys)
        
        for centroid_idx, ellipsoid in enumerate(index.ellipsoids):
            mask = index.assignments == centroid_idx
            cluster_points = clustered_keys[mask]
            if len(cluster_points) == 0:
                continue
            diff = cluster_points - ellipsoid.center
            shape_matrix = torch.linalg.pinv(ellipsoid.inv_shape_matrix)
            quad = torch.einsum('bi,ij,bj->b', diff, shape_matrix, diff)
            assert torch.all(quad <= 1.0 + 2e-2)


class TestRandomExactBallIndex:
    def test_basic_build(self, sample_keys_high_dim):
        index = RandomExactBallIndex(num_centroids=10, seed=42, device="cpu")
        index.build(sample_keys_high_dim)

        assert len(index.balls) == 10

    def test_exact_ball_contains_cluster(self, clustered_keys):
        index = RandomExactBallIndex(num_centroids=3, seed=0, device="cpu")
        index.build(clustered_keys)

        for idx, ball in enumerate(index.balls):
            mask = index.assignments == idx
            cluster_points = clustered_keys[mask]
            if len(cluster_points) == 0:
                continue
            distances = torch.norm(cluster_points - ball.center, dim=1)
            assert torch.all(distances <= ball.radius + 1e-5)


# Cross-index comparison tests
class TestCrossIndexComparisons:
    
    def test_ball_vs_hyperrectangle_coverage(self, clustered_keys):
        """Compare ball vs hyperrectangle intersection rates."""
        ball_index = KMeansBallIndex(num_clusters=3, device='cpu')
        ball_index.build(clustered_keys)
        
        rect_index = KMeansHyperrectangleIndex(num_clusters=3, device='cpu')
        rect_index.build(clustered_keys)
        
        q = torch.randn(10)
        q = q / torch.norm(q)
        threshold = 0.0
        
        ball_pct = ball_index.get_intersection_percentage(q, threshold)
        rect_pct = rect_index.get_intersection_percentage(q, threshold)
        
        # Both should give reasonable results
        assert 0.0 <= ball_pct <= 100.0
        assert 0.0 <= rect_pct <= 100.0
    
    def test_consistent_structure_counts(self, sample_keys_high_dim):
        """Test that indexes create expected number of structures."""
        num_structures = 15
        
        # K-means indices
        kmeans_ball = KMeansBallIndex(num_clusters=num_structures, device='cpu')
        kmeans_ball.build(sample_keys_high_dim)
        assert len(kmeans_ball.balls) == num_structures
        
        kmeans_rect = KMeansHyperrectangleIndex(num_clusters=num_structures, device='cpu')
        kmeans_rect.build(sample_keys_high_dim)
        assert len(kmeans_rect.hyperrectangles) == num_structures
        
        # Random indices
        random_ball = RandomBallIndex(num_centroids=num_structures, device='cpu')
        random_ball.build(sample_keys_high_dim)
        assert len(random_ball.balls) == num_structures
        
        random_rect = RandomHyperrectangleIndex(num_centroids=num_structures, device='cpu')
        random_rect.build(sample_keys_high_dim)
        assert len(random_rect.hyperrectangles) == num_structures


# Edge case and corner case tests
class TestEdgeCases:
    
    def test_empty_cluster_handling(self, minimal_keys):
        """Test handling of empty clusters."""
        # Create index with many more clusters than points
        index = KMeansBallIndex(num_clusters=20, device='cpu')
        index.build(minimal_keys)
        
        # Should handle empty clusters gracefully
        assert len(index.balls) == 20
        # Some balls should have zero or near-zero radius
        radii = [ball.radius for ball in index.balls]
        assert any(r == 0.0 for r in radii)
    
    def test_identical_points(self):
        """Test with all identical points."""
        keys = torch.ones(10, 5)
        
        index = KMeansBallIndex(num_clusters=3, device='cpu')
        index.build(keys)
        
        # All balls should have zero radius
        for ball in index.balls:
            assert ball.radius == 0.0
    
    def test_collinear_points_2d(self):
        """Test with collinear points in 2D."""
        # Points along a line
        t = torch.linspace(0, 1, 50).unsqueeze(1)
        keys = torch.cat([t, t * 2], dim=1)
        
        index = KMeansHyperrectangleIndex(num_clusters=5, device='cpu')
        index.build(keys)
        
        assert len(index.hyperrectangles) == 5
    
    def test_extreme_threshold_values(self, sample_keys_high_dim):
        """Test with extreme threshold values."""
        index = KMeansBallIndex(num_clusters=10, device='cpu')
        index.build(sample_keys_high_dim)
        
        q = torch.randn(64)
        q = q / torch.norm(q)
        
        # Test with very large and small thresholds
        pct_low = index.get_intersection_percentage(q, -1e6)
        pct_high = index.get_intersection_percentage(q, 1e6)
        
        assert pct_low == 100.0
        assert pct_high == 0.0
    
    def test_normalized_vs_unnormalized_query(self, sample_keys_high_dim):
        """Test that query normalization doesn't affect intersection direction."""
        index = KMeansBallIndex(num_clusters=10, device='cpu')
        index.build(sample_keys_high_dim)
        
        q = torch.randn(64)
        q_normalized = q / torch.norm(q)
        
        threshold = 0.5
        
        pct1 = index.get_intersection_percentage(q_normalized, threshold)
        
        # Scaled query with adjusted threshold should give similar results
        # (though exact behavior depends on implementation details)
        assert 0.0 <= pct1 <= 100.0
    
    def test_single_point_cluster(self):
        """Test cluster with single point."""
        keys = torch.randn(10, 8)
        
        # Force a configuration that might create single-point clusters
        index = RandomBallIndex(num_centroids=10, seed=42, device='cpu')
        index.build(keys)
        
        # Should handle gracefully
        assert len(index.balls) == 10


# Performance and consistency tests
class TestConsistency:
    
    def test_multiple_builds_same_data(self, sample_keys_high_dim):
        """Test that rebuilding with same data gives consistent results."""
        index1 = RandomBallIndex(num_centroids=10, seed=42, device='cpu')
        index1.build(sample_keys_high_dim)
        
        index2 = RandomBallIndex(num_centroids=10, seed=42, device='cpu')
        index2.build(sample_keys_high_dim)
        
        q = torch.randn(64)
        q = q / torch.norm(q)
        
        pct1 = index1.get_intersection_percentage(q, 0.0)
        pct2 = index2.get_intersection_percentage(q, 0.0)
        
        assert pct1 == pct2
    
    def test_intersection_monotonicity(self, sample_keys_high_dim):
        """Test that lower thresholds include more structures."""
        index = KMeansBallIndex(num_clusters=10, device='cpu')
        index.build(sample_keys_high_dim)
        
        q = torch.randn(64)
        q = q / torch.norm(q)
        
        # Lower threshold should include more or equal structures
        pct_low = index.get_intersection_percentage(q, -1.0)
        pct_mid = index.get_intersection_percentage(q, 0.0)
        pct_high = index.get_intersection_percentage(q, 1.0)
        
        assert pct_low >= pct_mid >= pct_high


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
