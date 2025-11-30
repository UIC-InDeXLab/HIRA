"""
Unit tests for hierarchical index building and structure.
"""

import pytest
import torch

from hira.index import (
    KMeansIndexBuilder,
    HierarchicalIndex,
    RebuildUpdater,
    AllGPUPolicy,
)


class TestIndexBuilder:
    """Tests for index builders."""
    
    def test_kmeans_builder_basic(self):
        """Test basic k-means index building."""
        # Create some random keys
        num_keys = 1000
        head_dim = 64
        keys = torch.randn(num_keys, head_dim)
        
        # Build index
        builder = KMeansIndexBuilder(max_iterations=50)
        index = builder.build(
            keys=keys,
            num_levels=2,
            branching_factor=10,
            device=torch.device("cpu"),
        )
        
        # Verify structure
        assert index.num_levels() == 2
        assert index.num_keys == num_keys
        assert index.head_dim == head_dim
        
        # Check level 0
        level_0 = index.get_level(0)
        assert level_0.num_clusters() == 10
        assert level_0.centroids.shape == (10, head_dim)
        assert level_0.assignments.shape == (num_keys,)
        
        # Check level 1
        level_1 = index.get_level(1)
        assert level_1.centroids.shape[1] == head_dim
        assert level_1.assignments.shape == (num_keys,)
    
    def test_single_level_index(self):
        """Test building a single-level index."""
        keys = torch.randn(100, 32)
        builder = KMeansIndexBuilder()
        
        index = builder.build(
            keys=keys,
            num_levels=1,
            branching_factor=5,
            device=torch.device("cpu"),
        )
        
        assert index.num_levels() == 1
        assert index.get_level(0).num_clusters() == 5
    
    def test_empty_keys_error(self):
        """Test that empty keys raise an error."""
        keys = torch.tensor([]).reshape(0, 64)
        builder = KMeansIndexBuilder()
        
        with pytest.raises(ValueError):
            builder.build(
                keys=keys,
                num_levels=2,
                branching_factor=10,
                device=torch.device("cpu"),
            )
    
    def test_hirarchy_path(self):
        """Test getting hierarchical path for a key."""
        keys = torch.randn(100, 32)
        builder = KMeansIndexBuilder()
        
        index = builder.build(
            keys=keys,
            num_levels=2,
            branching_factor=5,
            device=torch.device("cpu"),
        )
        
        # Get path for first key
        path = index.get_hirarchy_path(0)
        assert len(path) == 2
        assert all(isinstance(p, int) for p in path)


class TestIndexUpdater:
    """Tests for index updaters."""
    
    def test_rebuild_updater(self):
        """Test rebuild updater."""
        keys_initial = torch.randn(100, 32)
        keys_new = torch.randn(20, 32)
        keys_all = torch.cat([keys_initial, keys_new], dim=0)
        
        builder = KMeansIndexBuilder()
        updater = RebuildUpdater(update_frequency="always")
        
        # Build initial index
        index = builder.build(
            keys=keys_initial,
            num_levels=2,
            branching_factor=5,
            device=torch.device("cpu"),
        )
        
        # Check if update is needed
        should_update = updater.should_update(
            current_index=index,
            num_new_keys=20,
            total_keys=120,
        )
        assert should_update
        
        # Update index
        new_index = updater.update(
            current_index=index,
            new_keys=keys_new,
            all_keys=keys_all,
            builder=builder,
            num_levels=2,
            branching_factor=5,
            device=torch.device("cpu"),
        )
        
        assert new_index.num_keys == 120
    
    def test_rebuild_every_n(self):
        """Test rebuild updater with every_n frequency."""
        updater = RebuildUpdater(
            update_frequency="every_n",
            update_interval=50,
        )
        
        # Should not update with 30 new keys
        assert not updater.should_update(None, 30, 100)
        
        # Should update with 50+ new keys
        assert updater.should_update(None, 50, 100)


class TestMemoryPolicy:
    """Tests for memory tiering policies."""
    
    def test_all_gpu_policy(self):
        """Test all-GPU memory policy."""
        keys = torch.randn(100, 32)
        builder = KMeansIndexBuilder()
        index = builder.build(
            keys=keys,
            num_levels=3,
            branching_factor=5,
            device=torch.device("cpu"),
        )
        
        policy = AllGPUPolicy(device="cpu")  # Use CPU for testing
        assignments = policy.get_device_assignments(index)
        
        assert len(assignments) == 3
        assert all(dev == torch.device("cpu") for dev in assignments.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
