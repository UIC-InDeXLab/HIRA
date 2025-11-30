"""
Unit tests for range searching.
"""

import pytest
import torch

from hira.index import KMeansIndexBuilder
from hira.search import HalfspaceRangeSearcher


class TestRangeSearcher:
    """Tests for range searchers."""
    
    def test_basic_range_search(self):
        """Test basic halfspace range search."""
        # Create keys and query
        num_keys = 200
        head_dim = 32
        keys = torch.randn(num_keys, head_dim)
        query = torch.randn(head_dim)
        
        # Build index
        builder = KMeansIndexBuilder(max_iterations=30)
        index = builder.build(
            keys=keys,
            num_levels=2,
            branching_factor=10,
            device=torch.device("cpu"),
        )
        
        # Perform range search
        searcher = HalfspaceRangeSearcher()
        threshold = 0.0
        results = searcher.search(
            query=query,
            threshold=threshold,
            index=index,
            keys=keys,
        )
        
        # Verify results satisfy the condition
        assert isinstance(results, torch.Tensor)
        if len(results) > 0:
            selected_keys = keys[results]
            scores = torch.matmul(selected_keys, query)
            assert torch.all(scores >= threshold - 1e-5)  # Allow small numerical error
    
    def test_high_threshold(self):
        """Test range search with high threshold (few results)."""
        keys = torch.randn(100, 32)
        query = torch.randn(32)
        
        builder = KMeansIndexBuilder()
        index = builder.build(
            keys=keys,
            num_levels=2,
            branching_factor=5,
            device=torch.device("cpu"),
        )
        
        # Very high threshold - should return few or no results
        searcher = HalfspaceRangeSearcher()
        threshold = 10.0
        results = searcher.search(
            query=query,
            threshold=threshold,
            index=index,
            keys=keys,
        )
        
        # Should return very few keys
        assert len(results) < len(keys)
    
    def test_low_threshold(self):
        """Test range search with low threshold (many results)."""
        keys = torch.randn(100, 32)
        query = torch.randn(32)
        
        builder = KMeansIndexBuilder()
        index = builder.build(
            keys=keys,
            num_levels=2,
            branching_factor=5,
            device=torch.device("cpu"),
        )
        
        # Very low threshold - should return many results
        searcher = HalfspaceRangeSearcher()
        threshold = -10.0
        results = searcher.search(
            query=query,
            threshold=threshold,
            index=index,
            keys=keys,
        )
        
        # Should return many keys
        assert len(results) > 0
    
    def test_max_candidates(self):
        """Test range search with max_candidates limit."""
        keys = torch.randn(100, 32)
        query = torch.randn(32)
        
        builder = KMeansIndexBuilder()
        index = builder.build(
            keys=keys,
            num_levels=2,
            branching_factor=5,
            device=torch.device("cpu"),
        )
        
        # Search with max_candidates
        searcher = HalfspaceRangeSearcher(max_candidates=10)
        threshold = -5.0  # Low threshold to get many candidates
        results = searcher.search(
            query=query,
            threshold=threshold,
            index=index,
            keys=keys,
        )
        
        # Should respect max_candidates limit
        assert len(results) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
