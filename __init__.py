"""
Hira: Hierarchical Range-Searching Attention

A novel attention mechanism using hierarchical indexing for efficient key selection.
"""

__version__ = "0.1.0"

from .index import (
    IndexBuilder,
    HierarchicalIndex,
    IndexUpdater,
    MemoryTieringPolicy,
    KMeansIndexBuilder,
)

from .search import RangeSearcher, HalfspaceRangeSearcher

from .attention import HiraAttention, HiraAttentionProcessor, patch_model_with_hira_attention

from .cache import HiraCache

__all__ = [
    # Index components
    "IndexBuilder",
    "HierarchicalIndex",
    "IndexUpdater",
    "MemoryTieringPolicy",
    "KMeansIndexBuilder",
    
    # Search components
    "RangeSearcher",
    "HalfspaceRangeSearcher",
    
    # Attention components
    "HiraAttention",
    "HiraAttentionProcessor",
    "patch_model_with_hira_attention",
    
    # Cache
    "HiraCache",
]
