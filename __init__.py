"""
Hira: Hierarchical Range-Searching Attention

A novel attention mechanism using hierarchical indexing for efficient key selection.
"""

__version__ = "0.1.0"

from .index import (
    Index,
    KMeansIndex,
    IndexConfig,
    KMeansIndexConfig,
)

from .search import HalfspaceSearcher

from .attention import HiraAttention, HiraAttentionProcessor, patch_model_with_hira_attention

from .cache import HiraCache

__all__ = [
    # Index components
    "Index",
    "KMeansIndex",
    "IndexConfig",
    "KMeansIndexConfig",
    
    # Search components
    "HalfspaceSearcher",
    
    # Attention components
    "HiraAttention",
    "HiraAttentionProcessor",
    "patch_model_with_hira_attention",
    
    # Cache
    "HiraCache",
]
