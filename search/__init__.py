"""
Search module for range searching over hierarchical indexes.

This module provides efficient range search capabilities, particularly
halfspace range searching for finding keys with high query-key dot products.
"""

from .range_searcher import HalfspaceSearcher

__all__ = [
    "HalfspaceSearcher",
]
