"""
Cache module for managing KV cache with hierarchical indexing.

This module provides a custom HuggingFace Cache that maintains a hierarchical
index alongside the standard KV cache, enabling efficient range-based key selection.
"""

from .hira_cache import HiraCache

__all__ = [
    "HiraCache",
]
