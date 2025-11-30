"""
HiraCache: Custom HuggingFace Cache with hierarchical indexing.

This cache extends the standard DynamicCache to maintain a hierarchical index
over key vectors, enabling efficient range-based key selection during attention.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers.cache_utils import Cache

from ..index import (
    HierarchicalIndex,
    IndexBuilder,
    KMeansIndexBuilder,
    IndexUpdater,
    RebuildUpdater,
    MemoryTieringPolicy,
    AllGPUPolicy,
)


class HiraCache(Cache):
    """
    Custom cache that maintains hierarchical indexes over key vectors.

    This cache behaves like a standard HuggingFace Cache but additionally
    maintains a hierarchical index for each layer, enabling efficient
    range-based key selection.

    The cache stores:
    - Key and value tensors (like DynamicCache)
    - Hierarchical indexes built from key vectors
    - Configuration for index building and updating

    Args:
        num_levels: Number of levels in the hirarchy
        branching_factor: Number of clusters per level
        builder: IndexBuilder instance (default: KMeansIndexBuilder)
        updater: IndexUpdater instance (default: RebuildUpdater)
        memory_policy: MemoryTieringPolicy instance (default: AllGPUPolicy)
        build_index_every_n: Build/update index every N tokens (0 = every update)
    """

    def __init__(
        self,
        num_levels: int = 3,
        branching_factor: int = 32,
        builder: Optional[IndexBuilder] = None,
        updater: Optional[IndexUpdater] = None,
        memory_policy: Optional[MemoryTieringPolicy] = None,
        build_index_every_n: int = 0,
    ):
        super().__init__()

        # Standard cache storage
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        # Hierarchical index storage (one per layer)
        self.indexes: List[Optional[HierarchicalIndex]] = []

        # Configuration
        self.num_levels = num_levels
        self.branching_factor = branching_factor
        self.builder = builder or KMeansIndexBuilder()
        self.updater = updater or RebuildUpdater(
            update_frequency="every_n", update_interval=128
        )
        self.memory_policy = memory_policy or AllGPUPolicy()
        self.build_index_every_n = build_index_every_n

        # Tracking
        self._seen_tokens = 0
        self._tokens_since_last_index_build: List[int] = []

    def __len__(self):
        """Returns the number of layers in the cache."""
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new key and value states.

        This method:
        1. Updates the standard KV cache
        2. Updates or rebuilds the hierarchical index for this layer

        Args:
            key_states: New key tensor [batch_size, num_heads, seq_len, head_dim]
            value_states: New value tensor [batch_size, num_heads, seq_len, head_dim]
            layer_idx: Index of the layer
            cache_kwargs: Additional arguments (unused)

        Returns:
            Tuple of (key_cache, value_cache) for this layer
        """
        # Track seen tokens (for layer 0 only to avoid double counting)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Ensure we have storage for this layer
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.tensor([]))
            self.value_cache.append(torch.tensor([]))
            self.indexes.append(None)
            self._tokens_since_last_index_build.append(0)

        # Update KV cache
        if self.key_cache[layer_idx].numel() == 0:
            # First update
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # Concatenate with existing cache
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        # Update hierarchical index
        num_new_keys = key_states.shape[-2]
        self._tokens_since_last_index_build[layer_idx] += num_new_keys

        # Check if we should build/update the index
        if self.build_index_every_n > 0:
            should_build = (
                self._tokens_since_last_index_build[layer_idx]
                >= self.build_index_every_n
            )
        else:
            # Use the updater's policy
            total_keys = self.key_cache[layer_idx].shape[-2]
            should_build = self.updater.should_update(
                current_index=self.indexes[layer_idx],
                num_new_keys=num_new_keys,
                total_keys=total_keys,
            )

        if should_build:
            self._update_index(layer_idx)
            self._tokens_since_last_index_build[layer_idx] = 0

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _update_index(self, layer_idx: int):
        """
        Update or build the hierarchical index for a specific layer.

        The index is built from the key vectors in the cache. We flatten
        the key tensor from [batch_size, num_heads, seq_len, head_dim] to
        [seq_len, num_heads * head_dim] or similar, depending on indexing strategy.

        TODO: Decide on indexing strategy:
        - Option 1: One index per head (num_heads separate indexes)
        - Option 2: One index for all heads (concatenate or average)
        - Option 3: One index per layer (current approach)

        For now, we use Option 3: build one index per layer, treating each
        (batch, head, seq_pos) as a separate key vector.

        Args:
            layer_idx: Layer index
        """
        keys_tensor = self.key_cache[
            layer_idx
        ]  # [batch_size, num_heads, seq_len, head_dim]

        if keys_tensor.numel() == 0:
            return

        # Flatten to [batch_size * num_heads * seq_len, head_dim]
        # This treats each position in each head as a separate key
        batch_size, num_heads, seq_len, head_dim = keys_tensor.shape
        keys_flat = keys_tensor.reshape(-1, head_dim)  # [batch*heads*seq, head_dim]

        # Extract new keys (if this is an update)
        num_new_tokens = self._tokens_since_last_index_build[layer_idx]
        if num_new_tokens > 0 and self.indexes[layer_idx] is not None:
            # Incremental update case
            new_keys_start = (seq_len - num_new_tokens) * batch_size * num_heads
            new_keys = keys_flat[new_keys_start:]
        else:
            # Full rebuild case
            new_keys = keys_flat

        # Update or build index
        self.indexes[layer_idx] = self.updater.update(
            current_index=self.indexes[layer_idx],
            new_keys=new_keys,
            all_keys=keys_flat,
            builder=self.builder,
            num_levels=self.num_levels,
            branching_factor=self.branching_factor,
            device=keys_tensor.device,
        )

        # Apply memory tiering policy
        if self.indexes[layer_idx] is not None:
            self.indexes[layer_idx].apply_memory_policy(self.memory_policy)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return (
            self.key_cache[layer_idx].shape[-2]
            if self.key_cache[layer_idx].numel() > 0
            else 0
        )

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length (None for dynamic cache)."""
        return None

    def get_index(self, layer_idx: int) -> Optional[HierarchicalIndex]:
        """
        Get the hierarchical index for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            HierarchicalIndex or None if not yet built
        """
        if layer_idx >= len(self.indexes):
            return None
        return self.indexes[layer_idx]

    def get_keys_flat(self, layer_idx: int) -> torch.Tensor:
        """
        Get flattened key vectors for a specific layer.

        This returns keys in the same format used for indexing:
        [batch_size * num_heads * seq_len, head_dim]

        Args:
            layer_idx: Layer index

        Returns:
            Flattened key tensor
        """
        if layer_idx >= len(self.key_cache):
            return torch.tensor([])

        keys_tensor = self.key_cache[layer_idx]
        if keys_tensor.numel() == 0:
            return torch.tensor([])

        # Flatten to [batch*heads*seq, head_dim]
        batch_size, num_heads, seq_len, head_dim = keys_tensor.shape
        return keys_tensor.reshape(-1, head_dim)

    def reset(self):
        """Reset the cache to empty state."""
        self.key_cache = []
        self.value_cache = []
        self.indexes = []
        self._seen_tokens = 0
        self._tokens_since_last_index_build = []

    def get_cache_info(self, layer_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Get information about the cache and indexes.

        Args:
            layer_idx: Specific layer to get info for (None = all layers)

        Returns:
            Dictionary with cache statistics
        """
        if layer_idx is not None:
            # Info for specific layer
            if layer_idx >= len(self):
                return {}

            info = {
                "layer_idx": layer_idx,
                "seq_length": self.get_seq_length(layer_idx),
                "has_index": self.indexes[layer_idx] is not None,
            }

            if self.indexes[layer_idx] is not None:
                index = self.indexes[layer_idx]
                info["index_info"] = {
                    "num_levels": index.num_levels(),
                    "num_keys": index.num_keys,
                    "memory_usage": index.total_memory_usage(),
                }

            return info
        else:
            # Info for all layers
            return {
                "num_layers": len(self),
                "total_tokens": self._seen_tokens,
                "layers": [self.get_cache_info(i) for i in range(len(self))],
            }
