"""
HiraCache: Custom HuggingFace Cache with hierarchical indexing.

This cache extends the standard DynamicCache to maintain a hierarchical index
over key vectors, enabling efficient range-based key selection during attention.
"""

from typing import Any, Dict, List, Optional, Tuple, Type
import torch
from transformers.cache_utils import Cache

from ..index import Index, KMeansIndex
from ..index.config import (
    IndexConfig,
    KMeansIndexConfig,
)


class HiraCache(Cache):
    """
    Custom cache that maintains hierarchical indexes over key vectors.

    This cache behaves like a standard HuggingFace Cache but additionally
    maintains a separate hierarchical index per layer, enabling efficient
    range-based key selection during attention.

    Each layer gets its own independent index since different layers capture
    different semantic information and should not share indexes.

    The cache stores:
    - Key and value tensors per layer (like DynamicCache)
    - One hierarchical index per layer

    Usage:
        # Create a configuration
        from hira import KMeansIndexConfig, HiraCache

        config = KMeansIndexConfig(
            num_levels=3,
            branching_factor=16,
            max_iterations=50
        )

        # Pass the config to the cache
        # Indexes are created per-layer automatically
        cache = HiraCache(config)

    Args:
        index_config: IndexConfig instance (KMeansIndexConfig, IncrementalIndexConfig, etc.)
                     Each layer will get its own Index instance created from this config.
    """

    def __init__(self, index_config: IndexConfig):
        super().__init__()

        # Standard cache storage
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        # Store config to create indexes per layer
        self.index_config = index_config

        # Determine index class from config type
        self.index_class = self._get_index_class(index_config)

        # Per-layer indexes (created lazily)
        self.indexes: List[Optional[Index]] = []

        # Tracking
        self._seen_tokens = 0

    def __len__(self):
        """Returns the number of layers in the cache."""
        return len(self.key_cache)

    def _get_index_class(self, config: IndexConfig) -> Type[Index]:
        """Map config type to index class."""
        if isinstance(config, KMeansIndexConfig):
            return KMeansIndex
        elif isinstance(config, RandomizedClusteringConfig):
            return RandomizedClustering
        else:
            raise ValueError(f"Unknown config type: {type(config)}")

    def _get_or_create_index(self, layer_idx: int) -> Index:
        """Get existing index for layer or create a new one."""
        # Ensure list is long enough
        while len(self.indexes) <= layer_idx:
            self.indexes.append(None)

        # Create index if not exists
        if self.indexes[layer_idx] is None:
            self.indexes[layer_idx] = self.index_class(self.index_config)

        return self.indexes[layer_idx]

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

        # Update index for this specific layer
        self._update_index(layer_idx)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _update_index(self, layer_idx: int):
        """
        Update the index for a specific layer.

        Each layer has its own independent index that is built and updated
        separately. The Index class handles all decisions about building,
        updating, hierarchy levels, etc. based on its internal configuration.

        Args:
            layer_idx: Index of the layer to update
        """
        if len(self.key_cache) <= layer_idx:
            return

        keys_tensor = self.key_cache[layer_idx]
        if keys_tensor.numel() == 0:
            return

        # Get or create index for this layer
        index = self._get_or_create_index(layer_idx)

        # Flatten keys for this layer
        batch_size, num_heads, seq_len, head_dim = keys_tensor.shape
        keys_flat = keys_tensor.reshape(-1, head_dim)
        device = keys_tensor.device

        # Build or update the index for this layer
        if index.num_keys == 0:
            # First build - index uses its config to determine parameters
            index.build(
                keys=keys_flat,
                device=device,
            )
        else:
            # Update with new keys
            # TODO: track new vs old keys properly
            new_keys = keys_flat  # Simplified for now
            index.update(
                current_index=index,
                new_keys=new_keys,
                all_keys=keys_flat,
                device=device,
            )

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

    def get_index(self, layer_idx: int) -> Optional[Index]:
        """
        Get the index for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            The Index instance for this layer, or None if not created yet
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
        # Reset all per-layer indexes
        self.indexes = []
        self._seen_tokens = 0

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
            }

            # Add index info if exists for this layer
            index = self.get_index(layer_idx)
            if index is not None and index.num_keys > 0:
                info["index_info"] = {
                    "num_levels": index.num_levels(),
                    "num_keys": index.num_keys,
                    "memory_usage": index.total_memory_usage(),
                }

            return info
        else:
            # Info for entire cache
            layers_info = []
            total_indexed_keys = 0
            num_built_indexes = 0

            for i in range(len(self)):
                layer_info = {
                    "layer_idx": i,
                    "seq_length": self.get_seq_length(i),
                }

                index = self.get_index(i)
                if index is not None and index.num_keys > 0:
                    layer_info["index_info"] = {
                        "num_levels": index.num_levels(),
                        "num_keys": index.num_keys,
                        "memory_usage": index.total_memory_usage(),
                    }
                    total_indexed_keys += index.num_keys
                    num_built_indexes += 1

                layers_info.append(layer_info)

            info = {
                "num_layers": len(self),
                "total_tokens": self._seen_tokens,
                "num_built_indexes": num_built_indexes,
                "total_indexed_keys": total_indexed_keys,
                "layers": layers_info,
            }

            return info
