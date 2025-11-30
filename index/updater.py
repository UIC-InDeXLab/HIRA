"""
Index update mechanisms for maintaining the hierarchical index as the KV cache grows.

The IndexUpdater is responsible for updating the hierarchical index when new keys
are added to the cache during generation. Different strategies can be used,
from simple rebuilds to incremental updates.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch

from .structure import HierarchicalIndex
from .builders import IndexBuilder


class IndexUpdater(ABC):
    """
    Abstract base class for index update strategies.

    An IndexUpdater handles how the hierarchical index is updated when new
    keys are added to the KV cache during token generation.
    """

    @abstractmethod
    def update(
        self,
        current_index: Optional[HierarchicalIndex],
        new_keys: torch.Tensor,
        all_keys: torch.Tensor,
        builder: IndexBuilder,
        **kwargs,
    ) -> HierarchicalIndex:
        """
        Update the index with new keys.

        Args:
            current_index: Existing index (None if building from scratch)
            new_keys: Newly added keys [num_new_keys, head_dim]
            all_keys: All keys including new ones [total_keys, head_dim]
            builder: IndexBuilder to use for rebuilding/updating
            **kwargs: Additional updater-specific parameters

        Returns:
            Updated HierarchicalIndex
        """
        pass

    @abstractmethod
    def should_update(
        self,
        current_index: Optional[HierarchicalIndex],
        num_new_keys: int,
        total_keys: int,
        **kwargs,
    ) -> bool:
        """
        Determine whether an update is needed.

        Args:
            current_index: Current index (None if no index exists)
            num_new_keys: Number of new keys added
            total_keys: Total number of keys
            **kwargs: Additional parameters

        Returns:
            True if update should be performed
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of this updater.

        Returns:
            Dictionary with updater configuration
        """
        pass


class RebuildUpdater(IndexUpdater):
    """
    Simple updater that rebuilds the entire index from scratch.

    This is the simplest strategy: whenever new keys are added and an update
    is triggered, the entire index is rebuilt from all keys. This ensures
    optimal clustering quality but can be expensive for large caches.

    Args:
        update_frequency: How often to rebuild ("always", "every_n", "threshold")
        update_interval: For "every_n", rebuild every N new keys
        update_threshold: For "threshold", rebuild when new_keys/total_keys > threshold
    """

    def __init__(
        self,
        update_frequency: str = "every_n",
        update_interval: int = 128,
        update_threshold: float = 0.1,
    ):
        if update_frequency not in ["always", "every_n", "threshold"]:
            raise ValueError(f"Invalid update_frequency: {update_frequency}")

        self.update_frequency = update_frequency
        self.update_interval = update_interval
        self.update_threshold = update_threshold
        self._keys_since_last_update = 0

    def update(
        self,
        current_index: Optional[HierarchicalIndex],
        new_keys: torch.Tensor,
        all_keys: torch.Tensor,
        builder: IndexBuilder,
        **kwargs,
    ) -> HierarchicalIndex:
        """
        Rebuild the index from scratch using all keys.

        Args:
            current_index: Existing index (ignored, will be rebuilt)
            new_keys: Newly added keys
            all_keys: All keys
            builder: IndexBuilder to use
            **kwargs: Passed to builder.build()

        Returns:
            Newly built HierarchicalIndex
        """
        # Extract build parameters from kwargs or use defaults
        num_levels = kwargs.get("num_levels", 3)
        branching_factor = kwargs.get("branching_factor", 32)
        device = kwargs.get("device", all_keys.device)

        # Rebuild from scratch
        new_index = builder.build(
            keys=all_keys,
            num_levels=num_levels,
            branching_factor=branching_factor,
            device=device,
        )

        self._keys_since_last_update = 0
        return new_index

    def should_update(
        self,
        current_index: Optional[HierarchicalIndex],
        num_new_keys: int,
        total_keys: int,
        **kwargs,
    ) -> bool:
        """
        Determine if rebuild is needed based on update frequency.

        Args:
            current_index: Current index
            num_new_keys: Number of new keys
            total_keys: Total keys
            **kwargs: Additional parameters

        Returns:
            True if rebuild should happen
        """
        if current_index is None:
            # No index exists, must build
            return True

        self._keys_since_last_update += num_new_keys

        if self.update_frequency == "always":
            return True
        elif self.update_frequency == "every_n":
            if self._keys_since_last_update >= self.update_interval:
                return True
        elif self.update_frequency == "threshold":
            ratio = self._keys_since_last_update / total_keys
            if ratio >= self.update_threshold:
                return True

        return False

    def get_config(self) -> Dict[str, Any]:
        """Return updater configuration."""
        return {
            "updater_type": "rebuild",
            "update_frequency": self.update_frequency,
            "update_interval": self.update_interval,
            "update_threshold": self.update_threshold,
        }


class IncrementalUpdater(IndexUpdater):
    """
    Incremental updater that updates the index without full rebuilding.

    This updater assigns new keys to existing clusters without rebuilding
    the entire hirarchy. This is faster but may lead to suboptimal clustering
    over time. A periodic rebuild can be scheduled to maintain quality.

    TODO: Implement incremental assignment strategy
    TODO: Add periodic rebuild scheduling
    TODO: Consider using online clustering algorithms

    Args:
        rebuild_every_n: Rebuild from scratch every N incremental updates
        assignment_method: How to assign new keys ("nearest_centroid", "adaptive")
    """

    def __init__(
        self,
        rebuild_every_n: int = 10,
        assignment_method: str = "nearest_centroid",
    ):
        self.rebuild_every_n = rebuild_every_n
        self.assignment_method = assignment_method
        self._update_count = 0

    def update(
        self,
        current_index: Optional[HierarchicalIndex],
        new_keys: torch.Tensor,
        all_keys: torch.Tensor,
        builder: IndexBuilder,
        **kwargs,
    ) -> HierarchicalIndex:
        """
        Incrementally update the index with new keys.

        TODO: Implement proper incremental update logic
        For now, falls back to rebuild for correctness.

        Args:
            current_index: Existing index
            new_keys: Newly added keys
            all_keys: All keys
            builder: IndexBuilder to use
            **kwargs: Additional parameters

        Returns:
            Updated HierarchicalIndex
        """
        # TODO: Implement incremental update
        # For now, fall back to rebuild
        if current_index is None or self._update_count % self.rebuild_every_n == 0:
            # Full rebuild
            num_levels = kwargs.get("num_levels", 3)
            branching_factor = kwargs.get("branching_factor", 32)
            device = kwargs.get("device", all_keys.device)

            new_index = builder.build(
                keys=all_keys,
                num_levels=num_levels,
                branching_factor=branching_factor,
                device=device,
            )
            self._update_count += 1
            return new_index
        else:
            # Incremental update: assign new keys to existing clusters
            # TODO: Implement this properly
            # For now, just return the current index unchanged
            # (This will lead to incorrect behavior - needs proper implementation)
            self._update_count += 1
            return current_index

    def should_update(
        self,
        current_index: Optional[HierarchicalIndex],
        num_new_keys: int,
        total_keys: int,
        **kwargs,
    ) -> bool:
        """
        Incremental updater updates on every call (when new keys are added).

        Args:
            current_index: Current index
            num_new_keys: Number of new keys
            total_keys: Total keys
            **kwargs: Additional parameters

        Returns:
            True if any new keys were added
        """
        if current_index is None:
            return True
        return num_new_keys > 0

    def get_config(self) -> Dict[str, Any]:
        """Return updater configuration."""
        return {
            "updater_type": "incremental",
            "rebuild_every_n": self.rebuild_every_n,
            "assignment_method": self.assignment_method,
        }
