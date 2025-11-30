"""
Data structures for representing hierarchical indexes.

The HierarchicalIndex encapsulates the multi-level structure, including centroids,
assignments, and device placement information. It is designed to be agnostic to
the specific clustering/indexing algorithm used.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch


@dataclass
class IndexLevel:
    """
    Represents a single level in the hierarchical index.

    Attributes:
        level_idx: Level index (0 = coarsest, higher = finer)
        centroids: Cluster centroids at this level [num_clusters, head_dim]
        assignments: Assignment of keys to clusters [num_keys]
                    Values are cluster indices at this level
        parent_assignments: For levels > 0, maps each cluster to its parent cluster
                           [num_clusters_at_this_level]
        device: Device where this level resides (GPU or CPU)
        metadata: Additional level-specific metadata
    """

    level_idx: int
    centroids: torch.Tensor  # [num_clusters, head_dim]
    assignments: torch.Tensor  # [num_keys], cluster indices
    parent_assignments: Optional[torch.Tensor]  # [num_clusters], parent cluster IDs
    device: torch.device
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        # Validate shapes
        num_clusters = self.centroids.shape[0]
        if self.parent_assignments is not None:
            if self.parent_assignments.shape[0] != num_clusters:
                raise ValueError(
                    f"parent_assignments shape {self.parent_assignments.shape} "
                    f"doesn't match num_clusters {num_clusters}"
                )

    def to(self, device: torch.device) -> "IndexLevel":
        """
        Move this level to a different device.

        Args:
            device: Target device

        Returns:
            New IndexLevel on the target device
        """
        return IndexLevel(
            level_idx=self.level_idx,
            centroids=self.centroids.to(device),
            assignments=self.assignments.to(device),
            parent_assignments=(
                self.parent_assignments.to(device)
                if self.parent_assignments is not None
                else None
            ),
            device=device,
            metadata=self.metadata.copy(),
        )

    def num_clusters(self) -> int:
        """Return the number of clusters at this level."""
        return self.centroids.shape[0]

    def head_dim(self) -> int:
        """Return the dimensionality of centroids."""
        return self.centroids.shape[1]

    def get_cluster_members(self, cluster_id: int) -> torch.Tensor:
        """
        Get indices of keys assigned to a specific cluster.

        Args:
            cluster_id: Cluster ID

        Returns:
            Tensor of key indices belonging to this cluster
        """
        mask = self.assignments == cluster_id
        return torch.nonzero(mask, as_tuple=False).squeeze(-1)

    def memory_usage(self) -> Dict[str, float]:
        """
        Compute memory usage of this level.

        Returns:
            Dictionary with memory usage in MB
        """
        centroid_mem = (
            self.centroids.element_size() * self.centroids.nelement() / (1024**2)
        )
        assignment_mem = (
            self.assignments.element_size() * self.assignments.nelement() / (1024**2)
        )
        parent_mem = 0.0
        if self.parent_assignments is not None:
            parent_mem = (
                self.parent_assignments.element_size()
                * self.parent_assignments.nelement()
                / (1024**2)
            )

        return {
            "centroids_mb": centroid_mem,
            "assignments_mb": assignment_mem,
            "parent_assignments_mb": parent_mem,
            "total_mb": centroid_mem + assignment_mem + parent_mem,
        }


class HierarchicalIndex:
    """
    Hierarchical index over key vectors.

    This class represents a multi-level hierarchical index, where each level
    contains clusters (centroids) and assignments of keys to those clusters.
    The index is designed to support efficient range searching and can be
    stored partially on GPU and partially on CPU.

    Attributes:
        levels: List of IndexLevel objects, ordered from coarse to fine
        num_keys: Total number of keys indexed
        head_dim: Dimensionality of key vectors
        device: Primary device (can be overridden per level)
        metadata: Additional index-wide metadata (e.g., builder config)
    """

    def __init__(
        self,
        levels: List[IndexLevel],
        num_keys: int,
        head_dim: int,
        device: torch.device,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.levels = levels
        self.num_keys = num_keys
        self.head_dim = head_dim
        self.device = device
        self.metadata = metadata or {}

        # Validate consistency
        self._validate()

    def _validate(self):
        """Validate index consistency."""
        if len(self.levels) == 0:
            raise ValueError("Index must have at least one level")

        for level in self.levels:
            if level.centroids.shape[1] != self.head_dim:
                raise ValueError(
                    f"Level {level.level_idx} centroid dim {level.centroids.shape[1]} "
                    f"doesn't match head_dim {self.head_dim}"
                )
            if level.assignments.shape[0] != self.num_keys:
                raise ValueError(
                    f"Level {level.level_idx} assignments shape {level.assignments.shape[0]} "
                    f"doesn't match num_keys {self.num_keys}"
                )

    def num_levels(self) -> int:
        """Return the number of levels in the hirarchy."""
        return len(self.levels)

    def get_level(self, level_idx: int) -> IndexLevel:
        """
        Get a specific level.

        Args:
            level_idx: Level index (0 = coarsest)

        Returns:
            IndexLevel at the specified index
        """
        if level_idx < 0 or level_idx >= len(self.levels):
            raise ValueError(
                f"Invalid level index {level_idx}, index has {len(self.levels)} levels"
            )
        return self.levels[level_idx]

    def apply_memory_policy(self, policy: "MemoryTieringPolicy"):
        """
        Apply a memory tiering policy to move levels between devices.

        Args:
            policy: MemoryTieringPolicy instance specifying device placement
        """
        device_assignments = policy.get_device_assignments(self)

        for level_idx, target_device in device_assignments.items():
            if level_idx >= len(self.levels):
                continue

            current_device = self.levels[level_idx].device
            if current_device != target_device:
                self.levels[level_idx] = self.levels[level_idx].to(target_device)

    def total_memory_usage(self) -> Dict[str, float]:
        """
        Compute total memory usage across all levels.

        Returns:
            Dictionary with memory usage breakdown
        """
        total = {"total_mb": 0.0}

        for level in self.levels:
            level_mem = level.memory_usage()
            total[f"level_{level.level_idx}_mb"] = level_mem["total_mb"]
            total["total_mb"] += level_mem["total_mb"]

        return total

    def get_leaf_assignments(self) -> torch.Tensor:
        """
        Get assignments at the finest (leaf) level.

        Returns:
            Tensor of cluster assignments [num_keys]
        """
        return self.levels[-1].assignments

    def get_hirarchy_path(self, key_idx: int) -> List[int]:
        """
        Get the hierarchical path for a specific key.

        Args:
            key_idx: Index of the key

        Returns:
            List of cluster IDs at each level, from coarse to fine
        """
        path = []
        for level in self.levels:
            cluster_id = level.assignments[key_idx].item()
            path.append(cluster_id)
        return path

    def __repr__(self) -> str:
        """String representation of the index."""
        levels_str = ", ".join(
            [
                f"L{i}: {level.num_clusters()} clusters on {level.device}"
                for i, level in enumerate(self.levels)
            ]
        )
        return (
            f"HierarchicalIndex(num_keys={self.num_keys}, head_dim={self.head_dim}, "
            f"num_levels={len(self.levels)}, [{levels_str}])"
        )
