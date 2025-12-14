from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
import numpy as np
import faiss
from .config import KMeansIndexConfig


class Index(ABC):
    def __init__(self):
        self.levels: List[Any] = []
        self.num_keys: int = 0
        self.keys: Optional[torch.Tensor] = None
        self.dim: int = 0
        self.device: Optional[torch.device] = None
        self.metadata: Dict[str, Any] = {}

    def _validate(self):
        pass  # TODO

    def total_memory_usage(self) -> Dict[str, float]:
        """Calculate total memory usage of the index."""
        total_bytes = 0
        breakdown = {}

        # Keys
        if self.keys is not None:
            keys_bytes = self.keys.element_size() * self.keys.numel()
            breakdown["keys"] = keys_bytes
            total_bytes += keys_bytes

        # Per-level data structures
        for i, level in enumerate(self.levels):
            level_bytes = 0

            # key_centers
            if hasattr(level, "key_centers") and level.key_centers is not None:
                level_bytes += (
                    level.key_centers.element_size() * level.key_centers.numel()
                )

            # key_radii
            if hasattr(level, "key_radii") and level.key_radii is not None:
                level_bytes += level.key_radii.element_size() * level.key_radii.numel()

            # key_ptrs
            if hasattr(level, "key_ptrs") and level.key_ptrs is not None:
                level_bytes += level.key_ptrs.element_size() * level.key_ptrs.numel()

            # cluster_assignments (dict overhead is approximate)
            if hasattr(level, "cluster_assignments") and level.cluster_assignments:
                # Approximate: dict overhead + tensor data
                dict_overhead = len(level.cluster_assignments) * 100  # rough estimate
                level_bytes += dict_overhead
                for tensor in level.cluster_assignments.values():
                    if isinstance(tensor, torch.Tensor):
                        level_bytes += tensor.element_size() * tensor.numel()

            breakdown[f"level_{i}"] = level_bytes
            total_bytes += level_bytes

        breakdown["total"] = total_bytes
        breakdown["total_mb"] = total_bytes / (1024 * 1024)

        return breakdown

    def __repr__(self) -> str:
        """String representation of the index."""
        levels_str = ", ".join(
            [
                f"L{i}: {level.size} clusters on {level.device}"
                for i, level in enumerate(self.levels)
            ]
        )
        return (
            f"{self.__class__.__name__}(num_keys={self.num_keys}, head_dim={self.dim}, "
            f"num_levels={len(self.levels)}, [{levels_str}])"
        )

    @abstractmethod
    def build(
        self,
        keys: torch.Tensor,
        num_levels: int,
        branching_factor: int,
        device: torch.device,
        **kwargs,
    ) -> "Index":
        """
        Build a hierarchical index from key vectors.

        Args:
            keys: Key vectors of shape [num_keys, head_dim]
            num_levels: Number of hierarchy levels to create
            branching_factor: Number of clusters per level
            device: Device to build the index on
            **kwargs: Additional index-specific parameters

        Returns:
            Index: Self, with populated hierarchical structure
        """
        pass


class KMeansIndex(Index):

    @dataclass
    class Level:
        level_idx: int
        key_ptrs: torch.Tensor  # index in the original list (cluster representatives)
        child2parent: Optional[
            torch.Tensor
        ]  # child2parent[i]: local index of parent to local key index i
        key_centers: torch.Tensor  # center of ball in the lower level (# key_ptrs)
        key_radii: torch.Tensor  # radius of ball in the lower level (# key_ptrs)
        device: torch.device
        size: int

    def __init__(self, config: "KMeansIndexConfig"):
        super().__init__()
        self.max_iterations = config.max_iterations
        self.device = (
            config.device == "cuda"
            and torch.cuda.is_available()
            and torch.device("cuda")
            or torch.device("cpu")
        )
        self.num_levels = config.num_levels
        self.branching_factor = config.branching_factor

    def build(self, keys: torch.Tensor):
        self.keys = keys.to(self.device)
        self.num_keys, self.dim = keys.shape

        current_size = self.num_keys
        current_indexes = torch.arange(self.num_keys, device=self.device)
        current_level_idx = 0

        # first level (all the points)
        level_0 = KMeansIndex.Level(
            level_idx=current_level_idx,
            key_ptrs=current_indexes.contiguous(),
            child2parent=None,
            key_centers=self.keys.contiguous(),
            key_radii=torch.zeros(self.num_keys, device=self.device),
            device=self.device,
            size=current_size,
        )
        self.levels.append(level_0)

        while True:
            # K Means Clustering
            num_clusters = current_size // self.branching_factor

            if num_clusters < 1 or current_level_idx + 1 >= self.num_levels:
                break

            pointers = current_indexes
            points = self.keys[pointers]

            (
                cluster_reps,
                assignment_dict,
                cluster_centers,
                cluster_radii,
            ) = self._k_means(points, pointers, num_clusters)

            # BUG FIX: 100% RECALL
            # child level: the level we are clustering to build its parents
            child_level = self.levels[current_level_idx]

            # Recompute radii so they bound *all leaves* in each parent's subtree
            # using triangle inequality: R_parent = max_c (R_child[c] + ||mu_child[c] - mu_parent||)
            if child_level.key_radii is not None and child_level.key_radii.numel() > 0:
                new_radii = torch.empty_like(cluster_radii)

                child_centers = child_level.key_centers  # [num_children, dim]
                child_radii = child_level.key_radii  # [num_children]

                for parent_idx, child_local_idx in assignment_dict.items():
                    # child_local_idx: indices of child clusters for this parent
                    # NOTE: these indices are consistent with child_level.key_centers order
                    parent_center = cluster_centers[parent_idx].unsqueeze(0)  # [1, dim]

                    dists = torch.norm(
                        child_centers[child_local_idx] - parent_center, dim=1
                    )
                    # radius up to leaves under this parent
                    new_radii[parent_idx] = torch.max(
                        child_radii[child_local_idx] + dists
                    )

                cluster_radii = new_radii  # overwrite with subtree-aware radii

            # Create new level
            parent_level = KMeansIndex.Level(
                level_idx=current_level_idx + 1,
                key_ptrs=cluster_reps.contiguous(),
                # cluster_assignments={},  # to be assigned (none for highest level)
                child2parent=None,
                key_centers=cluster_centers.contiguous(),
                key_radii=cluster_radii.contiguous(),
                device=self.device,
                size=len(cluster_reps),
            )
            self.levels.append(parent_level)

            # OPTIMIZED
            self.levels[current_level_idx].child2parent = (
                self._flatten_child_parent_dict(assignment_dict, len(current_indexes))
            )

            current_level_idx += 1
            current_size = len(cluster_reps)
            current_indexes = cluster_reps

        # validate
        self._validate()

        return self

    def _k_means(
        self, points, pointers: torch.Tensor, num_clusters
    ):  # returns idx of centroids and assignments
        # pointers means the indices of points in the original list
        # verify num_clusters
        if num_clusters >= points.shape[0]:
            # Handle edge case: each point is its own cluster
            cluster_reps = pointers
            assignment_dict = {
                idx: torch.tensor([idx], device=points.device)
                for idx in range(len(pointers))
            }
            cluster_centers = points
            cluster_radii = torch.zeros(len(pointers), device=points.device)
            return cluster_reps, assignment_dict, cluster_centers, cluster_radii

        points_np = points.cpu().float().numpy()
        kmeans = faiss.Kmeans(
            d=self.dim,
            k=num_clusters,
            niter=self.max_iterations,
            verbose=False,
            gpu=False,  # self.device.type == "cuda",  # TODO: gpu support
        )

        kmeans.train(points_np)
        _, assignments = kmeans.index.search(points_np, 1)

        centroids = torch.from_numpy(kmeans.centroids).to(points.device)
        assignments = torch.from_numpy(assignments.squeeze()).to(points.device)

        cluster_reps = []  # cluster representative points
        assignment_dict = {}
        cluster_centers = []  # mean of points in each cluster
        cluster_radii = []  # max distance from center
        counter = 0

        for cluster_idx, centroid in enumerate(centroids):
            cluster_points_indexes = (assignments == cluster_idx).nonzero(
                as_tuple=True
            )[0]
            cluster_members = pointers[cluster_points_indexes]  # original indices

            if len(cluster_points_indexes) > 0:
                cluster_centers.append(centroid)
                dists = torch.norm(
                    points[cluster_points_indexes] - centroid.unsqueeze(0), dim=1
                )

                # closest point to centroid as representative
                cluster_rep = cluster_members[torch.argmin(dists)]
                cluster_reps.append(cluster_rep)
                assignment_dict[counter] = cluster_points_indexes  # the local index
                counter += 1

                max_dist = torch.max(dists)
                cluster_radii.append(max_dist)
            else:
                # ignore empty clusters
                pass

        # Convert to tensors
        cluster_centers = (
            torch.stack(cluster_centers)
            if cluster_centers
            else torch.empty(0, self.dim, device=points.device)
        )
        cluster_radii = (
            torch.stack(cluster_radii)
            if cluster_radii
            else torch.empty(0, device=points.device)
        )
        cluster_reps = (
            torch.stack(cluster_reps)
            if cluster_reps
            else torch.empty(0, dtype=torch.long, device=points.device)
        )

        return cluster_reps, assignment_dict, cluster_centers, cluster_radii

    def _flatten_child_parent_dict(
        self, assignment: Dict[int, torch.Tensor], num_children: int
    ) -> torch.Tensor:
        child2parent = -torch.ones(num_children, dtype=torch.long, device=self.device)
        for parent, child in assignment.items():
            child2parent[child] = parent
        return child2parent.contiguous()
