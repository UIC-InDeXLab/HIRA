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
        pass  # TODO

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
        cluster_assignments: Optional[
            Dict[int, torch.Tensor]
        ]  # parent_key_ptr -> index of key inside key_ptrs
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
            key_ptrs=current_indexes,
            cluster_assignments={},  # to be updated
            key_centers=self.keys,
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

            # Create new level
            level = KMeansIndex.Level(
                level_idx=current_level_idx + 1,
                key_ptrs=cluster_reps,
                cluster_assignments={},  # to be assigned (none for highest level)
                key_centers=cluster_centers,
                key_radii=cluster_radii,
                device=self.device,
                size=len(cluster_reps),
            )
            self.levels.append(level)
            self.levels[current_level_idx].cluster_assignments = assignment_dict

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
                ptr.item(): torch.tensor([idx], device=points.device)
                for idx, ptr in enumerate(pointers)
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
                assignment_dict[cluster_rep.item()] = (
                    cluster_points_indexes  # the local index
                )
                cluster_reps.append(cluster_rep)
                
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
