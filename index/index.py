from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, TYPE_CHECKING, Tuple
from dataclasses import dataclass
import torch
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
        pass


class KMeansIndex(Index):

    @dataclass
    class Level:
        level_idx: int
        ball_centers: torch.Tensor  # center of ball in the lower level (# key_ptrs)
        ball_radii: torch.Tensor  # radius of ball in the lower level (# key_ptrs)
        device: torch.device
        size: int

        # store parent-child relationships (THIS IS CHILD LEVEL)
        # local child idx (current level) -> local parent idx
        child2parent: Optional[torch.Tensor]
        # parent2child[p_pointer[idx]: p_pointer[idx+1]] gives child local indexes.
        parent2child: Optional[torch.Tensor]
        # idx is local parent idx
        p_pointer: Optional[torch.Tensor]  # pointers into parent2child
        num_parents: Optional[int] = None

    def __init__(self, config: "KMeansIndexConfig"):
        super().__init__()
        self.max_iterations = config.max_iterations
        self.device = config.device
        self.num_levels = config.num_levels
        self.branching_factor = config.branching_factor

    def build(self, keys: torch.Tensor):
        self.keys = keys.to(self.device)
        self.num_keys, self.dim = keys.shape

        level_size = self.num_keys
        level_idx = 0
        ball_centers = self.keys

        # first level (all the points)
        # print(f"Building level {level_idx}...")
        level_0 = KMeansIndex.Level(
            level_idx=level_idx,
            ball_centers=ball_centers.contiguous(),
            ball_radii=torch.zeros(self.num_keys, device=self.device),
            device=self.device,
            size=level_size,
            # child <-> parent
            child2parent=None,
            parent2child=None,
            p_pointer=None,
            num_parents=None,
        )
        self.levels.append(level_0)

        while True:
            level_size = level_size // self.branching_factor

            if level_size < 1 or level_idx + 1 >= self.num_levels:
                break

            # print(f"Building level {level_idx + 1}...")

            # STEP 1: cluster
            (
                ball_centers,
                ball_radii,
                assignments,  # assignments[child] = parent
            ) = self._k_means(ball_centers, level_size)

            # child level: the level we are clustering to build its parent
            child_level = self.levels[level_idx]

            # STEP 2: fill child2parent mapping in child level
            self.levels[level_idx].child2parent = assignments.contiguous()

            # STEP 3: approximate small enclosing balls
            ball_radii = self._refine_ball_radii(child_level, assignments, ball_centers)

            # STEP 4: CSR parent -> child mapping
            parent2child, p_pointer = self._parent_csr(assignments, level_size)

            # STEP 5: assign child <-> parent for previous level
            self.levels[level_idx].parent2child = parent2child.contiguous()
            self.levels[level_idx].p_pointer = p_pointer.contiguous()
            self.levels[level_idx].num_parents = level_size

            # Create new level
            parent_level = KMeansIndex.Level(
                level_idx=level_idx + 1,
                ball_centers=ball_centers.contiguous(),
                ball_radii=ball_radii.contiguous(),
                device=self.device,
                size=len(ball_centers),
                # child <-> parent
                child2parent=None,  # to be filled later
                parent2child=None,
                p_pointer=None,
                num_parents=None,
            )
            self.levels.append(parent_level)

            level_idx += 1

        # validate
        self._validate()

        return self

    def _k_means(self, points, num_centroids):
        # verify num_centroids
        if num_centroids >= points.shape[0]:
            # each point is its own cluster
            p2cluster = torch.tensor(
                [i for i in range(len(points))], device=points.device
            )
            ball_centers = points
            ball_radii = torch.zeros(len(points), device=points.device)
            return ball_centers, ball_radii, p2cluster

        points_np = points.cpu().float().numpy()
        kmeans = faiss.Kmeans(
            d=self.dim,
            k=num_centroids,
            niter=self.max_iterations,
            verbose=False,
            gpu=False,  # self.device.type == "cuda",  # TODO: gpu support
        )

        kmeans.train(points_np)
        _, assignments = kmeans.index.search(points_np, 1)

        centroids = torch.from_numpy(kmeans.centroids).to(points.device)
        # assignment[i] = j means points[i] assigned to cluster j
        assignments = torch.from_numpy(assignments.squeeze()).to(points.device)

        ball_radii = torch.zeros(num_centroids, device=points.device)

        for cluster_idx, centroid in enumerate(centroids):
            cluster_points_indexes = (assignments == cluster_idx).nonzero(
                as_tuple=True
            )[0]

            if cluster_points_indexes.numel() == 0:
                ball_radii[cluster_idx] = 0.0
                continue

            cluster_members = points[cluster_points_indexes]
            dists = torch.norm(cluster_members - centroid.unsqueeze(0), dim=1)
            ball_radii[cluster_idx] = dists.max()

        return centroids, ball_radii, assignments

    def _refine_ball_radii(self, child_level, assignments, ball_centers):
        ball_radii = torch.zeros(len(ball_centers), device=self.device)

        for parent_idx, ball_center in enumerate(ball_centers):
            # children of this parent
            children = (assignments == parent_idx).nonzero(as_tuple=True)[0]

            if children.numel() == 0:
                ball_radii[parent_idx] = 0.0
                continue

            child_centers = child_level.ball_centers[children]
            child_radii = child_level.ball_radii[children]

            # get max of ||child_center - parent_center|| + child_radius
            dists = torch.norm(child_centers - ball_center.unsqueeze(0), dim=1)
            refined_radius = torch.max(dists + child_radii)
            ball_radii[parent_idx] = refined_radius

        return ball_radii

    def _parent_csr(self, child2parent, num_parents):
        children_sorted = torch.argsort(child2parent)  # [Children]
        parents_sorted = child2parent[children_sorted]  # [Children]

        counts = torch.bincount(parents_sorted, minlength=num_parents)  # [Parents]
        rowptr = torch.empty(num_parents + 1, device=self.device, dtype=torch.long)
        rowptr[0] = 0
        rowptr[1:] = torch.cumsum(counts, dim=0)  # [Parents+1]

        return children_sorted, rowptr
