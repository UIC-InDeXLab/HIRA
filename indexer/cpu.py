import math
import torch
from dataclasses import dataclass
from typing import Any, List, Optional
import faiss

from .base import BaseIndexer


class CPUIndexer(BaseIndexer):

    @dataclass
    class Level:
        level_idx: int
        ball_centers: torch.Tensor  # center of ball in the lower level (# key_ptrs)
        ball_radii: torch.Tensor  # radius of ball in the lower level (# key_ptrs)
        size: int

        # store parent-child relationships (THIS IS CHILD LEVEL)
        # local child idx (current level) -> local parent idx
        child2parent: Optional[torch.Tensor]

    def __init__(
        self,
        num_levels: int,
        branching_factor: int,
        max_iterations: int = 1,
        verbose: bool = False,
    ):
        super().__init__()
        self.max_iterations = max_iterations
        self.device = "cpu"
        self.num_levels = num_levels
        self.branching_factor = branching_factor
        self.verbose = verbose
        self.levels: List[Any] = []
        self.num_keys: int = 0
        self.values: Optional[torch.Tensor] = None
        self.dim: int = 0

        # ====== UPDATING ======
        self.update_count = 0

    @property
    def keys(self):
        return self.levels[0].ball_centers

    @torch.no_grad()
    def build(self, keys: torch.Tensor, values: Optional[torch.Tensor] = None):
        """
        input keys = (1, H, L, D)
        input values = (1, H, L, D)
        """

        keys = keys.to(self.device).squeeze(0).contiguous()
        self.num_heads, self.num_keys, self.dim = keys.shape
        if values is not None:
            self.values = values.to(self.device).contiguous()
        else:
            self.values = None

        level_size = self.num_keys
        level_idx = 0
        ball_centers = keys

        # first level (all the points)
        print(f"Building level {level_idx}...") if self.verbose else None

        level_0 = CPUIndexer.Level(
            level_idx=level_idx,
            ball_centers=ball_centers.contiguous(),
            ball_radii=torch.zeros((self.num_heads, self.num_keys), device=self.device),
            size=level_size,
            # child <-> parent
            child2parent=None,
        )
        self.levels.append(level_0)

        while True:
            level_size = level_size // self.branching_factor

            if level_size < 1 or level_idx + 1 >= self.num_levels:
                break

            print(f"Building level {level_idx + 1}...") if self.verbose else None

            # STEP 1: cluster
            (
                ball_centers,  # (H, K, D)
                _,  # (H, K)
                assign,  # (H, L)
            ) = self._sample_centroids_faiss(ball_centers, level_size)

            # child level: the level we are clustering to build its parent
            child_level = self.levels[level_idx]

            # STEP 2: fill child2parent mapping in child level
            child_level.child2parent = assign.contiguous()

            # STEP 3: approximate small enclosing balls
            ball_radii = self._refine_parent_ball_radii(
                child_centers=child_level.ball_centers,
                child_radii=child_level.ball_radii,
                assign=assign,
                parent_centroids=ball_centers,
            )

            # STEP 4: assign child <-> parent for previous level
            # Create new level
            parent_level = CPUIndexer.Level(
                level_idx=level_idx + 1,
                ball_centers=ball_centers.contiguous(),
                ball_radii=ball_radii.contiguous(),
                size=ball_centers.shape[1],
                # child <-> parent
                child2parent=None,  # to be filled later
            )
            self.levels.append(parent_level)

            level_idx += 1

        return self

    @torch.no_grad()
    def update(self, new_keys: torch.Tensor, new_values: Optional[torch.Tensor] = None):
        """Build an independent subtree for new keys, then concatenate per level.

        The update is equivalent to:
          1) build a standalone index over ``new_keys`` using the current hierarchy depth,
          2) append (concatenate) every level of that subtree to the existing hierarchy,
          3) shift appended child->parent indices by old parent sizes.

        dims:
            new_keys: (1, H, m, D)
            new_values: (1, H, m, D) | None
                -> They should match existing index.
        """
        assert (new_values is None) or (new_keys.shape == new_values.shape)
        assert new_keys.squeeze(0).shape[0] == self.keys.shape[0]
        assert new_keys.squeeze(0).shape[-1] == self.keys.shape[-1]

        depth = len(self.levels)
        old_level_sizes = [lvl.size for lvl in self.levels]

        # Build a temporary independent subtree with the same depth as the
        # current hierarchy. Small batches can terminate early; pad those with
        # 1-parent levels so level-wise concatenation remains valid.
        sub = CPUIndexer(
            num_levels=depth,
            branching_factor=self.branching_factor,
            max_iterations=self.max_iterations,
            verbose=False,
        ).build(new_keys)

        self._extend_levels_to_depth(sub, target_depth=depth)
        assert len(sub.levels) == depth

        for i in range(depth):
            dst = self.levels[i]
            src = sub.levels[i]

            dst.ball_centers = torch.cat(
                [dst.ball_centers, src.ball_centers], dim=1
            ).contiguous()
            dst.ball_radii = torch.cat(
                [dst.ball_radii, src.ball_radii], dim=1
            ).contiguous()
            dst.size = dst.ball_centers.shape[1]

            if i >= depth - 1:
                continue

            assert src.child2parent is not None

            shifted = src.child2parent + old_level_sizes[i + 1]
            if dst.child2parent is None:
                dst.child2parent = shifted.contiguous()
            else:
                dst.child2parent = torch.cat(
                    [dst.child2parent, shifted], dim=1
                ).contiguous()

        self.levels[0].ball_centers = self.levels[0].ball_centers.contiguous()
        self.num_keys = self.levels[0].ball_centers.shape[1]
        if new_values is not None:
            self.values = torch.cat(
                [self.values, new_values.to(self.device)], dim=-2
            ).contiguous()

        return self

    def _extend_levels_to_depth(self, idx: "CPUIndexer", target_depth: int):
        """Extend ``idx`` with singleton parent levels until ``target_depth``."""
        assert target_depth >= 1

        while len(idx.levels) < target_depth:
            child = idx.levels[-1]
            assert child.size >= 1

            parent_centers, _, assign = idx._sample_centroids_faiss(
                child.ball_centers, K=1
            )

            child.child2parent = assign.contiguous()

            parent_radii = idx._refine_parent_ball_radii(
                child_centers=child.ball_centers,
                child_radii=child.ball_radii,
                assign=assign,
                parent_centroids=parent_centers,
            )

            idx.levels.append(
                CPUIndexer.Level(
                    level_idx=len(idx.levels),
                    ball_centers=parent_centers.contiguous(),
                    ball_radii=parent_radii.contiguous(),
                    size=1,
                    child2parent=None,
                )
            )

    def _sample_centroids_faiss(self, keys: torch.Tensor, K: int, seed: int = None):
        assert keys.ndim == 3, "Expected keys shape (H, L, D)"
        H, L, D = keys.shape
        assert 1 <= K <= L, "Need 1 <= K <= L"

        device = keys.device
        centroids = torch.empty((H, K, D), device=device, dtype=keys.dtype)
        assign = torch.empty((H, L), device=device, dtype=torch.long)

        # Keep clustering cost bounded by the existing indexer setting.
        niter = max(int(self.max_iterations), 1)
        base_seed = 12345 if seed is None else int(seed)

        for h in range(H):
            x_np = keys[h].detach().to(torch.float32).contiguous().cpu().numpy()
            km_seed = base_seed + 9973 * h
            km = faiss.Kmeans(
                d=D,
                k=K,
                niter=niter,
                verbose=False,
                # gpu=self.require_faiss_gpu,
                gpu=False,
                seed=km_seed,
                min_points_per_centroid=1,
            )
            km.train(x_np)
            _, idx_np = km.index.search(x_np, 1)

            centroids_h = torch.from_numpy(km.centroids).to(
                device=device, dtype=keys.dtype
            )
            assign_h = torch.from_numpy(idx_np.reshape(-1)).to(
                device=device, dtype=torch.long
            )
            centroids[h] = centroids_h
            assign[h] = assign_h

        return centroids.contiguous(), None, assign.contiguous()

    def _refine_parent_ball_radii(
        self,
        child_centers: torch.Tensor,  # (H, C, D)
        child_radii: torch.Tensor,  # (H, C)
        assign: torch.Tensor,  # (H, C) long indices in [0, K-1], mapping each child -> parent centroid
        parent_centroids: torch.Tensor,  # (H, K, D)   centers of parent balls (centroids)
    ):
        """
        Uses triangle inequality to refine parent radii from children balls.
        """

        assert child_centers.ndim == 3, "child_centers should be (H, C, D)"
        assert parent_centroids.ndim == 3, "parent_centroids should be (H, K, D)"
        assert child_radii.ndim == 2 and assign.ndim == 2
        H, C, D = child_centers.shape
        H2, K, D2 = parent_centroids.shape
        assert H == H2 and D == D2, "Head / dim mismatch"
        assert child_radii.shape == (H, C)
        assert assign.shape == (H, C)
        assert assign.dtype == torch.long, "assign must be torch.long"

        # Gather parent centroid for each child based on assign:
        # parent_for_child: (H, C, D)
        parent_for_child = parent_centroids.gather(
            dim=1,
            index=assign.unsqueeze(-1).expand(H, C, D),
        )

        # Dist from each child center to its assigned parent centroid: (H, C)
        # Use float32 for the norm for stability if inputs are fp16/bf16
        diff = (child_centers - parent_for_child).float()
        dist = torch.sqrt((diff * diff).sum(dim=-1))

        # Upper bound contribution to parent radius from each child: (H, C)
        contrib = dist.to(child_radii.dtype) + child_radii

        # Scatter max over parent index -> parent_radii: (H, K)
        parent_radii = torch.full(
            (H, K), float("-inf"), device=self.device, dtype=contrib.dtype
        )
        parent_radii.scatter_reduce_(
            dim=1,
            index=assign,
            src=contrib,
            reduce="amax",
            include_self=True,
        )

        return parent_radii  # (H, K)
