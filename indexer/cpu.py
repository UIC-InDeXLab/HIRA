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
        num_parents: Optional[int] = None
        parent_mask_buf: Optional[torch.Tensor] = None

    def __init__(
        self,
        num_levels: int,
        branching_factor: int,
        max_iterations: int = 1,
        verbose: bool = False,
        centroid_refine_iters: int = 0,  # 0 keeps random-centroid behavior
    ):
        super().__init__()
        self.max_iterations = max_iterations
        self.device = "cpu"
        self.num_levels = num_levels
        self.branching_factor = branching_factor
        self.verbose = verbose
        self.levels: List[Any] = []
        self.num_keys: int = 0
        self.keys: Optional[torch.Tensor] = None
        self.dim: int = 0

        # ====== UPDATE ======
        # Cache for fast nearest-parent assignment during incremental updates.
        # Valid as long as level-1 parent centers stay unchanged.
        # self._faiss_l1_index: Optional[Any] = None
        self.update_count = 0
        self.centroid_refine_iters = int(centroid_refine_iters)

    @torch.no_grad()
    def build(self, keys: torch.Tensor):
        # keys = (1, H, L, D)

        self.keys = keys.to(self.device).squeeze(0)
        # self.keys = keys.to(self.device)

        self.num_heads, self.num_keys, self.dim = self.keys.shape
        # self.num_keys, self.dim = keys.shape

        level_size = self.num_keys
        level_idx = 0
        ball_centers = self.keys

        # first level (all the points)
        print(f"Building level {level_idx}...") if self.verbose else None

        level_0 = CPUIndexer.Level(
            level_idx=level_idx,
            ball_centers=ball_centers.contiguous(),
            ball_radii=torch.zeros((self.num_heads, self.num_keys), device=self.device),
            # ball_radii=torch.zeros(self.num_keys, device=self.device),
            size=level_size,
            # child <-> parent
            child2parent=None,
            num_parents=None,
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

            if self.centroid_refine_iters > 0:
                ball_centers, assign = self._refine_centroids_lloyd(
                    points=child_level.ball_centers,
                    centroids=ball_centers,
                    assign=assign,
                    num_iters=self.centroid_refine_iters,
                )

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
            child_level.num_parents = level_size

            # Create new level
            parent_level = CPUIndexer.Level(
                level_idx=level_idx + 1,
                ball_centers=ball_centers.contiguous(),
                ball_radii=ball_radii.contiguous(),
                size=ball_centers.shape[1],
                # child <-> parent
                child2parent=None,  # to be filled later
                num_parents=None,
            )
            self.levels.append(parent_level)

            level_idx += 1

        # add buffers
        # if len(self.levels) >= 2:
        # self._build_l1_faiss(self.levels[1])

        self.preallocate_search_buffers()

        return self

    # @torch.no_grad()
    # def update_v2(
    #     self,
    #     new_keys: torch.Tensor,
    #     *,
    #     use_faiss_kernel: Optional[bool] = None,
    #     top_levels: Optional[torch.Tensor] = None,
    # ):
    #     """
    #     Old, not used
    #     Skip-list style hierarchical update with per-key sampled top levels.

    #     For each new key j with sampled top level L_j:
    #       - levels 0..L_j include a representation of key j,
    #       - for i < L_j, child(i,j) -> parent(i+1,j) (self chain),
    #       - at i == L_j, parent is nearest centroid in level i+1.

    #     The nearest-parent search backend is switchable:
    #       - torch (default): fully batched and vectorized
    #       - FAISS: per-head C++ backend, useful for benchmarking speedups
    #     """
    #     if new_keys.numel() == 0:
    #         return self
    #     if new_keys.ndim != 4:
    #         raise ValueError(f"new_keys must be (1,H,m,D), got {tuple(new_keys.shape)}")
    #     _, H, m, D = new_keys.shape
    #     if D != self.dim or H != self.num_heads:
    #         raise ValueError(
    #             f"Expected (H={self.num_heads}, m, D={self.dim}), got {tuple(new_keys.shape)}"
    #         )
    #     if not self.levels:
    #         raise RuntimeError("update_v2() called before build()")

    #     use_faiss = (
    #         self.update_v2_use_faiss_kernel
    #         if use_faiss_kernel is None
    #         else bool(use_faiss_kernel)
    #     )
    #     new_keys_cpu = new_keys.squeeze(0).contiguous()  # (H,m,D)
    #     old_level_sizes = [lvl.size for lvl in self.levels]

    #     # 1) Append raw keys to level-0 storage.
    #     base_level = self.levels[0]
    #     self.keys = torch.cat([self.keys, new_keys_cpu], dim=1).contiguous()
    #     self.num_keys = self.keys.shape[1]
    #     base_level.ball_centers = self.keys
    #     base_level.size = self.num_keys
    #     base_level.ball_radii = torch.zeros(
    #         (H, self.num_keys), device=self.device, dtype=new_keys_cpu.dtype
    #     )

    #     # Single-level hierarchy has no parent links.
    #     if len(self.levels) < 2:
    #         self.update_count += m
    #         return self

    #     # Highest child level with an existing parent above it.
    #     max_top_level = len(self.levels) - 2
    #     sampled_top = self._sample_update_v2_top_levels(
    #         m=m,
    #         max_top_level=max_top_level,
    #         top_levels=top_levels,
    #     )  # (m,)

    #     # keys_by_level[i]: key ids that exist as children in level i for this update.
    #     keys_by_level: List[torch.Tensor] = [
    #         torch.arange(m, device=self.device, dtype=torch.long)
    #     ]
    #     for i in range(1, max_top_level + 1):
    #         keys_by_level.append((sampled_top >= i).nonzero(as_tuple=True)[0])

    #     # Append newly-created nodes to intermediate levels (1..max_top_level).
    #     # Track per-level map key_id -> appended position.
    #     new_node_pos: dict[int, torch.Tensor] = {}
    #     for i in range(1, max_top_level + 1):
    #         key_idx = keys_by_level[i]
    #         n_i = int(key_idx.numel())
    #         if n_i == 0:
    #             continue

    #         lvl = self.levels[i]
    #         new_centers_i = new_keys_cpu.index_select(1, key_idx)  # (H,n_i,D)
    #         new_radii_i = torch.zeros(
    #             (H, n_i), device=self.device, dtype=lvl.ball_radii.dtype
    #         )
    #         lvl.ball_centers = torch.cat(
    #             [lvl.ball_centers, new_centers_i], dim=1
    #         ).contiguous()
    #         lvl.ball_radii = torch.cat(
    #             [lvl.ball_radii, new_radii_i], dim=1
    #         ).contiguous()
    #         lvl.size = lvl.ball_centers.shape[1]

    #         pos_map_i = torch.full((m,), -1, device=self.device, dtype=torch.long)
    #         pos_map_i[key_idx] = torch.arange(n_i, device=self.device, dtype=torch.long)
    #         new_node_pos[i] = pos_map_i

    #     # Update child->parent links and parent radii on each edge i -> i+1.
    #     for i in range(0, max_top_level + 1):
    #         child_level = self.levels[i]
    #         parent_level = self.levels[i + 1]
    #         child_key_idx = keys_by_level[i]
    #         n_child_new = int(child_key_idx.numel())
    #         if n_child_new == 0:
    #             continue

    #         child_centers_new = new_keys_cpu.index_select(
    #             1, child_key_idx
    #         )  # (H,n_child_new,D)
    #         child_top = sampled_top.index_select(0, child_key_idx)  # (n_child_new,)

    #         # For keys that continue to higher levels, chain to their own parent node.
    #         # For keys whose top is this level, connect to nearest pre-update centroid above.
    #         self_chain_mask = (
    #             child_top > i
    #             if i < max_top_level
    #             else torch.zeros((n_child_new,), device=self.device, dtype=torch.bool)
    #         )
    #         nearest_mask = ~self_chain_mask

    #         assign_new = torch.empty(
    #             (H, n_child_new), device=self.device, dtype=torch.long
    #         )

    #         if nearest_mask.any():
    #             nearest_key_idx = child_key_idx[nearest_mask]
    #             q = new_keys_cpu.index_select(1, nearest_key_idx)  # (H,n_nearest,D)

    #             # Restrict nearest search to pre-existing centroids at level i+1.
    #             old_parent_size = old_level_sizes[i + 1]
    #             parent_centers_old = parent_level.ball_centers[
    #                 :, :old_parent_size, :
    #             ].contiguous()
    #             nearest_parent_idx, _ = self._nearest_l2_batched(
    #                 points=q,
    #                 centers=parent_centers_old,
    #                 use_faiss=use_faiss,
    #             )  # (H,n_nearest)
    #             assign_new[:, nearest_mask] = nearest_parent_idx

    #         if self_chain_mask.any():
    #             key_self = child_key_idx[self_chain_mask]
    #             if (i + 1) not in new_node_pos:
    #                 raise RuntimeError(
    #                     f"Missing new-node map for level {i+1} while building self-chain"
    #                 )
    #             pos_next = new_node_pos[i + 1].index_select(0, key_self)
    #             if (pos_next < 0).any():
    #                 raise RuntimeError("Invalid self-chain mapping in update_v2")
    #             assign_self = old_level_sizes[i + 1] + pos_next  # (k,)
    #             assign_new[:, self_chain_mask] = assign_self.unsqueeze(0).expand(H, -1)

    #         # Append child->parent links for this child level.
    #         if child_level.child2parent is None:
    #             child_level.child2parent = assign_new.contiguous()
    #         else:
    #             child_level.child2parent = torch.cat(
    #                 [child_level.child2parent, assign_new], dim=1
    #             ).contiguous()

    #         # Incremental parent-radii update:
    #         # r_parent <- max(r_parent, ||child-parent|| + r_child)
    #         parent_for_child = parent_level.ball_centers.gather(
    #             1,
    #             assign_new.unsqueeze(-1).expand(H, n_child_new, D),
    #         )
    #         dist = torch.linalg.norm(
    #             (child_centers_new - parent_for_child).float(), dim=-1
    #         )  # (H,n_child_new)

    #         child_r_new = child_level.ball_radii[:, -n_child_new:].float()
    #         contrib = dist + child_r_new  # (H,n_child_new)

    #         P = parent_level.size
    #         upd = torch.full(
    #             (H, P), float("-inf"), device=self.device, dtype=contrib.dtype
    #         )
    #         upd.scatter_reduce_(
    #             dim=1,
    #             index=assign_new,
    #             src=contrib,
    #             reduce="amax",
    #             include_self=True,
    #         )
    #         upd = torch.where(torch.isfinite(upd), upd, torch.zeros_like(upd))
    #         parent_level.ball_radii = torch.maximum(
    #             parent_level.ball_radii,
    #             upd.to(dtype=parent_level.ball_radii.dtype),
    #         )

    #     # Keep level metadata consistent after size changes.
    #     for i in range(len(self.levels) - 1):
    #         self.levels[i].num_parents = self.levels[i + 1].size
    #     self.preallocate_search_buffers()

    #     # Rebuild level-1 FAISS cache if level-1 centers changed.
    #     if len(self.levels) >= 2 and self.levels[1].size != old_level_sizes[1]:
    #         self._build_l1_faiss(self.levels[1])

    #     self.update_count += m
    #     if self.balance_every > 0 and self.update_count >= self.balance_every:
    #         # self._balance()
    #         self.update_count = 0

    #     return self

    @torch.no_grad()
    def update(self, new_keys: torch.Tensor):
        """Build an independent subtree for new keys, then concatenate per level.

        The update is equivalent to:
          1) build a standalone index over ``new_keys`` using the current hierarchy depth,
          2) append (concatenate) every level of that subtree to the existing hierarchy,
          3) shift appended child->parent indices by old parent sizes.
        """
        if new_keys.numel() == 0:
            return self
        if new_keys.ndim != 4:
            raise ValueError(f"new_keys must be (1,H,m,D), got {tuple(new_keys.shape)}")
        _, H, m, D = new_keys.shape
        if D != self.dim or H != self.num_heads:
            raise ValueError(
                f"Expected (H={self.num_heads}, m, D={self.dim}), got {tuple(new_keys.shape)}"
            )
        if not self.levels:
            raise RuntimeError("update_v3() called before build()")

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
            centroid_refine_iters=self.centroid_refine_iters,
        ).build(new_keys)

        self._extend_levels_to_depth(sub, target_depth=depth)
        if len(sub.levels) != depth:
            raise RuntimeError(
                f"Temporary subtree depth mismatch: got {len(sub.levels)}, expected {depth}"
            )

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

            if src.child2parent is None:
                raise RuntimeError(f"Missing child2parent in temporary level {i}")

            shifted = src.child2parent + old_level_sizes[i + 1]
            if dst.child2parent is None:
                dst.child2parent = shifted.contiguous()
            else:
                dst.child2parent = torch.cat(
                    [dst.child2parent, shifted], dim=1
                ).contiguous()

        self.keys = self.levels[0].ball_centers.contiguous()
        self.levels[0].ball_centers = self.keys
        self.num_keys = self.keys.shape[1]

        for i in range(depth - 1):
            self.levels[i].num_parents = self.levels[i + 1].size

        self.preallocate_search_buffers()
        # if len(self.levels) >= 2:
        # self._build_l1_faiss(self.levels[1])

        return self

    def _extend_levels_to_depth(self, idx: "CPUIndexer", target_depth: int):
        """Extend ``idx`` with singleton parent levels until ``target_depth``."""
        if target_depth < 1:
            raise ValueError("target_depth must be >= 1")

        while len(idx.levels) < target_depth:
            child = idx.levels[-1]
            if child.size < 1:
                raise RuntimeError("Cannot extend an empty level")

            parent_centers, _, assign = idx._sample_centroids_faiss(
                child.ball_centers, K=1
            )
            if idx.centroid_refine_iters > 0:
                parent_centers, assign = idx._refine_centroids_lloyd(
                    points=child.ball_centers,
                    centroids=parent_centers,
                    assign=assign,
                    num_iters=idx.centroid_refine_iters,
                )

            child.child2parent = assign.contiguous()
            child.num_parents = 1

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
                    num_parents=None,
                )
            )

    # def _sample_update_v2_top_levels(
    #     self,
    #     m: int,
    #     max_top_level: int,
    #     top_levels: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     """Sample/update per-key highest levels with P(top >= i) = bf^{-i}."""
    #     if m <= 0:
    #         return torch.empty((0,), device=self.device, dtype=torch.long)
    #     if max_top_level < 0:
    #         raise ValueError("max_top_level must be >= 0 for hierarchical updates")

    #     if top_levels is not None:
    #         tl = top_levels.to(device=self.device, dtype=torch.long).contiguous()
    #         if tl.ndim != 1 or tl.shape[0] != m:
    #             raise ValueError(f"top_levels must be (m,), got {tuple(tl.shape)}")
    #         if tl.numel() > 0 and (tl.min() < 0 or tl.max() > max_top_level):
    #             raise ValueError(
    #                 f"top_levels must be in [0, {max_top_level}], got min={int(tl.min())}, max={int(tl.max())}"
    #             )
    #         return tl

    #     bf = float(self.branching_factor)
    #     if bf <= 1.0:
    #         # Degenerate case: keep all at max available top level.
    #         return torch.full(
    #             (m,),
    #             fill_value=max_top_level,
    #             device=self.device,
    #             dtype=torch.long,
    #         )

    #     # If E ~ Exp(rate=ln(bf)), then floor(E) satisfies:
    #     # P(floor(E) >= i) = exp(-ln(bf)*i) = bf^{-i}.
    #     rate = math.log(bf)
    #     exp_sample = torch.empty(
    #         (m,), device=self.device, dtype=torch.float32
    #     ).exponential_(rate)
    #     return exp_sample.floor().to(torch.long).clamp_(0, max_top_level)

    # def _nearest_l2_batched(
    #     self,
    #     points: torch.Tensor,  # (H,M,D)
    #     centers: torch.Tensor,  # (H,P,D)
    #     *,
    #     use_faiss: bool,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Return nearest parent indices and distances for each (H,M) point."""
    #     if use_faiss:
    #         return self._nearest_l2_batched_faiss(points, centers)
    #     return self._nearest_l2_batched_torch(points, centers)

    # def _nearest_l2_batched_torch(
    #     self, points: torch.Tensor, centers: torch.Tensor
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     if points.ndim != 3 or centers.ndim != 3:
    #         raise ValueError("points and centers must be rank-3 tensors")
    #     H, M, D = points.shape
    #     H2, P, D2 = centers.shape
    #     if H != H2 or D != D2:
    #         raise ValueError(
    #             f"Head/dim mismatch: points={tuple(points.shape)} centers={tuple(centers.shape)}"
    #         )
    #     if P < 1:
    #         raise ValueError("centers must have at least one item on dim=1")

    #     p = points.float()
    #     c = centers.float()
    #     p2 = (p * p).sum(dim=-1, keepdim=True)  # (H,M,1)
    #     c2 = (c * c).sum(dim=-1).unsqueeze(1)  # (H,1,P)
    #     d2 = p2 + c2 - 2.0 * torch.bmm(p, c.transpose(1, 2))  # (H,M,P)
    #     idx = d2.argmin(dim=-1)  # (H,M)
    #     d2_min = d2.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1).clamp_min_(0.0)
    #     return idx.to(torch.long), d2_min.sqrt_()

    # def _nearest_l2_batched_faiss(
    #     self, points: torch.Tensor, centers: torch.Tensor
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     if points.ndim != 3 or centers.ndim != 3:
    #         raise ValueError("points and centers must be rank-3 tensors")
    #     H, M, D = points.shape
    #     H2, P, D2 = centers.shape
    #     if H != H2 or D != D2:
    #         raise ValueError(
    #             f"Head/dim mismatch: points={tuple(points.shape)} centers={tuple(centers.shape)}"
    #         )
    #     if P < 1:
    #         raise ValueError("centers must have at least one item on dim=1")

    #     idx_all = []
    #     dist_all = []
    #     for h in range(H):
    #         c_np = centers[h].detach().cpu().float().contiguous().numpy()
    #         x_np = points[h].detach().cpu().float().contiguous().numpy()

    #         index = faiss.IndexFlatL2(D)
    #         index.add(c_np)
    #         d2_np, i_np = index.search(x_np, 1)

    #         idx_h = torch.from_numpy(i_np.reshape(-1)).to(self.device, dtype=torch.long)
    #         dist_h = (
    #             torch.from_numpy(d2_np.reshape(-1))
    #             .to(self.device, dtype=torch.float32)
    #             .clamp_min_(0.0)
    #             .sqrt_()
    #         )
    #         idx_all.append(idx_h)
    #         dist_all.append(dist_h)

    #     return torch.stack(idx_all, dim=0), torch.stack(dist_all, dim=0)

    # def _balance(self):
    #     """Torch-only rebalance for per-head keys (H,N,D), with random 2-way split.

    #     Splits each oversized level-1 parent at most once per call (per head).
    #     """
    #     if len(self.levels) < 2:
    #         return

    #     level0 = self.levels[0]
    #     level1 = self.levels[1]
    #     bf = int(self.branching_factor)

    #     X0 = level0.ball_centers  # (H,N,D)
    #     c2p0 = level0.child2parent  # (H,N)
    #     C1 = level1.ball_centers  # (H,P,D)
    #     R1 = level1.ball_radii  # (H,P)

    #     if X0.ndim != 3 or C1.ndim != 3:
    #         raise RuntimeError(
    #             "Expected level0 centers (H,N,D) and level1 centers (H,P,D)."
    #         )

    #     H, N, D = X0.shape
    #     H1, P, D1 = C1.shape
    #     if H1 != H or D1 != D or D != self.dim:
    #         raise RuntimeError("Head/dim mismatch between levels.")

    #     # Optional level-2 support (Torch-only nearest-grandparent)
    #     has_l2 = len(self.levels) >= 3
    #     if has_l2:
    #         level2 = self.levels[2]
    #         C2 = level2.ball_centers  # (H,G,D)
    #         R2 = level2.ball_radii  # (H,G)
    #         H2, G, D2 = C2.shape
    #         if H2 != H or D2 != D:
    #             raise RuntimeError("Level-2 head/dim mismatch.")
    #         if level1.child2parent is None:
    #             raise RuntimeError(
    #                 "Level-1 child2parent is None; cannot rebalance with 3+ levels."
    #             )
    #         c2p1 = level1.child2parent  # (H,P)

    #     any_change = False
    #     device = self.device

    #     for h in range(H):
    #         # counts per parent for this head: (P,)
    #         counts = torch.bincount(c2p0[h], minlength=P)
    #         large_parents = (counts > 2 * bf).nonzero(as_tuple=True)[0]
    #         if large_parents.numel() == 0:
    #             continue

    #         if getattr(self, "verbose", False):
    #             print(
    #                 f"[head {h}] Rebalancing: splitting {large_parents.numel()} oversized parents"
    #             )

    #         for parent_idx in large_parents.tolist():
    #             child_idx = (c2p0[h] == parent_idx).nonzero(as_tuple=True)[0]
    #             M = int(child_idx.numel())
    #             if M <= 2 * bf or M < 2:
    #                 continue

    #             pts = X0[h].index_select(0, child_idx)  # (M,D), on device

    #             # ---- random 2-way split (GPU)
    #             perm = torch.randperm(M, device=device)
    #             half = M // 2
    #             a = torch.zeros(M, device=device, dtype=torch.long)
    #             a[perm[half:]] = 1  # cluster id in {0,1}

    #             # ---- centroids
    #             # Avoid boolean indexing twice; compute means via sums and counts
    #             mask1 = a == 1
    #             mask0 = ~mask1

    #             # ensure neither side is empty (rare but possible when M==2 and half==1 it's safe)
    #             if mask0.sum() == 0 or mask1.sum() == 0:
    #                 # fallback: force a balanced split
    #                 a.zero_()
    #                 a[perm[half:]] = 1
    #                 mask1 = a == 1
    #                 mask0 = ~mask1

    #             # sums: (D,)
    #             sum0 = pts[mask0].sum(dim=0)
    #             sum1 = pts[mask1].sum(dim=0)
    #             cnt0 = mask0.sum().clamp_min(1)
    #             cnt1 = mask1.sum().clamp_min(1)

    #             c0 = sum0 / cnt0
    #             c1 = sum1 / cnt1
    #             centroids = torch.stack([c0, c1], dim=0).to(dtype=C1.dtype)  # (2,D)

    #             # ---- radii (leaf children have radius 0)
    #             chosen = centroids.index_select(0, a)  # (M,D)
    #             dists = torch.linalg.norm((pts - chosen).float(), dim=1)  # (M,) float32

    #             r0 = (
    #                 dists[mask0].max()
    #                 if mask0.any()
    #                 else torch.zeros((), device=device, dtype=dists.dtype)
    #             )
    #             r1 = (
    #                 dists[mask1].max()
    #                 if mask1.any()
    #                 else torch.zeros((), device=device, dtype=dists.dtype)
    #             )

    #             # ---- overwrite existing parent (cluster 0)
    #             C1[h, parent_idx] = centroids[0]
    #             R1[h, parent_idx] = r0.to(dtype=R1.dtype)

    #             # ---- append a new parent globally to keep rectangular (H,P,D)
    #             new_parent_idx = P
    #             dummy_center = torch.zeros((H, 1, D), device=device, dtype=C1.dtype)
    #             dummy_radius = torch.zeros((H, 1), device=device, dtype=R1.dtype)

    #             C1 = torch.cat([C1, dummy_center], dim=1)  # (H,P+1,D)
    #             R1 = torch.cat([R1, dummy_radius], dim=1)  # (H,P+1)

    #             C1[h, new_parent_idx] = centroids[1]
    #             R1[h, new_parent_idx] = r1.to(dtype=R1.dtype)

    #             P += 1
    #             any_change = True

    #             # keep level1 child2parent (to level2) aligned if present
    #             if has_l2:
    #                 if c2p1.shape != (H, P - 1):
    #                     raise RuntimeError(
    #                         f"level1.child2parent out of sync: got {tuple(c2p1.shape)}, expected {(H, P-1)}"
    #                     )
    #                 c2p1 = torch.cat(
    #                     [c2p1, torch.zeros((H, 1), device=device, dtype=c2p1.dtype)],
    #                     dim=1,
    #                 )

    #             # ---- reassign moved children (cluster 1) to new parent
    #             moved_child = child_idx[mask1]
    #             if moved_child.numel() > 0:
    #                 c2p0[h].index_fill_(0, moved_child, int(new_parent_idx))

    #             # ---- update bookkeeping buffers
    #             level0.num_parents = P
    #             level0.parent_mask_buf = torch.empty(
    #                 (H, P), dtype=torch.bool, device=device
    #             )

    #             # ---- update level1 -> level2 assignment and level2 radii (Torch-only)
    #             if has_l2:
    #                 affected = torch.tensor(
    #                     [parent_idx, new_parent_idx], device=device, dtype=torch.long
    #                 )
    #                 pts_aff = C1[h].index_select(0, affected)  # (2,D)
    #                 gidx, d2min = self._nearest_center_indices(
    #                     pts_aff, C2[h]
    #                 )  # (2,), (2,)
    #                 c2p1[h].index_copy_(0, affected, gidx)

    #                 # level2 radius update: max(||parent - gp|| + parent_radius)
    #                 dist_gp = d2min.sqrt_()  # (2,)
    #                 total = dist_gp + R1[h].index_select(0, affected).float()  # (2,)

    #                 upd = torch.full(
    #                     (G,), float("-inf"), device=device, dtype=total.dtype
    #                 )
    #                 upd.scatter_reduce_(
    #                     0, gidx, total, reduce="amax", include_self=True
    #                 )
    #                 upd = torch.where(torch.isfinite(upd), upd, torch.zeros_like(upd))
    #                 R2[h] = torch.maximum(R2[h], upd.to(dtype=R2.dtype))

    #         # end parent loop
    #     # end head loop

    #     if any_change:
    #         # write back possibly reallocated tensors
    #         level1.ball_centers = C1
    #         level1.ball_radii = R1
    #         level1.size = P
    #         level0.child2parent = c2p0
    #         if has_l2:
    #             level1.child2parent = c2p1

    #         # rebuild per-head level-1 FAISS indexes
    #         self._build_l1_faiss(level1)

    # Helper: compute nearest grandparent for a small set of points (M,D) to (G,D) using matmul
    # def _nearest_center_indices(
    #     self, points: torch.Tensor, centers: torch.Tensor
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     points:  (M,D)
    #     centers: (G,D)
    #     Returns:
    #     idx: (M,) argmin L2
    #     d2:  (M,) min squared distance
    #     """
    #     # Use float32 for stability even if model is fp16/bf16
    #     p = points.float()
    #     c = centers.float()
    #     # d2 = ||p||^2 + ||c||^2 - 2 p c^T
    #     p2 = (p * p).sum(dim=1, keepdim=True)  # (M,1)
    #     c2 = (c * c).sum(dim=1).unsqueeze(0)  # (1,G)
    #     d2_all = p2 + c2 - 2.0 * (p @ c.t())  # (M,G)
    #     idx = d2_all.argmin(dim=1)  # (M,)
    #     d2 = d2_all.gather(1, idx.view(-1, 1)).squeeze(1).clamp_min_(0.0)
    #     return idx.to(torch.long), d2

    # def _build_l1_faiss(self, level1):
    #     centers = level1.ball_centers
    #     H, P, D = centers.shape
    #     if D != self.dim:
    #         raise ValueError(f"Expected D={self.dim}, got D={D}")

    #     # FAISS wants float32 CPU contiguous arrays
    #     centers_cpu = centers.detach().to(dtype=torch.float32).contiguous()
    #     centers_np = centers_cpu.numpy()  # (H, P, D), float32

    #     indexes = []
    #     for h in range(H):
    #         idx = faiss.IndexFlatL2(self.dim)
    #         idx.add(centers_np[h])  # (P, D)
    #         indexes.append(idx)

    #     # self._faiss_l1_index = indexes
    #     return self

    def preallocate_search_buffers(self):
        for lvl in self.levels:
            if lvl.num_parents is not None:
                lvl.parent_mask_buf = torch.empty(
                    (self.num_heads, lvl.num_parents),
                    dtype=torch.bool,
                    device=self.device,
                )

    # def _sample_centroids(
    #     self,
    #     keys: torch.Tensor,  # (H, L, D)
    #     K: int,  # number of centroids
    #     seed: int = None,
    # ):
    #     assert keys.ndim == 3, "Expected keys shape (H, L, D)"
    #     H, L, D = keys.shape
    #     assert 1 <= K <= L, "Need 1 <= K <= L"

    #     cent_idx = torch.randint(0, L, (H, K), device=self.device)

    #     centroids = keys[
    #         torch.arange(H, device=self.device)[:, None], cent_idx
    #     ]  # (H, K, D)

    #     diff = keys.unsqueeze(2) - centroids.unsqueeze(1)  # (H, L, K, D)
    #     dist2 = (diff * diff).sum(dim=-1)  # (H, L, K)

    #     assign = dist2.argmin(dim=-1)

    #     # distance to assigned centroid: (H, L)
    #     min_dist = dist2.gather(dim=-1, index=assign.unsqueeze(-1)).squeeze(-1).sqrt()

    #     radii = None  # will be refined later
    #     # radii = torch.full(
    #     #     (H, K), float("-inf"), device=self.device, dtype=min_dist.dtype
    #     # )
    #     # radii.scatter_reduce_(
    #     #     dim=1, index=assign, src=min_dist, reduce="amax", include_self=True
    #     # )

    #     # centroids with no assigned points stay -inf → set to 0.0 (or torch.nan if you prefer)
    #     # radii = torch.where(torch.isfinite(radii), radii, torch.zeros_like(radii))

    #     return (
    #         centroids,  # (H, K, D)
    #         radii,  # (H, K)
    #         assign,  # (H, L) for each (head, child), which parent (centroid) it is assigned
    #     )

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

        # Replace empty parents (-inf) with empty_value
        # if empty_value is not None:
        #     parent_radii = torch.where(
        #         torch.isfinite(parent_radii),
        #         parent_radii,
        #         torch.full_like(parent_radii, float(empty_value)),
        #     )

        return parent_radii  # (H, K)

    def _refine_centroids_lloyd(
        self,
        points: torch.Tensor,  # (H, L, D)
        centroids: torch.Tensor,  # (H, K, D)
        assign: torch.Tensor,  # (H, L)
        num_iters: int,
    ):
        """Run a few Lloyd steps to tighten parent centroids per head.

        This improves pruning quality (smaller/tighter parent radii) while
        keeping build-time overhead controllable for CPU.
        """
        if num_iters <= 0:
            return centroids, assign

        H, L, D = points.shape
        _, K, _ = centroids.shape

        ones = torch.ones((H, L), device=self.device, dtype=points.dtype)
        for _ in range(num_iters):
            sums = torch.zeros((H, K, D), device=self.device, dtype=points.dtype)
            counts = torch.zeros((H, K), device=self.device, dtype=points.dtype)

            sums.scatter_add_(
                dim=1,
                index=assign.unsqueeze(-1).expand(H, L, D),
                src=points,
            )
            counts.scatter_add_(dim=1, index=assign, src=ones)

            nonempty = counts > 0
            safe_counts = counts.clamp_min(1.0).unsqueeze(-1)
            new_centroids = sums / safe_counts
            centroids = torch.where(nonempty.unsqueeze(-1), new_centroids, centroids)

            # Re-assign by nearest centroid.
            diff = points.unsqueeze(2) - centroids.unsqueeze(1)  # (H, L, K, D)
            dist2 = (diff * diff).sum(dim=-1)  # (H, L, K)
            assign = dist2.argmin(dim=-1)

        return centroids.contiguous(), assign.contiguous()
