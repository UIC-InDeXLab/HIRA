from abc import ABC
import torch
from typing import Optional, Tuple
from enum import Enum

from .base import BaseIndexer
from hira.kernels.triton_update_kernels import (
    fill_existing_children_atomic_batched,
    update_parent_radii_atomic_batched_masked,
    nearest_l2_triton_batched,
)


class CUDAIndexer(BaseIndexer):

    class DEPTH(Enum):
        # currently only up to three levels on GPU
        TWO_LEVELS = 2
        THREE_LEVELS = 3

    def __init__(
        self,
        depth: DEPTH,
        max_iterations: int,
        branching_factor: int,
        verbose: bool = False,
        pad_value: float = 0.0,
    ):
        self.depth = depth
        self.max_iterations = max_iterations
        self.branching_factor = branching_factor
        self.verbose = verbose
        self.pad_value = pad_value  # value used for padding unfilled slots

        # to build
        self.dim: int = 0
        self.children: Optional[torch.Tensor] = None  # padded keys
        self.parents: Optional[torch.Tensor] = None  # level 1
        self.parent_radii: Optional[torch.Tensor] = None  # level 1
        self.grand_parents: Optional[torch.Tensor] = None  # level 2
        self.grand_parent_radii: Optional[torch.Tensor] = None  # level 2
        # buffer
        self.buffer: Optional[torch.Tensor] = None

        # ====== UPDATE_V2 CACHE ======
        # Per-parent count of filled children slots (assumes contiguous fill from slot 0).
        self._child_counts: Optional[torch.Tensor] = None  # (num_parents,) int32 CUDA
        # Valid parent rows (needed for THREE_LEVELS where layout may contain padded parents).
        self._parent_valid: Optional[torch.Tensor] = None  # (num_parents,) bool CUDA

    @torch.no_grad()
    def build(self, keys: torch.Tensor):
        # keys: (1, H, L, D)

        # make sure keys are on GPU
        keys = keys.to("cuda").squeeze(0)  # (H, L, D)
        self.num_heads, _, self.dim = keys.shape

        if self.depth == CUDAIndexer.DEPTH.TWO_LEVELS:
            (
                self.parents,
                self.children,
            ) = self._build_parents_children_from_keys(keys, self.branching_factor)
            self.parent_radii = self._compute_parent_radii_from_layout()
        else:  # THREE_LEVELS
            (
                self.grand_parents,
                self.parents,
                self.children,
            ) = self._build_grandparents_parents_children_from_keys(
                keys, self.branching_factor
            )
            self.parent_radii = self._compute_parent_radii_from_layout()
            self.grand_parent_radii = self._compute_grandparent_radii_from_layout()

        # buffer for storing the scores
        self.buffer = torch.zeros(
            (self.children.shape[0], self.children.shape[1]),
            device="cuda",
            dtype=torch.float32,
        )

        # Initialize update_v2 cache state.
        self._init_update_state()

        return self

    # ------------------------------------------------------------------
    # Build Helpers
    # ------------------------------------------------------------------

    def _compute_parent_radii_from_layout(self) -> torch.Tensor:
        """
        Returns:
        radii: (H, m) float32 CUDA
        """
        if self.parents.ndim != 3:
            raise ValueError(
                f"parents must be (H,m,d), got {tuple(self.parents.shape)}"
            )
        if self.children.ndim != 3:
            raise ValueError(
                f"children must be (H,m*bf,d), got {tuple(self.children.shape)}"
            )

        H, m, d = self.parents.shape
        bf = int(self.branching_factor)

        parents_f = self.parents.float().contiguous()  # (H,m,d)
        children_f = self.children.float().contiguous().view(H, m, bf, d)  # (H,m,bf,d)

        diffs = children_f - parents_f[:, :, None, :]  # (H,m,bf,d)
        dists = torch.linalg.norm(diffs, dim=-1)  # (H,m,bf)

        if self.pad_value is not None:
            pad = float(self.pad_value)
            valid = ~torch.all(children_f == pad, dim=-1)  # (H,m,bf)
            dists = torch.where(
                valid,
                dists,
                torch.tensor(float("-inf"), device=dists.device, dtype=dists.dtype),
            )

        radii = torch.max(dists, dim=2).values  # (H,m)
        radii = torch.where(torch.isfinite(radii), radii, torch.zeros_like(radii))

        assert radii.is_cuda
        return radii

    def _compute_grandparent_radii_from_layout(self) -> torch.Tensor:
        """
        Returns:
            radii : (H, g) float32 CUDA
        """
        if self.grand_parents.ndim != 3:
            raise ValueError(
                f"grand_parents must be (H,g,d), got {tuple(self.grand_parents.shape)}"
            )

        H, g, d = self.grand_parents.shape
        bf = int(self.branching_factor)

        gp_f = self.grand_parents.float().contiguous()  # (H,g,d)
        parents_f = self.parents.float().contiguous().view(H, g, bf, d)  # (H,g,bf,d)

        pr = self.parent_radii.float().contiguous().view(H, g, bf)  # (H,g,bf)

        # Distance between grandparent and each parent
        dists = torch.linalg.norm(parents_f - gp_f[:, :, None, :], dim=-1)  # (H,g,bf)

        totals = dists + pr  # (H,g,bf)

        if self.pad_value is not None:
            pad = float(self.pad_value)
            valid = ~torch.all(parents_f == pad, dim=-1)  # (H,g,bf)

            totals = torch.where(
                valid,
                totals,
                torch.tensor(
                    float("-inf"),
                    device=totals.device,
                    dtype=totals.dtype,
                ),
            )

        radii = torch.max(totals, dim=2).values  # (H,g)

        radii = torch.where(
            torch.isfinite(radii),
            radii,
            torch.zeros_like(radii),
        )

        assert radii.is_cuda
        return radii

    def _build_random_bf_level_batched(
        self,
        x: torch.Tensor,  # (H, n, d) float32 CUDA
        bf: int,
        *,
        seed: int = 1234,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One-level builder, batched over heads.

        Returns:
        - C: (H, K, d) centroids (randomly selected points from x per head)
        - selected: (H, K, bf) indices into x[h] (0..n-1), or -1 if padded

        Where:
        H = #heads, n = #points per head, d = dim
        K = max(1, ceil(n / bf))  -- ceiling so every key gets a slot
        """
        assert x.is_cuda and x.dtype == torch.float32 and x.ndim == 3
        device = x.device
        H, n, d = x.shape
        assert bf >= 1
        # Ceiling division: K*bf >= n so every key has a slot.
        K = max(1, (n + bf - 1) // bf)

        # ----------------------------
        # 1) Random centroid selection (per head)
        # ----------------------------
        # 1) generate random scores per head
        rand_scores = torch.rand(H, n, device=device)

        # 2) take top-K per head (unique indices)
        cent_idx = torch.topk(rand_scores, K, dim=1).indices  # (H, K)

        # 3) gather centroids
        C = x[
            torch.arange(H, device=device)[:, None], cent_idx  # (H,1)  # (H,K)
        ]  # -> (H, K, d)

        # ----------------------------
        # 2) Assign each point -> nearest centroid (batched over heads)
        # ----------------------------
        # dist: (H, n, K)
        # torch.cdist supports batched inputs: (H,n,d) vs (H,K,d)
        dist = torch.cdist(x, C)  # Euclidean
        dist1, idx1 = dist.min(dim=2)  # (H, n), (H, n) centroid assignment

        # ----------------------------
        # 3) Choose up to bf closest per centroid (true lexicographic by (centroid, dist))
        #    Vectorized version of your "composite sort" trick, but batched.
        # ----------------------------
        # dist_rank: rank of each point by distance within each head
        dist_order = torch.argsort(
            dist1, dim=1
        )  # (H, n) indices of points by increasing dist
        dist_rank = torch.empty_like(dist_order)
        dist_rank.scatter_(
            1, dist_order, torch.arange(n, device=device).view(1, n).expand(H, n)
        )

        # composite key to group by centroid then by distance rank
        composite = idx1.to(torch.int64) * (n + 1) + dist_rank.to(torch.int64)  # (H, n)
        order = torch.argsort(
            composite, dim=1
        )  # (H, n) grouped by centroid, then closest first

        idx_grp = idx1.gather(1, order)  # (H, n) centroid id per grouped position
        pts_grp = order  # (H, n) original point indices in grouped order

        # counts per centroid (batched bincount via scatter_add)
        counts = torch.zeros((H, K), device=device, dtype=torch.int64)
        ones = torch.ones((H, n), device=device, dtype=torch.int64)
        counts.scatter_add_(1, idx1.clamp_(0, K - 1), ones)

        # offsets: (H, K+1), offsets[h, c] = starting index in grouped list for centroid c
        offsets = torch.zeros((H, K + 1), device=device, dtype=torch.int64)
        offsets[:, 1:] = torch.cumsum(counts, dim=1)

        # For each grouped position p, compute local_rank within its centroid-run:
        pos = (
            torch.arange(n, device=device, dtype=torch.int64).view(1, n).expand(H, n)
        )  # (H, n)
        start = offsets[:, :-1].gather(1, idx_grp)  # (H, n)
        local_rank = pos - start  # (H, n), 0..count-1 inside centroid block

        mask = local_rank < bf  # take first bf per centroid
        # Prepare output (H, K, bf)
        selected = torch.full((H, K, bf), -1, device=device, dtype=torch.int64)

        # Scatter selected points into (H,K,bf) using flat indexing
        h_idx = (
            torch.arange(H, device=device, dtype=torch.int64).view(H, 1).expand(H, n)
        )
        c_idx = idx_grp.to(torch.int64)
        r_idx = local_rank.to(torch.int64)

        # Keep only valid entries (first bf per centroid)
        h_m = h_idx[mask]
        c_m = c_idx[mask]
        r_m = r_idx[mask]
        p_m = pts_grp[mask].to(torch.int64)

        flat = h_m * (K * bf) + c_m * bf + r_m
        selected.view(-1).scatter_(0, flat, p_m)

        # ----------------------------
        # 4) Optional: fill deficits (-1) using random leftovers per head
        #    (simple + deterministic; loops over H only)
        # ----------------------------
        if (selected == -1).any():
            for h in range(H):
                sel_h = selected[h]  # (K, bf)
                missing = (sel_h == -1).view(-1)
                m = int(missing.sum().item())
                if m == 0:
                    continue

                used = torch.zeros((n,), device=device, dtype=torch.bool)
                taken = sel_h[sel_h != -1]
                if taken.numel() > 0:
                    used[taken] = True
                leftovers = torch.nonzero(~used, as_tuple=False).view(-1)
                if leftovers.numel() == 0:
                    continue

                gh = torch.Generator(device=device)
                gh.manual_seed(seed + 2003 * h + 17)
                perm = torch.randperm(leftovers.numel(), generator=gh, device=device)
                fill = leftovers[perm[: min(m, leftovers.numel())]]

                flat_sel = sel_h.view(-1)
                flat_sel[missing.nonzero(as_tuple=False).view(-1)[: fill.numel()]] = (
                    fill
                )
                selected[h] = flat_sel.view(K, bf)

        return C, selected

    def _build_parents_children_from_keys(
        self,
        keys: torch.Tensor,  # (H, n, d)
        bf: int,
        *,
        seed: int = 1234,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build one level from raw keys using ONLY _build_faiss_kmeans_bf_level.

        Returns:
        parents  : (H, m, d) float32 CUDA, where m = max(1, ceil(n / bf))
        children : (H, m*bf, d) float32 CUDA, where children[i*bf:(i+1)*bf] are parent i's children.
                    If a slot is unfilled (selected == -1), it is padded with pad_value.

        Ordering:
        - For each parent i, children[i*bf:(i+1)*bf] follow the selection order from _build_faiss_kmeans_bf_level
            (centroid-grouped, closest-first within group).
        """
        if keys.ndim != 3:
            raise ValueError(f"keys must be (H,n,d), got {tuple(keys.shape)}")
        if not keys.is_cuda:
            raise ValueError("keys must be on CUDA")
        if bf <= 0:
            raise ValueError("bf must be positive")

        x = keys.detach()
        if x.dtype != torch.float32:
            x = x.float()
        x = x.contiguous()

        device = x.device
        H, n, d = x.shape

        # Build one level (batched over heads)
        parents, selected = self._build_random_bf_level_batched(x, bf, seed=seed)
        # parents : (H, m, d)  where m = max(1, ceil(n/bf))
        # selected: (H, m, bf) indices into x[h] or -1

        # Derive m from the actual output shape (ceil-division, not floor).
        m = parents.shape[1]

        # Materialize children: (H, m*bf, d)
        children = torch.full(
            (H, m * bf, d),
            float(self.pad_value),
            device=device,
            dtype=torch.float32,
        )

        sel_flat = selected.reshape(H, m * bf)  # (H, m*bf)
        valid = sel_flat != -1

        if valid.any():
            # For invalid positions, set index to 0 so gather is safe; we'll mask them out anyway.
            safe_idx = sel_flat.clamp_min(0)  # (H, m*bf)

            # Gather chosen children from x: (H, m*bf, d)
            gathered = x.gather(
                1,
                safe_idx.unsqueeze(-1).expand(-1, -1, d),
            )

            # Write only valid slots; keep pad_value in invalid slots
            children[valid] = gathered[valid]

        assert parents.is_cuda and children.is_cuda
        return parents, children

    def _build_grandparents_parents_children_from_keys(
        self,
        keys: torch.Tensor,  # (H,n,d) CUDA float/half ok
        bf: int,
        *,
        seed: int = 1234,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched (over heads) two-level build.

        Returns:
        grand_parents      : (H, g, d) float32 CUDA, g = max(1, m // bf) where m=max(1, n // bf)
        parents_reordered  : (H, g*bf, d) float32 CUDA, contiguous bf-block per grandparent
        children_reordered : (H, g*bf*bf, d) float32 CUDA, contiguous bf-block per parent (in reordered parent order)

        Invariants (per head h):
        - parents_reordered[h, i*bf:(i+1)*bf] are the parents of grand_parent[h, i] (padding possible)
        - children_reordered[h, p*bf:(p+1)*bf] are children of parent p in parents_reordered[h]
        """
        if keys.ndim != 3:
            raise ValueError(f"keys must be (H,n,d), got {tuple(keys.shape)}")
        if not keys.is_cuda:
            raise ValueError("keys must be on CUDA")
        if bf <= 0:
            raise ValueError("bf must be positive")

        # --- first level: keys -> parents, children ---
        parents, children = self._build_parents_children_from_keys(keys, bf, seed=seed)
        # parents:  (H, m, d)
        # children: (H, m*bf, d)
        device = parents.device
        H, m, d = parents.shape

        # --- second level: parents -> grand_parents, select parent-ids per grandparent ---
        gp, sel_par = self._build_random_bf_level_batched(
            parents.contiguous(), bf, seed=seed + 1
        )
        # gp:      (H, g, d)
        # sel_par: (H, g, bf)
        _, g, _ = gp.shape

        # Flatten selection: (H, g*bf)
        sel_flat = sel_par.reshape(H, g * bf)
        valid = sel_flat != -1  # (H, g*bf)

        # ----------------------------
        # Reorder parents into contiguous bf-blocks per grandparent
        # ----------------------------
        parents_reordered = torch.full(
            (H, g * bf, d),
            float(self.pad_value),
            device=device,
            dtype=torch.float32,
        )

        if valid.any():
            safe_idx = sel_flat.clamp_min(0)  # make gather safe
            gathered_parents = parents.gather(
                1,
                safe_idx.unsqueeze(-1).expand(-1, -1, d),
            )  # (H, g*bf, d)
            parents_reordered[valid] = gathered_parents[valid]

        # ----------------------------
        # Reorder children by permuting whole parent child-blocks
        # children aligned with original parents: children.view(H, m, bf, d)
        # ----------------------------
        child_blocks = children.view(H, m, bf, d)  # (H, m, bf, d)

        children_reordered_blocks = torch.full(
            (H, g * bf, bf, d),
            float(self.pad_value),
            device=device,
            dtype=torch.float32,
        )

        if valid.any():
            safe_idx = sel_flat.clamp_min(0)  # (H, g*bf)
            gathered_blocks = child_blocks.gather(
                1,
                safe_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, bf, d),
            )  # (H, g*bf, bf, d)
            # mask-fill only valid parent slots
            children_reordered_blocks[valid] = gathered_blocks[valid]

        children_reordered = children_reordered_blocks.view(H, g * bf * bf, d)

        assert gp.is_cuda and parents_reordered.is_cuda and children_reordered.is_cuda
        return gp, parents_reordered, children_reordered

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, new_keys: torch.Tensor):
        """Incremental update of the index with new key vectors.

        Args:
            new_keys: (1, H, M, D) new key vectors per head.

        Algorithm:
            1. Find nearest parent for each new key; fill if the parent
               has a free children slot.
            2. Orphan keys (parent full) -> randomly select O//bf new parents
               from orphans, assign orphans to nearest new parent, build
               children blocks. Append new parents + children.
            3. THREE_LEVELS only: repeat the same pattern for the newly
               created parents at the grandparent level, then append new
               grandparent blocks for any overflow.
            4. Update all radii and refresh internal state.
        """
        if new_keys.numel() == 0:
            return self
        if self.parents is None or self.children is None:
            raise RuntimeError("update() called before build()")

        new_keys = new_keys.squeeze(0).contiguous()
        assert new_keys.ndim == 3

        H, M, D = new_keys.shape
        bf = int(self.branching_factor)
        device = new_keys.device

        self._ensure_update_state()

        # ==============================================================
        # Phase 1 - place new keys into existing parent children blocks
        # ==============================================================
        nearest_parent, _ = nearest_l2_triton_batched(
            new_keys, self.parents, valid_mask=self._parent_valid
        )
        nearest_parent = nearest_parent.to(torch.int32)

        placed_mask, _placed_flat = fill_existing_children_atomic_batched(
            x=new_keys,
            parent_idx=nearest_parent,
            child_counts=self._child_counts,
            children=self.children,
            bf=bf,
        )

        # Update parent radii for placed keys (atomic max).
        if placed_mask.any() and self.parent_radii.dtype == torch.float32:
            update_parent_radii_atomic_batched_masked(
                new_keys,
                nearest_parent,
                placed_mask.to(torch.uint8),
                self.parents,
                self.parent_radii,
            )

        # ==============================================================
        # Phase 2 - build new parents from orphan keys
        # ==============================================================
        overflow_mask = ~placed_mask  # (H, M)
        orphan_counts = overflow_mask.sum(dim=1)  # (H,)
        total_orphans = orphan_counts.sum().item()

        if total_orphans == 0:
            if self.depth == CUDAIndexer.DEPTH.THREE_LEVELS and placed_mask.any():
                self._update_gp_radii_for_placed_children(placed_mask, nearest_parent)
            self._refresh_after_update()
            return self

        new_parents, new_children_flat, new_parent_radii = (
            self._build_level_from_orphans(
                all_items=new_keys,
                overflow_mask=overflow_mask,
                orphan_counts=orphan_counts,
            )
        )
        # new_parents:       (H, K_max, D)
        # new_children_flat: (H, K_max * bf, D)
        # new_parent_radii:  (H, K_max)

        # ==============================================================
        # Phase 3 - incorporate new parents
        # ==============================================================
        if self.depth == CUDAIndexer.DEPTH.TWO_LEVELS:
            self.parents = torch.cat([self.parents, new_parents], dim=1).contiguous()
            self.parent_radii = torch.cat(
                [self.parent_radii, new_parent_radii], dim=1
            ).contiguous()
            self.children = torch.cat(
                [self.children, new_children_flat], dim=1
            ).contiguous()
            self._refresh_after_update()
            return self

        # ---------- THREE_LEVELS ----------
        assert self.grand_parents is not None and self.grand_parent_radii is not None
        H, G_old, _ = self.grand_parents.shape
        P_old = self.parents.shape[1]
        assert P_old == G_old * bf

        # Update GP radii for keys placed in Phase 1.
        if placed_mask.any():
            self._update_gp_radii_for_placed_children(placed_mask, nearest_parent)

        # Try to place new parents into existing GP blocks.
        pad = float(self.pad_value)
        new_parent_valid = ~torch.all(new_parents == pad, dim=-1)  # (H, K_max)
        K_max = new_parents.shape[1]

        parent_placed_mask = torch.zeros((H, K_max), device=device, dtype=torch.bool)

        if new_parent_valid.any():
            self._place_parents_into_gp_blocks(
                new_parents,
                new_parent_radii,
                new_children_flat,
                new_parent_valid,
                parent_placed_mask,
            )

        # Orphan parents -> create new grandparent blocks.
        parent_overflow_mask = new_parent_valid & (~parent_placed_mask)
        parent_orphan_counts = parent_overflow_mask.sum(dim=1)

        if parent_orphan_counts.sum().item() > 0:
            self._grow_three_levels_from_orphan_parents(
                new_parents,
                new_parent_radii,
                new_children_flat,
                parent_overflow_mask,
                parent_orphan_counts,
            )

        self._refresh_after_update()
        return self

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def _build_level_from_orphans(
        self,
        all_items: torch.Tensor,  # (H, M, D)
        overflow_mask: torch.Tensor,  # (H, M) bool
        orphan_counts: torch.Tensor,  # (H,)
    ):
        """Build new parents + children blocks from orphan keys.

        Returns (all padded to max K across heads):
            new_parents       : (H, K_max, D)
            new_children_flat : (H, K_max * bf, D)
            new_parent_radii  : (H, K_max)
        """
        H, M, D = all_items.shape
        bf = int(self.branching_factor)
        device = all_items.device
        pad = float(self.pad_value)

        # Ceiling division so K_h * bf >= O_h -- every orphan key gets a slot.
        K_per_head = torch.where(
            orphan_counts > 0,
            torch.clamp((orphan_counts + bf - 1) // bf, min=1),
            torch.zeros_like(orphan_counts),
        )
        K_max = K_per_head.max().item()

        new_parents = torch.full((H, K_max, D), pad, device=device, dtype=torch.float32)
        new_children_flat = torch.full(
            (H, K_max * bf, D), pad, device=device, dtype=torch.float32
        )
        new_parent_radii = torch.zeros((H, K_max), device=device, dtype=torch.float32)

        for h in range(H):
            O_h = orphan_counts[h].item()
            if O_h == 0:
                continue
            K_h = K_per_head[h].item()

            orphans_h = all_items[h][overflow_mask[h]]  # (O_h, D)

            parents_h, children_h, radii_h = self._build_one_level_from_points(
                orphans_h, K_h, bf, pad, device, D
            )

            new_parents[h, :K_h] = parents_h
            new_children_flat[h, : K_h * bf] = children_h
            new_parent_radii[h, :K_h] = radii_h

        return new_parents, new_children_flat, new_parent_radii

    @staticmethod
    def _build_one_level_from_points(
        points: torch.Tensor,  # (N, D)
        K: int,  # number of parents to create
        bf: int,
        pad: float,
        device: torch.device,
        D: int,
    ):
        """Build K parents and K*bf children from N points (single head).

        Returns:
            parents  : (K, D)
            children : (K*bf, D), padded with *pad*
            radii    : (K,) float32
        """
        N = points.shape[0]

        # Random parent selection
        perm = torch.randperm(N, device=device)
        parents = points[perm[:K]].contiguous()  # (K, D)

        # Assign every point to nearest parent
        dist = torch.cdist(points.unsqueeze(0), parents.unsqueeze(0)).squeeze(
            0
        )  # (N, K)
        assign = dist.argmin(dim=1)  # (N,)
        dist_assigned = dist.gather(1, assign.unsqueeze(1)).squeeze(1)  # (N,)

        # Group by (parent, distance) - closest first
        dist_order = torch.argsort(dist_assigned)
        dist_rank = torch.empty_like(dist_order)
        dist_rank[dist_order] = torch.arange(N, device=device, dtype=dist_order.dtype)

        composite = assign.to(torch.int64) * (N + 1) + dist_rank.to(torch.int64)
        order = torch.argsort(composite)

        idx_sorted = assign[order]
        pts_sorted = points[order]

        counts = torch.bincount(idx_sorted, minlength=K).to(torch.int64)
        offsets = torch.zeros(K + 1, device=device, dtype=torch.int64)
        offsets[1:] = torch.cumsum(counts, dim=0)

        pos = torch.arange(N, device=device, dtype=torch.int64)
        start_pos = offsets[:-1].gather(0, idx_sorted)
        local_rank = pos - start_pos

        placed = local_rank < bf
        children = torch.full((K * bf, D), pad, device=device, dtype=torch.float32)

        if placed.any():
            dst_idx = idx_sorted[placed] * bf + local_rank[placed]
            children.index_copy_(0, dst_idx, pts_sorted[placed])

        # Leftovers (overflow beyond bf per centroid) go into any remaining
        # free slots.  With ceiling-division K, K*bf >= N guarantees
        # free.numel() >= leftovers.shape[0] always holds; the min() guards
        # against any unexpected edge case without silently dropping keys.
        if not placed.all():
            leftovers = pts_sorted[~placed]
            empty_slots = torch.all(children == pad, dim=-1)
            free = torch.nonzero(empty_slots, as_tuple=False).view(-1)
            n_fill = min(free.numel(), leftovers.shape[0])
            if n_fill > 0:
                children[free[:n_fill]] = leftovers[:n_fill]

        # Compute parent radii
        c_view = children.view(K, bf, D)
        valid = ~torch.all(c_view == pad, dim=-1)
        dists_r = torch.linalg.norm((c_view - parents[:, None, :]).float(), dim=-1)
        dists_r = torch.where(
            valid,
            dists_r,
            torch.tensor(float("-inf"), device=device),
        )
        radii = torch.max(dists_r, dim=1).values
        radii = torch.where(torch.isfinite(radii), radii, torch.zeros_like(radii))

        return parents, children, radii

    def _place_parents_into_gp_blocks(
        self,
        new_parents: torch.Tensor,  # (H, K, D)
        new_parent_radii: torch.Tensor,  # (H, K)
        new_children_flat: torch.Tensor,  # (H, K*bf, D)
        new_parent_valid: torch.Tensor,  # (H, K) bool
        parent_placed_mask: torch.Tensor,  # (H, K) bool - updated in-place
    ):
        """Try to place new parents into empty slots of existing GP blocks.

        Modifies self.parents, self.parent_radii, self.children,
        self.grand_parent_radii, and *parent_placed_mask* in place.
        """
        bf = int(self.branching_factor)
        device = self.parents.device
        H, G_old, D = self.grand_parents.shape
        K = new_parents.shape[1]

        # Find nearest GP for each new parent.
        nearest_gp, _ = nearest_l2_triton_batched(
            new_parents, self.grand_parents, valid_mask=None
        )
        nearest_gp = nearest_gp.to(torch.int64)

        # Current fill per GP: (H, G_old)
        gp_counts = self._gp_child_counts.to(torch.int64)

        for h in range(H):
            valid_k = torch.nonzero(new_parent_valid[h], as_tuple=False).view(-1)
            if valid_k.numel() == 0:
                continue
            n_valid = valid_k.numel()
            gp_assign = nearest_gp[h, valid_k]  # (n_valid,)

            avail = (bf - gp_counts[h]).clamp_min(0)  # (G_old,)

            # Rank within GP groups
            order = torch.argsort(gp_assign)
            gp_sorted = gp_assign[order]

            counts = torch.zeros(G_old, device=device, dtype=torch.int64)
            counts.scatter_add_(
                0,
                gp_sorted,
                torch.ones(n_valid, device=device, dtype=torch.int64),
            )
            offsets = torch.zeros(G_old + 1, device=device, dtype=torch.int64)
            offsets[1:] = torch.cumsum(counts, dim=0)

            pos = torch.arange(n_valid, device=device, dtype=torch.int64)
            start_pos = offsets[:-1].gather(0, gp_sorted)
            rank = pos - start_pos

            avail_sorted = avail.gather(0, gp_sorted)
            can_place = rank < avail_sorted

            if not can_place.any():
                continue

            placed_o = order[can_place]
            placed_k = valid_k[placed_o]
            placed_gp = gp_sorted[can_place]
            placed_slot = gp_counts[h].gather(0, placed_gp) + rank[can_place]
            dst_p = placed_gp * bf + placed_slot  # position in parents array

            # Write parent vectors + radii
            self.parents[h, dst_p] = new_parents[h, placed_k]
            self.parent_radii[h, dst_p] = new_parent_radii[h, placed_k].to(
                self.parent_radii.dtype
            )

            # Write children blocks
            P_old = self.parents.shape[1]
            src_c = new_children_flat[h].view(K, bf, D)
            dst_c = self.children[h].view(P_old, bf, D)
            dst_c[dst_p] = src_c[placed_k]

            parent_placed_mask[h, placed_k] = True

            # Update GP radii (monotonic increase)
            gp_centers = self.grand_parents[h, placed_gp].float()
            par_vecs = self.parents[h, dst_p].float()
            dist_gp = torch.linalg.norm(par_vecs - gp_centers, dim=1)
            total = dist_gp + self.parent_radii[h, dst_p].float()

            upd = torch.full(
                (G_old,), float("-inf"), device=device, dtype=torch.float32
            )
            upd.scatter_reduce_(0, placed_gp, total, reduce="amax", include_self=True)
            upd = torch.where(torch.isfinite(upd), upd, torch.zeros_like(upd))
            self.grand_parent_radii[h] = torch.maximum(
                self.grand_parent_radii[h].float(), upd
            ).to(self.grand_parent_radii.dtype)

    def _grow_three_levels_from_orphan_parents(
        self,
        new_parents: torch.Tensor,  # (H, K, D)
        new_parent_radii: torch.Tensor,  # (H, K)
        new_children_flat: torch.Tensor,  # (H, K*bf, D)
        parent_overflow_mask: torch.Tensor,  # (H, K) bool
        parent_orphan_counts: torch.Tensor,  # (H,)
    ):
        """Create new GP blocks from orphan parents and append to index."""
        bf = int(self.branching_factor)
        pad = float(self.pad_value)
        device = self.parents.device
        H = new_parents.shape[0]
        D = new_parents.shape[2]
        K = new_parents.shape[1]

        # Ceiling division so K_gp * bf >= n_orphan_parents -- none dropped.
        K_gp_per_head = torch.where(
            parent_orphan_counts > 0,
            torch.clamp((parent_orphan_counts + bf - 1) // bf, min=1),
            torch.zeros_like(parent_orphan_counts),
        )
        K_gp_max = K_gp_per_head.max().item()

        new_gps = torch.full((H, K_gp_max, D), pad, device=device, dtype=torch.float32)
        new_parents_block = torch.full(
            (H, K_gp_max * bf, D), pad, device=device, dtype=torch.float32
        )
        new_pr_block = torch.zeros(
            (H, K_gp_max * bf), device=device, dtype=torch.float32
        )
        new_children_block = torch.full(
            (H, K_gp_max * bf * bf, D),
            pad,
            device=device,
            dtype=torch.float32,
        )
        new_gp_radii = torch.zeros((H, K_gp_max), device=device, dtype=torch.float32)

        for h in range(H):
            n_orphan = parent_orphan_counts[h].item()
            if n_orphan == 0:
                continue
            K_gp = K_gp_per_head[h].item()

            orphan_mask_h = parent_overflow_mask[h]
            orphan_parents_h = new_parents[h][orphan_mask_h]  # (n_orphan, D)
            orphan_radii_h = new_parent_radii[h][orphan_mask_h]  # (n_orphan,)
            orphan_children_h = new_children_flat[h].view(K, bf, D)[
                orphan_mask_h
            ]  # (n_orphan, bf, D)

            # Random GP selection from orphan parents
            perm = torch.randperm(n_orphan, device=device)
            gps_h = orphan_parents_h[perm[:K_gp]]  # (K_gp, D)

            # Assign orphan parents to nearest GP
            dist = torch.cdist(
                orphan_parents_h.unsqueeze(0), gps_h.unsqueeze(0)
            ).squeeze(0)
            gp_assign = dist.argmin(dim=1)
            dist_assigned = dist.gather(1, gp_assign.unsqueeze(1)).squeeze(1)

            # Group by (GP, distance)
            dist_order = torch.argsort(dist_assigned)
            dist_rank = torch.empty_like(dist_order)
            dist_rank[dist_order] = torch.arange(
                n_orphan, device=device, dtype=dist_order.dtype
            )
            composite = gp_assign.to(torch.int64) * (n_orphan + 1) + dist_rank.to(
                torch.int64
            )
            order = torch.argsort(composite)

            gp_sorted = gp_assign[order]
            parents_sorted = orphan_parents_h[order]
            radii_sorted = orphan_radii_h[order]
            children_sorted = orphan_children_h[order]  # (n_orphan, bf, D)

            counts = torch.bincount(gp_sorted, minlength=K_gp).to(torch.int64)
            offsets = torch.zeros(K_gp + 1, device=device, dtype=torch.int64)
            offsets[1:] = torch.cumsum(counts, dim=0)

            pos = torch.arange(n_orphan, device=device, dtype=torch.int64)
            start_pos = offsets[:-1].gather(0, gp_sorted)
            local_rank = pos - start_pos
            placed_gp = local_rank < bf

            parents_blk = torch.full(
                (K_gp * bf, D), pad, device=device, dtype=torch.float32
            )
            radii_blk = torch.zeros(K_gp * bf, device=device, dtype=torch.float32)
            children_blk = torch.full(
                (K_gp * bf * bf, D),
                pad,
                device=device,
                dtype=torch.float32,
            )
            children_blk_view = children_blk.view(K_gp * bf, bf, D)

            if placed_gp.any():
                dst_p = gp_sorted[placed_gp] * bf + local_rank[placed_gp]
                parents_blk[dst_p] = parents_sorted[placed_gp]
                radii_blk[dst_p] = radii_sorted[placed_gp]
                children_blk_view[dst_p] = children_sorted[placed_gp]

            # Leftovers: orphan parents that overflowed their assigned GP.
            # With ceiling-division K_gp, free slots >= leftovers always.
            if not placed_gp.all():
                left_p = parents_sorted[~placed_gp]
                left_r = radii_sorted[~placed_gp]
                left_c = children_sorted[~placed_gp]

                empty = torch.all(parents_blk == pad, dim=-1)
                free = torch.nonzero(empty, as_tuple=False).view(-1)
                n_fill = min(free.numel(), left_p.shape[0])
                if n_fill > 0:
                    parents_blk[free[:n_fill]] = left_p[:n_fill]
                    radii_blk[free[:n_fill]] = left_r[:n_fill]
                    children_blk_view[free[:n_fill]] = left_c[:n_fill]

            # GP radii
            gp_f = gps_h.float()
            pv = parents_blk.float().view(K_gp, bf, D)
            rv = radii_blk.float().view(K_gp, bf)
            valid_p = ~torch.all(pv == pad, dim=-1)
            dists_gp = torch.linalg.norm(pv - gp_f[:, None, :], dim=-1)
            totals = dists_gp + rv
            totals = torch.where(
                valid_p,
                totals,
                torch.tensor(float("-inf"), device=device),
            )
            gpr = torch.max(totals, dim=1).values
            gpr = torch.where(torch.isfinite(gpr), gpr, torch.zeros_like(gpr))

            new_gps[h, :K_gp] = gps_h
            new_gp_radii[h, :K_gp] = gpr
            new_parents_block[h, : K_gp * bf] = parents_blk
            new_pr_block[h, : K_gp * bf] = radii_blk
            new_children_block[h, : K_gp * bf * bf] = children_blk

        # Append
        self.grand_parents = torch.cat(
            [self.grand_parents, new_gps], dim=1
        ).contiguous()
        self.grand_parent_radii = torch.cat(
            [self.grand_parent_radii, new_gp_radii], dim=1
        ).contiguous()
        self.parents = torch.cat([self.parents, new_parents_block], dim=1).contiguous()
        self.parent_radii = torch.cat(
            [self.parent_radii, new_pr_block], dim=1
        ).contiguous()
        self.children = torch.cat(
            [self.children, new_children_block], dim=1
        ).contiguous()

    def _update_gp_radii_for_placed_children(
        self,
        placed_mask: torch.Tensor,  # (H, M) bool
        nearest_parent: torch.Tensor,  # (H, M) int32
    ):
        """Update grand_parent_radii after placing children under existing parents."""
        H = placed_mask.shape[0]
        bf = int(self.branching_factor)
        G = self.grand_parents.shape[1]
        device = placed_mask.device

        h_idx = torch.arange(H, device=device)[:, None].expand_as(placed_mask)
        h_m = h_idx[placed_mask]
        p_m = nearest_parent[placed_mask].to(torch.int64)
        gp_m = p_m // bf

        gp_centers = self.grand_parents[h_m, gp_m].float()
        par_vecs = self.parents[h_m, p_m].float()
        dist = torch.linalg.norm(par_vecs - gp_centers, dim=1)
        total = dist + self.parent_radii[h_m, p_m].float()

        upd_flat = torch.full(
            (H * G,), float("-inf"), device=device, dtype=torch.float32
        )
        upd_flat.scatter_reduce_(
            0, h_m * G + gp_m, total, reduce="amax", include_self=True
        )
        upd = upd_flat.view(H, G)
        upd = torch.where(torch.isfinite(upd), upd, torch.zeros_like(upd))
        self.grand_parent_radii = torch.maximum(
            self.grand_parent_radii.float(), upd
        ).to(self.grand_parent_radii.dtype)

    def _refresh_after_update(self):
        """Refresh buffer and cached state after an update."""
        self.buffer = torch.zeros(
            (self.children.shape[0], self.children.shape[1]),
            device="cuda",
            dtype=torch.float32,
        )
        self._init_update_state()

    # ------------------------------------------------------------------
    # Update state management
    # ------------------------------------------------------------------

    def _init_update_state(self) -> None:
        if self.parents is None or self.children is None:
            self._child_counts = None
            self._parent_valid = None
            self._gp_child_counts = None
            return

        bf = int(self.branching_factor)

        if self.parents.ndim != 3:
            raise ValueError(
                f"parents must be (H,P,d), got {tuple(self.parents.shape)}"
            )
        if self.children.ndim != 3:
            raise ValueError(
                f"children must be (H,P*bf,d), got {tuple(self.children.shape)}"
            )

        H, P, d = self.parents.shape

        # Parent validity (padded parents are all pad_value).
        if self.pad_value is None:
            self._parent_valid = torch.ones(
                (H, P), device=self.parents.device, dtype=torch.bool
            )
        else:
            pad = float(self.pad_value)
            self._parent_valid = ~torch.all(self.parents == pad, dim=-1)  # (H,P)

        # Child counts per parent.
        children3 = self.children.view(H, P, bf, d)
        if self.pad_value is None:
            valid_child = torch.ones(
                (H, P, bf), device=self.children.device, dtype=torch.bool
            )
        else:
            pad = float(self.pad_value)
            valid_child = ~torch.all(children3 == pad, dim=-1)

        self._child_counts = valid_child.sum(dim=2).to(torch.int32).contiguous()

        # GP child counts (parent fill per GP) for THREE_LEVELS.
        if (
            self.depth == CUDAIndexer.DEPTH.THREE_LEVELS
            and self.grand_parents is not None
        ):
            G = self.grand_parents.shape[1]
            parents4 = self.parents.view(H, G, bf, d)
            if self.pad_value is None:
                gp_valid = torch.ones(
                    (H, G, bf), device=self.parents.device, dtype=torch.bool
                )
            else:
                pad = float(self.pad_value)
                gp_valid = ~torch.all(parents4 == pad, dim=-1)
            self._gp_child_counts = gp_valid.sum(dim=2).to(torch.int32).contiguous()
        else:
            self._gp_child_counts = None

    def _ensure_update_state(self) -> None:
        if self._child_counts is None or self._parent_valid is None:
            self._init_update_state()
