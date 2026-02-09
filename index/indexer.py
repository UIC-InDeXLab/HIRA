from abc import ABC
import torch
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import faiss
from enum import Enum
import faiss.contrib.torch_utils  # enables torch<->faiss GPU interop

from hira.kernels.update_triton_kernels import (
    fill_existing_children_atomic,
    nearest_l2_triton,
    scatter_parent_children_blocks,
    update_parent_radii_atomic,
)


class Indexer(ABC):
    def build(self, keys: torch.Tensor):
        raise NotImplementedError

    def update(self, new_keys: torch.Tensor):
        raise NotImplementedError


class CPUIndexer(Indexer):

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
        balance_every: int = 0,  # balance every # new added keys (0 = never)
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
        self._faiss_l1_index: Optional[Any] = None
        self._faiss_l1_index_ntotal: int = 0
        self.balance_every = balance_every
        self.update_count = 0

    @torch.no_grad()
    def build(self, keys: torch.Tensor):
        self.keys = keys.to(self.device)
        self.num_keys, self.dim = keys.shape

        level_size = self.num_keys
        level_idx = 0
        ball_centers = self.keys

        # first level (all the points)
        print(f"Building level {level_idx}...") if self.verbose else None

        level_0 = CPUIndexer.Level(
            level_idx=level_idx,
            ball_centers=ball_centers.contiguous(),
            ball_radii=torch.zeros(self.num_keys, device=self.device),
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
                ball_centers,
                ball_radii,
                assignments,  # assignments[child] = parent
            ) = self._k_means(ball_centers, level_size)

            # child level: the level we are clustering to build its parent
            child_level = self.levels[level_idx]

            # STEP 2: fill child2parent mapping in child level
            child_level.child2parent = assignments.contiguous()

            # STEP 3: approximate small enclosing balls
            ball_radii = self._refine_ball_radii(child_level, assignments, ball_centers)

            # STEP 4: assign child <-> parent for previous level
            child_level.num_parents = level_size

            # Create new level
            parent_level = CPUIndexer.Level(
                level_idx=level_idx + 1,
                ball_centers=ball_centers.contiguous(),
                ball_radii=ball_radii.contiguous(),
                size=len(ball_centers),
                # child <-> parent
                child2parent=None,  # to be filled later
                num_parents=None,
            )
            self.levels.append(parent_level)

            level_idx += 1

        # add buffers (ASSUMED STATIC LEVELS)
        if len(self.levels) >= 2:
            self._build_l1_faiss(self.levels[1])

        self.preallocate_search_buffers()

        return self

    @torch.no_grad()
    def update(self, new_keys: torch.Tensor, fast: bool = True):
        if new_keys.numel() == 0:
            return self
        if new_keys.ndim != 2:
            raise ValueError(f"new_keys must be (n,d), got {tuple(new_keys.shape)}")
        if not self.levels:
            raise RuntimeError("CPUIndexer.update() called before build()")

        new_keys_cpu = new_keys.to(self.device).contiguous()

        # 1) Append to base storage (keep self.keys and level-0 centers consistent)
        base_level = self.levels[0]

        self.keys = torch.cat([self.keys, new_keys_cpu], dim=0).contiguous()

        self.num_keys = self.keys.shape[0]
        base_level.ball_centers = self.keys
        base_level.size = self.num_keys

        # Leaf radii are always 0 (leaf balls are points)
        base_level.ball_radii = torch.zeros(
            (self.num_keys,), device=self.device, dtype=new_keys_cpu.dtype
        )

        # If there is no parent level, nothing to connect to.
        if len(self.levels) < 2:
            return self

        # 2) Assign each new key to the closest parent (level 1) using FAISS
        parent_level = self.levels[1]
        parent_centers = parent_level.ball_centers
        if parent_centers.ndim != 2 or parent_centers.shape[1] != self.dim:
            raise RuntimeError(
                f"Invalid parent centers shape: {tuple(parent_centers.shape)} (expected (*,{self.dim}))"
            )

        x_np = new_keys_cpu.detach().cpu().float().contiguous().numpy()
        d2_np, idx_np = self._faiss_l1_index.search(x_np, 1)  # (m,1), (m,1)

        closest_parent_idx = torch.from_numpy(idx_np.reshape(-1)).to(
            device=self.device, dtype=torch.long
        )
        base_level.child2parent = torch.cat(
            [base_level.child2parent, closest_parent_idx], dim=0
        )

        # 3) Incrementally update parent radii if new keys extend the enclosing ball.
        # parent ball radius is max(||child_center - parent_center|| + child_radius).
        # leaf child_radius is 0, so this is just the max distance.
        dist = (
            torch.from_numpy(d2_np.reshape(-1))
            .to(device=self.device, dtype=torch.float32)
            .clamp_min_(0.0)
            .sqrt_()
        )

        pr = parent_level.ball_radii
        if pr.shape[0] != parent_level.size:
            raise RuntimeError(
                f"Parent radii shape mismatch: {tuple(pr.shape)} vs size {parent_level.size}"
            )

        # Fast path (PyTorch scatter_reduce_)
        if fast:
            upd = torch.full(
                (parent_level.size,),
                float("-inf"),
                device=self.device,
                dtype=dist.dtype,
            )
            upd.scatter_reduce_(
                0, closest_parent_idx, dist, reduce="amax", include_self=True
            )
            upd = torch.where(torch.isfinite(upd), upd, torch.zeros_like(upd))
            parent_level.ball_radii = torch.maximum(pr, upd.to(dtype=pr.dtype))
        else:
            for p in closest_parent_idx.unique().tolist():
                mask = closest_parent_idx == p
                m = dist[mask].max()
                if m > parent_level.ball_radii[p]:
                    parent_level.ball_radii[p] = m.to(
                        dtype=parent_level.ball_radii.dtype
                    )

        # 4) Rebalance if needed
        self.update_count += new_keys_cpu.shape[0]
        if self.balance_every > 0 and self.update_count >= self.balance_every:
            self._balance(use_faiss=True)
            self.update_count = 0

        return self

    def _balance(self, use_faiss: bool = True):
        """Rebalance the hierarchy by splitting oversized level-1 parents.

        Policy:
        - If a level-1 parent has > 2*branching_factor level-0 children, split its children
          into two groups (2-means), keep one centroid in-place, append the other as a new parent.
        - Update level-0 child2parent for the split children.
                - Assign both affected parents (the overwritten one and the appended one) to the closest
                    grandparent (level 2) if present, and update the affected level-2 radii.

        Notes:
        - Radii are only ever increased (never decreased), so this is safe for pruning.
        - This intentionally does NOT propagate radius updates to levels >= 3.
        """
        if len(self.levels) < 2:
            return

        level0 = self.levels[0]
        level1 = self.levels[1]

        bf = int(self.branching_factor)

        any_change = False

        # Single-pass: split each currently-oversized parent at most once.
        counts = torch.bincount(level0.child2parent, minlength=level1.size)
        large_parents = (counts > 2 * bf).nonzero(as_tuple=True)[0]
        if large_parents.numel() == 0:
            return

        if self.verbose:
            print(f"Rebalancing: splitting {large_parents.numel()} oversized parents")

        for parent_idx in large_parents.tolist():
            child_idx = (level0.child2parent == parent_idx).nonzero(as_tuple=True)[0]
            if child_idx.numel() <= 2 * bf:
                continue
            if child_idx.numel() < 2:
                continue

            pts = level0.ball_centers.index_select(0, child_idx)
            pts_f = pts.detach().cpu().float().contiguous()
            pts_np = pts_f.numpy()

            # --- split with 2-means (faiss) ---
            if use_faiss:
                kmeans = faiss.Kmeans(
                    d=self.dim,
                    k=2,
                    niter=1,
                    nredo=1,
                    verbose=False,
                    gpu=False,
                )
                kmeans.train(pts_np)
                d2_np, a_np = kmeans.index.search(pts_np, 1)
                assign = torch.from_numpy(a_np.reshape(-1)).to(
                    device=self.device, dtype=torch.long
                )
                centroids = torch.from_numpy(kmeans.centroids).to(
                    device=self.device, dtype=level1.ball_centers.dtype
                )
            else:
                # Fallback: random half split
                perm = torch.randperm(child_idx.numel(), device=self.device)
                assign = torch.zeros(
                    child_idx.numel(), device=self.device, dtype=torch.long
                )
                assign[perm[child_idx.numel() // 2 :]] = 1
                c0 = pts.index_select(0, perm[: child_idx.numel() // 2]).mean(dim=0)
                c1 = pts.index_select(0, perm[child_idx.numel() // 2 :]).mean(dim=0)
                centroids = torch.stack([c0, c1], dim=0).to(
                    dtype=level1.ball_centers.dtype
                )

            # Compute radii for the two clusters (leaf children have radius 0)
            pts_dev = pts.to(self.device)
            dists = torch.norm(pts_dev - centroids.index_select(0, assign), dim=1)
            r0 = dists[assign == 0].max() if (assign == 0).any() else torch.tensor(0.0)
            r1 = dists[assign == 1].max() if (assign == 1).any() else torch.tensor(0.0)

            # Overwrite existing parent with centroid 0, append centroid 1
            level1.ball_centers[parent_idx] = centroids[0]
            level1.ball_radii[parent_idx] = r0.to(dtype=level1.ball_radii.dtype)

            new_parent_idx = level1.size
            level1.ball_centers = torch.cat(
                [level1.ball_centers, centroids[1:2].contiguous()], dim=0
            )
            level1.ball_radii = torch.cat(
                [
                    level1.ball_radii,
                    r1.to(device=self.device, dtype=level1.ball_radii.dtype).view(1),
                ],
                dim=0,
            )
            level1.size = level1.ball_centers.shape[0]

            # Keep level-1 child2parent length consistent with level-1 size (if it exists)
            if (
                level1.child2parent is not None
                and level1.child2parent.numel() != level1.size
            ):
                if level1.child2parent.numel() != level1.size - 1:
                    raise RuntimeError(
                        "Level-1 child2parent is out of sync with level-1 size. "
                        f"Expected size-1, got {level1.child2parent.numel()} vs {level1.size}."
                    )
                level1.child2parent = torch.cat(
                    [
                        level1.child2parent,
                        torch.zeros(
                            (1,),
                            device=self.device,
                            dtype=level1.child2parent.dtype,
                        ),
                    ],
                    dim=0,
                )

            # Reassign children: cluster 1 -> new parent index
            moved_child = child_idx[assign == 1]
            if moved_child.numel() > 0:
                level0.child2parent.index_fill_(0, moved_child, int(new_parent_idx))

            any_change = True

            # Update bookkeeping for L0 after parent count changes
            level0.num_parents = level1.size
            level0.parent_mask_buf = torch.empty(
                level0.num_parents, dtype=torch.bool, device=self.device
            )

            # Update level1->level2 assignment and radii (no propagation above level 2)
            if len(self.levels) >= 3:
                level2 = self.levels[2]
                if level1.child2parent is None:
                    raise RuntimeError(
                        "Level-1 child2parent is None; cannot rebalance with 3+ levels."
                    )

                # Build temporary FAISS index for level-2 centers
                gp_index = faiss.IndexFlatL2(self.dim)
                gp_np = level2.ball_centers.detach().cpu().float().contiguous().numpy()
                gp_index.add(gp_np)

                affected = torch.tensor(
                    [parent_idx, new_parent_idx], device=self.device, dtype=torch.long
                )
                aff_np = (
                    level1.ball_centers.index_select(0, affected)
                    .detach()
                    .cpu()
                    .float()
                    .contiguous()
                    .numpy()
                )
                d2g_np, gidx_np = gp_index.search(aff_np, 1)
                gidx = torch.from_numpy(gidx_np.reshape(-1)).to(
                    device=self.device, dtype=torch.long
                )
                level1.child2parent.index_copy_(0, affected, gidx)

                # Update level-2 radii for affected grandparents
                dist_gp = (
                    torch.from_numpy(d2g_np.reshape(-1))
                    .to(device=self.device, dtype=torch.float32)
                    .clamp_min_(0.0)
                    .sqrt_()
                )
                total = dist_gp + level1.ball_radii.index_select(0, affected).float()

                upd = torch.full(
                    (level2.size,),
                    float("-inf"),
                    device=self.device,
                    dtype=total.dtype,
                )
                upd.scatter_reduce_(0, gidx, total, reduce="amax", include_self=True)
                upd = torch.where(torch.isfinite(upd), upd, torch.zeros_like(upd))
                level2.ball_radii = torch.maximum(
                    level2.ball_radii, upd.to(level2.ball_radii.dtype)
                )

        if any_change:
            self._build_l1_faiss(level1)

    def _build_l1_faiss(self, level1):
        parents_np = level1.ball_centers.contiguous().numpy()
        index = faiss.IndexFlatL2(self.dim)
        index.add(parents_np)
        self._faiss_l1_index = index
        self._faiss_l1_index_ntotal = parents_np.shape[0]

    def preallocate_search_buffers(self):
        for lvl in self.levels:
            if lvl.num_parents is not None:
                lvl.parent_mask_buf = torch.empty(
                    lvl.num_parents, dtype=torch.bool, device=self.device
                )

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
            gpu=True,  # self.device.type == "cuda",  # TODO: gpu support
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


class CUDAIndexer(Indexer):

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
        # make sure keys are on GPU
        keys = keys.to("cuda")
        self.dim = keys.shape[1]

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
            (self.children.shape[0],), device="cuda", dtype=torch.float32
        )

        # Initialize update_v2 cache state.
        self._init_update_v2_state()

        return self

    def _init_update_v2_state(self) -> None:
        if self.parents is None or self.children is None:
            self._child_counts = None
            self._parent_valid = None
            return

        bf = int(self.branching_factor)
        d = int(self.parents.shape[1])
        P = int(self.parents.shape[0])

        # Parent validity (padded parents are all pad_value).
        if self.pad_value is None:
            self._parent_valid = torch.ones(
                (P,), device=self.parents.device, dtype=torch.bool
            )
        else:
            self._parent_valid = ~(
                torch.all(self.parents == float(self.pad_value), dim=-1)
            )

        # Child counts (assumes contiguous fill).
        children3 = self.children.view(P, bf, d)
        if self.pad_value is None:
            valid_child = torch.ones(
                (P, bf), device=self.children.device, dtype=torch.bool
            )
        else:
            valid_child = ~torch.all(children3 == float(self.pad_value), dim=-1)
        self._child_counts = valid_child.sum(dim=1).to(torch.int32).contiguous()

    def _ensure_update_v2_state(self) -> None:
        if self._child_counts is None or self._parent_valid is None:
            self._init_update_v2_state()

    def _compute_parent_radii_from_layout(self) -> torch.Tensor:
        m, d = self.parents.shape

        parents_f = self.parents.float().contiguous()
        children_f = (
            self.children.float().contiguous().view(m, self.branching_factor, d)
        )
        diffs = children_f - parents_f[:, None, :]
        dists = torch.linalg.norm(diffs, dim=-1)  # (m,bf)

        if self.pad_value is not None:
            valid = ~torch.all(children_f == float(self.pad_value), dim=-1)  # (m,bf)
            dists = torch.where(
                valid, dists, torch.tensor(float("-inf"), device=dists.device)
            )

        radii = torch.max(dists, dim=1).values
        radii = torch.where(torch.isfinite(radii), radii, torch.zeros_like(radii))

        assert radii.is_cuda
        return radii

    def _compute_grandparent_radii_from_layout(self) -> torch.Tensor:
        g, d = self.grand_parents.shape

        gp_f = self.grand_parents.float().contiguous()
        parents_f = self.parents.float().contiguous().view(g, self.branching_factor, d)
        pr = self.parent_radii.float().contiguous().view(g, self.branching_factor)

        dists = torch.linalg.norm(parents_f - gp_f[:, None, :], dim=-1)  # (g,bf)
        totals = dists + pr

        if self.pad_value is not None:
            valid = ~torch.all(parents_f == float(self.pad_value), dim=-1)
            totals = torch.where(
                valid,
                totals,
                torch.tensor(float("-inf"), device=totals.device),
            )

        radii = torch.max(totals, dim=1).values
        radii = torch.where(torch.isfinite(radii), radii, torch.zeros_like(radii))

        assert radii.is_cuda
        return radii

    def _build_faiss_kmeans_bf_level(
        self,
        x: torch.Tensor,  # (N, d) float32 CUDA
        bf: int,
        *,
        seed: int = 1234,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generic 'one level' builder:
        - returns centroids C (K, d)
        - returns selected indices S (K, bf) into x, or -1 if padded
        """
        assert x.is_cuda and x.dtype == torch.float32 and x.ndim == 2
        device = x.device
        N, d = x.shape
        K = max(1, N // bf)

        # --- KMeans (faiss; may use GPU if available) ---
        x_cpu = x.contiguous().cpu().numpy()
        kmeans = faiss.Kmeans(
            d=d,
            k=K,
            niter=self.max_iterations,
            nredo=1,
            verbose=self.verbose,
            seed=seed,
            gpu=False,
        )
        kmeans.train(x_cpu)

        C = (
            torch.from_numpy(kmeans.centroids)
            .to(device=device, dtype=torch.float32)
            .contiguous()
        )

        # --- assign points -> nearest centroid ---
        index = faiss.IndexFlatL2(d)
        index.add(kmeans.centroids.astype("float32", copy=False))
        dist1_np, idx1_np = index.search(x_cpu, 1)  # (N,1)
        dist1 = torch.from_numpy(dist1_np.reshape(-1)).to(
            device=device, dtype=torch.float32
        )
        idx1 = torch.from_numpy(idx1_np.reshape(-1)).to(
            device=device, dtype=torch.int64
        )

        # --- choose up to bf closest per centroid (true lexicographic by (centroid, dist)) ---
        dist_order = torch.argsort(dist1)
        dist_rank = torch.empty_like(dist_order)
        dist_rank[dist_order] = torch.arange(N, device=device, dtype=dist_order.dtype)

        composite = idx1 * (N + 1) + dist_rank.to(torch.int64)
        order = torch.argsort(composite)

        idx_grp = idx1[order]
        pts_grp = order

        counts = torch.bincount(idx_grp, minlength=K)
        offsets = torch.zeros(K + 1, device=device, dtype=torch.int64)
        offsets[1:] = torch.cumsum(counts.to(torch.int64), dim=0)

        selected = torch.full((K, bf), -1, device=device, dtype=torch.int64)
        used = torch.zeros(N, device=device, dtype=torch.bool)

        for c in range(K):
            s = int(offsets[c].item())
            e = int(offsets[c + 1].item())
            if s == e:
                continue
            take = min(bf, e - s)
            take_pts = pts_grp[s : s + take]
            selected[c, :take] = take_pts
            used[take_pts] = True

        # --- fill deficits from leftovers (heuristic) ---
        if (selected == -1).any():
            leftovers = torch.nonzero(~used, as_tuple=False).view(-1)
            if leftovers.numel() > 0:
                k_try = min(K, 8)
                leftovers_cpu = leftovers.cpu().numpy()
                _, idxk_np = index.search(x_cpu[leftovers_cpu], k_try)  # (L,k_try)
                idxk = torch.from_numpy(idxk_np).to(device=device, dtype=torch.int64)

                next_slot = (selected != -1).sum(dim=1)

                for j in range(leftovers.numel()):
                    p = int(leftovers[j].item())
                    for t in range(k_try):
                        c = int(idxk[j, t].item())
                        s = int(next_slot[c].item())
                        if s < bf:
                            selected[c, s] = p
                            next_slot[c] += 1
                            break

        return C, selected

    def _build_parents_children_from_keys(
        self,
        keys: torch.Tensor,  # (n,d) CUDA float/half ok
        bf: int,
        *,
        seed: int = 1234,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build one level from raw keys using ONLY _build_faiss_kmeans_bf_level.

        Returns:
        parents  : (m, d) float32 CUDA, where m = max(1, n // bf)
        children : (m*bf, d) float32 CUDA, where children[i*bf:(i+1)*bf] are parent i's children.
                    If a slot is unfilled (selected == -1), it is padded with pad_value.

        Ordering:
        - For each parent i, children[i*bf:(i+1)*bf] follow the selection order from _build_faiss_kmeans_bf_level
            (centroid-grouped, closest-first within group).
        """
        if keys.ndim != 2:
            raise ValueError(f"keys must be (n,d), got {tuple(keys.shape)}")
        if not keys.is_cuda:
            raise ValueError("keys must be on CUDA")
        if bf <= 0:
            raise ValueError("bf must be positive")

        x = keys.detach()
        if x.dtype != torch.float32:
            x = x.float()
        x = x.contiguous()

        device = x.device
        n, d = x.shape
        m = max(1, n // bf)

        # Use the provided level builder directly on keys
        parents, selected = self._build_faiss_kmeans_bf_level(x, bf, seed=seed)
        # parents: (m,d) ; selected: (m,bf) indices into x or -1

        # Materialize children in the requested layout
        children = torch.full(
            (m * bf, d), self.pad_value, device=device, dtype=torch.float32
        )

        sel_flat = selected.view(-1)  # (m*bf,)
        valid = sel_flat != -1
        if valid.any():
            children[valid] = x[sel_flat[valid]]

        assert parents.is_cuda and children.is_cuda
        return parents, children

    def _build_grandparents_parents_children_from_keys(
        self,
        keys: torch.Tensor,  # (n,d) CUDA float/half ok
        bf: int,
        *,
        seed: int = 1234,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build two levels from raw keys using ONLY _build_faiss_kmeans_bf_level.

        Returns:
        grand_parents     : (g, d) float32 CUDA, g = max(1, m // bf) where m=max(1, n // bf)
        parents_reordered : (g*bf, d) float32 CUDA, contiguous bf-block per grandparent
        children_reordered: (g*bf*bf, d) float32 CUDA, contiguous bf-block per parent (in reordered parent order)

        Invariants:
        - parents_reordered[i*bf:(i+1)*bf] are the parents of grand_parent i (padding possible)
        - children_reordered[p*bf:(p+1)*bf] are children of parent p in parents_reordered
        - ordering is consistent with the selection order produced by _build_faiss_kmeans_bf_level
        """
        # --- first level: keys -> parents, children ---
        parents, children = self._build_parents_children_from_keys(keys, bf, seed=seed)
        device = parents.device
        m, d = parents.shape

        # --- second level: parents -> grand_parents, select parent-ids per grandparent ---
        gp, sel_par = self._build_faiss_kmeans_bf_level(
            parents.contiguous(), bf, seed=seed + 1
        )
        g = gp.shape[0]

        # Reorder parents into contiguous bf-blocks per grandparent
        sel_flat = sel_par.view(-1)  # (g*bf,)
        valid = sel_flat != -1

        parents_reordered = torch.full(
            (g * bf, d), self.pad_value, device=device, dtype=torch.float32
        )
        if valid.any():
            parents_reordered[valid] = parents[sel_flat[valid]]

        # Reorder children by permuting whole parent child-blocks to match parents_reordered order
        # children is aligned with original parents: children.view(m, bf, d)
        child_blocks = children.view(m, bf, d)  # (m,bf,d)

        children_reordered_blocks = torch.full(
            (g * bf, bf, d), self.pad_value, device=device, dtype=torch.float32
        )
        if valid.any():
            children_reordered_blocks[valid] = child_blocks[sel_flat[valid]]

        children_reordered = children_reordered_blocks.view(g * bf * bf, d)

        assert gp.is_cuda and parents_reordered.is_cuda and children_reordered.is_cuda
        return gp, parents_reordered, children_reordered

    def _nearest_l2(
        self,
        x: torch.Tensor,
        centers: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor] = None,
        x_block: int = 2048,
        c_block: int = 8192,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (nearest_center_index, nearest_squared_distance) on CUDA.

        Computes squared L2 distance using a GEMM-based formula and processes in blocks
        to avoid materializing the full (N,M) distance matrix.

        If valid_mask is provided, only centers[valid_mask] are considered and returned
        indices are in the original centers index space.
        """
        if x.numel() == 0:
            return (
                torch.empty((0,), device=x.device, dtype=torch.int64),
                torch.empty((0,), device=x.device, dtype=torch.float32),
            )

        if valid_mask is not None:
            valid_idx = torch.nonzero(valid_mask, as_tuple=False).view(-1)
            if valid_idx.numel() == 0:
                raise RuntimeError("No valid centers to search")
            c = centers.index_select(0, valid_idx)
        else:
            valid_idx = None
            c = centers

        x = x.float().contiguous()
        c = c.float().contiguous()

        N = x.shape[0]
        M = c.shape[0]

        c_norm = (c * c).sum(dim=1)  # (M,)

        best_idx = torch.empty((N,), device=x.device, dtype=torch.int64)
        best_d2 = torch.empty((N,), device=x.device, dtype=torch.float32)

        for xs in range(0, N, x_block):
            xe = min(N, xs + x_block)
            xb = x[xs:xe]
            xb_norm = (xb * xb).sum(dim=1, keepdim=True)  # (B,1)

            bd2 = torch.full((xb.shape[0],), float("inf"), device=x.device)
            bidx = torch.full((xb.shape[0],), -1, device=x.device, dtype=torch.int64)

            for cs in range(0, M, c_block):
                ce = min(M, cs + c_block)
                cb = c[cs:ce]
                cb_norm = c_norm[cs:ce]
                # squared L2: ||x||^2 + ||c||^2 - 2 x·c
                d2 = xb_norm + cb_norm[None, :] - 2.0 * (xb @ cb.t())
                d2 = torch.clamp_min(d2, 0.0)

                local_d2, local_j = torch.min(d2, dim=1)
                better = local_d2 < bd2
                bd2 = torch.where(better, local_d2, bd2)
                bidx = torch.where(better, local_j.to(torch.int64) + cs, bidx)

            best_idx[xs:xe] = bidx
            best_d2[xs:xe] = bd2

        if valid_idx is not None:
            best_idx = valid_idx.index_select(0, best_idx)

        return best_idx, best_d2

    def _fill_existing_children(
        self,
        *,
        x: torch.Tensor,
        parent_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fill padded child slots for assigned parents.

        Returns:
            placed_mask (len(x),) bool
            placed_child_flat_idx (num_placed,) int64 indices into children flat rows
        """
        P = self.parents.shape[0]
        d = self.parents.shape[1]
        bf = self.branching_factor
        device = x.device
        children3 = self.children.view(P, bf, d)
        empty = torch.all(children3 == self.pad_value, dim=-1)  # (P,bf)
        avail = empty.sum(dim=1).to(torch.int64)  # (P,)

        # Prepare sorted empty slots per parent (bf as sentinel)
        slot_ids = torch.arange(bf, device=device, dtype=torch.int64)
        slot_mat = slot_ids.view(1, bf).expand(P, bf)
        empty_first = torch.where(empty, slot_mat, torch.full_like(slot_mat, bf))
        empty_sorted = torch.sort(empty_first, dim=1).values  # (P,bf)

        # Sort keys by parent for group-wise ranking
        order = torch.argsort(parent_idx)
        p_sorted = parent_idx[order]
        x_sorted = x[order]

        counts = torch.bincount(parent_idx, minlength=P).to(torch.int64)
        prefix = torch.cumsum(counts, dim=0)
        start = prefix.index_select(0, p_sorted) - counts.index_select(0, p_sorted)
        rank = torch.arange(x.shape[0], device=device, dtype=torch.int64) - start

        can_place = rank < avail.index_select(0, p_sorted)
        if not can_place.any():
            placed_mask = torch.zeros((x.shape[0],), device=device, dtype=torch.bool)
            return placed_mask, torch.empty((0,), device=device, dtype=torch.int64)

        p_place = p_sorted[can_place]
        x_place = x_sorted[can_place]
        r_place = rank[can_place]
        slot = empty_sorted[p_place, r_place]
        child_flat = p_place * bf + slot

        # Write children in-place
        self.children.index_copy_(0, child_flat, x_place)

        placed_mask = torch.zeros((x.shape[0],), device=device, dtype=torch.bool)
        placed_mask.index_fill_(0, order[can_place], True)
        return placed_mask, child_flat

    def _update_parent_radii_for_inserted(
        self,
        *,
        inserted_keys: torch.Tensor,
        inserted_parent_idx: torch.Tensor,
    ) -> None:
        if inserted_keys.numel() == 0:
            return
        p = inserted_parent_idx
        dist = torch.linalg.norm(
            inserted_keys - self.parents.index_select(0, p), dim=1
        ).float()
        upd = torch.full(
            (self.parents.shape[0],),
            float("-inf"),
            device=inserted_keys.device,
            dtype=torch.float32,
        )
        upd.scatter_reduce_(0, p, dist, reduce="amax", include_self=True)
        upd = torch.where(torch.isfinite(upd), upd, torch.zeros_like(upd))
        self.parent_radii = torch.maximum(self.parent_radii.float(), upd).to(
            dtype=self.parent_radii.dtype
        )

    def _build_new_parents_from_overflow(
        self, x_over: torch.Tensor, *, seed: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create new parents for overflow keys.

        Returns:
            new_parents (K,d)
            new_children_flat (K*bf,d)
            new_parent_radii (K,)
        """
        device = x_over.device
        bf = self.branching_factor
        if x_over.numel() == 0:
            return (
                torch.empty((0, self.dim), device=device, dtype=torch.float32),
                torch.empty((0, self.dim), device=device, dtype=torch.float32),
                torch.empty((0,), device=device, dtype=torch.float32),
            )

        torch.manual_seed(seed)
        m_over = x_over.shape[0]
        K = int((m_over + bf - 1) // bf)  # ceil
        K = max(1, K)

        perm = torch.randperm(m_over, device=device)
        choose = perm[:K]
        new_parents = x_over.index_select(0, choose).contiguous()  # (K,d)

        # Initial nearest assignment to these new parents
        assign, d2 = self._nearest_l2(x_over, new_parents, valid_mask=None)
        # assign in [0,K)

        # Sort by (assign, dist)
        order = torch.argsort(assign * (m_over + 1) + torch.argsort(d2))
        a_sorted = assign[order]
        x_sorted = x_over[order]
        d2_sorted = d2[order]

        counts = torch.bincount(a_sorted, minlength=K).to(torch.int64)
        prefix = torch.cumsum(counts, dim=0)
        start = prefix.index_select(0, a_sorted) - counts.index_select(0, a_sorted)
        rank = torch.arange(m_over, device=device, dtype=torch.int64) - start

        # First pass: fill up to bf closest per new parent
        placed = rank < bf
        children_new = torch.full(
            (K * bf, self.dim), self.pad_value, device=device, dtype=torch.float32
        )
        if placed.any():
            a_p = a_sorted[placed]
            r_p = rank[placed]
            idx = a_p * bf + r_p
            children_new.index_copy_(0, idx, x_sorted[placed])

        # Second pass: place leftovers into any remaining slots
        if (~placed).any():
            leftovers = x_sorted[~placed]
            d2_left = d2_sorted[~placed]
            # recompute full distance matrix for leftovers to all new parents (K is small)
            # use fast formula
            lp = leftovers
            lp_norm = (lp * lp).sum(dim=1, keepdim=True)
            np = new_parents
            np_norm = (np * np).sum(dim=1)
            d2mat = lp_norm + np_norm[None, :] - 2.0 * (lp @ np.t())
            d2mat = torch.clamp_min(d2mat, 0.0)

            # capacity tracking
            next_slot = (
                (children_new.view(K, bf, self.dim) != self.pad_value)
                .any(dim=-1)
                .sum(dim=1)
                .to(torch.int64)
            )
            cap = next_slot < bf
            # Iteratively assign (leftovers count expected small)
            for j in range(leftovers.shape[0]):
                cap = next_slot < bf
                if not bool(cap.any().item()):
                    raise RuntimeError(
                        "Invariant broken: no capacity to place leftovers"
                    )
                d2j = d2mat[j]
                d2j = torch.where(cap, d2j, torch.tensor(float("inf"), device=device))
                c = int(torch.argmin(d2j).item())
                s = int(next_slot[c].item())
                children_new[c * bf + s] = leftovers[j]
                next_slot[c] += 1

        # Compute radii for new parents
        c3 = children_new.view(K, bf, self.dim)
        valid = ~torch.all(c3 == self.pad_value, dim=-1)
        dists = torch.linalg.norm(c3 - new_parents[:, None, :], dim=-1)
        dists = torch.where(valid, dists, torch.tensor(float("-inf"), device=device))
        radii = torch.max(dists, dim=1).values
        radii = torch.where(torch.isfinite(radii), radii, torch.zeros_like(radii))

        return new_parents, children_new, radii

    def _build_new_parents_from_overflow_v2(
        self,
        x_over: torch.Tensor,
        *,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create new parents for overflow keys (v2).

        v2 removes the Python-side leftover placement loop (and associated DtoH sync)
        by placing any leftovers into globally available free slots.

        Returns:
            new_parents (K,d)
            new_children_flat (K*bf,d)
            new_parent_radii (K,)
        """
        device = x_over.device
        bf = int(self.branching_factor)
        if x_over.numel() == 0:
            return (
                torch.empty((0, self.dim), device=device, dtype=torch.float32),
                torch.empty((0, self.dim), device=device, dtype=torch.float32),
                torch.empty((0,), device=device, dtype=torch.float32),
            )

        torch.manual_seed(seed)
        m_over = int(x_over.shape[0])
        K = int((m_over + bf - 1) // bf)
        K = max(1, K)

        perm = torch.randperm(m_over, device=device)
        choose = perm[:K]
        new_parents = x_over.index_select(0, choose).contiguous()  # (K,d)

        # Initial nearest assignment to these new parents
        assign, d2 = nearest_l2_triton(x_over, new_parents, valid_mask=None)

        # Group by assign (no secondary sort by distance; avoids extra argsort)
        order = torch.argsort(assign)
        a_sorted = assign[order]
        x_sorted = x_over[order]

        counts = torch.bincount(a_sorted, minlength=K).to(torch.int64)
        prefix = torch.cumsum(counts, dim=0)
        start = prefix.index_select(0, a_sorted) - counts.index_select(0, a_sorted)
        rank = torch.arange(m_over, device=device, dtype=torch.int64) - start

        placed = rank < bf
        children_new = torch.full(
            (K * bf, self.dim), self.pad_value, device=device, dtype=torch.float32
        )

        if placed.any():
            a_p = a_sorted[placed]
            r_p = rank[placed]
            idx = a_p * bf + r_p
            children_new.index_copy_(0, idx, x_sorted[placed])

        # Vectorized leftover placement: assign to any globally free slots.
        if (~placed).any():
            leftovers = x_sorted[~placed]
            empty_slots = torch.all(children_new == float(self.pad_value), dim=-1)
            free = torch.nonzero(empty_slots, as_tuple=False).view(-1)
            if free.numel() < leftovers.shape[0]:
                raise RuntimeError("Invariant broken: insufficient free slots")
            children_new.index_copy_(0, free[: leftovers.shape[0]], leftovers)

        # Compute radii for new parents
        c3 = children_new.view(K, bf, self.dim)
        valid = ~torch.all(c3 == float(self.pad_value), dim=-1)
        dists = torch.linalg.norm(c3 - new_parents[:, None, :], dim=-1)
        dists = torch.where(valid, dists, torch.tensor(float("-inf"), device=device))
        radii = torch.max(dists, dim=1).values
        radii = torch.where(torch.isfinite(radii), radii, torch.zeros_like(radii))

        return new_parents, children_new, radii

    @torch.no_grad()
    def update(
        self,
        new_keys: torch.Tensor,
    ):
        """Faster CUDA update path.

        Keeps `update()` intact; this method is for A/B comparisons.

        Current scope:
        - Uses Triton nearest-center (two-stage) when enabled.
        - Uses Triton atomic child-slot filling based on per-parent counters.
        - Uses Triton atomic parent-radii updates when possible.
        - Uses v2 overflow parent creation (no Python leftover loop).
        """
        if new_keys.numel() == 0:
            return self
        if self.parents is None or self.children is None or self.parent_radii is None:
            raise RuntimeError("CUDAIndexer.update_v2() called before build()")
        if new_keys.ndim != 2:
            raise ValueError(f"new_keys must be (n,d), got {tuple(new_keys.shape)}")

        if self.pad_value is None:
            raise ValueError(
                "CUDAIndexer.update_v2() requires pad_value to detect empty slots"
            )

        self._ensure_update_v2_state()

        new_keys = new_keys.to("cuda").contiguous()
        bf = int(self.branching_factor)
        device = new_keys.device

        # 1) Assign each new key to nearest valid parent
        valid_parent = self._parent_valid

        nearest_parent, _nearest_d2 = nearest_l2_triton(
            new_keys, self.parents, valid_mask=valid_parent
        )

        # 2) Try to fill child slots under those parents using atomic counters
        placed_mask: torch.Tensor

        placed_mask, _placed_flat = fill_existing_children_atomic(
            x=new_keys,
            parent_idx=nearest_parent,
            child_counts=self._child_counts,
            children_flat=self.children,
            bf=bf,
        )

        # 3) Update parent radii for placed keys
        if placed_mask.any():
            inserted_keys = new_keys[placed_mask]
            inserted_parents = nearest_parent[placed_mask]

            did_atomic = False

            if self.parent_radii.dtype == torch.float32:
                update_parent_radii_atomic(
                    inserted_keys=inserted_keys,
                    inserted_parent_idx=inserted_parents,
                    parents=self.parents,
                    parent_radii=self.parent_radii,
                )
                did_atomic = True

            # THREE_LEVELS: update affected grandparent radii (monotonic increase)
            if self.depth == CUDAIndexer.DEPTH.THREE_LEVELS:
                if self.grand_parents is None or self.grand_parent_radii is None:
                    raise RuntimeError("THREE_LEVELS build state incomplete")
                g = self.grand_parents.shape[0]
                gp_idx = (inserted_parents // bf).to(torch.int64)
                gp_centers = self.grand_parents.index_select(0, gp_idx)
                dist_gp = torch.linalg.norm(
                    self.parents.index_select(0, inserted_parents) - gp_centers, dim=1
                ).float()
                total = (
                    dist_gp
                    + self.parent_radii.index_select(0, inserted_parents).float()
                )

                upd_gp = torch.full(
                    (g,), float("-inf"), device=device, dtype=torch.float32
                )
                upd_gp.scatter_reduce_(
                    0, gp_idx, total, reduce="amax", include_self=True
                )
                upd_gp = torch.where(
                    torch.isfinite(upd_gp), upd_gp, torch.zeros_like(upd_gp)
                )
                self.grand_parent_radii = torch.maximum(
                    self.grand_parent_radii.float(), upd_gp
                ).to(dtype=self.grand_parent_radii.dtype)

        # 4) Overflow keys => create new parents and append
        overflow = new_keys[~placed_mask]
        if overflow.numel() == 0:
            return self

        new_parents, new_children_flat, new_parent_radii = (
            self._build_new_parents_from_overflow_v2(overflow, seed=1234)
        )
        if new_parents.numel() == 0:
            return self

        K = int(new_parents.shape[0])
        d = int(new_parents.shape[1])

        if self.depth == CUDAIndexer.DEPTH.TWO_LEVELS:
            self.parents = torch.cat([self.parents, new_parents], dim=0).contiguous()
            self.parent_radii = torch.cat(
                [self.parent_radii, new_parent_radii.to(self.parent_radii.dtype)], dim=0
            ).contiguous()
            self.children = torch.cat(
                [self.children, new_children_flat], dim=0
            ).contiguous()

            # Update v2 cache: appended parents are valid
            self._parent_valid = torch.cat(
                [self._parent_valid, torch.ones((K,), device=device, dtype=torch.bool)],
                dim=0,
            ).contiguous()
            children_blocks = new_children_flat.view(K, bf, d)
            valid_child = ~torch.all(children_blocks == float(self.pad_value), dim=-1)
            new_counts = valid_child.sum(dim=1).to(torch.int32)
            self._child_counts = torch.cat(
                [self._child_counts, new_counts], dim=0
            ).contiguous()

            # Extend buffer
            if self.buffer is None:
                self.buffer = torch.zeros(
                    (self.children.shape[0],), device="cuda", dtype=torch.float32
                )
            else:
                add = new_children_flat.shape[0]
                if add > 0:
                    self.buffer = torch.cat(
                        [
                            self.buffer,
                            torch.zeros((add,), device="cuda", dtype=torch.float32),
                        ],
                        dim=0,
                    )

            return self

        # THREE_LEVELS: place new parents into existing grandparent blocks when possible,
        # otherwise append new grandparent blocks.

        G = int(self.grand_parents.shape[0])
        P_total = int(self.parents.shape[0])
        if P_total != G * bf:
            raise RuntimeError(
                f"Invalid THREE_LEVELS layout: parents={P_total} but grand_parents={G} and bf={bf}"
            )
        if int(self.children.shape[0]) != P_total * bf:
            raise RuntimeError(
                "Invalid THREE_LEVELS layout: children not aligned with parents blocks"
            )

        # Determine empty parent slots per grandparent.
        parents2 = self.parents.view(G, bf, d)
        parent_empty = torch.all(parents2 == float(self.pad_value), dim=-1)  # (G,bf)
        parent_avail = parent_empty.sum(dim=1).to(torch.int64)  # (G,)
        slot_ids = torch.arange(bf, device=device, dtype=torch.int64)
        slot_mat = slot_ids.view(1, bf).expand(G, bf)
        empty_first = torch.where(parent_empty, slot_mat, torch.full_like(slot_mat, bf))
        empty_sorted = torch.sort(empty_first, dim=1).values  # (G,bf)

        gp_near, gp_d2 = nearest_l2_triton(
            new_parents, self.grand_parents, valid_mask=None
        )

        # Group-wise place into existing grandparent blocks.
        order = torch.argsort(gp_near)
        gp_sorted = gp_near[order]

        counts = torch.bincount(gp_near, minlength=G).to(torch.int64)
        prefix = torch.cumsum(counts, dim=0)
        start = prefix.index_select(0, gp_sorted) - counts.index_select(0, gp_sorted)
        rank = torch.arange(K, device=device, dtype=torch.int64) - start

        can_place = rank < parent_avail.index_select(0, gp_sorted)

        # Indices in new_parents that go into existing blocks.
        place_src = order[can_place]
        place_gp = gp_sorted[can_place]
        place_rank = rank[can_place]
        place_slot = empty_sorted[place_gp, place_rank]
        place_parent_global = place_gp * bf + place_slot

        if place_src.numel() > 0:
            self.parents.index_copy_(
                0, place_parent_global, new_parents.index_select(0, place_src)
            )
            self.parent_radii.index_copy_(
                0,
                place_parent_global,
                new_parent_radii.index_select(0, place_src).to(self.parent_radii.dtype),
            )

            children_blocks = new_children_flat.view(K, bf, d)
            dst_children3 = self.children.view(P_total, bf, d)
            dst_children3.index_copy_(
                0, place_parent_global, children_blocks.index_select(0, place_src)
            )

            # Update grandparent radii (monotonic increase).
            gp_centers = self.grand_parents.index_select(0, place_gp)
            dist_gp = torch.linalg.norm(
                self.parents.index_select(0, place_parent_global) - gp_centers, dim=1
            ).float()
            total = (
                dist_gp + self.parent_radii.index_select(0, place_parent_global).float()
            )

            upd_gp = torch.full((G,), float("-inf"), device=device, dtype=torch.float32)
            upd_gp.scatter_reduce_(0, place_gp, total, reduce="amax", include_self=True)
            upd_gp = torch.where(
                torch.isfinite(upd_gp), upd_gp, torch.zeros_like(upd_gp)
            )
            self.grand_parent_radii = torch.maximum(
                self.grand_parent_radii.float(), upd_gp
            ).to(dtype=self.grand_parent_radii.dtype)

        # Remaining new parents require new grandparents.
        if can_place.all():
            # Ensure buffer matches children (no shape change here).
            if self.buffer is None:
                self.buffer = torch.zeros(
                    (self.children.shape[0],), device="cuda", dtype=torch.float32
                )
            elif self.buffer.shape[0] != self.children.shape[0]:
                self.buffer = torch.zeros(
                    (self.children.shape[0],), device="cuda", dtype=torch.float32
                )
            # Layout changed; refresh cached masks/counters.
            self._init_update_v2_state()
            return self

        remaining_src = order[~can_place]
        rem_parents = new_parents.index_select(0, remaining_src)
        rem_pr = new_parent_radii.index_select(0, remaining_src)
        rem_children_blocks = new_children_flat.view(K, bf, d).index_select(
            0, remaining_src
        )

        M = int(rem_parents.shape[0])
        g_new = int((M + bf - 1) // bf)
        g_new = max(1, g_new)

        # Choose new grandparent centers from the remaining parents.
        torch.manual_seed(4321)
        perm = torch.randperm(M, device=device)
        gp_choose = perm[:g_new]
        gp_new = rem_parents.index_select(0, gp_choose).contiguous()  # (g_new,d)

        # Assign remaining parents to new grandparents.

        gp_assign, gp_d2 = nearest_l2_triton(rem_parents, gp_new, valid_mask=None)

        # Sort by (gp_assign, dist rank) to pick up to bf per new gp.
        dist_rank = torch.argsort(gp_d2)
        dist_rank2 = torch.empty_like(dist_rank)
        dist_rank2[dist_rank] = torch.arange(M, device=device, dtype=dist_rank.dtype)
        order2 = torch.argsort(
            gp_assign.to(torch.int64) * (M + 1) + dist_rank2.to(torch.int64)
        )

        ga_sorted = gp_assign[order2]
        rp_sorted = rem_parents[order2]
        rr_sorted = rem_pr[order2]
        rc_sorted = rem_children_blocks[order2]

        counts2 = torch.bincount(ga_sorted, minlength=g_new).to(torch.int64)
        prefix2 = torch.cumsum(counts2, dim=0)
        start2 = prefix2.index_select(0, ga_sorted) - counts2.index_select(0, ga_sorted)
        rank2 = torch.arange(M, device=device, dtype=torch.int64) - start2

        placed2 = rank2 < bf

        parents_block = torch.full(
            (g_new * bf, d), float(self.pad_value), device=device, dtype=torch.float32
        )
        pr_block = torch.zeros(
            (g_new * bf,), device=device, dtype=self.parent_radii.dtype
        )
        children_block_flat = torch.full(
            (g_new * bf * bf, d),
            float(self.pad_value),
            device=device,
            dtype=torch.float32,
        )

        # def _scatter_fallback(
        #     *,
        #     dst_idx: torch.Tensor,
        #     src_p: torch.Tensor,
        #     src_r: torch.Tensor,
        #     src_c_blocks: torch.Tensor,
        # ) -> None:
        #     parents_block.index_copy_(0, dst_idx, src_p)
        #     pr_block.index_copy_(0, dst_idx, src_r.to(self.parent_radii.dtype))
        #     dst_children3 = children_block_flat.view(g_new * bf, bf, d)
        #     dst_children3.index_copy_(0, dst_idx, src_c_blocks)

        # def _scatter_try_triton(
        #     *,
        #     dst_idx: torch.Tensor,
        #     src_p: torch.Tensor,
        #     src_r: torch.Tensor,
        #     src_c_blocks: torch.Tensor,
        # ) -> None:
        #     try:

        #     except Exception:
        #         _scatter_fallback(
        #             dst_idx=dst_idx,
        #             src_p=src_p,
        #             src_r=src_r,
        #             src_c_blocks=src_c_blocks,
        #         )

        if placed2.any():
            ga_p = ga_sorted[placed2]
            r_p = rank2[placed2]
            dst = ga_p.to(torch.int64) * bf + r_p
            # _scatter_try_triton(
            #     dst_idx=dst,
            #     src_p=rp_sorted[placed2],
            #     src_r=rr_sorted[placed2],
            #     src_c_blocks=rc_sorted[placed2],
            # )

            scatter_parent_children_blocks(
                src_parents=rp_sorted[placed2],
                src_parent_radii=rr_sorted[placed2].to(torch.float32),
                src_children_flat=rc_sorted[placed2].reshape(-1, d),
                dst_parent_idx=dst,
                dst_parents=parents_block,
                dst_parent_radii=pr_block,
                dst_children_flat=children_block_flat,
                bf=bf,
            )

        # Place any leftovers into remaining free slots (vectorized).
        if (~placed2).any():
            leftovers_p = rp_sorted[~placed2]
            leftovers_r = rr_sorted[~placed2]
            leftovers_c = rc_sorted[~placed2]

            empty_slots = torch.all(parents_block == float(self.pad_value), dim=-1)
            free = torch.nonzero(empty_slots, as_tuple=False).view(-1)
            if free.numel() < leftovers_p.shape[0]:
                raise RuntimeError("Invariant broken: insufficient free parent slots")

            take = free[: leftovers_p.shape[0]]
            # _scatter_try_triton(
            #     dst_idx=take,
            #     src_p=leftovers_p,
            #     src_r=leftovers_r,
            #     src_c_blocks=leftovers_c,
            # )
            scatter_parent_children_blocks(
                src_parents=leftovers_p,
                src_parent_radii=leftovers_r.to(torch.float32),
                src_children_flat=leftovers_c.reshape(-1, d),
                dst_parent_idx=take,
                dst_parents=parents_block,
                dst_parent_radii=pr_block,
                dst_children_flat=children_block_flat,
                bf=bf,
            )

        # Compute radii for new grandparents.
        gp_f = gp_new.float()
        parents_f = parents_block.float().view(g_new, bf, d)
        pr_f = pr_block.float().view(g_new, bf)
        validp = ~torch.all(parents_f == float(self.pad_value), dim=-1)
        dists_gp = torch.linalg.norm(parents_f - gp_f[:, None, :], dim=-1)
        totals = dists_gp + pr_f
        totals = torch.where(validp, totals, torch.tensor(float("-inf"), device=device))
        gp_radii_new = torch.max(totals, dim=1).values
        gp_radii_new = torch.where(
            torch.isfinite(gp_radii_new), gp_radii_new, torch.zeros_like(gp_radii_new)
        ).to(self.grand_parent_radii.dtype)

        # Append new blocks.
        self.grand_parents = torch.cat([self.grand_parents, gp_new], dim=0).contiguous()
        self.grand_parent_radii = torch.cat(
            [self.grand_parent_radii, gp_radii_new], dim=0
        ).contiguous()
        self.parents = torch.cat([self.parents, parents_block], dim=0).contiguous()
        self.parent_radii = torch.cat([self.parent_radii, pr_block], dim=0).contiguous()
        self.children = torch.cat(
            [self.children, children_block_flat], dim=0
        ).contiguous()

        # Extend/refresh buffer to match new children length.
        self.buffer = torch.zeros(
            (self.children.shape[0],), device="cuda", dtype=torch.float32
        )

        # Refresh cached masks/counters for future fast updates.
        self._init_update_v2_state()
        return self

    @torch.no_grad()
    def update_v1(self, new_keys: torch.Tensor):
        if new_keys.numel() == 0:
            return self
        if self.parents is None or self.children is None or self.parent_radii is None:
            raise RuntimeError("CUDAIndexer.update() called before build()")
        if new_keys.ndim != 2:
            raise ValueError(f"new_keys must be (n,d), got {tuple(new_keys.shape)}")

        new_keys = new_keys.to("cuda").contiguous()

        bf = int(self.branching_factor)

        if self.pad_value is None:
            raise ValueError(
                "CUDAIndexer.update() requires pad_value to detect empty slots"
            )

        device = new_keys.device

        # 1) Assign each new key to its nearest *existing* valid parent
        valid_parent = ~(torch.all(self.parents == self.pad_value, dim=-1))
        nearest_parent, nearest_d2 = self._nearest_l2(
            new_keys, self.parents, valid_mask=valid_parent
        )

        # 2) Try to fill padded child slots under those parents
        placed_mask, _ = self._fill_existing_children(
            x=new_keys, parent_idx=nearest_parent
        )

        if placed_mask.any():
            inserted_keys = new_keys[placed_mask]
            inserted_parents = nearest_parent[placed_mask]
            self._update_parent_radii_for_inserted(
                inserted_keys=inserted_keys,
                inserted_parent_idx=inserted_parents,
            )

            # For 3-level, update affected grandparent radii (monotonic increase)
            if self.depth == CUDAIndexer.DEPTH.THREE_LEVELS:
                if self.grand_parents is None or self.grand_parent_radii is None:
                    raise RuntimeError("THREE_LEVELS build state incomplete")
                g = self.grand_parents.shape[0]
                # Map parent idx -> grandparent idx via block structure
                gp_idx = (inserted_parents // bf).to(torch.int64)
                gp_centers = self.grand_parents.index_select(0, gp_idx)
                # total radius contribution: ||parent - gp|| + parent_radius
                dist_gp = torch.linalg.norm(
                    self.parents.index_select(0, inserted_parents) - gp_centers, dim=1
                ).float()
                total = (
                    dist_gp
                    + self.parent_radii.index_select(0, inserted_parents).float()
                )

                upd_gp = torch.full(
                    (g,), float("-inf"), device=device, dtype=torch.float32
                )
                upd_gp.scatter_reduce_(
                    0, gp_idx, total, reduce="amax", include_self=True
                )
                upd_gp = torch.where(
                    torch.isfinite(upd_gp), upd_gp, torch.zeros_like(upd_gp)
                )
                self.grand_parent_radii = torch.maximum(
                    self.grand_parent_radii.float(), upd_gp
                ).to(dtype=self.grand_parent_radii.dtype)

        # 3) Overflow keys (assigned to full parents) => create new parents and append/insert
        overflow = new_keys[~placed_mask]
        if overflow.numel() == 0:
            return self

        new_parents, new_children_flat, new_parent_radii = (
            self._build_new_parents_from_overflow(overflow, seed=1234)
        )
        if new_parents.numel() == 0:
            return self

        K = new_parents.shape[0]
        d = new_parents.shape[1]

        if self.depth == CUDAIndexer.DEPTH.TWO_LEVELS:
            # Append new parent blocks
            self.parents = torch.cat([self.parents, new_parents], dim=0).contiguous()
            self.parent_radii = torch.cat(
                [self.parent_radii, new_parent_radii.to(self.parent_radii.dtype)], dim=0
            ).contiguous()
            self.children = torch.cat(
                [self.children, new_children_flat], dim=0
            ).contiguous()

            # Extend buffer to match new children length
            if self.buffer is None:
                self.buffer = torch.zeros(
                    (self.children.shape[0],), device="cuda", dtype=torch.float32
                )
            else:
                add = new_children_flat.shape[0]
                if add > 0:
                    self.buffer = torch.cat(
                        [
                            self.buffer,
                            torch.zeros((add,), device="cuda", dtype=torch.float32),
                        ],
                        dim=0,
                    )

            return self

        # THREE_LEVELS: insert into nearest grandparent blocks if possible, else create new grandparents.
        if self.grand_parents is None or self.grand_parent_radii is None:
            raise RuntimeError("THREE_LEVELS build state incomplete")

        G = self.grand_parents.shape[0]
        P_total = self.parents.shape[0]
        if P_total != G * bf:
            raise RuntimeError(
                f"Invalid THREE_LEVELS layout: parents={P_total} but grand_parents={G} and bf={bf}"
            )
        if self.children.shape[0] != P_total * bf:
            raise RuntimeError(
                "Invalid THREE_LEVELS layout: children not aligned with parents blocks"
            )

        # Determine empty parent slots per grandparent
        parents2 = self.parents.view(G, bf, d)
        parent_empty = torch.all(parents2 == self.pad_value, dim=-1)  # (G,bf)
        parent_avail = parent_empty.sum(dim=1).to(torch.int64)  # (G,)
        slot_ids = torch.arange(bf, device=device, dtype=torch.int64)
        slot_mat = slot_ids.view(1, bf).expand(G, bf)
        empty_first = torch.where(parent_empty, slot_mat, torch.full_like(slot_mat, bf))
        empty_sorted = torch.sort(empty_first, dim=1).values  # (G,bf)

        # Nearest grandparent for each new parent
        gp_near, _ = self._nearest_l2(new_parents, self.grand_parents, valid_mask=None)
        # gp_near is in [0,G)

        # Group-wise place into existing G blocks
        order = torch.argsort(gp_near)
        gp_sorted = gp_near[order]
        src_sorted = order

        counts = torch.bincount(gp_near, minlength=G).to(torch.int64)
        prefix = torch.cumsum(counts, dim=0)
        start = prefix.index_select(0, gp_sorted) - counts.index_select(0, gp_sorted)
        rank = torch.arange(K, device=device, dtype=torch.int64) - start

        can_place = rank < parent_avail.index_select(0, gp_sorted)

        # Indices in new_parents that go into existing blocks
        place_src = src_sorted[can_place]
        place_gp = gp_sorted[can_place]
        place_rank = rank[can_place]
        place_slot = empty_sorted[place_gp, place_rank]
        place_parent_global = place_gp * bf + place_slot  # (num_place,)

        # Write parents + parent_radii
        if place_src.numel() > 0:
            self.parents.index_copy_(
                0, place_parent_global, new_parents.index_select(0, place_src)
            )
            self.parent_radii.index_copy_(
                0,
                place_parent_global,
                new_parent_radii.index_select(0, place_src).to(self.parent_radii.dtype),
            )

            # Write children blocks into corresponding parent slots
            children_blocks = new_children_flat.view(K, bf, d)
            dst_children3 = self.children.view(P_total, bf, d)
            dst_children3.index_copy_(
                0, place_parent_global, children_blocks.index_select(0, place_src)
            )

            # Update grandparent radii (monotonic increase)
            gp_centers = self.grand_parents.index_select(0, place_gp)
            dist_gp = torch.linalg.norm(
                self.parents.index_select(0, place_parent_global) - gp_centers, dim=1
            ).float()
            total = (
                dist_gp + self.parent_radii.index_select(0, place_parent_global).float()
            )

            upd_gp = torch.full((G,), float("-inf"), device=device, dtype=torch.float32)
            upd_gp.scatter_reduce_(0, place_gp, total, reduce="amax", include_self=True)
            upd_gp = torch.where(
                torch.isfinite(upd_gp), upd_gp, torch.zeros_like(upd_gp)
            )
            self.grand_parent_radii = torch.maximum(
                self.grand_parent_radii.float(), upd_gp
            ).to(dtype=self.grand_parent_radii.dtype)

        # Remaining new parents require new grandparents
        if can_place.all():
            return self

        remaining_src = src_sorted[~can_place]
        rem_parents = new_parents.index_select(0, remaining_src)
        rem_pr = new_parent_radii.index_select(0, remaining_src)
        rem_children_blocks = new_children_flat.view(K, bf, d).index_select(
            0, remaining_src
        )

        M = rem_parents.shape[0]
        g_new = int((M + bf - 1) // bf)
        g_new = max(1, g_new)
        torch.manual_seed(4321)
        perm = torch.randperm(M, device=device)
        gp_choose = perm[:g_new]
        gp_new = rem_parents.index_select(0, gp_choose).contiguous()  # (g_new,d)

        # Assign remaining parents to new grandparents and fill bf slots per new gp
        gp_assign, gp_d2 = self._nearest_l2(rem_parents, gp_new, valid_mask=None)
        # gp_assign in [0,g_new)

        # Sort by (gp_assign, dist)
        order2 = torch.argsort(gp_assign * (M + 1) + torch.argsort(gp_d2))
        ga_sorted = gp_assign[order2]
        rp_sorted = rem_parents[order2]
        rr_sorted = rem_pr[order2]
        rc_sorted = rem_children_blocks[order2]

        counts2 = torch.bincount(ga_sorted, minlength=g_new).to(torch.int64)
        prefix2 = torch.cumsum(counts2, dim=0)
        start2 = prefix2.index_select(0, ga_sorted) - counts2.index_select(0, ga_sorted)
        rank2 = torch.arange(M, device=device, dtype=torch.int64) - start2

        placed2 = rank2 < bf
        parents_block = torch.full(
            (g_new * bf, d), self.pad_value, device=device, dtype=torch.float32
        )
        pr_block = torch.zeros(
            (g_new * bf,), device=device, dtype=self.parent_radii.dtype
        )
        children_block = torch.full(
            (g_new * bf, bf, d), self.pad_value, device=device, dtype=torch.float32
        )

        if placed2.any():
            ga_p = ga_sorted[placed2]
            r_p = rank2[placed2]
            dst = ga_p * bf + r_p
            parents_block.index_copy_(0, dst, rp_sorted[placed2])
            pr_block.index_copy_(0, dst, rr_sorted[placed2].to(self.parent_radii.dtype))
            children_block.index_copy_(0, dst, rc_sorted[placed2])

        # Place leftovers into any free slots (should exist)
        if (~placed2).any():
            leftovers_p = rp_sorted[~placed2]
            leftovers_r = rr_sorted[~placed2]
            leftovers_c = rc_sorted[~placed2]

            next_slot = (
                ((torch.all(parents_block == self.pad_value, dim=-1)).logical_not())
                .view(g_new, bf)
                .sum(dim=1)
                .to(torch.int64)
            )
            # Build distance matrix once (M_left x g_new)
            lp = leftovers_p
            lp_norm = (lp * lp).sum(dim=1, keepdim=True)
            gp_norm = (gp_new * gp_new).sum(dim=1)
            d2mat = lp_norm + gp_norm[None, :] - 2.0 * (lp @ gp_new.t())
            d2mat = torch.clamp_min(d2mat, 0.0)

            for j in range(lp.shape[0]):
                cap = next_slot < bf
                if not bool(cap.any().item()):
                    raise RuntimeError(
                        "Invariant broken: no capacity for new grandparent slots"
                    )
                d2j = torch.where(
                    cap, d2mat[j], torch.tensor(float("inf"), device=device)
                )
                gsel = int(torch.argmin(d2j).item())
                s = int(next_slot[gsel].item())
                idx = gsel * bf + s
                parents_block[idx] = leftovers_p[j]
                pr_block[idx] = leftovers_r[j].to(self.parent_radii.dtype)
                children_block[idx] = leftovers_c[j]
                next_slot[gsel] += 1

        children_block_flat = children_block.view(g_new * bf * bf, d)

        # Compute radii for new grandparents
        gp_f = gp_new.float()
        parents_f = parents_block.float().view(g_new, bf, d)
        pr_f = pr_block.float().view(g_new, bf)
        validp = ~torch.all(parents_f == self.pad_value, dim=-1)
        dists_gp = torch.linalg.norm(parents_f - gp_f[:, None, :], dim=-1)
        totals = dists_gp + pr_f
        totals = torch.where(validp, totals, torch.tensor(float("-inf"), device=device))
        gp_radii_new = torch.max(totals, dim=1).values
        gp_radii_new = torch.where(
            torch.isfinite(gp_radii_new), gp_radii_new, torch.zeros_like(gp_radii_new)
        ).to(self.grand_parent_radii.dtype)

        # Append new blocks
        self.grand_parents = torch.cat([self.grand_parents, gp_new], dim=0).contiguous()
        self.grand_parent_radii = torch.cat(
            [self.grand_parent_radii, gp_radii_new], dim=0
        ).contiguous()
        self.parents = torch.cat([self.parents, parents_block], dim=0).contiguous()
        self.parent_radii = torch.cat([self.parent_radii, pr_block], dim=0).contiguous()
        self.children = torch.cat(
            [self.children, children_block_flat], dim=0
        ).contiguous()

        # Extend buffer to match new children length
        if self.buffer is None:
            self.buffer = torch.zeros(
                (self.children.shape[0],), device="cuda", dtype=torch.float32
            )
        else:
            add = children_block_flat.shape[0]
            if add > 0:
                self.buffer = torch.cat(
                    [
                        self.buffer,
                        torch.zeros((add,), device="cuda", dtype=torch.float32),
                    ],
                    dim=0,
                )

        return self


class CPUCUDAIndexer(Indexer):
    def __init__(
        self,
        gpu_cache_size: int,
        gpu_max_size: int,
        cpu_num_levels: int,
        cpu_branching_factor: int,
        cpu_max_iterations: int = 1,
    ):
        super().__init__()
        self.gpu_cache_size = gpu_cache_size
        self.gpu_max_size = gpu_max_size
        self.cpu_indexer = CPUIndexer(
            num_levels=cpu_num_levels,
            branching_factor=cpu_branching_factor,
            max_iterations=cpu_max_iterations,
        )

        self.gpu_cached_keys: Optional[torch.Tensor] = None

    @torch.no_grad()
    def build(self, keys: torch.Tensor):  # keys are on CUDA
        self.cpu_indexer.build(keys)

        # store last gpu_cache_size keys on GPU
        num_keys = keys.shape[0]
        cache_start = max(0, num_keys - self.gpu_cache_size)
        self.gpu_cached_keys = keys[cache_start:]

        return self
