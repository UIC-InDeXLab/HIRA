from abc import ABC
import torch
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import faiss
from enum import Enum
import faiss.contrib.torch_utils  # enables torch<->faiss GPU interop


class Indexer(ABC):
    def build(self, keys: torch.Tensor):
        raise NotImplementedError

    def update(self, new_keys: torch.Tensor):
        raise NotImplementedError

    @staticmethod
    def sample_max_level(new_keys, branching_factor: int):
        U = torch.rand(new_keys.shape[0], device=new_keys.device)
        L = 1 + torch.floor(
            torch.log(U)
            / torch.log(torch.tensor(1.0 / branching_factor, device=new_keys.device))
        )
        return L.long()


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

        return self

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
