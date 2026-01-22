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
        self.preallocate_search_buffers()

        return self

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
    pass
