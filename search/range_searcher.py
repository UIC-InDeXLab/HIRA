"""
Halfspace range searcher for efficient key selection using hierarchical indexes.

The HalfspaceSearcher performs halfspace range searches to identify keys whose
dot product with a query exceeds a threshold. It uses hierarchical pruning
with cluster radii to efficiently traverse the index.
"""

import torch
from ..index.index import KMeansIndex

# Dummy profile decorator (can be overridden by line_profiler if installed)
try:
    profile
except NameError:

    def profile(func):
        return func


class HalfspaceSearcher:
    def __init__(
        self,
        enable_profiling: bool = False,
        optimized: bool = True,
    ):
        self.enable_profiling = enable_profiling
        self.reset_stats()
        self.optimized = optimized

    def reset_stats(self):
        self.stats = {"all_keys": [], "active_keys": [], "exact_checks": -1}

    @profile
    def search(self, query, threshold, index: "KMeansIndex"):
        # normalize query
        query = query / torch.norm(query, p=2)

        # corner case
        if len(index.levels) == 1:
            scores = index.keys @ query  # [N]
            qualifying_idx = (scores >= threshold).nonzero(as_tuple=True)[0]
            if self.enable_profiling:
                self.stats["all_keys"] = [index.keys.shape[0]]
                self.stats["active_keys"] = [qualifying_idx.numel()]
                self.stats["exact_checks"] = index.keys.shape[0]
            return qualifying_idx

        # search root
        root = index.levels[-1]

        scores = torch.matmul(root.ball_centers, query)
        mask = (scores + root.ball_radii) >= threshold
        active_cluster_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

        if self.enable_profiling:
            self.stats["all_keys"].append(root.size)
            self.stats["active_keys"].append(len(active_cluster_idx))

        # recursively search down the tree
        levels = index.levels[::-1][1:-1]  # skip root, reverse order, skip level 0
        for level in levels:
            # level here is the child level of previous one

            if active_cluster_idx.numel() == 0:
                return torch.tensor([], dtype=torch.long, device=query.device)

            # CSR is fast for small fraction of parents, otherwise use child2parent
            use_csr = False  # hardcoded for now

            if use_csr:
                active_parents = active_cluster_idx
                rowptr = level.p_pointer
                p2c = level.parent2child

                active_parents, _ = torch.sort(active_parents)

                starts = rowptr[active_parents]  # [A]
                ends = rowptr[active_parents + 1]  # [A]
                counts = ends - starts  # [A]
                total = counts.sum()

                if total.item() == 0:
                    return torch.empty((0,), dtype=torch.long, device=query.device)

                base = torch.repeat_interleave(starts, counts)  # [total]
                csum = torch.cumsum(counts, dim=0)
                seg_starts = csum - counts
                inner = torch.arange(
                    total, device=query.device
                ) - torch.repeat_interleave(seg_starts, counts)

                idx_in_p2c = base + inner
                child_idx = p2c[idx_in_p2c]  # [total] child local ids
            else:
                p = level.child2parent  # [C]
                parent_mask = torch.zeros(
                    level.num_parents, dtype=torch.bool, device=query.device
                )
                parent_mask[active_cluster_idx] = True

                child_mask = parent_mask[p]  # [C] bool
                child_idx = child_mask.nonzero(as_tuple=True)[0]  # child local ids

            child_radii = level.ball_radii.index_select(0, child_idx)
            child_centers = level.ball_centers.index_select(0, child_idx)

            # scores
            scores = torch.matmul(child_centers, query)
            mask = (scores + child_radii) >= threshold
            active_cluster_idx = child_idx[mask]

            if self.enable_profiling:
                self.stats["all_keys"].append(len(child_centers))
                self.stats["active_keys"].append(len(active_cluster_idx))

        # LEVEL 1 -> 0
        level0 = index.levels[0]  # keys
        level1 = index.levels[1]  # clusters of keys

        # active_cluster_idx are indices into level1 (parents of level0)
        P1 = level1.size

        parent_mask = torch.zeros(P1, dtype=torch.bool, device=query.device)
        parent_mask[active_cluster_idx] = True

        # level0.child2parent: [num_keys] mapping key_id -> parent_id in level1
        leaf_idx = parent_mask[level0.child2parent].nonzero(as_tuple=True)[0]

        # TODO: DANGER Remove
        # child_idx = child_idx[:len(index.keys) // 5]
        # self.stats["exact_checks"] = len(child_idx)?

        if self.enable_profiling:
            self.stats["exact_checks"] = len(leaf_idx)

        # exact check
        scores = index.keys[leaf_idx] @ query
        qualifying_idx = leaf_idx[scores >= threshold]

        return qualifying_idx  # indices of keys satisfying q.k >= threshold

    def print_stats(self):
        """Print detailed profiling statistics"""
        if not self.enable_profiling:
            print("Profiling not enabled")
            return

        print("Half-space Searcher Profiling Statistics:")
        for level_idx, allof in enumerate(self.stats["all_keys"]):
            print(
                f"Level {level_idx} | all: {allof} | active: {self.stats['active_keys'][level_idx]}"
            )
        print("Exact checks at final level:", self.stats["exact_checks"])
