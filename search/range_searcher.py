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
        """Reset profiling statistics"""
        self.stats = {"all_keys": [], "active_keys": [], "exact_checks": -1}

    @profile
    def search(self, query, threshold, index: "KMeansIndex"):
        """
        Args:
            threshold (_type_): The points with x.q >= threshold are returned
        """

        # normalize query
        query = query / torch.norm(query, p=2)

        # search root
        root = index.levels[-1]

        scores = torch.matmul(root.key_centers, query)
        mask = (scores + root.key_radii) >= threshold
        active_cluster_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        parent_size = len(root.key_ptrs)

        if self.enable_profiling:
            self.stats["all_keys"].append(len(root.key_ptrs))
            self.stats["active_keys"].append(len(active_cluster_idx))

        # recursively search down the tree
        levels = index.levels[::-1][1:-1]  # skip root, reverse order, skip level 0
        for level_idx, level in enumerate(levels):
            # level here is the child level of previous one
            # OPTIMIZED
            parent_mask = torch.zeros(
                parent_size, dtype=torch.bool, device=query.device
            )
            parent_mask[active_cluster_idx] = True
            child_mask = parent_mask[level.child2parent]

            # get radii and centers
            child_idx = torch.nonzero(child_mask, as_tuple=False).squeeze(1)
            child_radii = level.key_radii.index_select(0, child_idx)
            child_centers = level.key_centers.index_select(0, child_idx)

            # scores
            scores = torch.matmul(child_centers, query)
            mask = (scores + child_radii) >= threshold
            active_cluster_idx = child_idx[mask]

            # option 2
            # mask = torch.add(child_radii, torch.mv(child_centers, query)).ge(threshold)
            # active_cluster_ptrs = pointers[mask]

            parent_size = len(level.key_ptrs)

            if self.enable_profiling:
                self.stats["all_keys"].append(len(child_centers))
                self.stats["active_keys"].append(len(active_cluster_idx))

            if len(active_cluster_idx) == 0:
                return torch.tensor([], dtype=torch.long, device=query.device)

        # ONLY LAST LEVEL: level 1->0
        parent_mask = torch.zeros(parent_size, dtype=torch.bool, device=query.device)
        parent_mask[active_cluster_idx] = True
        child_mask = parent_mask[index.levels[0].child2parent]

        # get radii and centers
        child_idx = torch.nonzero(child_mask, as_tuple=False).squeeze(1)

        if self.enable_profiling:
            self.stats["exact_checks"] = len(child_idx)

        # TODO: DANGER Remove
        # child_idx = child_idx[:len(index.keys) // 5]
        # self.stats["exact_checks"] = len(child_idx)

        # exact filter on final candidates
        active_cluster_ptrs = index.levels[0].key_ptrs[child_idx]
        keys = index.keys[active_cluster_ptrs]
        final_scores = torch.matmul(keys, query)
        final_mask = final_scores >= threshold
        qualifying_keys = active_cluster_ptrs[final_mask]

        return qualifying_keys  # indices of keys satisfying q.k >= threshold

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
