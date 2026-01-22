from abc import ABC
import torch
from hira.index.indexer import CPUIndexer, CUDAIndexer
from hira.kernels.numba_kernels import exact_filter_mask_numba
from hira.kernels.triton_wrappers import (
    triton_two_level_filter,
    triton_three_level_filter_v1,
)


class Searcher(ABC):
    pass


class CPUSearcher(Searcher):
    def __init__(
        self,
        chunk_size=8 * 1024,
        profiling: bool = False,
        numba=True,  # use numba for faster kernel
    ):
        super().__init__()
        self.profiling = profiling
        self.chunk_size = chunk_size
        self.stats = {"all_keys": [], "active_keys": [], "exact_checks": -1}
        self.use_numba = numba

    def search(self, query, threshold, indexer: CPUIndexer):
        # normalize query
        query = query / torch.norm(query, p=2)

        # corner case
        if len(indexer.levels) == 1:
            scores = indexer.keys @ query  # [N]
            qualifying_idx = (scores >= threshold).nonzero(as_tuple=True)[0]
            if self.profiling:
                self.stats["all_keys"] = [indexer.keys.shape[0]]
                self.stats["active_keys"] = [qualifying_idx.numel()]
                self.stats["exact_checks"] = indexer.keys.shape[0]
            return qualifying_idx

        # search root
        root = indexer.levels[-1]

        scores = torch.matmul(root.ball_centers, query)
        mask = (scores + root.ball_radii) >= threshold
        active_cluster_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

        if self.profiling:
            self.stats["all_keys"].append(root.size)
            self.stats["active_keys"].append(len(active_cluster_idx))

        # recursively search down the tree
        levels = indexer.levels[::-1][1:-1]  # skip root, reverse order, skip level 0
        for level in levels:
            # level here is the child level of previous one

            if active_cluster_idx.numel() == 0:
                return torch.tensor([], dtype=torch.long, device=query.device)

            p = level.child2parent  # [C]
            buf = level.parent_mask_buf
            buf.zero_()
            buf[active_cluster_idx] = True
            child_idx = torch.nonzero(buf[p], as_tuple=False).squeeze(1)

            child_radii = level.ball_radii.index_select(0, child_idx)
            child_centers = level.ball_centers.index_select(0, child_idx)

            # scores
            scores = torch.matmul(child_centers, query)
            mask = (scores + child_radii) >= threshold
            active_cluster_idx = child_idx[mask]

            if self.profiling:
                self.stats["all_keys"].append(len(child_centers))
                self.stats["active_keys"].append(len(active_cluster_idx))

        # LEVEL 1 -> 0 (child2parent expansion)
        level0 = indexer.levels[0]  # keys
        buf = level0.parent_mask_buf
        buf.zero_()
        buf[active_cluster_idx] = True
        leaf_idx = buf[level0.child2parent].nonzero(as_tuple=True)[0]

        if self.profiling:
            self.stats["exact_checks"] = len(leaf_idx)

        # exact check
        if self.use_numba:
            qualifying_idx = self.exact_filter_kernel_call(
                indexer.keys, leaf_idx, query, threshold
            )
        else:
            qualifying_idx = self.exact_filter_chunked(
                indexer.keys, leaf_idx, query, threshold, chunk=self.chunk_size
            )

        return qualifying_idx  # indices of keys satisfying q.k >= threshold

    def exact_filter_kernel_call(self, keys, leaf_idx, query, threshold):
        # Numba works on numpy arrays (CPU)
        keys_np = keys.detach().cpu().numpy()
        leaf_idx_np = leaf_idx.detach().cpu().numpy()
        query_np = query.detach().cpu().numpy()

        mask = exact_filter_mask_numba(keys_np, leaf_idx_np, query_np, float(threshold))
        out_np = leaf_idx_np[mask]
        return torch.from_numpy(out_np).to(leaf_idx.device)

    def exact_filter_chunked(self, keys, leaf_idx, query, threshold, chunk=8 * 1024):
        out = []
        for s in range(0, leaf_idx.numel(), chunk):
            sub = leaf_idx[s : s + chunk]
            scores = keys.index_select(0, sub).matmul(query)  # matmul or @
            keep = scores >= threshold
            if keep.any():
                out.append(sub[keep])
        return torch.cat(out) if out else leaf_idx.new_empty((0,))

    def print_stats(self):
        if not self.enable_profiling:
            print("Profiling not enabled")
            return

        print("Half-space Searcher Profiling Statistics:")
        for level_idx, allof in enumerate(self.stats["all_keys"]):
            print(
                f"Level {level_idx} | all: {allof} | active: {self.stats['active_keys'][level_idx]}"
            )
        print("Exact checks at final level:", self.stats["exact_checks"])


class CUDASearcher(Searcher):
    def __init__(self, block_c):
        super().__init__()
        self.block_c = block_c

    def search(self, query, threshold, indexer):
        # normalize query
        query = query / torch.norm(query, p=2)

        if indexer.depth == CUDAIndexer.DEPTH.TWO_LEVELS:
            output = triton_two_level_filter(
                indexer.children,
                indexer.parents,
                indexer.parent_radii,
                query,
                threshold,
                out=None,  # check
                BLOCK_C=self.block_c,
                branch=indexer.branching_factor,
            )
        else:
            output = triton_three_level_filter_v1(
                indexer.children,
                indexer.parents,
                indexer.parent_radii,
                indexer.grand_parents,
                indexer.grand_parent_radii,
                query,
                threshold,
                out=None,  # check
                branch=indexer.branching_factor,
            )
        return output


class CPUCUDASearcher(Searcher):
    pass
