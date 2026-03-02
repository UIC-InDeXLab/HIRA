import torch

from hira.indexer import CPUIndexer
from .base import BaseSearcher

from hira.kernels.cpp.torch_ext_loader import (
    hira_torch_ext_v2,
    hira_torch_ext_v3,
    hira_torch_ext_v4,
    hira_torch_ext,
)


class CPUSearcher(BaseSearcher):
    """Multi-head CPU searcher for HIRA index."""

    def __init__(
        self,
        chunk_size: int = 8 * 1024,
        profiling: bool = False,
        search_strategy: str = "fused_v3",
    ):
        super().__init__()
        self.profiling = profiling
        self.chunk_size = chunk_size
        self._ext_cache = {}
        self._reset_stats()

        self.strategies = {
            # "vectorized_cpp_filter": self._search_vectorized,
            # "exact_torch": self._search_exact_torch,
            "fused_v1": self._search_fused_v1,
            "fused_v2": self._search_fused_v2,
            "fused_v3": self._search_fused_v3,  # the best
            "fused_v4": self._search_fused_v4,
        }

        self.search_alg = self.strategies[search_strategy]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: torch.Tensor,
        threshold: torch.Tensor,
        indexer: CPUIndexer,
        scaling: torch.Tensor | None = None,
    ):
        """Tree-pruned inner-product search over indexed keys.

        High-level behavior:
        - Tree traversal uses the gate ``dot(q, center) + radius >= threshold``
          at each level to prune subtrees.
        - Leaves that survive traversal are exact-filtered with
          ``dot(q, key) >= threshold``.
        - Returned kept scores are multiplied by ``scaling``; pruned/rejected
          entries stay ``0.0``.

        ``q`` is expected to be L2-normalized. The tree bound above assumes a
        unit query norm; for non-unit queries the conservative bound is
        ``dot(q, center) + ||q|| * radius``.

        Args:
            query:     ``(1, H, 1, D)`` query tensor.
            threshold: ``(H,)`` per-head threshold tensor.
            scaling:   Optional ``(H,)`` per-head scaling tensor for returned scores.
            indexer:   Built :class:`CPUIndexer` (all tensors ``(H, …)``).

        Returns:
            ``(H, N)`` float tensor -- dot-product scores for qualifying
            keys, 0.0 for pruned / below-threshold keys.
        """
        # Prepare query: (1, H, 1, D) -> (H, D)
        query = query.squeeze(0).squeeze(-2).contiguous()

        assert threshold.shape == (
            query.shape[0],
        ), "threshold shape must match number of heads in query"

        if scaling is None:
            scaling = torch.ones(
                (query.shape[0],), device=query.device, dtype=torch.float32
            )
        assert scaling.shape == threshold.shape

        q_head_to_kv = self._resolve_q_head_to_kv(
            num_query_heads=query.shape[0], num_kv_heads=indexer.keys.shape[0]
        )

        if self.profiling:
            self._reset_stats()

        return self.search_alg(
            query,
            threshold,
            indexer,
            q_head_to_kv=q_head_to_kv,
            scaling=scaling,
        )

    @staticmethod
    def _resolve_q_head_to_kv(num_query_heads: int, num_kv_heads: int) -> torch.Tensor:
        if num_query_heads == num_kv_heads:
            return torch.arange(num_query_heads, dtype=torch.long)
        assert num_query_heads % num_kv_heads == 0
        group_size = num_query_heads // num_kv_heads
        return torch.arange(num_query_heads, dtype=torch.long) // group_size

    # ------------------------------------------------------------------
    # Zero-copy fused search (PyTorch C++ extension)
    # ------------------------------------------------------------------

    def _search_fused_v1(
        self,
        query: torch.Tensor,
        threshold: torch.Tensor,
        indexer: CPUIndexer,
        q_head_to_kv: torch.Tensor,
        scaling: torch.Tensor | None = None,
    ):
        """Fused search using the zero-copy PyTorch C++ extension.

        Same algorithm as :meth:`search_fused` but accepts ``torch::Tensor``
        directly in C++ — no numpy conversion or memory copies.  The kernel
        is JIT-compiled on first call via ``torch.utils.cpp_extension.load()``.

        Args:
            query:     ``(1, H, 1, D)`` query tensor.
            threshold: ``(H,)`` per-head threshold tensor.
            indexer:   Built :class:`CPUIndexer`.

        Returns:
            ``(H, N)`` float tensor – qualifying scores, 0.0 elsewhere.
        """
        # Collect level data as lists of tensors (no copies – just ensure
        # contiguous + float32/int64 which they already are from the indexer)
        centers_list = []
        radii_list = []
        c2p_list = []
        sizes_list = []

        for level in indexer.levels:
            centers_list.append(level.ball_centers.float().contiguous())
            radii_list.append(level.ball_radii.float().contiguous())
            if level.child2parent is not None:
                c2p_list.append(level.child2parent.long().contiguous())
            else:
                c2p_list.append(torch.empty(0, dtype=torch.long))
            sizes_list.append(level.size)

        return hira_torch_ext.fused_tree_search(
            indexer.keys.float().contiguous(),
            query,
            threshold,
            centers_list,
            radii_list,
            c2p_list,
            sizes_list,
            q_head_to_kv.long().contiguous(),
            scaling.float().contiguous(),
        )

    def _search_fused_v2(
        self,
        query: torch.Tensor,
        threshold: torch.Tensor,
        indexer: CPUIndexer,
        q_head_to_kv: torch.Tensor,
        scaling: torch.Tensor | None = None,
    ):
        """Fused search using the v2 zero-copy PyTorch C++ extension.

        The v2 kernel keeps the same semantics as ``search_fused_torch_ext``
        but uses a different OpenMP strategy and internal buffering to reduce
        high-head overhead.

        Args:
            query:     ``(1, H, 1, D)`` query tensor.
            threshold: ``(H,)`` per-head threshold tensor.
            indexer:   Built :class:`CPUIndexer`.

        Returns:
            ``(H, N)`` float tensor – qualifying scores, 0.0 elsewhere.
        """
        (
            centers_list,
            radii_list,
            c2p_list,
            sizes_list,
        ) = self._prepare_ext_level_data(indexer=indexer, with_adjacency=False)

        return hira_torch_ext_v2.fused_tree_search_v2(
            indexer.keys.float().contiguous(),
            query,
            threshold,
            centers_list,
            radii_list,
            c2p_list,
            sizes_list,
            q_head_to_kv.long().contiguous(),
            scaling.float().contiguous(),
        )

    def _search_fused_v3(
        self,
        query: torch.Tensor,
        threshold: torch.Tensor,
        indexer: CPUIndexer,
        q_head_to_kv: torch.Tensor,
        scaling: torch.Tensor | None = None,
    ):
        """Fused search using the v3 zero-copy PyTorch C++ extension.

        v3 uses active-parent traversal with precomputed parent->children
        adjacency to avoid scanning all children at each level.
        """
        (
            centers_list,
            radii_list,
            c2p_list,
            sizes_list,
            offsets_list,
            children_list,
        ) = self._prepare_ext_level_data(indexer=indexer, with_adjacency=True)

        return hira_torch_ext_v3.fused_tree_search_v3(
            indexer.keys.float().contiguous(),
            query,
            threshold,
            centers_list,
            radii_list,
            c2p_list,
            sizes_list,
            offsets_list,
            children_list,
            q_head_to_kv.long().contiguous(),
            scaling.float().contiguous(),
        )

    def _search_fused_v4(
        self,
        query: torch.Tensor,
        threshold: torch.Tensor,
        indexer: CPUIndexer,
        q_head_to_kv: torch.Tensor,
        scaling: torch.Tensor | None = None,
    ):
        """Fused search using the v4 zero-copy PyTorch C++ extension.

        v4 is specialized for D=128 and uses active-parent traversal +
        flattened-candidate exact filtering for better load balance.
        """
        assert query.shape[-1] == 128, "v4 kernel is specialized for D=128"

        (
            centers_list,
            radii_list,
            c2p_list,
            sizes_list,
            offsets_list,
            children_list,
        ) = self._prepare_ext_level_data(indexer=indexer, with_adjacency=True)

        return hira_torch_ext_v4.fused_tree_search_v4(
            indexer.keys.float().contiguous(),
            query,
            threshold,
            centers_list,
            radii_list,
            c2p_list,
            sizes_list,
            offsets_list,
            children_list,
            q_head_to_kv.long().contiguous(),
            scaling.float().contiguous(),
        )

    def _prepare_ext_level_data(
        self, indexer: CPUIndexer, with_adjacency: bool = False
    ):
        """Prepare and cache level tensors for torch extensions."""
        levels = indexer.levels
        key = (
            id(indexer),
            indexer.num_keys,
            tuple(lvl.size for lvl in levels),
            with_adjacency,
        )
        cached = self._ext_cache.get(key)
        if cached is not None:
            return cached

        centers_list = []
        radii_list = []
        c2p_list = []
        sizes_list = []
        offsets_list = []
        children_list = []

        num_levels = len(levels)
        H = indexer.keys.shape[0]

        for l, level in enumerate(levels):
            centers_list.append(level.ball_centers.float().contiguous())
            radii_list.append(level.ball_radii.float().contiguous())
            sizes_list.append(level.size)

            if level.child2parent is not None:
                c2p = level.child2parent.long().contiguous()
                c2p_list.append(c2p)
            else:
                c2p = None
                c2p_list.append(torch.empty(0, dtype=torch.long))

            if with_adjacency:
                if c2p is None:
                    offsets_list.append(torch.empty(0, dtype=torch.long))
                    children_list.append(torch.empty(0, dtype=torch.long))
                else:
                    # Number of parents at upper level is next level size.
                    P = levels[l + 1].size if l + 1 < num_levels else 0
                    off, ch = self._build_parent_child_adjacency(c2p, P, H)
                    offsets_list.append(off)
                    children_list.append(ch)

        if with_adjacency:
            value = (
                centers_list,
                radii_list,
                c2p_list,
                sizes_list,
                offsets_list,
                children_list,
            )
        else:
            value = (centers_list, radii_list, c2p_list, sizes_list)

        # Keep only the latest cache entry to avoid unbounded growth.
        self._ext_cache.clear()
        self._ext_cache[key] = value
        return value

    @staticmethod
    def _build_parent_child_adjacency(c2p: torch.Tensor, P: int, H: int):
        """Build per-head CSR-like parent->children adjacency from child2parent."""
        C = c2p.shape[1]
        offsets = torch.empty((H, P + 1), dtype=torch.long)
        children = torch.empty((H, C), dtype=torch.long)

        for h in range(H):
            p = c2p[h]  # (C,)
            order = torch.argsort(p)
            sorted_parents = p.index_select(0, order)
            counts = torch.bincount(sorted_parents, minlength=P)
            off = torch.empty(P + 1, dtype=torch.long)
            off[0] = 0
            off[1:] = counts.cumsum(0)
            offsets[h] = off
            children[h] = order

        return offsets.contiguous(), children.contiguous()

    def _reset_stats(self):
        self.stats = {"all_keys": [], "active_keys": [], "exact_checks": 0}

    def print_stats(self):
        if not self.profiling:
            print("Profiling not enabled")
            return

        print("HIRA CPUSearcher profiling statistics:")
        for i, total in enumerate(self.stats["all_keys"]):
            active = self.stats["active_keys"][i]
            print(f"  Level {i} | total: {total} | active: {active}")
        print(f"  Exact checks: {self.stats['exact_checks']}")
