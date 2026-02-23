import torch

from indexer import CPUIndexer
from .base import BaseSearcher

try:
    from hira.kernels.cpp import hira_cpp_kernels
except ImportError as e:
    raise ImportError("C++ kernels not built. Run: cd hira/kernels/cpp && make") from e
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
            "vectorized_cpp_filter": self._search_vectorized,
            "fused_v1": self._search_fused,
            "fused_v2": self._search_fused_v2,
            "fused_v3": self._search_fused_v3,  # the best
            "fused_v4": self._search_fused_v4,
            "exact_torch": self._search_exact_torch,
        }

        self.search_alg = self.strategies[search_strategy]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: torch.Tensor, threshold: torch.Tensor, indexer: CPUIndexer):
        """Search for keys satisfying ``dot(query, key) >= threshold``.

        Args:
            query:     ``(1, H, 1, D)`` query tensor.
            threshold: ``(H,)`` per-head threshold tensor.
            indexer:   Built :class:`CPUIndexer` (all tensors ``(H, …)``).

        Returns:
            ``(H, N)`` float tensor -- dot-product scores for qualifying
            keys, 0.0 for pruned / below-threshold keys.
        """
        # Prepare query: (1, H, 1, D) -> (H, D), normalise
        query = query.squeeze(0).squeeze(-2).float().contiguous()
        query = query / query.norm(dim=-1, keepdim=True)

        threshold = self._coerce_threshold(
            threshold=threshold, num_query_heads=query.shape[0]
        )
        q_head_to_kv = self._resolve_q_head_to_kv(
            num_query_heads=query.shape[0], num_kv_heads=indexer.keys.shape[0]
        )

        if self.profiling:
            self._reset_stats()

        return self.search_alg(query, threshold, indexer, q_head_to_kv=q_head_to_kv)

    @staticmethod
    def _coerce_threshold(
        threshold: torch.Tensor, num_query_heads: int
    ) -> torch.Tensor:
        t = threshold.float().contiguous().reshape(-1)
        if t.numel() == 1 and num_query_heads > 1:
            t = t.expand(num_query_heads)
        if t.numel() != num_query_heads:
            raise ValueError(
                f"threshold must be scalar or length H_q={num_query_heads}, got {t.numel()}"
            )
        return t.contiguous()

    @staticmethod
    def _resolve_q_head_to_kv(num_query_heads: int, num_kv_heads: int) -> torch.Tensor:
        if num_query_heads <= 0 or num_kv_heads <= 0:
            raise ValueError(
                f"Invalid head counts: H_q={num_query_heads}, H_kv={num_kv_heads}"
            )
        if num_query_heads == num_kv_heads:
            return torch.arange(num_query_heads, dtype=torch.long)
        if (num_query_heads % num_kv_heads) != 0:
            raise ValueError(
                f"Unsupported head mapping: H_q={num_query_heads} must be divisible by H_kv={num_kv_heads}"
            )
        group_size = num_query_heads // num_kv_heads
        return torch.arange(num_query_heads, dtype=torch.long) // group_size

    # ------------------------------------------------------------------
    # Vectorized path (batched across H heads)
    # ------------------------------------------------------------------

    def _search_vectorized(self, query, threshold, indexer, q_head_to_kv=None):
        """All heads processed in parallel via batched ops."""
        if q_head_to_kv is not None and query.shape[0] != indexer.keys.shape[0]:
            raise ValueError(
                "vectorized_cpp_filter currently requires H_q == H_kv; use fused_v1/v2/v3/v4 for grouped attention"
            )
        H, D = query.shape
        N = indexer.num_keys
        th = threshold.unsqueeze(-1)  # (H, 1)  for broadcasting

        # --- Corner case: single level (flat scan) ---
        if len(indexer.levels) == 1:
            scores = torch.bmm(indexer.keys, query.unsqueeze(-1)).squeeze(-1)
            result = torch.where(scores >= th, scores, torch.zeros_like(scores))
            if self.profiling:
                self.stats["all_keys"].append(N)
                self.stats["active_keys"].append(int((result != 0).sum()))
                self.stats["exact_checks"] = N * H
            return result

        # --- Root level ---
        root = indexer.levels[-1]
        scores = torch.bmm(root.ball_centers, query.unsqueeze(-1)).squeeze(
            -1
        )  # (H, K_root)
        active_mask = (scores + root.ball_radii) >= th  # (H, K_root)

        if self.profiling:
            self.stats["all_keys"].append(root.size)
            self.stats["active_keys"].append(int(active_mask.sum()))

        # --- Intermediate levels (from root-1 down to level 1) ---
        for level in indexer.levels[::-1][1:-1]:
            # Expand active parents to children via gather
            # level.child2parent: (H, C) -> parent idx
            child_active = torch.gather(active_mask, 1, level.child2parent)  # (H, C)

            # Score children (all of them; masking applied after)
            scores = torch.bmm(level.ball_centers, query.unsqueeze(-1)).squeeze(
                -1
            )  # (H, C)
            active_mask = child_active & ((scores + level.ball_radii) >= th)  # (H, C)

            if self.profiling:
                self.stats["all_keys"].append(level.size)
                self.stats["active_keys"].append(int(active_mask.sum()))

        # --- Level-0 expansion (leaves) ---
        level0 = indexer.levels[0]
        leaf_mask = torch.gather(active_mask, 1, level0.child2parent)  # (H, N)

        if self.profiling:
            self.stats["exact_checks"] = int(leaf_mask.sum())

        # --- Exact filter ---
        return self._exact_filter_cpp(indexer.keys, leaf_mask, query, threshold)

    # ----- C++ / OpenMP -----

    def _exact_filter_cpp(self, keys, leaf_mask, query, threshold):
        """H-batched C++/OpenMP kernel via pybind11. Returns (H, N) float scores."""
        k_np = keys.detach().cpu().float().numpy()
        m_np = leaf_mask.detach().cpu().numpy()
        q_np = query.detach().cpu().float().numpy()
        th_np = threshold.detach().cpu().float().numpy()
        result_np = hira_cpp_kernels.exact_filter_mask_batched(k_np, m_np, q_np, th_np)
        return torch.from_numpy(result_np).to(device=keys.device, dtype=query.dtype)

    # ------------------------------------------------------------------
    # Zero-copy fused search (PyTorch C++ extension)
    # ------------------------------------------------------------------

    def _search_fused(
        self,
        query: torch.Tensor,
        threshold: torch.Tensor,
        indexer: CPUIndexer,
        q_head_to_kv: torch.Tensor,
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

        # Prepare query: (1, H, 1, D) -> (H, D), normalise
        # query = query.squeeze(0).squeeze(-2).float().contiguous()
        # query = query / query.norm(dim=-1, keepdim=True)

        # threshold = threshold.float().contiguous()

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
        )

    def _search_fused_v2(
        self,
        query: torch.Tensor,
        threshold: torch.Tensor,
        indexer: CPUIndexer,
        q_head_to_kv: torch.Tensor,
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
        # Prepare query: (1, H, 1, D) -> (H, D), normalise
        # query = query.squeeze(0).squeeze(-2).float().contiguous()
        # query = query / query.norm(dim=-1, keepdim=True)

        # threshold = threshold.float().contiguous()

        centers_list, radii_list, c2p_list, sizes_list = self._prepare_ext_level_data(
            indexer=indexer, with_adjacency=False
        )

        return hira_torch_ext_v2.fused_tree_search_v2(
            indexer.keys.float().contiguous(),
            query,
            threshold,
            centers_list,
            radii_list,
            c2p_list,
            sizes_list,
            q_head_to_kv.long().contiguous(),
        )

    def _search_fused_v3(
        self,
        query: torch.Tensor,
        threshold: torch.Tensor,
        indexer: CPUIndexer,
        q_head_to_kv: torch.Tensor,
    ):
        """Fused search using the v3 zero-copy PyTorch C++ extension.

        v3 uses active-parent traversal with precomputed parent->children
        adjacency to avoid scanning all children at each level.
        """
        # from hira.kernels.cpp.torch_ext_loader import hira_torch_ext_v3

        # query = query.squeeze(0).squeeze(-2).float().contiguous()
        # query = query / query.norm(dim=-1, keepdim=True)
        # threshold = threshold.float().contiguous()

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
        )

    def _search_fused_v4(
        self,
        query: torch.Tensor,
        threshold: torch.Tensor,
        indexer: CPUIndexer,
        q_head_to_kv: torch.Tensor,
    ):
        """Fused search using the v4 zero-copy PyTorch C++ extension.

        v4 is specialized for D=128 and uses active-parent traversal +
        flattened-candidate exact filtering for better load balance.
        """
        # from hira.kernels.cpp.torch_ext_loader import hira_torch_ext_v4

        # query = query.squeeze(0).squeeze(-2).float().contiguous()
        # query = query / query.norm(dim=-1, keepdim=True)
        # threshold = threshold.float().contiguous()

        if query.shape[-1] != 128:
            raise ValueError(
                f"v4 kernel is specialized for D=128, got D={query.shape[-1]}"
            )

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
        )

    def _search_exact_torch(
        self,
        query: torch.Tensor,
        threshold: torch.Tensor,
        indexer: CPUIndexer,
        q_head_to_kv=None,
    ):
        """Vectorized search using the zero-copy exact-filter kernel.

        Tree traversal is done in PyTorch (same as ``search()`` with
        ``vectorize=True``), but the leaf-level exact filter runs through
        the zero-copy C++ kernel — no numpy copies.

        Args:
            query:     ``(1, H, 1, D)`` query tensor.
            threshold: ``(H,)`` per-head threshold tensor.
            indexer:   Built :class:`CPUIndexer`.

        Returns:
            ``(H, N)`` float tensor – qualifying scores, 0.0 elsewhere.
        """
        # from hira.kernels.cpp.torch_ext_loader import hira_torch_ext

        # (1, H, 1, D) -> (H, D), normalise
        # query = query.squeeze(0).squeeze(-2).float().contiguous()
        # query = query / query.norm(dim=-1, keepdim=True)

        # threshold = threshold.float().contiguous()
        th = threshold.unsqueeze(-1)  # (H, 1)

        H, D = query.shape
        N = indexer.num_keys

        # --- Corner case: single level ---
        if len(indexer.levels) == 1:
            leaf_mask = torch.ones(H, N, dtype=torch.bool)
            return hira_torch_ext.exact_filter(
                indexer.keys.float().contiguous(), leaf_mask, query, threshold
            )

        # --- Root ---
        root = indexer.levels[-1]
        scores = torch.bmm(root.ball_centers, query.unsqueeze(-1)).squeeze(-1)
        active_mask = (scores + root.ball_radii) >= th

        # --- Intermediate levels ---
        for level in indexer.levels[::-1][1:-1]:
            child_active = torch.gather(active_mask, 1, level.child2parent)
            scores = torch.bmm(level.ball_centers, query.unsqueeze(-1)).squeeze(-1)
            active_mask = child_active & ((scores + level.ball_radii) >= th)

        # --- Level-0 expansion ---
        level0 = indexer.levels[0]
        leaf_mask = torch.gather(active_mask, 1, level0.child2parent)

        # --- Zero-copy exact filter ---
        return hira_torch_ext.exact_filter(
            indexer.keys.float().contiguous(), leaf_mask, query, threshold
        )

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    def _reset_stats(self):
        self.stats = {"all_keys": [], "active_keys": [], "exact_checks": 0}

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

    def print_stats(self):
        if not self.profiling:
            print("Profiling not enabled")
            return

        print("HIRA CPUSearcher profiling statistics:")
        for i, total in enumerate(self.stats["all_keys"]):
            active = self.stats["active_keys"][i]
            print(f"  Level {i} | total: {total} | active: {active}")
        print(f"  Exact checks: {self.stats['exact_checks']}")
