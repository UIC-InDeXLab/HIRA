import pytest
import torch
import faiss
from hira.index.indexer import CUDAIndexer  # type: ignore
from hira.index.searcher import CUDASearcher


def _normalize_query(q: torch.Tensor) -> torch.Tensor:
    return q / torch.norm(q, p=2)


def _brute_force_halfspace_idx(
    keys: torch.Tensor, q: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Brute-force halfspace search over the *given* key matrix.

    Note: CUDA kernels gate parents via strict `>` comparisons, but the search API
    is typically interpreted as `score >= threshold`. We use `>=` here and avoid
    exact-boundary thresholds in tests.
    """
    q = _normalize_query(q)
    scores = keys @ q
    return (scores >= threshold).nonzero(as_tuple=True)[0]


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _valid_rows_mask(keys: torch.Tensor, *, pad_value: float) -> torch.Tensor:
    """Mask for non-padded rows (rows that correspond to real keys)."""
    return ~torch.all(keys == float(pad_value), dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestCUDAIndexer:
    """End-to-end tests for CUDAIndexer (no monkeypatching)."""

    def _assert_key_accounting_exact(
        self, *, keys: torch.Tensor, built_children: torch.Tensor, pad_value: float
    ) -> None:
        """Every original key appears exactly once among non-padded children."""
        assert keys.ndim == 2 and built_children.ndim == 2

        keys_cpu = keys.detach().cpu()
        children_cpu = built_children.detach().cpu()

        valid = ~torch.all(children_cpu == float(pad_value), dim=-1)
        non_padded = children_cpu[valid]

        assert (
            non_padded.shape[0] == keys_cpu.shape[0]
        ), f"Expected {keys_cpu.shape[0]} non-padded children, got {non_padded.shape[0]}"

        used = torch.zeros(keys_cpu.shape[0], dtype=torch.bool)
        for row in non_padded:
            eq = torch.all(keys_cpu == row[None, :], dim=-1)
            idxs = torch.nonzero(eq, as_tuple=False).view(-1)
            assert idxs.numel() == 1, "Child row not found uniquely in original keys"
            idx = int(idxs.item())
            assert not used[idx], "Duplicate key found in children"
            used[idx] = True

        assert bool(used.all().item()), "Some original keys were missing"

    def test_invalid_inputs_raise(self):
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=1,
            branching_factor=2,
        )

        with pytest.raises(ValueError, match=r"keys must be \(n,d\)"):
            indexer.build(torch.zeros(2, 3, 4, device="cuda"))

    def test_build_two_levels_sets_fields_and_radii(self):
        torch.manual_seed(0)
        bf = 16
        n = 400
        d = 32
        pad_value = -1.0e9
        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)

        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=10,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        assert indexer.dim == d
        assert indexer.parents is not None
        assert indexer.children is not None
        assert indexer.parent_radii is not None
        assert indexer.grand_parents is None
        assert indexer.grand_parent_radii is None

        m = max(1, n // bf)
        assert indexer.parents.shape == (m, d)
        assert indexer.children.shape == (m * bf, d)
        assert indexer.parent_radii.shape == (m,)
        assert torch.all(indexer.parent_radii >= 0)

        # Radii match the internal layout computation.
        expected = indexer._compute_parent_radii_from_layout()
        torch.testing.assert_close(indexer.parent_radii, expected, atol=0.0, rtol=0.0)

    def test_build_three_levels_sets_fields_and_radii(self):
        torch.manual_seed(0)
        bf = 16
        n = 4000
        d = 64
        pad_value = -1.0e9
        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)

        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.THREE_LEVELS,
            max_iterations=10,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        assert indexer.dim == d
        assert indexer.grand_parents is not None
        assert indexer.parents is not None
        assert indexer.children is not None
        assert indexer.parent_radii is not None
        assert indexer.grand_parent_radii is not None

        m = max(1, n // bf)
        g = max(1, m // bf)
        assert indexer.grand_parents.shape == (g, d)
        assert indexer.parents.shape == (g * bf, d)
        assert indexer.children.shape == (g * bf * bf, d)
        assert indexer.parent_radii.shape == (g * bf,)
        assert indexer.grand_parent_radii.shape == (g,)
        assert torch.all(indexer.parent_radii >= 0)
        assert torch.all(indexer.grand_parent_radii >= 0)

        expected_parent = indexer._compute_parent_radii_from_layout()
        torch.testing.assert_close(
            indexer.parent_radii, expected_parent, atol=0.0, rtol=0.0
        )
        expected_gp = indexer._compute_grandparent_radii_from_layout()
        torch.testing.assert_close(
            indexer.grand_parent_radii, expected_gp, atol=0.0, rtol=0.0
        )

    def _make_clustered_keys_2level(self, *, m: int, bf: int, d: int) -> torch.Tensor:
        """Create m clusters with exactly bf points each (N=m*bf)."""
        device = "cuda"
        # Cluster centers far apart on different dimensions.
        centers = torch.zeros((m, d), device=device, dtype=torch.float32)
        for i in range(m):
            centers[i, i % d] = 1000.0 + 10.0 * i
        # Unique per-point deterministic offsets so points are unique.
        keys = []
        for i in range(m):
            for j in range(bf):
                off = torch.zeros((d,), device=device, dtype=torch.float32)
                off[(i + 1) % d] = (j + 1) * 1e-3
                keys.append(centers[i] + off)
        return torch.stack(keys, dim=0)

    def _make_clustered_keys_3level(self, *, g: int, bf: int, d: int) -> torch.Tensor:
        """Create hierarchical data: g grand-clusters; each has bf parent-clusters; each parent has bf children.

        Total N = g * bf * bf.
        """
        device = "cuda"
        # Grandparent centers *very* widely separated to make gp clustering stable/balanced.
        gp_centers = torch.zeros((g, d), device=device, dtype=torch.float32)
        for j in range(g):
            gp_centers[j, j % d] = 1.0e6 * (j + 1)

        keys = []
        parent_global = 0
        for j in range(g):
            for p_in_gp in range(bf):
                # Parent centers around each grandparent.
                parent_center = gp_centers[j].clone()
                parent_center[(j + 1) % d] += 10.0 * (p_in_gp + 1)

                for c in range(bf):
                    # Deterministic unique offsets around each parent.
                    off = torch.zeros((d,), device=device, dtype=torch.float32)
                    off[(j + 2) % d] = (c + 1) * 1e-3
                    off[(j + 3) % d] = (parent_global + 1) * 1e-4
                    keys.append(parent_center + off)

                parent_global += 1

        return torch.stack(keys, dim=0)

    def test_two_level_ordering_matches_nearest_centroid(self):
        torch.manual_seed(0)
        # Use a reasonably large bf to reduce k-means instability.
        bf = 64
        m = 8
        d = 16
        keys = self._make_clustered_keys_2level(m=m, bf=bf, d=d)
        n = keys.shape[0]
        assert n == m * bf

        pad_value = -1.0e9
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=25,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        parents = indexer.parents
        children = indexer.children
        assert parents is not None and children is not None
        assert parents.shape == (m, d)
        assert children.shape == (m * bf, d)

        # Ordering/layout check using only final build outputs:
        # each contiguous child block should be closest (in aggregate) to its matching parent.
        child_blocks = children.view(m, bf, d)
        block_means = child_blocks.mean(dim=1)  # (m,d)
        dist = torch.cdist(block_means, parents)  # (m,m)
        assigned = torch.argmin(dist, dim=1)
        assert torch.equal(assigned, torch.arange(m, device=assigned.device))

    def test_two_level_strict_child_assignments_and_key_accounting(self):
        """Strict check: each non-padded child in block p is nearest to parent p, and all keys appear once."""
        torch.manual_seed(0)
        # Use a large enough N to avoid FAISS k-means small-sample issues.
        bf = 64
        m = 8
        d = 16
        pad_value = -1.0e9

        # Deterministic, well-separated equal clusters.
        keys = self._make_clustered_keys_2level(m=m, bf=bf, d=d)

        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=25,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        parents = indexer.parents
        children = indexer.children
        assert parents is not None and children is not None
        assert parents.shape == (m, d)
        assert children.shape == (m * bf, d)

        # End-to-end key accounting
        self._assert_key_accounting_exact(
            keys=keys, built_children=children, pad_value=pad_value
        )

        # Strict nearest-parent assignment
        blocks = children.view(m, bf, d)
        for p in range(m):
            block = blocks[p]
            dists = torch.cdist(block, parents)
            assigned = torch.argmin(dists, dim=1)
            assert torch.all(assigned == p), "Child not nearest to its block parent"

    def test_two_level_padding_layout_when_n_lt_bf(self):
        """When n < bf, output must still be a single parent with a bf-sized children block, padded per selection."""
        torch.manual_seed(0)
        bf = 32
        n = 7
        d = 8
        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)

        pad_value = -1.0e9
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=10,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        parents = indexer.parents
        children = indexer.children
        assert parents is not None and children is not None

        assert parents.shape == (1, d)
        assert children.shape == (bf, d)

        valid = ~torch.all(children == float(pad_value), dim=-1)
        assert int(valid.sum().item()) == n

        # Every non-padded row must exactly match one of the input keys.
        non_padded = children[valid]
        min_d = torch.cdist(non_padded, keys).min(dim=1).values
        assert torch.all(min_d == 0)

    def test_three_level_padding_layout_small_n(self):
        """3-level build should preserve gp->parent and parent->child block layout even with heavy padding."""
        torch.manual_seed(0)
        bf = 16
        n = 50  # m=max(1,n//bf)=3; g=max(1,m//bf)=1 => lots of padding.
        d = 16
        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)

        pad_value = -1.0e9
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.THREE_LEVELS,
            max_iterations=10,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        gp = indexer.grand_parents
        parents_reordered = indexer.parents
        children_reordered = indexer.children
        assert (
            gp is not None
            and parents_reordered is not None
            and children_reordered is not None
        )

        assert gp.shape == (1, d)
        assert parents_reordered.shape == (bf, d)
        assert children_reordered.shape == (bf * bf, d)

        # Parent slots beyond the true m should be padded.
        m = max(1, n // bf)
        parents_valid = ~torch.all(parents_reordered == float(pad_value), dim=-1)
        assert int(parents_valid.sum().item()) == m

        # Any padded parent slot must have an entirely padded child block.
        child_blocks = children_reordered.view(bf, bf, d)
        padded_parent_idx = torch.nonzero(~parents_valid, as_tuple=False).view(-1)
        if padded_parent_idx.numel() > 0:
            blocks = child_blocks[padded_parent_idx]
            assert torch.all(torch.all(blocks == float(pad_value), dim=-1))

    def test_three_level_build_outputs_no_padding_when_n_equals_g_bf_bf(self):
        torch.manual_seed(0)
        # Use bf >= ~39 to reduce k-means instability at both levels.
        bf = 40
        g = 3
        d = 16
        keys = self._make_clustered_keys_3level(g=g, bf=bf, d=d)
        n = keys.shape[0]
        assert n == g * bf * bf

        pad_value = -1.0e9
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.THREE_LEVELS,
            max_iterations=25,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        gp = indexer.grand_parents
        parents_reordered = indexer.parents
        children_reordered = indexer.children
        assert (
            gp is not None
            and parents_reordered is not None
            and children_reordered is not None
        )

        assert gp.shape == (g, d)
        assert parents_reordered.shape == (g * bf, d)
        assert children_reordered.shape == (g * bf * bf, d)

    def test_three_level_strict_parent_and_child_assignments_and_key_accounting(self):
        """Strict check: parents in gp-block are nearest to that grandparent; children in parent-block nearest to parent;
        plus no missing/duplicate keys among non-padded children.
        """
        torch.manual_seed(0)
        bf = 2
        g = 2
        d = 8
        pad_value = -1.0e9

        # Hierarchical-separated construction: total keys = g*bf*bf.
        keys = []
        parent_global = 0
        for gp_idx in range(g):
            gp_center = torch.zeros((d,), device="cuda", dtype=torch.float32)
            gp_center[gp_idx % d] = 1.0e6 * (gp_idx + 1)
            for p_in_gp in range(bf):
                parent_center = gp_center.clone()
                parent_center[(gp_idx + 1) % d] += 1.0e3 * (p_in_gp + 1)
                for c in range(bf):
                    off = torch.zeros((d,), device="cuda", dtype=torch.float32)
                    off[(gp_idx + 2) % d] = (c + 1) * 1e-2
                    off[(gp_idx + 3) % d] = (parent_global + 1) * 1e-3
                    keys.append(parent_center + off)
                parent_global += 1
        keys = torch.stack(keys, dim=0)

        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.THREE_LEVELS,
            max_iterations=25,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        gp = indexer.grand_parents
        parents = indexer.parents
        children = indexer.children
        assert gp is not None and parents is not None and children is not None

        assert gp.shape == (g, d)
        assert parents.shape == (g * bf, d)
        assert children.shape == (g * bf * bf, d)

        # End-to-end key accounting
        self._assert_key_accounting_exact(
            keys=keys, built_children=children, pad_value=pad_value
        )

        # Strict grandparent->parent assignment by gp block
        parent_blocks = parents.view(g, bf, d)
        for gp_idx in range(g):
            block = parent_blocks[gp_idx]
            dists = torch.cdist(block, gp)
            assigned = torch.argmin(dists, dim=1)
            assert torch.all(assigned == gp_idx), "Parent not nearest to its gp block"

        # Strict parent->child assignment by parent block
        p_total = g * bf
        child_blocks = children.view(p_total, bf, d)
        for p in range(p_total):
            block = child_blocks[p]
            dists = torch.cdist(block, parents)
            assigned = torch.argmin(dists, dim=1)
            assert torch.all(assigned == p), "Child not nearest to its parent block"

    def test_radii_functions_match_manual_formula_after_build(self):
        """Explicitly validate the refine-ball-style radii formulas on the built layout."""
        torch.manual_seed(0)
        bf = 8
        n = 1200
        d = 32
        pad_value = -1.0e9
        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)

        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.THREE_LEVELS,
            max_iterations=10,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        g, _ = indexer.grand_parents.shape

        # Parent radii: max ||child - parent|| per parent block, ignoring padded.
        parents = indexer.parents
        children = indexer.children.view(g * bf, bf, d)
        dists = torch.linalg.norm(children - parents[:, None, :], dim=-1)
        valid = ~torch.all(children == float(pad_value), dim=-1)
        dists = torch.where(
            valid, dists, torch.tensor(float("-inf"), device=dists.device)
        )
        parent_r_expected = torch.max(dists, dim=1).values
        parent_r_expected = torch.where(
            torch.isfinite(parent_r_expected),
            parent_r_expected,
            torch.zeros_like(parent_r_expected),
        )

        torch.testing.assert_close(
            indexer.parent_radii, parent_r_expected, atol=0.0, rtol=0.0
        )

        # Grandparent radii: max (||parent - gp|| + parent_radius) per gp block, ignoring padded.
        gp = indexer.grand_parents
        parents_blocks = parents.view(g, bf, d)
        pr_blocks = indexer.parent_radii.view(g, bf)
        totals = torch.linalg.norm(parents_blocks - gp[:, None, :], dim=-1) + pr_blocks
        valid_gp = ~torch.all(parents_blocks == float(pad_value), dim=-1)
        totals = torch.where(
            valid_gp, totals, torch.tensor(float("-inf"), device=totals.device)
        )
        gp_r_expected = torch.max(totals, dim=1).values
        gp_r_expected = torch.where(
            torch.isfinite(gp_r_expected),
            gp_r_expected,
            torch.zeros_like(gp_r_expected),
        )

        torch.testing.assert_close(
            indexer.grand_parent_radii, gp_r_expected, atol=0.0, rtol=0.0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestCUDASearcherRangeSearch:
    """End-to-end correctness checks for CUDASearcher + CUDAIndexer."""

    @pytest.mark.parametrize(
        "bf,n,d,threshold",
        [
            (16, 1024, 32, 0.25),
            (32, 2048, 64, 0.25),
        ],
    )
    def test_two_level_search_matches_bruteforce(self, bf, n, d, threshold):
        torch.manual_seed(0)
        pad_value = -1.0e9
        assert n % bf == 0, "Choose n multiple of bf to avoid padding"

        if not _is_power_of_two(d):
            pytest.skip("Triton kernels require power-of-two feature dim")

        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=15,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        # Search happens over the reordered/padded children matrix.
        children = indexer.children
        assert children is not None
        valid = _valid_rows_mask(children, pad_value=pad_value)
        assert valid.any(), "Expected at least one non-padded child row"

        q = _normalize_query(torch.randn(d, device="cuda", dtype=torch.float32))
        searcher = CUDASearcher(block_c=min(16, bf))

        out = searcher.search(q, threshold, indexer)
        assert out.shape == (children.shape[0],)

        got = (out >= threshold).nonzero(as_tuple=True)[0]
        # Brute-force only over valid (non-padded) rows, and map back to global indices.
        expected_local = _brute_force_halfspace_idx(children[valid], q, threshold)
        expected = torch.nonzero(valid, as_tuple=True)[0][expected_local]

        # Padded rows represent no key; they should never be returned.
        if got.numel() > 0:
            assert torch.all(valid[got])

        # No false positives: if out[idx] >= t then true score must be >= t.
        true_scores = children @ q
        if got.numel() > 0:
            assert torch.all(true_scores[got] >= threshold - 1e-4)
            torch.testing.assert_close(out[got], true_scores[got], rtol=1e-4, atol=1e-3)

        # Recall should be (near) perfect with correct radii upper bounds.
        if expected.numel() > 0:
            inter = torch.isin(expected, got).sum().item()
            recall = inter / expected.numel()
            assert recall >= 0.99, f"Recall {recall:.3f} < 0.99"

    def test_two_level_extreme_thresholds(self):
        torch.manual_seed(1)
        bf, n, d = 16, 1024, 64
        pad_value = -1.0e9
        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=10,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)
        children = indexer.children
        assert children is not None

        q = _normalize_query(torch.randn(d, device="cuda", dtype=torch.float32))
        scores = children @ q
        searcher = CUDASearcher(block_c=min(16, bf))

        # Very high threshold => empty.
        t_hi = float(scores.max().item() + 1.0)
        out_hi = searcher.search(q, t_hi, indexer)
        got_hi = (out_hi >= t_hi).nonzero(as_tuple=True)[0]
        assert got_hi.numel() == 0

        # Very low threshold => all.
        t_lo = float(scores.min().item() - 1.0)
        out_lo = searcher.search(q, t_lo, indexer)
        got_lo = (out_lo >= t_lo).nonzero(as_tuple=True)[0]
        assert got_lo.numel() == children.shape[0]

    @pytest.mark.parametrize(
        "bf,g,d,threshold",
        [
            (8, 16, 32, 0.25),
            (16, 8, 64, 0.25),
        ],
    )
    def test_three_level_search_matches_bruteforce(self, bf, g, d, threshold):
        torch.manual_seed(2)
        pad_value = -1.0e9
        n = g * bf * bf

        if not _is_power_of_two(d):
            pytest.skip("Triton kernels require power-of-two feature dim")

        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.THREE_LEVELS,
            max_iterations=15,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        children = indexer.children
        assert children is not None
        valid = _valid_rows_mask(children, pad_value=pad_value)
        assert valid.any(), "Expected at least one non-padded child row"

        q = _normalize_query(torch.randn(d, device="cuda", dtype=torch.float32))
        searcher = CUDASearcher(block_c=min(16, bf))

        out = searcher.search(q, threshold, indexer)
        assert out.shape == (children.shape[0],)

        got = (out >= threshold).nonzero(as_tuple=True)[0]
        expected_local = _brute_force_halfspace_idx(children[valid], q, threshold)
        expected = torch.nonzero(valid, as_tuple=True)[0][expected_local]

        if got.numel() > 0:
            assert torch.all(valid[got])

        true_scores = children @ q
        if got.numel() > 0:
            assert torch.all(true_scores[got] >= threshold - 1e-4)
            torch.testing.assert_close(out[got], true_scores[got], rtol=1e-4, atol=1e-3)

        if expected.numel() > 0:
            inter = torch.isin(expected, got).sum().item()
            recall = inter / expected.numel()
            assert recall >= 0.99, f"Recall {recall:.3f} < 0.99"

    def test_two_level_recall_across_thresholds(self):
        """Sweep several threshold types and verify recall stays high.

        Thresholds include:
        - very low: should return all valid keys
        - quantile-based
        - zero
        - very high: should return none
        """
        torch.manual_seed(3)
        bf, n, d = 32, 4096, 64
        pad_value = -1.0e9

        if not _is_power_of_two(d):
            pytest.skip("Triton kernels require power-of-two feature dim")

        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=15,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        children = indexer.children
        assert children is not None
        valid = _valid_rows_mask(children, pad_value=pad_value)
        assert valid.any(), "Expected at least one non-padded child row"

        q = _normalize_query(torch.randn(d, device="cuda", dtype=torch.float32))
        true_scores = children[valid] @ q

        # Build thresholds from distribution to exercise different regimes.
        t_all = float(true_scores.min().item() - 1.0)
        t_none = float(true_scores.max().item() + 1.0)
        t_q10 = float(torch.quantile(true_scores, 0.10).item())
        t_q50 = float(torch.quantile(true_scores, 0.50).item())
        t_q90 = float(torch.quantile(true_scores, 0.90).item())
        thresholds = [t_all, t_q10, 0.0, t_q50, t_q90, t_none]

        searcher = CUDASearcher(block_c=min(16, bf))
        valid_idx = torch.nonzero(valid, as_tuple=True)[0]

        for t in thresholds:
            out = searcher.search(q, float(t), indexer)
            got = ((out >= float(t)) & valid).nonzero(as_tuple=True)[0]

            expected_local = (true_scores >= float(t)).nonzero(as_tuple=True)[0]
            expected = valid_idx[expected_local]

            if expected.numel() == 0:
                assert got.numel() == 0
                continue

            inter = torch.isin(expected, got).sum().item()
            recall = inter / expected.numel()
            assert recall >= 0.99, f"threshold={t:.4f}: recall {recall:.3f} < 0.99"

    def test_three_level_recall_across_thresholds(self):
        """Same as two-level sweep but for three-level index."""
        torch.manual_seed(4)
        bf, g, d = 8, 64, 64
        n = g * bf * bf
        pad_value = -1.0e9

        if not _is_power_of_two(d):
            pytest.skip("Triton kernels require power-of-two feature dim")

        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.THREE_LEVELS,
            max_iterations=15,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        children = indexer.children
        assert children is not None
        valid = _valid_rows_mask(children, pad_value=pad_value)
        assert valid.any(), "Expected at least one non-padded child row"

        q = _normalize_query(torch.randn(d, device="cuda", dtype=torch.float32))
        true_scores = children[valid] @ q

        t_all = float(true_scores.min().item() - 1.0)
        t_none = float(true_scores.max().item() + 1.0)
        t_q10 = float(torch.quantile(true_scores, 0.10).item())
        t_q50 = float(torch.quantile(true_scores, 0.50).item())
        t_q90 = float(torch.quantile(true_scores, 0.90).item())
        thresholds = [t_all, t_q10, 0.0, t_q50, t_q90, t_none]

        searcher = CUDASearcher(block_c=min(16, bf))
        valid_idx = torch.nonzero(valid, as_tuple=True)[0]

        for t in thresholds:
            out = searcher.search(q, float(t), indexer)
            got = ((out >= float(t)) & valid).nonzero(as_tuple=True)[0]

            expected_local = (true_scores >= float(t)).nonzero(as_tuple=True)[0]
            expected = valid_idx[expected_local]

            if expected.numel() == 0:
                assert got.numel() == 0
                continue

            inter = torch.isin(expected, got).sum().item()
            recall = inter / expected.numel()
            assert recall >= 0.99, f"threshold={t:.4f}: recall {recall:.3f} < 0.99"
