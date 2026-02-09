import pytest
import torch


from hira.index.indexer import CUDAIndexer  # type: ignore
from hira.index.searcher import CUDASearcher


def _valid_rows_mask(x: torch.Tensor, *, pad_value: float) -> torch.Tensor:
    return ~torch.all(x == float(pad_value), dim=-1)


def _choose_block_c(branching_factor: int) -> int:
    # Must satisfy: BLOCK_C <= branch and branch % BLOCK_C == 0
    for c in (16, 8, 4, 2, 1):
        if c <= branching_factor and (branching_factor % c == 0):
            return c
    raise ValueError(f"No valid BLOCK_C for branch={branching_factor}")


def _row_bytes(row: torch.Tensor) -> bytes:
    # Exact match by bytes is safe since keys are copied verbatim.
    return row.detach().cpu().contiguous().numpy().tobytes()


def _keyset_from_scores(
    *,
    children: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
    pad_value: float,
) -> set[bytes]:
    assert children.ndim == 2 and scores.ndim == 1
    valid = _valid_rows_mask(children, pad_value=pad_value)
    keep = (scores >= threshold) & valid
    idx = torch.nonzero(keep, as_tuple=False).view(-1)
    out: set[bytes] = set()
    # Keep test sizes modest; a Python loop is fine here.
    for i in idx.tolist():
        out.add(_row_bytes(children[i]))
    return out


def _keyset_from_matmul(
    *,
    keys: torch.Tensor,
    q: torch.Tensor,
    threshold: float,
) -> set[bytes]:
    scores = (keys @ q).detach()
    idx = torch.nonzero(scores >= threshold, as_tuple=False).view(-1)
    out: set[bytes] = set()
    for i in idx.tolist():
        out.add(_row_bytes(keys[i]))
    return out


def _indexed_keyset(children: torch.Tensor, *, pad_value: float) -> set[bytes]:
    valid = _valid_rows_mask(children, pad_value=pad_value)
    idx = torch.nonzero(valid, as_tuple=False).view(-1)
    out: set[bytes] = set()
    for i in idx.tolist():
        out.add(_row_bytes(children[i]))
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestCUDAIndexerUpdate:
    def test_update_empty_is_noop(self):
        torch.manual_seed(0)
        bf = 16
        d = 32
        pad_value = -1.0e9
        n = bf * 8
        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=3,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        parents_before = indexer.parents.clone()
        children_before = indexer.children.clone()
        radii_before = indexer.parent_radii.clone()

        indexer.update(torch.empty((0, d), device="cuda", dtype=torch.float32))
        torch.testing.assert_close(indexer.parents, parents_before)
        torch.testing.assert_close(indexer.children, children_before)
        torch.testing.assert_close(indexer.parent_radii, radii_before)

    def test_update_invalid_shape_raises(self):
        torch.manual_seed(0)
        bf = 16
        d = 32
        pad_value = -1.0e9
        keys = torch.randn(bf * 8, d, device="cuda", dtype=torch.float32)
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=3,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        with pytest.raises(ValueError, match=r"new_keys must be \(n,d\)"):
            indexer.update(torch.zeros(2, 3, 4, device="cuda"))

    def test_update_two_levels_fills_and_appends_and_radii_match(self):
        torch.manual_seed(0)
        bf = 16
        d = 32
        pad_value = -1.0e9

        # Pick n not divisible by bf to ensure padded slots exist.
        n = 257
        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)

        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=5,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        before_valid = (
            _valid_rows_mask(indexer.children, pad_value=pad_value).sum().item()
        )
        assert before_valid > 0

        # Add enough keys to fill remaining padding and force at least one append.
        new_keys = torch.randn(100, d, device="cuda", dtype=torch.float32)
        indexer.update(new_keys)

        after_valid = (
            _valid_rows_mask(indexer.children, pad_value=pad_value).sum().item()
        )
        assert after_valid == before_valid + new_keys.shape[0]

        # Layout/radii consistency.
        expected = indexer._compute_parent_radii_from_layout()
        torch.testing.assert_close(indexer.parent_radii, expected, atol=1e-5, rtol=0.0)

        assert indexer.buffer is not None
        assert indexer.buffer.shape[0] == indexer.children.shape[0]

    def test_update_two_levels_small_update_increases_nonpad_and_radii_consistent(self):
        torch.manual_seed(42)
        bf = 16
        d = 64
        pad_value = -1.0e9

        # n not divisible by bf => build leaves padding.
        n = bf * 8 + 3
        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=3,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        before_valid = (
            _valid_rows_mask(indexer.children, pad_value=pad_value).sum().item()
        )
        # Small update: should always insert all keys and keep radii consistent.
        new_keys = torch.randn(5, d, device="cuda", dtype=torch.float32)
        indexer.update(new_keys)

        after_valid = (
            _valid_rows_mask(indexer.children, pad_value=pad_value).sum().item()
        )
        assert after_valid == before_valid + new_keys.shape[0]

        expected = indexer._compute_parent_radii_from_layout()
        torch.testing.assert_close(indexer.parent_radii, expected, atol=1e-5, rtol=0.0)

    def test_update_two_levels_overflow_appends_parents(self):
        torch.manual_seed(7)
        bf = 16
        d = 64
        pad_value = -1.0e9

        # Large enough that update overflow is likely; we will compute exact free slots.
        n = bf * 32
        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=3,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)
        parents_before = indexer.parents.shape[0]
        children_before = indexer.children.shape[0]

        valid_before = (
            _valid_rows_mask(indexer.children, pad_value=pad_value).sum().item()
        )
        free = int(children_before - valid_before)
        # Force capacity overflow deterministically.
        m_new = free + 1
        new_keys = torch.randn(m_new, d, device="cuda", dtype=torch.float32)
        indexer.update(new_keys)

        assert indexer.parents.shape[0] > parents_before
        assert indexer.children.shape[0] > children_before
        dp = indexer.parents.shape[0] - parents_before
        dc = indexer.children.shape[0] - children_before
        assert dp >= 1
        assert dc >= bf
        assert dc % bf == 0
        assert dc == dp * bf
        valid_after = (
            _valid_rows_mask(indexer.children, pad_value=pad_value).sum().item()
        )
        assert valid_after == valid_before + m_new

        expected = indexer._compute_parent_radii_from_layout()
        torch.testing.assert_close(indexer.parent_radii, expected, atol=1e-5, rtol=0.0)

    def test_update_three_levels_updates_layout_and_radii(self):
        torch.manual_seed(1)
        bf = 8
        d = 16
        pad_value = -1.0e9

        # Ensure both parent slots and child slots have padding initially.
        n = 513
        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)

        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.THREE_LEVELS,
            max_iterations=5,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        before_valid = (
            _valid_rows_mask(indexer.children, pad_value=pad_value).sum().item()
        )
        assert before_valid > 0

        # Add enough keys to require new parents and (likely) new grandparents.
        new_keys = torch.randn(200, d, device="cuda", dtype=torch.float32)
        indexer.update(new_keys)

        after_valid = (
            _valid_rows_mask(indexer.children, pad_value=pad_value).sum().item()
        )
        assert after_valid == before_valid + new_keys.shape[0]

        # Radii consistency with layout recomputation.
        pr_expected = indexer._compute_parent_radii_from_layout()
        torch.testing.assert_close(
            indexer.parent_radii, pr_expected, atol=1e-5, rtol=0.0
        )

        gp_expected = indexer._compute_grandparent_radii_from_layout()
        torch.testing.assert_close(
            indexer.grand_parent_radii, gp_expected, atol=1e-4, rtol=0.0
        )

        assert indexer.buffer is not None
        assert indexer.buffer.shape[0] == indexer.children.shape[0]

    def test_update_three_levels_small_update_increases_nonpad_and_radii_consistent(
        self,
    ):
        torch.manual_seed(123)
        bf = 8
        d = 32
        pad_value = -1.0e9

        # Choose n not divisible by bf^2 so build leaves padding.
        n = bf * bf * 8 + 5
        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.THREE_LEVELS,
            max_iterations=3,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        before_valid = (
            _valid_rows_mask(indexer.children, pad_value=pad_value).sum().item()
        )

        # Small update should fill existing child slots only.
        new_keys = torch.randn(10, d, device="cuda", dtype=torch.float32)
        indexer.update(new_keys)

        after_valid = (
            _valid_rows_mask(indexer.children, pad_value=pad_value).sum().item()
        )
        assert after_valid == before_valid + new_keys.shape[0]

        pr_expected = indexer._compute_parent_radii_from_layout()
        torch.testing.assert_close(
            indexer.parent_radii, pr_expected, atol=1e-5, rtol=0.0
        )
        gp_expected = indexer._compute_grandparent_radii_from_layout()
        torch.testing.assert_close(
            indexer.grand_parent_radii, gp_expected, atol=1e-4, rtol=0.0
        )

    def test_update_three_levels_overflow_appends_new_grandparents_when_full(self):
        torch.manual_seed(999)
        bf = 8
        d = 32
        pad_value = -1.0e9

        # Deterministically force overflow based on computed free slots.
        n = bf * bf * 16  # multiple of bf^2
        keys = torch.randn(n, d, device="cuda", dtype=torch.float32)
        indexer = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.THREE_LEVELS,
            max_iterations=3,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)
        gp_before = indexer.grand_parents.shape[0]
        p_before = indexer.parents.shape[0]
        c_before = indexer.children.shape[0]

        valid_before = (
            _valid_rows_mask(indexer.children, pad_value=pad_value).sum().item()
        )
        free = int(c_before - valid_before)
        m_new = free + 1
        new_keys = torch.randn(m_new, d, device="cuda", dtype=torch.float32)
        indexer.update(new_keys)

        assert indexer.grand_parents.shape[0] > gp_before
        assert indexer.parents.shape[0] > p_before
        assert indexer.children.shape[0] > c_before

        dgp = indexer.grand_parents.shape[0] - gp_before
        dp = indexer.parents.shape[0] - p_before
        dc = indexer.children.shape[0] - c_before
        assert dgp >= 1
        assert dp % bf == 0
        assert dp == dgp * bf
        assert dc % (bf * bf) == 0
        assert dc == dgp * bf * bf
        valid_after = (
            _valid_rows_mask(indexer.children, pad_value=pad_value).sum().item()
        )
        assert valid_after == valid_before + m_new

        pr_expected = indexer._compute_parent_radii_from_layout()
        torch.testing.assert_close(
            indexer.parent_radii, pr_expected, atol=1e-5, rtol=0.0
        )
        gp_expected = indexer._compute_grandparent_radii_from_layout()
        torch.testing.assert_close(
            indexer.grand_parent_radii, gp_expected, atol=1e-4, rtol=0.0
        )

    def test_incremental_update_quality_similar_to_build_two_levels(self):
        torch.manual_seed(2024)
        bf = 16
        d = 128
        pad_value = -1.0e9
        total = bf * 256
        base = bf * 64

        keys = torch.randn(total, d, device="cuda", dtype=torch.float32)
        keys_base = keys[:base].contiguous()
        keys_add = keys[base:].contiguous()

        index_full = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=5,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        index_inc = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.TWO_LEVELS,
            max_iterations=5,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys_base)

        valid_before = (
            _valid_rows_mask(index_inc.children, pad_value=pad_value).sum().item()
        )
        index_inc.update(keys_add)

        # Update must insert all new keys (even if build drops some base keys).
        valid_after = (
            _valid_rows_mask(index_inc.children, pad_value=pad_value).sum().item()
        )
        assert valid_after == valid_before + keys_add.shape[0]

        searcher = CUDASearcher(block_c=_choose_block_c(bf))
        thresholds = [0.0, 0.2]
        n_queries = 8

        indexed_full = _indexed_keyset(index_full.children, pad_value=pad_value)
        indexed_inc = _indexed_keyset(index_inc.children, pad_value=pad_value)
        common_indexed = indexed_full & indexed_inc

        for qi in range(n_queries):
            q = torch.randn(d, device="cuda", dtype=torch.float32)
            q = q / q.norm(p=2)
            for t in thresholds:
                out_full = searcher.search(q, t, index_full)
                out_inc = searcher.search(q, t, index_inc)

                brute = _keyset_from_matmul(keys=keys, q=q, threshold=t)
                got_full = _keyset_from_scores(
                    children=index_full.children,
                    scores=out_full,
                    threshold=t,
                    pad_value=pad_value,
                )
                got_inc = _keyset_from_scores(
                    children=index_inc.children,
                    scores=out_inc,
                    threshold=t,
                    pad_value=pad_value,
                )

                # Similarity between incremental and full rebuild, restricted to keys
                # that both indices actually contain.
                got_full_common = got_full & common_indexed
                got_inc_common = got_inc & common_indexed
                if len(got_full_common) > 0:
                    overlap = len(got_full_common & got_inc_common) / len(
                        got_full_common
                    )
                    assert overlap >= 0.9

                # Conditional recall vs brute force, restricted to keys indexed by each.
                brute_full = brute & indexed_full
                brute_inc = brute & indexed_inc
                if len(brute_full) > 0:
                    rec_full = len(got_full & brute_full) / len(brute_full)
                    assert rec_full >= 0.9
                else:
                    rec_full = 1.0
                if len(brute_inc) > 0:
                    rec_inc = len(got_inc & brute_inc) / len(brute_inc)
                    assert rec_inc >= 0.9
                else:
                    rec_inc = 1.0
                assert abs(rec_full - rec_inc) <= 0.10

    def test_incremental_update_quality_similar_to_build_three_levels(self):
        torch.manual_seed(2025)
        bf = 8
        d = 128
        pad_value = -1.0e9
        total = bf * bf * 128
        base = bf * bf * 32

        keys = torch.randn(total, d, device="cuda", dtype=torch.float32)
        keys_base = keys[:base].contiguous()
        keys_add = keys[base:].contiguous()

        index_full = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.THREE_LEVELS,
            max_iterations=5,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys)

        index_inc = CUDAIndexer(
            depth=CUDAIndexer.DEPTH.THREE_LEVELS,
            max_iterations=5,
            branching_factor=bf,
            pad_value=pad_value,
        ).build(keys_base)

        valid_before = (
            _valid_rows_mask(index_inc.children, pad_value=pad_value).sum().item()
        )
        index_inc.update(keys_add)

        valid_after = (
            _valid_rows_mask(index_inc.children, pad_value=pad_value).sum().item()
        )
        assert valid_after == valid_before + keys_add.shape[0]

        searcher = CUDASearcher(block_c=_choose_block_c(bf))
        thresholds = [0.0, 0.2]
        n_queries = 6

        indexed_full = _indexed_keyset(index_full.children, pad_value=pad_value)
        indexed_inc = _indexed_keyset(index_inc.children, pad_value=pad_value)
        common_indexed = indexed_full & indexed_inc

        for qi in range(n_queries):
            q = torch.randn(d, device="cuda", dtype=torch.float32)
            q = q / q.norm(p=2)
            for t in thresholds:
                out_full = searcher.search(q, t, index_full)
                out_inc = searcher.search(q, t, index_inc)

                brute = _keyset_from_matmul(keys=keys, q=q, threshold=t)
                got_full = _keyset_from_scores(
                    children=index_full.children,
                    scores=out_full,
                    threshold=t,
                    pad_value=pad_value,
                )
                got_inc = _keyset_from_scores(
                    children=index_inc.children,
                    scores=out_inc,
                    threshold=t,
                    pad_value=pad_value,
                )

                got_full_common = got_full & common_indexed
                got_inc_common = got_inc & common_indexed
                if len(got_full_common) > 0:
                    overlap = len(got_full_common & got_inc_common) / len(
                        got_full_common
                    )
                    assert overlap >= 0.85

                brute_full = brute & indexed_full
                brute_inc = brute & indexed_inc
                if len(brute_full) > 0:
                    rec_full = len(got_full & brute_full) / len(brute_full)
                    assert rec_full >= 0.85
                else:
                    rec_full = 1.0
                if len(brute_inc) > 0:
                    rec_inc = len(got_inc & brute_inc) / len(brute_inc)
                    assert rec_inc >= 0.85
                else:
                    rec_inc = 1.0
                assert abs(rec_full - rec_inc) <= 0.10
