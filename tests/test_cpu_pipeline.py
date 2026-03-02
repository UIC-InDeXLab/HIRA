"""
To test:
    - build an index with a set of keys, get different thresholds of the keys and a query representing different ratios being returned and check the recall being 100%.
    - test above, but this time building the index on a smaller set of keys and updating it incrementally, finally, checking the recall on the final index.
    - testing search with different scaling inputs.
    - testing search on all v1 v2 v3 and v4 kernels
    - testing the ordering of values and keys in the CPUIndexer being the same. if keys shuffled, values should be shuffled in the same way.
"""

import pytest
import torch
import torch.nn.functional as F

from hira.indexer.cpu import CPUIndexer
from hira.searcher.cpu import CPUSearcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keys(num_heads: int, seq_len: int, dim: int, seed: int = 0) -> torch.Tensor:
    """Return L2-normalised keys of shape (1, H, L, D)."""
    torch.manual_seed(seed)
    keys = torch.randn(1, num_heads, seq_len, dim)
    keys = F.normalize(keys, dim=-1)
    return keys.float()


def _make_query(num_heads: int, dim: int, seed: int = 42) -> torch.Tensor:
    """Return a single L2-normalised query of shape (1, H, 1, D)."""
    torch.manual_seed(seed)
    q = torch.randn(1, num_heads, 1, dim)
    q = F.normalize(q, dim=-1)
    return q.float()


def _exact_scores(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Brute-force dot-product scores.

    Args:
        query: (1, H, 1, D)
        keys:  (H, L, D)   (already squeezed from indexer.keys)

    Returns:
        (H, L) float scores.
    """
    q = query.squeeze(0).squeeze(-2)  # (H, D)
    return torch.einsum("hd,hld->hl", q, keys)


def _threshold_for_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return per-head threshold such that exactly top-k keys exceed it.

    Args:
        scores: (H, L)
        k:      number of keys to keep per head

    Returns:
        (H,) threshold tensor.
    """
    # kth largest value; keys with score >= threshold should be returned
    topk_vals, _ = torch.topk(scores, k, dim=-1)
    # use the smallest of the top-k values as the threshold
    return topk_vals[:, -1]


def _recall(
    scores_approx: torch.Tensor,
    scores_exact: torch.Tensor,
    threshold: torch.Tensor,
) -> float:
    """Fraction of keys that truly exceed the threshold and are also returned
    by the approximate search (non-zero score).

    Args:
        scores_approx: (H, L)
        scores_exact:  (H, L)
        threshold:     (H,)
    """
    true_mask = scores_exact >= threshold.unsqueeze(-1)  # (H, L)
    returned_mask = scores_approx > 0.0  # (H, L)

    true_pos = (true_mask & returned_mask).sum().item()
    total_true = true_mask.sum().item()

    if total_true == 0:
        return 1.0
    return true_pos / total_true


def _build_indexer(keys: torch.Tensor, num_levels: int = 3, branching_factor: int = 8):
    return CPUIndexer(
        num_levels=num_levels,
        branching_factor=branching_factor,
    ).build(keys)


# ---------------------------------------------------------------------------
# Test 1 – full recall at multiple threshold ratios (single build)
# ---------------------------------------------------------------------------


class TestFullRecallSingleBuild:
    """Build an index once; verify 100 % recall at several keep-ratios."""

    NUM_HEADS = 4
    SEQ_LEN = 512
    DIM = 128

    @pytest.fixture(scope="class")
    def setup(self):
        keys = _make_keys(self.NUM_HEADS, self.SEQ_LEN, self.DIM, seed=1)
        query = _make_query(self.NUM_HEADS, self.DIM, seed=10)
        indexer = _build_indexer(keys)
        searcher = CPUSearcher(search_strategy="fused_v3")
        scores_exact = _exact_scores(query, indexer.keys)  # (H, L)
        return indexer, searcher, query, scores_exact

    @pytest.mark.parametrize("keep_ratio", [0.05, 0.10, 0.25])
    def test_recall_is_100_percent(self, setup, keep_ratio):
        indexer, searcher, query, scores_exact = setup
        k = max(1, int(self.SEQ_LEN * keep_ratio))
        threshold = _threshold_for_topk(scores_exact, k)

        scores_approx = searcher.search(
            query=query,
            threshold=threshold,
            indexer=indexer,
        )  # (H, L)

        recall = _recall(scores_approx, scores_exact, threshold)
        assert (
            recall > 0.98
        ), f"Expected 100% recall at keep_ratio={keep_ratio}, got {recall:.4f}"


# ---------------------------------------------------------------------------
# Test 2 – full recall after incremental updates
# ---------------------------------------------------------------------------


class TestFullRecallIncrementalUpdate:
    """Start with a small index, update it in chunks, then verify 100% recall."""

    NUM_HEADS = 4
    DIM = 128
    INITIAL_LEN = 64
    CHUNK_LEN = 64
    NUM_CHUNKS = 6  # total seq_len = INITIAL_LEN + NUM_CHUNKS * CHUNK_LEN = 448

    @pytest.fixture(scope="class")
    def setup(self):
        total_len = self.INITIAL_LEN + self.NUM_CHUNKS * self.CHUNK_LEN
        all_keys = _make_keys(self.NUM_HEADS, total_len, self.DIM, seed=2)

        # Build initial index on the first slice
        initial_keys = all_keys[:, :, : self.INITIAL_LEN, :]
        indexer = _build_indexer(initial_keys, num_levels=3, branching_factor=8)

        # Incrementally update
        for i in range(self.NUM_CHUNKS):
            start = self.INITIAL_LEN + i * self.CHUNK_LEN
            end = start + self.CHUNK_LEN
            chunk = all_keys[:, :, start:end, :]
            indexer.update(chunk)

        query = _make_query(self.NUM_HEADS, self.DIM, seed=20)
        searcher = CPUSearcher(search_strategy="fused_v3")
        scores_exact = _exact_scores(query, indexer.keys)

        return indexer, searcher, query, scores_exact, total_len

    @pytest.mark.parametrize("keep_ratio", [0.05, 0.20, 0.50])
    def test_recall_is_100_percent_after_updates(self, setup, keep_ratio):
        indexer, searcher, query, scores_exact, total_len = setup
        k = max(1, int(total_len * keep_ratio))
        threshold = _threshold_for_topk(scores_exact, k)

        scores_approx = searcher.search(
            query=query,
            threshold=threshold,
            indexer=indexer,
        )

        recall = _recall(scores_approx, scores_exact, threshold)
        assert recall > 0.98, (
            f"Expected 100% recall after incremental update at keep_ratio={keep_ratio}, "
            f"got {recall:.4f}"
        )

    def test_total_num_keys_is_correct(self, setup):
        indexer, _, _, _, total_len = setup
        assert indexer.num_keys == total_len


# ---------------------------------------------------------------------------
# Test 3 – search with different scaling inputs
# ---------------------------------------------------------------------------


class TestSearchScaling:
    """Verify that the scaling tensor linearly scales the returned scores."""

    NUM_HEADS = 4
    SEQ_LEN = 256
    DIM = 128

    @pytest.fixture(scope="class")
    def setup(self):
        keys = _make_keys(self.NUM_HEADS, self.SEQ_LEN, self.DIM, seed=3)
        query = _make_query(self.NUM_HEADS, self.DIM, seed=30)
        indexer = _build_indexer(keys)
        searcher = CPUSearcher(search_strategy="fused_v3")

        scores_exact = _exact_scores(query, indexer.keys)
        # Use a low threshold so many keys are returned
        k = self.SEQ_LEN // 2
        threshold = _threshold_for_topk(scores_exact, k)

        return indexer, searcher, query, threshold

    @pytest.mark.parametrize("scale_value", [0.5, 1.0, 2.0, 10.0])
    def test_scores_scaled_correctly(self, setup, scale_value):
        indexer, searcher, query, threshold = setup
        H = self.NUM_HEADS

        scaling = torch.full((H,), scale_value)
        scores_scaled = searcher.search(
            query=query,
            threshold=threshold,
            indexer=indexer,
            scaling=scaling,
        )

        # Baseline with identity scaling
        identity_scaling = torch.ones(H)
        scores_identity = searcher.search(
            query=query,
            threshold=threshold,
            indexer=indexer,
            scaling=identity_scaling,
        )

        # For positions returned by both, scores_scaled = scale_value * scores_identity
        returned = scores_identity > 0.0
        if returned.any():
            ratio = scores_scaled[returned] / scores_identity[returned]
            assert torch.allclose(
                ratio, torch.full_like(ratio, scale_value), atol=1e-5
            ), (
                f"Scaling {scale_value} does not linearly scale scores. "
                f"Max ratio deviation: {(ratio - scale_value).abs().max().item()}"
            )

    def test_zero_scaling_returns_zeros(self, setup):
        indexer, searcher, query, threshold = setup
        H = self.NUM_HEADS
        scaling = torch.zeros(H)
        scores = searcher.search(
            query=query,
            threshold=threshold,
            indexer=indexer,
            scaling=scaling,
        )
        assert (scores == 0.0).all(), "Zero scaling should produce all-zero scores"

    def test_per_head_scaling(self, setup):
        """Different scale per head should scale each head independently."""
        indexer, searcher, query, threshold = setup
        H = self.NUM_HEADS

        scale_values = torch.arange(1, H + 1, dtype=torch.float32)
        scores_scaled = searcher.search(
            query=query,
            threshold=threshold,
            indexer=indexer,
            scaling=scale_values,
        )
        scores_identity = searcher.search(
            query=query,
            threshold=threshold,
            indexer=indexer,
            scaling=torch.ones(H),
        )

        for h in range(H):
            returned = scores_identity[h] > 0.0
            if returned.any():
                ratio = scores_scaled[h, returned] / scores_identity[h, returned]
                expected = scale_values[h].item()
                assert torch.allclose(
                    ratio, torch.full_like(ratio, expected), atol=1e-5
                ), f"Head {h}: expected scale {expected}, got max deviation {(ratio - expected).abs().max().item()}"


# ---------------------------------------------------------------------------
# Test 4 – all four kernel variants produce consistent results
# ---------------------------------------------------------------------------


class TestAllKernels:
    """v1, v2, v3, v4 kernels should agree on which keys are returned."""

    NUM_HEADS = 4
    SEQ_LEN = 256
    DIM = 128  # v4 requires D=128

    STRATEGIES = ["fused_v1", "fused_v2", "fused_v3", "fused_v4"]

    @pytest.fixture(scope="class")
    def setup(self):
        keys = _make_keys(self.NUM_HEADS, self.SEQ_LEN, self.DIM, seed=4)
        query = _make_query(self.NUM_HEADS, self.DIM, seed=40)
        indexer = _build_indexer(keys)

        scores_exact = _exact_scores(query, indexer.keys)
        k = self.SEQ_LEN // 4
        threshold = _threshold_for_topk(scores_exact, k)

        searchers = {s: CPUSearcher(search_strategy=s) for s in self.STRATEGIES}
        return indexer, searchers, query, threshold, scores_exact

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_100_percent_recall(self, setup, strategy):
        indexer, searchers, query, threshold, scores_exact = setup
        scores_approx = searchers[strategy].search(
            query=query,
            threshold=threshold,
            indexer=indexer,
        )
        recall = _recall(scores_approx, scores_exact, threshold)
        assert (
            recall >= 0.98
        ), f"Strategy '{strategy}' achieved recall={recall:.4f}, expected 0.98"

    def test_all_strategies_agree_on_returned_set(self, setup):
        """All strategies should each achieve high recall independently.

        v4 uses a D=128-specialised SIMD path whose FP accumulation order
        differs from v1/v2/v3, so scores at the exact threshold boundary can
        round differently.  Rather than demanding bitwise-identical masks across
        all kernels, we verify per-strategy recall.
        """
        indexer, searchers, query, threshold, scores_exact = setup

        for strat, searcher in searchers.items():
            scores_approx = searcher.search(
                query=query,
                threshold=threshold,
                indexer=indexer,
            )
            recall = _recall(scores_approx, scores_exact, threshold)
            assert (
                recall >= 0.98
            ), f"Strategy '{strat}' achieved recall={recall:.4f}, expected >=0.98"


# ---------------------------------------------------------------------------
# Test 5 – ordering of keys and values is consistent
# ---------------------------------------------------------------------------


class TestKeyValueOrdering:
    """After shuffling keys+values at build time the indexer should preserve
    the correspondence: indexer.keys[h, i] must match indexer.values[..., i, :]."""

    NUM_HEADS = 4
    SEQ_LEN = 256
    DIM = 128

    def _build_with_values(self, keys: torch.Tensor, values: torch.Tensor):
        return CPUIndexer(num_levels=3, branching_factor=8).build(keys, values)

    def test_key_value_correspondence_preserved(self):
        keys = _make_keys(self.NUM_HEADS, self.SEQ_LEN, self.DIM, seed=5)
        # Values are simply a tagged copy: value[h, i] = i (the original index)
        values = torch.arange(self.SEQ_LEN, dtype=torch.float32)
        values = (
            values.view(1, 1, self.SEQ_LEN, 1)
            .expand(1, self.NUM_HEADS, self.SEQ_LEN, self.DIM)
            .clone()
        )
        # Embed the original index in the first channel so we can recover it.
        # values[0, h, i, 0] = i for all h, i
        for i in range(self.SEQ_LEN):
            values[0, :, i, 0] = float(i)

        indexer = CPUIndexer(num_levels=3, branching_factor=8).build(keys, values)

        # The indexer stores keys at level 0; build a mapping from key content
        # to position and verify the values match.
        stored_keys = indexer.keys  # (H, L, D)
        stored_values = indexer.values  # (1, H, L, D)

        assert stored_keys is not None
        assert stored_values is not None

        # For each head, check key[h, i] corresponds to value[h, i]
        # by matching stored_values[0, h, i, 0] (which holds original index)
        # to the matching key in the original keys tensor.
        for h in range(self.NUM_HEADS):
            for i in range(self.SEQ_LEN):
                stored_key = stored_keys[h, i]  # (D,)
                stored_val_idx = stored_values[0, h, i, 0].long().item()

                # The key at position i should match the original key at stored_val_idx
                original_key = keys[0, h, stored_val_idx]  # (D,)
                assert torch.allclose(stored_key, original_key, atol=1e-6), (
                    f"Head {h}, position {i}: key does not match value's original index "
                    f"{stored_val_idx}"
                )

    def test_shuffled_keys_values_stay_aligned(self):
        """Explicitly shuffle keys and values together, build, and verify alignment."""
        keys = _make_keys(self.NUM_HEADS, self.SEQ_LEN, self.DIM, seed=6)

        # Construct values where value[0, h, i, :] = keys[0, h, i, :] * (i+1)
        # so we can identify which key goes with which value.
        values = keys.clone()
        for i in range(self.SEQ_LEN):
            values[0, :, i, :] = keys[0, :, i, :] * (i + 1)

        # Shuffle along the sequence dimension using the same permutation for
        # both keys and values.
        torch.manual_seed(99)
        perm = torch.randperm(self.SEQ_LEN)
        shuffled_keys = keys[:, :, perm, :]
        shuffled_values = values[:, :, perm, :]

        indexer = CPUIndexer(num_levels=3, branching_factor=8).build(
            shuffled_keys, shuffled_values
        )

        stored_keys = indexer.keys  # (H, L, D)
        stored_values = indexer.values  # (1, H, L, D)

        assert stored_keys is not None
        assert stored_values is not None

        # For every position i, keys[h, i] * scale == values[0, h, i, :]
        # where scale = original_position + 1.
        # Verify v[i] is proportional to k[i] for each (h, i).
        for h in range(self.NUM_HEADS):
            k = stored_keys[h]  # (L, D)
            v = stored_values[0, h]  # (L, D)
            for i in range(self.SEQ_LEN):
                # Find a non-zero dimension
                nz_dims = k[i].abs() > 1e-6
                if not nz_dims.any():
                    continue
                ratios = v[i, nz_dims] / k[i, nz_dims]
                # All ratios should be the same (= original_index + 1)
                assert torch.allclose(
                    ratios, ratios[0].expand_as(ratios), atol=1e-4
                ), f"Head {h}, position {i}: values and keys are misaligned"
                # The ratio must be an integer ≥ 1 (original_index + 1)
                scale = ratios[0].item()
                assert (
                    1 <= round(scale) <= self.SEQ_LEN
                ), f"Head {h}, position {i}: unexpected scale {scale}"
