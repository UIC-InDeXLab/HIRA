import os
import sys
from pathlib import Path

import pytest
import torch


# Keep imports compatible with the current code layout and avoid writing to
# the home cache path during extension loading.
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
os.environ.setdefault("MAX_JOBS", "4")
Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)

HIRA_ROOT = Path(__file__).resolve().parents[1]
if str(HIRA_ROOT) not in sys.path:
    sys.path.insert(0, str(HIRA_ROOT))

from indexer.cpu import CPUIndexer
from searcher.cpu import CPUSearcher


def _normalized_keys(h: int, n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((1, h, n, d), generator=g, dtype=torch.float32)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _random_query(h: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn((1, h, 1, d), generator=g, dtype=torch.float32)


def _expected_level_sizes(n: int, branching_factor: int, num_levels: int) -> list[int]:
    sizes = [n]
    while len(sizes) < num_levels:
        nxt = sizes[-1] // branching_factor
        if nxt < 1:
            break
        sizes.append(nxt)
    return sizes


def _assert_parent_child_structure(indexer: CPUIndexer):
    for li in range(len(indexer.levels) - 1):
        child = indexer.levels[li]
        parent = indexer.levels[li + 1]
        c2p = child.child2parent
        assert c2p is not None, f"level {li} child2parent missing"
        assert c2p.dtype == torch.long
        assert c2p.shape == (indexer.num_heads, child.size)
        assert (c2p >= 0).all(), f"level {li} has negative parent index"
        assert (c2p < parent.size).all(), f"level {li} parent index out of range"
        assert child.num_parents == parent.size


def _assert_radius_upper_bound(indexer: CPUIndexer, atol: float = 1e-4):
    for li in range(len(indexer.levels) - 1):
        child = indexer.levels[li]
        parent = indexer.levels[li + 1]
        c2p = child.child2parent
        assert c2p is not None

        h, c, d = child.ball_centers.shape
        parent_for_child = parent.ball_centers.gather(
            dim=1,
            index=c2p.unsqueeze(-1).expand(h, c, d),
        )
        dist = torch.linalg.norm(
            (child.ball_centers - parent_for_child).float(), dim=-1
        )
        contrib = dist + child.ball_radii.float()
        pr = parent.ball_radii.float().gather(1, c2p)

        bad = pr + atol < contrib
        assert not bad.any(), f"radius bound violated at edge {li}->{li+1}"


def _brute_force(keys_hnd: torch.Tensor, query_1h1d: torch.Tensor) -> torch.Tensor:
    q = query_1h1d.squeeze(0).squeeze(-2).float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.einsum("hd,hnd->hn", q, keys_hnd.float())


def _recall(
    pred_scores: torch.Tensor, gt_scores: torch.Tensor, threshold: torch.Tensor
) -> float:
    gt_mask = gt_scores >= threshold.unsqueeze(-1)
    pred_mask = pred_scores != 0
    denom = int(gt_mask.sum().item())
    if denom == 0:
        return 1.0
    tp = int((pred_mask & gt_mask).sum().item())
    return tp / denom


@pytest.mark.parametrize(
    "h,n,d,num_levels,bf,seed",
    [
        (1, 64, 16, 3, 4, 11),
        (2, 96, 32, 3, 8, 12),
        (4, 128, 64, 4, 4, 13),
    ],
)
def test_cpu_indexer_build_structure_and_relationships(h, n, d, num_levels, bf, seed):
    keys = _normalized_keys(h, n, d, seed=seed)
    indexer = CPUIndexer(num_levels=num_levels, branching_factor=bf, max_iterations=2)
    indexer.build(keys)

    expected = _expected_level_sizes(n, bf, num_levels)
    assert len(indexer.levels) == len(expected)
    assert indexer.keys.shape == (h, n, d)
    assert indexer.levels[0].ball_radii.abs().max().item() == 0.0

    for li, sz in enumerate(expected):
        lvl = indexer.levels[li]
        assert lvl.level_idx == li
        assert lvl.size == sz
        assert lvl.ball_centers.shape == (h, sz, d)
        assert lvl.ball_radii.shape == (h, sz)

    if len(indexer.levels) > 1:
        _assert_parent_child_structure(indexer)
        _assert_radius_upper_bound(indexer)


@pytest.mark.parametrize(
    "h,n,d,num_levels,bf,seed",
    [
        (1, 128, 32, 3, 8, 21),
        (2, 160, 64, 4, 4, 22),
    ],
)
def test_cpu_indexer_update_incremental_recall_matches_full_build(
    h, n, d, num_levels, bf, seed
):
    total = n
    n0 = int(total * 0.7)
    n1 = total - n0
    all_keys = _normalized_keys(h, total, d, seed=seed)
    keys_base = all_keys[:, :, :n0, :].contiguous()
    keys_new = all_keys[:, :, n0:, :].contiguous()

    full = CPUIndexer(num_levels=num_levels, branching_factor=bf, max_iterations=2)
    full.build(all_keys)

    inc = CPUIndexer(num_levels=num_levels, branching_factor=bf, max_iterations=2)
    inc.build(keys_base)
    inc.update(keys_new)

    assert inc.keys.shape == full.keys.shape
    assert inc.num_keys == full.num_keys == total
    _assert_parent_child_structure(inc)
    _assert_radius_upper_bound(inc)

    searcher = CPUSearcher(search_strategy="vectorized_cpp_filter")

    recalls_full = []
    recalls_inc = []
    for i in range(6):
        q = _random_query(h, d, seed=seed + 100 + i)
        gt = _brute_force(all_keys.squeeze(0), q)
        th = torch.quantile(gt, 0.7, dim=-1)

        pred_full = searcher.search(q, th, full)
        pred_inc = searcher.search(q, th, inc)

        recalls_full.append(_recall(pred_full, gt, th))
        recalls_inc.append(_recall(pred_inc, gt, th))

    mean_full = sum(recalls_full) / len(recalls_full)
    mean_inc = sum(recalls_inc) / len(recalls_inc)
    assert mean_full >= 0.98
    assert mean_inc >= 0.98
    assert abs(mean_full - mean_inc) <= 0.03


@pytest.mark.parametrize(
    "h,n,d,num_levels,bf,seed",
    [
        (1, 128, 32, 3, 8, 31),
        (2, 192, 64, 4, 4, 32),
        (4, 256, 128, 4, 8, 33),
    ],
)
def test_cpu_searcher_high_recall_vectorized_cpp_filter(h, n, d, num_levels, bf, seed):
    keys = _normalized_keys(h, n, d, seed=seed)
    indexer = CPUIndexer(num_levels=num_levels, branching_factor=bf, max_iterations=2)
    indexer.build(keys)

    searcher = CPUSearcher(search_strategy="vectorized_cpp_filter")

    recalls = []
    for i in range(8):
        q = _random_query(h, d, seed=seed + 1000 + i)
        gt = _brute_force(keys.squeeze(0), q)
        th = torch.quantile(gt, 0.75, dim=-1)
        pred = searcher.search(q, th, indexer)
        recalls.append(_recall(pred, gt, th))

    assert (sum(recalls) / len(recalls)) >= 0.99


@pytest.mark.parametrize(
    "strategy",
    [
        "vectorized_cpp_filter",
        "fused",
        "fused_v2",
        "fused_v3",
        "exact_torch",
    ],
)
def test_cpu_searcher_methods_have_almost_full_recall(strategy):
    h, n, d = 2, 256, 64
    keys = _normalized_keys(h, n, d, seed=41)
    q = _random_query(h, d, seed=42)

    indexer = CPUIndexer(num_levels=4, branching_factor=8, max_iterations=2)
    indexer.build(keys)

    gt = _brute_force(keys.squeeze(0), q)
    th = torch.quantile(gt, 0.75, dim=-1)

    searcher = CPUSearcher(search_strategy=strategy)
    pred = searcher.search(q, th, indexer)

    assert pred.shape == (h, n)
    assert _recall(pred, gt, th) >= 0.99


def test_cpu_searcher_fused_v4_runs_for_d128():
    h, n, d = 2, 192, 128
    keys = _normalized_keys(h, n, d, seed=51)
    q = _random_query(h, d, seed=52)

    indexer = CPUIndexer(num_levels=4, branching_factor=8, max_iterations=2)
    indexer.build(keys)

    gt = _brute_force(keys.squeeze(0), q)
    th = torch.quantile(gt, 0.8, dim=-1)

    searcher = CPUSearcher(search_strategy="fused_v4")
    pred = searcher.search(q, th, indexer)

    assert pred.shape == (h, n)
    assert _recall(pred, gt, th) >= 0.99


def test_cpu_searcher_single_level_equals_bruteforce():
    h, n, d = 2, 96, 32
    keys = _normalized_keys(h, n, d, seed=61)
    q = _random_query(h, d, seed=62)

    indexer = CPUIndexer(num_levels=1, branching_factor=8, max_iterations=1)
    indexer.build(keys)

    gt = _brute_force(keys.squeeze(0), q)
    th = torch.quantile(gt, 0.7, dim=-1)
    gt_thresholded = torch.where(gt >= th.unsqueeze(-1), gt, torch.zeros_like(gt))

    searcher = CPUSearcher(search_strategy="vectorized_cpp_filter")
    pred = searcher.search(q, th, indexer)
    torch.testing.assert_close(pred, gt_thresholded, atol=1e-5, rtol=1e-5)


def test_cpu_indexer_update_rejects_invalid_shape():
    h, n, d = 2, 64, 16
    keys = _normalized_keys(h, n, d, seed=71)
    indexer = CPUIndexer(num_levels=3, branching_factor=8, max_iterations=1)
    indexer.build(keys)

    bad = torch.randn(h, 8, d, dtype=torch.float32)  # expected (1,H,m,D)
    with pytest.raises(ValueError):
        indexer.update(bad)
