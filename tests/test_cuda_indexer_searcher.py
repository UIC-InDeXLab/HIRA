import os
import sys
from pathlib import Path

import pytest
import torch


os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("MAX_JOBS", "4")
Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

HIRA_ROOT = Path(__file__).resolve().parents[1]
if str(HIRA_ROOT) not in sys.path:
    sys.path.insert(0, str(HIRA_ROOT))

from indexer.cuda import CUDAIndexer
from searcher.cuda import CUDASearcher


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDA indexer/searcher tests",
)


def _normalized_keys(h: int, n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        (1, h, n, d),
        generator=g,
        device="cuda",
        dtype=torch.float32,
    )
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _random_query(h: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((1, h, 1, d), generator=g, device="cuda", dtype=torch.float32)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _choose_block_c(branching_factor: int) -> int:
    for c in (16, 8, 4, 2, 1):
        if c <= branching_factor and branching_factor % c == 0:
            return c
    raise ValueError(f"No valid BLOCK_C for branching_factor={branching_factor}")


def _valid_rows(children: torch.Tensor, pad_value: float) -> torch.Tensor:
    return ~torch.all(children == float(pad_value), dim=-1)


def _brute_force_children_scores(children: torch.Tensor, query_1h1d: torch.Tensor):
    q = query_1h1d.squeeze(0).squeeze(-2).float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.einsum("hd,hnd->hn", q, children.float())


def _per_head_quantile_threshold(
    scores: torch.Tensor, valid_mask: torch.Tensor, q: float
) -> torch.Tensor:
    h = scores.shape[0]
    out = []
    for i in range(h):
        s = scores[i][valid_mask[i]]
        out.append(torch.quantile(s, q))
    return torch.stack(out, dim=0)


def _recall(
    pred_scores: torch.Tensor,
    gt_scores: torch.Tensor,
    threshold: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    gt_mask = (gt_scores >= threshold.unsqueeze(-1)) & valid_mask
    pred_mask = (pred_scores != 0) & valid_mask
    denom = int(gt_mask.sum().item())
    if denom == 0:
        return 1.0
    tp = int((pred_mask & gt_mask).sum().item())
    return tp / denom


@pytest.mark.parametrize(
    "h,n,d,depth,bf,seed",
    [
        (2, 129, 32, CUDAIndexer.DEPTH.TWO_LEVELS, 8, 11),
        (3, 513, 64, CUDAIndexer.DEPTH.THREE_LEVELS, 8, 12),
    ],
)
def test_cuda_indexer_build_structure_and_relationships(h, n, d, depth, bf, seed):
    keys = _normalized_keys(h, n, d, seed=seed)
    assert keys.is_cuda
    indexer = CUDAIndexer(
        depth=depth,
        branching_factor=bf,
        max_iterations=2,
        pad_value=0.0,
    ).build(keys)

    m = max(1, (n + bf - 1) // bf)
    children_n = m * bf
    expected_g = None
    expected_parent_n = m
    if depth == CUDAIndexer.DEPTH.THREE_LEVELS:
        expected_g = max(1, (m + bf - 1) // bf)
        expected_parent_n = expected_g * bf
        children_n = expected_g * bf * bf

    assert indexer.children is not None
    assert indexer.parents is not None
    assert indexer.parent_radii is not None
    assert indexer.children.is_cuda
    assert indexer.parents.is_cuda
    assert indexer.parent_radii.is_cuda
    assert indexer.children.shape == (h, children_n, d)
    assert indexer.parents.shape == (h, expected_parent_n, d)
    assert indexer.parent_radii.shape == (h, expected_parent_n)
    assert indexer.buffer is not None
    assert indexer.buffer.is_cuda
    assert indexer.buffer.shape == (h, children_n)

    valid = _valid_rows(indexer.children, pad_value=indexer.pad_value)
    assert int(valid.sum().item()) == h * n

    pr_expected = indexer._compute_parent_radii_from_layout()
    torch.testing.assert_close(indexer.parent_radii, pr_expected, atol=1e-5, rtol=0.0)

    assert indexer._child_counts is not None
    assert indexer._child_counts.shape == (h, expected_parent_n)
    assert (indexer._child_counts >= 0).all()
    assert (indexer._child_counts <= bf).all()

    assert indexer._parent_valid is not None
    assert indexer._parent_valid.shape == (h, expected_parent_n)
    assert indexer._parent_valid.dtype == torch.bool

    if depth == CUDAIndexer.DEPTH.TWO_LEVELS:
        assert indexer.grand_parents is None
        assert indexer.grand_parent_radii is None
        assert indexer._gp_child_counts is None
    else:
        assert expected_g is not None
        assert indexer.grand_parents is not None
        assert indexer.grand_parent_radii is not None
        assert indexer.grand_parents.shape == (h, expected_g, d)
        assert indexer.grand_parent_radii.shape == (h, expected_g)
        gp_expected = indexer._compute_grandparent_radii_from_layout()
        torch.testing.assert_close(
            indexer.grand_parent_radii, gp_expected, atol=1e-5, rtol=0.0
        )
        assert indexer._gp_child_counts is not None
        assert indexer._gp_child_counts.shape == (h, expected_g)
        assert (indexer._gp_child_counts >= 0).all()
        assert (indexer._gp_child_counts <= bf).all()


@pytest.mark.parametrize(
    "h,n,d,depth,bf,seed",
    [
        (2, 192, 32, CUDAIndexer.DEPTH.TWO_LEVELS, 8, 21),
        (2, 640, 64, CUDAIndexer.DEPTH.THREE_LEVELS, 8, 22),
    ],
)
def test_cuda_indexer_update_incremental_recall_matches_full_build(
    h, n, d, depth, bf, seed
):
    n0 = int(n * 0.7)
    n1 = n - n0
    all_keys = _normalized_keys(h, n, d, seed=seed)
    assert all_keys.is_cuda
    keys_base = all_keys[:, :, :n0, :].contiguous()
    keys_new = all_keys[:, :, n0:, :].contiguous()

    full = CUDAIndexer(
        depth=depth,
        branching_factor=bf,
        max_iterations=2,
        pad_value=0.0,
    ).build(all_keys)

    inc = CUDAIndexer(
        depth=depth,
        branching_factor=bf,
        max_iterations=2,
        pad_value=0.0,
    ).build(keys_base)
    inc.update(keys_new)

    assert full.children is not None
    assert inc.children is not None
    assert full.children.is_cuda
    assert inc.children.is_cuda
    valid_full = _valid_rows(full.children, pad_value=full.pad_value)
    valid_inc = _valid_rows(inc.children, pad_value=inc.pad_value)
    assert int(valid_full.sum().item()) == h * n
    assert int(valid_inc.sum().item()) == h * n

    searcher = CUDASearcher(block_c=_choose_block_c(bf))

    recalls_full = []
    recalls_inc = []
    for i in range(6):
        q = _random_query(h, d, seed=seed + 100 + i)

        gt_full = _brute_force_children_scores(full.children, q)
        th_full = _per_head_quantile_threshold(gt_full, valid_full, q=0.75)
        pred_full = searcher.search(q, th_full, full)
        assert pred_full.is_cuda
        recalls_full.append(_recall(pred_full, gt_full, th_full, valid_full))

        gt_inc = _brute_force_children_scores(inc.children, q)
        th_inc = _per_head_quantile_threshold(gt_inc, valid_inc, q=0.75)
        pred_inc = searcher.search(q, th_inc, inc)
        assert pred_inc.is_cuda
        recalls_inc.append(_recall(pred_inc, gt_inc, th_inc, valid_inc))

    mean_full = sum(recalls_full) / len(recalls_full)
    mean_inc = sum(recalls_inc) / len(recalls_inc)
    assert mean_full >= 0.98
    assert mean_inc >= 0.98
    assert abs(mean_full - mean_inc) <= 0.05


@pytest.mark.parametrize(
    "h,n,d,depth,bf,seed",
    [
        (2, 256, 32, CUDAIndexer.DEPTH.TWO_LEVELS, 8, 31),
        (3, 640, 64, CUDAIndexer.DEPTH.THREE_LEVELS, 8, 32),
    ],
)
def test_cuda_searcher_high_recall(h, n, d, depth, bf, seed):
    keys = _normalized_keys(h, n, d, seed=seed)
    assert keys.is_cuda
    indexer = CUDAIndexer(
        depth=depth,
        branching_factor=bf,
        max_iterations=2,
        pad_value=0.0,
    ).build(keys)

    assert indexer.children is not None
    assert indexer.children.is_cuda
    valid = _valid_rows(indexer.children, pad_value=indexer.pad_value)
    searcher = CUDASearcher(block_c=_choose_block_c(bf))

    recalls = []
    for i in range(8):
        q = _random_query(h, d, seed=seed + 1000 + i)
        gt = _brute_force_children_scores(indexer.children, q)
        th = _per_head_quantile_threshold(gt, valid, q=0.75)
        pred = searcher.search(q, th, indexer)
        assert pred.is_cuda
        recalls.append(_recall(pred, gt, th, valid))

    assert (sum(recalls) / len(recalls)) >= 0.99


def test_cuda_searcher_query_layouts_match():
    h, n, d, bf = 3, 192, 32, 8
    keys = _normalized_keys(h, n, d, seed=41)
    assert keys.is_cuda
    indexer = CUDAIndexer(
        depth=CUDAIndexer.DEPTH.TWO_LEVELS,
        branching_factor=bf,
        max_iterations=2,
        pad_value=0.0,
    ).build(keys)

    assert indexer.children is not None
    assert indexer.children.is_cuda
    valid = _valid_rows(indexer.children, pad_value=indexer.pad_value)
    q = _random_query(h, d, seed=42)
    gt = _brute_force_children_scores(indexer.children, q)
    th = _per_head_quantile_threshold(gt, valid, q=0.7)

    searcher = CUDASearcher(block_c=_choose_block_c(bf))
    out_4d = searcher.search(q, th, indexer)
    out_2d = searcher.search(q.squeeze(0).squeeze(-2).contiguous(), th, indexer)
    assert out_4d.is_cuda
    assert out_2d.is_cuda
    torch.testing.assert_close(out_2d, out_4d, atol=1e-4, rtol=1e-4)


def test_cuda_searcher_low_threshold_matches_bruteforce_on_valid_rows():
    h, n, d, bf = 2, 129, 32, 8
    keys = _normalized_keys(h, n, d, seed=51)
    assert keys.is_cuda
    indexer = CUDAIndexer(
        depth=CUDAIndexer.DEPTH.TWO_LEVELS,
        branching_factor=bf,
        max_iterations=2,
        pad_value=0.0,
    ).build(keys)

    assert indexer.children is not None
    assert indexer.children.is_cuda
    valid = _valid_rows(indexer.children, pad_value=indexer.pad_value)
    q = _random_query(h, d, seed=52)
    gt = _brute_force_children_scores(indexer.children, q)
    threshold = torch.full((h,), -2.0, device="cuda", dtype=torch.float32)

    searcher = CUDASearcher(block_c=_choose_block_c(bf))
    pred = searcher.search(q, threshold, indexer)
    assert pred.is_cuda

    expected = torch.where(valid, gt, torch.zeros_like(gt))
    torch.testing.assert_close(
        torch.where(valid, pred, torch.zeros_like(pred)),
        expected,
        atol=1e-4,
        rtol=1e-4,
    )


def test_cuda_indexer_update_rejects_invalid_shape():
    h, n, d, bf = 2, 128, 32, 8
    keys = _normalized_keys(h, n, d, seed=61)
    assert keys.is_cuda
    indexer = CUDAIndexer(
        depth=CUDAIndexer.DEPTH.THREE_LEVELS,
        branching_factor=bf,
        max_iterations=2,
        pad_value=0.0,
    ).build(keys)

    bad = torch.randn(2, h, 7, d, device="cuda", dtype=torch.float32)
    with pytest.raises(AssertionError):
        indexer.update(bad)
