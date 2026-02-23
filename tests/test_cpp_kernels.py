import os
import sys
from pathlib import Path

import pytest
import torch


os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
os.environ.setdefault("MAX_JOBS", "4")
Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)

HIRA_ROOT = Path(__file__).resolve().parents[1]
if str(HIRA_ROOT) not in sys.path:
    sys.path.insert(0, str(HIRA_ROOT))

from indexer.cpu import CPUIndexer


def _normalized_keys(h: int, n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((1, h, n, d), generator=g, dtype=torch.float32)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _random_query_hd(h: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn((h, d), generator=g, dtype=torch.float32)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _thresholded_bruteforce(
    keys_hnd: torch.Tensor, query_hd: torch.Tensor, threshold_h: torch.Tensor
) -> torch.Tensor:
    scores = torch.einsum("hnd,hd->hn", keys_hnd.float(), query_hd.float())
    return torch.where(scores >= threshold_h.unsqueeze(-1), scores, torch.zeros_like(scores))


def _prepare_level_data(indexer: CPUIndexer):
    centers_list = []
    radii_list = []
    c2p_list = []
    sizes_list = []
    for level in indexer.levels:
        centers_list.append(level.ball_centers.float().contiguous())
        radii_list.append(level.ball_radii.float().contiguous())
        if level.child2parent is None:
            c2p_list.append(torch.empty(0, dtype=torch.long))
        else:
            c2p_list.append(level.child2parent.long().contiguous())
        sizes_list.append(level.size)
    return centers_list, radii_list, c2p_list, sizes_list


def _build_parent_child_adjacency(c2p: torch.Tensor, num_parents: int):
    h, c = c2p.shape
    offsets = torch.empty((h, num_parents + 1), dtype=torch.long)
    children = torch.empty((h, c), dtype=torch.long)

    for head in range(h):
        p = c2p[head]
        order = torch.argsort(p)
        sorted_parents = p.index_select(0, order)
        counts = torch.bincount(sorted_parents, minlength=num_parents)
        off = torch.empty(num_parents + 1, dtype=torch.long)
        off[0] = 0
        off[1:] = counts.cumsum(0)
        offsets[head] = off
        children[head] = order

    return offsets.contiguous(), children.contiguous()


def _prepare_level_data_with_adjacency(indexer: CPUIndexer):
    centers_list, radii_list, c2p_list, sizes_list = _prepare_level_data(indexer)
    offsets_list = []
    children_list = []

    for level_idx, level in enumerate(indexer.levels):
        if level.child2parent is None:
            offsets_list.append(torch.empty(0, dtype=torch.long))
            children_list.append(torch.empty(0, dtype=torch.long))
            continue

        num_parents = indexer.levels[level_idx + 1].size
        off, ch = _build_parent_child_adjacency(
            level.child2parent.long().contiguous(),
            num_parents=num_parents,
        )
        offsets_list.append(off)
        children_list.append(ch)

    return (
        centers_list,
        radii_list,
        c2p_list,
        sizes_list,
        offsets_list,
        children_list,
    )


@pytest.fixture(scope="module")
def cpp_numpy_kernel():
    return pytest.importorskip("hira.kernels.cpp.hira_cpp_kernels")


@pytest.fixture(scope="module")
def cpp_torch_ext():
    return pytest.importorskip("hira.kernels.cpp.torch_ext_loader")


def test_cpp_numpy_exact_filter_matches_reference(cpp_numpy_kernel):
    h, n, d = 3, 97, 32
    g = torch.Generator().manual_seed(11)
    keys = torch.randn((h, n, d), generator=g, dtype=torch.float32)
    keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    query = _random_query_hd(h, d, seed=12)
    leaf_mask = torch.rand((h, n), generator=g) > 0.35
    all_scores = torch.einsum("hnd,hd->hn", keys, query)
    threshold = torch.quantile(all_scores, 0.70, dim=-1)

    out_np = cpp_numpy_kernel.exact_filter_mask_batched(
        keys.numpy(),
        leaf_mask.numpy(),
        query.numpy(),
        threshold.numpy(),
    )
    out = torch.from_numpy(out_np)

    expected_scores = torch.einsum("hnd,hd->hn", keys, query)
    expected = torch.where(
        leaf_mask & (expected_scores >= threshold.unsqueeze(-1)),
        expected_scores,
        torch.zeros_like(expected_scores),
    )
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_cpp_torch_exact_filter_matches_reference(cpp_torch_ext):
    h, n, d = 2, 129, 64
    g = torch.Generator().manual_seed(21)
    keys = torch.randn((h, n, d), generator=g, dtype=torch.float32)
    keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    query = _random_query_hd(h, d, seed=22)
    leaf_mask = torch.rand((h, n), generator=g) > 0.30
    all_scores = torch.einsum("hnd,hd->hn", keys, query)
    threshold = torch.quantile(all_scores, 0.75, dim=-1)

    out = cpp_torch_ext.hira_torch_ext.exact_filter(keys, leaf_mask, query, threshold)

    expected_scores = torch.einsum("hnd,hd->hn", keys, query)
    expected = torch.where(
        leaf_mask & (expected_scores >= threshold.unsqueeze(-1)),
        expected_scores,
        torch.zeros_like(expected_scores),
    )
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "variant",
    ["v1", "v2", "v3"],
)
def test_cpp_fused_variants_match_bruteforce(variant, cpp_torch_ext):
    h, n, d, num_levels, bf = 2, 192, 64, 4, 8
    keys = _normalized_keys(h, n, d, seed=31)
    indexer = CPUIndexer(
        num_levels=num_levels,
        branching_factor=bf,
        max_iterations=2,
    ).build(keys)

    query = _random_query_hd(h, d, seed=32)
    gt_scores = torch.einsum("hnd,hd->hn", indexer.keys.float(), query.float())
    threshold = torch.quantile(gt_scores, 0.75, dim=-1)
    expected = _thresholded_bruteforce(indexer.keys, query, threshold)

    if variant == "v1":
        centers_list, radii_list, c2p_list, sizes_list = _prepare_level_data(indexer)
        out = cpp_torch_ext.hira_torch_ext.fused_tree_search(
            indexer.keys.float().contiguous(),
            query,
            threshold,
            centers_list,
            radii_list,
            c2p_list,
            sizes_list,
        )
    elif variant == "v2":
        centers_list, radii_list, c2p_list, sizes_list = _prepare_level_data(indexer)
        out = cpp_torch_ext.hira_torch_ext_v2.fused_tree_search_v2(
            indexer.keys.float().contiguous(),
            query,
            threshold,
            centers_list,
            radii_list,
            c2p_list,
            sizes_list,
        )
    else:
        (
            centers_list,
            radii_list,
            c2p_list,
            sizes_list,
            offsets_list,
            children_list,
        ) = _prepare_level_data_with_adjacency(indexer)
        out = cpp_torch_ext.hira_torch_ext_v3.fused_tree_search_v3(
            indexer.keys.float().contiguous(),
            query,
            threshold,
            centers_list,
            radii_list,
            c2p_list,
            sizes_list,
            offsets_list,
            children_list,
        )

    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


def test_cpp_fused_v4_matches_bruteforce(cpp_torch_ext):
    h, n, d, num_levels, bf = 2, 192, 128, 4, 8
    keys = _normalized_keys(h, n, d, seed=41)
    indexer = CPUIndexer(
        num_levels=num_levels,
        branching_factor=bf,
        max_iterations=2,
    ).build(keys)

    query = _random_query_hd(h, d, seed=42)
    gt_scores = torch.einsum("hnd,hd->hn", indexer.keys.float(), query.float())
    threshold = torch.quantile(gt_scores, 0.75, dim=-1)
    expected = _thresholded_bruteforce(indexer.keys, query, threshold)

    (
        centers_list,
        radii_list,
        c2p_list,
        sizes_list,
        offsets_list,
        children_list,
    ) = _prepare_level_data_with_adjacency(indexer)
    out = cpp_torch_ext.hira_torch_ext_v4.fused_tree_search_v4(
        indexer.keys.float().contiguous(),
        query,
        threshold,
        centers_list,
        radii_list,
        c2p_list,
        sizes_list,
        offsets_list,
        children_list,
    )

    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
