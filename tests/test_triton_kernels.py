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
from hira.kernels.triton_search_wrappers import (
    triton_three_level_filter_kernel_v1,
    triton_three_level_filter_kernel_v2,
    triton_three_level_filter_v1,
    triton_two_level_filter,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton kernel tests",
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


def _bruteforce_children_scores(children: torch.Tensor, query_1h1d: torch.Tensor):
    q = query_1h1d.squeeze(0).squeeze(-2).float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.einsum("hd,hnd->hn", q, children.float())


def _grouped_bruteforce_children_scores(
    children_kv: torch.Tensor, query_1h1d: torch.Tensor, q_head_to_kv: torch.Tensor
):
    q = query_1h1d.squeeze(0).squeeze(-2).float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    children_for_query = children_kv.index_select(0, q_head_to_kv.long())
    return torch.einsum("hd,hnd->hn", q, children_for_query.float())


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
    "h,n,d,bf,seed",
    [
        (2, 257, 64, 8, 11),
        (4, 513, 128, 16, 12),
    ],
)
def test_triton_two_level_filter_high_recall_against_bruteforce(h, n, d, bf, seed):
    keys = _normalized_keys(h, n, d, seed=seed)
    assert keys.is_cuda
    indexer = CUDAIndexer(
        num_levels=CUDAIndexer.DEPTH.TWO_LEVELS,
        max_iterations=2,
        branching_factor=bf,
        pad_value=0.0,
    ).build(keys)

    assert indexer.children is not None
    assert indexer.parents is not None
    assert indexer.parent_radii is not None
    assert indexer.children.is_cuda
    assert indexer.parents.is_cuda
    assert indexer.parent_radii.is_cuda

    q = _random_query(h, d, seed=seed + 100)
    assert q.is_cuda
    valid = _valid_rows(indexer.children, pad_value=indexer.pad_value)
    gt = _bruteforce_children_scores(indexer.children, q)
    th = _per_head_quantile_threshold(gt, valid, q=0.75)
    assert th.is_cuda

    out = triton_two_level_filter(
        indexer.children,
        indexer.parents,
        indexer.parent_radii,
        q,
        th,
        branch=bf,
        BLOCK_C=_choose_block_c(bf),
    )

    assert out.shape == gt.shape
    assert out.is_cuda
    assert _recall(out, gt, th, valid) >= 0.99

    for head in range(h):
        keep = (out[head] >= th[head]) & valid[head]
        if keep.any():
            assert torch.all(gt[head][keep] >= (th[head] - 1e-4))


def test_triton_two_level_filter_accepts_2d_and_4d_query_layouts():
    h, n, d, bf = 3, 192, 64, 8
    keys = _normalized_keys(h, n, d, seed=21)
    assert keys.is_cuda
    indexer = CUDAIndexer(
        num_levels=CUDAIndexer.DEPTH.TWO_LEVELS,
        max_iterations=2,
        branching_factor=bf,
        pad_value=0.0,
    ).build(keys)

    assert indexer.children is not None
    assert indexer.parents is not None
    assert indexer.parent_radii is not None
    assert indexer.children.is_cuda
    assert indexer.parents.is_cuda
    assert indexer.parent_radii.is_cuda

    q4 = _random_query(h, d, seed=22)
    q2 = q4.squeeze(0).squeeze(-2).contiguous()
    assert q4.is_cuda
    assert q2.is_cuda
    valid = _valid_rows(indexer.children, pad_value=indexer.pad_value)
    gt = _bruteforce_children_scores(indexer.children, q4)
    th = _per_head_quantile_threshold(gt, valid, q=0.70)

    out4 = triton_two_level_filter(
        indexer.children,
        indexer.parents,
        indexer.parent_radii,
        q4,
        th,
        branch=bf,
        BLOCK_C=_choose_block_c(bf),
    )
    out2 = triton_two_level_filter(
        indexer.children,
        indexer.parents,
        indexer.parent_radii,
        q2,
        th,
        branch=bf,
        BLOCK_C=_choose_block_c(bf),
    )

    assert out4.is_cuda
    assert out2.is_cuda
    torch.testing.assert_close(out2, out4, atol=1e-4, rtol=1e-4)


def test_triton_two_level_filter_writes_to_provided_output_buffer():
    h, n, d, bf = 2, 160, 32, 8
    keys = _normalized_keys(h, n, d, seed=31)
    assert keys.is_cuda
    indexer = CUDAIndexer(
        num_levels=CUDAIndexer.DEPTH.TWO_LEVELS,
        max_iterations=2,
        branching_factor=bf,
        pad_value=0.0,
    ).build(keys)

    assert indexer.children is not None
    assert indexer.parents is not None
    assert indexer.parent_radii is not None
    assert indexer.children.is_cuda
    assert indexer.parents.is_cuda
    assert indexer.parent_radii.is_cuda

    q = _random_query(h, d, seed=32)
    assert q.is_cuda
    valid = _valid_rows(indexer.children, pad_value=indexer.pad_value)
    gt = _bruteforce_children_scores(indexer.children, q)
    th = _per_head_quantile_threshold(gt, valid, q=0.80)

    out_buf = torch.full_like(gt, -7.0)
    assert out_buf.is_cuda
    out = triton_two_level_filter(
        indexer.children,
        indexer.parents,
        indexer.parent_radii,
        q,
        th,
        out=out_buf,
        branch=bf,
        BLOCK_C=_choose_block_c(bf),
    )
    assert out.data_ptr() == out_buf.data_ptr()
    assert out.shape == gt.shape
    assert out.is_cuda


def test_triton_three_level_variants_match_each_other_and_bruteforce():
    h, n, d, bf = 3, 769, 64, 8
    keys = _normalized_keys(h, n, d, seed=41)
    assert keys.is_cuda
    indexer = CUDAIndexer(
        num_levels=CUDAIndexer.DEPTH.THREE_LEVELS,
        max_iterations=2,
        branching_factor=bf,
        pad_value=0.0,
    ).build(keys)

    assert indexer.children is not None
    assert indexer.parents is not None
    assert indexer.parent_radii is not None
    assert indexer.grand_parents is not None
    assert indexer.grand_parent_radii is not None
    assert indexer.children.is_cuda
    assert indexer.parents.is_cuda
    assert indexer.parent_radii.is_cuda
    assert indexer.grand_parents.is_cuda
    assert indexer.grand_parent_radii.is_cuda

    q = _random_query(h, d, seed=42)
    assert q.is_cuda
    valid = _valid_rows(indexer.children, pad_value=indexer.pad_value)
    gt = _bruteforce_children_scores(indexer.children, q)
    th = _per_head_quantile_threshold(gt, valid, q=0.75)
    assert th.is_cuda
    block_c = _choose_block_c(bf)

    out_wrapper = triton_three_level_filter_v1(
        indexer.children,
        indexer.parents,
        indexer.parent_radii,
        indexer.grand_parents,
        indexer.grand_parent_radii,
        q,
        th,
        branch=bf,
        BLOCK_C=block_c,
    )
    out_k1 = triton_three_level_filter_kernel_v1(
        indexer.children,
        indexer.parents,
        indexer.parent_radii,
        indexer.grand_parents,
        indexer.grand_parent_radii,
        q,
        th,
        branch=bf,
        BLOCK_C=block_c,
    )
    out_k2 = triton_three_level_filter_kernel_v2(
        indexer.children,
        indexer.parents,
        indexer.parent_radii,
        indexer.grand_parents,
        indexer.grand_parent_radii,
        q,
        th,
        branch=bf,
        BLOCK_C=block_c,
    )

    assert out_wrapper.is_cuda
    assert out_k1.is_cuda
    assert out_k2.is_cuda
    assert _recall(out_wrapper, gt, th, valid) >= 0.99
    assert _recall(out_k1, gt, th, valid) >= 0.99
    assert _recall(out_k2, gt, th, valid) >= 0.99

    torch.testing.assert_close(out_k1, out_k2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_wrapper, out_k1, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "depth,d,seed",
    [
        (CUDAIndexer.DEPTH.TWO_LEVELS, 64, 91),
        (CUDAIndexer.DEPTH.THREE_LEVELS, 64, 92),
    ],
)
def test_triton_kernels_support_grouped_attention(depth, d, seed):
    h_kv, group_size, n, bf = 2, 3, 512, 8
    h_q = h_kv * group_size

    keys = _normalized_keys(h_kv, n, d, seed=seed)
    indexer = CUDAIndexer(
        num_levels=depth,
        max_iterations=2,
        branching_factor=bf,
        pad_value=0.0,
    ).build(keys)

    assert indexer.children is not None
    assert indexer.parents is not None
    assert indexer.parent_radii is not None
    valid_kv = _valid_rows(indexer.children, pad_value=indexer.pad_value)

    q = _random_query(h_q, d, seed=seed + 1)
    q_head_to_kv = torch.arange(h_q, device="cuda", dtype=torch.long) // group_size
    valid_q = valid_kv.index_select(0, q_head_to_kv)
    gt = _grouped_bruteforce_children_scores(indexer.children, q, q_head_to_kv)
    th = _per_head_quantile_threshold(gt, valid_q, q=0.75)
    block_c = _choose_block_c(bf)

    if depth == CUDAIndexer.DEPTH.TWO_LEVELS:
        out = triton_two_level_filter(
            indexer.children,
            indexer.parents,
            indexer.parent_radii,
            q,
            th,
            branch=bf,
            BLOCK_C=block_c,
        )
        assert out.shape == gt.shape
        assert _recall(out, gt, th, valid_q) >= 0.99
        return

    assert indexer.grand_parents is not None
    assert indexer.grand_parent_radii is not None
    out_k1 = triton_three_level_filter_kernel_v1(
        indexer.children,
        indexer.parents,
        indexer.parent_radii,
        indexer.grand_parents,
        indexer.grand_parent_radii,
        q,
        th,
        branch=bf,
        BLOCK_C=block_c,
    )
    out_k2 = triton_three_level_filter_kernel_v2(
        indexer.children,
        indexer.parents,
        indexer.parent_radii,
        indexer.grand_parents,
        indexer.grand_parent_radii,
        q,
        th,
        branch=bf,
        BLOCK_C=block_c,
    )

    assert out_k1.shape == gt.shape
    assert out_k2.shape == gt.shape
    assert _recall(out_k1, gt, th, valid_q) >= 0.99
    assert _recall(out_k2, gt, th, valid_q) >= 0.99
    torch.testing.assert_close(out_k1, out_k2, atol=1e-4, rtol=1e-4)


def test_triton_two_level_filter_validates_inputs():
    h, n, d, bf = 2, 64, 32, 8
    keys = _normalized_keys(h, n, d, seed=51)
    assert keys.is_cuda
    indexer = CUDAIndexer(
        num_levels=CUDAIndexer.DEPTH.TWO_LEVELS,
        max_iterations=1,
        branching_factor=bf,
        pad_value=0.0,
    ).build(keys)

    assert indexer.children is not None
    assert indexer.parents is not None
    assert indexer.parent_radii is not None
    assert indexer.children.is_cuda
    assert indexer.parents.is_cuda
    assert indexer.parent_radii.is_cuda

    q = _random_query(h, d, seed=52)
    th = torch.zeros((h,), device="cuda", dtype=torch.float32)
    assert q.is_cuda
    assert th.is_cuda

    with pytest.raises(ValueError):
        triton_two_level_filter(
            indexer.children,
            indexer.parents,
            indexer.parent_radii,
            q,
            th,
            branch=bf,
            BLOCK_C=3,
        )

    with pytest.raises(ValueError):
        triton_two_level_filter(
            indexer.children,
            indexer.parents,
            indexer.parent_radii,
            q[:, :1, :, :],
            th,
            branch=bf,
            BLOCK_C=_choose_block_c(bf),
        )

    with pytest.raises(ValueError):
        triton_two_level_filter(
            indexer.children,
            indexer.parents,
            indexer.parent_radii,
            q,
            torch.zeros((h + 1,), device="cuda", dtype=torch.float32),
            branch=bf,
            BLOCK_C=_choose_block_c(bf),
        )
