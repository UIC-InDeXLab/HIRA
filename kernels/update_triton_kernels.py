from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_TRITON_AVAILABLE = True


@dataclass
class NearestL2TritonConfig:
    block_m: int = 256


@triton.jit
def _nearest_l2_stage1_kernel(
    x_ptr,
    centers_ptr,
    centers_norm2_ptr,
    valid_mask_ptr,  # may be None (0)
    out_best_score_ptr,
    out_best_idx_ptr,
    x_row_stride: tl.constexpr,
    c_row_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    d: tl.constexpr,
    m,
    use_valid_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # 2D launch: (query_id, center_block_id)
    qid = tl.program_id(0)
    bid = tl.program_id(1)

    # Load x[qid, :]
    offs_d = tl.arange(0, d)
    x = tl.load(x_ptr + qid * x_row_stride + offs_d, mask=offs_d < d, other=0.0)

    # Load centers block
    center_ids = bid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_c = center_ids < m

    c_ptrs = centers_ptr + center_ids[:, None] * c_row_stride + offs_d[None, :]
    c = tl.load(c_ptrs, mask=mask_c[:, None] & (offs_d[None, :] < d), other=0.0)

    # Dot products: (BLOCK_M,)
    dot = tl.sum(c * x[None, :], axis=1)
    c_norm2 = tl.load(centers_norm2_ptr + center_ids, mask=mask_c, other=0.0)

    # score = ||c||^2 - 2 * x·c  (||x||^2 is constant across centers)
    score = c_norm2 - 2.0 * dot

    if use_valid_mask:
        v = tl.load(valid_mask_ptr + center_ids, mask=mask_c, other=0).to(tl.int1)
        # invalid centers => score = +inf
        score = tl.where(v, score, float("inf"))

    neg_score = -score
    # invalid/masked => -inf so it won't win argmax
    neg_score = tl.where(mask_c, neg_score, float("-inf"))

    local_j = tl.argmax(neg_score, axis=0)
    local_best = -tl.max(neg_score, axis=0)
    local_idx = bid * BLOCK_M + local_j

    tl.store(out_best_score_ptr + qid * out_row_stride + bid, local_best)
    tl.store(out_best_idx_ptr + qid * out_row_stride + bid, local_idx)


def nearest_l2_triton(
    x: torch.Tensor,
    centers: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    cfg: Optional[NearestL2TritonConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute nearest center indices and squared distances using Triton (two-stage).

    Returns (best_idx, best_d2) where best_idx is in the original centers index space.

    Notes:
    - Uses a stage-1 kernel to compute best per center-block.
    - Reduces across blocks using a small PyTorch reduction.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")

    if x.numel() == 0:
        return (
            torch.empty((0,), device=x.device, dtype=torch.int64),
            torch.empty((0,), device=x.device, dtype=torch.float32),
        )

    if x.ndim != 2 or centers.ndim != 2:
        raise ValueError("x and centers must be 2D")

    if x.shape[1] != centers.shape[1]:
        raise ValueError("x and centers must have same dim")

    if not x.is_cuda or not centers.is_cuda:
        raise ValueError("x and centers must be CUDA tensors")

    x_f = x.float().contiguous()
    c_f = centers.float().contiguous()

    n, d = x_f.shape
    m = c_f.shape[0]

    if cfg is None:
        cfg = NearestL2TritonConfig()

    block_m = int(cfg.block_m)
    nb = triton.cdiv(m, block_m)

    centers_norm2 = (c_f * c_f).sum(dim=1).contiguous()  # (m,)
    x_norm2 = (x_f * x_f).sum(dim=1).contiguous()  # (n,)

    out_best_score = torch.empty((n, nb), device=x.device, dtype=torch.float32)
    out_best_idx = torch.empty((n, nb), device=x.device, dtype=torch.int32)

    use_valid = valid_mask is not None
    if use_valid:
        vm = valid_mask.to(device=x.device)
        if vm.dtype != torch.bool:
            vm = vm.to(torch.bool)
        vm_i32 = vm.to(torch.int32).contiguous()
        vm_ptr = vm_i32
    else:
        vm_ptr = torch.empty((1,), device=x.device, dtype=torch.int32)

    grid = (n, nb)
    _nearest_l2_stage1_kernel[grid](
        x_f,
        c_f,
        centers_norm2,
        vm_ptr,
        out_best_score,
        out_best_idx,
        x_row_stride=x_f.stride(0),
        c_row_stride=c_f.stride(0),
        out_row_stride=out_best_score.stride(0),
        d=d,
        m=m,
        use_valid_mask=use_valid,
        BLOCK_M=block_m,
        num_warps=4,
    )

    # Reduce over blocks on GPU
    best_score, best_block = torch.min(out_best_score, dim=1)  # (n,)
    ar = torch.arange(n, device=x.device)
    best_idx = out_best_idx[ar, best_block].to(torch.int64)

    best_d2 = best_score + x_norm2
    best_d2 = torch.clamp_min(best_d2, 0.0)

    return best_idx, best_d2


@triton.jit
def _fill_children_atomic_kernel(
    x_ptr,
    parent_idx_ptr,
    child_counts_ptr,
    children_ptr,
    placed_ptr,
    placed_flat_idx_ptr,
    n,
    d: tl.constexpr,
    bf: tl.constexpr,
    x_row_stride: tl.constexpr,
    children_row_stride: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n:
        return

    p = tl.load(parent_idx_ptr + pid).to(tl.int32)

    # Reserve a slot.
    slot = tl.atomic_add(child_counts_ptr + p, 1).to(tl.int32)

    if slot < bf:
        row = p * bf + slot
        offs = tl.arange(0, d)
        v = tl.load(x_ptr + pid * x_row_stride + offs, mask=offs < d, other=0.0)
        tl.store(children_ptr + row * children_row_stride + offs, v, mask=offs < d)
        tl.store(placed_ptr + pid, 1)
        tl.store(placed_flat_idx_ptr + pid, row)
    else:
        tl.store(placed_ptr + pid, 0)
        tl.store(placed_flat_idx_ptr + pid, -1)


def fill_existing_children_atomic(
    *,
    x: torch.Tensor,
    parent_idx: torch.Tensor,
    child_counts: torch.Tensor,
    children_flat: torch.Tensor,
    bf: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fill children slots using per-parent atomic counters.

    Assumes that for each parent, filled children occupy slots [0, child_counts[parent]).

    Returns:
      placed_mask: (N,) bool
      placed_flat_idx: (N,) int64 (row index into children_flat), -1 if not placed
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")

    if x.numel() == 0:
        return (
            torch.empty((0,), device=x.device, dtype=torch.bool),
            torch.empty((0,), device=x.device, dtype=torch.int64),
        )

    if x.dtype != torch.float32:
        x = x.float()
    x = x.contiguous()

    if parent_idx.dtype != torch.int64:
        parent_idx = parent_idx.to(torch.int64)
    parent_i32 = parent_idx.to(torch.int32).contiguous()

    if child_counts.dtype != torch.int32:
        raise ValueError("child_counts must be int32")

    if not (
        x.is_cuda
        and parent_i32.is_cuda
        and child_counts.is_cuda
        and children_flat.is_cuda
    ):
        raise ValueError("all inputs must be CUDA")

    n, d = x.shape

    placed_i8 = torch.empty((n,), device=x.device, dtype=torch.int8)
    placed_flat_i32 = torch.empty((n,), device=x.device, dtype=torch.int32)

    grid = (triton.cdiv(n, 256) * 256,)
    _fill_children_atomic_kernel[grid](
        x,
        parent_i32,
        child_counts,
        children_flat,
        placed_i8,
        placed_flat_i32,
        n=n,
        d=d,
        bf=int(bf),
        x_row_stride=x.stride(0),
        children_row_stride=children_flat.stride(0),
        num_warps=4,
    )

    placed_mask = placed_i8.to(torch.bool)
    placed_flat_idx = placed_flat_i32.to(torch.int64)
    return placed_mask, placed_flat_idx


@triton.jit
def _update_parent_radii_atomic_kernel(
    x_ptr,
    parent_idx_ptr,
    parents_ptr,
    parent_radii_ptr,
    n: tl.constexpr,
    d: tl.constexpr,
    x_row_stride: tl.constexpr,
    p_row_stride: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n:
        return

    p = tl.load(parent_idx_ptr + pid).to(tl.int32)

    offs = tl.arange(0, d)
    x = tl.load(x_ptr + pid * x_row_stride + offs, mask=offs < d, other=0.0)
    pc = tl.load(parents_ptr + p * p_row_stride + offs, mask=offs < d, other=0.0)

    diff = x - pc
    dist2 = tl.sum(diff * diff, axis=0)
    dist = tl.sqrt(dist2)

    tl.atomic_max(parent_radii_ptr + p, dist)


def update_parent_radii_atomic(
    *,
    inserted_keys: torch.Tensor,
    inserted_parent_idx: torch.Tensor,
    parents: torch.Tensor,
    parent_radii: torch.Tensor,
) -> None:
    """Atomic max-update of parent radii.

    Requirements:
    - parent_radii must be float32 CUDA (atomic_max support).
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")

    if inserted_keys.numel() == 0:
        return

    if parent_radii.dtype != torch.float32:
        raise ValueError("parent_radii must be float32 for atomic update")

    x = inserted_keys.float().contiguous()
    p = inserted_parent_idx.to(torch.int32).contiguous()
    parents_f = parents.float().contiguous()

    n, d = x.shape

    grid = (triton.cdiv(n, 256) * 256,)
    _update_parent_radii_atomic_kernel[grid](
        x,
        p,
        parents_f,
        parent_radii,
        n=n,
        d=d,
        x_row_stride=x.stride(0),
        p_row_stride=parents_f.stride(0),
        num_warps=4,
    )


@triton.jit
def _scatter_parent_children_blocks_kernel(
    src_parents_ptr,
    src_pr_ptr,
    src_children_ptr,  # flat: (n*bf, d)
    dst_parent_idx_ptr,
    dst_parents_ptr,
    dst_pr_ptr,
    dst_children_ptr,  # flat: (P*bf, d)
    n,
    d: tl.constexpr,
    bf: tl.constexpr,
    src_p_row_stride: tl.constexpr,
    src_c_row_stride: tl.constexpr,
    dst_p_row_stride: tl.constexpr,
    dst_c_row_stride: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n:
        return

    dst = tl.load(dst_parent_idx_ptr + pid).to(tl.int32)

    offs = tl.arange(0, d)
    p = tl.load(
        src_parents_ptr + pid * src_p_row_stride + offs,
        mask=offs < d,
        other=0.0,
    )
    tl.store(
        dst_parents_ptr + dst * dst_p_row_stride + offs,
        p,
        mask=offs < d,
    )

    pr = tl.load(src_pr_ptr + pid)
    tl.store(dst_pr_ptr + dst, pr)

    # Copy bf child rows.
    # src_children is flattened, where child row j for parent pid is at (pid*bf + j).
    for j in tl.static_range(0, bf):
        c = tl.load(
            src_children_ptr + (pid * bf + j) * src_c_row_stride + offs,
            mask=offs < d,
            other=0.0,
        )
        tl.store(
            dst_children_ptr + (dst * bf + j) * dst_c_row_stride + offs,
            c,
            mask=offs < d,
        )


def scatter_parent_children_blocks(
    *,
    src_parents: torch.Tensor,  # (n,d)
    src_parent_radii: torch.Tensor,  # (n,)
    src_children_flat: torch.Tensor,  # (n*bf, d)
    dst_parent_idx: torch.Tensor,  # (n,) int64/int32, indices in [0, P)
    dst_parents: torch.Tensor,  # (P,d)
    dst_parent_radii: torch.Tensor,  # (P,)
    dst_children_flat: torch.Tensor,  # (P*bf,d)
    bf: int,
) -> None:
    """Scatter parent rows + radii + bf-child blocks into destination parent slots.

    This is used to speed packing of new parent blocks when creating new grandparents.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")

    if src_parents.numel() == 0:
        return

    if src_parents.ndim != 2:
        raise ValueError("src_parents must be (n,d)")
    n, d = src_parents.shape

    if src_children_flat.ndim != 2 or src_children_flat.shape[0] != n * int(bf):
        raise ValueError("src_children_flat must be (n*bf,d)")

    if not (
        src_parents.is_cuda
        and src_parent_radii.is_cuda
        and src_children_flat.is_cuda
        and dst_parent_idx.is_cuda
        and dst_parents.is_cuda
        and dst_parent_radii.is_cuda
        and dst_children_flat.is_cuda
    ):
        raise ValueError("all inputs must be CUDA tensors")

    src_p = src_parents.contiguous()
    src_pr = src_parent_radii.contiguous()
    src_c = src_children_flat.contiguous()

    dst_idx = dst_parent_idx
    if dst_idx.dtype != torch.int32:
        dst_idx = dst_idx.to(torch.int32)
    dst_idx = dst_idx.contiguous()

    dst_p = dst_parents.contiguous()
    dst_pr = dst_parent_radii.contiguous()
    dst_c = dst_children_flat.contiguous()

    grid = (triton.cdiv(n, 256) * 256,)
    _scatter_parent_children_blocks_kernel[grid](
        src_p,
        src_pr,
        src_c,
        dst_idx,
        dst_p,
        dst_pr,
        dst_c,
        n=n,
        d=d,
        bf=int(bf),
        src_p_row_stride=src_p.stride(0),
        src_c_row_stride=src_c.stride(0),
        dst_p_row_stride=dst_p.stride(0),
        dst_c_row_stride=dst_c.stride(0),
        num_warps=4,
    )
