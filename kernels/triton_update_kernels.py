import torch
import triton
import triton.language as tl


@triton.jit
def _nearest_l2_stage1_kernel_batched(
    x_ptr,  # *fp16/fp32, (H,N,D)
    centers_ptr,  # *fp16/fp32, (H,M,D)
    centers_norm2_ptr,  # *fp32,      (H,M)
    valid_mask_ptr,  # *int8/bool, (H,M) or 0
    out_best_score_ptr,  # *fp32,      (H,N,nblocks)
    out_best_idx_ptr,  # *int32,     (H,N,nblocks)
    # strides in *elements* (not bytes)
    x_h_stride: tl.constexpr,  # stride between heads in x
    x_n_stride: tl.constexpr,  # stride between rows (queries) in x within a head
    c_h_stride: tl.constexpr,  # stride between heads in centers
    c_m_stride: tl.constexpr,  # stride between rows (centers) in centers within a head
    cn_h_stride: tl.constexpr,  # stride between heads in centers_norm2
    vm_h_stride: tl.constexpr,  # stride between heads in valid_mask
    out_h_stride: tl.constexpr,  # stride between heads in output
    out_n_stride: tl.constexpr,  # stride between query rows in output (within head)
    out_b_stride: tl.constexpr,  # stride between block columns in output (within (H,N,*))
    d: tl.constexpr,
    m: tl.constexpr,
    use_valid_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # 3D launch: (qid, bid, hid)
    qid = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)

    # ---- base pointers for this head ----
    x_ptr_h = x_ptr + hid * x_h_stride
    c_ptr_h = centers_ptr + hid * c_h_stride
    cn_ptr_h = centers_norm2_ptr + hid * cn_h_stride

    if use_valid_mask:
        vm_ptr_h = valid_mask_ptr + hid * vm_h_stride

    out_score_h = out_best_score_ptr + hid * out_h_stride
    out_idx_h = out_best_idx_ptr + hid * out_h_stride

    # ---- load x[hid, qid, :] ----
    offs_d = tl.arange(0, d)
    x = tl.load(x_ptr_h + qid * x_n_stride + offs_d, mask=offs_d < d, other=0.0).to(
        tl.float32
    )

    # ---- load centers block: centers[hid, center_ids, :] ----
    center_ids = bid * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    mask_c = center_ids < m

    c_ptrs = c_ptr_h + center_ids[:, None] * c_m_stride + offs_d[None, :]
    c = tl.load(
        c_ptrs,
        mask=mask_c[:, None] & (offs_d[None, :] < d),
        other=0.0,
    ).to(tl.float32)

    # dot: (BLOCK_M,)
    dot = tl.sum(c * x[None, :], axis=1)
    c_norm2 = tl.load(cn_ptr_h + center_ids, mask=mask_c, other=0.0).to(tl.float32)

    # score = ||c||^2 - 2*x·c  (||x||^2 constant across centers)
    score = c_norm2 - 2.0 * dot

    if use_valid_mask:
        v = tl.load(vm_ptr_h + center_ids, mask=mask_c, other=0).to(tl.int1)
        score = tl.where(v, score, float("inf"))

    neg_score = -score
    neg_score = tl.where(mask_c, neg_score, float("-inf"))

    local_j = tl.argmax(neg_score, axis=0)
    local_best = -tl.max(neg_score, axis=0)
    local_idx = bid * BLOCK_M + local_j

    # write out for this (hid, qid, bid)
    tl.store(out_score_h + qid * out_n_stride + bid * out_b_stride, local_best)
    tl.store(out_idx_h + qid * out_n_stride + bid * out_b_stride, local_idx)


def nearest_l2_stage1_batched(x, centers, centers_norm2, valid_mask=None, BLOCK_M=128):
    # x: (H,N,D), centers: (H,M,D), centers_norm2: (H,M)
    assert x.ndim == 3 and centers.ndim == 3
    H, N, D = x.shape
    _, M, _ = centers.shape
    nblocks = triton.cdiv(M, BLOCK_M)

    out_best_score = torch.empty((H, N, nblocks), device=x.device, dtype=torch.float32)
    out_best_idx = torch.empty((H, N, nblocks), device=x.device, dtype=torch.int32)

    use_valid_mask = valid_mask is not None
    vm_ptr = valid_mask if use_valid_mask else x  # dummy (won't be used if flag false)

    grid = (N, nblocks, H)

    _nearest_l2_stage1_kernel_batched[grid](
        x,
        centers,
        centers_norm2,
        vm_ptr,
        out_best_score,
        out_best_idx,
        x_h_stride=x.stride(0),
        x_n_stride=x.stride(1),
        c_h_stride=centers.stride(0),
        c_m_stride=centers.stride(1),
        cn_h_stride=centers_norm2.stride(0),
        vm_h_stride=valid_mask.stride(0) if use_valid_mask else 0,
        out_h_stride=out_best_score.stride(0),
        out_n_stride=out_best_score.stride(1),
        out_b_stride=out_best_score.stride(2),
        d=D,
        m=M,
        use_valid_mask=use_valid_mask,
        BLOCK_M=BLOCK_M,
        num_warps=4,
    )

    return out_best_idx, out_best_score


@triton.jit
def _fill_children_atomic_kernel_batched(
    x_ptr,  # (H,N,D)
    parent_idx_ptr,  # (H,N)
    child_counts_ptr,  # (H,P)
    children_ptr,  # (H,P*bf,D)
    placed_ptr,  # (H,N) uint8/int8
    placed_flat_idx_ptr,  # (H,N) int32
    n,  # runtime N
    d: tl.constexpr,  # constexpr D
    bf: tl.constexpr,  # constexpr bf
    # strides in elements
    x_h_stride: tl.constexpr,  # x.stride(0)
    x_row_stride: tl.constexpr,  # x.stride(1)
    x_d_stride: tl.constexpr,  # x.stride(2)
    p_h_stride: tl.constexpr,  # parent_idx.stride(0)
    p_row_stride: tl.constexpr,  # parent_idx.stride(1)
    cc_h_stride: tl.constexpr,  # child_counts.stride(0)
    cc_p_stride: tl.constexpr,  # child_counts.stride(1)
    ch_h_stride: tl.constexpr,  # children.stride(0)
    ch_row_stride: tl.constexpr,  # children.stride(1)
    ch_d_stride: tl.constexpr,  # children.stride(2)
    placed_h_stride: tl.constexpr,  # placed.stride(0)
    placed_row_stride: tl.constexpr,  # placed.stride(1)
    placedf_h_stride: tl.constexpr,  # placed_flat_idx.stride(0)
    placedf_row_stride: tl.constexpr,  # placed_flat_idx.stride(1)
):
    pid = tl.program_id(0)  # 0..N-1
    hid = tl.program_id(1)  # 0..H-1

    if pid >= n:
        return

    # --- base pointers for head ---
    x_h = x_ptr + hid * x_h_stride
    p_h = parent_idx_ptr + hid * p_h_stride
    cc_h = child_counts_ptr + hid * cc_h_stride
    ch_h = children_ptr + hid * ch_h_stride

    placed_h = placed_ptr + hid * placed_h_stride
    placedf_h = placed_flat_idx_ptr + hid * placedf_h_stride

    # read parent id for this (hid, pid)
    p = tl.load(p_h + pid * p_row_stride).to(tl.int32)

    # reserve a slot (atomic, per (hid,p))
    slot = tl.atomic_add(cc_h + p * cc_p_stride, 1).to(tl.int32)

    if slot < bf:
        row = p * bf + slot  # in [0, P*bf)
        offs = tl.arange(0, d)

        v = tl.load(
            x_h + pid * x_row_stride + offs * x_d_stride,
            mask=offs < d,
            other=0.0,
        )
        tl.store(
            ch_h + row * ch_row_stride + offs * ch_d_stride,
            v,
            mask=offs < d,
        )

        tl.store(placed_h + pid * placed_row_stride, 1)
        tl.store(placedf_h + pid * placedf_row_stride, row)
    else:
        tl.store(placed_h + pid * placed_row_stride, 0)
        tl.store(placedf_h + pid * placedf_row_stride, -1)


def fill_existing_children_atomic_batched(
    x, parent_idx, child_counts, children, bf: int
):
    # x: (H,N,D)
    H, N, D = x.shape

    placed = torch.empty((H, N), device=x.device, dtype=torch.uint8)
    placed_flat = torch.empty((H, N), device=x.device, dtype=torch.int32)

    grid = (N, H)

    _fill_children_atomic_kernel_batched[grid](
        x,
        parent_idx,
        child_counts,
        children,
        placed,
        placed_flat,
        n=N,
        d=D,
        bf=bf,
        x_h_stride=x.stride(0),
        x_row_stride=x.stride(1),
        x_d_stride=x.stride(2),
        p_h_stride=parent_idx.stride(0),
        p_row_stride=parent_idx.stride(1),
        cc_h_stride=child_counts.stride(0),
        cc_p_stride=child_counts.stride(1),
        ch_h_stride=children.stride(0),
        ch_row_stride=children.stride(1),
        ch_d_stride=children.stride(2),
        placed_h_stride=placed.stride(0),
        placed_row_stride=placed.stride(1),
        placedf_h_stride=placed_flat.stride(0),
        placedf_row_stride=placed_flat.stride(1),
        num_warps=4,
    )

    placed_mask = placed.to(torch.bool)
    return placed_mask, placed_flat


@triton.jit
def _update_parent_radii_atomic_kernel_batched_masked(
    x_ptr,  # (H,N,D)
    parent_idx_ptr,  # (H,N)
    placed_ptr,  # (H,N) uint8/bool (0/1)
    parents_ptr,  # (H,P,D)
    parent_radii_ptr,  # (H,P) fp32
    n,  # runtime N
    d: tl.constexpr,
    # strides
    x_h_stride: tl.constexpr,
    x_n_stride: tl.constexpr,
    x_d_stride: tl.constexpr,
    idx_h_stride: tl.constexpr,
    idx_n_stride: tl.constexpr,
    placed_h_stride: tl.constexpr,
    placed_n_stride: tl.constexpr,
    p_h_stride: tl.constexpr,
    p_p_stride: tl.constexpr,
    p_d_stride: tl.constexpr,
    r_h_stride: tl.constexpr,
    r_p_stride: tl.constexpr,
):
    pid = tl.program_id(0)
    hid = tl.program_id(1)
    if pid >= n:
        return

    x_h = x_ptr + hid * x_h_stride
    idx_h = parent_idx_ptr + hid * idx_h_stride
    pl_h = placed_ptr + hid * placed_h_stride
    p_h = parents_ptr + hid * p_h_stride
    r_h = parent_radii_ptr + hid * r_h_stride

    placed = tl.load(pl_h + pid * placed_n_stride).to(tl.int1)
    if not placed:
        return

    p = tl.load(idx_h + pid * idx_n_stride).to(tl.int32)

    offs = tl.arange(0, d)
    x = tl.load(
        x_h + pid * x_n_stride + offs * x_d_stride, mask=offs < d, other=0.0
    ).to(tl.float32)
    pc = tl.load(p_h + p * p_p_stride + offs * p_d_stride, mask=offs < d, other=0.0).to(
        tl.float32
    )

    diff = x - pc
    dist2 = tl.sum(diff * diff, axis=0)
    dist = tl.sqrt(dist2)

    tl.atomic_max(r_h + p * r_p_stride, dist)


def update_parent_radii_atomic_batched_masked(
    inserted_keys,  # (H,N,D)
    inserted_parent_idx,  # (H,N)
    placed_mask_u8,  # (H,N) uint8 (0/1)
    parents,  # (H,P,D)
    parent_radii,  # (H,P) fp32
):
    H, N, D = inserted_keys.shape
    if parent_radii.dtype != torch.float32:
        raise ValueError("parent_radii must be float32 for atomic_max")

    grid = (N, H)
    _update_parent_radii_atomic_kernel_batched_masked[grid](
        inserted_keys,
        inserted_parent_idx,
        placed_mask_u8,
        parents,
        parent_radii,
        n=N,
        d=D,
        x_h_stride=inserted_keys.stride(0),
        x_n_stride=inserted_keys.stride(1),
        x_d_stride=inserted_keys.stride(2),
        idx_h_stride=inserted_parent_idx.stride(0),
        idx_n_stride=inserted_parent_idx.stride(1),
        placed_h_stride=placed_mask_u8.stride(0),
        placed_n_stride=placed_mask_u8.stride(1),
        p_h_stride=parents.stride(0),
        p_p_stride=parents.stride(1),
        p_d_stride=parents.stride(2),
        r_h_stride=parent_radii.stride(0),
        r_p_stride=parent_radii.stride(1),
        num_warps=4,
    )


@triton.jit
def _nearest_l2_stage2_kernel_batched(
    best_score_ptr,  # (H,N,B) fp32
    best_idx_ptr,  # (H,N,B) int32
    out_score_ptr,  # (H,N) fp32
    out_idx_ptr,  # (H,N) int32
    B: tl.constexpr,
    B_PAD: tl.constexpr,
    # strides (elements)
    s_h: tl.constexpr,
    s_n: tl.constexpr,
    s_b: tl.constexpr,
    i_h: tl.constexpr,
    i_n: tl.constexpr,
    i_b: tl.constexpr,
    o_h: tl.constexpr,
    o_n: tl.constexpr,
):
    nid = tl.program_id(0)  # 0..N-1
    hid = tl.program_id(1)  # 0..H-1

    s_base = best_score_ptr + hid * s_h + nid * s_n
    i_base = best_idx_ptr + hid * i_h + nid * i_n

    offs = tl.arange(0, B_PAD)
    mask = offs < B
    scores = tl.load(s_base + offs * s_b, mask=mask, other=float("inf")).to(tl.float32)

    # pick min score => argmax(-score)
    neg = -scores
    j = tl.argmax(neg, axis=0)
    best_s = -tl.max(neg, axis=0)
    best_i = tl.load(i_base + j * i_b).to(tl.int32)

    tl.store(out_score_ptr + hid * o_h + nid * o_n, best_s)
    tl.store(out_idx_ptr + hid * o_h + nid * o_n, best_i)


def nearest_l2_triton_batched(
    x, centers, valid_mask=None, BLOCK_M=128, num_warps_stage1=4, num_warps_stage2=2
):
    """
    x:       (H,N,D)
    centers: (H,M,D)
    valid_mask: (H,M) or None
    returns:
      best_idx: (H,N) int32
      best_d2_part: (H,N) fp32  where d2_part = ||c||^2 - 2 x·c  (argmin-equivalent to true L2^2)
    """
    H, N, D = x.shape
    _, M, _ = centers.shape
    if M == 0:
        raise ValueError("centers must have non-zero size along dim=1")
    B = (M + BLOCK_M - 1) // BLOCK_M
    B_PAD = triton.next_power_of_2(B)

    # Precompute ||c||^2 for stage1 kernel
    centers_f = centers.float().contiguous()
    centers_norm2 = (centers_f * centers_f).sum(dim=-1)  # (H, M)

    best_idx_blk, best_score_blk = nearest_l2_stage1_batched(
        x,
        centers,
        centers_norm2,
        valid_mask=valid_mask,
        BLOCK_M=BLOCK_M,
    )  # (H,N,B) int32, fp32

    out_score = torch.empty((H, N), device=x.device, dtype=torch.float32)
    out_idx = torch.empty((H, N), device=x.device, dtype=torch.int32)

    grid = (N, H)
    _nearest_l2_stage2_kernel_batched[grid](
        best_score_blk,
        best_idx_blk,
        out_score,
        out_idx,
        B=B,
        B_PAD=B_PAD,
        s_h=best_score_blk.stride(0),
        s_n=best_score_blk.stride(1),
        s_b=best_score_blk.stride(2),
        i_h=best_idx_blk.stride(0),
        i_n=best_idx_blk.stride(1),
        i_b=best_idx_blk.stride(2),
        o_h=out_score.stride(0),
        o_n=out_score.stride(1),
        num_warps=num_warps_stage2,
    )
    return out_idx, out_score
