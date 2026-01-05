import triton
import triton.language as tl
import torch


@triton.jit
def two_level_filter_kernel(
    K_ptr,  # Keys fp32
    P_ptr,  # Parent centers fp32
    R_ptr,  # Parent radii fp32
    q_ptr,  # Query vector fp32
    out_ptr,
    t,  # threshold fp32
    K_row_stride: tl.constexpr,
    P_row_stride: tl.constexpr,
    n_cols: tl.constexpr,  # 128
    BLOCK_C: tl.constexpr,  # e.g. 64
    branching_factor: tl.constexpr,  # e.g. 16
):
    pid = tl.program_id(0)  # parent id
    child_base = pid * branching_factor

    # --- load q once
    d = tl.arange(0, n_cols)  # 0..127
    q = tl.load(q_ptr + d)

    # --- parent dot: tl.dot(P[pid], q)
    p_ptrs = P_ptr + pid * P_row_stride + d
    p = tl.load(p_ptrs)

    parent_dot = tl.sum(p * q, axis=0)  # (safe fallback)
    # parent_dot = tl.dot(p, q)  # check if your Triton version supports 1D tl.dot

    r = tl.load(R_ptr + pid)
    passes = (parent_dot + r) > t

    if passes:
        # process 'branching_factor' children in chunks of BLOCK_C
        for c0 in tl.static_range(0, branching_factor, BLOCK_C):
            c = c0 + tl.arange(0, BLOCK_C)  # local child ids
            k_row_ids = child_base + c  # global child ids

            # load K tile: shape (BLOCK_C, 128)
            # addresses: K[k_row_ids, d]
            k_ptrs = K_ptr + k_row_ids[:, None] * K_row_stride + d[None, :]
            K_tile = tl.load(k_ptrs)

            # matvec: scores for this chunk
            scores = tl.sum(K_tile * q[None, :], axis=1)
            # scores = tl.dot(K_tile, q)

            # store to flat output
            tl.store(out_ptr + k_row_ids, scores)


import triton
import triton.language as tl


@triton.jit
def two_level_filter_kernel_masked(
    K_ptr,
    P_ptr,
    R_ptr,
    q_ptr,
    out_ptr,
    parent_mask_ptr,  # use this mask to quickly filter out some parents without calc dot prod.
    t,
    K_row_stride: tl.constexpr,
    P_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_C: tl.constexpr,
    branching_factor: tl.constexpr,
):
    pid = tl.program_id(0)
    child_base = pid * branching_factor

    # ---- parent mask gate (early)
    # tl.load returns integer type; convert to predicate
    m = tl.load(parent_mask_ptr + pid)
    if m == 0:
        return  # treat as filtered-out parent; leave output untouched

    # ---- load q once
    d = tl.arange(0, n_cols)
    q = tl.load(q_ptr + d)

    # ---- parent dot
    p = tl.load(P_ptr + pid * P_row_stride + d)
    parent_dot = tl.sum(p * q, axis=0)
    r = tl.load(R_ptr + pid)

    if (parent_dot + r) <= t:
        return

    # ---- children
    for c0 in tl.static_range(0, branching_factor, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        k_row_ids = child_base + c

        k_ptrs = K_ptr + k_row_ids[:, None] * K_row_stride + d[None, :]
        K_tile = tl.load(k_ptrs)

        scores = tl.sum(K_tile * q[None, :], axis=1)
        tl.store(out_ptr + k_row_ids, scores)


@triton.jit
def three_level_filter_kernel_v1(
    K_ptr,
    P1_ptr,
    R1_ptr,
    P2_ptr,
    R2_ptr,
    q_ptr,
    out_ptr,
    t,
    K_row_stride: tl.constexpr,
    P1_row_stride: tl.constexpr,
    P2_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    branch: tl.constexpr,
):
    # K -> P1 -> P2 (bottom up layers)
    # K = branch * (branch * |P2|)
    # Each program handles one p2 parent
    pid = tl.program_id(0)

    # load query
    d = tl.arange(0, n_cols)
    q = tl.load(q_ptr + d)

    p2_ptr = P2_ptr + pid * P2_row_stride + d
    p2 = tl.load(p2_ptr)

    parent2_dot = tl.sum(p2 * q, axis=0)

    r2 = tl.load(R2_ptr + pid)
    passes2 = (parent2_dot + r2) > t

    if not passes2:
        return

    # filter the middle level (P1)
    # load P1 tile: shape (branch, 128)
    p1_ids = pid * branch + tl.arange(0, branch)
    p1_ptrs = P1_ptr + p1_ids[:, None] * P1_row_stride + d[None, :]
    P1_tile = tl.load(p1_ptrs)

    p1_dots = tl.sum(P1_tile * q[None, :], axis=1)
    r1 = tl.load(R1_ptr + p1_ids)
    passes1 = (p1_dots + r1) > t

    j = tl.arange(0, branch)

    # calculate all, but only store the passed ones
    k_ids = p1_ids[:, None] * branch + j[None, :]
    k_ptrs = K_ptr + k_ids[:, :, None] * K_row_stride + d[None, None, :]
    K_tile = tl.load(k_ptrs)

    scores = tl.sum(K_tile * q[None, None, :], axis=2)  # [B, B]

    store_mask = passes1[:, None]  # [B, 1] -> broadcast to [B, B]
    tl.store(out_ptr + k_ids, scores, mask=store_mask)


@triton.jit
def three_level_filter_kernel_v2(
    K_ptr,
    P1_ptr,
    R1_ptr,
    P2_ptr,
    R2_ptr,
    q_ptr,
    out_ptr,
    t,
    K_row_stride: tl.constexpr,
    P1_row_stride: tl.constexpr,
    P2_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    branch: tl.constexpr,
):
    # Each program id handles one p1 node
    p1_id = tl.program_id(0)
    p2_id = p1_id // branch

    # load query
    d = tl.arange(0, n_cols)
    q = tl.load(q_ptr + d)

    # filter by p2
    p2 = tl.load(P2_ptr + p2_id * P2_row_stride + d)
    p2_dot = tl.sum(p2 * q, axis=0)
    r2 = tl.load(R2_ptr + p2_id)

    if (p2_dot + r2) <= t:
        return  # leave untouched

    # filter by p1
    p1 = tl.load(P1_ptr + p1_id * P1_row_stride + d)
    p1_dot = tl.sum(p1 * q, axis=0)
    r1 = tl.load(R1_ptr + p1_id)

    if (p1_dot + r1) <= t:
        return  # leave output untouched

    j = tl.arange(0, branch)
    k_ids = p1_id * branch + j

    k_ptrs = K_ptr + k_ids[:, None] * K_row_stride + d[None, :]
    K_tile = tl.load(k_ptrs)

    scores = tl.sum(K_tile * q[None, :], axis=1)  # [branch]
    tl.store(out_ptr + k_ids, scores)
