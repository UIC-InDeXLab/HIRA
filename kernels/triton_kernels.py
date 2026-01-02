import triton
import triton.language as tl


@triton.jit
def parent_child_filter_kernel(
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
