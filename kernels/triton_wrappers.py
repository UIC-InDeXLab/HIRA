from triton_kernels import *


# wrapper
def triton_two_level_filter(K, P, R, q, t, *, branch, BLOCK_C, n_warps=4, out=None):
    """
    K: (n, 128) fp16/fp32
    P: (m, 128)
    R: (m,)
    q: (128,)
    t: scalar (python float or 0-d torch tensor)
    out: (n,) preallocated and zeroed outside (or we allocate here)
    """
    assert K.is_cuda and P.is_cuda and R.is_cuda and q.is_cuda
    n, d = K.shape
    assert n % branch == 0
    m = n // branch
    assert P.shape == (m, d)
    assert R.shape == (m,)
    assert BLOCK_C <= branch
    assert branch % BLOCK_C == 0
    assert (
        K.stride(1) == 1 and P.stride(1) == 1 and q.stride(0) == 1
    )  # not sure about this!

    if out is None:
        out = torch.zeros((n,), device=K.device, dtype=K.dtype)

    grid = (m,)
    two_level_filter_kernel[grid](
        K,
        P,
        R,
        q,
        out,
        t,
        K_row_stride=K.stride(0),
        P_row_stride=P.stride(0),
        n_cols=d,
        BLOCK_C=BLOCK_C,
        num_warps=n_warps,
        branching_factor=branch,
    )
    return out


# wrapper
def triton_three_level_filter_v1(
    K, P1, R1, P2, R2, q, t, *, branch, n_warps=4, out=None
):
    assert (
        K.is_cuda
        and P2.is_cuda
        and R2.is_cuda
        and P1.is_cuda
        and R1.is_cuda
        and q.is_cuda
    )

    N_K, d = K.shape
    assert q.shape == (d,)

    assert N_K % branch == 0
    N_P1 = N_K // branch
    assert N_P1 % branch == 0
    N_P2 = N_P1 // branch

    assert P1.shape == (N_P1, d)
    assert R1.shape == (N_P1,)
    assert P2.shape == (N_P2, d)
    assert R2.shape == (N_P2,)

    if out is None:
        out = torch.zeros((N_K,), device=K.device, dtype=K.dtype)

    grid = (N_P2,)

    # Call the correct kernel name you actually defined:
    three_level_filter_kernel_v1[grid](
        K_ptr=K,
        P1_ptr=P1,
        R1_ptr=R1,
        P2_ptr=P2,
        R2_ptr=R2,
        q_ptr=q,
        out_ptr=out,
        t=t,
        K_row_stride=K.stride(0),
        P1_row_stride=P1.stride(0),
        P2_row_stride=P2.stride(0),
        n_cols=d,
        branch=branch,
        num_warps=n_warps,
    )
    return out


# wrapper
def triton_three_level_filter_v2(
    K, P1, R1, P2, R2, q, t, *, branch, n_warps=4, out=None
):
    assert (
        K.is_cuda
        and P2.is_cuda
        and R2.is_cuda
        and P1.is_cuda
        and R1.is_cuda
        and q.is_cuda
    )

    N_K, d = K.shape
    assert q.shape == (d,)

    assert N_K % branch == 0
    N_P1 = N_K // branch
    assert N_P1 % branch == 0
    N_P2 = N_P1 // branch

    assert P1.shape == (N_P1, d)
    assert R1.shape == (N_P1,)
    assert P2.shape == (N_P2, d)
    assert R2.shape == (N_P2,)

    if out is None:
        out = torch.zeros((N_K,), device=K.device, dtype=K.dtype)

    grid = (N_P1,)

    # Call the correct kernel name you actually defined:
    three_level_filter_kernel_v2[grid](
        K_ptr=K,
        P1_ptr=P1,
        R1_ptr=R1,
        P2_ptr=P2,
        R2_ptr=R2,
        q_ptr=q,
        out_ptr=out,
        t=t,
        K_row_stride=K.stride(0),
        P1_row_stride=P1.stride(0),
        P2_row_stride=P2.stride(0),
        n_cols=d,
        branch=branch,
        num_warps=n_warps,
    )
    return out


import torch


def triton_three_level_filter_v3(
    K, P1, R1, P2, R2, q, t, *, branch, n_warps=4, out=None
):
    """
    Three-level filtering implemented by simply calling the masked *two-level* kernel twice:
    """
    BLOCK_C = branch

    assert (
        K.is_cuda
        and P1.is_cuda
        and R1.is_cuda
        and P2.is_cuda
        and R2.is_cuda
        and q.is_cuda
    )
    N_K, d = K.shape
    assert q.shape == (d,)

    assert N_K % branch == 0
    N_P1 = N_K // branch
    assert N_P1 % branch == 0
    N_P2 = N_P1 // branch

    assert P1.shape == (N_P1, d)
    assert R1.shape == (N_P1,)
    assert P2.shape == (N_P2, d)
    assert R2.shape == (N_P2,)

    # re-use the same sanity checks as your 2-level wrapper
    assert BLOCK_C <= branch
    assert branch % BLOCK_C == 0
    assert (
        K.stride(1) == 1
        and P1.stride(1) == 1
        and P2.stride(1) == 1
        and q.stride(0) == 1
    )

    if out is None:
        out = torch.zeros((N_K,), device=K.device, dtype=K.dtype)

    # -------------------------
    # Stage 1: P2 -> P1
    # -------------------------
    out_p1 = torch.full((N_P1,), float("-inf"), device=K.device, dtype=K.dtype)

    parent_mask_p2 = torch.ones((N_P2,), device=K.device, dtype=torch.uint8)  # all pass

    grid1 = (N_P2,)
    two_level_filter_kernel_masked[grid1](
        K_ptr=P1,  # "keys" are P1 rows
        P_ptr=P2,  # parents are P2 rows
        R_ptr=R2,
        q_ptr=q,
        out_ptr=out_p1,  # writes scores for P1 children of passing P2
        parent_mask_ptr=parent_mask_p2,
        t=t,
        K_row_stride=P1.stride(0),
        P_row_stride=P2.stride(0),
        n_cols=d,
        BLOCK_C=BLOCK_C,
        branching_factor=branch,
        num_warps=n_warps,
    )

    # Build P1 parent mask from stage-1 output:
    # - out_p1 is -inf for untouched entries => those P1 did NOT survive P2
    parent_mask_p1 = torch.isfinite(out_p1).to(torch.uint8)

    # -------------------------
    # Stage 2: P1 -> K
    # -------------------------
    grid2 = (N_P1,)
    two_level_filter_kernel_masked[grid2](
        K_ptr=K,
        P_ptr=P1,
        R_ptr=R1,
        q_ptr=q,
        out_ptr=out,
        parent_mask_ptr=parent_mask_p1,
        t=t,
        K_row_stride=K.stride(0),
        P_row_stride=P1.stride(0),
        n_cols=d,
        BLOCK_C=BLOCK_C,
        branching_factor=branch,
        num_warps=n_warps,
    )

    return out
