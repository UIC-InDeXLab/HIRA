import torch

from hira.kernels.triton_search_kernels import (
    two_level_filter_kernel_batched,
    two_level_filter_kernel_masked_batched,
    three_level_filter_kernel_v1_batched,
    three_level_filter_kernel_v2_batched,
)


def _coerce_query(q: torch.Tensor, *, H: int, d: int, device: torch.device):
    if not isinstance(q, torch.Tensor):
        raise TypeError(f"q must be a torch.Tensor, got {type(q)}")
    if q.ndim == 4:
        # cache-style format: (1, H, 1, D)
        if q.shape[0] != 1 or q.shape[2] != 1:
            raise ValueError(f"4D query must be (1,H,1,D), got {tuple(q.shape)}")
        q = q.squeeze(0).squeeze(-2)
    elif q.ndim != 2:
        raise ValueError(f"q must be (H,d) or (1,H,1,D), got {tuple(q.shape)}")

    if q.shape[-1] != d:
        raise ValueError(f"query dim mismatch: expected d={d}, got {q.shape[-1]}")

    if q.shape[0] != H:
        raise ValueError(f"query head mismatch: expected H={H}, got {q.shape[0]}")

    q = q.to(device=device).contiguous()
    if not q.is_cuda:
        raise ValueError("q must be CUDA")
    return q


def _coerce_threshold(t, *, H: int, device: torch.device):
    if torch.is_tensor(t):
        t = t.to(device=device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.reshape(1)
        else:
            t = t.reshape(-1)
    else:
        t = torch.tensor([float(t)], device=device, dtype=torch.float32)

    if t.numel() == 1 and H > 1:
        t = t.expand(H)
    if t.numel() != H:
        raise ValueError(f"threshold must be scalar or length H={H}, got {t.numel()}")
    return t.contiguous()


def _coerce_out(out, *, H: int, n: int, device: torch.device, dtype: torch.dtype):
    if out is None:
        return torch.zeros((H, n), device=device, dtype=dtype)

    if not isinstance(out, torch.Tensor):
        raise TypeError(f"out must be a torch.Tensor, got {type(out)}")
    if out.ndim != 2:
        raise ValueError(f"out must be 2D (H,n), got {tuple(out.shape)}")
    if out.shape != (H, n):
        raise ValueError(f"out shape mismatch: expected {(H, n)}, got {out.shape}")

    if not out.is_cuda:
        raise ValueError("out must be CUDA")
    return out


def _check_layout_3d_last_contiguous(x: torch.Tensor, name: str):
    if x.stride(-1) != 1:
        raise ValueError(
            f"{name} must be contiguous in the last dim; got stride={x.stride()}"
        )


def triton_two_level_filter(K, P, R, q, t, *, branch, BLOCK_C, n_warps=4, out=None):
    """
    Multi-head only:
    - K: (H, N, D)
    - P: (H, M, D)
    - R: (H, M)
    - q: (H, D) or (1, H, 1, D)
    - out: (H, N)
    """
    if K.ndim != 3:
        raise ValueError(f"K must be (H,n,d), got {tuple(K.shape)}")
    if P.ndim != 3:
        raise ValueError(f"P must be (H,m,d), got {tuple(P.shape)}")
    if R.ndim != 2:
        raise ValueError(f"R must be (H,m), got {tuple(R.shape)}")

    if not (K.is_cuda and P.is_cuda and R.is_cuda):
        raise ValueError("K, P, R must be CUDA tensors")

    H, n, d = K.shape
    if n % branch != 0:
        raise ValueError(f"n must be divisible by branch; got n={n}, branch={branch}")
    m = n // branch

    if P.shape != (H, m, d):
        raise ValueError(
            f"P shape mismatch: expected {(H, m, d)}, got {tuple(P.shape)}"
        )
    if R.shape != (H, m):
        raise ValueError(f"R shape mismatch: expected {(H, m)}, got {tuple(R.shape)}")
    if BLOCK_C > branch or (branch % BLOCK_C) != 0:
        raise ValueError(
            f"BLOCK_C must divide branch and be <= branch; got {BLOCK_C}, {branch}"
        )

    _check_layout_3d_last_contiguous(K, "K")
    _check_layout_3d_last_contiguous(P, "P")

    q = _coerce_query(q, H=H, d=d, device=K.device)
    t = _coerce_threshold(t, H=H, device=K.device)

    if q.stride(-1) != 1:
        raise ValueError(
            f"q must be contiguous in the last dim; got stride={q.stride()}"
        )

    out = _coerce_out(out, H=H, n=n, device=K.device, dtype=K.dtype)

    grid = (m, H)
    two_level_filter_kernel_batched[grid](
        K_ptr=K,
        P_ptr=P,
        R_ptr=R,
        q_ptr=q,
        out_ptr=out,
        t_ptr=t,
        K_h_stride=K.stride(0),
        K_row_stride=K.stride(1),
        P_h_stride=P.stride(0),
        P_row_stride=P.stride(1),
        R_h_stride=R.stride(0),
        R_row_stride=R.stride(1),
        q_h_stride=q.stride(0),
        q_d_stride=q.stride(1),
        out_h_stride=out.stride(0),
        out_row_stride=out.stride(1),
        t_h_stride=t.stride(0),
        n_cols=d,
        BLOCK_C=BLOCK_C,
        branching_factor=branch,
        num_warps=n_warps,
    )

    return out


def triton_three_level_filter_v1(
    K, P1, R1, P2, R2, q, t, *, branch, BLOCK_C=None, n_warps=4, out=None
):
    """
    Multi-head only.
    Uses two batched masked two-level passes:
    1) P2 -> P1
    2) P1 -> K
    """
    if BLOCK_C is None:
        BLOCK_C = branch

    if K.ndim != 3:
        raise ValueError(f"K must be (H,n,d), got {tuple(K.shape)}")
    if P1.ndim != 3:
        raise ValueError(f"P1 must be (H,n_p1,d), got {tuple(P1.shape)}")
    if R1.ndim != 2:
        raise ValueError(f"R1 must be (H,n_p1), got {tuple(R1.shape)}")
    if P2.ndim != 3:
        raise ValueError(f"P2 must be (H,n_p2,d), got {tuple(P2.shape)}")
    if R2.ndim != 2:
        raise ValueError(f"R2 must be (H,n_p2), got {tuple(R2.shape)}")

    if not (K.is_cuda and P1.is_cuda and R1.is_cuda and P2.is_cuda and R2.is_cuda):
        raise ValueError("K, P1, R1, P2, R2 must be CUDA tensors")

    H, N_K, d = K.shape
    if N_K % branch != 0:
        raise ValueError(
            f"N_K must be divisible by branch; got N_K={N_K}, branch={branch}"
        )
    N_P1 = N_K // branch
    if N_P1 % branch != 0:
        raise ValueError(
            f"N_P1 must be divisible by branch; got N_P1={N_P1}, branch={branch}"
        )
    N_P2 = N_P1 // branch

    if P1.shape != (H, N_P1, d):
        raise ValueError(
            f"P1 shape mismatch: expected {(H, N_P1, d)}, got {tuple(P1.shape)}"
        )
    if R1.shape != (H, N_P1):
        raise ValueError(
            f"R1 shape mismatch: expected {(H, N_P1)}, got {tuple(R1.shape)}"
        )
    if P2.shape != (H, N_P2, d):
        raise ValueError(
            f"P2 shape mismatch: expected {(H, N_P2, d)}, got {tuple(P2.shape)}"
        )
    if R2.shape != (H, N_P2):
        raise ValueError(
            f"R2 shape mismatch: expected {(H, N_P2)}, got {tuple(R2.shape)}"
        )
    if BLOCK_C > branch or (branch % BLOCK_C) != 0:
        raise ValueError(
            f"BLOCK_C must divide branch and be <= branch; got {BLOCK_C}, {branch}"
        )

    _check_layout_3d_last_contiguous(K, "K")
    _check_layout_3d_last_contiguous(P1, "P1")
    _check_layout_3d_last_contiguous(P2, "P2")

    q = _coerce_query(q, H=H, d=d, device=K.device)
    t = _coerce_threshold(t, H=H, device=K.device)
    if q.stride(-1) != 1:
        raise ValueError(
            f"q must be contiguous in the last dim; got stride={q.stride()}"
        )

    out = _coerce_out(out, H=H, n=N_K, device=K.device, dtype=K.dtype)

    out_p1 = torch.full((H, N_P1), float("-inf"), device=K.device, dtype=K.dtype)
    parent_mask_p2 = torch.ones((H, N_P2), device=K.device, dtype=torch.uint8)

    grid1 = (N_P2, H)
    two_level_filter_kernel_masked_batched[grid1](
        K_ptr=P1,
        P_ptr=P2,
        R_ptr=R2,
        q_ptr=q,
        out_ptr=out_p1,
        parent_mask_ptr=parent_mask_p2,
        t_ptr=t,
        K_h_stride=P1.stride(0),
        K_row_stride=P1.stride(1),
        P_h_stride=P2.stride(0),
        P_row_stride=P2.stride(1),
        R_h_stride=R2.stride(0),
        R_row_stride=R2.stride(1),
        q_h_stride=q.stride(0),
        q_d_stride=q.stride(1),
        out_h_stride=out_p1.stride(0),
        out_row_stride=out_p1.stride(1),
        pm_h_stride=parent_mask_p2.stride(0),
        pm_row_stride=parent_mask_p2.stride(1),
        t_h_stride=t.stride(0),
        n_cols=d,
        BLOCK_C=BLOCK_C,
        branching_factor=branch,
        num_warps=n_warps,
    )

    parent_mask_p1 = torch.isfinite(out_p1).to(torch.uint8)

    grid2 = (N_P1, H)
    two_level_filter_kernel_masked_batched[grid2](
        K_ptr=K,
        P_ptr=P1,
        R_ptr=R1,
        q_ptr=q,
        out_ptr=out,
        parent_mask_ptr=parent_mask_p1,
        t_ptr=t,
        K_h_stride=K.stride(0),
        K_row_stride=K.stride(1),
        P_h_stride=P1.stride(0),
        P_row_stride=P1.stride(1),
        R_h_stride=R1.stride(0),
        R_row_stride=R1.stride(1),
        q_h_stride=q.stride(0),
        q_d_stride=q.stride(1),
        out_h_stride=out.stride(0),
        out_row_stride=out.stride(1),
        pm_h_stride=parent_mask_p1.stride(0),
        pm_row_stride=parent_mask_p1.stride(1),
        t_h_stride=t.stride(0),
        n_cols=d,
        BLOCK_C=BLOCK_C,
        branching_factor=branch,
        num_warps=n_warps,
    )

    return out


def _prepare_three_level_inputs(K, P1, R1, P2, R2, q, t, *, branch, BLOCK_C, out):
    if BLOCK_C is None:
        BLOCK_C = branch

    if K.ndim != 3:
        raise ValueError(f"K must be (H,n,d), got {tuple(K.shape)}")
    if P1.ndim != 3:
        raise ValueError(f"P1 must be (H,n_p1,d), got {tuple(P1.shape)}")
    if R1.ndim != 2:
        raise ValueError(f"R1 must be (H,n_p1), got {tuple(R1.shape)}")
    if P2.ndim != 3:
        raise ValueError(f"P2 must be (H,n_p2,d), got {tuple(P2.shape)}")
    if R2.ndim != 2:
        raise ValueError(f"R2 must be (H,n_p2), got {tuple(R2.shape)}")

    if not (K.is_cuda and P1.is_cuda and R1.is_cuda and P2.is_cuda and R2.is_cuda):
        raise ValueError("K, P1, R1, P2, R2 must be CUDA tensors")

    H, N_K, d = K.shape
    if N_K % branch != 0:
        raise ValueError(
            f"N_K must be divisible by branch; got N_K={N_K}, branch={branch}"
        )
    N_P1 = N_K // branch
    if N_P1 % branch != 0:
        raise ValueError(
            f"N_P1 must be divisible by branch; got N_P1={N_P1}, branch={branch}"
        )
    N_P2 = N_P1 // branch

    if P1.shape != (H, N_P1, d):
        raise ValueError(
            f"P1 shape mismatch: expected {(H, N_P1, d)}, got {tuple(P1.shape)}"
        )
    if R1.shape != (H, N_P1):
        raise ValueError(
            f"R1 shape mismatch: expected {(H, N_P1)}, got {tuple(R1.shape)}"
        )
    if P2.shape != (H, N_P2, d):
        raise ValueError(
            f"P2 shape mismatch: expected {(H, N_P2, d)}, got {tuple(P2.shape)}"
        )
    if R2.shape != (H, N_P2):
        raise ValueError(
            f"R2 shape mismatch: expected {(H, N_P2)}, got {tuple(R2.shape)}"
        )
    if BLOCK_C > branch or (branch % BLOCK_C) != 0:
        raise ValueError(
            f"BLOCK_C must divide branch and be <= branch; got {BLOCK_C}, {branch}"
        )

    _check_layout_3d_last_contiguous(K, "K")
    _check_layout_3d_last_contiguous(P1, "P1")
    _check_layout_3d_last_contiguous(P2, "P2")

    q = _coerce_query(q, H=H, d=d, device=K.device)
    t = _coerce_threshold(t, H=H, device=K.device)
    if q.stride(-1) != 1:
        raise ValueError(
            f"q must be contiguous in the last dim; got stride={q.stride()}"
        )

    out = _coerce_out(out, H=H, n=N_K, device=K.device, dtype=K.dtype)
    return H, N_P1, N_P2, d, q, t, out, BLOCK_C


def triton_three_level_filter_kernel_v1(
    K, P1, R1, P2, R2, q, t, *, branch, BLOCK_C=None, n_warps=4, out=None
):
    """
    One-pass three-level traversal.
    Program granularity: one program per (head, P2 node).
    """
    H, _N_P1, N_P2, d, q, t, out, BLOCK_C = _prepare_three_level_inputs(
        K, P1, R1, P2, R2, q, t, branch=branch, BLOCK_C=BLOCK_C, out=out
    )

    grid = (N_P2, H)
    three_level_filter_kernel_v1_batched[grid](
        K_ptr=K,
        P1_ptr=P1,
        R1_ptr=R1,
        P2_ptr=P2,
        R2_ptr=R2,
        q_ptr=q,
        out_ptr=out,
        t_ptr=t,
        K_h_stride=K.stride(0),
        K_row_stride=K.stride(1),
        P1_h_stride=P1.stride(0),
        P1_row_stride=P1.stride(1),
        R1_h_stride=R1.stride(0),
        R1_row_stride=R1.stride(1),
        P2_h_stride=P2.stride(0),
        P2_row_stride=P2.stride(1),
        R2_h_stride=R2.stride(0),
        R2_row_stride=R2.stride(1),
        q_h_stride=q.stride(0),
        q_d_stride=q.stride(1),
        out_h_stride=out.stride(0),
        out_row_stride=out.stride(1),
        t_h_stride=t.stride(0),
        n_cols=d,
        branch=branch,
        BLOCK_C=BLOCK_C,
        num_warps=n_warps,
    )
    return out


def triton_three_level_filter_kernel_v2(
    K, P1, R1, P2, R2, q, t, *, branch, BLOCK_C=None, n_warps=4, out=None
):
    """
    One-pass three-level traversal.
    Program granularity: one program per (head, P1 node).
    """
    H, N_P1, _N_P2, d, q, t, out, BLOCK_C = _prepare_three_level_inputs(
        K, P1, R1, P2, R2, q, t, branch=branch, BLOCK_C=BLOCK_C, out=out
    )

    grid = (N_P1, H)
    three_level_filter_kernel_v2_batched[grid](
        K_ptr=K,
        P1_ptr=P1,
        R1_ptr=R1,
        P2_ptr=P2,
        R2_ptr=R2,
        q_ptr=q,
        out_ptr=out,
        t_ptr=t,
        K_h_stride=K.stride(0),
        K_row_stride=K.stride(1),
        P1_h_stride=P1.stride(0),
        P1_row_stride=P1.stride(1),
        R1_h_stride=R1.stride(0),
        R1_row_stride=R1.stride(1),
        P2_h_stride=P2.stride(0),
        P2_row_stride=P2.stride(1),
        R2_h_stride=R2.stride(0),
        R2_row_stride=R2.stride(1),
        q_h_stride=q.stride(0),
        q_d_stride=q.stride(1),
        out_h_stride=out.stride(0),
        out_row_stride=out.stride(1),
        t_h_stride=t.stride(0),
        n_cols=d,
        branch=branch,
        BLOCK_C=BLOCK_C,
        num_warps=n_warps,
    )
    return out
