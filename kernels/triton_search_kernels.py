import triton
import triton.language as tl
import torch


@triton.jit
def two_level_filter_kernel_batched(
    K_ptr,  # (H_kv, N, D)
    P_ptr,  # (H_kv, M, D)
    R_ptr,  # (H_kv, M)
    q_ptr,  # (H_q, D)
    q_head_to_kv_ptr,  # (H_q,)
    out_ptr,  # (H_q, N)
    t_ptr,  # (H_q,)
    scaling_ptr,  # (H_q,)
    K_h_stride: tl.constexpr,
    K_row_stride: tl.constexpr,
    P_h_stride: tl.constexpr,
    P_row_stride: tl.constexpr,
    R_h_stride: tl.constexpr,
    R_row_stride: tl.constexpr,
    q_h_stride: tl.constexpr,
    q_d_stride: tl.constexpr,
    out_h_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    t_h_stride: tl.constexpr,
    scaling_h_stride: tl.constexpr,
    q2kv_h_stride: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_C: tl.constexpr,
    branching_factor: tl.constexpr,
):
    pid = tl.program_id(0)  # parent id
    qid = tl.program_id(1)  # query head id
    child_base = pid * branching_factor

    d = tl.arange(0, n_cols)

    kv_hid = tl.load(q_head_to_kv_ptr + qid * q2kv_h_stride).to(tl.int64)
    q = tl.load(q_ptr + qid * q_h_stride + d * q_d_stride).to(tl.float32)

    p_ptrs = P_ptr + kv_hid * P_h_stride + pid * P_row_stride + d
    p = tl.load(p_ptrs).to(tl.float32)

    parent_dot = tl.sum(p * q, axis=0)
    r = tl.load(R_ptr + kv_hid * R_h_stride + pid * R_row_stride).to(tl.float32)
    t = tl.load(t_ptr + qid * t_h_stride).to(tl.float32)
    scaling = tl.load(scaling_ptr + qid * scaling_h_stride).to(tl.float32)
    if (parent_dot + r) <= t:
        return

    for c0 in tl.static_range(0, branching_factor, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        k_row_ids = child_base + c

        k_ptrs = (
            K_ptr + kv_hid * K_h_stride + k_row_ids[:, None] * K_row_stride + d[None, :]
        )
        K_tile = tl.load(k_ptrs).to(tl.float32)

        scores = tl.sum(K_tile * q[None, :], axis=1)
        scores = scores * scaling
        tl.store(out_ptr + qid * out_h_stride + k_row_ids * out_row_stride, scores)


@triton.jit
def three_level_filter_kernel_v1_batched(
    K_ptr,  # (H_kv, N_K, D)
    P1_ptr,  # (H_kv, N_P1, D)
    R1_ptr,  # (H_kv, N_P1)
    P2_ptr,  # (H_kv, N_P2, D)
    R2_ptr,  # (H_kv, N_P2)
    q_ptr,  # (H_q, D)
    q_head_to_kv_ptr,  # (H_q,)
    out_ptr,  # (H_q, N_K)
    t_ptr,  # (H_q,)
    scaling_ptr,  # (H_q,)
    K_h_stride: tl.constexpr,
    K_row_stride: tl.constexpr,
    P1_h_stride: tl.constexpr,
    P1_row_stride: tl.constexpr,
    R1_h_stride: tl.constexpr,
    R1_row_stride: tl.constexpr,
    P2_h_stride: tl.constexpr,
    P2_row_stride: tl.constexpr,
    R2_h_stride: tl.constexpr,
    R2_row_stride: tl.constexpr,
    q_h_stride: tl.constexpr,
    q_d_stride: tl.constexpr,
    out_h_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    t_h_stride: tl.constexpr,
    scaling_h_stride: tl.constexpr,
    q2kv_h_stride: tl.constexpr,
    n_cols: tl.constexpr,
    branch: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # One program per (query-head, p2)
    p2_id = tl.program_id(0)
    qid = tl.program_id(1)

    d = tl.arange(0, n_cols)
    kv_hid = tl.load(q_head_to_kv_ptr + qid * q2kv_h_stride).to(tl.int64)
    q = tl.load(q_ptr + qid * q_h_stride + d * q_d_stride).to(tl.float32)
    t = tl.load(t_ptr + qid * t_h_stride).to(tl.float32)
    scaling = tl.load(scaling_ptr + qid * scaling_h_stride).to(tl.float32)
    p2 = tl.load(P2_ptr + kv_hid * P2_h_stride + p2_id * P2_row_stride + d).to(
        tl.float32
    )
    p2_dot = tl.sum(p2 * q, axis=0)
    r2 = tl.load(R2_ptr + kv_hid * R2_h_stride + p2_id * R2_row_stride).to(tl.float32)

    if (p2_dot + r2) <= t:
        return

    p1_base = p2_id * branch

    # Traverse all p1 children under this p2.
    for p1_local in tl.static_range(0, branch):
        p1_id = p1_base + p1_local

        p1 = tl.load(P1_ptr + kv_hid * P1_h_stride + p1_id * P1_row_stride + d).to(
            tl.float32
        )
        p1_dot = tl.sum(p1 * q, axis=0)
        r1 = tl.load(R1_ptr + kv_hid * R1_h_stride + p1_id * R1_row_stride).to(
            tl.float32
        )

        if (p1_dot + r1) > t:
            child_base = p1_id * branch
            for c0 in tl.static_range(0, branch, BLOCK_C):
                c = c0 + tl.arange(0, BLOCK_C)
                k_row_ids = child_base + c

                k_ptrs = (
                    K_ptr
                    + kv_hid * K_h_stride
                    + k_row_ids[:, None] * K_row_stride
                    + d[None, :]
                )
                K_tile = tl.load(k_ptrs).to(tl.float32)
                scores = tl.sum(K_tile * q[None, :], axis=1)
                scores = scores * scaling
                tl.store(
                    out_ptr + qid * out_h_stride + k_row_ids * out_row_stride, scores
                )
