"""v10.0 Triton kernels for S=4, bf=4 cluster-streaming attention."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _next_pow2_min16(x: int) -> int:
    return max(_next_pow2(x), 16)


if HAS_TRITON:

    @triton.jit
    def _ta_v10_depth_selected_kernel(
        Order_ptr,         # (H_q, 4, K) int64/int32
        SortedScores_ptr,  # (H_q, 4, K) fp16/fp32
        Threshold_ptr,     # (H_q,) fp32
        Depth_ptr,         # (H_q,) int32
        Selected_ptr,      # (H_q, 4, K) int8
        K_CLUSTERS,
        K_BLOCK: tl.constexpr,
    ):
        hq = tl.program_id(0)
        offs = tl.arange(0, K_BLOCK)
        valid = offs < K_CLUSTERS

        row_sum = tl.zeros([K_BLOCK], dtype=tl.float32)
        for s_idx in tl.static_range(4):
            vals = tl.load(
                SortedScores_ptr + (hq * 4 + s_idx) * K_CLUSTERS + offs,
                mask=valid,
                other=0.0,
            )
            row_sum += vals
        th = tl.load(Threshold_ptr + hq).to(tl.float32)
        below = (row_sum < th) & valid
        first = tl.min(tl.where(below, offs, K_CLUSTERS), axis=0)
        depth = tl.where(first < K_CLUSTERS, first + 1, K_CLUSTERS)
        tl.store(Depth_ptr + hq, depth)

        for s_idx in tl.static_range(4):
            base = (hq * 4 + s_idx) * K_CLUSTERS
            tl.store(
                Selected_ptr + base + offs,
                tl.zeros([K_BLOCK], dtype=tl.int8),
                mask=valid,
            )
            in_top = valid & (offs < depth)
            clusters = tl.load(Order_ptr + base + offs, mask=in_top, other=0).to(tl.int32)
            c_valid = in_top & (clusters >= 0) & (clusters < K_CLUSTERS)
            tl.store(
                Selected_ptr + base + clusters,
                tl.full([K_BLOCK], 1, dtype=tl.int8),
                mask=c_valid,
            )

    @triton.jit
    def _ta_v10_build_selected_from_depth_kernel(
        Order_ptr,       # (H_q, 4, K) int64/int32
        Depth_ptr,       # (H_q,) int32
        Selected_ptr,    # (H_q, 4, K) int8
        K_CLUSTERS,
        K_BLOCK: tl.constexpr,
    ):
        hq = tl.program_id(0)
        s_idx = tl.program_id(1)

        offs = tl.arange(0, K_BLOCK)
        valid = offs < K_CLUSTERS
        base = (hq * 4 + s_idx) * K_CLUSTERS

        tl.store(
            Selected_ptr + base + offs,
            tl.zeros([K_BLOCK], dtype=tl.int8),
            mask=valid,
        )

        depth = tl.load(Depth_ptr + hq).to(tl.int32)
        in_top = valid & (offs < depth)
        clusters = tl.load(Order_ptr + base + offs, mask=in_top, other=0).to(tl.int32)
        c_valid = in_top & (clusters >= 0) & (clusters < K_CLUSTERS)
        tl.store(
            Selected_ptr + base + clusters,
            tl.full([K_BLOCK], 1, dtype=tl.int8),
            mask=c_valid,
        )


    @triton.jit
    def _ta_v10_cluster_attn_kernel(
        Q_ptr,               # (H_q, D) fp16
        Order_ptr,           # (H_q, 4, K) int64/int32
        Depth_ptr,           # (H_q,) int32
        Selected_ptr,        # (H_q, 4, K) int8
        Assigns_ptr,         # (4, H_kv, N_pad) int32
        ClusterIds_ptr,      # (4, H_kv, K, 4) int32
        ClusterKeysT_ptr,    # (4, H_kv, K, D, 4) fp16
        ClusterValues_ptr,   # (4, H_kv, K, 4, D_v) fp16
        Threshold_ptr,       # (H_q,) fp32
        BufKeysT_ptr,        # (H_kv, D, L_buf) fp16
        BufValues_ptr,       # (H_kv, L_buf, D_v) fp16
        BufInvalid_ptr,      # (H_kv, L_buf) int8
        M_out_ptr,           # (H_q, NUM_SPLITS) fp32
        L_out_ptr,           # (H_q, NUM_SPLITS) fp32
        O_out_ptr,           # (H_q, NUM_SPLITS, D_v) fp32
        K_CLUSTERS,
        N_PAD,
        ASSIGN_STRIDE_S,
        SCALE_LOG2E: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        H_KV: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BF: tl.constexpr,
        CANDS: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        HAS_BUFFER: tl.constexpr,
        L_BUF_MAX: tl.constexpr,
        BUF_COLS_PER_SPLIT: tl.constexpr,
    ):
        hq = tl.program_id(0)
        split = tl.program_id(1)
        kvh = hq // GROUPS

        d_range = tl.arange(0, D)
        dv_range = tl.arange(0, D_V)
        row_inner = tl.arange(0, BLOCK_ROWS)
        child_inner = tl.arange(0, BF)
        cand_inner = tl.arange(0, CANDS)

        q_vals = tl.load(Q_ptr + hq * D + d_range)
        threshold = tl.load(Threshold_ptr + hq).to(tl.float32)
        depth = tl.load(Depth_ptr + hq).to(tl.int32)

        m = tl.full((), -1.0e30, dtype=tl.float32)
        l_acc = tl.full((), 0.0, dtype=tl.float32)
        o_acc = tl.zeros([D_V], dtype=tl.float32)

        rows_per_split = (depth + NUM_SPLITS - 1) // NUM_SPLITS
        r_start = split * rows_per_split
        r_stop = tl.minimum(r_start + rows_per_split, depth)

        for r_chunk in range(r_start, r_stop, BLOCK_ROWS):
            rows = r_chunk + row_inner
            row_valid = rows < r_stop
            rows_safe = tl.where(row_valid, rows, 0)

            for s_idx in tl.static_range(4):
                clusters = tl.load(
                    Order_ptr + (hq * 4 + s_idx) * K_CLUSTERS + rows_safe,
                    mask=row_valid,
                    other=0,
                ).to(tl.int32)
                cluster_valid = row_valid & (clusters >= 0) & (clusters < K_CLUSTERS)
                clusters_safe = tl.where(cluster_valid, clusters, 0)

                key_ids_2d = tl.load(
                    ClusterIds_ptr
                    + (((s_idx * H_KV + kvh) * K_CLUSTERS + clusters_safe[:, None]) * BF)
                    + child_inner[None, :],
                    mask=cluster_valid[:, None],
                    other=-1,
                ).to(tl.int32)

                cluster_2d = clusters_safe[:, None] + tl.zeros([BLOCK_ROWS, BF], dtype=tl.int32)
                child_2d = child_inner[None, :] + tl.zeros([BLOCK_ROWS, BF], dtype=tl.int32)
                row_valid_2d = cluster_valid[:, None] & (key_ids_2d >= 0) & (key_ids_2d < N_PAD)

                cluster_flat = tl.reshape(cluster_2d, [CANDS])
                child_flat = tl.reshape(child_2d, [CANDS])
                key_flat = tl.reshape(key_ids_2d, [CANDS])
                owned = tl.reshape(row_valid_2d, [CANDS])
                key_safe = tl.where(owned, key_flat, 0)

                # First-selected-subspace ownership removes duplicate keys
                # across subspaces without a visited table.
                for p in tl.static_range(4):
                    parent_p = tl.load(
                        Assigns_ptr + p * ASSIGN_STRIDE_S + kvh * N_PAD + key_safe,
                        mask=owned,
                        other=0,
                    ).to(tl.int32)
                    prev_sel = tl.load(
                        Selected_ptr + (hq * 4 + p) * K_CLUSTERS + parent_p,
                        mask=owned,
                        other=0,
                    )
                    earlier = (p < s_idx) & (prev_sel != 0)
                    owned = owned & ~earlier

                if tl.max(owned.to(tl.int32), axis=0) != 0:
                    keys_tile = tl.load(
                        ClusterKeysT_ptr
                        + ((((s_idx * H_KV + kvh) * K_CLUSTERS + cluster_flat[None, :]) * D
                            + d_range[:, None]) * BF)
                        + child_flat[None, :],
                        mask=owned[None, :],
                        other=0.0,
                    )
                    raw = tl.dot(q_vals[None, :], keys_tile)
                    raw = tl.reshape(raw, [CANDS]).to(tl.float32)
                    survive = owned & (raw >= threshold)
                    scaled = tl.where(survive, raw * SCALE_LOG2E, -1.0e30)

                    chunk_max = tl.max(scaled, axis=0)
                    m_new = tl.maximum(m, chunk_max)
                    alpha = tl.exp2(m - m_new)
                    pvals = tl.exp2(scaled - m_new)
                    pvals = tl.where(survive, pvals, 0.0)
                    l_acc = alpha * l_acc + tl.sum(pvals, axis=0)

                    v_tile = tl.load(
                        ClusterValues_ptr
                        + ((((s_idx * H_KV + kvh) * K_CLUSTERS + cluster_flat[:, None]) * BF
                            + child_flat[:, None]) * D_V)
                        + dv_range[None, :],
                        mask=survive[:, None],
                        other=0.0,
                    )
                    o_acc = alpha * o_acc + tl.reshape(
                        tl.dot(pvals[None, :].to(tl.float16), v_tile),
                        [D_V],
                    )
                    m = m_new

        if HAS_BUFFER:
            buf_start = split * BUF_COLS_PER_SPLIT
            buf_end = tl.minimum(buf_start + BUF_COLS_PER_SPLIT, L_BUF_MAX)
            b_inner = tl.arange(0, BUF_COLS_PER_SPLIT)
            buf_idx = buf_start + b_inner
            buf_valid = buf_idx < buf_end
            buf_safe = tl.where(buf_valid, buf_idx, 0)
            buf_inv = tl.load(
                BufInvalid_ptr + kvh * L_BUF_MAX + buf_safe,
                mask=buf_valid,
                other=1,
            )
            buf_live = buf_valid & (buf_inv == 0)

            if tl.max(buf_live.to(tl.int32), axis=0) != 0:
                buf_keys = tl.load(
                    BufKeysT_ptr
                    + (kvh * D + d_range[:, None]) * L_BUF_MAX
                    + buf_safe[None, :],
                    mask=buf_live[None, :],
                    other=0.0,
                )
                buf_raw = tl.dot(q_vals[None, :], buf_keys)
                buf_raw = tl.reshape(buf_raw, [BUF_COLS_PER_SPLIT]).to(tl.float32)
                buf_scaled = tl.where(buf_live, buf_raw * SCALE_LOG2E, -1.0e30)

                buf_max = tl.max(buf_scaled, axis=0)
                m_new = tl.maximum(m, buf_max)
                alpha = tl.exp2(m - m_new)
                bp = tl.exp2(buf_scaled - m_new)
                bp = tl.where(buf_live, bp, 0.0)
                l_acc = alpha * l_acc + tl.sum(bp, axis=0)

                buf_values = tl.load(
                    BufValues_ptr
                    + (kvh * L_BUF_MAX + buf_safe[:, None]) * D_V
                    + dv_range[None, :],
                    mask=buf_live[:, None],
                    other=0.0,
                )
                o_acc = alpha * o_acc + tl.reshape(
                    tl.dot(bp[None, :].to(tl.float16), buf_values),
                    [D_V],
                )
                m = m_new

        tl.store(M_out_ptr + hq * NUM_SPLITS + split, m)
        tl.store(L_out_ptr + hq * NUM_SPLITS + split, l_acc)
        tl.store(O_out_ptr + (hq * NUM_SPLITS + split) * D_V + dv_range, o_acc)


def run_ta_v10_build_selected_from_depth(
    order_i64: torch.Tensor,
    depth_i32: torch.Tensor,
    selected_i8: torch.Tensor,
    *,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, k_clusters = order_i64.shape
    if int(s_sub) != 4:
        raise ValueError(f"v10 selected builder requires S=4, got {s_sub}")
    _ta_v10_build_selected_from_depth_kernel[(int(h_q), 4)](
        order_i64,
        depth_i32,
        selected_i8,
        int(k_clusters),
        K_BLOCK=_next_pow2(int(k_clusters)),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v10_depth_and_selected(
    order_i64: torch.Tensor,
    sorted_scores: torch.Tensor,
    threshold_f32: torch.Tensor,
    depth_i32: torch.Tensor,
    selected_i8: torch.Tensor,
    *,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, k_clusters = order_i64.shape
    if int(s_sub) != 4:
        raise ValueError(f"v10 depth/selected requires S=4, got {s_sub}")
    _ta_v10_depth_selected_kernel[(int(h_q),)](
        order_i64,
        sorted_scores,
        threshold_f32,
        depth_i32,
        selected_i8,
        int(k_clusters),
        K_BLOCK=_next_pow2(int(k_clusters)),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v10_cluster_attn(
    q: torch.Tensor,
    order_i64: torch.Tensor,
    depth_i32: torch.Tensor,
    selected_i8: torch.Tensor,
    assigns_i32: torch.Tensor,
    cluster_ids_i32: torch.Tensor,
    cluster_keys_t_f16: torch.Tensor,
    cluster_values_f16: torch.Tensor,
    threshold_f32: torch.Tensor,
    buf_keys_t_f16: torch.Tensor | None,
    buf_values_f16: torch.Tensor | None,
    buf_invalid_i8: torch.Tensor | None,
    *,
    n_pad: int,
    scale_log2e: float,
    groups: int,
    block_rows: int,
    num_splits: int,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    d = int(q.shape[-1])
    d_v = int(cluster_values_f16.shape[-1])
    s_sub, h_kv, k_clusters, bf = cluster_ids_i32.shape
    if int(s_sub) != 4 or int(bf) != 4:
        raise ValueError(f"v10 cluster attention requires S=4, bf=4; got S={s_sub}, bf={bf}")

    has_buffer = buf_keys_t_f16 is not None and int(buf_keys_t_f16.shape[-1]) > 0
    if has_buffer:
        l_buf_max = int(buf_keys_t_f16.shape[-1])
        buf_cols_per_split = _next_pow2_min16(
            max(1, (l_buf_max + int(num_splits) - 1) // int(num_splits))
        )
        bk = buf_keys_t_f16
        bv = buf_values_f16
        bi = buf_invalid_i8
    else:
        l_buf_max = 16
        buf_cols_per_split = 16
        bk = q
        bv = q
        bi = selected_i8

    _ta_v10_cluster_attn_kernel[(int(q.shape[0]), int(num_splits))](
        q,
        order_i64,
        depth_i32,
        selected_i8,
        assigns_i32,
        cluster_ids_i32,
        cluster_keys_t_f16,
        cluster_values_f16,
        threshold_f32,
        bk,
        bv,
        bi,
        out_m,
        out_l,
        out_o,
        int(k_clusters),
        int(n_pad),
        int(h_kv) * int(n_pad),
        SCALE_LOG2E=float(scale_log2e),
        D=d,
        D_V=d_v,
        H_KV=int(h_kv),
        GROUPS=int(groups),
        BLOCK_ROWS=int(block_rows),
        BF=4,
        CANDS=int(block_rows) * 4,
        NUM_SPLITS=int(num_splits),
        HAS_BUFFER=bool(has_buffer),
        L_BUF_MAX=int(l_buf_max),
        BUF_COLS_PER_SPLIT=int(buf_cols_per_split),
        num_warps=num_warps,
        num_stages=num_stages,
    )
