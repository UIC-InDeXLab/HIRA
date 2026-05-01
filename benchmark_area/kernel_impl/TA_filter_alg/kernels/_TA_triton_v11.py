"""v11 Triton kernels for S=4, bf=4 cluster-streaming attention.

v11 drops the second-pass per-key threshold filter used by v10.  Once a
cluster is selected (either by sort+depth or by per-subspace top-l), every
key belonging to that cluster contributes to softmax.  Reports show the
selected cluster set is small relative to N, so attention runs over a tiny
fraction of keys; softmax assigns near-zero weight to low-scoring keys, so
output remains close to the threshold-filtered baseline.

Two selectors share a single attention kernel:

  - sort   : reuse v10's order / depth_and_selected helpers, no threshold.
  - top-l  : compute topk indices per (h, s) in torch, then build selected
             with `_ta_v11_build_selected_from_topl_kernel`; depth = L.
"""

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
    def _ta_v11_build_selected_from_topl_kernel(
        Indices_ptr,    # (H_q, 4, L) int64/int32 — top-L cluster ids per (h, s)
        Selected_ptr,   # (H_q, 4, K) int8
        L,
        K_CLUSTERS,
        L_BLOCK: tl.constexpr,
        K_BLOCK: tl.constexpr,
    ):
        hq = tl.program_id(0)
        s_idx = tl.program_id(1)

        k_offs = tl.arange(0, K_BLOCK)
        k_valid = k_offs < K_CLUSTERS
        base_k = (hq * 4 + s_idx) * K_CLUSTERS
        tl.store(
            Selected_ptr + base_k + k_offs,
            tl.zeros([K_BLOCK], dtype=tl.int8),
            mask=k_valid,
        )

        l_offs = tl.arange(0, L_BLOCK)
        l_valid = l_offs < L
        base_l = (hq * 4 + s_idx) * L
        indices = tl.load(
            Indices_ptr + base_l + l_offs,
            mask=l_valid,
            other=0,
        ).to(tl.int32)
        c_valid = l_valid & (indices >= 0) & (indices < K_CLUSTERS)
        tl.store(
            Selected_ptr + base_k + indices,
            tl.full([L_BLOCK], 1, dtype=tl.int8),
            mask=c_valid,
        )

    @triton.jit
    def _ta_v11_cluster_attn_kernel(
        Q_ptr,               # (H_q, D) fp16
        Order_ptr,           # (H_q, 4, ROW_STRIDE) int64/int32 (sort: K, topl: L)
        Depth_ptr,           # (H_q,) int32 — sort: depth, topl: L
        Selected_ptr,        # (H_q, 4, K) int8
        Assigns_ptr,         # (4, H_kv, N_pad) int32
        ClusterIds_ptr,      # (4, H_kv, K, 4) int32
        ClusterKeysT_ptr,    # (4, H_kv, K, D, 4) fp16
        ClusterValues_ptr,   # (4, H_kv, K, 4, D_v) fp16
        BufKeysT_ptr,        # (H_kv, D, L_buf) fp16
        BufValues_ptr,       # (H_kv, L_buf, D_v) fp16
        BufInvalid_ptr,      # (H_kv, L_buf) int8
        M_out_ptr,           # (H_q, NUM_SPLITS) fp32
        L_out_ptr,           # (H_q, NUM_SPLITS) fp32
        O_out_ptr,           # (H_q, NUM_SPLITS, D_v) fp32
        K_CLUSTERS,
        N_PAD,
        ASSIGN_STRIDE_S,
        ROW_STRIDE,          # K (sort) or L (topl)
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

        q_vals = tl.load(Q_ptr + hq * D + d_range)
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
                    Order_ptr + (hq * 4 + s_idx) * ROW_STRIDE + rows_safe,
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
                    # No threshold filter — every key in selected clusters
                    # contributes; softmax suppresses low-score noise.
                    scaled = tl.where(owned, raw * SCALE_LOG2E, -1.0e30)

                    chunk_max = tl.max(scaled, axis=0)
                    m_new = tl.maximum(m, chunk_max)
                    alpha = tl.exp2(m - m_new)
                    pvals = tl.exp2(scaled - m_new)
                    pvals = tl.where(owned, pvals, 0.0)
                    l_acc = alpha * l_acc + tl.sum(pvals, axis=0)

                    v_tile = tl.load(
                        ClusterValues_ptr
                        + ((((s_idx * H_KV + kvh) * K_CLUSTERS + cluster_flat[:, None]) * BF
                            + child_flat[:, None]) * D_V)
                        + dv_range[None, :],
                        mask=owned[:, None],
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


def run_ta_v11_build_selected_from_topl(
    indices: torch.Tensor,        # (H_q, 4, L) int64/int32
    selected_i8: torch.Tensor,    # (H_q, 4, K) int8
    *,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, s_sub, l = indices.shape
    k_clusters = int(selected_i8.shape[-1])
    if int(s_sub) != 4:
        raise ValueError(f"v11 topl selected requires S=4, got {s_sub}")
    _ta_v11_build_selected_from_topl_kernel[(int(h_q), 4)](
        indices,
        selected_i8,
        int(l),
        int(k_clusters),
        L_BLOCK=_next_pow2(int(l)),
        K_BLOCK=_next_pow2(int(k_clusters)),
        num_warps=num_warps,
        num_stages=num_stages,
    )


if HAS_TRITON:

    @triton.jit
    def _ta_v11_cluster_attn_fp16_kernel(
        Q_ptr,               # (H_q, D) fp16
        Order_ptr,           # (H_q, 4, ROW_STRIDE) int64/int32
        Depth_ptr,           # (H_q,) int32
        Selected_ptr,        # (H_q, 4, K) int8
        Assigns_ptr,         # (4, H_kv, N_pad) int32
        ClusterIds_ptr,      # (4, H_kv, K, 4) int32
        ClusterKeysT_ptr,    # (4, H_kv, K, D, 4) fp16
        ClusterValues_ptr,   # (4, H_kv, K, 4, D_v) fp16
        BufKeysT_ptr,        # (H_kv, D, L_buf) fp16
        BufValues_ptr,       # (H_kv, L_buf, D_v) fp16
        BufInvalid_ptr,      # (H_kv, L_buf) int8
        M_out_ptr,           # (H_q, NUM_SPLITS) fp16
        L_out_ptr,           # (H_q, NUM_SPLITS) fp16
        O_out_ptr,           # (H_q, NUM_SPLITS, D_v) fp16
        K_CLUSTERS,
        N_PAD,
        ASSIGN_STRIDE_S,
        ROW_STRIDE,
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

        q_vals = tl.load(Q_ptr + hq * D + d_range)
        depth = tl.load(Depth_ptr + hq).to(tl.int32)

        m = tl.full((), -65000.0, dtype=tl.float16)
        l_acc = tl.full((), 0.0, dtype=tl.float16)
        o_acc = tl.zeros([D_V], dtype=tl.float16)
        scale_log2e_f16 = tl.full((), SCALE_LOG2E, dtype=tl.float16)
        neg_inf_f16 = tl.full((), -65000.0, dtype=tl.float16)

        rows_per_split = (depth + NUM_SPLITS - 1) // NUM_SPLITS
        r_start = split * rows_per_split
        r_stop = tl.minimum(r_start + rows_per_split, depth)

        for r_chunk in range(r_start, r_stop, BLOCK_ROWS):
            rows = r_chunk + row_inner
            row_valid = rows < r_stop
            rows_safe = tl.where(row_valid, rows, 0)

            for s_idx in tl.static_range(4):
                clusters = tl.load(
                    Order_ptr + (hq * 4 + s_idx) * ROW_STRIDE + rows_safe,
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
                    raw = tl.dot(q_vals[None, :], keys_tile, out_dtype=tl.float16)
                    raw = tl.reshape(raw, [CANDS])
                    scaled = tl.where(owned, raw * scale_log2e_f16, neg_inf_f16)

                    chunk_max = tl.max(scaled, axis=0).to(tl.float16)
                    m_new = tl.maximum(m, chunk_max).to(tl.float16)
                    alpha = tl.exp2((m - m_new).to(tl.float32)).to(tl.float16)
                    pvals = tl.exp2((scaled - m_new).to(tl.float32)).to(tl.float16)
                    pvals = tl.where(owned, pvals, tl.full([CANDS], 0.0, dtype=tl.float16))
                    l_acc = alpha * l_acc + tl.sum(pvals, axis=0)

                    v_tile = tl.load(
                        ClusterValues_ptr
                        + ((((s_idx * H_KV + kvh) * K_CLUSTERS + cluster_flat[:, None]) * BF
                            + child_flat[:, None]) * D_V)
                        + dv_range[None, :],
                        mask=owned[:, None],
                        other=0.0,
                    )
                    o_acc = alpha * o_acc + tl.reshape(
                        tl.dot(pvals[None, :], v_tile, out_dtype=tl.float16),
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
                buf_raw = tl.dot(q_vals[None, :], buf_keys, out_dtype=tl.float16)
                buf_raw = tl.reshape(buf_raw, [BUF_COLS_PER_SPLIT])
                buf_scaled = tl.where(buf_live, buf_raw * scale_log2e_f16, neg_inf_f16)

                buf_max = tl.max(buf_scaled, axis=0).to(tl.float16)
                m_new = tl.maximum(m, buf_max).to(tl.float16)
                alpha = tl.exp2((m - m_new).to(tl.float32)).to(tl.float16)
                bp = tl.exp2((buf_scaled - m_new).to(tl.float32)).to(tl.float16)
                bp = tl.where(buf_live, bp, tl.full([BUF_COLS_PER_SPLIT], 0.0, dtype=tl.float16))
                l_acc = alpha * l_acc + tl.sum(bp, axis=0)

                buf_values = tl.load(
                    BufValues_ptr
                    + (kvh * L_BUF_MAX + buf_safe[:, None]) * D_V
                    + dv_range[None, :],
                    mask=buf_live[:, None],
                    other=0.0,
                )
                o_acc = alpha * o_acc + tl.reshape(
                    tl.dot(bp[None, :], buf_values, out_dtype=tl.float16),
                    [D_V],
                )
                m = m_new

        tl.store(M_out_ptr + hq * NUM_SPLITS + split, m)
        tl.store(L_out_ptr + hq * NUM_SPLITS + split, l_acc)
        tl.store(O_out_ptr + (hq * NUM_SPLITS + split) * D_V + dv_range, o_acc)


    @triton.jit
    def _ta_v11_reduce_fp16_kernel(
        M_ptr,            # (H_q, NUM_SPLITS) fp16
        L_ptr,            # (H_q, NUM_SPLITS) fp16
        O_ptr,            # (H_q, NUM_SPLITS, D_V) fp16
        Out_ptr,          # (H_q, D_V) fp16
        NUM_SPLITS: tl.constexpr,
        D_V: tl.constexpr,
        SPLITS_POW: tl.constexpr,
    ):
        hq = tl.program_id(0)
        s_range = tl.arange(0, SPLITS_POW)
        s_valid = s_range < NUM_SPLITS
        dv = tl.arange(0, D_V)

        m = tl.load(M_ptr + hq * NUM_SPLITS + s_range, mask=s_valid, other=-65000.0)
        l_ = tl.load(L_ptr + hq * NUM_SPLITS + s_range, mask=s_valid, other=0.0)
        m_global = tl.max(m, axis=0).to(tl.float16)
        alpha = tl.exp2((m - m_global).to(tl.float32)).to(tl.float16)
        l_sum = tl.sum(alpha * l_, axis=0)

        o = tl.load(
            O_ptr + (hq * NUM_SPLITS + s_range[:, None]) * D_V + dv[None, :],
            mask=s_valid[:, None],
            other=0.0,
        )
        o_sum = tl.sum(alpha[:, None] * o, axis=0)

        l_safe = tl.where(l_sum > tl.full((), 0.0, dtype=tl.float16), l_sum, tl.full((), 1.0, dtype=tl.float16))
        out = o_sum / l_safe
        tl.store(Out_ptr + hq * D_V + dv, out)


if HAS_TRITON:

    @triton.jit
    def _ta_v11_3_cluster_attn_kernel(
        Q_ptr,               # (H_q, D) fp16
        Order_ptr,           # (H_q, 4, ROW_STRIDE) int64
        Depth_ptr,           # (H_q,) int32
        ClusterIds_ptr,      # (4, H_kv, K, 4) int32
        ClusterKeysT_ptr,    # (4, H_kv, K, D, 4) fp16
        ClusterValues_ptr,   # (4, H_kv, K, 4, D_v) fp16
        BufKeysT_ptr,
        BufValues_ptr,
        BufInvalid_ptr,
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        K_CLUSTERS,
        N_PAD,
        ROW_STRIDE,
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
        # No-dedup variant: every selected (s, cluster) contributes its BF keys
        # without checking earlier-subspace ownership.  Duplicates across
        # subspaces are scored multiple times; user accepts the error.
        hq = tl.program_id(0)
        split = tl.program_id(1)
        kvh = hq // GROUPS

        d_range = tl.arange(0, D)
        dv_range = tl.arange(0, D_V)
        row_inner = tl.arange(0, BLOCK_ROWS)
        child_inner = tl.arange(0, BF)

        q_vals = tl.load(Q_ptr + hq * D + d_range)
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
                    Order_ptr + (hq * 4 + s_idx) * ROW_STRIDE + rows_safe,
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
                owned = tl.reshape(row_valid_2d, [CANDS])

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
                    scaled = tl.where(owned, raw * SCALE_LOG2E, -1.0e30)

                    chunk_max = tl.max(scaled, axis=0)
                    m_new = tl.maximum(m, chunk_max)
                    alpha = tl.exp2(m - m_new)
                    pvals = tl.exp2(scaled - m_new)
                    pvals = tl.where(owned, pvals, 0.0)
                    l_acc = alpha * l_acc + tl.sum(pvals, axis=0)

                    v_tile = tl.load(
                        ClusterValues_ptr
                        + ((((s_idx * H_KV + kvh) * K_CLUSTERS + cluster_flat[:, None]) * BF
                            + child_flat[:, None]) * D_V)
                        + dv_range[None, :],
                        mask=owned[:, None],
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


    @triton.jit
    def _ta_v11_4_cand_mask_kernel(
        Selected_ptr,    # (H_q, 4, K) int8
        Assigns_ptr,     # (4, H_kv, N_pad) int32
        Mask_ptr,        # (H_q, N_pad) int8 — 1 if any of 4 parents selected
        N_PAD,
        K_CLUSTERS,
        ASSIGN_STRIDE_S,
        H_KV: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        hq = tl.program_id(0)
        n_block = tl.program_id(1)
        kvh = hq // GROUPS

        offs = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = offs < N_PAD

        acc = tl.zeros([BLOCK_N], dtype=tl.int8)
        for s_idx in tl.static_range(4):
            parents = tl.load(
                Assigns_ptr + s_idx * ASSIGN_STRIDE_S + kvh * N_PAD + offs,
                mask=valid,
                other=0,
            ).to(tl.int32)
            p_valid = valid & (parents >= 0) & (parents < K_CLUSTERS)
            p_safe = tl.where(p_valid, parents, 0)
            sel = tl.load(
                Selected_ptr + (hq * 4 + s_idx) * K_CLUSTERS + p_safe,
                mask=p_valid,
                other=0,
            )
            acc = acc | sel

        tl.store(Mask_ptr + hq * N_PAD + offs, acc, mask=valid)


    @triton.jit
    def _ta_v11_4_flat_attn_kernel(
        Q_ptr,               # (H_q, D) fp16
        FlatIds_ptr,         # (H_q, M_MAX) int32 — packed candidate key ids; -1 = invalid
        FlatCount_ptr,       # (H_q,) int32 — number of valid entries per head
        KeysT_ptr,           # (H_kv, D, N_pad) fp16 — pre-transposed keys
        Values_ptr,          # (H_kv, N_pad, D_v) fp16
        BufKeysT_ptr,
        BufValues_ptr,
        BufInvalid_ptr,
        M_out_ptr,           # (H_q, NUM_SPLITS) fp32
        L_out_ptr,           # (H_q, NUM_SPLITS) fp32
        O_out_ptr,           # (H_q, NUM_SPLITS, D_v) fp32
        N_PAD,
        M_MAX,
        SCALE_LOG2E: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        H_KV: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
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
        m_inner = tl.arange(0, BLOCK_M)

        q_vals = tl.load(Q_ptr + hq * D + d_range)
        cnt = tl.load(FlatCount_ptr + hq).to(tl.int32)

        m = tl.full((), -1.0e30, dtype=tl.float32)
        l_acc = tl.full((), 0.0, dtype=tl.float32)
        o_acc = tl.zeros([D_V], dtype=tl.float32)

        rows_per_split = (cnt + NUM_SPLITS - 1) // NUM_SPLITS
        r_start = split * rows_per_split
        r_stop = tl.minimum(r_start + rows_per_split, cnt)

        for r_chunk in range(r_start, r_stop, BLOCK_M):
            rows = r_chunk + m_inner
            row_valid = rows < r_stop
            rows_safe = tl.where(row_valid, rows, 0)

            key_ids = tl.load(
                FlatIds_ptr + hq * M_MAX + rows_safe,
                mask=row_valid,
                other=-1,
            ).to(tl.int32)
            owned = row_valid & (key_ids >= 0) & (key_ids < N_PAD)
            ids_safe = tl.where(owned, key_ids, 0)

            keys_tile = tl.load(
                KeysT_ptr + (kvh * D + d_range[:, None]) * N_PAD + ids_safe[None, :],
                mask=owned[None, :],
                other=0.0,
            )
            raw = tl.dot(q_vals[None, :], keys_tile)
            raw = tl.reshape(raw, [BLOCK_M]).to(tl.float32)
            scaled = tl.where(owned, raw * SCALE_LOG2E, -1.0e30)

            chunk_max = tl.max(scaled, axis=0)
            m_new = tl.maximum(m, chunk_max)
            alpha = tl.exp2(m - m_new)
            pvals = tl.exp2(scaled - m_new)
            pvals = tl.where(owned, pvals, 0.0)
            l_acc = alpha * l_acc + tl.sum(pvals, axis=0)

            v_tile = tl.load(
                Values_ptr + (kvh * N_PAD + ids_safe[:, None]) * D_V + dv_range[None, :],
                mask=owned[:, None],
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


if HAS_TRITON:

    @triton.jit
    def _ta_v11_5_merged_cluster_attn_kernel(
        Q_ptr,
        Order_ptr,           # (H_q, 4, K) int64 — sort order per subspace
        Depth_ptr,           # (H_q,) int32
        ClusterIds_ptr,      # (4, H_kv, K, 4) int32
        ClusterKeysT_ptr,    # (4, H_kv, K, D, 4) fp16
        ClusterValues_ptr,   # (4, H_kv, K, 4, D_v) fp16
        BufKeysT_ptr,
        BufValues_ptr,
        BufInvalid_ptr,
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        K_CLUSTERS,
        N_PAD,
        SCALE_LOG2E: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        H_KV: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BF: tl.constexpr,
        SUBS: tl.constexpr,
        CANDS: tl.constexpr,             # = BLOCK_ROWS * SUBS * BF
        NUM_SPLITS: tl.constexpr,
        HAS_BUFFER: tl.constexpr,
        L_BUF_MAX: tl.constexpr,
        BUF_COLS_PER_SPLIT: tl.constexpr,
    ):
        # No-dedup, merged subspace variant.  For each row chunk we gather
        # SUBS*BLOCK_ROWS clusters across the 4 subspaces and run ONE matmul
        # of shape (1, D) x (D, CANDS).  Amortises the per-program fixed
        # cost (q load, m/l/o init, dot fixed overhead) over 4× more keys
        # than v11.3.  Same correctness profile as v11.3 (duplicate keys
        # across subspaces are scored multiple times — user accepts).
        hq = tl.program_id(0)
        split = tl.program_id(1)
        kvh = hq // GROUPS

        d_range = tl.arange(0, D)
        dv_range = tl.arange(0, D_V)
        row_inner = tl.arange(0, BLOCK_ROWS)
        sub_inner = tl.arange(0, SUBS)
        child_inner = tl.arange(0, BF)

        q_vals = tl.load(Q_ptr + hq * D + d_range)
        depth = tl.load(Depth_ptr + hq).to(tl.int32)

        m = tl.full((), -1.0e30, dtype=tl.float32)
        l_acc = tl.full((), 0.0, dtype=tl.float32)
        o_acc = tl.zeros([D_V], dtype=tl.float32)

        rows_per_split = (depth + NUM_SPLITS - 1) // NUM_SPLITS
        r_start = split * rows_per_split
        r_stop = tl.minimum(r_start + rows_per_split, depth)

        for r_chunk in range(r_start, r_stop, BLOCK_ROWS):
            rows = r_chunk + row_inner
            row_valid = rows < r_stop                                          # (BR,)
            rows_safe = tl.where(row_valid, rows, 0)

            # Gather clusters for all 4 subspaces simultaneously.
            # cluster_2d_rs : (BR, SUBS) — cluster id per (row, subspace)
            cluster_2d_rs = tl.load(
                Order_ptr
                + (hq * SUBS + sub_inner[None, :]) * K_CLUSTERS
                + rows_safe[:, None],
                mask=row_valid[:, None],
                other=0,
            ).to(tl.int32)
            cluster_valid_rs = (
                row_valid[:, None]
                & (cluster_2d_rs >= 0)
                & (cluster_2d_rs < K_CLUSTERS)
            )
            clusters_safe_rs = tl.where(cluster_valid_rs, cluster_2d_rs, 0)

            # Gather BF child key ids per (row, subspace).  Layout:
            # cands_3d[row, sub, child] -> (BR, SUBS, BF), then flatten.
            cands_3d = tl.load(
                ClusterIds_ptr
                + (((sub_inner[None, :, None] * H_KV + kvh) * K_CLUSTERS
                    + clusters_safe_rs[:, :, None]) * BF)
                + child_inner[None, None, :],
                mask=cluster_valid_rs[:, :, None],
                other=-1,
            ).to(tl.int32)
            cluster_3d = clusters_safe_rs[:, :, None] + tl.zeros(
                [BLOCK_ROWS, SUBS, BF], dtype=tl.int32
            )
            sub_3d = sub_inner[None, :, None] + tl.zeros(
                [BLOCK_ROWS, SUBS, BF], dtype=tl.int32
            )
            owned_3d = cluster_valid_rs[:, :, None] & (cands_3d >= 0) & (cands_3d < N_PAD)

            cluster_flat = tl.reshape(cluster_3d, [CANDS])
            sub_flat = tl.reshape(sub_3d, [CANDS])
            child_flat = tl.reshape(
                child_inner[None, None, :] + tl.zeros([BLOCK_ROWS, SUBS, BF], dtype=tl.int32),
                [CANDS],
            )
            owned = tl.reshape(owned_3d, [CANDS])

            if tl.max(owned.to(tl.int32), axis=0) != 0:
                # Single big keys tile: (D, CANDS).  Fully amortised matmul.
                keys_tile = tl.load(
                    ClusterKeysT_ptr
                    + ((((sub_flat[None, :] * H_KV + kvh) * K_CLUSTERS
                         + cluster_flat[None, :]) * D
                        + d_range[:, None]) * BF)
                    + child_flat[None, :],
                    mask=owned[None, :],
                    other=0.0,
                )
                raw = tl.dot(q_vals[None, :], keys_tile)
                raw = tl.reshape(raw, [CANDS]).to(tl.float32)
                scaled = tl.where(owned, raw * SCALE_LOG2E, -1.0e30)

                chunk_max = tl.max(scaled, axis=0)
                m_new = tl.maximum(m, chunk_max)
                alpha = tl.exp2(m - m_new)
                pvals = tl.exp2(scaled - m_new)
                pvals = tl.where(owned, pvals, 0.0)
                l_acc = alpha * l_acc + tl.sum(pvals, axis=0)

                v_tile = tl.load(
                    ClusterValues_ptr
                    + ((((sub_flat[:, None] * H_KV + kvh) * K_CLUSTERS
                         + cluster_flat[:, None]) * BF
                        + child_flat[:, None]) * D_V)
                    + dv_range[None, :],
                    mask=owned[:, None],
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


def run_ta_v11_5_merged_cluster_attn(
    q: torch.Tensor,
    order_i64: torch.Tensor,
    depth_i32: torch.Tensor,
    cluster_ids_i32: torch.Tensor,
    cluster_keys_t_f16: torch.Tensor,
    cluster_values_f16: torch.Tensor,
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
        raise ValueError(f"v11.5 requires S=4, bf=4; got S={s_sub}, bf={bf}")

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
        bi = depth_i32

    _ta_v11_5_merged_cluster_attn_kernel[(int(q.shape[0]), int(num_splits))](
        q,
        order_i64,
        depth_i32,
        cluster_ids_i32,
        cluster_keys_t_f16,
        cluster_values_f16,
        bk,
        bv,
        bi,
        out_m,
        out_l,
        out_o,
        int(k_clusters),
        int(n_pad),
        SCALE_LOG2E=float(scale_log2e),
        D=d,
        D_V=d_v,
        H_KV=int(h_kv),
        GROUPS=int(groups),
        BLOCK_ROWS=int(block_rows),
        BF=4,
        SUBS=4,
        CANDS=int(block_rows) * 4 * 4,
        NUM_SPLITS=int(num_splits),
        HAS_BUFFER=bool(has_buffer),
        L_BUF_MAX=int(l_buf_max),
        BUF_COLS_PER_SPLIT=int(buf_cols_per_split),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v11_3_cluster_attn(
    q: torch.Tensor,
    order_i64: torch.Tensor,
    depth_i32: torch.Tensor,
    cluster_ids_i32: torch.Tensor,
    cluster_keys_t_f16: torch.Tensor,
    cluster_values_f16: torch.Tensor,
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
        raise ValueError(f"v11.3 requires S=4, bf=4; got S={s_sub}, bf={bf}")
    row_stride = int(order_i64.shape[-1])

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
        bi = depth_i32

    _ta_v11_3_cluster_attn_kernel[(int(q.shape[0]), int(num_splits))](
        q,
        order_i64,
        depth_i32,
        cluster_ids_i32,
        cluster_keys_t_f16,
        cluster_values_f16,
        bk,
        bv,
        bi,
        out_m,
        out_l,
        out_o,
        int(k_clusters),
        int(n_pad),
        int(row_stride),
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


if HAS_TRITON:

    @triton.jit
    def _ta_v11_4_count_kernel(
        Mask_ptr,        # (H_q, N_pad) int8
        Count_ptr,       # (H_q,) int32
        N_PAD,
        BLOCK_N: tl.constexpr,
    ):
        hq = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        valid = offs < N_PAD
        m = tl.load(Mask_ptr + hq * N_PAD + offs, mask=valid, other=0)
        total = tl.sum(m.to(tl.int32), axis=0)
        tl.store(Count_ptr + hq, total)


def run_ta_v11_4_count(
    mask_i8: torch.Tensor,
    count_i32: torch.Tensor,
) -> None:
    h_q, n_pad = mask_i8.shape
    block_n = max(_next_pow2(int(n_pad)), 16)
    _ta_v11_4_count_kernel[(int(h_q),)](
        mask_i8, count_i32, int(n_pad),
        BLOCK_N=int(block_n),
        num_warps=4,
        num_stages=2,
    )


def run_ta_v11_4_cand_mask(
    selected_i8: torch.Tensor,
    assigns_i32: torch.Tensor,
    mask_out_i8: torch.Tensor,
    *,
    groups: int,
    block_n: int = 256,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    h_q, n_pad = mask_out_i8.shape
    s_sub, h_kv, _ = assigns_i32.shape
    k_clusters = int(selected_i8.shape[-1])
    if int(s_sub) != 4:
        raise ValueError(f"v11.4 cand mask requires S=4, got {s_sub}")
    grid = (int(h_q), (int(n_pad) + int(block_n) - 1) // int(block_n))
    _ta_v11_4_cand_mask_kernel[grid](
        selected_i8,
        assigns_i32,
        mask_out_i8,
        int(n_pad),
        int(k_clusters),
        int(h_kv) * int(n_pad),
        H_KV=int(h_kv),
        GROUPS=int(groups),
        BLOCK_N=int(block_n),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v11_4_flat_attn(
    q: torch.Tensor,
    flat_ids_i32: torch.Tensor,
    flat_count_i32: torch.Tensor,
    keys_t_f16: torch.Tensor,
    values_f16: torch.Tensor,
    buf_keys_t_f16: torch.Tensor | None,
    buf_values_f16: torch.Tensor | None,
    buf_invalid_i8: torch.Tensor | None,
    *,
    n_pad: int,
    scale_log2e: float,
    groups: int,
    block_m: int,
    num_splits: int,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    d = int(q.shape[-1])
    d_v = int(values_f16.shape[-1])
    h_kv = int(keys_t_f16.shape[0])
    m_max = int(flat_ids_i32.shape[-1])

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
        bi = flat_count_i32

    _ta_v11_4_flat_attn_kernel[(int(q.shape[0]), int(num_splits))](
        q,
        flat_ids_i32,
        flat_count_i32,
        keys_t_f16,
        values_f16,
        bk,
        bv,
        bi,
        out_m,
        out_l,
        out_o,
        int(n_pad),
        int(m_max),
        SCALE_LOG2E=float(scale_log2e),
        D=d,
        D_V=d_v,
        H_KV=int(h_kv),
        GROUPS=int(groups),
        BLOCK_M=int(block_m),
        NUM_SPLITS=int(num_splits),
        HAS_BUFFER=bool(has_buffer),
        L_BUF_MAX=int(l_buf_max),
        BUF_COLS_PER_SPLIT=int(buf_cols_per_split),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_ta_v11_cluster_attn_fp16(
    q: torch.Tensor,
    order_i64: torch.Tensor,
    depth_i32: torch.Tensor,
    selected_i8: torch.Tensor,
    assigns_i32: torch.Tensor,
    cluster_ids_i32: torch.Tensor,
    cluster_keys_t_f16: torch.Tensor,
    cluster_values_f16: torch.Tensor,
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
        raise ValueError(f"v11 fp16 cluster attention requires S=4, bf=4; got S={s_sub}, bf={bf}")
    row_stride = int(order_i64.shape[-1])

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

    _ta_v11_cluster_attn_fp16_kernel[(int(q.shape[0]), int(num_splits))](
        q,
        order_i64,
        depth_i32,
        selected_i8,
        assigns_i32,
        cluster_ids_i32,
        cluster_keys_t_f16,
        cluster_values_f16,
        bk,
        bv,
        bi,
        out_m,
        out_l,
        out_o,
        int(k_clusters),
        int(n_pad),
        int(h_kv) * int(n_pad),
        int(row_stride),
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


def run_ta_v11_reduce_fp16(
    m_idx: torch.Tensor,
    l_idx: torch.Tensor,
    o_idx: torch.Tensor,
    out: torch.Tensor,
) -> None:
    h_q, num_splits = m_idx.shape
    d_v = int(o_idx.shape[-1])
    splits_pow = max(_next_pow2(num_splits), 1)
    _ta_v11_reduce_fp16_kernel[(h_q,)](
        m_idx, l_idx, o_idx, out,
        NUM_SPLITS=num_splits,
        D_V=d_v,
        SPLITS_POW=splits_pow,
    )


def run_ta_v11_cluster_attn(
    q: torch.Tensor,
    order_i64: torch.Tensor,           # (H_q, 4, ROW_STRIDE)
    depth_i32: torch.Tensor,
    selected_i8: torch.Tensor,
    assigns_i32: torch.Tensor,
    cluster_ids_i32: torch.Tensor,
    cluster_keys_t_f16: torch.Tensor,
    cluster_values_f16: torch.Tensor,
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
        raise ValueError(f"v11 cluster attention requires S=4, bf=4; got S={s_sub}, bf={bf}")
    row_stride = int(order_i64.shape[-1])

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

    _ta_v11_cluster_attn_kernel[(int(q.shape[0]), int(num_splits))](
        q,
        order_i64,
        depth_i32,
        selected_i8,
        assigns_i32,
        cluster_ids_i32,
        cluster_keys_t_f16,
        cluster_values_f16,
        bk,
        bv,
        bi,
        out_m,
        out_l,
        out_o,
        int(k_clusters),
        int(n_pad),
        int(h_kv) * int(n_pad),
        int(row_stride),
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
