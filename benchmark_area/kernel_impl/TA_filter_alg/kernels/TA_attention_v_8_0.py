"""TA_attention_v8.0 — sparse cluster-iteration attention.

Pipeline:

  centroid_scores  →  torch.sort  →  build_selected
                                 →  build_cand (fused: cand_b + cluster_active)
                                 →  v8 sparse cluster attention (skips dead chunks)
                                 →  reduce

Key change vs v6.0: instead of walking every key in N_pad and inlining the
S-way OR-of-selected per chunk, we iterate over clusters in the s=0
partition (K parents, BF children each) and skip whole parent chunks via a
1-byte ``cluster_active[h_kv, c]`` flag.  When a chunk is live, the per-head
candidate mask is read with a single ``cand_b[h_q, c, child]`` int8 lookup
in cluster-aligned layout, then keys/values come from the cluster-contiguous
``cluster_keys_t_f16``/``cluster_values_f16`` layouts produced by
``TA_build_v_7_0``.  This recovers the v5.14 cluster-iteration speed while
preserving exact TA-filter recall.
"""

from __future__ import annotations

import math

import torch

from ._TA_common import _LOG2E, next_pow2
from ._TA_triton_centroid_scores import run_ta_centroid_scores
from ._TA_triton_reduce import run_ta_reduce
from ._TA_triton_selected import run_ta_build_selected
from ._TA_triton_v8 import HAS_TRITON, run_ta_v8_attn, run_ta_v8_build_cand
from .TA_attention_v_1_2 import attend as attend_v1_2
from .TA_attention_v_3_0 import _grouped_or_identity_groups, _prep_buffer_cached

KERNEL_VERSION = "v8.0"

_CENTER_BLOCK_K = 64
_PARENTS_PER_PROG = 8
_ATTN_SPLITS = 32
_ATTN_WARPS = 4
_ATTN_STAGES = 3
_BUILD_KCHUNK = 8


def _workspace(
    state: dict,
    *,
    h_q: int,
    s_sub: int,
    k_clusters: int,
    n_pad: int,
    d_v: int,
    bf: int,
    device,
) -> dict:
    cache = state.setdefault("_TA_attention_v8_0_cache", {})
    key = (device.index, h_q, s_sub, k_clusters, n_pad, d_v, bf, _ATTN_SPLITS)
    if cache.get("key") != key:
        cache.clear()
        cache["key"] = key
        cache["scores"] = torch.empty(h_q, s_sub, k_clusters, device=device, dtype=torch.float16)
        cache["selected"] = torch.empty(h_q, s_sub, k_clusters, device=device, dtype=torch.int8)
        cache["cand_b"] = torch.empty(h_q, k_clusters, bf, device=device, dtype=torch.int8)
        cache["cluster_active"] = torch.empty(
            int(state["keys_padded_f16"].shape[0]), k_clusters, device=device, dtype=torch.int8
        )
        cache["assigns_i32"] = state["assigns_padded"].to(torch.int32).contiguous()
        cache["children_s0_i32"] = (
            state["children_padded_i32"][0].to(torch.int32).contiguous()
        )
        cache["invalid_i8"] = state["invalid_mask"].to(torch.int8).contiguous()
        cache["cluster_keys_t_s0"] = state["cluster_keys_t_f16"][0].contiguous()
        cache["cluster_values_s0"] = state["cluster_values_f16"][0].contiguous()
        cache["out_m"] = torch.empty(h_q, _ATTN_SPLITS, device=device, dtype=torch.float32)
        cache["out_l"] = torch.empty(h_q, _ATTN_SPLITS, device=device, dtype=torch.float32)
        cache["out_o"] = torch.empty(h_q, _ATTN_SPLITS, d_v, device=device, dtype=torch.float32)
        cache["out"] = torch.empty(h_q, d_v, device=device, dtype=torch.float32)
        cache["buf_key"] = None
        cache["buf_prepped"] = (None, None, None)
    return cache


def attend(
    q: torch.Tensor,
    threshold: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
    keys_children: torch.Tensor | None = None,
) -> torch.Tensor:
    del keys_children
    if not HAS_TRITON:
        raise RuntimeError("TA_attention_v8.0 requires Triton")
    if (
        "children_padded_i32" not in state
        or "cluster_keys_t_f16" not in state
        or "cluster_values_f16" not in state
    ):
        return attend_v1_2(q, threshold, state, buffer_keys, buffer_values, q_head_to_kv, scale)

    h_q, d = q.shape
    h_kv = int(state["keys_padded_f16"].shape[0])
    d_v = int(state["values_padded_f16"].shape[-1])
    s_sub = int(state["n_subspaces"])
    k_clusters = int(state["K"])
    n_pad = int(state["N_pad"])
    bf = int(state["bf"])
    groups = _grouped_or_identity_groups(q_head_to_kv, h_q, h_kv)
    if (
        groups is None
        or groups > 8
        or d != 128
        or d_v != 128
        or s_sub != 8
        or bf != 4
        or int(state["max_width"]) > 32
    ):
        return attend_v1_2(q, threshold, state, buffer_keys, buffer_values, q_head_to_kv, scale)

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale_log2e = float(scale) * _LOG2E

    q_f16 = q if q.dtype == torch.float16 and q.is_contiguous() else q.to(torch.float16).contiguous()
    threshold_f32 = (
        threshold
        if threshold.dtype == torch.float32 and threshold.is_contiguous()
        else threshold.float().contiguous()
    )

    ws = _workspace(
        state,
        h_q=h_q,
        s_sub=s_sub,
        k_clusters=k_clusters,
        n_pad=n_pad,
        d_v=d_v,
        bf=bf,
        device=q.device,
    )

    run_ta_centroid_scores(
        q_f16,
        state["centers_padded_f16"],
        state["dim_offsets"],
        state["dim_widths"],
        ws["scores"],
        groups=groups,
        block_k=_CENTER_BLOCK_K,
    )
    sorted_scores, order = torch.sort(ws["scores"], dim=-1, descending=True)

    run_ta_build_selected(
        order,
        sorted_scores,
        threshold_f32,
        ws["selected"],
    )

    run_ta_v8_build_cand(
        ws["selected"],
        ws["assigns_i32"],
        ws["children_s0_i32"],
        ws["invalid_i8"],
        ws["cand_b"],
        ws["cluster_active"],
        groups=groups,
        kchunk=_BUILD_KCHUNK,
    )

    bk_t, bv, b_inv = _prep_buffer_cached(ws, buffer_keys, buffer_values)
    groups_pow = max(next_pow2(groups), 4)
    run_ta_v8_attn(
        q=q_f16,
        cluster_keys_t_s0_f16=ws["cluster_keys_t_s0"],
        cluster_values_s0_f16=ws["cluster_values_s0"],
        cand_b_i8=ws["cand_b"],
        cluster_active_i8=ws["cluster_active"],
        threshold_f32=threshold_f32,
        buf_keys_t_f16=bk_t,
        buf_values_f16=bv,
        buf_invalid_i8=b_inv,
        h_kv_eff=h_kv,
        k_clusters=k_clusters,
        scale_log2e=scale_log2e,
        groups=groups,
        groups_pow=groups_pow,
        parents_per_prog=_PARENTS_PER_PROG,
        num_splits=_ATTN_SPLITS,
        out_m=ws["out_m"],
        out_l=ws["out_l"],
        out_o=ws["out_o"],
        num_warps=_ATTN_WARPS,
        num_stages=_ATTN_STAGES,
    )
    run_ta_reduce(ws["out_m"], ws["out_l"], ws["out_o"], ws["out"])
    return ws["out"]


KERNEL = attend
