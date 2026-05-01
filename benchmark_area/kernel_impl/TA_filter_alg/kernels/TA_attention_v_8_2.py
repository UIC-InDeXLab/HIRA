"""TA_attention_v8.2 — precomputed cand_b + cand_kvh, v6.0-pattern attention.

Pipeline:

  centroid_scores  →  torch.sort  →  build_selected
                                 →  build_cand (cand_b + cand_kvh)
                                 →  v8.2 split-N attention with cheap chunk skip
                                 →  reduce

At the bench's depth profile, both cluster-aligned skip (v8.0) and per-kvh
compact gather (v8.1) failed to beat v6.0 because the candidate density
is too high (~66%) for chunk-skip and per-kvh compact has its own cost.
v8.2 sticks with v6.0's split-N attention shape but moves the
``OR_s selected[hq, s, assigns[s, kvh, n]]`` logic out of the attention
inner loop into a parallel preprocessing kernel — saving ~2 KB of mask
traffic per chunk inside the hot inner loop, and using ``cand_kvh`` as a
single-byte chunk-skip predicate instead of v6.0's S=8 indexed lookups.
"""

from __future__ import annotations

import math

import torch

from ._TA_common import _LOG2E, next_pow2
from ._TA_triton_centroid_scores import run_ta_centroid_scores
from ._TA_triton_reduce import run_ta_reduce
from ._TA_triton_selected import run_ta_build_selected
from ._TA_triton_v8_2 import HAS_TRITON, run_ta_v8_2_attn, run_ta_v8_2_build_cand
from .TA_attention_v_1_2 import attend as attend_v1_2
from .TA_attention_v_3_0 import _grouped_or_identity_groups, _prep_buffer_cached

KERNEL_VERSION = "v8.2"

_CENTER_BLOCK_K = 64
_BLOCK_N = 32
_ATTN_SPLITS = 32
_ATTN_WARPS = 4
_ATTN_STAGES = 3
_BUILD_CHUNK_N = 64


def _workspace(
    state: dict,
    *,
    h_q: int,
    s_sub: int,
    k_clusters: int,
    n_pad: int,
    d_v: int,
    device,
) -> dict:
    cache = state.setdefault("_TA_attention_v8_2_cache", {})
    h_kv = int(state["keys_padded_f16"].shape[0])
    key = (device.index, h_q, s_sub, k_clusters, n_pad, d_v, _ATTN_SPLITS)
    if cache.get("key") != key:
        cache.clear()
        cache["key"] = key
        cache["scores"] = torch.empty(h_q, s_sub, k_clusters, device=device, dtype=torch.float16)
        cache["selected"] = torch.empty(h_q, s_sub, k_clusters, device=device, dtype=torch.int8)
        cache["cand_b"] = torch.empty(h_q, n_pad, device=device, dtype=torch.int8)
        cache["cand_kvh"] = torch.empty(h_kv, n_pad, device=device, dtype=torch.int8)
        cache["assigns_i32"] = state["assigns_padded"].to(torch.int32).contiguous()
        cache["invalid_i8"] = state["invalid_mask"].to(torch.int8).contiguous()
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
        raise RuntimeError("TA_attention_v8.2 requires Triton")

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

    run_ta_v8_2_build_cand(
        ws["selected"],
        ws["assigns_i32"],
        ws["invalid_i8"],
        ws["cand_b"],
        ws["cand_kvh"],
        groups=groups,
        chunk_n=_BUILD_CHUNK_N,
    )

    bk_t, bv, b_inv = _prep_buffer_cached(ws, buffer_keys, buffer_values)
    groups_pow = max(next_pow2(groups), 4)
    run_ta_v8_2_attn(
        q=q_f16,
        keys_t_f16=state["keys_padded_t_f16"],
        values_f16=state["values_padded_f16"],
        cand_b_i8=ws["cand_b"],
        cand_kvh_i8=ws["cand_kvh"],
        invalid_mask_i8=ws["invalid_i8"],
        threshold_f32=threshold_f32,
        buf_keys_t_f16=bk_t,
        buf_values_f16=bv,
        buf_invalid_i8=b_inv,
        h_kv_eff=h_kv,
        n_pad=n_pad,
        scale_log2e=scale_log2e,
        groups=groups,
        groups_pow=groups_pow,
        block_n=_BLOCK_N,
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
