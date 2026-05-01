"""TA_attention_v5.0 — bounded top-prefix prep experiment.

This tries to replace full centroid sorting with a fixed top-R prefix.  The
path is exact only when the stop row is found before R; otherwise it falls back
to v4.0.  The fallback branch intentionally synchronizes, so this version is an
experiment to measure whether prefix selection beats full sort enough to matter.
"""

from __future__ import annotations

import math

import torch

from ._TA_common import _LOG2E, next_pow2
from ._TA_triton_attn_premask import HAS_TRITON, run_ta_attn_premask
from ._TA_triton_centroid_scores import run_ta_centroid_scores
from ._TA_triton_depth import run_ta_stop_depth
from ._TA_triton_masks_v4 import run_ta_mark_candidates_prefix_from_children
from ._TA_triton_reduce import run_ta_reduce
from .TA_attention_v_1_2 import attend as attend_v1_2
from .TA_attention_v_3_0 import _grouped_or_identity_groups, _prep_buffer_cached
from .TA_attention_v_4_0 import attend as attend_v4_0

KERNEL_VERSION = "v5.0"

_TOP_R = 384
_CENTER_BLOCK_K = 64
_MARK_ROWS = 16
_MARK_SPLITS = 64
_BLOCK_N = 32
_ATTN_SPLITS = 32


def _workspace(state: dict, *, h_q: int, s_sub: int, k_clusters: int, n_pad: int, d_v: int, device) -> dict:
    cache = state.setdefault("_TA_attention_v5_0_cache", {})
    top_r = min(_TOP_R, k_clusters)
    key = (device.index, h_q, s_sub, k_clusters, top_r, n_pad, d_v, _ATTN_SPLITS)
    if cache.get("key") != key:
        cache.clear()
        cache["key"] = key
        cache["scores"] = torch.empty(h_q, s_sub, k_clusters, device=device, dtype=torch.float16)
        cache["depth"] = torch.empty(h_q, device=device, dtype=torch.int32)
        cache["cand_mask"] = torch.empty(h_q, n_pad, device=device, dtype=torch.int8)
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
        raise RuntimeError("TA_attention_v5.0 requires Triton")
    if "children_padded_i32" not in state:
        return attend_v1_2(q, threshold, state, buffer_keys, buffer_values, q_head_to_kv, scale)

    h_q, d = q.shape
    h_kv = int(state["keys_padded_f16"].shape[0])
    d_v = int(state["values_padded_f16"].shape[-1])
    s_sub = int(state["n_subspaces"])
    k_clusters = int(state["K"])
    n_pad = int(state["N_pad"])
    groups = _grouped_or_identity_groups(q_head_to_kv, h_q, h_kv)
    if (
        groups is None
        or groups > 8
        or d != 128
        or d_v != 128
        or s_sub != 8
        or int(state["bf"]) != 4
        or int(state["max_width"]) > 32
    ):
        return attend_v1_2(q, threshold, state, buffer_keys, buffer_values, q_head_to_kv, scale)

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale_log2e = float(scale) * _LOG2E
    q_f16 = q if q.dtype == torch.float16 and q.is_contiguous() else q.to(torch.float16).contiguous()
    threshold_f32 = threshold if threshold.dtype == torch.float32 and threshold.is_contiguous() else threshold.float().contiguous()

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
    top_r = min(_TOP_R, k_clusters)
    top_scores, order_prefix = torch.topk(ws["scores"], k=top_r, dim=-1, largest=True, sorted=True)
    run_ta_stop_depth(top_scores, threshold_f32, ws["depth"])

    # Exactness guard: depth == top_r can mean either "stopped at row R" or
    # "did not stop inside prefix".  Fall back conservatively.
    if bool((ws["depth"] >= top_r).any().item()):
        state["_TA_attention_v5_0_last_fallback"] = "prefix_exhausted"
        return attend_v4_0(q, threshold, state, buffer_keys, buffer_values, q_head_to_kv, scale)
    state["_TA_attention_v5_0_last_fallback"] = None

    ws["cand_mask"].zero_()
    run_ta_mark_candidates_prefix_from_children(
        order_prefix,
        ws["depth"],
        state["children_padded_i32"],
        ws["cand_mask"],
        n_pad=n_pad,
        groups=groups,
        block_rows=_MARK_ROWS,
        num_splits=_MARK_SPLITS,
        num_warps=8,
        num_stages=2,
    )

    bk_t, bv, b_inv = _prep_buffer_cached(ws, buffer_keys, buffer_values)
    run_ta_attn_premask(
        q=q_f16,
        keys_t_f16=state["keys_padded_t_f16"],
        values_f16=state["values_padded_f16"],
        cand_mask_i8=ws["cand_mask"],
        invalid_mask_i8=ws["invalid_i8"],
        threshold_f32=threshold_f32,
        buf_keys_t_f16=bk_t,
        buf_values_f16=bv,
        buf_invalid_i8=b_inv,
        h_kv_eff=h_kv,
        n_pad=n_pad,
        scale=float(scale),
        scale_log2e=scale_log2e,
        groups=groups,
        groups_pow=max(next_pow2(groups), 4),
        block_n=_BLOCK_N,
        num_splits=_ATTN_SPLITS,
        out_m=ws["out_m"],
        out_l=ws["out_l"],
        out_o=ws["out_o"],
        num_warps=4,
        num_stages=3,
    )
    run_ta_reduce(ws["out_m"], ws["out_l"], ws["out_o"], ws["out"])
    return ws["out"]


KERNEL = attend
