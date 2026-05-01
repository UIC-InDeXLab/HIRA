"""TA_attention_v3.0 — v2.4 with tuned launch geometry.

Changes vs v2.4:
1. Candidate compaction uses larger row tiles.
2. Candidate and compact-attention kernels use higher warp count.
"""

from __future__ import annotations

import math

import torch

from ._TA_common import _LOG2E
from ._TA_triton_attn_compact import HAS_TRITON, run_ta_attn_compact
from ._TA_triton_candidates_v2 import run_ta_compact_candidates_stamp
from ._TA_triton_centroid_scores import run_ta_centroid_scores
from ._TA_triton_depth import run_ta_stop_depth
from ._TA_triton_reduce import run_ta_reduce
from .TA_attention_v_1_2 import attend as attend_v1_2

KERNEL_VERSION = "v3.0"

_CANDIDATE_SPLITS = 128
_ATTN_SPLITS = 96
_BLOCK_ROWS = 16
_BLOCK_N = 32
_CENTER_BLOCK_K = 64
_CAND_NUM_WARPS = 8
_CAND_NUM_STAGES = 2
_ATTN_NUM_WARPS = 8
_ATTN_NUM_STAGES = 4


def _empty_buffer(buffer_keys, buffer_values) -> bool:
    return buffer_keys is None or buffer_values is None or int(buffer_keys.shape[1]) == 0


def _grouped_or_identity_groups(q_head_to_kv, h_q: int, h_kv: int) -> int | None:
    if q_head_to_kv is None:
        return 1 if h_q == h_kv else None
    if h_q % h_kv != 0:
        return None
    groups = h_q // h_kv
    expected = torch.arange(h_q, device=q_head_to_kv.device, dtype=q_head_to_kv.dtype) // groups
    if not torch.equal(q_head_to_kv, expected):
        return None
    return groups


def _prep_buffer(buffer_keys, buffer_values):
    if _empty_buffer(buffer_keys, buffer_values):
        return None, None, None
    bk = (
        buffer_keys
        if buffer_keys.dtype == torch.float16 and buffer_keys.is_contiguous()
        else buffer_keys.to(torch.float16).contiguous()
    )
    bv = (
        buffer_values
        if buffer_values.dtype == torch.float16 and buffer_values.is_contiguous()
        else buffer_values.to(torch.float16).contiguous()
    )
    h_eff, l_buf, _ = bk.shape
    bk_t = bk.transpose(-1, -2).contiguous()
    inv = torch.zeros(h_eff, l_buf, dtype=torch.int8, device=bk.device)
    return bk_t, bv, inv


def _workspace(state: dict, *, h_q: int, s_sub: int, k_clusters: int, n_pad: int, d_v: int, device) -> dict:
    cache = state.setdefault("_TA_attention_v3_0_cache", {})
    key = (device.index, h_q, s_sub, k_clusters, n_pad, d_v, _ATTN_SPLITS)
    if cache.get("key") != key:
        cache.clear()
        cache["key"] = key
        cache["stamp"] = 0
        cache["stamp_tensor"] = torch.empty((), device=device, dtype=torch.int32)
        cache["scores"] = torch.empty(h_q, s_sub, k_clusters, device=device, dtype=torch.float16)
        cache["depth"] = torch.empty(h_q, device=device, dtype=torch.int32)
        cache["visited"] = torch.zeros(h_q, n_pad, device=device, dtype=torch.int32)
        cache["counts"] = torch.empty(h_q, device=device, dtype=torch.int32)
        cache["cand_ids"] = torch.empty(h_q, n_pad, device=device, dtype=torch.int32)
        cache["out_m"] = torch.empty(h_q, _ATTN_SPLITS, device=device, dtype=torch.float32)
        cache["out_l"] = torch.empty(h_q, _ATTN_SPLITS, device=device, dtype=torch.float32)
        cache["out_o"] = torch.empty(h_q, _ATTN_SPLITS, d_v, device=device, dtype=torch.float32)
        cache["out"] = torch.empty(h_q, d_v, device=device, dtype=torch.float32)
        cache["buf_key"] = None
        cache["buf_prepped"] = (None, None, None)
    return cache


def _prep_buffer_cached(ws: dict, buffer_keys, buffer_values):
    key = None
    if not _empty_buffer(buffer_keys, buffer_values):
        key = (
            int(buffer_keys.data_ptr()),
            tuple(buffer_keys.shape),
            buffer_keys.dtype,
            int(buffer_values.data_ptr()),
            tuple(buffer_values.shape),
            buffer_values.dtype,
            buffer_keys.device.index,
        )
    if ws.get("buf_key") != key:
        ws["buf_key"] = key
        ws["buf_prepped"] = _prep_buffer(buffer_keys, buffer_values)
    return ws["buf_prepped"]


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
        raise RuntimeError("TA_attention_v3.0 requires Triton")
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
    run_ta_stop_depth(sorted_scores, threshold_f32, ws["depth"])

    stamp = int(ws["stamp"]) + 1
    if stamp >= 2_000_000_000:
        ws["visited"].zero_()
        stamp = 1
    ws["stamp"] = stamp
    ws["stamp_tensor"].fill_(stamp)
    ws["counts"].zero_()
    run_ta_compact_candidates_stamp(
        order,
        ws["depth"],
        state["children_padded_i32"],
        ws["visited"],
        ws["counts"],
        ws["cand_ids"],
        ws["stamp_tensor"],
        n_pad=n_pad,
        groups=groups,
        block_rows=_BLOCK_ROWS,
        num_splits=_CANDIDATE_SPLITS,
        num_warps=_CAND_NUM_WARPS,
        num_stages=_CAND_NUM_STAGES,
    )

    bk_t, bv, b_inv = _prep_buffer_cached(ws, buffer_keys, buffer_values)
    run_ta_attn_compact(
        q=q_f16,
        keys_t_f16=state["keys_padded_t_f16"],
        values_f16=state["values_padded_f16"],
        cand_ids_i32=ws["cand_ids"],
        counts_i32=ws["counts"],
        threshold_f32=threshold_f32,
        buf_keys_t_f16=bk_t,
        buf_values_f16=bv,
        buf_invalid_i8=b_inv,
        n_pad=n_pad,
        scale_log2e=scale_log2e,
        groups=groups,
        block_n=_BLOCK_N,
        num_splits=_ATTN_SPLITS,
        out_m=ws["out_m"],
        out_l=ws["out_l"],
        out_o=ws["out_o"],
        num_warps=_ATTN_NUM_WARPS,
        num_stages=_ATTN_NUM_STAGES,
    )
    run_ta_reduce(ws["out_m"], ws["out_l"], ws["out_o"], ws["out"])
    return ws["out"]


KERNEL = attend
