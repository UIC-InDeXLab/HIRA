"""TA_attention_v13.2 — stamped full-key mask.

Same full-key mask semantics as v13.0, but removes the mask-clear kernel.
Each query increments a device stamp, selected parent children store that stamp
into ``visited[h, key]``, and attention treats ``visited == stamp`` as True.
"""

from __future__ import annotations

import math

import torch

from ._TA_common import _LOG2E
from ._TA_triton_centroid_scores import run_ta_centroid_scores
from ._TA_triton_reduce import run_ta_reduce
from ._TA_triton_v8_3 import run_ta_v8_3_copy_q_th
from ._TA_triton_v13 import (
    HAS_TRITON,
    run_ta_v13_depth_count_stamp,
    run_ta_v13_mark_stamp,
    run_ta_v13_stamp_key_attn,
)
from .TA_attention_v_13_0 import (
    _CENTER_BLOCK_K,
    _ATTN_BLOCK_N,
    _ATTN_SPLITS,
    _ATTN_STAGES,
    _ATTN_WARPS,
    _bucket_for,
    _copy_buffer_into_stage,
    _make_buf_stage,
)
from .TA_attention_v_1_2 import attend as attend_v1_2
from .TA_attention_v_3_0 import _empty_buffer, _grouped_or_identity_groups

KERNEL_VERSION = "v13.2"


def _workspace(
    state: dict,
    *,
    h_q: int,
    s_sub: int,
    k_clusters: int,
    n_pad: int,
    d_v: int,
    d: int,
    device,
) -> dict:
    cache = state.setdefault("_TA_attention_v13_2_cache", {})
    key = (device.index, h_q, s_sub, k_clusters, n_pad, d_v, d, _ATTN_SPLITS, _ATTN_BLOCK_N)
    if cache.get("key") != key:
        cache.clear()
        cache["key"] = key
        cache["static_q"] = torch.empty(h_q, d, device=device, dtype=torch.float16)
        cache["static_th"] = torch.empty(h_q, device=device, dtype=torch.float32)
        cache["scores"] = torch.empty(h_q, s_sub, k_clusters, device=device, dtype=torch.float16)
        cache["sorted_scores"] = torch.empty(h_q, s_sub, k_clusters, device=device, dtype=torch.float16)
        cache["order"] = torch.empty(h_q, s_sub, k_clusters, device=device, dtype=torch.long)
        cache["depth"] = torch.empty(h_q, device=device, dtype=torch.int32)
        cache["counts"] = torch.empty(h_q, device=device, dtype=torch.int32)
        cache["stamp"] = torch.zeros((), device=device, dtype=torch.int32)
        cache["visited"] = torch.zeros(h_q, n_pad, device=device, dtype=torch.int32)
        cache["out_m"] = torch.empty(h_q, _ATTN_SPLITS, device=device, dtype=torch.float32)
        cache["out_l"] = torch.empty(h_q, _ATTN_SPLITS, device=device, dtype=torch.float32)
        cache["out_o"] = torch.empty(h_q, _ATTN_SPLITS, d_v, device=device, dtype=torch.float32)
        cache["out"] = torch.empty(h_q, d_v, device=device, dtype=torch.float32)
        cache["graph"] = None
        cache["capture_failed"] = False
        cache["graph_cache_key"] = None
        cache["buf_stage"] = {}
    return cache


def _launch_pipeline(
    *,
    static_q: torch.Tensor,
    static_th: torch.Tensor,
    state: dict,
    ws: dict,
    bk_t: torch.Tensor | None,
    bv: torch.Tensor | None,
    b_inv: torch.Tensor | None,
    groups: int,
    scale_log2e: float,
) -> None:
    run_ta_centroid_scores(
        static_q,
        state["centers_padded_f16"],
        state["dim_offsets"],
        state["dim_widths"],
        ws["scores"],
        groups=groups,
        block_k=_CENTER_BLOCK_K,
    )
    torch.sort(
        ws["scores"],
        dim=-1,
        descending=True,
        out=(ws["sorted_scores"], ws["order"]),
    )
    run_ta_v13_depth_count_stamp(
        ws["sorted_scores"],
        static_th,
        ws["depth"],
        ws["counts"],
        ws["stamp"],
    )
    run_ta_v13_mark_stamp(
        ws["order"],
        ws["depth"],
        state["parent_children_i32"],
        ws["visited"],
        ws["stamp"],
        groups=groups,
    )
    run_ta_v13_stamp_key_attn(
        q=static_q,
        visited_i32=ws["visited"],
        stamp_i32=ws["stamp"],
        keys_t_f16=state["keys_padded_t_f16"],
        values_f16=state["values_padded_f16"],
        buf_keys_t_f16=bk_t,
        buf_values_f16=bv,
        buf_invalid_i8=b_inv,
        scale_log2e=scale_log2e,
        groups=groups,
        block_n=_ATTN_BLOCK_N,
        num_splits=_ATTN_SPLITS,
        out_m=ws["out_m"],
        out_l=ws["out_l"],
        out_o=ws["out_o"],
        num_warps=_ATTN_WARPS,
        num_stages=_ATTN_STAGES,
    )
    run_ta_reduce(ws["out_m"], ws["out_l"], ws["out_o"], ws["out"])


def _capture_pipeline(ws: dict, launch_kwargs: dict, state: dict) -> None:
    if ws["graph"] is not None or ws["capture_failed"]:
        return
    if not bool(state.get("_TA_attention_v13_2_use_cuda_graphs", True)):
        ws["capture_failed"] = True
        return
    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                _launch_pipeline(**launch_kwargs)
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_pipeline(**launch_kwargs)
        ws["graph"] = graph
    except Exception:
        ws["capture_failed"] = True


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
        raise RuntimeError("TA_attention_v13.2 requires Triton")

    h_q, d = q.shape
    h_kv = int(state["keys_padded_f16"].shape[0])
    d_v = int(state["values_padded_f16"].shape[-1]) if "values_padded_f16" in state else 0
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
        or s_sub != 4
        or bf != 4
        or int(state["max_width"]) > 32
        or "parent_children_i32" not in state
        or "keys_padded_t_f16" not in state
        or "values_padded_f16" not in state
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
        d=d,
        device=q.device,
    )

    has_buf = not _empty_buffer(buffer_keys, buffer_values)
    if has_buf:
        l_buf = int(buffer_keys.shape[1])
        bucket = _bucket_for(l_buf)
        buf_keys_f16 = (
            buffer_keys
            if buffer_keys.dtype == torch.float16 and buffer_keys.is_contiguous()
            else buffer_keys.to(torch.float16).contiguous()
        )
        buf_values_f16 = (
            buffer_values
            if buffer_values.dtype == torch.float16 and buffer_values.is_contiguous()
            else buffer_values.to(torch.float16).contiguous()
        )
        stage = ws["buf_stage"].get(bucket)
        if stage is None:
            stage = _make_buf_stage(d, d_v, h_kv, bucket, q.device)
            ws["buf_stage"][bucket] = stage
        _copy_buffer_into_stage(stage, buf_keys_f16, buf_values_f16)
        bk_t = stage["buf_keys_t"]
        bv = stage["buf_values"]
        b_inv = stage["buf_invalid"]
        graph_cache_key = (bucket, int(stage["valid_len"]))
    else:
        bk_t = None
        bv = None
        b_inv = None
        graph_cache_key = (0, 0)

    if ws["graph_cache_key"] != graph_cache_key:
        ws["graph"] = None
        ws["capture_failed"] = False
        ws["graph_cache_key"] = graph_cache_key

    run_ta_v8_3_copy_q_th(q_f16, threshold_f32, ws["static_q"], ws["static_th"])

    launch_kwargs = dict(
        static_q=ws["static_q"],
        static_th=ws["static_th"],
        state=state,
        ws=ws,
        bk_t=bk_t,
        bv=bv,
        b_inv=b_inv,
        groups=groups,
        scale_log2e=scale_log2e,
    )

    _capture_pipeline(ws, launch_kwargs, state)
    if ws["graph"] is not None:
        ws["graph"].replay()
    else:
        _launch_pipeline(**launch_kwargs)
    return ws["out"]


KERNEL = attend
