"""Shared legacy attention helpers for modern TA kernels."""

from __future__ import annotations

import math

import torch

from ._TA_common import (
    _LOG2E,
    build_selected_clusters,
    compute_centroid_scores,
    next_pow2,
    stop_depth_per_head,
)
from ._TA_triton_attn_inline import HAS_TRITON, run_ta_attn_inline
from ._TA_triton_reduce import run_ta_reduce

_DEFAULT_BLOCK_N = 64
_DEFAULT_NUM_SPLITS = 32


def empty_buffer(buffer_keys, buffer_values) -> bool:
    return (
        buffer_keys is None
        or buffer_values is None
        or int(buffer_keys.shape[1]) == 0
    )


def grouped_or_identity_groups(q_head_to_kv, h_q: int, h_kv: int) -> int | None:
    if q_head_to_kv is None:
        return 1 if h_q == h_kv else None
    if h_q % h_kv != 0:
        return None
    groups = h_q // h_kv
    expected = torch.arange(h_q, device=q_head_to_kv.device, dtype=q_head_to_kv.dtype) // groups
    if not torch.equal(q_head_to_kv, expected):
        return None
    return groups


def _prep_buffer(buffer_keys, buffer_values, q_head_to_kv):
    if empty_buffer(buffer_keys, buffer_values):
        return None, None, None
    if q_head_to_kv is None:
        bk = buffer_keys.to(torch.float16).contiguous()
        bv = buffer_values.to(torch.float16).contiguous()
    else:
        bk = buffer_keys.index_select(0, q_head_to_kv).to(torch.float16).contiguous()
        bv = buffer_values.index_select(0, q_head_to_kv).to(torch.float16).contiguous()
    h_eff, l_buf, _ = bk.shape
    bk_t = bk.transpose(-1, -2).contiguous()
    inv = torch.zeros(h_eff, l_buf, dtype=torch.int8, device=bk.device)
    return bk_t, bv, inv


def attend_fallback_v1_2(
    q: torch.Tensor,
    threshold: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
    keys_children: torch.Tensor | None = None,
    block_n: int = _DEFAULT_BLOCK_N,
    num_splits: int = _DEFAULT_NUM_SPLITS,
) -> torch.Tensor:
    del keys_children
    if not HAS_TRITON:
        raise RuntimeError("fallback v1.2 path requires Triton")

    h_q, d = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale_log2e = float(scale) * _LOG2E

    n_pad = int(state["N_pad"])
    s_sub = int(state["n_subspaces"])
    k_clusters = int(state["K"])
    h_kv = int(state["keys_padded_f16"].shape[0])

    scores_h_s_k = compute_centroid_scores(
        q=q,
        centers_padded_f16=state["centers_padded_f16"],
        dim_slices=state["dim_slices"],
        q_head_to_kv=q_head_to_kv,
    )
    sorted_scores, order = torch.sort(scores_h_s_k, dim=-1, descending=True)
    threshold_f32 = threshold.float().contiguous()
    depth = stop_depth_per_head(sorted_scores, threshold_f32)
    selected = build_selected_clusters(order, depth)
    selected_i8 = selected.to(torch.int8).contiguous()

    if q_head_to_kv is None:
        groups = 1
        h_kv_eff = h_kv
        keys_t = state["keys_padded_t_f16"]
        values = state["values_padded_f16"]
        invalid = state["invalid_mask"].to(torch.int8).contiguous()
    else:
        groups = h_q // h_kv
        h_kv_eff = h_kv
        keys_t = state["keys_padded_t_f16"]
        values = state["values_padded_f16"]
        invalid = state["invalid_mask"].to(torch.int8).contiguous()

    assigns_eff = state["assigns_padded"].to(torch.int32).contiguous()
    if q_head_to_kv is not None and groups == 1:
        assigns_eff = state["assigns_padded"].index_select(1, q_head_to_kv).to(torch.int32).contiguous()

    groups_pow = max(next_pow2(groups), 4)
    bk_t, bv, b_inv = _prep_buffer(buffer_keys, buffer_values, q_head_to_kv)
    q_f16 = q.to(torch.float16).contiguous()

    d_v = int(values.shape[-1])
    device = q.device
    out_m = torch.empty(h_q, num_splits, device=device, dtype=torch.float32)
    out_l = torch.empty(h_q, num_splits, device=device, dtype=torch.float32)
    out_o = torch.empty(h_q, num_splits, d_v, device=device, dtype=torch.float32)
    out = torch.empty(h_q, d_v, device=device, dtype=torch.float32)

    run_ta_attn_inline(
        q=q_f16,
        keys_t_f16=keys_t,
        values_f16=values,
        selected_i8=selected_i8,
        assigns_i32=assigns_eff,
        invalid_mask_i8=invalid,
        threshold_f32=threshold_f32,
        buf_keys_t_f16=bk_t,
        buf_values_f16=bv,
        buf_invalid_i8=b_inv,
        h_kv_eff=h_kv_eff,
        n_pad=n_pad,
        s_sub=s_sub,
        k_clusters=k_clusters,
        scale_log2e=scale_log2e,
        groups=groups,
        groups_pow=groups_pow,
        block_n=block_n,
        num_splits=num_splits,
        out_m=out_m,
        out_l=out_l,
        out_o=out_o,
    )
    run_ta_reduce(out_m, out_l, out_o, out)
    return out
