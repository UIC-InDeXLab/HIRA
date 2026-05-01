"""TA_attention_v1.2 — torch prep + Triton split-N attention with INLINE mask.

Same prep as v1.1 (centroid dot + sort + L*) but the per-key OR-over-subspaces
candidate check is evaluated INSIDE the attention kernel rather than
materialised as an (H_q, N_pad) tensor in torch first.  Saves the mask write
and read at the cost of S extra small indexed loads per (chunk, group) inside
the kernel.

This is meant as a memory-bandwidth experiment: the precomputed mask in v1.1
is ~H_q * N_pad bytes that we have to write and re-read; v1.2 trades that for
inline gather work.  Whether it wins depends on H_q, N_pad, and the relative
costs.
"""

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

KERNEL_VERSION = "v1.2"

_DEFAULT_BLOCK_N = 64
_DEFAULT_NUM_SPLITS = 32


def _empty_buffer(buffer_keys, buffer_values) -> bool:
    return (
        buffer_keys is None
        or buffer_values is None
        or int(buffer_keys.shape[1]) == 0
    )


def _prep_buffer(buffer_keys, buffer_values, q_head_to_kv):
    if _empty_buffer(buffer_keys, buffer_values):
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


def _expand_assigns_for_query(assigns_padded, q_head_to_kv):
    if q_head_to_kv is None:
        return assigns_padded.contiguous()
    # assigns_padded is (S, H_kv, N_pad) — index along H_kv.
    return assigns_padded.index_select(1, q_head_to_kv).contiguous()


def attend(
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
        raise RuntimeError("TA_attention_v1.2 requires Triton")

    h_q, d = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale_log2e = float(scale) * _LOG2E

    n_pad = int(state["N_pad"])
    s_sub = int(state["n_subspaces"])
    k_clusters = int(state["K"])
    h_kv = int(state["keys_padded_f16"].shape[0])

    # ── prep: centroid scores → sort → L* → selected[h, s, c] ──
    scores_h_s_k = compute_centroid_scores(
        q=q,
        centers_padded_f16=state["centers_padded_f16"],
        dim_slices=state["dim_slices"],
        q_head_to_kv=q_head_to_kv,
    )
    sorted_scores, order = torch.sort(scores_h_s_k, dim=-1, descending=True)
    threshold_f32 = threshold.float().contiguous()
    depth = stop_depth_per_head(sorted_scores, threshold_f32)
    selected = build_selected_clusters(order, depth)                 # (H_q, S, K) bool
    selected_i8 = selected.to(torch.int8).contiguous()

    # ── Per-head (kv-effective) layout ──
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
    # ``assigns_padded`` is (S, H_kv, N_pad); for grouped GQA H_kv_eff = H_kv.
    # If we ever support "expanded" mode this would need re-indexing along axis 1.
    if q_head_to_kv is not None and groups == 1:
        # Heads have changed — index along axis 1.
        assigns_eff = state["assigns_padded"].index_select(1, q_head_to_kv).to(torch.int32).contiguous()

    groups_pow = max(next_pow2(groups), 4)

    # ── Buffer ──
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


KERNEL = attend
