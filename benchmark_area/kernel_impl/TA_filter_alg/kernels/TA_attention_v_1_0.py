"""TA_attention_v1.0 — pure torch reference.

This is the simplest, slowest, most-obviously-correct implementation of the
TA-filter algorithm. Use it as the correctness baseline for the other v1.x
kernels and to sanity-check the algorithm spec end-to-end.

Pipeline (per query):
  1. Compute centroid scores M[h, s, c] = q_s . mu_{s,c}.
  2. Sort each subspace's K clusters by descending M.
  3. Find L* = first row L where sum_s sorted M[h, s, L] < T (else K).
  4. Mark a key as a "candidate" iff at least one of its S parent clusters
     ranks within L* in its subspace.
  5. Score the candidate set with q.k, set scores below T to -inf.
  6. Softmax over the surviving scores; multiply by V; sum over keys.
  7. Fold the buffer keys/values into the same softmax (they always score).
"""

from __future__ import annotations

import math

import torch

from ._TA_common import (
    build_selected_clusters,
    compute_centroid_scores,
    expand_for_query,
    per_key_candidate_mask,
    stop_depth_per_head,
)

KERNEL_VERSION = "v1.0"


def _empty_buffer(buffer_keys, buffer_values) -> bool:
    return (
        buffer_keys is None
        or buffer_values is None
        or int(buffer_keys.shape[1]) == 0
    )


def attend(
    q: torch.Tensor,
    threshold: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
    keys_children: torch.Tensor | None = None,  # unused, accepted for parity
) -> torch.Tensor:
    """Compute TA-filter attention output (H_q, D_v).

    Args:
        q: (H_q, D) fp16 query vectors.
        threshold: (H_q,) fp16/fp32 scalar T per query head.
        state: TA build state dict using the v1-compatible layout.
        buffer_keys: (H_kv, L_buf, D) fp16 or None.
        buffer_values: (H_kv, L_buf, D_v) fp16 or None.
        q_head_to_kv: (H_q,) int64 GQA map or None.
        scale: float; defaults to 1/sqrt(D).
    """
    del keys_children
    if state.get("version") != "v1.0":
        # Allow newer compatible TA builds too.
        pass

    h_q, d = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # ── Step 1-2: centroid scores + sort ──
    scores_h_s_k = compute_centroid_scores(
        q=q,
        centers_padded_f16=state["centers_padded_f16"],
        dim_slices=state["dim_slices"],
        q_head_to_kv=q_head_to_kv,
    )                                                                # (H_q, S, K)
    sorted_scores, order = torch.sort(scores_h_s_k, dim=-1, descending=True)

    # ── Step 3: stop depth ──
    threshold_f32 = threshold.float()
    depth = stop_depth_per_head(sorted_scores, threshold_f32)        # (H_q,)

    # ── Step 4: per-key candidate mask ──
    selected = build_selected_clusters(order, depth)                 # (H_q, S, K)
    cand_mask = per_key_candidate_mask(
        selected=selected,
        assigns_padded=state["assigns_padded"],
        q_head_to_kv=q_head_to_kv,
    )                                                                # (H_q, N_pad)

    # ── Step 5-6: score + filter + softmax + V ──
    keys_padded_f16 = state["keys_padded_f16"]                       # (H_kv, N_pad, D)
    values_padded_f16 = state["values_padded_f16"]                   # (H_kv, N_pad, D_v)
    invalid_mask = state["invalid_mask"]                             # (H_kv, N_pad)
    keys_eff = expand_for_query(keys_padded_f16, q_head_to_kv).float()
    values_eff = expand_for_query(values_padded_f16, q_head_to_kv).float()
    invalid_eff = expand_for_query(invalid_mask, q_head_to_kv)

    qf = q.float()
    raw_scores = torch.einsum("hd,hnd->hn", qf, keys_eff)            # (H_q, N_pad)
    survive = cand_mask & (~invalid_eff) & (raw_scores >= threshold_f32.unsqueeze(-1))
    scaled = raw_scores * scale
    scaled = scaled.masked_fill(~survive, float("-inf"))             # (H_q, N_pad)

    # ── Step 7: optionally include buffer ──
    has_buf = not _empty_buffer(buffer_keys, buffer_values)
    if has_buf:
        kbuf = expand_for_query(buffer_keys, q_head_to_kv).float()
        vbuf = expand_for_query(buffer_values, q_head_to_kv).float()
        buf_scores = torch.einsum("hd,hnd->hn", qf, kbuf) * scale    # (H_q, L_buf)
        all_scaled = torch.cat([scaled, buf_scores], dim=-1)
        m = all_scaled.amax(dim=-1, keepdim=True)
        m = m.masked_fill(torch.isinf(m), 0.0)
        e = torch.exp(all_scaled - m)
        denom = e.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        probs = e / denom
        probs_idx = probs[:, : scaled.shape[-1]]
        probs_buf = probs[:, scaled.shape[-1] :]
        out = torch.einsum("hn,hnd->hd", probs_idx, values_eff) + torch.einsum(
            "hn,hnd->hd", probs_buf, vbuf
        )
    else:
        m = scaled.amax(dim=-1, keepdim=True)
        # Heads with zero candidates produce zeros (avoid -inf - -inf NaNs).
        m = m.masked_fill(torch.isinf(m), 0.0)
        e = torch.exp(scaled - m)
        denom = e.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        probs = e / denom
        out = torch.einsum("hn,hnd->hd", probs, values_eff)

    return out.float()


KERNEL = attend
