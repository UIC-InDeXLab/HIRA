"""Shared helpers for TA-filter attention kernels."""

from __future__ import annotations

import torch


_LOG2E = 1.4426950408889634


def next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def expand_for_query(
    tensor: torch.Tensor, q_head_to_kv: torch.Tensor | None
) -> torch.Tensor:
    if q_head_to_kv is None:
        return tensor
    return tensor.index_select(0, q_head_to_kv).contiguous()


def gqa_mode(q_head_to_kv: torch.Tensor | None, h_q: int, h_kv: int) -> str:
    """Pick a GQA layout strategy.

    "identity": H_q == H_kv, no remapping.
    "grouped":  q_head_to_kv == arange(H_q) // groups, can broadcast cleanly.
    "expanded": fall back to materializing per-H_q copies of K/V/centers.
    """
    if q_head_to_kv is None:
        return "identity"
    if h_q % h_kv == 0:
        groups = h_q // h_kv
        expected = (
            torch.arange(h_q, device=q_head_to_kv.device, dtype=q_head_to_kv.dtype)
            // groups
        )
        if torch.equal(q_head_to_kv, expected):
            return "grouped"
    return "expanded"


_Q_PACK_CACHE: dict = {}


def _q_subspace_pack_indices(
    dim_slices: list[tuple[int, int]],
    max_w: int,
    d: int,
    device: torch.device,
) -> torch.Tensor:
    """Pre-build (S, max_w) index map: q_packed[s, w] = q[dim_slices[s][0] + w]
    for w < width(s), and a sentinel index otherwise.

    The sentinel is ``d`` so we can index into a length-(d+1) padded q whose
    last entry is zero — this lets us avoid masked gathers.
    """
    key = (id(dim_slices), tuple(dim_slices), max_w, d, device)
    cached = _Q_PACK_CACHE.get(key)
    if cached is not None:
        return cached
    s_count = len(dim_slices)
    idx = torch.full((s_count, max_w), d, dtype=torch.long, device=device)
    for s_idx, (start, end) in enumerate(dim_slices):
        w = end - start
        idx[s_idx, :w] = torch.arange(start, end, device=device)
    _Q_PACK_CACHE[key] = idx
    return idx


def compute_centroid_scores(
    q: torch.Tensor,
    centers_padded_f16: torch.Tensor,
    dim_slices: list[tuple[int, int]],
    q_head_to_kv: torch.Tensor | None,
) -> torch.Tensor:
    """Centroid scores M[h, s, c] = q_s . mu_{s,c} (no radii).

    Single fused einsum over (H_q, S, K, max_w).  Q is packed once per
    dim_slices into shape (S, H_q, max_w) using a cached scatter index;
    centers_padded_f16 already lives in that layout.

    Args:
        q: (H_q, D) fp16/fp32 query vectors.
        centers_padded_f16: (S, H_kv, K, max_w) fp16, zero-padded across width.
        dim_slices: list of (start, end) per subspace.
        q_head_to_kv: (H_q,) int64 GQA map or None.

    Returns:
        scores: (H_q, S, K) fp32.
    """
    s_count, h_kv, k, max_w = centers_padded_f16.shape
    h_q, d = q.shape
    device = q.device

    pad_idx = _q_subspace_pack_indices(dim_slices, max_w, d, device)  # (S, max_w)
    q_padded = torch.cat(
        [q.float(), torch.zeros(h_q, 1, device=device, dtype=torch.float32)], dim=1
    )                                                                # (H_q, D+1)
    # Gather slices: result shape (H_q, S, max_w) — q_padded[h, pad_idx[s, w]].
    q_packed = q_padded.index_select(1, pad_idx.view(-1)).view(h_q, s_count, max_w)

    if q_head_to_kv is None:
        centers_eff = centers_padded_f16.float()                     # (S, H_kv, K, max_w)
        # einsum: (S, H_kv, K, w) * (H_kv=H_q, S, w) -> (H_q, S, K)
        out = torch.einsum("shkw,hsw->hsk", centers_eff, q_packed)
    else:
        # Index along H_kv → H_q.  Cheap (H_q rows of size K*max_w).
        centers_eff = centers_padded_f16.index_select(1, q_head_to_kv).float()
        out = torch.einsum("shkw,hsw->hsk", centers_eff, q_packed)
    return out.contiguous()


def stop_depth_per_head(
    sorted_scores: torch.Tensor, threshold: torch.Tensor
) -> torch.Tensor:
    """Smallest L (1-based, inclusive) such that row-sum_{L} of sorted scores < T.

    Args:
        sorted_scores: (H_q, S, K) fp32 — descending along K per (h, s).
        threshold: (H_q,) fp32 — scalar T per query head.

    Returns:
        depth: (H_q,) int64 in [1, K].
    """
    h_q, _s, k = sorted_scores.shape
    row_sums = sorted_scores.sum(dim=1)                              # (H_q, K)
    below = row_sums < threshold.unsqueeze(-1)                       # (H_q, K)
    has = below.any(dim=-1)
    first = below.float().argmax(dim=-1)
    depth = torch.where(has, first + 1, torch.full_like(first, k))
    return depth


def build_selected_clusters(
    order: torch.Tensor, depth: torch.Tensor
) -> torch.Tensor:
    """Boolean (H_q, S, K) — True if cluster c has rank <= L* in subspace s.

    Args:
        order: (H_q, S, K) int64 — cluster ids in descending centroid-score order.
        depth: (H_q,) int64 — L* per head.

    Returns:
        selected: (H_q, S, K) bool.
    """
    h_q, s_count, k = order.shape
    device = order.device
    rank_pos = torch.arange(k, device=device).view(1, 1, k)          # (1,1,K)
    in_top = rank_pos < depth.view(h_q, 1, 1)                        # (H_q, 1, K)
    in_top_b = in_top.expand(h_q, s_count, k).contiguous()
    selected = torch.zeros(h_q, s_count, k, dtype=torch.bool, device=device)
    selected.scatter_(2, order, in_top_b)
    return selected


def per_key_candidate_mask(
    selected: torch.Tensor,
    assigns_padded: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
) -> torch.Tensor:
    """Mark each (h, n) as candidate iff at least one of its S parents is selected.

    Args:
        selected: (H_q, S, K) bool.
        assigns_padded: (S, H_kv, N_pad) int16/int32.
        q_head_to_kv: (H_q,) int64 GQA map or None.

    Returns:
        cand_mask: (H_q, N_pad) bool.
    """
    s_count, h_kv, n_pad = assigns_padded.shape
    device = selected.device
    h_q = int(selected.shape[0])

    if q_head_to_kv is None:
        assigns_eff = assigns_padded                                 # (S, H_q, N_pad)
    else:
        assigns_eff = assigns_padded.index_select(1, q_head_to_kv).contiguous()

    cand = torch.zeros(h_q, n_pad, dtype=torch.bool, device=device)
    for s_idx in range(s_count):
        parents = assigns_eff[s_idx].to(torch.int64)                 # (H_q, N_pad)
        sel_s = selected[:, s_idx, :]                                # (H_q, K)
        passed = sel_s.gather(1, parents)                            # (H_q, N_pad)
        cand |= passed
    return cand
