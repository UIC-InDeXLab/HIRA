"""Core AABB plus a few explicit outlier points."""

from __future__ import annotations

import torch


def enclose_outlier_aabb(keys, assign, centers, K, bf, n_outliers: int = 1):
    """
    Remove a few farthest points from each cluster, build the AABB on the
    remaining core, and keep the removed points explicitly in the gate.
    """
    H, _, D = keys.shape
    device = keys.device
    M = max(1, n_outliers)

    core_lo = torch.full((H, K, D), float("inf"), device=device, dtype=keys.dtype)
    core_hi = torch.full((H, K, D), float("-inf"), device=device, dtype=keys.dtype)
    outlier_points = torch.zeros(H, K, M, D, device=device, dtype=keys.dtype)
    outlier_mask = torch.zeros(H, K, M, device=device, dtype=torch.bool)

    for h in range(H):
        for k in range(K):
            idx = (assign[h] == k).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                core_lo[h, k] = 0.0
                core_hi[h, k] = 0.0
                continue

            pts = keys[h, idx]
            if idx.numel() <= M:
                core_lo[h, k] = pts.min(dim=0).values
                core_hi[h, k] = pts.max(dim=0).values
                outlier_points[h, k, : idx.numel()] = pts[:M]
                outlier_mask[h, k, : idx.numel()] = True
                continue

            dist = (pts - centers[h, k]).square().sum(dim=-1)
            chosen = dist.topk(min(M, int(idx.numel()) - 1)).indices
            keep = torch.ones(idx.numel(), device=device, dtype=torch.bool)
            keep[chosen] = False

            core = pts[keep]
            core_lo[h, k] = core.min(dim=0).values
            core_hi[h, k] = core.max(dim=0).values
            outlier_points[h, k, : chosen.numel()] = pts[chosen]
            outlier_mask[h, k, : chosen.numel()] = True

    empty = core_lo[:, :, 0].isinf()
    if empty.any():
        core_lo[empty] = 0.0
        core_hi[empty] = 0.0

    def gate(q, th):
        q_exp = q.unsqueeze(1)
        core_upper = torch.maximum(q_exp * core_lo, q_exp * core_hi).sum(dim=-1)
        point_upper = torch.einsum("hkmd,hd->hkm", outlier_points, q)
        point_upper = point_upper.masked_fill(~outlier_mask, float("-inf"))
        best_point = point_upper.amax(dim=-1)
        return torch.maximum(core_upper, best_point) > th.unsqueeze(-1)

    span = (core_hi - core_lo).clamp_min(0)
    return gate, {
        "outliers": M,
        "core_span_mean": float(span.mean()),
    }
