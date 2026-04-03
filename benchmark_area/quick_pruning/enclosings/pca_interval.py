"""Local-PCA single-axis interval bound."""

from __future__ import annotations

import torch


def enclose_pca_interval(keys, assign, centers, K, bf):
    """
    Fit one local principal axis per cluster and bound the remaining energy with
    an orthogonal residual radius.

    With `bf` this small, a tiny per-cluster SVD is cheap enough to try while
    preserving a single-axis query gate.
    """
    H, _, D = keys.shape
    device = keys.device

    axis = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    coeff_lo = torch.zeros(H, K, device=device, dtype=keys.dtype)
    coeff_hi = torch.zeros(H, K, device=device, dtype=keys.dtype)
    residual = torch.zeros(H, K, device=device, dtype=keys.dtype)

    for h in range(H):
        for k in range(K):
            idx = (assign[h] == k).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue

            pts = keys[h, idx]
            if idx.numel() == 1:
                u = pts[0] / pts[0].norm().clamp_min(1e-12)
            else:
                _, _, vh = torch.linalg.svd(pts, full_matrices=False)
                u = vh[0]

            coeff = pts @ u
            perp = (pts.square().sum(dim=-1) - coeff.square()).clamp_min(0).sqrt()

            axis[h, k] = u
            coeff_lo[h, k] = coeff.min()
            coeff_hi[h, k] = coeff.max()
            residual[h, k] = perp.max()

    def gate(q, th):
        proj = torch.einsum("hkd,hd->hk", axis, q).clamp(-1, 1)
        proj_perp = (1.0 - proj.square()).clamp_min(0).sqrt()
        upper = torch.maximum(proj * coeff_lo, proj * coeff_hi) + residual * proj_perp
        return upper > th.unsqueeze(-1)

    return gate, {
        "coeff_span_mean": float((coeff_hi - coeff_lo).mean()),
        "residual_mean": float(residual.mean()),
    }
