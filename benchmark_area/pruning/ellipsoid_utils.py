"""Utility helpers for ellipsoid-based pruning indexes."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from .random_exact_ball_index import _minimum_enclosing_ball_numpy


@dataclass
class Ellipsoid:
    """Represents an ellipsoid defined by (x - c)^T A (x - c) <= 1."""

    center: torch.Tensor
    inv_shape_matrix: torch.Tensor  # A^{-1} to compute support functions efficiently

    def intersects_halfspace(self, q: torch.Tensor, threshold: float) -> bool:
        """Check if ellipsoid intersects halfspace {x: q^T x >= threshold}."""

        q = q.to(self.center.device)
        support_sq = torch.matmul(q, torch.matmul(self.inv_shape_matrix, q))
        support_sq = torch.clamp(support_sq, min=0.0)
        support = torch.sqrt(support_sq)
        max_dot = torch.dot(self.center, q) + support
        return max_dot >= threshold


def _ball_fallback(points_np: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback to a spherical ellipsoid defined by the smallest enclosing ball."""

    center_np, radius = _minimum_enclosing_ball_numpy(points_np, rng)
    dim = points_np.shape[1]
    inv_matrix = (radius ** 2) * np.eye(dim, dtype=np.float64)
    return center_np, inv_matrix


def minimum_volume_enclosing_ellipsoid_numpy(
    points_np: np.ndarray,
    rng: np.random.Generator,
    tol: float = 1e-5,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute center and inverse shape matrix for the minimum-volume enclosing ellipsoid."""

    num_points, dim = points_np.shape
    if num_points == 0:
        return np.zeros(dim, dtype=np.float64), np.zeros((dim, dim), dtype=np.float64)
    if num_points <= dim + 1:
        return _ball_fallback(points_np, rng)

    Q = np.vstack([points_np.T, np.ones(num_points, dtype=np.float64)])
    u = np.ones(num_points, dtype=np.float64) / num_points

    for _ in range(max_iter):
        X = Q @ np.diag(u) @ Q.T
        try:
            X_inv = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            return _ball_fallback(points_np, rng)

        M = np.einsum("ij,jk,ki->i", Q.T, X_inv, Q)
        j = int(np.argmax(M))
        max_value = M[j]
        if max_value <= dim + 1 + tol:
            break

        step = (max_value - dim - 1.0) / ((dim + 1.0) * (max_value - 1.0))
        step = float(np.clip(step, 0.0, 1.0))
        new_u = (1.0 - step) * u
        new_u[j] += step
        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break
        u = new_u

    center_np = points_np.T @ u
    centered = points_np - center_np
    weighted = centered * u[:, None]
    cov = centered.T @ weighted
    if np.linalg.matrix_rank(cov) < dim:
        return _ball_fallback(points_np, rng)

    inv_matrix = cov * dim
    return center_np, inv_matrix


def compute_ellipsoid_tensors(
    cluster_points: torch.Tensor,
    rng: np.random.Generator,
    tol: float = 1e-5,
    max_iter: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert numpy ellipsoid parameters into tensors on the same device/dtype as the cluster."""

    if cluster_points.numel() == 0:
        raise ValueError("Cannot build ellipsoid without points")

    device = cluster_points.device
    dtype = cluster_points.dtype
    points_np = cluster_points.detach().cpu().numpy().astype(np.float64)
    center_np, inv_matrix_np = minimum_volume_enclosing_ellipsoid_numpy(
        points_np, rng, tol=tol, max_iter=max_iter
    )
    center = torch.tensor(center_np, dtype=dtype, device=device)
    inv_matrix = torch.tensor(inv_matrix_np, dtype=dtype, device=device)
    return center, inv_matrix
