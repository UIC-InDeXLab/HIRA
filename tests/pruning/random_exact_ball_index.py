"""
Random Exact Ball Index: random centroid selection with exact smallest enclosing balls.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class Ball:
    """Represents a ball in d-dimensional space."""

    center: torch.Tensor  # [dim]
    radius: float

    def intersects_halfspace(self, q: torch.Tensor, threshold: float) -> bool:
        """Check if ball intersects with halfspace {x: q^T x >= threshold}."""

        max_dot = torch.dot(self.center, q) + self.radius
        return max_dot >= threshold


def _ball_from(points: List[np.ndarray], dim: int) -> Tuple[np.ndarray, float]:
    """Compute ball defined by <= dim+1 boundary points."""

    if not points:
        return np.zeros(dim, dtype=np.float64), 0.0
    if len(points) == 1:
        return points[0].copy(), 0.0
    if len(points) == 2:
        center = (points[0] + points[1]) / 2.0
        radius = np.linalg.norm(points[0] - center)
        return center, radius

    pts = np.stack(points)
    p0 = pts[0]
    a = 2.0 * (pts[1:] - p0)
    b = np.sum(pts[1:] ** 2, axis=1) - np.dot(p0, p0)

    if a.shape[0] == a.shape[1]:
        try:
            center = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            center = np.linalg.lstsq(a, b, rcond=None)[0]
    else:
        center = np.linalg.lstsq(a, b, rcond=None)[0]

    radius = np.linalg.norm(pts - center, axis=1).max()
    return center, radius


def _welzl(
    points: np.ndarray, r: List[np.ndarray], n: int, dim: int
) -> Tuple[np.ndarray, float]:
    """Welzl's algorithm for smallest enclosing ball."""

    if n == 0 or len(r) == dim + 1:
        return _ball_from(r, dim)

    p = points[n - 1].copy()
    center, radius = _welzl(points, r, n - 1, dim)
    if np.linalg.norm(p - center) <= radius + 1e-9:
        return center, radius

    return _welzl(points, r + [p], n - 1, dim)


def _minimum_enclosing_ball_numpy(
    points_np: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, float]:
    """Compute smallest enclosing ball for numpy points using Welzl's algorithm."""

    num_points, dim = points_np.shape
    if num_points == 0:
        return np.zeros(dim, dtype=np.float64), 0.0

    shuffled = points_np.astype(np.float64, copy=True)
    rng.shuffle(shuffled)
    center, radius = _welzl(shuffled, [], num_points, dim)
    return center, radius


class RandomExactBallIndex:
    """Index with random centroids and exact smallest enclosing balls."""

    def __init__(self, num_centroids: int = 100, seed: int = 42, device: str = "cpu"):
        self.num_centroids = num_centroids
        self.seed = seed
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )

        self.centroids = None
        self.balls: List[Ball] = []
        self.assignments = None
        self._rng = np.random.default_rng(self.seed)

    def build(self, keys: torch.Tensor) -> "RandomExactBallIndex":
        """Build the index from key vectors."""

        keys = keys.to(self.device)
        num_keys, dim = keys.shape

        torch.manual_seed(self.seed)
        self._rng = np.random.default_rng(self.seed)

        num_selected = min(self.num_centroids, num_keys)
        if num_selected == 0:
            self.centroids = torch.empty(0, dim, device=self.device)
            self.balls = []
            self.assignments = torch.empty(0, dtype=torch.long, device=self.device)
            return self

        indices = torch.randperm(num_keys)[:num_selected]
        centroids = keys[indices]
        self.centroids = centroids

        distances = torch.cdist(keys, centroids)
        self.assignments = torch.argmin(distances, dim=1)

        self.balls = []
        for centroid_idx in range(num_selected):
            mask = self.assignments == centroid_idx
            cluster_points = keys[mask]

            if cluster_points.numel() > 0:
                center, radius = self._compute_exact_ball(cluster_points)
                ball = Ball(center=center, radius=radius)
                self.balls.append(ball)
            else:
                centroid = centroids[centroid_idx]
                self.balls.append(Ball(center=centroid, radius=0.0))

        return self

    def _compute_exact_ball(
        self, cluster_points: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Compute exact smallest enclosing ball for the given cluster."""

        points_np = cluster_points.detach().cpu().numpy().astype(np.float64)
        center_np, radius = _minimum_enclosing_ball_numpy(points_np, self._rng)
        center = torch.tensor(center_np, dtype=cluster_points.dtype, device=self.device)
        return center, float(radius)

    def count_intersecting_balls(self, q: torch.Tensor, threshold: float) -> int:
        """Count how many balls intersect with halfspace {x: q^T x >= threshold}."""

        q = q.to(self.device)
        count = 0
        for ball in self.balls:
            if ball.intersects_halfspace(q, threshold):
                count += 1
        return count

    def get_intersection_percentage(self, q: torch.Tensor, threshold: float) -> float:
        """Get percentage of balls that intersect with the halfspace."""

        if len(self.balls) == 0:
            return 0.0
        count = self.count_intersecting_balls(q, threshold)
        return 100.0 * count / len(self.balls)
