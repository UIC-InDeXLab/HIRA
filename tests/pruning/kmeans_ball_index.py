"""
K-Means Ball Index: Build k-means clustering and create ball structures around centroids.
"""

import torch
import numpy as np
import faiss
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class Ball:
    """Represents a ball in d-dimensional space."""

    center: torch.Tensor  # [dim]
    radius: float

    def intersects_halfspace(self, q: torch.Tensor, threshold: float) -> bool:
        """
        Check if ball intersects with halfspace {x: q^T x >= threshold}.

        Args:
            q: Unit query vector [dim]
            threshold: Halfspace threshold

        Returns:
            True if ball intersects halfspace
        """
        # Maximum dot product point in ball: center + radius * q
        max_dot = torch.dot(self.center, q) + self.radius
        return max_dot >= threshold


class KMeansBallIndex:
    """
    Index based on k-means clustering with ball neighborhoods.

    For each centroid, we create a ball that contains all points assigned to that centroid.
    """

    def __init__(
        self, num_clusters: int = 100, max_iterations: int = 100, device: str = "cpu"
    ):
        """
        Initialize K-Means Ball Index.

        Args:
            num_clusters: Number of k-means clusters
            max_iterations: Maximum iterations for k-means
            device: Device to run on ('cpu' or 'cuda')
        """
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )

        self.centroids = None
        self.balls: List[Ball] = []
        self.assignments = None

    def build(self, keys: torch.Tensor) -> "KMeansBallIndex":
        """
        Build the index from key vectors.

        Args:
            keys: Key vectors of shape [num_keys, dim]

        Returns:
            Self
        """
        keys = keys.to(self.device)
        num_keys, dim = keys.shape

        # Run k-means using FAISS
        keys_np = keys.cpu().float().numpy()

        kmeans = faiss.Kmeans(
            d=dim,
            k=self.num_clusters,
            niter=self.max_iterations,
            verbose=False,
            gpu=False,
        )

        kmeans.train(keys_np)

        # Get assignments
        _, assignments = kmeans.index.search(keys_np, 1)
        assignments = assignments.squeeze()

        # Get centroids
        centroids = torch.from_numpy(kmeans.centroids).to(self.device)
        self.centroids = centroids
        self.assignments = torch.from_numpy(assignments).to(self.device)

        # Build balls for each cluster
        self.balls = []
        for cluster_idx in range(self.num_clusters):
            # Get points in this cluster
            mask = self.assignments == cluster_idx
            cluster_points = keys[mask]

            if len(cluster_points) > 0:
                centroid = centroids[cluster_idx]

                # Compute radius as max distance from centroid
                distances = torch.norm(cluster_points - centroid.unsqueeze(0), dim=1)
                radius = torch.max(distances).item()

                ball = Ball(center=centroid, radius=radius)
                self.balls.append(ball)
            else:
                # Empty cluster - create ball with zero radius
                ball = Ball(center=centroids[cluster_idx], radius=0.0)
                self.balls.append(ball)

        return self

    def count_intersecting_balls(self, q: torch.Tensor, threshold: float) -> int:
        """
        Count how many balls intersect with halfspace {x: q^T x >= threshold}.

        Args:
            q: Unit query vector [dim]
            threshold: Halfspace threshold

        Returns:
            Number of intersecting balls
        """
        q = q.to(self.device)
        count = 0
        for ball in self.balls:
            if ball.intersects_halfspace(q, threshold):
                count += 1
        return count

    def get_intersection_percentage(self, q: torch.Tensor, threshold: float) -> float:
        """
        Get percentage of balls that intersect with halfspace.

        Args:
            q: Unit query vector [dim]
            threshold: Halfspace threshold

        Returns:
            Percentage (0-100) of intersecting balls
        """
        if len(self.balls) == 0:
            return 0.0
        count = self.count_intersecting_balls(q, threshold)
        return 100.0 * count / len(self.balls)
