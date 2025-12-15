"""
Random Ball Index: Random centroid selection with ball structures around centroids.
"""

import torch
import numpy as np
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


class RandomBallIndex:
    """
    Index based on random centroid selection with ball neighborhoods.

    Randomly select centroids from the data points, then for each centroid,
    assign remaining points to nearest centroid and create a ball.
    """

    def __init__(self, num_centroids: int = 100, seed: int = 42, device: str = "cpu"):
        """
        Initialize Random Ball Index.

        Args:
            num_centroids: Number of random centroids to select
            seed: Random seed for reproducibility
            device: Device to run on ('cpu' or 'cuda')
        """
        self.num_centroids = num_centroids
        self.seed = seed
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )

        self.centroids = None
        self.balls: List[Ball] = []
        self.assignments = None

    def build(self, keys: torch.Tensor) -> "RandomBallIndex":
        """
        Build the index from key vectors.

        Args:
            keys: Key vectors of shape [num_keys, dim]

        Returns:
            Self
        """
        keys = keys.to(self.device)
        num_keys, dim = keys.shape

        # Set random seed for reproducibility
        torch.manual_seed(self.seed)

        # Randomly select centroids (cap at available keys)
        num_selected = min(self.num_centroids, num_keys)
        if num_selected == 0:
            self.centroids = torch.empty(0, dim, device=self.device)
            self.balls = []
            self.assignments = torch.empty(0, dtype=torch.long, device=self.device)
            return self

        indices = torch.randperm(num_keys)[:num_selected]
        centroids = keys[indices]
        self.centroids = centroids

        # Assign each point to nearest centroid
        # Compute pairwise distances: [num_keys, num_centroids]
        distances = torch.cdist(keys, centroids)
        self.assignments = torch.argmin(distances, dim=1)

        # Build balls for each centroid
        self.balls = []
        for centroid_idx in range(num_selected):
            # Get points assigned to this centroid
            mask = self.assignments == centroid_idx
            cluster_points = keys[mask]

            if len(cluster_points) > 0:
                centroid = centroids[centroid_idx]

                # Compute radius as max distance from centroid
                distances_to_centroid = torch.norm(
                    cluster_points - centroid.unsqueeze(0), dim=1
                )
                radius = torch.max(distances_to_centroid).item()

                ball = Ball(center=centroid, radius=radius)
                self.balls.append(ball)
            else:
                # Empty cluster - create ball with zero radius
                ball = Ball(center=centroids[centroid_idx], radius=0.0)
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
