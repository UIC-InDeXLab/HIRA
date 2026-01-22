"""
Random Hyperrectangle Index: Random centroid selection with hyperrectangle structures around centroids.
"""

import torch
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class Hyperrectangle:
    """Represents an axis-aligned hyperrectangle (bounding box) in d-dimensional space."""

    min_bounds: torch.Tensor  # [dim]
    max_bounds: torch.Tensor  # [dim]

    def intersects_halfspace(self, q: torch.Tensor, threshold: float) -> bool:
        """
        Check if hyperrectangle intersects with halfspace {x: q^T x >= threshold}.

        Args:
            q: Unit query vector [dim]
            threshold: Halfspace threshold

        Returns:
            True if hyperrectangle intersects halfspace
        """
        # Find the farthest corner of the hyperrectangle in direction q
        # For each dimension, choose max_bound if q[i] > 0, else min_bound
        farthest_corner = torch.where(q >= 0, self.max_bounds, self.min_bounds)

        # Maximum dot product in the hyperrectangle
        max_dot = torch.dot(farthest_corner, q)
        return max_dot >= threshold


class RandomHyperrectangleIndex:
    """
    Index based on random centroid selection with hyperrectangle neighborhoods.

    Randomly select centroids from the data points, then for each centroid,
    assign remaining points to nearest centroid and create a hyperrectangle.
    """

    def __init__(self, num_centroids: int = 100, seed: int = 42, device: str = "cpu"):
        """
        Initialize Random Hyperrectangle Index.

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
        self.hyperrectangles: List[Hyperrectangle] = []
        self.assignments = None

    def build(self, keys: torch.Tensor) -> "RandomHyperrectangleIndex":
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
            self.hyperrectangles = []
            self.assignments = torch.empty(0, dtype=torch.long, device=self.device)
            return self

        indices = torch.randperm(num_keys)[:num_selected]
        centroids = keys[indices]
        self.centroids = centroids

        # Assign each point to nearest centroid
        # Compute pairwise distances: [num_keys, num_centroids]
        distances = torch.cdist(keys, centroids)
        self.assignments = torch.argmin(distances, dim=1)

        # Build hyperrectangles for each centroid
        self.hyperrectangles = []
        for centroid_idx in range(num_selected):
            # Get points assigned to this centroid
            mask = self.assignments == centroid_idx
            cluster_points = keys[mask]

            if len(cluster_points) > 0:
                # Compute bounding box
                min_bounds = torch.min(cluster_points, dim=0)[0]
                max_bounds = torch.max(cluster_points, dim=0)[0]

                hyperrect = Hyperrectangle(min_bounds=min_bounds, max_bounds=max_bounds)
                self.hyperrectangles.append(hyperrect)
            else:
                # Empty cluster - create degenerate hyperrectangle
                centroid = centroids[centroid_idx]
                hyperrect = Hyperrectangle(min_bounds=centroid, max_bounds=centroid)
                self.hyperrectangles.append(hyperrect)

        return self

    def count_intersecting_hyperrectangles(
        self, q: torch.Tensor, threshold: float
    ) -> int:
        """
        Count how many hyperrectangles intersect with halfspace {x: q^T x >= threshold}.

        Args:
            q: Unit query vector [dim]
            threshold: Halfspace threshold

        Returns:
            Number of intersecting hyperrectangles
        """
        q = q.to(self.device)
        count = 0
        for hyperrect in self.hyperrectangles:
            if hyperrect.intersects_halfspace(q, threshold):
                count += 1
        return count

    def get_intersection_percentage(self, q: torch.Tensor, threshold: float) -> float:
        """
        Get percentage of hyperrectangles that intersect with halfspace.

        Args:
            q: Unit query vector [dim]
            threshold: Halfspace threshold

        Returns:
            Percentage (0-100) of intersecting hyperrectangles
        """
        if len(self.hyperrectangles) == 0:
            return 0.0
        count = self.count_intersecting_hyperrectangles(q, threshold)
        return 100.0 * count / len(self.hyperrectangles)
