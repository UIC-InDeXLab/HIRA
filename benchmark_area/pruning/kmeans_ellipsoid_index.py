"""
K-Means Ellipsoid Index: build k-means clusters with minimum-volume enclosing ellipsoids.
"""

import numpy as np
import torch
import faiss
from typing import List

from .ellipsoid_utils import Ellipsoid, compute_ellipsoid_tensors


class KMeansEllipsoidIndex:
    """Index using k-means clustering with minimum-volume enclosing ellipsoids."""

    def __init__(
        self,
        num_clusters: int = 100,
        max_iterations: int = 100,
        seed: int = 42,
        ellipsoid_tol: float = 1e-5,
        ellipsoid_max_iter: int = 1000,
        device: str = "cpu",
    ):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.seed = seed
        self.ellipsoid_tol = ellipsoid_tol
        self.ellipsoid_max_iter = ellipsoid_max_iter
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )

        self.centroids = None
        self.ellipsoids: List[Ellipsoid] = []
        self.assignments = None
        self._rng = np.random.default_rng(self.seed)

    def build(self, keys: torch.Tensor) -> "KMeansEllipsoidIndex":
        """Build the index from key vectors."""

        keys = keys.to(self.device)
        num_keys, dim = keys.shape

        keys_np = keys.cpu().float().numpy()

        kmeans = faiss.Kmeans(
            d=dim,
            k=self.num_clusters,
            niter=self.max_iterations,
            verbose=False,
            gpu=False,
        )
        kmeans.seed = self.seed
        kmeans.train(keys_np)

        _, assignments = kmeans.index.search(keys_np, 1)
        assignments = assignments.squeeze()

        centroids = torch.from_numpy(kmeans.centroids).to(self.device)
        self.centroids = centroids
        self.assignments = torch.from_numpy(assignments).to(self.device)

        self._rng = np.random.default_rng(self.seed)

        self.ellipsoids = []
        for cluster_idx in range(self.num_clusters):
            mask = self.assignments == cluster_idx
            cluster_points = keys[mask]

            if cluster_points.numel() > 0:
                center, inv_matrix = compute_ellipsoid_tensors(
                    cluster_points,
                    self._rng,
                    tol=self.ellipsoid_tol,
                    max_iter=self.ellipsoid_max_iter,
                )
                ellipsoid = Ellipsoid(center=center, inv_shape_matrix=inv_matrix)
                self.ellipsoids.append(ellipsoid)
            else:
                centroid = centroids[cluster_idx]
                zero_matrix = torch.zeros(
                    dim, dim, dtype=keys.dtype, device=self.device
                )
                ellipsoid = Ellipsoid(center=centroid, inv_shape_matrix=zero_matrix)
                self.ellipsoids.append(ellipsoid)

        return self

    def count_intersecting_ellipsoids(self, q: torch.Tensor, threshold: float) -> int:
        """Count ellipsoids intersecting the halfspace {x: q^T x >= threshold}."""

        q = q.to(self.device)
        count = 0
        for ellipsoid in self.ellipsoids:
            if ellipsoid.intersects_halfspace(q, threshold):
                count += 1
        return count

    def get_intersection_percentage(self, q: torch.Tensor, threshold: float) -> float:
        """Return percentage of ellipsoids intersecting the halfspace."""

        if len(self.ellipsoids) == 0:
            return 0.0
        count = self.count_intersecting_ellipsoids(q, threshold)
        return 100.0 * count / len(self.ellipsoids)
