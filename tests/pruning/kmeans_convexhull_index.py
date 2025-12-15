"""
K-Means Convex Hull Index: Build k-means clustering and create convex hull structures around centroids.
"""

import torch
import numpy as np
import faiss
from typing import List
from dataclasses import dataclass
from scipy.spatial import ConvexHull


@dataclass
class ConvexHullStructure:
    """Represents a convex hull in d-dimensional space."""
    hull: ConvexHull  # scipy ConvexHull object
    points: torch.Tensor  # Original points [num_points, dim]
    
    def intersects_halfspace(self, q: torch.Tensor, threshold: float) -> bool:
        """
        Check if convex hull intersects with halfspace {x: q^T x >= threshold}.
        
        Args:
            q: Unit query vector [dim]
            threshold: Halfspace threshold
            
        Returns:
            True if convex hull intersects halfspace
        """
        # Maximum dot product is at one of the vertices of the convex hull
        q_np = q.cpu().numpy()
        dots = np.dot(self.hull.points[self.hull.vertices], q_np)
        max_dot = np.max(dots)
        return max_dot >= threshold


class KMeansConvexHullIndex:
    """
    Index based on k-means clustering with convex hull neighborhoods.
    
    For each centroid, we create a convex hull of all points assigned to that centroid.
    """
    
    def __init__(
        self,
        num_clusters: int = 100,
        max_iterations: int = 100,
        device: str = "cpu"
    ):
        """
        Initialize K-Means Convex Hull Index.
        
        Args:
            num_clusters: Number of k-means clusters
            max_iterations: Maximum iterations for k-means
            device: Device to run on ('cpu' or 'cuda')
        """
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        
        self.centroids = None
        self.convex_hulls: List[ConvexHullStructure] = []
        self.assignments = None
        
    def build(self, keys: torch.Tensor) -> "KMeansConvexHullIndex":
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
            gpu=False
        )
        
        kmeans.train(keys_np)
        
        # Get assignments
        _, assignments = kmeans.index.search(keys_np, 1)
        assignments = assignments.squeeze()
        
        # Get centroids
        centroids = torch.from_numpy(kmeans.centroids).to(self.device)
        self.centroids = centroids
        self.assignments = torch.from_numpy(assignments).to(self.device)
        
        # Build convex hulls for each cluster
        self.convex_hulls = []
        for cluster_idx in range(self.num_clusters):
            # Get points in this cluster
            mask = self.assignments == cluster_idx
            cluster_points = keys[mask]
            
            # Need at least dim+1 points for a proper convex hull
            if len(cluster_points) <= dim:
                # Too few points, use the points themselves as vertices
                if len(cluster_points) > 0:
                    cluster_points_np = cluster_points.cpu().numpy()
                    hull = type('obj', (object,), {'points': cluster_points_np, 'vertices': np.arange(len(cluster_points))})()
                    hull_struct = ConvexHullStructure(hull=hull, points=cluster_points)
                    self.convex_hulls.append(hull_struct)
                else:
                    # Empty cluster - single point at centroid
                    centroid = centroids[cluster_idx]
                    single_point = centroid.unsqueeze(0).cpu().numpy()
                    hull = type('obj', (object,), {'points': single_point, 'vertices': np.array([0])})()
                    hull_struct = ConvexHullStructure(hull=hull, points=centroid.unsqueeze(0))
                    self.convex_hulls.append(hull_struct)
            else:
                try:
                    cluster_points_np = cluster_points.cpu().numpy()
                    hull = ConvexHull(cluster_points_np)
                    hull_struct = ConvexHullStructure(hull=hull, points=cluster_points)
                    self.convex_hulls.append(hull_struct)
                except Exception:
                    # If convex hull fails, fall back to single point
                    centroid = torch.mean(cluster_points, dim=0)
                    single_point = centroid.unsqueeze(0).cpu().numpy()
                    hull = type('obj', (object,), {'points': single_point, 'vertices': np.array([0])})()
                    hull_struct = ConvexHullStructure(hull=hull, points=centroid.unsqueeze(0))
                    self.convex_hulls.append(hull_struct)
        
        return self
    
    def count_intersecting_hulls(self, q: torch.Tensor, threshold: float) -> int:
        """
        Count how many convex hulls intersect with halfspace {x: q^T x >= threshold}.
        
        Args:
            q: Unit query vector [dim]
            threshold: Halfspace threshold
            
        Returns:
            Number of intersecting convex hulls
        """
        q = q.to(self.device)
        count = 0
        for hull_struct in self.convex_hulls:
            if hull_struct.intersects_halfspace(q, threshold):
                count += 1
        return count
    
    def get_intersection_percentage(self, q: torch.Tensor, threshold: float) -> float:
        """
        Get percentage of convex hulls that intersect with halfspace.
        
        Args:
            q: Unit query vector [dim]
            threshold: Halfspace threshold
            
        Returns:
            Percentage (0-100) of intersecting convex hulls
        """
        if len(self.convex_hulls) == 0:
            return 0.0
        count = self.count_intersecting_hulls(q, threshold)
        return 100.0 * count / len(self.convex_hulls)
