"""
Random Convex Hull Index: Random centroid selection with convex hull structures around centroids.
"""

import torch
import numpy as np
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


class RandomConvexHullIndex:
    """
    Index based on random centroid selection with convex hull neighborhoods.
    
    Randomly select centroids from the data points, then for each centroid,
    assign remaining points to nearest centroid and create a convex hull.
    """
    
    def __init__(
        self,
        num_centroids: int = 100,
        seed: int = 42,
        device: str = "cpu"
    ):
        """
        Initialize Random Convex Hull Index.
        
        Args:
            num_centroids: Number of random centroids to select
            seed: Random seed for reproducibility
            device: Device to run on ('cpu' or 'cuda')
        """
        self.num_centroids = num_centroids
        self.seed = seed
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        
        self.centroids = None
        self.convex_hulls: List[ConvexHullStructure] = []
        self.assignments = None
        
    def build(self, keys: torch.Tensor) -> "RandomConvexHullIndex":
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
            self.convex_hulls = []
            self.assignments = torch.empty(0, dtype=torch.long, device=self.device)
            return self

        indices = torch.randperm(num_keys)[:num_selected]
        centroids = keys[indices]
        self.centroids = centroids
        
        # Assign each point to nearest centroid
        # Compute pairwise distances: [num_keys, num_centroids]
        distances = torch.cdist(keys, centroids)
        self.assignments = torch.argmin(distances, dim=1)
        
        # Build convex hulls for each centroid
        self.convex_hulls = []
        for centroid_idx in range(num_selected):
            # Get points assigned to this centroid
            mask = self.assignments == centroid_idx
            cluster_points = keys[mask]
            
            if len(cluster_points) > dim:  # Need at least dim+1 points for convex hull
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
            else:
                # Too few points, use the points themselves as vertices
                if len(cluster_points) > 0:
                    cluster_points_np = cluster_points.cpu().numpy()
                    hull = type('obj', (object,), {'points': cluster_points_np, 'vertices': np.arange(len(cluster_points))})()
                    hull_struct = ConvexHullStructure(hull=hull, points=cluster_points)
                    self.convex_hulls.append(hull_struct)
                else:
                    # Empty cluster - single point at centroid
                    centroid = centroids[centroid_idx]
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
