"""
K-Means Hyperrectangle Index: Build k-means clustering and create hyperrectangle structures around centroids.
"""

import torch
import numpy as np
import faiss
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


class KMeansHyperrectangleIndex:
    """
    Index based on k-means clustering with hyperrectangle neighborhoods.
    
    For each centroid, we create a hyperrectangle that contains all points assigned to that centroid.
    """
    
    def __init__(
        self,
        num_clusters: int = 100,
        max_iterations: int = 100,
        device: str = "cpu"
    ):
        """
        Initialize K-Means Hyperrectangle Index.
        
        Args:
            num_clusters: Number of k-means clusters
            max_iterations: Maximum iterations for k-means
            device: Device to run on ('cpu' or 'cuda')
        """
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        
        self.centroids = None
        self.hyperrectangles: List[Hyperrectangle] = []
        self.assignments = None
        
    def build(self, keys: torch.Tensor) -> "KMeansHyperrectangleIndex":
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
        
        # Build hyperrectangles for each cluster
        self.hyperrectangles = []
        for cluster_idx in range(self.num_clusters):
            # Get points in this cluster
            mask = self.assignments == cluster_idx
            cluster_points = keys[mask]
            
            if len(cluster_points) > 0:
                # Compute bounding box
                min_bounds = torch.min(cluster_points, dim=0)[0]
                max_bounds = torch.max(cluster_points, dim=0)[0]
                
                hyperrect = Hyperrectangle(min_bounds=min_bounds, max_bounds=max_bounds)
                self.hyperrectangles.append(hyperrect)
            else:
                # Empty cluster - create degenerate hyperrectangle
                centroid = centroids[cluster_idx]
                hyperrect = Hyperrectangle(min_bounds=centroid, max_bounds=centroid)
                self.hyperrectangles.append(hyperrect)
        
        return self
    
    def count_intersecting_hyperrectangles(self, q: torch.Tensor, threshold: float) -> int:
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
