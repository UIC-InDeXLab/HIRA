"""
PQ Convex Hull Index: Build PQ encoding and create convex hull structures around centroids.
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


class PQConvexHullIndex:
    """
    Index based on Product Quantization with convex hull neighborhoods.
    
    For each centroid in the PQ codebook, we create a convex hull that contains 
    all points assigned to that centroid.
    """
    
    def __init__(
        self,
        M: int = 8,
        nbits: int = 8,
        device: str = "cpu"
    ):
        """
        Initialize PQ Convex Hull Index.
        
        Args:
            M: Number of subquantizers (dimension will be divided into M subspaces)
            nbits: Number of bits per subquantizer (2^nbits centroids per subspace)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.M = M
        self.nbits = nbits
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        
        self.pq_index = None
        self.codes = None
        self.convex_hulls: List[ConvexHullStructure] = []
        self.unique_codes = None
        
    def build(self, keys: torch.Tensor) -> "PQConvexHullIndex":
        """
        Build the index from key vectors.
        
        Args:
            keys: Key vectors of shape [num_keys, dim]
            
        Returns:
            Self
        """
        keys = keys.to(self.device)
        num_keys, dim = keys.shape
        
        # Train PQ index using FAISS
        keys_np = keys.cpu().float().numpy()
        
        self.pq_index = faiss.IndexPQ(dim, self.M, self.nbits)
        self.pq_index.train(keys_np)
        self.pq_index.add(keys_np)
        
        # Extract packed codes: shape (ntotal, code_size)
        packed = faiss.vector_to_array(self.pq_index.codes).astype(np.uint8)
        packed = packed.reshape(self.pq_index.ntotal, self.pq_index.code_size)

        # Unpack to per-subquantizer codes: shape (ntotal, M)
        codes_np = self.unpack_pq_codes(packed, self.M, self.nbits)

        self.codes = torch.from_numpy(codes_np).to(self.device)
        
        # Get unique codes (centroids in codebook space)
        self.unique_codes, inverse_indices = torch.unique(
            self.codes, dim=0, return_inverse=True
        )
        
        # Build convex hulls for each unique code
        self.convex_hulls = []
        
        for code_idx in range(len(self.unique_codes)):
            # Get points with this code
            mask = inverse_indices == code_idx
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
                    # Empty cluster - single point
                    zero_point = torch.zeros(dim, device=self.device).unsqueeze(0).cpu().numpy()
                    hull = type('obj', (object,), {'points': zero_point, 'vertices': np.array([0])})()
                    hull_struct = ConvexHullStructure(hull=hull, points=torch.zeros(1, dim, device=self.device))
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

    def unpack_pq_codes(self, packed: np.ndarray, M: int, nbits: int) -> np.ndarray:
        """
        packed: uint8 array of shape (N, code_size) where code_size = ceil(M*nbits/8)
        returns: int32 array of shape (N, M) with values in [0, 2^nbits)
        """
        assert packed.dtype == np.uint8
        N, code_size = packed.shape
        total_bits = code_size * 8

        bits = np.unpackbits(packed, axis=1, bitorder="little")  # (N, total_bits)
        bits = bits[:, : M * nbits]  # keep only used bits
        bits = bits.reshape(N, M, nbits)  # (N, M, nbits)

        weights = (1 << np.arange(nbits, dtype=np.int32))[None, None, :]  # (1,1,nbits)
        codes = (bits.astype(np.int32) * weights).sum(axis=2)  # (N, M)
        return codes
