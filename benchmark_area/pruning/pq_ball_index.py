"""
PQ Ball Index: Build PQ encoding and create ball structures around centroids.
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


class PQBallIndex:
    """
    Index based on Product Quantization with ball neighborhoods.
    
    For each centroid in the PQ codebook, we create a ball that contains 
    all points assigned to that centroid.
    """
    
    def __init__(
        self,
        M: int = 8,
        nbits: int = 8,
        device: str = "cpu"
    ):
        """
        Initialize PQ Ball Index.
        
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
        self.balls: List[Ball] = []
        self.unique_codes = None
        
    def build(self, keys: torch.Tensor) -> "PQBallIndex":
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
        
        # Build balls for each unique code
        self.balls = []
        print("Number of unique PQ codes (balls):", len(self.unique_codes))
        
        for code_idx in range(len(self.unique_codes)):
            # Get points with this code
            mask = inverse_indices == code_idx
            cluster_points = keys[mask]
            
            if len(cluster_points) > 0:
                # Compute centroid and radius
                centroid = torch.mean(cluster_points, dim=0)
                distances = torch.norm(cluster_points - centroid.unsqueeze(0), dim=1)
                radius = torch.max(distances).item()
                
                ball = Ball(center=centroid, radius=radius)
                self.balls.append(ball)
            else:
                # Empty cluster - create ball with zero radius
                zero_point = torch.zeros(dim, device=self.device)
                ball = Ball(center=zero_point, radius=0.0)
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
