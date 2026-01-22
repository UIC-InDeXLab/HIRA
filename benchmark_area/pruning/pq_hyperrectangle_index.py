"""
PQ Hyperrectangle Index: Build PQ encoding and create hyperrectangle structures around centroids.
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


class PQHyperrectangleIndex:
    """
    Index based on Product Quantization with hyperrectangle neighborhoods.

    For each centroid in the PQ codebook, we create a hyperrectangle that contains
    all points assigned to that centroid.
    """

    def __init__(self, M: int = 8, nbits: int = 8, device: str = "cpu"):
        """
        Initialize PQ Hyperrectangle Index.

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
        self.hyperrectangles: List[Hyperrectangle] = []
        self.unique_codes = None

    def build(self, keys: torch.Tensor) -> "PQHyperrectangleIndex":
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

        # # Extract codes
        # codes_np = faiss.vector_to_array(self.pq_index.codes).reshape(
        #     self.pq_index.ntotal, self.pq_index.pq.M
        # )
        # self.codes = torch.from_numpy(codes_np).to(self.device)

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

        # Build hyperrectangles for each unique code
        self.hyperrectangles = []
        # print("Number of unique PQ codes (hyperrectangles):", len(self.unique_codes))
        for code_idx in range(len(self.unique_codes)):
            # Get points with this code
            mask = inverse_indices == code_idx
            cluster_points = keys[mask]

            if len(cluster_points) > 0:
                # Compute bounding box

                # random subset from cluster_points
                # cluster_points = cluster_points[
                #     torch.randperm(len(cluster_points))[:1]
                # ]

                min_bounds = torch.min(cluster_points, dim=0)[0]
                max_bounds = torch.max(cluster_points, dim=0)[0]

                hyperrect = Hyperrectangle(min_bounds=min_bounds, max_bounds=max_bounds)
                self.hyperrectangles.append(hyperrect)
            else:
                # Empty cluster - create degenerate hyperrectangle
                zero_point = torch.zeros(dim, device=self.device)
                hyperrect = Hyperrectangle(min_bounds=zero_point, max_bounds=zero_point)
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
