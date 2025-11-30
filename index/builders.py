"""
Abstract base classes and concrete implementations for building hierarchical indexes.

The IndexBuilder is responsible for constructing a multi-level hierarchical index
from a batch of key vectors. Different builders can use different clustering or
partitioning strategies (e.g., k-means, hierarchical k-means, PQ, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import torch
import torch.nn.functional as F


class IndexBuilder(ABC):
    """
    Abstract base class for hierarchical index builders.
    
    An IndexBuilder constructs a hierarchical index from key vectors, typically
    organizing them into a tree-like structure with multiple levels.
    """
    
    @abstractmethod
    def build(
        self,
        keys: torch.Tensor,
        num_levels: int,
        branching_factor: int,
        device: torch.device,
        **kwargs
    ) -> "HierarchicalIndex":
        """
        Build a hierarchical index from key vectors.
        
        Args:
            keys: Key vectors of shape [num_keys, head_dim]
            num_levels: Number of hirarchy levels to create
            branching_factor: Number of clusters per level (can vary by level)
            device: Device to build the index on
            **kwargs: Additional builder-specific parameters
            
        Returns:
            HierarchicalIndex: The constructed hierarchical index
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of this builder.
        
        Returns:
            Dictionary with builder configuration
        """
        pass


class KMeansIndexBuilder(IndexBuilder):
    """
    Hierarchical index builder using k-means clustering.
    
    This builder creates a hierarchical index by recursively applying k-means
    clustering at each level. At level 0, all keys are clustered into k groups.
    At level 1, each cluster from level 0 is further subdivided, and so on.
    
    Args:
        max_iterations: Maximum k-means iterations per level
        tolerance: Convergence tolerance for k-means
        init_method: Centroid initialization method ("kmeans++", "random")
        use_faiss: Whether to use FAISS for faster clustering (if available)
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        init_method: str = "kmeans++",
        use_faiss: bool = False,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.init_method = init_method
        self.use_faiss = use_faiss
        
        # TODO: Add FAISS support for faster clustering on large datasets
        if use_faiss:
            try:
                import faiss
                self.faiss_available = True
            except ImportError:
                print("Warning: FAISS not available, falling back to PyTorch k-means")
                self.faiss_available = False
        else:
            self.faiss_available = False
    
    def build(
        self,
        keys: torch.Tensor,
        num_levels: int,
        branching_factor: int,
        device: torch.device,
        **kwargs
    ) -> "HierarchicalIndex":
        """
        Build hierarchical index using recursive k-means clustering.
        
        Args:
            keys: Key vectors [num_keys, head_dim]
            num_levels: Number of hirarchy levels
            branching_factor: Number of clusters per level
            device: Device for computation
            **kwargs: Additional parameters
            
        Returns:
            HierarchicalIndex with k-means clustering at each level
        """
        from .structure import HierarchicalIndex, IndexLevel
        
        if keys.shape[0] == 0:
            raise ValueError("Cannot build index from empty key set")
        
        keys = keys.to(device)
        num_keys, head_dim = keys.shape
        
        # Storage for all levels
        levels = []
        
        # Level 0: Cluster all keys
        centroids_0, assignments_0 = self._kmeans(
            keys, min(branching_factor, num_keys), device
        )
        
        level_0 = IndexLevel(
            level_idx=0,
            centroids=centroids_0,
            assignments=assignments_0,
            parent_assignments=None,
            device=device,
        )
        levels.append(level_0)
        
        # Build subsequent levels by clustering within each parent cluster
        for level_idx in range(1, num_levels):
            parent_level = levels[level_idx - 1]
            parent_centroids = parent_level.centroids
            parent_assignments = parent_level.assignments
            
            # For each parent cluster, subdivide its members
            all_centroids = []
            all_assignments = []
            parent_of_child = []
            
            for parent_cluster_id in range(parent_centroids.shape[0]):
                # Get keys belonging to this parent cluster
                mask = parent_assignments == parent_cluster_id
                cluster_keys = keys[mask]
                
                if cluster_keys.shape[0] == 0:
                    continue
                
                # Cluster within this parent cluster
                num_subclusters = min(branching_factor, cluster_keys.shape[0])
                sub_centroids, sub_assignments = self._kmeans(
                    cluster_keys, num_subclusters, device
                )
                
                # Remap assignments to global indexing
                global_cluster_offset = len(all_centroids)
                sub_assignments_global = sub_assignments + global_cluster_offset
                
                all_centroids.append(sub_centroids)
                
                # Store the remapped assignments for keys in this parent cluster
                # Need to fill in the full assignment array
                full_assignments = torch.full(
                    (num_keys,), -1, dtype=torch.long, device=device
                )
                full_assignments[mask] = sub_assignments_global
                all_assignments.append(full_assignments)
                
                # Track which parent cluster each child cluster belongs to
                for _ in range(num_subclusters):
                    parent_of_child.append(parent_cluster_id)
            
            if len(all_centroids) == 0:
                # No more clusters to create, stop building levels
                break
            
            # Combine centroids from all parent clusters
            level_centroids = torch.cat(all_centroids, dim=0)
            
            # Combine assignments: take the first non-(-1) value for each key
            # (each key belongs to exactly one parent cluster)
            level_assignments = torch.full(
                (num_keys,), -1, dtype=torch.long, device=device
            )
            for assign_arr in all_assignments:
                mask = assign_arr != -1
                level_assignments[mask] = assign_arr[mask]
            
            # Create parent assignment mapping
            parent_of_child_tensor = torch.tensor(
                parent_of_child, dtype=torch.long, device=device
            )
            
            level = IndexLevel(
                level_idx=level_idx,
                centroids=level_centroids,
                assignments=level_assignments,
                parent_assignments=parent_of_child_tensor,
                device=device,
            )
            levels.append(level)
        
        # Create the hierarchical index
        index = HierarchicalIndex(
            levels=levels,
            num_keys=num_keys,
            head_dim=head_dim,
            device=device,
        )
        
        return index
    
    def _kmeans(
        self,
        data: torch.Tensor,
        num_clusters: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform k-means clustering on the given data.
        
        Args:
            data: Data points [num_points, dim]
            num_clusters: Number of clusters
            device: Device for computation
            
        Returns:
            Tuple of (centroids [num_clusters, dim], assignments [num_points])
        """
        # TODO: Integrate FAISS for large-scale clustering
        if self.faiss_available:
            return self._kmeans_faiss(data, num_clusters, device)
        else:
            return self._kmeans_torch(data, num_clusters, device)
    
    def _kmeans_torch(
        self,
        data: torch.Tensor,
        num_clusters: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        PyTorch-based k-means clustering.
        
        Args:
            data: Data points [num_points, dim]
            num_clusters: Number of clusters
            device: Device for computation
            
        Returns:
            Tuple of (centroids, assignments)
        """
        num_points, dim = data.shape
        
        if num_clusters >= num_points:
            # Degenerate case: one point per cluster
            centroids = data.clone()
            assignments = torch.arange(num_points, device=device, dtype=torch.long)
            return centroids, assignments
        
        # Initialize centroids
        if self.init_method == "kmeans++":
            centroids = self._kmeans_plusplus_init(data, num_clusters, device)
        else:
            # Random initialization
            indices = torch.randperm(num_points, device=device)[:num_clusters]
            centroids = data[indices].clone()
        
        # K-means iterations
        for iteration in range(self.max_iterations):
            # Assign points to nearest centroids
            # distances: [num_points, num_clusters]
            distances = torch.cdist(data, centroids, p=2)
            assignments = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(num_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centroids[k] = data[mask].mean(dim=0)
                else:
                    # Empty cluster: reinitialize with a random point
                    new_centroids[k] = data[torch.randint(num_points, (1,), device=device)]
            
            # Check convergence
            centroid_shift = torch.norm(new_centroids - centroids, dim=1).max()
            centroids = new_centroids
            
            if centroid_shift < self.tolerance:
                break
        
        # Final assignment
        distances = torch.cdist(data, centroids, p=2)
        assignments = torch.argmin(distances, dim=1)
        
        return centroids, assignments
    
    def _kmeans_plusplus_init(
        self,
        data: torch.Tensor,
        num_clusters: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        K-means++ initialization for better initial centroids.
        
        Args:
            data: Data points [num_points, dim]
            num_clusters: Number of clusters
            device: Device for computation
            
        Returns:
            Initial centroids [num_clusters, dim]
        """
        num_points = data.shape[0]
        centroids = []
        
        # Choose first centroid randomly
        first_idx = torch.randint(num_points, (1,), device=device).item()
        centroids.append(data[first_idx])
        
        # Choose remaining centroids
        for _ in range(1, num_clusters):
            # Compute distance to nearest centroid for each point
            centroid_tensor = torch.stack(centroids, dim=0)
            distances = torch.cdist(data, centroid_tensor, p=2)
            min_distances = distances.min(dim=1)[0]
            
            # Sample proportional to squared distance
            probabilities = min_distances ** 2
            probabilities = probabilities / probabilities.sum()
            
            next_idx = torch.multinomial(probabilities, 1).item()
            centroids.append(data[next_idx])
        
        return torch.stack(centroids, dim=0)
    
    def _kmeans_faiss(
        self,
        data: torch.Tensor,
        num_clusters: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        FAISS-based k-means clustering (faster for large datasets).
        
        TODO: Implement FAISS integration
        
        Args:
            data: Data points [num_points, dim]
            num_clusters: Number of clusters
            device: Device for computation
            
        Returns:
            Tuple of (centroids, assignments)
        """
        # TODO: Implement FAISS k-means
        # For now, fall back to PyTorch
        return self._kmeans_torch(data, num_clusters, device)
    
    def get_config(self) -> Dict[str, Any]:
        """Return builder configuration."""
        return {
            "builder_type": "kmeans",
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "init_method": self.init_method,
            "use_faiss": self.use_faiss,
        }
