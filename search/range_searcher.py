"""
Range searchers for efficient key selection using hierarchical indexes.

The RangeSearcher performs halfspace range searches to identify keys whose
dot product with a query exceeds a threshold. It uses hierarchical pruning
to avoid exhaustive searching.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from ..index.structure import HierarchicalIndex, IndexLevel


class RangeSearcher(ABC):
    """
    Abstract base class for range searching over hierarchical indexes.
    
    A RangeSearcher takes a query and a threshold and returns the set of
    key indices that satisfy the range condition (e.g., q·k >= τ).
    """
    
    @abstractmethod
    def search(
        self,
        query: torch.Tensor,
        threshold: float,
        index: HierarchicalIndex,
        keys: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform range search to find keys satisfying the condition.
        
        Args:
            query: Query vector [head_dim]
            threshold: Score threshold
            index: HierarchicalIndex to search
            keys: All key vectors [num_keys, head_dim]
            **kwargs: Additional search parameters
            
        Returns:
            Tensor of key indices satisfying the condition
        """
        pass


class HalfspaceRangeSearcher(RangeSearcher):
    """
    Halfspace range searcher using hierarchical pruning.
    
    This searcher identifies keys k where q·k >= τ (halfspace defined by
    query q and threshold τ). It uses the hierarchical index to prune
    clusters that cannot contain any qualifying keys.
    
    Strategy:
    1. At the coarsest level, compute bounds on q·k for each cluster
    2. Prune clusters where max(q·k) < τ (no keys in cluster can qualify)
    3. Recurse into qualifying clusters at finer levels
    4. At the leaf level, check individual keys
    
    TODO: Optimize with custom CUDA kernels for coarse-level pruning
    TODO: Add support for approximate search (return top-k instead of threshold)
    TODO: Implement cluster caching for repeated queries
    
    Args:
        use_bounds: Whether to use centroid-based bounds for pruning
        max_candidates: Maximum number of candidate keys to return
        early_stopping: Stop when max_candidates are found
    """
    
    def __init__(
        self,
        use_bounds: bool = True,
        max_candidates: Optional[int] = None,
        early_stopping: bool = False,
    ):
        self.use_bounds = use_bounds
        self.max_candidates = max_candidates
        self.early_stopping = early_stopping
    
    def search(
        self,
        query: torch.Tensor,
        threshold: float,
        index: HierarchicalIndex,
        keys: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform halfspace range search using hierarchical pruning.
        
        Args:
            query: Query vector [head_dim]
            threshold: Score threshold τ
            index: HierarchicalIndex
            keys: All key vectors [num_keys, head_dim]
            **kwargs: Additional parameters
            
        Returns:
            Tensor of key indices where q·k >= τ
        """
        if index.num_levels() == 0:
            raise ValueError("Index has no levels")
        
        query = query.to(keys.device)
        
        # Start search from the coarsest level
        candidate_clusters = self._search_level(
            query=query,
            threshold=threshold,
            level=index.get_level(0),
            index=index,
            level_idx=0,
        )
        
        # Recursively refine through finer levels
        for level_idx in range(1, index.num_levels()):
            level = index.get_level(level_idx)
            candidate_clusters = self._refine_search(
                query=query,
                threshold=threshold,
                level=level,
                parent_candidates=candidate_clusters,
                index=index,
            )
        
        # Get candidate key indices from finest level
        if len(candidate_clusters) == 0:
            return torch.tensor([], dtype=torch.long, device=keys.device)
        
        candidate_keys = self._get_keys_from_clusters(
            cluster_ids=candidate_clusters,
            level=index.get_level(index.num_levels() - 1),
        )
        
        # Final filtering: compute exact scores and filter by threshold
        qualifying_keys = self._exact_filter(
            query=query,
            threshold=threshold,
            candidate_keys=candidate_keys,
            keys=keys,
        )
        
        # Apply max_candidates limit if specified
        if self.max_candidates is not None and len(qualifying_keys) > self.max_candidates:
            # Return top-k by score
            scores = torch.matmul(keys[qualifying_keys], query)
            top_k_indices = torch.topk(scores, k=self.max_candidates, largest=True)[1]
            qualifying_keys = qualifying_keys[top_k_indices]
        
        return qualifying_keys
    
    def _search_level(
        self,
        query: torch.Tensor,
        threshold: float,
        level: IndexLevel,
        index: HierarchicalIndex,
        level_idx: int,
    ) -> torch.Tensor:
        """
        Search a single level to find candidate clusters.
        
        Args:
            query: Query vector [head_dim]
            threshold: Score threshold
            level: IndexLevel to search
            index: Parent HierarchicalIndex
            level_idx: Index of this level
            
        Returns:
            Tensor of cluster IDs that may contain qualifying keys
        """
        centroids = level.centroids.to(query.device)
        
        if not self.use_bounds:
            # Conservative: consider all clusters
            return torch.arange(centroids.shape[0], device=query.device, dtype=torch.long)
        
        # Compute scores for centroids
        centroid_scores = torch.matmul(centroids, query)  # [num_clusters]
        
        # TODO: Compute bounds more precisely using cluster radius
        # For now, use a simple heuristic: include cluster if centroid score is close to threshold
        # A more sophisticated approach would compute the maximum possible score
        # for any key in the cluster: max_score = centroid_score + cluster_radius
        
        # Heuristic: use a margin based on a rough estimate of cluster spread
        # This is conservative - we may include some clusters that don't qualify
        # TODO: Precompute and store cluster radii during index building
        margin = self._estimate_cluster_margin(level, index)
        
        # Include clusters where centroid_score + margin >= threshold
        candidate_mask = (centroid_scores + margin) >= threshold
        candidate_clusters = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        
        return candidate_clusters
    
    def _refine_search(
        self,
        query: torch.Tensor,
        threshold: float,
        level: IndexLevel,
        parent_candidates: torch.Tensor,
        index: HierarchicalIndex,
    ) -> torch.Tensor:
        """
        Refine search at a finer level given parent candidates.
        
        Args:
            query: Query vector [head_dim]
            threshold: Score threshold
            level: Current (finer) level
            parent_candidates: Cluster IDs from parent level
            index: Parent HierarchicalIndex
            
        Returns:
            Tensor of cluster IDs at this level that may contain qualifying keys
        """
        if len(parent_candidates) == 0:
            return torch.tensor([], dtype=torch.long, device=query.device)
        
        centroids = level.centroids.to(query.device)
        parent_assignments = level.parent_assignments.to(query.device)
        
        # Find clusters at this level that belong to candidate parent clusters
        child_mask = torch.isin(parent_assignments, parent_candidates)
        child_clusters = torch.nonzero(child_mask, as_tuple=False).squeeze(-1)
        
        if len(child_clusters) == 0:
            return torch.tensor([], dtype=torch.long, device=query.device)
        
        # Among these child clusters, prune those that can't contain qualifying keys
        child_centroids = centroids[child_clusters]
        centroid_scores = torch.matmul(child_centroids, query)
        
        margin = self._estimate_cluster_margin(level, index)
        qualifying_mask = (centroid_scores + margin) >= threshold
        
        return child_clusters[qualifying_mask]
    
    def _get_keys_from_clusters(
        self,
        cluster_ids: torch.Tensor,
        level: IndexLevel,
    ) -> torch.Tensor:
        """
        Get all key indices belonging to the given clusters.
        
        Args:
            cluster_ids: Cluster IDs [num_clusters]
            level: IndexLevel
            
        Returns:
            Tensor of key indices [num_candidate_keys]
        """
        if len(cluster_ids) == 0:
            return torch.tensor([], dtype=torch.long, device=cluster_ids.device)
        
        assignments = level.assignments.to(cluster_ids.device)
        
        # Find all keys assigned to any of the candidate clusters
        mask = torch.isin(assignments, cluster_ids)
        key_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        
        return key_indices
    
    def _exact_filter(
        self,
        query: torch.Tensor,
        threshold: float,
        candidate_keys: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform exact filtering on candidate keys.
        
        Args:
            query: Query vector [head_dim]
            threshold: Score threshold
            candidate_keys: Candidate key indices [num_candidates]
            keys: All key vectors [num_keys, head_dim]
            
        Returns:
            Tensor of key indices that exactly satisfy q·k >= τ
        """
        if len(candidate_keys) == 0:
            return torch.tensor([], dtype=torch.long, device=keys.device)
        
        # Compute exact scores
        candidate_key_vectors = keys[candidate_keys]
        scores = torch.matmul(candidate_key_vectors, query)
        
        # Filter by threshold
        qualifying_mask = scores >= threshold
        qualifying_keys = candidate_keys[qualifying_mask]
        
        return qualifying_keys
    
    def _estimate_cluster_margin(
        self,
        level: IndexLevel,
        index: HierarchicalIndex,
    ) -> float:
        """
        Estimate a margin for cluster bounds.
        
        This is a rough heuristic to account for cluster spread. A better
        approach would precompute and store cluster radii during index building.
        
        TODO: Precompute cluster radii and store in IndexLevel.metadata
        TODO: Use more sophisticated bounds (e.g., ball bounds, cone bounds)
        
        Args:
            level: IndexLevel
            index: Parent HierarchicalIndex
            
        Returns:
            Estimated margin value
        """
        # Simple heuristic: use a fraction of the average centroid norm
        # This is very conservative and can be improved
        centroid_norms = torch.norm(level.centroids, dim=1)
        avg_norm = centroid_norms.mean().item()
        
        # Use 20% of average norm as margin (this is a rough guess)
        # TODO: Make this configurable or learn from data
        margin = 0.2 * avg_norm
        
        return margin
    
    def batch_search(
        self,
        queries: torch.Tensor,
        thresholds: torch.Tensor,
        index: HierarchicalIndex,
        keys: torch.Tensor,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Perform batch range search for multiple queries.
        
        TODO: Optimize with batched operations instead of looping
        
        Args:
            queries: Query vectors [num_queries, head_dim]
            thresholds: Thresholds for each query [num_queries]
            index: HierarchicalIndex
            keys: All key vectors [num_keys, head_dim]
            **kwargs: Additional parameters
            
        Returns:
            List of qualifying key index tensors, one per query
        """
        results = []
        for i in range(queries.shape[0]):
            query = queries[i]
            threshold = thresholds[i].item()
            result = self.search(query, threshold, index, keys, **kwargs)
            results.append(result)
        
        return results
