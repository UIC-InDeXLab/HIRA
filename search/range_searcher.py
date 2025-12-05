"""
Halfspace range searcher for efficient key selection using hierarchical indexes.

The HalfspaceSearcher performs halfspace range searches to identify keys whose
dot product with a query exceeds a threshold. It uses hierarchical pruning
with cluster radii to efficiently traverse the index.
"""

from typing import List, Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from ..index.index import KMeansIndex


class HalfspaceSearcher:
    """
    Halfspace range searcher using hierarchical pruning with cluster radii.

    This searcher identifies keys k where q·k >= τ (halfspace defined by
    query q and threshold τ). It uses the hierarchical index with cluster
    radii to prune clusters that cannot contain any qualifying keys.

    Search Strategy:
    1. Start from the coarsest level (top of hierarchy)
    2. For each cluster centroid, decide if children might intersect the halfspace
    3. Use cluster radius to make pruning decisions:
       - Compute centroid score: s = q·c
       - If s - radius >= τ, all points in cluster qualify (include all)
       - If s + radius < τ, no points in cluster qualify (prune)
       - Otherwise, recurse into children at finer level
    4. At the leaf level, check individual keys exactly

    Args:
        max_candidates: Maximum number of candidate keys to return (optional)
    """

    def __init__(
        self,
        max_candidates: Optional[int] = None,
    ):
        """
        Initialize the halfspace searcher.

        Args:
            max_candidates: Maximum number of keys to return (None for all qualifying keys)
        """
        self.max_candidates = max_candidates

    def search(
        self,
        query: torch.Tensor,
        threshold: float,
        index: "KMeansIndex",
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform halfspace range search using hierarchical pruning.

        Args:
            query: Query vector [head_dim]
            threshold: Score threshold τ
            index: KMeansIndex with hierarchy
            keys: All key vectors [num_keys, head_dim]

        Returns:
            Tensor of key indices where q·k >= τ
        """
        if index.num_levels() == 0:
            raise ValueError("Index has no levels")

        # Ensure query is on the same device as keys
        query = query.to(keys.device)

        # Normalize query if using spherical k-means
        if hasattr(index, "spherical") and index.spherical:
            query = torch.nn.functional.normalize(query, dim=0)

        # Start search from the coarsest level
        candidate_clusters = self._search_level(
            query=query,
            threshold=threshold,
            level_idx=0,
            index=index,
            keys=keys,
        )

        # Recursively refine through finer levels
        for level_idx in range(1, index.num_levels()):
            candidate_clusters = self._refine_search(
                query=query,
                threshold=threshold,
                level_idx=level_idx,
                parent_candidates=candidate_clusters,
                index=index,
                keys=keys,
            )

            # Early exit if no candidates remain
            if len(candidate_clusters) == 0:
                return torch.tensor([], dtype=torch.long, device=keys.device)

        # Get candidate key indices from finest level
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
        if (
            self.max_candidates is not None
            and len(qualifying_keys) > self.max_candidates
        ):
            # Return top-k by score
            scores = torch.matmul(keys[qualifying_keys], query)
            top_k_indices = torch.topk(scores, k=self.max_candidates, largest=True)[1]
            qualifying_keys = qualifying_keys[top_k_indices]

        return qualifying_keys

    def _search_level(
        self,
        query: torch.Tensor,
        threshold: float,
        level_idx: int,
        index: "KMeansIndex",
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Search a single level to find candidate clusters using cluster radii.

        Args:
            query: Query vector [head_dim]
            threshold: Score threshold
            level_idx: Index of this level
            index: KMeansIndex
            keys: All key vectors

        Returns:
            Tensor of cluster IDs that may contain qualifying keys
        """
        level = index.get_level(level_idx)
        ball_centers = level.cluster_centers.to(query.device)
        radii = level.cluster_radii.to(query.device)

        # Compute scores for ball centers: s = q·c
        center_scores = torch.matmul(ball_centers, query)  # [num_clusters]

        # Pruning decisions based on smallest enclosing ball:
        # - If s - radius >= threshold: all points qualify (mark for inclusion)
        # - If s + radius < threshold: no points qualify (prune)
        # - Otherwise: may contain qualifying points (need to recurse)

        # For non-leaf levels, we return clusters that may contain qualifying keys
        # This includes clusters where s + radius >= threshold
        candidate_mask = (center_scores + radii) >= threshold
        candidate_clusters = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)

        return candidate_clusters

    def _refine_search(
        self,
        query: torch.Tensor,
        threshold: float,
        level_idx: int,
        parent_candidates: torch.Tensor,
        index: "KMeansIndex",
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine search at a finer level given parent candidates.

        Args:
            query: Query vector [head_dim]
            threshold: Score threshold
            level_idx: Current (finer) level index
            parent_candidates: Cluster IDs from parent level
            index: KMeansIndex
            keys: All key vectors

        Returns:
            Tensor of cluster IDs at this level that may contain qualifying keys
        """
        if len(parent_candidates) == 0:
            return torch.tensor([], dtype=torch.long, device=query.device)

        level = index.get_level(level_idx)
        centroids = level.centroids.to(query.device)
        radii = level.cluster_radii.to(query.device)
        parent_assignments = level.parent_assignments.to(query.device)

        # Find clusters at this level that belong to candidate parent clusters
        child_mask = torch.isin(parent_assignments, parent_candidates)
        child_clusters = torch.nonzero(child_mask, as_tuple=False).squeeze(-1)

        if len(child_clusters) == 0:
            return torch.tensor([], dtype=torch.long, device=query.device)

        # Among these child clusters, prune those that can't contain qualifying keys
        child_centroids = centroids[child_clusters]
        child_radii = radii[child_clusters]
        centroid_scores = torch.matmul(child_centroids, query)

        # Keep clusters where s + radius >= threshold
        qualifying_mask = (centroid_scores + child_radii) >= threshold

        return child_clusters[qualifying_mask]

    def _get_keys_from_clusters(
        self,
        cluster_ids: torch.Tensor,
        level,
    ) -> torch.Tensor:
        """
        Get all key indices belonging to the given clusters.

        Args:
            cluster_ids: Cluster IDs [num_clusters]
            level: KMeansIndex.Level

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

    def batch_search(
        self,
        queries: torch.Tensor,
        thresholds: torch.Tensor,
        index: "KMeansIndex",
        keys: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Perform batch range search for multiple queries.

        Args:
            queries: Query vectors [num_queries, head_dim]
            thresholds: Thresholds for each query [num_queries]
            index: KMeansIndex
            keys: All key vectors [num_keys, head_dim]

        Returns:
            List of qualifying key index tensors, one per query
        """
        results = []
        for i in range(queries.shape[0]):
            query = queries[i]
            threshold = thresholds[i].item()
            result = self.search(query, threshold, index, keys)
            results.append(result)

        return results
