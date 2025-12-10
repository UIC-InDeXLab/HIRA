"""
Halfspace range searcher for efficient key selection using hierarchical indexes.

The HalfspaceSearcher performs halfspace range searches to identify keys whose
dot product with a query exceeds a threshold. It uses hierarchical pruning
with cluster radii to efficiently traverse the index.
"""

from typing import List, Optional, TYPE_CHECKING
import torch
from ..index.index import KMeansIndex


class HalfspaceSearcher:
    def __init__(
        self,
        enable_profiling: bool = False,
    ):
        self.enable_profiling = enable_profiling
        self.reset_stats()

    def reset_stats(self):
        """Reset profiling statistics"""
        self.stats = {
            "num_distance_computations": 0,
            "num_clusters_visited": 0,
            "num_clusters_pruned": 0,
            "num_clusters_fully_included": 0,
            "time_per_level": [],
            "time_centroid_scoring": 0,
            "time_radius_checking": 0,
            "time_exact_filtering": 0,
            "time_data_movement": 0,
            "num_keys_per_level": [],
            "pruning_rate_per_level": [],
            "clusters_examined_per_level": [],
            "clusters_kept_per_level": [],
        }

    def search(self, query, threshold, index: "KMeansIndex"):
        """
        Args:
            threshold (_type_): The points with x.q >= threshold are returned
        """
        # normalize query
        query = query / torch.norm(query, p=2)

        # search root
        root = index.levels[-1]
        scores = torch.matmul(root.key_centers, query)
        mask = (scores + root.key_radii) >= threshold
        intersecting_cluster_ptrs = root.key_ptrs[mask]

        # recursively search down the tree
        levels = index.levels[::-1][1:]  # skip root, reverse order
        for level in levels:
            # find children
            children_indexes = []
            for parent_ptr in intersecting_cluster_ptrs.tolist():
                if parent_ptr in level.cluster_assignments:
                    children_indexes.append(level.cluster_assignments[parent_ptr])
            children_indexes = (
                torch.cat(children_indexes)
                if children_indexes
                else torch.tensor([], device=query.device, dtype=torch.long)
            )

            # get centers and radii
            child_centers = level.key_centers[children_indexes]
            child_radii = level.key_radii[children_indexes]
            pointers = level.key_ptrs[children_indexes]

            # scores
            scores = torch.matmul(child_centers, query)
            mask = (scores + child_radii) >= threshold
            intersecting_cluster_ptrs = pointers[mask]

            if len(intersecting_cluster_ptrs) == 0:
                return torch.tensor([], dtype=torch.long, device=query.device)

        # exact filter on final candidates
        keys = index.keys[intersecting_cluster_ptrs]
        final_scores = torch.matmul(keys, query)
        final_mask = final_scores >= threshold
        qualifying_keys = intersecting_cluster_ptrs[final_mask]

        return qualifying_keys  # indices of keys satisfying q.k >= threshold

    def print_stats(self):
        """Print detailed profiling statistics"""
        if not self.enable_profiling:
            print("Profiling not enabled")
            return

        if "total_time" not in self.stats:
            print("No profiling data available (search may have failed)")
            return

        print("\n" + "=" * 80)
        print("HIERARCHICAL SEARCH PROFILING STATISTICS")
        print("=" * 80)
        print(f"Total time: {self.stats['total_time']*1000:.3f} ms")
        print(f"\nTime breakdown:")
        print(f"  Centroid scoring: {self.stats['time_centroid_scoring']*1000:.3f} ms")
        print(f"  Radius checking: {self.stats['time_radius_checking']*1000:.3f} ms")
        print(f"  Data movement: {self.stats['time_data_movement']*1000:.3f} ms")
        print(f"  Exact filtering: {self.stats['time_exact_filtering']*1000:.3f} ms")

        print(f"\nPer-level timing:")
        for i, t in enumerate(self.stats["time_per_level"]):
            print(f"  Level {i}: {t*1000:.3f} ms")

        print(f"\nDistance computations:")
        print(f"  Total: {self.stats['num_distance_computations']}")
        print(f"  Clusters visited: {self.stats['num_clusters_visited']}")
        print(f"  Clusters pruned: {self.stats['num_clusters_pruned']}")

        print(f"\nPruning effectiveness (higher = better):")
        for i, (examined, kept) in enumerate(
            zip(
                self.stats["clusters_examined_per_level"],
                self.stats["clusters_kept_per_level"],
            )
        ):
            if examined > 0:
                pruning_rate = 1.0 - (kept / examined)
                print(
                    f"  Level {i}: examined {examined}, kept {kept} → {pruning_rate*100:.1f}% pruned"
                )
            else:
                print(f"  Level {i}: examined {examined}, kept {kept} → N/A")

        print(f"\nCandidate reduction:")
        for i, count in enumerate(self.stats["num_keys_per_level"]):
            print(f"  After level {i}: {count} candidates")
        print(f"  Final candidates: {self.stats['final_candidates']}")
        print(f"  Final qualifying: {self.stats['final_qualifying']}")
        print("=" * 80)
