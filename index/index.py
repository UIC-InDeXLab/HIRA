"""
Unified hierarchical index classes that handle both building and updating.

Each index type encapsulates its own building and updating logic, making it
easier to implement different indexing strategies with their specific requirements.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
import numpy as np
import faiss

if TYPE_CHECKING:
    from .memory_policy import MemoryTieringPolicy
    from .config import (
        IndexConfig,
        KMeansIndexConfig,
        RandomizedClusteringConfig,
        IncrementalIndexConfig,
        ProductQuantizationIndexConfig,
    )


class Index(ABC):
    """
    Abstract base class for hierarchical indexes.

    An Index is responsible for:
    1. Building a hierarchical index from key vectors
    2. Updating the index when new keys are added
    3. Determining when updates are needed

    Each index type implements its own building and updating strategy.
    The index contains a multi-level hierarchical structure with centroids,
    assignments, and device placement information.

    Attributes:
        levels: List of Level objects, ordered from coarse to fine
        num_keys: Total number of keys indexed
        head_dim: Dimensionality of key vectors
        device: Primary device (can be overridden per level)
        metadata: Additional index-wide metadata (e.g., builder config)
    """

    def __init__(self):
        """Initialize an empty index structure."""
        self.levels: List[Any] = []
        self.num_keys: int = 0
        self.head_dim: int = 0
        self.device: Optional[torch.device] = None
        self.metadata: Dict[str, Any] = {}

    def _validate(self):
        """Validate index consistency."""
        if len(self.levels) == 0:
            return  # Empty index is valid

        for level in self.levels:
            if level.centroids.shape[1] != self.head_dim:
                raise ValueError(
                    f"Level {level.level_idx} centroid dim {level.centroids.shape[1]} "
                    f"doesn't match head_dim {self.head_dim}"
                )
            if level.assignments.shape[0] != self.num_keys:
                raise ValueError(
                    f"Level {level.level_idx} assignments shape {level.assignments.shape[0]} "
                    f"doesn't match num_keys {self.num_keys}"
                )

    def num_levels(self) -> int:
        """Return the number of levels in the hierarchy."""
        return len(self.levels)

    def get_level(self, level_idx: int):
        """
        Get a specific level.

        Args:
            level_idx: Level index (0 = coarsest)

        Returns:
            Level at the specified index
        """
        if level_idx < 0 or level_idx >= len(self.levels):
            raise ValueError(
                f"Invalid level index {level_idx}, index has {len(self.levels)} levels"
            )
        return self.levels[level_idx]

    def apply_memory_policy(self, policy: "MemoryTieringPolicy"):
        """
        Apply a memory tiering policy to move levels between devices.

        Args:
            policy: MemoryTieringPolicy instance specifying device placement
        """
        device_assignments = policy.get_device_assignments(self)

        for level_idx, target_device in device_assignments.items():
            if level_idx >= len(self.levels):
                continue

            current_device = self.levels[level_idx].device
            if current_device != target_device:
                self.levels[level_idx] = self.levels[level_idx].to(target_device)

    def total_memory_usage(self) -> Dict[str, float]:
        """
        Compute total memory usage across all levels.

        Returns:
            Dictionary with memory usage breakdown
        """
        total = {"total_mb": 0.0}

        for level in self.levels:
            level_mem = level.memory_usage()
            total[f"level_{level.level_idx}_mb"] = level_mem["total_mb"]
            total["total_mb"] += level_mem["total_mb"]

        return total

    def get_leaf_assignments(self) -> torch.Tensor:
        """
        Get assignments at the finest (leaf) level.

        Returns:
            Tensor of cluster assignments [num_keys]
        """
        if len(self.levels) == 0:
            raise ValueError("Index has no levels")
        return self.levels[-1].assignments

    def get_hierarchy_path(self, key_idx: int) -> List[int]:
        """
        Get the hierarchical path for a specific key.

        Args:
            key_idx: Index of the key

        Returns:
            List of cluster IDs at each level, from coarse to fine
        """
        path = []
        for level in self.levels:
            cluster_id = level.assignments[key_idx].item()
            path.append(cluster_id)
        return path

    def __repr__(self) -> str:
        """String representation of the index."""
        levels_str = ", ".join(
            [
                f"L{i}: {level.num_clusters()} clusters on {level.device}"
                for i, level in enumerate(self.levels)
            ]
        )
        return (
            f"{self.__class__.__name__}(num_keys={self.num_keys}, head_dim={self.head_dim}, "
            f"num_levels={len(self.levels)}, [{levels_str}])"
        )

    @abstractmethod
    def build(
        self,
        keys: torch.Tensor,
        num_levels: int,
        branching_factor: int,
        device: torch.device,
        **kwargs,
    ) -> "Index":
        """
        Build a hierarchical index from key vectors.

        Args:
            keys: Key vectors of shape [num_keys, head_dim]
            num_levels: Number of hierarchy levels to create
            branching_factor: Number of clusters per level
            device: Device to build the index on
            **kwargs: Additional index-specific parameters

        Returns:
            Index: Self, with populated hierarchical structure
        """
        pass

    @abstractmethod
    def update(
        self,
        current_index: Optional["Index"],
        new_keys: torch.Tensor,
        all_keys: torch.Tensor,
        **kwargs,
    ) -> "Index":
        """
        Update the index with new keys.

        Args:
            current_index: Existing index (None if building from scratch)
            new_keys: Newly added keys [num_new_keys, head_dim]
            all_keys: All keys including new ones [total_keys, head_dim]
            **kwargs: Additional index-specific parameters

        Returns:
            Updated Index
        """
        pass

    @abstractmethod
    def should_update(
        self,
        current_index: Optional["Index"],
        num_new_keys: int,
        total_keys: int,
        **kwargs,
    ) -> bool:
        """
        Determine whether an update is needed.

        Args:
            current_index: Current index (None if no index exists)
            num_new_keys: Number of new keys added
            total_keys: Total number of keys
            **kwargs: Additional parameters

        Returns:
            True if update should be performed
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of this index.

        Returns:
            Dictionary with index configuration
        """
        pass


class KMeansIndex(Index):
    """
    Hierarchical index using FAISS-optimized k-means clustering.

    This index creates a hierarchical structure by recursively applying k-means
    clustering at each level using FAISS for maximum performance. It rebuilds
    the entire index when updates are needed.

    FAISS provides highly optimized implementations with:
    - SSE/AVX vectorization for distance computations
    - Multi-threading support
    - GPU acceleration (if faiss-gpu is installed)
    - Advanced initialization (k-means++)

    Args:
        config: KMeansIndexConfig with all configuration parameters
    """

    @dataclass
    class Level:
        """
        Represents a single level in the hierarchical index.

        Attributes:
            level_idx: Level index (0 = coarsest, higher = finer)
            centroids: Cluster centroids at this level [num_clusters, head_dim]
            assignments: Assignment of keys to clusters [num_keys]
                        Values are cluster indices at this level
            parent_assignments: For levels > 0, maps each cluster to its parent cluster
                               [num_clusters_at_this_level]
            cluster_centers: Center of smallest enclosing ball [num_clusters, head_dim]
            cluster_radii: Radius of smallest enclosing ball [num_clusters]
                          Used for hierarchical pruning in search
            device: Device where this level resides (GPU or CPU)
            metadata: Additional level-specific metadata
        """

        level_idx: int
        centroids: torch.Tensor  # [num_clusters, head_dim]
        assignments: torch.Tensor  # [num_keys], cluster indices
        parent_assignments: Optional[torch.Tensor]  # [num_clusters], parent cluster IDs
        cluster_centers: torch.Tensor  # [num_clusters, head_dim], ball centers
        cluster_radii: torch.Tensor  # [num_clusters], ball radii
        device: torch.device
        metadata: Dict[str, Any] = field(default_factory=dict)

        def __post_init__(self):
            # Validate shapes
            num_clusters = self.centroids.shape[0]
            if self.parent_assignments is not None:
                if self.parent_assignments.shape[0] != num_clusters:
                    raise ValueError(
                        f"parent_assignments shape {self.parent_assignments.shape} "
                        f"doesn't match num_clusters {num_clusters}"
                    )

        def to(self, device: torch.device) -> "KMeansIndex.Level":
            """
            Move this level to a different device.

            Args:
                device: Target device

            Returns:
                New Level on the target device
            """
            return KMeansIndex.Level(
                level_idx=self.level_idx,
                centroids=self.centroids.to(device),
                assignments=self.assignments.to(device),
                parent_assignments=(
                    self.parent_assignments.to(device)
                    if self.parent_assignments is not None
                    else None
                ),
                cluster_centers=self.cluster_centers.to(device),
                cluster_radii=self.cluster_radii.to(device),
                device=device,
                metadata=self.metadata.copy(),
            )

        def num_clusters(self) -> int:
            """Return the number of clusters at this level."""
            return self.centroids.shape[0]

        def head_dim(self) -> int:
            """Return the dimensionality of centroids."""
            return self.centroids.shape[1]

        def get_cluster_members(self, cluster_id: int) -> torch.Tensor:
            """
            Get indices of keys assigned to a specific cluster.

            Args:
                cluster_id: Cluster ID

            Returns:
                Tensor of key indices belonging to this cluster
            """
            mask = self.assignments == cluster_id
            return torch.nonzero(mask, as_tuple=False).squeeze(-1)

        def memory_usage(self) -> Dict[str, float]:
            """
            Compute memory usage of this level.

            Returns:
                Dictionary with memory usage in MB
            """
            centroid_mem = (
                self.centroids.element_size() * self.centroids.nelement() / (1024**2)
            )
            assignment_mem = (
                self.assignments.element_size()
                * self.assignments.nelement()
                / (1024**2)
            )
            parent_mem = 0.0
            if self.parent_assignments is not None:
                parent_mem = (
                    self.parent_assignments.element_size()
                    * self.parent_assignments.nelement()
                    / (1024**2)
                )
            radii_mem = (
                self.cluster_radii.element_size()
                * self.cluster_radii.nelement()
                / (1024**2)
            )

            return {
                "centroids_mb": centroid_mem,
                "assignments_mb": assignment_mem,
                "parent_assignments_mb": parent_mem,
                "cluster_radii_mb": radii_mem,
                "total_mb": centroid_mem + assignment_mem + parent_mem + radii_mem,
            }

    def __init__(self, config: "KMeansIndexConfig"):
        super().__init__()
        # Store config
        self.config = config

        # K-means parameters from config
        self.max_iterations = config.max_iterations
        self.tolerance = config.tolerance
        self.init_method = config.init_method
        self.nredo = config.nredo
        self.verbose = config.verbose
        self.spherical = config.spherical
        self.use_float16 = config.use_float16

        # GPU setup
        self.use_gpu = config.use_gpu
        self.gpu_available = False
        if config.use_gpu:
            try:
                self.gpu_available = hasattr(faiss, "StandardGpuResources")
                if self.gpu_available:
                    self.gpu_resources = faiss.StandardGpuResources()
                else:
                    print(
                        "Warning: GPU requested but faiss-gpu not available, using CPU"
                    )
                    self.use_gpu = False
            except Exception as e:
                print(f"Warning: GPU initialization failed ({e}), using CPU")
                self.use_gpu = False
                self.gpu_available = False

        # Update parameters from config
        if config.update_frequency not in ["always", "every_n", "threshold"]:
            raise ValueError(f"Invalid update_frequency: {config.update_frequency}")
        self.update_frequency = config.update_frequency
        self.update_interval = config.update_interval
        self.update_threshold = config.update_threshold
        self._keys_since_last_update = 0

    def build(
        self,
        keys: torch.Tensor,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "Index":
        """
        Build hierarchical index using FAISS-optimized recursive k-means clustering.

        Args:
            keys: Key vectors [num_keys, head_dim]
            device: Device for computation (overrides config if provided)
            **kwargs: Additional parameters

        Returns:
            Self, with populated hierarchical structure
        """
        if keys.shape[0] == 0:
            raise ValueError("Cannot build index from empty key set")

        # Use device from config if not provided
        if device is None:
            device = self.config.to_device()

        keys = keys.to(device)
        num_keys, head_dim = keys.shape

        # Get hierarchy parameters from config
        num_levels = self.config.num_levels
        branching_factor = self.config.branching_factor

        # Initialize index attributes
        self.num_keys = num_keys
        self.head_dim = head_dim
        self.device = device
        self.levels = []

        # Level 0: Cluster all keys
        centroids_0, assignments_0 = self._kmeans_faiss(
            keys, min(branching_factor, num_keys), device
        )

        # Compute smallest enclosing ball for level 0
        centers_0, radii_0 = self._compute_cluster_radii(keys, centroids_0, assignments_0, device)

        level_0 = KMeansIndex.Level(
            level_idx=0,
            centroids=centroids_0,
            assignments=assignments_0,
            parent_assignments=None,
            cluster_centers=centers_0,
            cluster_radii=radii_0,
            device=device,
        )
        self.levels.append(level_0)

        # Build subsequent levels by clustering within each parent cluster
        for level_idx in range(1, num_levels):
            parent_level = self.levels[level_idx - 1]
            parent_centroids = parent_level.centroids
            parent_assignments = parent_level.assignments

            # For each parent cluster, subdivide its members
            all_centroids = []
            all_assignments = []
            all_centers = []
            all_radii = []
            parent_of_child = []

            for parent_cluster_id in range(parent_centroids.shape[0]):
                # Get keys belonging to this parent cluster
                mask = parent_assignments == parent_cluster_id
                cluster_keys = keys[mask]

                if cluster_keys.shape[0] == 0:
                    continue

                # Cluster within this parent cluster
                num_subclusters = min(branching_factor, cluster_keys.shape[0])
                sub_centroids, sub_assignments = self._kmeans_faiss(
                    cluster_keys, num_subclusters, device
                )

                # Compute radii for subclusters
                sub_centers, sub_radii = self._compute_cluster_radii(
                    cluster_keys, sub_centroids, sub_assignments, device
                )

                # Remap assignments to global indexing
                global_cluster_offset = len(all_centroids)
                sub_assignments_global = sub_assignments + global_cluster_offset

                all_centroids.append(sub_centroids)
                all_centers.append(sub_centers)
                all_radii.append(sub_radii)

                # Store the remapped assignments for keys in this parent cluster
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
            level_centers = torch.cat(all_centers, dim=0)
            level_radii = torch.cat(all_radii, dim=0)

            # Combine assignments: take the first non-(-1) value for each key
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

            level = KMeansIndex.Level(
                level_idx=level_idx,
                centroids=level_centroids,
                assignments=level_assignments,
                parent_assignments=parent_of_child_tensor,
                cluster_centers=level_centers,
                cluster_radii=level_radii,
                device=device,
            )
            self.levels.append(level)

        # Validate the constructed index
        self._validate()

        return self

    def update(
        self,
        current_index: Optional["Index"],
        new_keys: torch.Tensor,
        all_keys: torch.Tensor,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "Index":
        """
        Update index by rebuilding from scratch using all keys.

        Args:
            current_index: Existing index (ignored, will be rebuilt)
            new_keys: Newly added keys
            all_keys: All keys
            device: Device for computation (overrides config if provided)
            **kwargs: Additional parameters

        Returns:
            Self, rebuilt with all keys
        """
        # Use device from config if not provided
        if device is None:
            device = kwargs.get("device", self.config.to_device())

        # Rebuild from scratch using config parameters
        self.build(
            keys=all_keys,
            device=device,
        )

        self._keys_since_last_update = 0
        return self

    def should_update(
        self,
        current_index: Optional["Index"],
        num_new_keys: int,
        total_keys: int,
        **kwargs,
    ) -> bool:
        """
        Determine if rebuild is needed based on update frequency.

        Args:
            current_index: Current index
            num_new_keys: Number of new keys
            total_keys: Total keys
            **kwargs: Additional parameters

        Returns:
            True if rebuild should happen
        """
        if current_index is None:
            return True

        self._keys_since_last_update += num_new_keys

        if self.update_frequency == "always":
            return True
        elif self.update_frequency == "every_n":
            if self._keys_since_last_update >= self.update_interval:
                return True
        elif self.update_frequency == "threshold":
            ratio = self._keys_since_last_update / total_keys
            if ratio >= self.update_threshold:
                return True

        return False

    def _kmeans_faiss(
        self,
        data: torch.Tensor,
        num_clusters: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        FAISS-optimized k-means clustering with GPU support.

        Args:
            data: Data points [num_points, dim]
            num_clusters: Number of clusters
            device: Device for computation

        Returns:
            Tuple of (centroids [num_clusters, dim], assignments [num_points])
        """
        num_points, dim = data.shape

        if num_clusters >= num_points:
            # Degenerate case: one point per cluster
            centroids = data.clone()
            assignments = torch.arange(num_points, device=device, dtype=torch.long)
            return centroids, assignments

        # Convert to numpy for FAISS (contiguous float32)
        data_np = data.cpu().float().numpy().astype("float32")
        if not data_np.flags["C_CONTIGUOUS"]:
            data_np = np.ascontiguousarray(data_np)

        # Normalize for spherical k-means if requested
        if self.spherical:
            faiss.normalize_L2(data_np)

        # Create k-means clusterer
        kmeans = faiss.Kmeans(
            d=dim,
            k=num_clusters,
            niter=self.max_iterations,
            nredo=self.nredo,
            verbose=self.verbose,
            spherical=self.spherical,
            gpu=self.use_gpu and self.gpu_available,
            seed=42,
        )

        # Train the k-means model
        kmeans.train(data_np)

        # Get cluster assignments
        _, assignments_np = kmeans.index.search(data_np, 1)
        assignments_np = assignments_np.squeeze(1)

        # Convert back to torch tensors
        centroids = torch.from_numpy(kmeans.centroids).to(device)
        assignments = torch.from_numpy(assignments_np).long().to(device)

        return centroids, assignments

    def _compute_cluster_radii(
        self,
        data: torch.Tensor,
        centroids: torch.Tensor,
        assignments: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the smallest enclosing ball for each cluster.

        Args:
            data: Data points [num_points, dim]
            centroids: Cluster centroids [num_clusters, dim]
            assignments: Cluster assignments [num_points]
            device: Device for computation

        Returns:
            Tuple of (centers, radii) where:
            - centers: Center of smallest enclosing ball [num_clusters, dim]
            - radii: Radius of smallest enclosing ball [num_clusters]
        """
        num_clusters = centroids.shape[0]
        dim = centroids.shape[1]
        centers = torch.zeros(num_clusters, dim, device=device)
        radii = torch.zeros(num_clusters, device=device)

        for cluster_id in range(num_clusters):
            # Get all points belonging to this cluster
            mask = assignments == cluster_id
            cluster_points = data[mask]

            if cluster_points.shape[0] == 0:
                # Empty cluster, use centroid as center with radius 0
                centers[cluster_id] = centroids[cluster_id]
                radii[cluster_id] = 0.0
            elif cluster_points.shape[0] == 1:
                # Single point, center is the point with radius 0
                centers[cluster_id] = cluster_points[0]
                radii[cluster_id] = 0.0
            else:
                # Compute smallest enclosing ball
                # Simple approximation: use mean as center, max distance as radius
                center = cluster_points.mean(dim=0)
                distances = torch.norm(cluster_points - center, dim=1)
                radius = distances.max()
                centers[cluster_id] = center
                radii[cluster_id] = radius

        return centers, radii

    def get_config(self) -> Dict[str, Any]:
        """Return index configuration."""
        return {
            "index_type": "kmeans",
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "init_method": self.init_method,
            "use_gpu": self.use_gpu,
            "gpu_available": self.gpu_available,
            "nredo": self.nredo,
            "verbose": self.verbose,
            "spherical": self.spherical,
            "use_float16": self.use_float16,
            "update_frequency": self.update_frequency,
            "update_interval": self.update_interval,
            "update_threshold": self.update_threshold,
            "faiss_version": faiss.__version__,
        }


class RandomizedClustering(Index):
    """
    Hierarchical index using randomized sampling for clustering.

    This index creates a hierarchical structure by randomly sampling points
    to serve as cluster centroids at each level, then assigning all points
    to their nearest sampled centroid. This is much faster than k-means
    clustering but may result in less optimal cluster assignments.

    Building Strategy:
    1. Start with all N keys at the bottom
    2. Randomly sample N/branching_factor points as centroids for Level 0
    3. Assign all keys to their nearest Level 0 centroid
    4. At Level 1, sample from Level 0 centroids: M/branching_factor points
       where M = N/branching_factor
    5. Continue recursively up the hierarchy

    This approach trades clustering quality for speed - useful for large-scale
    scenarios where building time is critical.

    Args:
        config: RandomizedClusteringConfig with all configuration parameters
    """

    @dataclass
    class Level:
        """
        Represents a single level in the hierarchical index.

        Attributes:
            level_idx: Level index (0 = coarsest, higher = finer)
            centroids: Cluster centroids at this level [num_clusters, head_dim]
            assignments: Assignment of keys to clusters [num_keys]
                        Values are cluster indices at this level
            parent_assignments: For levels > 0, maps each cluster to its parent cluster
                               [num_clusters_at_this_level]
            cluster_radii: Maximum distance from each centroid to its members [num_clusters]
                          Used for hierarchical pruning in search
            device: Device where this level resides (GPU or CPU)
            metadata: Additional level-specific metadata
        """

        level_idx: int
        centroids: torch.Tensor  # [num_clusters, head_dim]
        assignments: torch.Tensor  # [num_keys], cluster indices
        parent_assignments: Optional[torch.Tensor]  # [num_clusters], parent cluster IDs
        cluster_centers: torch.Tensor  # [num_clusters, head_dim], center of smallest enclosing ball
        cluster_radii: torch.Tensor  # [num_clusters], radius of smallest enclosing ball
        device: torch.device
        metadata: Dict[str, Any] = field(default_factory=dict)

        def __post_init__(self):
            # Validate shapes
            num_clusters = self.centroids.shape[0]
            if self.parent_assignments is not None:
                if self.parent_assignments.shape[0] != num_clusters:
                    raise ValueError(
                        f"parent_assignments shape {self.parent_assignments.shape} "
                        f"doesn't match num_clusters {num_clusters}"
                    )

        def to(self, device: torch.device) -> "RandomizedClustering.Level":
            """
            Move this level to a different device.

            Args:
                device: Target device

            Returns:
                New Level on the target device
            """
            return RandomizedClustering.Level(
                level_idx=self.level_idx,
                centroids=self.centroids.to(device),
                assignments=self.assignments.to(device),
                parent_assignments=(
                    self.parent_assignments.to(device)
                    if self.parent_assignments is not None
                    else None
                ),
                cluster_centers=self.cluster_centers.to(device),
                cluster_radii=self.cluster_radii.to(device),
                device=device,
                metadata=self.metadata.copy(),
            )

        def num_clusters(self) -> int:
            """Return the number of clusters at this level."""
            return self.centroids.shape[0]

        def head_dim(self) -> int:
            """Return the dimensionality of centroids."""
            return self.centroids.shape[1]

        def get_cluster_members(self, cluster_id: int) -> torch.Tensor:
            """
            Get indices of keys assigned to a specific cluster.

            Args:
                cluster_id: Cluster ID

            Returns:
                Tensor of key indices belonging to this cluster
            """
            mask = self.assignments == cluster_id
            return torch.nonzero(mask, as_tuple=False).squeeze(-1)

        def memory_usage(self) -> Dict[str, float]:
            """
            Compute memory usage of this level.

            Returns:
                Dictionary with memory usage in MB
            """
            centroid_mem = (
                self.centroids.element_size() * self.centroids.nelement() / (1024**2)
            )
            assignment_mem = (
                self.assignments.element_size()
                * self.assignments.nelement()
                / (1024**2)
            )
            parent_mem = 0.0
            if self.parent_assignments is not None:
                parent_mem = (
                    self.parent_assignments.element_size()
                    * self.parent_assignments.nelement()
                    / (1024**2)
                )
            radii_mem = (
                self.cluster_radii.element_size()
                * self.cluster_radii.nelement()
                / (1024**2)
            )

            return {
                "centroids_mb": centroid_mem,
                "assignments_mb": assignment_mem,
                "parent_assignments_mb": parent_mem,
                "cluster_radii_mb": radii_mem,
                "total_mb": centroid_mem + assignment_mem + parent_mem + radii_mem,
            }

    def __init__(self, config: "RandomizedClusteringConfig"):
        super().__init__()
        # Store config
        self.config = config

        # Random sampling parameters
        self.random_seed = config.random_seed

        # K-means parameters (same as KMeans but with niter=1)
        self.max_iterations = 1  # Single iteration = random initialization
        self.tolerance = config.tolerance
        self.nredo = config.nredo
        self.verbose = config.verbose
        self.spherical = config.spherical
        self.use_gpu = config.use_gpu
        self.gpu_available = faiss.get_num_gpus() > 0

        # Update parameters from config
        if config.update_frequency not in ["always", "every_n", "threshold"]:
            raise ValueError(f"Invalid update_frequency: {config.update_frequency}")
        self.update_frequency = config.update_frequency
        self.update_interval = config.update_interval
        self.update_threshold = config.update_threshold
        self._keys_since_last_update = 0

    def build(
        self,
        keys: torch.Tensor,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "Index":
        """
        Build hierarchical index using FAISS k-means with niter=1 (random initialization).

        Args:
            keys: Key vectors [num_keys, head_dim]
            device: Device for computation (overrides config if provided)
            **kwargs: Additional parameters

        Returns:
            Self, with populated hierarchical structure
        """
        if keys.shape[0] == 0:
            raise ValueError("Cannot build index from empty key set")

        # Use device from config if not provided
        if device is None:
            device = self.config.to_device()

        keys = keys.to(device)
        num_keys, head_dim = keys.shape

        # Get hierarchy parameters from config
        num_levels = self.config.num_levels
        branching_factor = self.config.branching_factor

        # Initialize index attributes
        self.num_keys = num_keys
        self.head_dim = head_dim
        self.device = device
        self.levels = []

        # Level 0: Cluster all keys
        centroids_0, assignments_0 = self._kmeans_faiss(
            keys, min(branching_factor, num_keys), device
        )

        # Compute smallest enclosing ball for level 0
        centers_0, radii_0 = self._compute_cluster_radii(keys, centroids_0, assignments_0, device)

        level_0 = RandomizedClustering.Level(
            level_idx=0,
            centroids=centroids_0,
            assignments=assignments_0,
            parent_assignments=None,
            cluster_centers=centers_0,
            cluster_radii=radii_0,
            device=device,
        )
        self.levels.append(level_0)

        # Build subsequent levels by clustering within each parent cluster
        for level_idx in range(1, num_levels):
            parent_level = self.levels[level_idx - 1]
            parent_centroids = parent_level.centroids
            parent_assignments = parent_level.assignments

            # For each parent cluster, subdivide its members
            all_centroids = []
            all_assignments = []
            all_centers = []
            all_radii = []
            parent_of_child = []

            for parent_cluster_id in range(parent_centroids.shape[0]):
                # Get keys belonging to this parent cluster
                mask = parent_assignments == parent_cluster_id
                cluster_keys = keys[mask]

                if cluster_keys.shape[0] == 0:
                    continue

                # Cluster within this parent cluster
                num_subclusters = min(branching_factor, cluster_keys.shape[0])
                sub_centroids, sub_assignments = self._kmeans_faiss(
                    cluster_keys, num_subclusters, device
                )

                # Compute radii for subclusters
                sub_centers, sub_radii = self._compute_cluster_radii(
                    cluster_keys, sub_centroids, sub_assignments, device
                )

                # Remap assignments to global indexing
                global_cluster_offset = len(all_centroids)
                sub_assignments_global = sub_assignments + global_cluster_offset

                all_centroids.append(sub_centroids)
                all_centers.append(sub_centers)
                all_radii.append(sub_radii)

                # Store the remapped assignments for keys in this parent cluster
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
            level_centers = torch.cat(all_centers, dim=0)
            level_radii = torch.cat(all_radii, dim=0)

            # Combine assignments: take the first non-(-1) value for each key
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

            level = RandomizedClustering.Level(
                level_idx=level_idx,
                centroids=level_centroids,
                assignments=level_assignments,
                parent_assignments=parent_of_child_tensor,
                cluster_centers=level_centers,
                cluster_radii=level_radii,
                device=device,
            )
            self.levels.append(level)

        # Validate the constructed index
        self._validate()

        return self

    def update(
        self,
        current_index: Optional["Index"],
        new_keys: torch.Tensor,
        all_keys: torch.Tensor,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "Index":
        """
        Update index by rebuilding from scratch using all keys.

        Args:
            current_index: Existing index (ignored, will be rebuilt)
            new_keys: Newly added keys
            all_keys: All keys
            device: Device for computation (overrides config if provided)
            **kwargs: Additional parameters

        Returns:
            Self, rebuilt with all keys
        """
        # Use device from config if not provided
        if device is None:
            device = kwargs.get("device", self.config.to_device())

        # Rebuild from scratch using config parameters
        self.build(
            keys=all_keys,
            device=device,
        )

        self._keys_since_last_update = 0
        return self

    def should_update(
        self,
        current_index: Optional["Index"],
        num_new_keys: int,
        total_keys: int,
        **kwargs,
    ) -> bool:
        """
        Determine if rebuild is needed based on update frequency.

        Args:
            current_index: Current index
            num_new_keys: Number of new keys
            total_keys: Total keys
            **kwargs: Additional parameters

        Returns:
            True if rebuild should happen
        """
        if current_index is None:
            return True

        self._keys_since_last_update += num_new_keys

        if self.update_frequency == "always":
            return True
        elif self.update_frequency == "every_n":
            if self._keys_since_last_update >= self.update_interval:
                return True
        elif self.update_frequency == "threshold":
            ratio = self._keys_since_last_update / total_keys
            if ratio >= self.update_threshold:
                return True

        return False

    def _kmeans_faiss(
        self,
        data: torch.Tensor,
        num_clusters: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        FAISS-optimized k-means clustering with niter=1 (random initialization).

        Args:
            data: Data points [num_points, dim]
            num_clusters: Number of clusters
            device: Device for computation

        Returns:
            Tuple of (centroids [num_clusters, dim], assignments [num_points])
        """
        num_points, dim = data.shape

        if num_clusters >= num_points:
            # Degenerate case: one point per cluster
            centroids = data.clone()
            assignments = torch.arange(num_points, device=device, dtype=torch.long)
            return centroids, assignments

        # Convert to numpy for FAISS (contiguous float32)
        data_np = data.cpu().float().numpy().astype("float32")
        if not data_np.flags["C_CONTIGUOUS"]:
            data_np = np.ascontiguousarray(data_np)

        # Normalize for spherical k-means if requested
        if self.spherical:
            faiss.normalize_L2(data_np)

        # Create k-means clusterer with niter=1 for random initialization
        kmeans = faiss.Kmeans(
            d=dim,
            k=num_clusters,
            niter=1,  # Single iteration = random initialization
            nredo=self.nredo,
            verbose=self.verbose,
            spherical=self.spherical,
            gpu=self.use_gpu and self.gpu_available,
            seed=self.random_seed,
        )

        # Train the k-means model
        kmeans.train(data_np)

        # Get cluster assignments
        _, assignments_np = kmeans.index.search(data_np, 1)
        assignments_np = assignments_np.squeeze(1)

        # Convert back to torch tensors
        centroids = torch.from_numpy(kmeans.centroids).to(device)
        assignments = torch.from_numpy(assignments_np).long().to(device)

        return centroids, assignments

    def _compute_cluster_radii(
        self,
        data: torch.Tensor,
        centroids: torch.Tensor,
        assignments: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the smallest enclosing ball for each cluster.

        Args:
            data: Data points [num_points, dim]
            centroids: Cluster centroids [num_clusters, dim]
            assignments: Cluster assignments [num_points]
            device: Device for computation

        Returns:
            Tuple of (centers, radii) where:
            - centers: Center of smallest enclosing ball [num_clusters, dim]
            - radii: Radius of smallest enclosing ball [num_clusters]
        """
        num_clusters = centroids.shape[0]
        dim = centroids.shape[1]
        centers = torch.zeros(num_clusters, dim, device=device)
        radii = torch.zeros(num_clusters, device=device)

        for cluster_id in range(num_clusters):
            # Get all points belonging to this cluster
            mask = assignments == cluster_id
            cluster_points = data[mask]

            if cluster_points.shape[0] == 0:
                # Empty cluster, use centroid as center with radius 0
                centers[cluster_id] = centroids[cluster_id]
                radii[cluster_id] = 0.0
            elif cluster_points.shape[0] == 1:
                # Single point, center is the point with radius 0
                centers[cluster_id] = cluster_points[0]
                radii[cluster_id] = 0.0
            else:
                # Compute smallest enclosing ball
                # Simple approximation: use mean as center, max distance as radius
                center = cluster_points.mean(dim=0)
                distances = torch.norm(cluster_points - center, dim=1)
                radius = distances.max()
                centers[cluster_id] = center
                radii[cluster_id] = radius

        return centers, radii

    def get_config(self) -> Dict[str, Any]:
        """Return index configuration."""
        return {
            "index_type": "randomized_clustering",
            "random_seed": self.random_seed,
            "update_frequency": self.update_frequency,
            "update_interval": self.update_interval,
            "update_threshold": self.update_threshold,
        }
