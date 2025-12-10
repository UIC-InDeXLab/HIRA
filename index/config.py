"""
Index configuration classes for hierarchical indexing.

This module provides configuration classes that encapsulate all parameters
needed to build and update hierarchical indexes. Each index type has its
own configuration class that inherits from the base IndexConfig.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import torch


@dataclass
class IndexConfig:
    """
    Base configuration for hierarchical index building and updating.

    This class encapsulates common parameters needed by all Index classes.
    Specific index types should inherit from this and add their own parameters.

    Attributes:
        num_levels: Number of levels in the hierarchy (default: 3)
        branching_factor: Number of clusters per level (default: 32)
        device: Device for index operations (default: "cpu")
        metadata: Additional custom configuration parameters
    """

    # Core hierarchy parameters
    num_levels: int = 3
    branching_factor: int = 32
    device: str = "cpu"

    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_device(self) -> torch.device:
        """Convert device string to torch.device."""
        return torch.device(self.device)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "num_levels": self.num_levels,
            "branching_factor": self.branching_factor,
            "device": self.device,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "IndexConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(num_levels={self.num_levels}, "
            f"branching_factor={self.branching_factor}, "
            f"device={self.device})"
        )


@dataclass
class KMeansIndexConfig(IndexConfig):
    """
    Configuration specific to KMeansIndex.

    Extends IndexConfig with KMeans-specific parameters for clustering
    and update policies.

    Attributes:
        max_iterations: Maximum k-means iterations per level (default: 25)
        tolerance: Convergence tolerance for k-means (default: 1e-4)
        init_method: Centroid initialization method (default: "kmeans++")
        use_gpu: Whether to use GPU acceleration (default: False)
        nredo: Number of k-means runs to perform (default: 1)
        verbose: Whether to print FAISS clustering progress (default: False)
        use_float16: Use float16 for faster computation (default: False)
        update_frequency: When to rebuild ("always", "every_n", "threshold")
        update_interval: For "every_n", rebuild every N keys (default: 128)
        update_threshold: For "threshold", rebuild ratio (default: 0.1)
    """

    # Clustering parameters
    max_iterations: int = 25
    tolerance: float = 1e-4
    init_method: str = "kmeans++"

    # GPU and performance
    use_gpu: bool = False
    nredo: int = 1
    verbose: bool = False
    use_float16: bool = False

    # Update policy
    update_frequency: str = "every_n"  # "always", "every_n", "threshold"
    update_interval: int = 128
    update_threshold: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "max_iterations": self.max_iterations,
                "tolerance": self.tolerance,
                "init_method": self.init_method,
                "use_gpu": self.use_gpu,
                "nredo": self.nredo,
                "verbose": self.verbose,
                "use_float16": self.use_float16,
                "update_frequency": self.update_frequency,
                "update_interval": self.update_interval,
                "update_threshold": self.update_threshold,
            }
        )
        return base_dict
