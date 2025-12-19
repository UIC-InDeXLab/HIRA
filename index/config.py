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
    # Core hierarchy parameters
    num_levels: int = 3
    branching_factor: int = 32
    device: str = "cpu"

    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KMeansIndexConfig(IndexConfig):
    # Clustering parameters
    max_iterations: int = 25
