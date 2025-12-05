"""
Example demonstrating the new config-based architecture for HIRA.

This example shows how to:
1. Create an IndexConfig (specific to the index type)
2. Create an Index with the config
3. Pass the Index to HiraCache
"""

import torch
from hira import (
    HiraCache,
    KMeansIndexConfig,
    IncrementalIndexConfig,
)


def example_kmeans_index():
    """Example using KMeansIndex with custom configuration."""
    print("=" * 60)
    print("Example 1: KMeansIndex with custom config")
    print("=" * 60)

    # Create a configuration for KMeansIndex
    config = KMeansIndexConfig(
        num_levels=3,
        branching_factor=16,
        max_iterations=50,
        update_frequency="threshold",
        update_threshold=0.2,
        use_gpu=False,
        verbose=False,
    )

    # Create cache with config (indexes created per-layer automatically)
    cache = HiraCache(config)

    print(f"Created cache with KMeansIndexConfig")
    print(f"Config: num_levels={config.num_levels}, branching_factor={config.branching_factor}")
    print(f"Each layer will get its own independent index")

    # Use the cache
    for layer_idx in range(2):
        keys = torch.randn(1, 4, 10, 64)  # [batch, heads, seq, dim]
        values = torch.randn(1, 4, 10, 64)
        cache.update(keys, values, layer_idx)

    # Check per-layer indexes
    print(f"Layer 0 index: {cache.get_index(0).num_keys} keys")
    print(f"Layer 1 index: {cache.get_index(1).num_keys} keys")
    print()


def example_incremental_index():
    """Example using IncrementalIndex with custom configuration."""
    print("=" * 60)
    print("Example 2: IncrementalIndex with custom config")
    print("=" * 60)

    # Create a configuration for IncrementalIndex
    config = IncrementalIndexConfig(
        num_levels=2,
        branching_factor=8,
        rebuild_every_n=10,
        assignment_method="nearest_centroid",
        use_gpu=False,
    )

    # Create cache with config
    cache = HiraCache(config)

    print(f"Created cache with IncrementalIndexConfig")
    print(
        f"Config: num_levels={config.num_levels}, branching_factor={config.branching_factor}"
    )
    print(f"Will rebuild every {config.rebuild_every_n} updates per layer")

    # Use the cache
    for layer_idx in range(2):
        keys = torch.randn(1, 4, 10, 64)
        values = torch.randn(1, 4, 10, 64)
        cache.update(keys, values, layer_idx)

    print(f"Layer 0 index: {cache.get_index(0).num_keys} keys")
    print(f"Layer 1 index: {cache.get_index(1).num_keys} keys")
    print()


def example_default_config():
    """Example using default configuration."""
    print("=" * 60)
    print("Example 3: Using default config")
    print("=" * 60)

    # Use default configuration
    config = KMeansIndexConfig()  # All defaults
    cache = HiraCache(config)

    print(f"Created cache with default KMeansIndexConfig")
    print(f"Config: num_levels={config.num_levels}, branching_factor={config.branching_factor}")

    # Use the cache
    for layer_idx in range(2):
        keys = torch.randn(1, 4, 10, 64)
        values = torch.randn(1, 4, 10, 64)
        cache.update(keys, values, layer_idx)

    print(f"Layer 0 index: {cache.get_index(0).num_keys} keys")
    print(f"Layer 1 index: {cache.get_index(1).num_keys} keys")
    print()


if __name__ == "__main__":
    example_kmeans_index()
    example_incremental_index()
    example_default_config()

    print("✓ All examples completed successfully!")
    print("\nKey takeaway: Pass IndexConfig to HiraCache.")
    print("Each layer automatically gets its own independent index.")
