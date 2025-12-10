# HIRA Usage Guide

## Per-Layer Index Architecture

HIRA maintains **one independent index per layer** because different layers in an LLM capture different semantic information:
- **Early layers**: Syntax, local patterns
- **Middle layers**: Entities, relationships  
- **Late layers**: High-level reasoning, task-specific features

Each layer's keys have different distributions and should be indexed separately for correct attention computation.

## Quick Start

### Basic Usage

```python
import torch
from hira import HiraCache, KMeansIndexConfig

# Step 1: Create a configuration
config = KMeansIndexConfig(
    num_levels=3,
    branching_factor=16,
    max_iterations=50,
    update_frequency="every_n",
    update_interval=128,
)

# Step 2: Create cache with config
# Each layer will automatically get its own index
cache = HiraCache(config)

# Step 3: Use the cache (indexes created lazily per layer)
for layer_idx in range(num_layers):
    keys = ...  # [batch, heads, seq, dim]
    values = ...
    cache.update(keys, values, layer_idx)

# Access per-layer indexes
index_layer_0 = cache.get_index(0)  # Index for layer 0
index_layer_1 = cache.get_index(1)  # Index for layer 1
```

### Using IncrementalIndex

```python
from hira import IncrementalIndexConfig

# Create config for incremental updates
config = IncrementalIndexConfig(
    num_levels=2,
    branching_factor=8,
    rebuild_every_n=10,  # Rebuild from scratch every 10 updates
    assignment_method="nearest_centroid",
)

cache = HiraCache(config)
# Each layer gets its own IncrementalIndex instance
```

### Using Default Configuration

```python
from hira import KMeansIndexConfig

# Use all default values
config = KMeansIndexConfig()
cache = HiraCache(config)
```

## Configuration Classes

### IndexConfig (Base Class)

Common parameters for all index types:

- `num_levels`: Number of hierarchy levels (default: 3)
- `branching_factor`: Number of clusters per level (default: 32)
- `device`: Device for index operations (default: "cpu")
- `metadata`: Additional custom parameters (default: {})

### KMeansIndexConfig

Extends `IndexConfig` with K-means specific parameters:

**Clustering Parameters:**
- `max_iterations`: Maximum k-means iterations per level (default: 25)
- `tolerance`: Convergence tolerance (default: 1e-4)
- `init_method`: Initialization method - "kmeans++" or "random" (default: "kmeans++")

**Performance Parameters:**
- `use_gpu`: Whether to use GPU acceleration (default: False)
- `nredo`: Number of k-means runs (keeps best result) (default: 1)
- `verbose`: Print FAISS clustering progress (default: False)
- `use_float16`: Use float16 for faster computation (default: False)

**Update Policy:**
- `update_frequency`: When to rebuild - "always", "every_n", or "threshold" (default: "every_n")
- `update_interval`: For "every_n", rebuild every N keys (default: 128)
- `update_threshold`: For "threshold", rebuild ratio (default: 0.1)

### IncrementalIndexConfig

Extends `IndexConfig` for incremental updates:

- `max_iterations`: K-means iterations for initial build (default: 25)
- `rebuild_every_n`: Rebuild from scratch every N updates (default: 10)
- `assignment_method`: How to assign new keys (default: "nearest_centroid")
- `use_gpu`: GPU acceleration (default: False)

### ProductQuantizationIndexConfig

Extends `IndexConfig` for product quantization:

- `num_subquantizers`: Number of subvectors (must divide head_dim) (default: 8)
- `bits_per_subquantizer`: Bits per subquantizer (default: 8)
- `max_iterations`: Training iterations (default: 25)
- `use_gpu`: GPU acceleration (default: False)

## Advanced Usage

### Accessing Per-Layer Indexes

```python
# Get index for specific layer
index = cache.get_index(layer_idx=0)

if index is not None:
    print(f"Layer 0: {index.num_keys} keys, {index.num_levels()} levels")
```

### Getting Cache Statistics

```python
# Get info for all layers
info = cache.get_cache_info()
print(f"Total layers: {info['num_layers']}")
print(f"Built indexes: {info['num_built_indexes']}")
print(f"Total indexed keys: {info['total_indexed_keys']}")

for layer_info in info['layers']:
    layer_idx = layer_info['layer_idx']
    if 'index_info' in layer_info:
        num_keys = layer_info['index_info']['num_keys']
        print(f"  Layer {layer_idx}: {num_keys} keys indexed")

# Get info for specific layer
layer_0_info = cache.get_cache_info(layer_idx=0)
```

### Custom Configuration

```python
config = KMeansIndexConfig(
    num_levels=4,           # Deeper hierarchy
    branching_factor=64,    # More clusters per level
    max_iterations=100,     # More iterations for better quality
    use_gpu=True,           # Use GPU if available
    nredo=5,                # Try 5 times, keep best
    verbose=True,           # Show progress
    update_frequency="threshold",
    update_threshold=0.15,  # Rebuild when 15% new keys
)

cache = HiraCache(config)
```

### Configuration Serialization

```python
# Save config to dictionary
config_dict = config.to_dict()

# Restore config from dictionary
restored_config = KMeansIndexConfig.from_dict(config_dict)
```

## Key Concepts

### Per-Layer Independence

Each layer maintains its own index completely independently:

```python
# Layer 0 and Layer 1 have separate indexes
cache.update(keys_0, values_0, layer_idx=0)  # Builds index for layer 0
cache.update(keys_1, values_1, layer_idx=1)  # Builds index for layer 1

# These are different objects
assert cache.get_index(0) is not cache.get_index(1)

# Each tracks its own keys
print(cache.get_index(0).num_keys)  # Keys in layer 0
print(cache.get_index(1).num_keys)  # Keys in layer 1
```

### Lazy Index Creation

Indexes are created on-demand when a layer is first updated:

```python
cache = HiraCache(config)

# No indexes yet
assert cache.get_index(0) is None

# Update layer 0 - creates index for layer 0 only
cache.update(keys, values, layer_idx=0)
assert cache.get_index(0) is not None  # Created
assert cache.get_index(1) is None      # Not created yet
```

### Why Per-Layer Indexes?

When computing attention at layer `i`:
```
Attention(Q_i, K_i, V_i)
```

You need to search within `K_i` (keys from layer i), not keys from all layers mixed together. Mixing layers would give incorrect attention scores because:
1. Different layers have different semantic spaces
2. Query from layer i is semantically compatible with keys from layer i
3. Cross-layer key matching doesn't make sense in standard transformer attention

## Migration Guide

### Old API (if you had a single unified index)

```python
# Old (hypothetical):
index = KMeansIndex(config)
cache = HiraCache(index)  # Single index for all layers
```

### New API (per-layer indexes)

```python
# New:
config = KMeansIndexConfig(...)
cache = HiraCache(config)  # Each layer gets its own index

# Access per-layer:
layer_0_index = cache.get_index(0)
layer_1_index = cache.get_index(1)
```

## Examples

See `examples/config_usage_example.py` for complete working examples demonstrating:
- KMeansIndexConfig usage
- IncrementalIndexConfig usage
- Default configurations
- Per-layer index access

## Benefits

1. **Semantic Correctness**: Each layer indexes its own semantic space
2. **Independence**: Updates to one layer don't affect others
3. **Memory Efficiency**: Can apply different policies per layer
4. **Correct Attention**: Queries search keys from the same layer
5. **Lazy Creation**: Indexes created only for layers that are actually used
