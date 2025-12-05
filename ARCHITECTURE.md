# Hira Architecture Documentation

## Overview

Hira implements a hierarchical range-searching attention mechanism that efficiently selects high-scoring keys from the KV cache using multi-level indexing and halfspace range queries.

## Core Design Principles

1. **Modularity**: All components are swappable via abstract interfaces
2. **Extensibility**: Easy to add new indexing strategies, search methods, or threshold policies
3. **HuggingFace Compatibility**: Integrates seamlessly with HF models and Cache API
4. **Performance**: Designed for future C++/CUDA optimizations

## Architecture Layers

### 1. Index Layer (`hira/index/`)

**Purpose**: Build and maintain hierarchical indexes over key vectors

**Components**:
- `Index`: Abstract base class for hierarchical indexes (build, update, should_update)
  - `KMeansIndex`: Implementation using hierarchical k-means clustering
  - `IncrementalIndex`: Incremental updates without full rebuilding (WIP)
  - `ProductQuantizationIndex`: PQ-based compression (placeholder)
- `HierarchicalIndex`: Data structure representing multi-level index
  - `IndexLevel`: Single level with centroids, assignments, and metadata
- `MemoryTieringPolicy`: Controls GPU/CPU placement of levels
  - `AllGPUPolicy`: Keep all levels on GPU
  - `HybridGPUCPUPolicy`: Split between GPU and CPU

**Key Design Decisions**:
- Index structure is independent of clustering algorithm
- Each layer has its own index (per-layer indexing)
- Indexes can span multiple devices (GPU/CPU tiering)

### 2. Search Layer (`hira/search/`)

**Purpose**: Efficiently find keys satisfying range conditions

**Components**:
- `RangeSearcher`: Abstract interface for range searching
  - `HalfspaceRangeSearcher`: Find keys where q·k >= τ using hierarchical pruning

**Algorithm**:
```
1. Start at coarsest level (level 0)
2. For each cluster centroid c:
   - Compute bound: max_score = c·q + cluster_radius
   - If max_score < threshold: prune cluster
3. Recurse into qualifying clusters at finer levels
4. At leaf level, compute exact scores and filter
```

**Optimization Opportunities**:
- Precompute cluster radii during index building
- Use tighter bounds (ball bounds, cone bounds)
- Batch coarse-level operations in CUDA

### 3. Cache Layer (`hira/cache/`)

**Purpose**: Manage KV cache with integrated hierarchical indexing

**Components**:
- `HiraCache`: Extends HF `Cache` to maintain indexes alongside KV pairs

**Key Features**:
- Standard KV cache storage (compatible with HF)
- Per-layer hierarchical indexes
- Configurable index update frequency
- Memory policy application

**Index Update Strategy**:
- Keys are flattened: [batch, heads, seq, dim] → [batch*heads*seq, dim]
- One index per layer (shared across batch and heads)
- Alternative strategies possible (per-head indexing, etc.)

### 4. Attention Layer (`hira/attention/`)

**Purpose**: Compute sparse attention using hierarchical key selection

**Components**:
- `HiraAttention`: Core attention computation
- `HiraAttentionProcessor`: HF processor interface (placeholder)

**Computation Flow**:
```
For each query q:
  1. Compute threshold τ using ThresholdStrategy
  2. Use RangeSearcher to find qualifying key indices
  3. Gather selected keys and values
  4. Compute scores: s = (selected_keys @ q) * scale
  5. Apply softmax over selected keys only
  6. Weighted sum: output = softmax(s) @ selected_values
```

**Key Properties**:
- Sparse attention: only selected keys contribute
- Softmax normalization over selected keys (not all keys)
- Falls back to standard attention during prefill (configurable)

### 5. Utilities Layer (`hira/utils/`)

**Purpose**: Threshold computation and helper functions

**Components**:
- `ThresholdStrategy`: Abstract interface for threshold selection
  - `FixedThresholdStrategy`: Constant threshold
  - `TopKThresholdStrategy`: Select top K keys (placeholder)
  - `PercentileThresholdStrategy`: Percentile-based (placeholder)
  - `AdaptiveThresholdStrategy`: Dynamic adjustment (placeholder)

## Data Flow

### Generation Pipeline

```
Input → Model
  ↓
Hidden States → Q/K/V Projections
  ↓
RoPE → Query, Key, Value
  ↓
HiraCache.update()
  ├─ Store K, V
  └─ Update/Build Index
  ↓
HiraAttention.forward()
  ├─ Compute threshold τ
  ├─ RangeSearcher.search() → Selected key indices
  ├─ Gather selected K, V
  ├─ Compute attention scores
  ├─ Softmax (sparse)
  └─ Weighted sum
  ↓
Output
```

## Memory Layout

### KV Cache
```
Standard format: [batch_size, num_heads, seq_len, head_dim]
```

### Hierarchical Index (per layer)
```
Flattened keys: [batch * num_heads * seq_len, head_dim]

Level 0:
  - Centroids: [num_clusters_0, head_dim]
  - Assignments: [total_keys]  # cluster ID for each key

Level 1:
  - Centroids: [num_clusters_1, head_dim]
  - Assignments: [total_keys]
  - Parent assignments: [num_clusters_1]  # parent cluster for each cluster

...
```

## Configuration Options

### Index Configuration
```python
HiraCache(
    num_levels=3,              # Hierarchy depth
    branching_factor=32,       # Clusters per level
    index=KMeansIndex(
        max_iterations=100,
        update_frequency="every_n",
        update_interval=128,
    ),
    memory_policy=AllGPUPolicy(),
    build_index_every_n=128,   # Update frequency override
)
```

### Attention Configuration
```python
HiraAttention(
    threshold_strategy=FixedThresholdStrategy(threshold=0.0),
    range_searcher=HalfspaceRangeSearcher(
        use_bounds=True,
        max_candidates=None,
        early_stopping=False,
    ),
    use_hira_during_prefill=False,  # Use standard attention for prefill
)
```

## Extension Points

### Adding New Index Strategies
1. Subclass `Index`
2. Implement `build()`, `update()`, and `should_update()` methods
3. Return `HierarchicalIndex` with appropriate structure

Example: PQ-based indexing, LSH, learned indexes

### Adding New Search Strategies
1. Subclass `RangeSearcher`
2. Implement `search()` method
3. Can use different pruning strategies or data structures

Example: Approximate search, GPU-accelerated search, learned pruning

### Adding New Threshold Strategies
1. Subclass `ThresholdStrategy`
2. Implement `compute_threshold()` method
3. Can use query/key statistics, historical data, etc.

Example: Adaptive based on attention entropy, learned thresholds

### Adding Custom Memory Policies
1. Subclass `MemoryTieringPolicy`
2. Implement `get_device_assignments()` method
3. Can use dynamic memory monitoring, access patterns, etc.

Example: Dynamic based on GPU memory pressure, LRU-style eviction

## Future Optimizations

### High Priority
1. **Cluster radius computation**: Precompute and store during index building
2. **Batched range search**: Process multiple queries in parallel
3. **CUDA coarse-level pruning**: Fuse centroid scoring and pruning
4. **Per-head indexing**: Separate indexes per attention head

### Medium Priority
1. **Incremental index updates**: Proper online clustering
2. **Index compression**: Reduce memory footprint
3. **Learned cluster boundaries**: ML-based pruning
4. **Attention kernel fusion**: End-to-end sparse attention kernel

### Low Priority
1. **Multi-GPU support**: Distribute index across GPUs
2. **Index checkpointing**: Save/load indexes
3. **Dynamic branching factors**: Adaptive cluster sizes
4. **Approximate attention**: Trade accuracy for speed

## Performance Characteristics

### Time Complexity

**Index Building**: O(n × k × d × iterations)
- n: number of keys
- k: branching factor
- d: head dimension
- iterations: k-means iterations

**Range Search**: O(L × k + m × d)
- L: number of levels
- k: branching factor (clusters checked per level)
- m: number of qualifying keys
- d: head dimension

**Attention**: O(m × d)
- m: number of selected keys (typically << n)
- d: head dimension

**Speedup**: Proportional to compression ratio (m/n)

### Memory Complexity

**Index Storage per Layer**:
- Centroids: O(Σ k^i × d) for i=0..L-1
- Assignments: O(n × L)
- Total: Approximately O(k^L × d + n × L)

**Typical Values** (for 1M keys, 3 levels, k=32, d=64):
- Centroids: ~8 MB
- Assignments: ~12 MB
- Total: ~20 MB per layer

## Testing Strategy

### Unit Tests
- Index building correctness
- Range search correctness
- Cache update logic
- Memory policy application

### Integration Tests
- End-to-end generation with HiraCache
- Attention computation correctness
- Model patching

### Performance Tests
- Benchmark vs standard attention
- Memory profiling
- Scaling tests (varying cache size, num_levels, etc.)

## Debugging Tips

1. **Index quality**: Check cluster distribution and hirarchy depth
2. **Search recall**: Verify selected keys actually satisfy threshold
3. **Attention output**: Compare with standard attention output
4. **Memory usage**: Monitor index size vs KV cache size
5. **Build frequency**: Balance index freshness vs computation cost

## References

- HuggingFace Transformers Cache API
- K-means clustering algorithms
- Range searching in metric spaces
- Sparse attention mechanisms
- MagicPIG and RetrievalAttention implementations
