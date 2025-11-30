# Kernels - C++/CUDA Optimizations

This directory contains placeholders for future C++ and CUDA kernel implementations that will accelerate critical operations in Hira.

## Planned Optimizations

### 1. Hierarchical Clustering Kernels (`cpp/clustering/`)
- **Fast k-means clustering** for index building
- **Online clustering updates** for incremental index maintenance
- **Batched clustering** across multiple layers
- Integration with FAISS for large-scale clustering

### 2. Range Search Kernels (`cuda/range_search/`)
- **Coarse-level pruning** on GPU with fused operations
- **Hierarchical traversal** with optimized memory access patterns
- **Batched range search** for multiple queries
- **Approximate range search** with early termination

### 3. Sparse Attention Kernels (`cuda/sparse_attention/`)
- **Gather-scatter operations** for sparse key-value access
- **Sparse softmax** over non-contiguous indices
- **Fused attention computation** (score + softmax + weighted sum)
- **Memory-efficient attention** with on-the-fly computation

### 4. Memory Management Kernels (`cpp/memory/`)
- **Efficient CPU-GPU transfers** for hierarchical tiering
- **Pinned memory management** for index structures
- **Prefetching strategies** for fine-level access
- **Compression** for offloaded index levels

### 5. Index Operations (`cpp/index/`)
- **Fast centroid scoring** with SIMD vectorization
- **Cluster assignment** with optimized distance computation
- **Index serialization/deserialization** for checkpointing
- **Multi-threaded index building**

## Directory Structure

```
kernels/
├── README.md                    # This file
├── cpp/                         # C++ implementations
│   ├── clustering/              # Clustering algorithms
│   │   ├── kmeans.h
│   │   ├── kmeans.cpp
│   │   └── online_clustering.cpp
│   ├── index/                   # Index operations
│   │   ├── build.h
│   │   ├── build.cpp
│   │   └── update.cpp
│   ├── memory/                  # Memory management
│   │   ├── transfer.h
│   │   ├── transfer.cpp
│   │   └── prefetch.cpp
│   └── bindings/                # Python bindings (pybind11)
│       ├── clustering_bindings.cpp
│       ├── index_bindings.cpp
│       └── setup.py
├── cuda/                        # CUDA implementations
│   ├── range_search/            # Range search kernels
│   │   ├── coarse_prune.cu
│   │   ├── hierarchical_search.cu
│   │   └── batch_search.cu
│   ├── sparse_attention/        # Sparse attention kernels
│   │   ├── gather_scatter.cu
│   │   ├── sparse_softmax.cu
│   │   └── fused_attention.cu
│   ├── utils/                   # Utility kernels
│   │   ├── dot_product.cu
│   │   ├── reduce.cu
│   │   └── sorting.cu
│   └── bindings/                # CUDA Python bindings
│       └── setup.py
└── tests/                       # Kernel tests
    ├── test_clustering.py
    ├── test_range_search.py
    └── test_sparse_attention.py
```

## Integration Points

### Python API
Kernels will be exposed through Python bindings that integrate seamlessly with the existing Hira API:

```python
from hira.kernels import (
    fast_kmeans_cuda,           # Accelerated clustering
    hierarchical_search_cuda,   # Optimized range search
    sparse_attention_cuda,      # Fused sparse attention
)
```

### Fallback Behavior
All kernel operations have PyTorch fallback implementations in the main codebase. Kernels are optional and only used when available:

```python
try:
    from hira.kernels import fast_kmeans_cuda
    USE_CUDA_KERNELS = True
except ImportError:
    USE_CUDA_KERNELS = False
```

## Building Kernels

### C++ Kernels
```bash
cd kernels/cpp/bindings
python setup.py install
```

### CUDA Kernels
```bash
cd kernels/cuda/bindings
python setup.py install
```

## Performance Goals

Target speedups over PyTorch implementations:
- **Clustering**: 5-10x faster with CUDA k-means
- **Range Search**: 3-5x faster with fused hierarchical traversal
- **Sparse Attention**: 2-4x faster with custom gather-scatter
- **Index Building**: 2-3x faster with multi-threading

## Dependencies

- **C++ Kernels**: C++17, OpenMP, Eigen (optional)
- **CUDA Kernels**: CUDA 11.8+, cuBLAS, Thrust
- **Bindings**: pybind11, PyTorch C++ API
- **Optional**: FAISS (for clustering), CUB (for primitives)

## References

Inspiration and prior art:
- **FAISS**: Fast similarity search and clustering
- **FlashAttention**: Efficient attention kernels
- **Cutlass**: High-performance GEMM operations
- **MagicPIG**: LSH-based attention
- **RetroInfer**: Retrieval-based attention

## TODOs

- [ ] Implement basic k-means CUDA kernel
- [ ] Create hierarchical search kernel with coarse pruning
- [ ] Develop sparse attention gather-scatter operations
- [ ] Add memory transfer optimizations
- [ ] Benchmark against PyTorch baselines
- [ ] Integrate with main Hira codebase
- [ ] Add comprehensive kernel tests
- [ ] Document kernel APIs and usage

## Contributing

When adding new kernels:
1. Start with a PyTorch reference implementation
2. Profile to identify bottlenecks
3. Implement kernel with optimization strategy
4. Add unit tests comparing against reference
5. Benchmark and document speedups
6. Ensure fallback behavior works correctly
