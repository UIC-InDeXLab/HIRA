"""
Placeholder for clustering kernel bindings.

Future C++ clustering implementations will be exposed here.
"""

# TODO: Implement pybind11 bindings for:
# - fast_kmeans_cpu
# - fast_kmeans_cuda
# - online_clustering
# - hierarchical_kmeans

def fast_kmeans_available() -> bool:
    """Check if fast k-means kernels are available."""
    return False
