from .kmeans import cluster_kmeans
from .random_projection import cluster_random_projection
from .pq_subspace import cluster_pq_subspace
from .kcenter import cluster_kcenter
from .pca_pq import cluster_pca_pq
from .whitened_pq import cluster_whitened_pq
from .batch_nn import cluster_batch_nn

CLUSTERING_METHODS = {
    "kmeans": cluster_kmeans,
    "random_proj": cluster_random_projection,
    "pq_subspace": cluster_pq_subspace,
    "kcenter": cluster_kcenter,
    "pca_pq": cluster_pca_pq,
    "whitened_pq": cluster_whitened_pq,
    "batch_nn": cluster_batch_nn,
}
