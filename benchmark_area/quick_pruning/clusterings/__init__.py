from .kmeans import cluster_kmeans
from .spherical_kmeans import cluster_spherical_kmeans
from .random_projection import cluster_random_projection
from .random_partition import cluster_random_partition
from .pq_subspace import cluster_pq_subspace
from .gmm_diag import cluster_gmm_diag
from .kcenter import cluster_kcenter
from .pca_bisect import cluster_pca_bisect
from .kmeans_pp import cluster_kmeans_pp
from .linf_kmeans import cluster_linf_kmeans
from .pq_linf import cluster_pq_linf
from .pq_l2 import cluster_pq_l2
from .span_kmeans import cluster_span_kmeans
from .pca_pq import cluster_pca_pq
from .pq_span import cluster_pq_span
from .whitened_pq import cluster_whitened_pq

CLUSTERING_METHODS = {
    "kmeans": cluster_kmeans,
    # "spherical_kmeans": cluster_spherical_kmeans, => not good pruning
    "random_proj": cluster_random_projection,
    "random_partition": cluster_random_partition,
    "pq_subspace": cluster_pq_subspace,
    "gmm_diag": cluster_gmm_diag,
    "kcenter": cluster_kcenter,
    "kmeans_pp": cluster_kmeans_pp,
    # "kdtree": cluster_kdtree,
    "linf_kmeans": cluster_linf_kmeans,
    # "pq8": cluster_pq8,
    "pq_linf": cluster_pq_linf,
    "pq_l2": cluster_pq_l2,
    "span_kmeans": cluster_span_kmeans,
    "pca_pq": cluster_pca_pq,
    "pq_span": cluster_pq_span,
    "whitened_pq": cluster_whitened_pq,
    # "whitened_pq_span": cluster_whitened_pq_span,
    # "interleaved_whitened_pq": cluster_interleaved_whitened_pq,
    # "pca_bisect": cluster_pca_bisect, => Too slow
}
