from .kmeans import cluster_kmeans
from .spherical_kmeans import cluster_spherical_kmeans
from .random_projection import cluster_random_projection
from .random_partition import cluster_random_partition
from .pq_subspace import cluster_pq_subspace
from .gmm_diag import cluster_gmm_diag

CLUSTERING_METHODS = {
    "kmeans": cluster_kmeans,
    "spherical_kmeans": cluster_spherical_kmeans,
    "random_proj": cluster_random_projection,
    "random_partition": cluster_random_partition,
    "pq_subspace": cluster_pq_subspace,
    "gmm_diag": cluster_gmm_diag,
}
