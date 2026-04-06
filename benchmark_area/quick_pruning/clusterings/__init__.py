from .kmeans import cluster_kmeans
from .spherical_kmeans import cluster_spherical_kmeans
from .random_projection import cluster_random_projection
from .random_partition import cluster_random_partition
from .pq_subspace import cluster_pq_subspace
# from .gmm_diag import cluster_gmm_diag
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
from .direction_kmeans import cluster_direction_kmeans
from .shell_kmeans import cluster_shell_kmeans
from .dirnorm_pq import cluster_dirnorm_pq
from .balanced_kmeans import cluster_balanced_kmeans
from .balanced_kcenter import cluster_balanced_kcenter
from .balanced_pca_tree import cluster_balanced_pca_tree
from .pca_morton_chunk import cluster_pca_morton_chunk
from .balanced_ray_kmeans import cluster_balanced_ray_kmeans
from .balanced_ray_kcenter import cluster_balanced_ray_kcenter
from .ball_ratio_kmeans import cluster_ball_ratio_kmeans, cluster_ball_ratio_kcenter
from .ray_kmeans import cluster_ray_kmeans, cluster_ray_kcenter, cluster_ray_kcenter_meb
from .aabb_chunks import cluster_pca_axis_chunk, cluster_pca_morton_span
from .pq_aabb_refine import cluster_pq_span_refine, cluster_pq_balanced_span
from .batch_nn import cluster_batch_nn, cluster_batch_nn_l1, cluster_batch_nn_linf, cluster_batch_nn_aabb_aware
from .batch_nn_lp import cluster_batch_nn_lp, make_cluster_batch_nn_lp
from .kcenter_meb import cluster_kcenter_meb
from .kcenter_minimax import cluster_kcenter_minimax
from .kcenter_lp import cluster_kcenter_lp, make_cluster_kcenter_lp
# from .kdtree_partition import cluster_kcenter_linf

CLUSTERING_METHODS = {
    "kmeans": cluster_kmeans,
    # "spherical_kmeans": cluster_spherical_kmeans, => not good pruning
    "random_proj": cluster_random_projection,
    "random_partition": cluster_random_partition,
    "pq_subspace": cluster_pq_subspace,
    # "gmm_diag": cluster_gmm_diag,
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
    "direction_kmeans": cluster_direction_kmeans,
    "shell_kmeans": cluster_shell_kmeans,
    "dirnorm_pq": cluster_dirnorm_pq,
    "balanced_kmeans": cluster_balanced_kmeans,
    "balanced_kcenter": cluster_balanced_kcenter,
    "balanced_pca_tree": cluster_balanced_pca_tree,
    "pca_morton_chunk": cluster_pca_morton_chunk,
    "balanced_ray_kmeans": cluster_balanced_ray_kmeans,
    "balanced_ray_kcenter": cluster_balanced_ray_kcenter,
    "ball_ratio_kmeans": cluster_ball_ratio_kmeans,
    "ball_ratio_kcenter": cluster_ball_ratio_kcenter,
    "ray_kmeans": cluster_ray_kmeans,
    "ray_kcenter": cluster_ray_kcenter,
    "ray_kcenter_meb": cluster_ray_kcenter_meb,
    "pca_axis_chunk": cluster_pca_axis_chunk,
    "pca_morton_span": cluster_pca_morton_span,
    "pq_span_refine": cluster_pq_span_refine,
    "pq_balanced_span": cluster_pq_balanced_span,
    "batch_nn": cluster_batch_nn,
    "batch_nn_l1": cluster_batch_nn_l1,
    "batch_nn_linf": cluster_batch_nn_linf,
    "batch_nn_aabb_aware": cluster_batch_nn_aabb_aware,
    "batch_nn_lp": cluster_batch_nn_lp,
    "kcenter_meb": cluster_kcenter_meb,
    "kcenter_minimax": cluster_kcenter_minimax,
    "kcenter_lp": cluster_kcenter_lp,
    # "kcenter_linf": cluster_kcenter_linf,
    # "whitened_pq_span": cluster_whitened_pq_span,
    # "interleaved_whitened_pq": cluster_interleaved_whitened_pq,
    # "pca_bisect": cluster_pca_bisect, => Too slow
}
