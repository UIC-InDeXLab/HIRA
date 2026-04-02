from .ball_centroid import enclose_ball_centroid
from .min_enclosing_ball import enclose_min_ball
from .aabb import enclose_aabb
from .cone import enclose_cone
from .hybrid import enclose_hybrid
from .ellipsoid import enclose_ellipsoid
from .multi_ball import enclose_multi_ball
from .subspace_box import enclose_subspace_box
from .tight_hybrid import enclose_tight_hybrid
from .topk_aabb_residual import enclose_topk_aabb_residual
from .centerline import enclose_centerline
from .pca_obb import enclose_pca_obb
from .hybrid_plus import enclose_hybrid_plus
from .split_aabb import enclose_split_aabb
from .quad_aabb import enclose_quad_aabb
from .bisect_aabb import enclose_bisect_aabb
from .slab_bundle import enclose_slab_bundle
from .span_ball import enclose_span_ball

ENCLOSING_METHODS = {
    "ball_centroid": enclose_ball_centroid,
    "min_enclosing_ball": enclose_min_ball,
    "aabb": enclose_aabb,
    "cone": enclose_cone,
    "hybrid": enclose_hybrid,
    "ellipsoid": enclose_ellipsoid,
    # "multi_ball": enclose_multi_ball, => too slow
    # "subspace_box": enclose_subspace_box, => too slow
    # "tight_hybrid": enclose_tight_hybrid, => too slow
    "topk_aabb_residual": enclose_topk_aabb_residual,
    "centerline": enclose_centerline,
    "pca_obb": enclose_pca_obb,
    # "hybrid_plus": enclose_hybrid_plus,
    "split_aabb": enclose_split_aabb,
    # "quad_aabb": enclose_quad_aabb,
    # "split_hybrid": enclose_split_hybrid,
    "bisect_aabb": enclose_bisect_aabb,
    "slab_bundle": enclose_slab_bundle,
    # "split_full_hybrid": enclose_split_full_hybrid,
    "span_ball": enclose_span_ball,
}
