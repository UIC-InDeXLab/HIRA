from .ball_centroid import enclose_ball_centroid
from .l1_ball import enclose_l1_ball
from .lp_ball import enclose_lp_ball, make_enclose_lp_ball
from .min_enclosing_ball import enclose_min_ball
from .aabb import enclose_aabb
from .cone import enclose_cone
from .hybrid import enclose_hybrid
from .ellipsoid import enclose_ellipsoid
from .multi_ball import enclose_multi_ball
from .subspace_box import enclose_subspace_box
from .tight_hybrid import enclose_tight_hybrid
from .outlier_aabb import enclose_outlier_aabb
from .outlier_ball_centroid import enclose_outlier_ball_centroid
from .outlier_span_ball import enclose_outlier_span_ball
# from .pca_aabb_residual import enclose_pca_aabb_residual
from .topk_aabb_residual import enclose_topk_aabb_residual
from .centerline import enclose_centerline
from .pca_obb import enclose_pca_obb
from .hybrid_plus import enclose_hybrid_plus
from .split_aabb import enclose_split_aabb
from .quad_aabb import enclose_quad_aabb
from .bisect_aabb import enclose_bisect_aabb
from .slab_bundle import enclose_slab_bundle
from .span_ball import enclose_span_ball
from .axis_interval import enclose_axis_interval
from .dual_axis_interval import enclose_dual_axis_interval
# from .pca_interval import enclose_pca_interval

ENCLOSING_METHODS = {
    "ball_centroid": enclose_ball_centroid,
    "l1_ball": enclose_l1_ball,
    "min_enclosing_ball": enclose_min_ball,
    "aabb": enclose_aabb,
    "cone": enclose_cone,
    # "hybrid": enclose_hybrid,
    "ellipsoid": enclose_ellipsoid,
    "outlier_aabb": enclose_outlier_aabb,
    "outlier_ball_centroid": enclose_outlier_ball_centroid,
    "outlier_span_ball": enclose_outlier_span_ball,
    # "pca_aabb_resid": enclose_pca_aabb_residual,
    # "multi_ball": enclose_multi_ball, => too slow
    # "subspace_box": enclose_subspace_box, => too slow
    # "tight_hybrid": enclose_tight_hybrid, => too slow
    "topk_aabb_residual": enclose_topk_aabb_residual,
    "centerline": enclose_centerline,
    "pca_obb": enclose_pca_obb,
    # "hybrid_plus": enclose_hybrid_plus,
    # "split_aabb": enclose_split_aabb,
    # "quad_aabb": enclose_quad_aabb,
    # "split_hybrid": enclose_split_hybrid,
    # "bisect_aabb": enclose_bisect_aabb,
    "slab_bundle": enclose_slab_bundle,
    # "split_full_hybrid": enclose_split_full_hybrid,
    "span_ball": enclose_span_ball,
    "axis_interval": enclose_axis_interval,
    "dual_axis_interval": enclose_dual_axis_interval,
    # "pca_interval": enclose_pca_interval,
}
