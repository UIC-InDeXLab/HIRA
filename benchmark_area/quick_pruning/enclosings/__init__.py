from .ball_centroid import enclose_ball_centroid
from .min_enclosing_ball import enclose_min_ball
from .aabb import enclose_aabb
from .ellipsoid import enclose_ellipsoid
from .outlier_aabb import enclose_outlier_aabb
from .outlier_ball_centroid import enclose_outlier_ball_centroid
from .outlier_span_ball import enclose_outlier_span_ball

ENCLOSING_METHODS = {
    "ball_centroid": enclose_ball_centroid,
    "min_enclosing_ball": enclose_min_ball,
    "aabb": enclose_aabb,
    "ellipsoid": enclose_ellipsoid,
    "outlier_aabb": enclose_outlier_aabb,
    "outlier_ball_centroid": enclose_outlier_ball_centroid,
}
