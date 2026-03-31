from .ball_centroid import enclose_ball_centroid
from .min_enclosing_ball import enclose_min_ball
from .aabb import enclose_aabb
from .cone import enclose_cone

ENCLOSING_METHODS = {
    "ball_centroid": enclose_ball_centroid,
    "min_enclosing_ball": enclose_min_ball,
    "aabb": enclose_aabb,
    "cone": enclose_cone,
}
