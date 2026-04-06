"""Batch-NN balling methods exposed as clustering algorithms."""

from __future__ import annotations

import sys
from pathlib import Path

_bench_root = Path(__file__).resolve().parents[1]
if str(_bench_root) not in sys.path:
    sys.path.insert(0, str(_bench_root))

from balling.batch_nn_ball import ball_batch_nn
from balling.l1_batch_nn import ball_batch_nn_l1, ball_batch_nn_linf, ball_batch_nn_aabb_aware


def cluster_batch_nn(keys, bf):
    return ball_batch_nn(keys, bf)


def cluster_batch_nn_l1(keys, bf):
    return ball_batch_nn_l1(keys, bf)


def cluster_batch_nn_linf(keys, bf):
    return ball_batch_nn_linf(keys, bf)


def cluster_batch_nn_aabb_aware(keys, bf):
    return ball_batch_nn_aabb_aware(keys, bf)
