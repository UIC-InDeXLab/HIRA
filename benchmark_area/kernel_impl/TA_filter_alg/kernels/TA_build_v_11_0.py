"""TA_build_v11.0 — alias of v10.0 cluster-streaming layout for v11 attention.

v11 attention uses the same cluster-contiguous (S, H_kv, K, D, BF) /
(S, H_kv, K, BF, D_v) layouts produced by v10.0; this builder just re-labels
the version string so bench output is consistent with the attention version.
"""

from __future__ import annotations

import torch

from .TA_build_v_10_0 import build as build_v10_0

KERNEL_VERSION = "v11.0"


def build(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    values: torch.Tensor | None = None,
) -> dict:
    state = build_v10_0(
        keys=keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=refine_iter,
        values=values,
    )
    state["version"] = KERNEL_VERSION
    return state


KERNEL = build
