"""TA_build_v11.0 — alias of v10.0 cluster-streaming layout for v11 attention.

v11 attention uses the same cluster-contiguous (S, H_kv, K, D, BF) /
(S, H_kv, K, BF, D_v) layouts produced by v10.0; this builder just re-labels
the version string so bench output is consistent with the attention version.
"""

from __future__ import annotations

import torch

from ..commons._TA_build_legacy import add_v10_layouts, build_v1_1_state

KERNEL_VERSION = "v11.0"


def build(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    values: torch.Tensor | None = None,
) -> dict:
    if bf != 4 or n_subspaces != 4:
        raise ValueError(
            f"TA_build_v11.0 is specialized for bf=4 and n_subspaces=4; "
            f"got bf={bf}, n_subspaces={n_subspaces}"
        )
    state = build_v1_1_state(
        keys=keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=refine_iter,
        values=values,
    )
    add_v10_layouts(state)
    state["version"] = KERNEL_VERSION
    return state


KERNEL = build
