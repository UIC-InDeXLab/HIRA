"""TA_build_v13.0 — parent-contiguous child layout for mask attention.

Starts from the v11 cluster-streaming build and adds an explicit flattened
parent -> children layout:

    parent_children_i32[s, h, parent * bf + child_slot]

This lets v13 attention mark candidate original key ids directly from selected
parent ids without touching cluster-key/value tiles.
"""

from __future__ import annotations

import torch

from ..v11.TA_build_v_11_0 import build as build_v11_0

KERNEL_VERSION = "v13.0"


def _add_v13_layouts(state: dict) -> None:
    children = state["children_padded_i32"].contiguous()  # (S, H, K, bf)
    state["parent_children_i32"] = children.view(
        int(children.shape[0]),
        int(children.shape[1]),
        int(children.shape[2]) * int(children.shape[3]),
    ).contiguous()


def build(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    values: torch.Tensor | None = None,
) -> dict:
    state = build_v11_0(
        keys=keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=refine_iter,
        values=values,
    )
    _add_v13_layouts(state)
    state["version"] = KERNEL_VERSION
    return state


KERNEL = build
