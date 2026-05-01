"""TA_build_v10.0 — S=4, bf=4 cluster-streaming layout.

This builder keeps the v1.1 TA index contract and adds the cluster-contiguous
key/value layouts consumed by TA_attention_v10.0.  It is intentionally scoped
to the latency experiment where ``n_subspaces == 4`` and ``bf == 4``.
"""

from __future__ import annotations

import torch

from .TA_build_v_1_1 import build as build_v1_1

KERNEL_VERSION = "v10.0"


def _add_v10_layouts(state: dict) -> None:
    children = state["children_padded_i32"].contiguous()  # (S, H, K, 4)
    keys = state["keys_padded_f16"]
    values = state.get("values_padded_f16")

    s_sub, h_kv, k_clusters, bf = children.shape
    _h, _n_pad, d = keys.shape
    if s_sub != 4 or bf != 4:
        raise ValueError(f"TA_build_v10.0 requires S=4 and bf=4, got S={s_sub}, bf={bf}")

    valid = children >= 0
    safe_ids = children.clamp_min(0).to(torch.long)
    cluster_keys = torch.empty(
        s_sub, h_kv, k_clusters, bf, d, device=keys.device, dtype=torch.float16
    )

    for h_idx in range(h_kv):
        gathered = keys[h_idx].index_select(0, safe_ids[:, h_idx].reshape(-1))
        gathered = gathered.view(s_sub, k_clusters, bf, d)
        cluster_keys[:, h_idx] = gathered.masked_fill(~valid[:, h_idx, :, :, None], 0.0)

    state["version"] = KERNEL_VERSION
    state["cluster_key_ids_i32"] = children
    state["cluster_keys_t_f16"] = cluster_keys.permute(0, 1, 2, 4, 3).contiguous()

    if values is not None:
        d_v = int(values.shape[-1])
        cluster_values = torch.empty(
            s_sub, h_kv, k_clusters, bf, d_v,
            device=values.device,
            dtype=torch.float16,
        )
        for h_idx in range(h_kv):
            gathered = values[h_idx].index_select(0, safe_ids[:, h_idx].reshape(-1))
            gathered = gathered.view(s_sub, k_clusters, bf, d_v)
            cluster_values[:, h_idx] = gathered.masked_fill(
                ~valid[:, h_idx, :, :, None], 0.0
            )
        state["cluster_values_f16"] = cluster_values.contiguous()


def build(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    values: torch.Tensor | None = None,
) -> dict:
    if bf != 4 or n_subspaces != 4:
        raise ValueError(
            f"TA_build_v10.0 is specialized for bf=4 and n_subspaces=4; "
            f"got bf={bf}, n_subspaces={n_subspaces}"
        )
    state = build_v1_1(
        keys=keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=refine_iter,
        values=values,
    )
    _add_v10_layouts(state)
    return state


KERNEL = build
