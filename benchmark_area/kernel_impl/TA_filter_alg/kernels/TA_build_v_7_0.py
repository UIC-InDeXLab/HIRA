"""TA_build_v7.0 — capped TA index plus cluster-contiguous v7 layouts.

This builder preserves the v1.1 state contract and adds layouts used by the
v7 attention experiments.  Build time and memory are intentionally traded for
decode-time locality.
"""

from __future__ import annotations

import torch

from .TA_build_v_1_1 import build as build_v1_1

KERNEL_VERSION = "v7.0"


def _add_cluster_layouts(state: dict, values: torch.Tensor | None) -> None:
    children = state["children_padded_i32"]
    keys = state["keys_padded_f16"]
    values_padded = state.get("values_padded_f16")

    s_sub, h_kv, k_clusters, bf = children.shape
    _h, _n_pad, d = keys.shape
    d_v = int(values_padded.shape[-1]) if values_padded is not None else 0
    device = keys.device

    cluster_keys = torch.zeros(
        s_sub, h_kv, k_clusters, bf, d, device=device, dtype=torch.float16
    )
    cluster_values = None
    if values_padded is not None:
        cluster_values = torch.zeros(
            s_sub, h_kv, k_clusters, bf, d_v, device=device, dtype=torch.float16
        )

    # Full-vector metadata for conservative Cauchy upper bounds in v7.1.
    full_centers = torch.zeros(
        s_sub, h_kv, k_clusters, d, device=device, dtype=torch.float32
    )
    full_radii = torch.zeros(s_sub, h_kv, k_clusters, device=device, dtype=torch.float32)
    subspace_radii = torch.zeros(
        s_sub, h_kv, k_clusters, device=device, dtype=torch.float32
    )

    for s_idx in range(s_sub):
        start, end = state["dim_slices"][s_idx]
        width = end - start
        for h_idx in range(h_kv):
            ids = children[s_idx, h_idx].to(torch.long)
            valid = ids >= 0
            safe_ids = ids.clamp_min(0)

            gathered_k = keys[h_idx].index_select(0, safe_ids.reshape(-1)).view(
                k_clusters, bf, d
            )
            gathered_k = gathered_k.masked_fill(~valid[..., None], 0.0)
            cluster_keys[s_idx, h_idx] = gathered_k

            counts = valid.sum(dim=1).clamp_min(1).to(torch.float32)
            centers = gathered_k.float().sum(dim=1) / counts[:, None]
            full_centers[s_idx, h_idx] = centers

            diff = (gathered_k.float() - centers[:, None, :]).masked_fill(
                ~valid[..., None], 0.0
            )
            dist = torch.linalg.vector_norm(diff, dim=-1).masked_fill(~valid, 0.0)
            # Small epsilon keeps the stored fp32 bound conservative under minor
            # arithmetic differences between build-time torch and Triton.
            full_radii[s_idx, h_idx] = dist.max(dim=1).values + 1e-4

            centers_sub = state["centers_padded_f16"][
                s_idx, h_idx, :, :width
            ].float()
            diff_sub = (
                gathered_k[..., start:end].float() - centers_sub[:, None, :]
            ).masked_fill(~valid[..., None], 0.0)
            dist_sub = torch.linalg.vector_norm(diff_sub, dim=-1).masked_fill(
                ~valid, 0.0
            )
            subspace_radii[s_idx, h_idx] = dist_sub.max(dim=1).values + 1e-4

            if cluster_values is not None:
                gathered_v = values_padded[h_idx].index_select(
                    0, safe_ids.reshape(-1)
                ).view(k_clusters, bf, d_v)
                cluster_values[s_idx, h_idx] = gathered_v.masked_fill(
                    ~valid[..., None], 0.0
                )

    state["version"] = KERNEL_VERSION
    state["cluster_key_ids_i32"] = children.contiguous()
    state["cluster_keys_f16"] = cluster_keys.contiguous()
    state["cluster_keys_t_f16"] = cluster_keys.permute(0, 1, 2, 4, 3).contiguous()
    state["cluster_full_centers_f32"] = full_centers.contiguous()
    state["cluster_full_radii_f32"] = full_radii.contiguous()
    state["cluster_subspace_radii_f32"] = subspace_radii.contiguous()
    if cluster_values is not None:
        state["cluster_values_f16"] = cluster_values.contiguous()


def build(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    values: torch.Tensor | None = None,
) -> dict:
    state = build_v1_1(
        keys=keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=refine_iter,
        values=values,
    )
    _add_cluster_layouts(state, values)
    return state


KERNEL = build
