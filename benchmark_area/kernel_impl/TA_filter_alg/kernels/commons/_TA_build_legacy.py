"""Shared legacy TA build helpers for v11+/v13+ builders."""

from __future__ import annotations

import math

import torch


def split_contiguous(d: int, s_count: int) -> list[tuple[int, int]]:
    sub = d // s_count
    rem = d % s_count
    out: list[tuple[int, int]] = []
    off = 0
    for idx in range(s_count):
        width = sub + (1 if idx < rem else 0)
        out.append((off, off + width))
        off += width
    return out


def assigns_dtype(k: int) -> torch.dtype:
    return torch.int16 if k < 32768 else torch.int32


def _dominant_axis(points: torch.Tensor) -> torch.Tensor:
    _, d = points.shape
    if points.shape[0] <= 1:
        axis = torch.zeros(d, device=points.device, dtype=points.dtype)
        axis[0] = 1.0
        return axis
    var = points.float().var(dim=0, unbiased=False)
    axis_idx = int(var.argmax().item())
    axis = torch.zeros(d, device=points.device, dtype=points.dtype)
    axis[axis_idx] = 1.0
    return axis


def _target_cluster_sizes(n_points: int, bf: int, device: torch.device) -> torch.Tensor:
    k = max(1, math.ceil(n_points / bf))
    base = n_points // k
    extra = n_points % k
    sizes = torch.full((k,), base, device=device, dtype=torch.long)
    if extra:
        sizes[:extra] += 1
    return sizes


def _assign_balanced_tree_h(
    points: torch.Tensor,
    indices: torch.Tensor,
    leaf_sizes: torch.Tensor,
    assign: torch.Tensor,
    cluster_offset: int,
) -> None:
    if int(leaf_sizes.numel()) == 1:
        assign[indices] = cluster_offset
        return

    mid = int(leaf_sizes.numel()) // 2
    left_sizes = leaf_sizes[:mid]
    right_sizes = leaf_sizes[mid:]
    left_count = int(left_sizes.sum().item())

    subset = points.index_select(0, indices)
    axis = _dominant_axis(subset)
    order = torch.argsort(subset @ axis)
    left_idx = indices.index_select(0, order[:left_count])
    right_idx = indices.index_select(0, order[left_count:])

    _assign_balanced_tree_h(points, left_idx, left_sizes, assign, cluster_offset)
    _assign_balanced_tree_h(points, right_idx, right_sizes, assign, cluster_offset + mid)


def _recompute_centers(keys_h: torch.Tensor, assign_h: torch.Tensor, k: int) -> torch.Tensor:
    n, d = keys_h.shape
    centers = torch.zeros(k, d, device=keys_h.device, dtype=keys_h.dtype)
    centers.scatter_add_(0, assign_h[:, None].expand(n, d), keys_h)
    counts = torch.bincount(assign_h, minlength=k).to(keys_h.dtype).clamp_min(1.0)
    return centers / counts[:, None]


def _balanced_pca_tree_subspace(keys_sub: torch.Tensor, bf: int) -> tuple[torch.Tensor, torch.Tensor]:
    h, n, _d = keys_sub.shape
    device = keys_sub.device
    k = max(1, math.ceil(n / bf))
    target_sizes = _target_cluster_sizes(n, bf, device)

    assign = torch.empty(h, n, device=device, dtype=torch.long)
    centers = torch.empty(h, k, keys_sub.shape[-1], device=device, dtype=keys_sub.dtype)
    all_idx = torch.arange(n, device=device)
    for head in range(h):
        assign_h = torch.empty(n, device=device, dtype=torch.long)
        _assign_balanced_tree_h(keys_sub[head], all_idx, target_sizes, assign_h, 0)
        assign[head] = assign_h
        centers[head] = _recompute_centers(keys_sub[head], assign_h, k)
    return assign, centers


def _children_from_assign(
    assign: torch.Tensor,
    *,
    k: int,
    bf: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, n = assign.shape
    device = assign.device
    children = torch.full((h, k, bf), -1, device=device, dtype=torch.int32)
    counts_out = torch.zeros(h, k, device=device, dtype=torch.int16)

    for head in range(h):
        assign_h = assign[head].to(torch.long)
        counts = torch.bincount(assign_h, minlength=k)
        if bool((counts > bf).any().item()):
            max_count = int(counts.max().item())
            raise RuntimeError(f"capacity violation: max cluster size {max_count} > bf={bf}")
        counts_out[head] = counts.to(torch.int16)

        order = torch.argsort(assign_h, stable=True)
        offsets = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long), counts.cumsum(dim=0)]
        )
        for cluster in range(k):
            count = int(counts[cluster].item())
            if count:
                start = int(offsets[cluster].item())
                children[head, cluster, :count] = order[start : start + count].to(torch.int32)

    return children.contiguous(), counts_out.contiguous()


def build_v1_1_state(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    values: torch.Tensor | None = None,
) -> dict:
    del refine_iter
    if keys.ndim != 3:
        raise ValueError(f"keys must be (H_kv, N, D); got shape {tuple(keys.shape)}")
    if bf <= 0:
        raise ValueError(f"bf must be positive; got {bf}")

    h_kv, n_real, d = keys.shape
    k = max(1, math.ceil(n_real / bf))
    n_pad = k * bf
    pad = n_pad - n_real
    device = keys.device
    dtype = keys.dtype

    if pad:
        keys_padded = torch.cat(
            [keys, torch.zeros(h_kv, pad, d, device=device, dtype=dtype)], dim=1
        )
    else:
        keys_padded = keys

    invalid_mask = torch.zeros(h_kv, n_pad, dtype=torch.bool, device=device)
    if pad:
        invalid_mask[:, n_real:] = True

    slices = split_contiguous(d, n_subspaces)
    widths = [end - start for start, end in slices]
    offsets = [start for start, _ in slices]
    max_w = max(widths)

    centers_per_sub: list[torch.Tensor] = []
    assigns_padded_list: list[torch.Tensor] = []
    children_list: list[torch.Tensor] = []
    counts_list: list[torch.Tensor] = []

    for start, end in slices:
        keys_sub = keys[:, :, start:end].contiguous()
        assign, centers = _balanced_pca_tree_subspace(keys_sub, bf)
        centers_per_sub.append(centers.contiguous())

        ap = torch.zeros(h_kv, n_pad, dtype=torch.long, device=device)
        ap[:, :n_real] = assign
        assigns_padded_list.append(ap.to(assigns_dtype(k)).contiguous())

        children_hkb, counts_hk = _children_from_assign(assign, k=k, bf=bf)
        children_list.append(children_hkb)
        counts_list.append(counts_hk)

    centers_padded = torch.zeros(
        n_subspaces, h_kv, k, max_w, device=device, dtype=torch.float16
    )
    for s_idx, centers in enumerate(centers_per_sub):
        centers_padded[s_idx, :, :, : centers.shape[-1]] = centers.to(torch.float16)
    centers_padded = centers_padded.contiguous()

    keys_padded_f16 = keys_padded.to(torch.float16).contiguous()
    keys_padded_t_f16 = keys_padded_f16.transpose(-1, -2).contiguous()
    assigns_stack = torch.stack(assigns_padded_list, dim=0).contiguous()
    children_stack = torch.stack(children_list, dim=0).contiguous()
    counts_stack = torch.stack(counts_list, dim=0).contiguous()

    state = {
        "version": "v1.1",
        "dim_slices": slices,
        "dim_offsets": torch.tensor(offsets, dtype=torch.int32, device=device),
        "dim_widths": torch.tensor(widths, dtype=torch.int32, device=device),
        "max_width": int(max_w),
        "centers_padded_f16": centers_padded,
        "centers_per_sub": centers_per_sub,
        "assigns_padded": assigns_stack,
        "children_padded_i32": children_stack,
        "cluster_counts_i16": counts_stack,
        "keys_padded_f16": keys_padded_f16,
        "keys_padded_t_f16": keys_padded_t_f16,
        "invalid_mask": invalid_mask.contiguous(),
        "K": k,
        "N": n_real,
        "N_pad": n_pad,
        "bf": bf,
        "D": d,
        "n_subspaces": n_subspaces,
    }

    if values is not None:
        if values.shape[0] != h_kv or values.shape[1] != n_real:
            raise ValueError(
                f"values shape mismatch: expected ({h_kv}, {n_real}, *); got {tuple(values.shape)}"
            )
        d_v = int(values.shape[-1])
        if pad:
            values_padded = torch.cat(
                [
                    values,
                    torch.zeros(h_kv, pad, d_v, device=device, dtype=values.dtype),
                ],
                dim=1,
            )
        else:
            values_padded = values
        values_padded = values_padded.masked_fill(invalid_mask[..., None], 0.0)
        state["values_padded_f16"] = values_padded.to(torch.float16).contiguous()
        state["D_v"] = d_v

    return state


def add_v10_layouts(state: dict) -> None:
    children = state["children_padded_i32"].contiguous()
    keys = state["keys_padded_f16"]
    values = state.get("values_padded_f16")

    s_sub, h_kv, k_clusters, bf = children.shape
    _h, _n_pad, d = keys.shape
    if s_sub != 4 or bf != 4:
        raise ValueError(f"v10 layout requires S=4 and bf=4, got S={s_sub}, bf={bf}")

    valid = children >= 0
    safe_ids = children.clamp_min(0).to(torch.long)
    cluster_keys = torch.empty(
        s_sub, h_kv, k_clusters, bf, d, device=keys.device, dtype=torch.float16
    )

    for h_idx in range(h_kv):
        gathered = keys[h_idx].index_select(0, safe_ids[:, h_idx].reshape(-1))
        gathered = gathered.view(s_sub, k_clusters, bf, d)
        cluster_keys[:, h_idx] = gathered.masked_fill(~valid[:, h_idx, :, :, None], 0.0)

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
