"""TA_build_v13.4 — arithmetic parent->child layout.

This build is specialized for the no-lookup v13.4 attention path.  Parent
``p`` owns key rows ``p * bf : (p + 1) * bf`` directly, so attention can turn a
selected-parent mask into candidate key rows with integer arithmetic only.

This deliberately does not store an explicit parent->children tensor.  It also
means parent ids are contiguous token chunks, not the older per-subspace
k-center leaves.
"""

from __future__ import annotations

import math

import torch

from .TA_build_v_1_1 import _assigns_dtype, _split_contiguous

KERNEL_VERSION = "v13.4"


def build(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    refine_iter: int = 5,
    values: torch.Tensor | None = None,
) -> dict:
    del refine_iter
    if keys.ndim != 3:
        raise ValueError(f"keys must be (H_kv, N, D); got shape {tuple(keys.shape)}")
    if bf != 4 or n_subspaces != 4:
        raise ValueError(
            f"TA_build_v13.4 is specialized for bf=4 and S=4; got bf={bf}, S={n_subspaces}"
        )

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
    keys_padded = keys_padded.contiguous()

    invalid_mask = torch.zeros(h_kv, n_pad, dtype=torch.bool, device=device)
    if pad:
        invalid_mask[:, n_real:] = True

    slices = _split_contiguous(d, n_subspaces)
    widths = [end - start for start, end in slices]
    offsets = [start for start, _ in slices]
    max_w = max(widths)

    valid = (~invalid_mask).to(keys_padded.dtype).view(h_kv, k, bf, 1)
    centers_per_sub: list[torch.Tensor] = []
    centers_padded = torch.zeros(
        n_subspaces, h_kv, k, max_w, device=device, dtype=torch.float16
    )
    for s_idx, (start, end) in enumerate(slices):
        width = end - start
        sub = keys_padded[:, :, start:end].view(h_kv, k, bf, width)
        sums = (sub * valid).sum(dim=2)
        counts = valid.sum(dim=2).clamp_min(1.0)
        centers = sums / counts
        centers_per_sub.append(centers.contiguous())
        centers_padded[s_idx, :, :, :width] = centers.to(torch.float16)
    centers_padded = centers_padded.contiguous()

    assign_base = (torch.arange(n_pad, device=device) // bf).to(_assigns_dtype(k))
    assigns_hn = assign_base.unsqueeze(0).expand(h_kv, n_pad).contiguous()
    assigns_stack = assigns_hn.unsqueeze(0).expand(n_subspaces, h_kv, n_pad).contiguous()

    counts_base = torch.full((k,), bf, device=device, dtype=torch.int16)
    if pad:
        counts_base[-1] = bf - pad
    counts_hk = counts_base.unsqueeze(0).expand(h_kv, k).contiguous()
    counts_stack = counts_hk.unsqueeze(0).expand(n_subspaces, h_kv, k).contiguous()

    keys_padded_f16 = keys_padded.to(torch.float16).contiguous()
    state = {
        "version": KERNEL_VERSION,
        "dim_slices": slices,
        "dim_offsets": torch.tensor(offsets, dtype=torch.int32, device=device),
        "dim_widths": torch.tensor(widths, dtype=torch.int32, device=device),
        "max_width": int(max_w),
        "centers_padded_f16": centers_padded,
        "centers_per_sub": centers_per_sub,
        "assigns_padded": assigns_stack,
        "cluster_counts_i16": counts_stack,
        "keys_padded_f16": keys_padded_f16,
        "keys_padded_t_f16": keys_padded_f16.transpose(-1, -2).contiguous(),
        "invalid_mask": invalid_mask.contiguous(),
        "K": k,
        "N": n_real,
        "N_pad": n_pad,
        "bf": bf,
        "D": d,
        "n_subspaces": n_subspaces,
        "arithmetic_parent_children": True,
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


KERNEL = build
