"""TA_attention_v12.0 — capped top-k depth + duplicate-corrected merged stream.

Specialized for bf=4, S=4.  Replaces full subspace sort with sorted top-k
centroid rows (256 for the 8k/12k captures), computes only TA depth, then
streams all 4 subspaces in one merged attention tile.  If a key appears via
multiple selected subspaces, each copy contributes ``exp(score) / multiplicity``
so the final softmax matches a unique-key candidate set.
"""

from __future__ import annotations

import torch

from ._TA_attention_v12_common import attend_v12

KERNEL_VERSION = "v12.0"


def attend(
    q: torch.Tensor,
    threshold: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
    keys_children: torch.Tensor | None = None,
) -> torch.Tensor:
    return attend_v12(
        q,
        threshold,
        state,
        buffer_keys,
        buffer_values,
        q_head_to_kv,
        scale,
        keys_children,
        selector="topk",
        cache_name="_TA_attention_v12_0_cache",
        graph_flag="_TA_attention_v12_0_use_cuda_graphs",
    )


KERNEL = attend
