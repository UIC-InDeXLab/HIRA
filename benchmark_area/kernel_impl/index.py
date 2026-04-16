"""Subspace k-center + ball_centroid index.

A clean wrapper around pluggable kernel implementations in `kernels/`.

Per subspace:
    assigns[s]: (H, N) int64 — point -> cluster id
    centers[s]: (H, K, d_s) — parent layer
    radii[s]:   (H, K) — ball radius
    child_order[s]:   (H, N) int64 — parent-major child permutation
    child_offsets[s]: (H, K + 1) int32 — offsets into child_order
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kernels import get_build, get_search, get_update


@dataclass
class IndexConfig:
    n_subspaces: int = 4
    bf: int = 4
    refine_iter: int = 5
    update_mode: str = "inc"                    # "full" | "inc"
    build_kernel: str = "build_v1_0"
    search_kernel: str = "search_v1_0"
    update_kernel: str = "update_v1_0"


class SubspaceKCenterIndex:
    """Holds the index state + key history + decoding buffer."""

    def __init__(self, cfg: IndexConfig):
        self.cfg = cfg
        self._build = get_build(cfg.build_kernel)
        self._search = get_search(cfg.search_kernel)
        self._update = get_update(cfg.update_kernel)

        self.state: dict | None = None
        self.keys: torch.Tensor | None = None          # (H, N, D) — in-index keys
        self.buffer: torch.Tensor | None = None        # (H, B, D) — pending keys
        self._steps_since_update = 0

    # ── Build ─────────────────────────────────────────────────────────

    def build(self, keys: torch.Tensor) -> "SubspaceKCenterIndex":
        self.keys = keys.contiguous()
        self.state = self._build(
            self.keys, self.cfg.bf, self.cfg.n_subspaces, self.cfg.refine_iter
        )
        self.buffer = torch.empty(
            keys.shape[0], 0, keys.shape[2], device=keys.device, dtype=keys.dtype
        )
        self._steps_since_update = 0
        return self

    # ── Decoding-time key handling ────────────────────────────────────

    def append_decoding_key(self, new_key: torch.Tensor):
        """new_key: (H, 1, D) or (H, D)."""
        if new_key.dim() == 2:
            new_key = new_key.unsqueeze(1)
        self.buffer = torch.cat([self.buffer, new_key], dim=1)
        self._steps_since_update += 1

    def needs_update(self, update_interval: int) -> bool:
        return self._steps_since_update >= update_interval

    def update(self) -> None:
        if self.buffer is None or self.buffer.shape[1] == 0:
            self._steps_since_update = 0
            return
        self.state, self.keys = self._update(
            self.state,
            self.keys,
            self.buffer,
            self.cfg.bf,
            self.cfg.n_subspaces,
            self.cfg.refine_iter,
            self.cfg.update_mode,
        )
        self.buffer = torch.empty_like(self.buffer[:, :0, :])
        self._steps_since_update = 0

    # ── Search ────────────────────────────────────────────────────────

    def search(
        self,
        q: torch.Tensor,                           # (H_q, D)
        th_per_subspace: torch.Tensor,             # (S, H_q)
        q_head_to_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (H_q, N_total) dot products with -inf at non-survivors.

        N_total = |index children| + |buffer|.
        """
        return self._search(
            q=q,
            th_per_subspace=th_per_subspace,
            state=self.state,
            buffer_keys=self.buffer,
            keys_children=self.keys,
            q_head_to_kv=q_head_to_kv,
        )

    # ── Introspection ─────────────────────────────────────────────────

    @property
    def n_children(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[1])

    @property
    def n_buffered(self) -> int:
        return 0 if self.buffer is None else int(self.buffer.shape[1])

    def memory_bytes(self) -> int:
        """Approximate GPU memory held by the index (keys + state)."""
        total = 0
        if self.keys is not None:
            total += self.keys.element_size() * self.keys.numel()
        if self.buffer is not None:
            total += self.buffer.element_size() * self.buffer.numel()
        if self.state is not None:
            tensors = self.state["assigns"] + self.state["centers"] + self.state["radii"]
            tensors += self.state.get("child_order", [])
            tensors += self.state.get("child_offsets", [])
            tensors += self.state.get("child_counts", [])
            for t in tensors:
                total += t.element_size() * t.numel()
        return total


def baseline_dot(
    q: torch.Tensor,
    keys: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Brute-force baseline: q · every key."""
    keys_q = keys if q_head_to_kv is None else keys[q_head_to_kv]
    return torch.einsum("hd,hnd->hn", q, keys_q)
