"""Subspace k-center + ball_centroid index — fused-attention path only.

Per subspace the index stores parents (centers/radii) and a parent-major,
block-packed child layout produced by build_v2_4. At attention time a fused
Triton pipeline gates parents against per-subspace thresholds, evaluates the
online softmax over survivors and the decoding buffer, and merges the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kernels import get_attention, get_build, get_update


@dataclass
class IndexConfig:
    n_subspaces: int = 8
    bf: int = 4
    refine_iter: int = 5
    update_mode: str = "inc"                        # "full" | "inc"
    build_kernel: str = "build_v2_4"
    update_kernel: str = "update_v2_1"
    attention_kernel: str = "attention_v2_6"


class SubspaceKCenterIndex:
    """Holds the index state + key/value history + decoding buffer."""

    def __init__(self, cfg: IndexConfig):
        self.cfg = cfg
        self._build = get_build(cfg.build_kernel)
        self._update = get_update(cfg.update_kernel)
        self._attention = get_attention(cfg.attention_kernel)

        self.state: dict | None = None
        self.keys: torch.Tensor | None = None          # (H_kv, N, D)
        self.values: torch.Tensor | None = None        # (H_kv, N, D_v)
        self.buffer: torch.Tensor | None = None        # (H_kv, B, D)
        self.values_buffer: torch.Tensor | None = None # (H_kv, B, D_v)
        self._steps_since_update = 0

    # ── Build ─────────────────────────────────────────────────────────

    def build(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> "SubspaceKCenterIndex":
        self.keys = keys.contiguous()
        self.values = values.contiguous()
        self.state = self._build(
            keys=self.keys,
            bf=self.cfg.bf,
            n_subspaces=self.cfg.n_subspaces,
            refine_iter=self.cfg.refine_iter,
            values=self.values,
        )
        self.buffer = torch.empty(
            self.keys.shape[0], 0, self.keys.shape[2],
            device=self.keys.device, dtype=self.keys.dtype,
        )
        self.values_buffer = torch.empty(
            self.values.shape[0], 0, self.values.shape[2],
            device=self.values.device, dtype=self.values.dtype,
        )
        self._steps_since_update = 0
        return self

    # ── Decoding-time key/value handling ──────────────────────────────

    def append_decoding_kv(
        self, new_key: torch.Tensor, new_value: torch.Tensor
    ) -> None:
        """Append a matching (k, v) pair to the decoding buffer."""
        if new_key.dim() == 2:
            new_key = new_key.unsqueeze(1)
        if new_value.dim() == 2:
            new_value = new_value.unsqueeze(1)
        self.buffer = torch.cat([self.buffer, new_key], dim=1)
        self.values_buffer = torch.cat([self.values_buffer, new_value], dim=1)
        self._steps_since_update += 1

    def needs_update(self, update_interval: int) -> bool:
        return self._steps_since_update >= update_interval

    def update(self) -> None:
        if self.buffer.shape[1] == 0:
            self._steps_since_update = 0
            return

        if self.cfg.update_mode == "full":
            new_keys = torch.cat([self.keys, self.buffer], dim=1).contiguous()
            new_values = torch.cat(
                [self.values, self.values_buffer], dim=1
            ).contiguous()
            self.state = self._build(
                keys=new_keys,
                bf=self.cfg.bf,
                n_subspaces=self.cfg.n_subspaces,
                refine_iter=self.cfg.refine_iter,
                values=new_values,
            )
            self.keys = new_keys
            self.values = new_values
        elif self.cfg.update_mode == "inc":
            self.state, self.keys, self.values = self._update(
                state=self.state,
                old_keys=self.keys,
                buffer_keys=self.buffer,
                bf=self.cfg.bf,
                n_subspaces=self.cfg.n_subspaces,
                refine_iter=self.cfg.refine_iter,
                old_values=self.values,
                buffer_values=self.values_buffer,
            )
        else:
            raise ValueError(f"Unknown update_mode: {self.cfg.update_mode!r}")

        self.buffer = torch.empty(
            self.keys.shape[0], 0, self.keys.shape[2],
            device=self.keys.device, dtype=self.keys.dtype,
        )
        self.values_buffer = torch.empty(
            self.values.shape[0], 0, self.values.shape[2],
            device=self.values.device, dtype=self.values.dtype,
        )
        self._steps_since_update = 0

    # ── Fused attention ───────────────────────────────────────────────

    def attend(
        self,
        q: torch.Tensor,                               # (H_q, D)
        th_per_subspace: torch.Tensor,                 # (S, H_q)
        q_head_to_kv: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Return (H_q, D_v) attention output over (index + buffer) K/V."""
        return self._attention(
            q=q,
            th_per_subspace=th_per_subspace,
            state=self.state,
            buffer_keys=self.buffer,
            buffer_values=self.values_buffer,
            keys_children=self.keys,
            q_head_to_kv=q_head_to_kv,
            scale=scale,
        )

    # ── Introspection ─────────────────────────────────────────────────

    def last_cluster_pass(self) -> torch.Tensor | None:
        """The most recent cluster-pass mask produced by attend().

        Shape: (S, H_q, K) int8, 1 = parent survives threshold in subspace s
        for query head h_q. AND across S = parents actually scanned by the
        index kernel. Returns None if the active attention kernel doesn't
        expose this or attend() hasn't run yet.

        Safe to read after `attend()` returns — it syncs via _time_gpu. The
        tensor gets overwritten on the next attend() call.
        """
        if self.state is None:
            return None
        for cache_name in (
            "_attn_v1_16_fixed",
            "_attn_v1_17_fixed",
            "_attn_v1_18_fixed",
            "_attn_v1_20_fixed",
            "_attn_v1_22_fixed",
            "_attn_v1_23_fixed",
            "_attn_v1_24_fixed",
            "_attn_v2_6_fixed",
            "_attn_v2_15_fixed",
        ):
            wrap = self.state.get(cache_name)
            if not wrap:
                continue
            fixed = wrap.get("fixed")
            if fixed is None:
                continue
            cluster_pass = fixed["shared"].get("cluster_pass")
            if cluster_pass is not None:
                return cluster_pass
        return None

    @property
    def n_children(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[1])

    @property
    def n_buffered(self) -> int:
        return 0 if self.buffer is None else int(self.buffer.shape[1])

    def memory_bytes(self) -> int:
        """Approximate GPU memory held by the index (keys + values + state)."""
        total = 0
        for t in (self.keys, self.values, self.buffer, self.values_buffer):
            if t is not None:
                total += t.element_size() * t.numel()
        if self.state is not None:
            for v in self.state.values():
                if isinstance(v, torch.Tensor):
                    total += v.element_size() * v.numel()
                elif isinstance(v, list):
                    for t in v:
                        if isinstance(t, torch.Tensor):
                            total += t.element_size() * t.numel()
        return total


def baseline_attention(
    q: torch.Tensor,                                  # (H_q, D)
    keys: torch.Tensor,                               # (H_kv, N, D)
    values: torch.Tensor,                             # (H_kv, N, D_v)
    q_head_to_kv: torch.Tensor | None = None,         # kept for API parity
    scale: float | None = None,
) -> torch.Tensor:
    """Brute-force dense attention: softmax(scale * q @ K^T) @ V → (H_q, D_v).

    GQA-aware: when H_q != H_kv we reshape Q into (groups, H_kv, D) and
    broadcast against the original (H_kv, N, D) keys rather than materializing
    an (H_q, N, D) expansion — same work as `enable_gqa=True` for SDPA.
    """
    import math

    h_q, d = q.shape
    h_kv, n, _ = keys.shape
    d_v = values.shape[-1]
    scale = 1.0 / math.sqrt(d) if scale is None else float(scale)

    if h_q == h_kv:
        scores = torch.einsum("hd,hnd->hn", q, keys) * scale
        probs = torch.softmax(scores.float(), dim=-1).to(values.dtype)
        return torch.einsum("hn,hnd->hd", probs, values)

    if h_q % h_kv != 0:
        raise ValueError(f"H_q={h_q} must be a multiple of H_kv={h_kv}")
    groups = h_q // h_kv
    # Q heads are kv-major: q_head i -> kv_head i // groups (matches
    # _q_to_kv_map and SDPA's enable_gqa convention).
    q_hg = q.view(h_kv, groups, d)
    scores = torch.einsum("hgd,hnd->hgn", q_hg, keys) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(values.dtype)
    out = torch.einsum("hgn,hnd->hgd", probs, values)
    return out.reshape(h_q, d_v)


def baseline_sdpa(
    q: torch.Tensor,                                  # (H_q, D)
    keys: torch.Tensor,                               # (H_kv, N, D)
    values: torch.Tensor,                             # (H_kv, N, D_v)
    q_head_to_kv: torch.Tensor | None = None,         # kept for API parity
    scale: float | None = None,
) -> torch.Tensor:
    """torch.nn.functional.scaled_dot_product_attention with native GQA.

    Uses `enable_gqa=True` so K/V stay at H_kv — no (H_q, N, D) expansion
    inside the timed region. Requires PyTorch >= 2.5.
    """
    import math

    h_q, d = q.shape
    h_kv, n, _ = keys.shape
    d_v = values.shape[-1]
    scale = 1.0 / math.sqrt(d) if scale is None else float(scale)

    q4 = q.view(1, h_q, 1, d)
    k4 = keys.view(1, h_kv, n, d)
    v4 = values.view(1, h_kv, n, d_v)
    out = torch.nn.functional.scaled_dot_product_attention(
        q4, k4, v4, is_causal=False, scale=scale,
        enable_gqa=(h_q != h_kv),
    )
    return out.view(h_q, d_v)
