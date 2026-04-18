"""Micro-benchmark: compare update_v2_* kernels (all operate on build_v2_4 state).

Measures how long it takes to fold a small buffer of new (key, value) rows
into an existing v2_4 index, and verifies that the attention kernel
(v1_5, the generic fallback) still produces correct output on the updated
state (loose gate vs dense reference).

Usage:
    python -m hira.benchmark_area.kernel_impl.kernels.kernel_bench.bench_update
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.kernels.build_v2_4 import build as build_v2_4
from hira.benchmark_area.kernel_impl.kernels import update_kernels, attention_kernels
from hira.benchmark_area.kernel_impl.kernels._attention_triton_v1_5 import (
    triton_fused_cluster_pass_rawq,
)

# Only include the v2.4-compatible updates.
UPDATE_WHITELIST = {
    "update_v2_0",
    "update_v2_1",
    "update_v2_2",
    "update_v2_3",
    "update_v2_4",
    "update_v2_5",
}
SUMMARY_ORDER = (
    "update_v2_0",
    "update_v2_1",
    "update_v2_2",
    "update_v2_3",
    "update_v2_4",
    "update_v2_5",
)


def time_call(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def _rebuild_reference(old_keys, buffer_keys, old_values, buffer_values, bf, S, refine_iter):
    full_keys = torch.cat([old_keys, buffer_keys], dim=1).contiguous()
    full_values = torch.cat([old_values, buffer_values], dim=1).contiguous()
    return build_v2_4(full_keys, bf, S, refine_iter, values=full_values), full_keys, full_values


def _dense_attention(q, keys_q, values_q, scale):
    scores = torch.einsum("hd,hnd->hn", q, keys_q) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hn,hnd->hd", probs, values_q)


def _subspace_topk_thresholds(q, keys, topk, dim_slices):
    """Per-subspace threshold = min over true top-k of q·k restricted to subspace."""
    scores = torch.einsum("hd,hnd->hn", q, keys)
    k = min(topk, scores.shape[-1])
    topk_idx = scores.topk(k, dim=-1).indices  # (H, topk)
    ths = []
    for s, e in dim_slices:
        qs = q[:, s:e]
        ks = keys[:, :, s:e]
        ss = torch.einsum("hd,hnd->hn", qs, ks)
        sub_top = ss.gather(1, topk_idx)
        ths.append(sub_top.min(dim=1).values)
    return torch.stack(ths, dim=0).contiguous(), topk_idx


def _pruning_stats(state, q, keys_q_expanded, full_keys, topk, q_head_to_kv):
    """Return (kept_frac, topk_recall) for this state under the tight gate.

    kept_frac    = # surviving real children / # real children (lower = more pruning)
    topk_recall  = # true top-k children that survive / topk (higher = better quality)
    """
    h_q = q.shape[0]
    h_kv = int(state["keys_reord"].shape[0])
    k_parents = int(state["K"])
    dim_slices = state["dim_slices"]
    groups = h_q // h_kv

    # Tight thresholds from the dense reference top-k over the expanded K keys.
    th, topk_idx_merged = _subspace_topk_thresholds(q, keys_q_expanded, topk, dim_slices)

    # Pack centers into (S, H_kv, K, max_d) padded tensor.
    widths = [e - s for s, e in dim_slices]
    max_d = max(widths)
    s_dim = len(dim_slices)
    centers = torch.zeros(s_dim, h_kv, k_parents, max_d, device=q.device, dtype=torch.float32)
    for idx, c in enumerate(state["centers"]):
        centers[idx, :, :, : c.shape[-1]] = c
    radii = torch.stack(state["radii"], dim=0).contiguous()

    dim_offsets_t = torch.tensor([s for s, _ in dim_slices], device=q.device, dtype=torch.int32)
    dim_widths_t = torch.tensor(widths, device=q.device, dtype=torch.int32)

    cluster_pass = triton_fused_cluster_pass_rawq(
        q=q.contiguous(), th=th.contiguous(),
        dim_offsets=dim_offsets_t, dim_widths=dim_widths_t,
        centers=centers.contiguous(), radii=radii,
        groups=groups,
    )  # (S, H_q, K) int8

    # Per-child survival: AND across subspaces of cluster_pass at that child's
    # per-subspace assigned cluster. assigns_reord[s] is (H_kv, N_pad) int.
    # Expand to (H_q, N_pad) via q_head_to_kv (GQA).
    invalid_mask = state["invalid_mask"].index_select(0, q_head_to_kv)  # (H_q, N_pad)
    survive = torch.ones_like(invalid_mask)  # (H_q, N_pad) bool
    for s_idx in range(s_dim):
        a = state["assigns_reord"][s_idx].index_select(0, q_head_to_kv).long()  # (H_q, N_pad)
        gate = cluster_pass[s_idx].gather(1, a) != 0                              # (H_q, N_pad)
        survive &= gate
    survive &= ~invalid_mask

    # reorder_perm[h, j] gives the original index (into merged keys) for physical j.
    perm = state["reorder_perm"].index_select(0, q_head_to_kv).long()  # (H_q, N_pad)
    # For invalid slots, perm may be arbitrary; mask them out.
    n_real = int(state["N"])

    kept_frac = survive.float().mean(dim=-1).mean().item()

    # Recall: are the true top-k original indices surviving?
    # topk_idx_merged is per-head in merged-key space, shape (H_q, topk).
    # Invert perm to lookup physical position for each original index.
    # We do it per head via scatter.
    n_pad = perm.shape[-1]
    inv_perm = torch.full((h_q, n_real), -1, device=q.device, dtype=torch.long)
    valid_phys = ~invalid_mask
    for h in range(h_q):
        valid_j = valid_phys[h].nonzero(as_tuple=True)[0]
        orig_idx = perm[h, valid_j]
        inv_perm[h, orig_idx] = valid_j
    recall_counts = []
    for h in range(h_q):
        phys_for_topk = inv_perm[h, topk_idx_merged[h]]
        in_index = phys_for_topk >= 0
        survived = survive[h, phys_for_topk.clamp_min(0)] & in_index
        recall_counts.append(survived.sum().item() / max(1, topk))
    recall = sum(recall_counts) / len(recall_counts)

    return kept_frac, recall


def _run_for_B(
    args,
    B: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    base_state: dict,
    q_head_to_kv: torch.Tensor,
    q_batch: torch.Tensor,
    *,
    verbose: bool,
) -> dict:
    """Run timing + pruning/correctness for a single buffer size B.

    Returns a dict keyed by kernel name with fields:
        ms, kept_frac, recall, corr_abs
    Plus an entry under "fresh (rebuild)" with ms/kept_frac/recall.
    """
    device = keys.device
    torch.manual_seed(1000 + B)  # Different buffer per B, reproducible.
    buffer_keys = torch.randn(args.H_kv, B, args.D, device=device, dtype=torch.float32)
    buffer_values = torch.randn(args.H_kv, B, args.D_v, device=device, dtype=torch.float32)

    def fresh_build():
        full_keys = torch.cat([keys, buffer_keys], dim=1).contiguous()
        full_values = torch.cat([values, buffer_values], dim=1).contiguous()
        build_v2_4(full_keys, args.bf, args.S, args.refine_iter, values=full_values)

    ms_fresh = time_call(fresh_build, iters=args.iters, warmup=2)
    if verbose:
        print(f"  {'build_v2_4 (fresh)':<26s} {'ref':<8s}  {ms_fresh:8.2f} ms")

    kept_states: dict[str, dict] = {}
    kernel_ms: dict[str, float] = {}
    for name, info in sorted(update_kernels().items()):
        if name not in UPDATE_WHITELIST:
            continue
        fn = info.fn

        def call(fn=fn):
            fn(
                base_state, keys, buffer_keys,
                args.bf, args.S, args.refine_iter,
                old_values=values, buffer_values=buffer_values,
            )

        try:
            ms = time_call(call, iters=args.iters, warmup=2)
        except Exception as exc:
            if verbose:
                print(f"  {name:<26s} {info.version:<8s}  FAIL {type(exc).__name__}: {exc}")
            continue
        if verbose:
            print(f"  {name:<26s} {info.version:<8s}  {ms:8.2f} ms  "
                  f"({ms_fresh / ms:5.1f}x vs fresh)")
        kernel_ms[name] = ms
        new_state, _, _ = fn(
            base_state, keys, buffer_keys,
            args.bf, args.S, args.refine_iter,
            old_values=values, buffer_values=buffer_values,
        )
        kept_states[name] = new_state

    # Fresh rebuild state for pruning reference.
    fresh_state, _, _ = _rebuild_reference(
        keys, buffer_keys, values, buffer_values,
        args.bf, args.S, args.refine_iter,
    )
    stats_states = {"fresh (rebuild)": fresh_state, **kept_states}

    # Correctness (attention_v1_5, loose gate vs dense).
    attn = attention_kernels().get("attention_v1_5")
    full_keys = torch.cat([keys, buffer_keys], dim=1)
    full_values = torch.cat([values, buffer_values], dim=1)
    keys_expanded = full_keys.index_select(0, q_head_to_kv)
    values_expanded = full_values.index_select(0, q_head_to_kv)
    scale = 1.0 / math.sqrt(args.D)
    corr_abs: dict[str, float] = {}
    if attn is not None:
        q0 = q_batch[0]
        out_ref = _dense_attention(q0, keys_expanded, values_expanded, scale)
        empty_buf = torch.empty(args.H_kv, 0, args.D, device=device, dtype=torch.float32)
        empty_val = torch.empty(args.H_kv, 0, args.D_v, device=device, dtype=torch.float32)
        for name, st in kept_states.items():
            s_eff = len(st["assigns_reord"])
            th_loose = torch.full((s_eff, q_batch.shape[1]), -1e9,
                                  device=device, dtype=torch.float32)
            out_ours = attn.fn(
                q=q0, th_per_subspace=th_loose, state=st,
                buffer_keys=empty_buf, buffer_values=empty_val,
                keys_children=full_keys, q_head_to_kv=q_head_to_kv, scale=scale,
            )
            corr_abs[name] = (out_ours.float() - out_ref.float()).abs().max().item()
            if verbose:
                print(f"  correctness[{name:<16s}]: max_abs_diff={corr_abs[name]:.4e}")

    # Pruning stats.
    n_queries = args.prune_queries
    stats: dict[str, tuple[float, float]] = {}
    for name, st in stats_states.items():
        kept_sum = 0.0
        recall_sum = 0.0
        for qi in range(n_queries):
            kept, recall = _pruning_stats(
                st, q_batch[qi], keys_expanded, full_keys,
                args.topk, q_head_to_kv,
            )
            kept_sum += kept
            recall_sum += recall
        stats[name] = (kept_sum / n_queries, recall_sum / n_queries)
        if verbose:
            print(f"  pruning[{name:<16s}]: kept_frac={stats[name][0]:.4f}  "
                  f"recall@{args.topk}={stats[name][1]:.4f}")

    out: dict = {
        "fresh (rebuild)": {
            "ms": ms_fresh,
            "kept_frac": stats["fresh (rebuild)"][0],
            "recall": stats["fresh (rebuild)"][1],
            "corr_abs": None,
        }
    }
    for name in kernel_ms:
        out[name] = {
            "ms": kernel_ms[name],
            "kept_frac": stats[name][0],
            "recall": stats[name][1],
            "corr_abs": corr_abs.get(name),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--H-kv", type=int, default=8)
    p.add_argument("--H-q", type=int, default=24)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--B", type=int, default=32, help="buffer size")
    p.add_argument("--B-sweep", type=str, default=None,
                   help="Comma-separated buffer sizes to sweep (overrides --B)")
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--D-v", type=int, default=128)
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--refine-iter", type=int, default=5)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--topk", type=int, default=32,
                   help="Top-k used to derive tight thresholds for pruning stats")
    p.add_argument("--prune-queries", type=int, default=8,
                   help="Number of distinct queries to average pruning stats over")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    if args.B_sweep is not None:
        b_list = [int(x) for x in args.B_sweep.split(",") if x.strip()]
    else:
        b_list = [args.B]

    torch.manual_seed(0)
    device = "cuda"
    keys = torch.randn(args.H_kv, args.N, args.D, device=device, dtype=torch.float32)
    values = torch.randn(args.H_kv, args.N, args.D_v, device=device, dtype=torch.float32)

    base_state = build_v2_4(keys, args.bf, args.S, args.refine_iter, values=values)

    h_q = args.H_q
    assert h_q % args.H_kv == 0
    q_head_to_kv = (torch.arange(h_q, device=device) // (h_q // args.H_kv)).to(torch.int64)

    q_batch = torch.randn(args.prune_queries, h_q, args.D, device=device, dtype=torch.float32)
    q_batch = q_batch / q_batch.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    print(f"update micro-bench: H_kv={args.H_kv} N={args.N} "
          f"D={args.D} D_v={args.D_v} bf={args.bf} S={args.S}")
    print("=" * 78)

    all_results: dict[int, dict] = {}
    verbose = len(b_list) == 1
    for B in b_list:
        if not verbose:
            print(f"[B={B}] ...")
        else:
            print(f"  B={B}")
            print("-" * 78)
        all_results[B] = _run_for_B(
            args, B, keys, values, base_state,
            q_head_to_kv, q_batch, verbose=verbose,
        )

    # Summary table across all Bs.
    print("=" * 78)
    print("Summary")
    header = (
        f"  {'B':>6s}  {'kernel':<18s}  {'ms':>9s}  "
        f"{'x_fresh':>8s}  {'kept':>7s}  {'recall':>7s}"
    )
    print(header)
    print("-" * 78)
    for B in b_list:
        res = all_results[B]
        ms_fresh = res["fresh (rebuild)"]["ms"]
        fresh = res["fresh (rebuild)"]
        print(f"  {B:>6d}  {'fresh (rebuild)':<18s}  {fresh['ms']:>9.2f}  "
              f"{'1.0x':>8s}  {fresh['kept_frac']:>7.4f}  {fresh['recall']:>7.4f}")
        for name in SUMMARY_ORDER:
            if name not in res:
                continue
            r = res[name]
            print(f"  {B:>6d}  {name:<18s}  {r['ms']:>9.2f}  "
                  f"{ms_fresh / r['ms']:>7.1f}x  "
                  f"{r['kept_frac']:>7.4f}  {r['recall']:>7.4f}")
        print()


if __name__ == "__main__":
    main()
