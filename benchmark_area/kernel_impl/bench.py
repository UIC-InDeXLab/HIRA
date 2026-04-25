"""End-to-end benchmark: decoding simulation with the subspace k-center index.

For each decoding step i:
  1. (excluded from timing) Compute per-subspace thresholds from the true
     top-k set over all keys up to step i.
  2. (timed) index.attend(query, thresholds) — fused attention over index
     survivors + buffer keys/values.
  3. (timed) baseline_attention(query, all_keys, all_values) — dense softmax.
  4. (timed) baseline_sdpa — torch SDPA reference.
  5. Append the new (k, v) to the decoding buffer.
  6. Every `--update-interval` steps: index.update() — timed separately.

Reports (incremental CSV) in kernel_impl/reports/:
  step, n_keys, attend_ours_ms, dense_attn_ms, sdpa_ms,
  update_ms, amortized_ours_ms, memory_bytes

Usage:
    python -m hira.benchmark_area.kernel_impl.bench \\
        --input-qkv capture.pt --n-steps 2000 --update-interval 256
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.index import (
    IndexConfig,
    SubspaceKCenterIndex,
    baseline_attention,
    baseline_sdpa,
)
from hira.benchmark_area.kernel_impl.kernels.attention_v1_17 import _BUCKETS as _ATTN_BUCKETS
from hira.benchmark_area.quick_pruning.pruning_bench_utils import (
    CaptureState,
    _capture_qkv,
    _q_to_kv_map,
)

DEFAULT_PROMPT = "Benchmark the index over a long decoding trace."

# attend() runs before the buffer append in the decode loop, so buffer size
# reaches `update_interval - 1` right before the flush. The attention kernels
# cap buffer at _BUCKETS[-1] (= 512).
_MAX_BUFFER = _ATTN_BUCKETS[-1]
_MAX_UPDATE_INTERVAL = _MAX_BUFFER + 1


def subspace_topk_thresholds(q, keys, topk, dim_slices):
    """Derive per-subspace thresholds from the full-space top-k set."""
    scores = torch.einsum("hd,hnd->hn", q, keys)
    k = min(topk, scores.shape[-1])
    topk_idx = scores.topk(k, dim=-1).indices
    ths = []
    for s, e in dim_slices:
        qs = q[:, s:e]
        ks = keys[:, :, s:e]
        ss = torch.einsum("hd,hnd->hn", qs, ks)
        sub_top = ss.gather(1, topk_idx)
        ths.append(sub_top.min(dim=1).values)
    return torch.stack(ths, dim=0)


def packed_subspace_topk_thresholds_fp16(q, keys, topk, dim_slices):
    """Return packed ``(2*S, H_q)`` fp16 thresholds + per-subspace q norms."""
    if q.dtype != torch.float16 or keys.dtype != torch.float16:
        raise RuntimeError(
            "bench.py only supports fp16 query/key inputs for fused attention. "
            f"Got q={q.dtype}, keys={keys.dtype}."
        )
    th = subspace_topk_thresholds(q, keys, topk, dim_slices)
    q_norms = torch.stack(
        [q[:, start:end].norm(dim=-1) for start, end in dim_slices],
        dim=0,
    )
    return torch.cat([th, q_norms], dim=0).contiguous()


def _time_gpu(fn):
    """Single-shot timing. Use for mutating ops (update) where looping is unsafe."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000  # ms


def _time_repeated(fn, iters=10, warmup=3):
    """Avg ms/call across ``iters`` runs with one sync per batch.

    Mirrors ``bench_attention.time_call`` so the per-kernel-launch Python+driver
    sync overhead is amortized instead of dominating (matters for 50-100µs calls).
    Only safe for read-only ops — do not pass anything that mutates index state.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end decoding simulation for the subspace k-center index."
    )

    # ── Input source ──
    p.add_argument("--input-qkv", type=Path, default=None,
                   help="Path to a captured QKV .pt file. If omitted, captures "
                        "fresh from --model.")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                   help="HF model id used when capturing QKV live.")
    p.add_argument("--n-tokens", type=int, default=2000,
                   help="Tokens to capture when --input-qkv is not given.")
    p.add_argument("--layer", type=int, default=15,
                   help="Which transformer layer's Q/K/V to simulate.")

    # ── Simulation window ──
    p.add_argument("--prefill-frac", type=float, default=0.5,
                   help="Fraction of total captured keys treated as prefill "
                        "(index is built on these). The remainder is the "
                        "decoding trace.")
    p.add_argument("--n-steps", type=int, default=None,
                   help="Max number of decoding steps to simulate. Defaults to "
                        "all available tokens after prefill. Actual count is "
                        "min(--n-steps, remaining keys, available generated queries).")

    # ── Threshold ──
    p.add_argument("--topk", type=int, default=20,
                   help="k for top-k-derived per-subspace thresholds (threshold "
                        "finding is excluded from timing).")

    # ── Index config ──
    p.add_argument("--bf", type=int, default=4,
                   help="Branching factor: cluster size target (K = ceil(N/bf)).")
    p.add_argument("--n-subspaces", type=int, default=8,
                   help="Number of contiguous dim splits.")
    p.add_argument("--refine-iter", type=int, default=5,
                   help="Lloyd refinement iterations per subspace during build.")
    p.add_argument("--update-mode", choices=["full", "inc"], default="inc",
                   help='"full": rebuild index on all keys via build kernel. '
                        '"inc": mini-index the buffer and merge (update kernel).')
    p.add_argument("--update-interval", type=int, default=256,
                   help=f"Flush the decoding buffer into the index every N "
                        f"steps. Must satisfy 1 <= N <= {_MAX_UPDATE_INTERVAL} "
                        f"(attention kernels bucket the buffer at "
                        f"{_ATTN_BUCKETS}; bucket-aligned values "
                        f"{_ATTN_BUCKETS} minimize CUDA-graph captures).")

    # ── Kernel selection (defaults = fused attention path) ──
    p.add_argument("--build-kernel", default="build_v2_4",
                   help="Module name under kernels/ for build (auto-discovered).")
    p.add_argument("--update-kernel", default="update_v2_1",
                   help="Module name under kernels/ for update (auto-discovered).")
    p.add_argument("--attention-kernel", default="attention_v2_6",
                   help="Module name under kernels/ for fused attention.")

    # ── Output ──
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Defaults to kernel_impl/reports/bench_<update_mode>.csv.")
    p.add_argument("--flush-every", type=int, default=50,
                   help="Flush CSV every N decoding steps.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for k-center seeding (affects build).")
    return p.parse_args()


def load_qkv(args):
    if args.input_qkv is not None:
        print(f"Loading capture from {args.input_qkv} ...")
        cap = CaptureState.load(args.input_qkv)
    else:
        print(f"Capturing {args.n_tokens} tokens from {args.model} ...")
        cap = _capture_qkv(
            model_name=args.model, prompt_text=DEFAULT_PROMPT,
            n=args.n_tokens, device="cuda",
            torch_dtype=torch.float16, show_progress=True,
        )
    layer_ids = cap.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, values_cpu = cap.to_layer_tensors(layer)
    if values_cpu is None:
        raise RuntimeError(
            "Fused attention requires captured values. Re-capture with values."
        )
    return queries_cpu, keys_cpu, values_cpu, layer


def _validate_args(args):
    if args.update_interval < 1 or args.update_interval > _MAX_UPDATE_INTERVAL:
        raise ValueError(
            f"--update-interval={args.update_interval} out of range. "
            f"attention kernels cap buffer at {_MAX_BUFFER} (buckets "
            f"{_ATTN_BUCKETS}), and buffer reaches update_interval-1 before "
            f"flush, so update_interval must be in [1, {_MAX_UPDATE_INTERVAL}]."
        )


def main():
    args = parse_args()
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(args.seed)

    queries_cpu, keys_cpu, values_cpu, layer = load_qkv(args)
    # Stay in fp16 for a fair comparison: captures are fp16, our attend kernel
    # packs fp16 internally, and SDPA's Flash/mem-efficient backends require
    # fp16/bf16 (fp32 falls back to the `math` backend and underperforms).
    dtype = torch.float16
    keys = keys_cpu.to(device="cuda", dtype=dtype)
    queries = queries_cpu.to(device="cuda", dtype=dtype)
    values = values_cpu.to(device="cuda", dtype=dtype)
    H_q = queries.shape[0]
    H_kv, N_total, D = keys.shape
    q_head_to_kv = _q_to_kv_map(H_q, H_kv, "cuda") if H_q != H_kv else None

    n_prefill = max(1, int(args.prefill_frac * N_total))
    max_decode = min(N_total - n_prefill, queries.shape[1] - n_prefill)
    n_decode = max_decode if args.n_steps is None else min(args.n_steps, max_decode)
    if n_decode <= 0:
        raise ValueError("Not enough keys for decoding — adjust --prefill-frac.")

    print(f"Layer {layer}: H_q={H_q} H_kv={H_kv} D={D}")
    print(f"prefill_keys={n_prefill}  decoding_steps={n_decode}")

    # ── Build index on prefill keys/values ──
    cfg = IndexConfig(
        n_subspaces=args.n_subspaces, bf=args.bf, refine_iter=args.refine_iter,
        update_mode=args.update_mode,
        build_kernel=args.build_kernel,
        update_kernel=args.update_kernel,
        attention_kernel=args.attention_kernel,
    )
    index = SubspaceKCenterIndex(cfg)
    prefill_keys = keys[:, :n_prefill, :].contiguous()
    prefill_values = values[:, :n_prefill, :].contiguous()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    index.build(prefill_keys, prefill_values)
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - t0) * 1000
    print(
        f"Build: {build_ms:.1f} ms  "
        f"(build={args.build_kernel}, update={args.update_kernel}, "
        f"attn={args.attention_kernel})"
    )

    # ── Correctness check: fused attend vs. dense attention, on first step. ──
    q0 = queries[:, n_prefill, :]
    qn0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    all_keys_check = keys[:, : n_prefill, :]
    all_values_check = values[:, : n_prefill, :]
    dim_slices = index.state["dim_slices"]
    keys_eval0 = all_keys_check if q_head_to_kv is None else all_keys_check[q_head_to_kv]
    th0 = packed_subspace_topk_thresholds_fp16(
        qn0, keys_eval0, args.topk, dim_slices
    )
    out_ours = index.attend(qn0, th0, q_head_to_kv=q_head_to_kv)
    out_ref = baseline_attention(
        qn0, all_keys_check, all_values_check, q_head_to_kv=q_head_to_kv
    )
    diff = (out_ours.float() - out_ref.float()).abs().max().item()
    rel = diff / (out_ref.float().abs().max().item() + 1e-9)
    print(f"Correctness (attend vs dense): max_abs_diff={diff:.4e}  rel={rel:.4e}")

    # ── Prep output CSV ──
    out_csv = args.output_csv or (
        Path(__file__).parent / "reports" / f"bench_{args.update_mode}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step", "n_keys", "attend_ours_ms", "dense_attn_ms", "sdpa_ms",
        "update_ms", "amortized_ours_ms", "memory_bytes",
        "scanned_parent_frac", "scanned_key_frac",
    ]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    rows: list[dict] = []
    update_costs: list[float] = []
    last_update_ms = 0.0

    sim_start = time.perf_counter()
    for step in range(n_decode):
        token_idx = n_prefill + step
        q = queries[:, token_idx, :]
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # ── Threshold computation (NOT timed) ──
        all_keys_so_far = keys[:, : token_idx + 1, :]
        all_values_so_far = values[:, : token_idx + 1, :]
        keys_eval = all_keys_so_far if q_head_to_kv is None else all_keys_so_far[q_head_to_kv]
        th = packed_subspace_topk_thresholds_fp16(
            qn, keys_eval, args.topk, index.state["dim_slices"]
        )

        # ── Timed: our fused attention ──
        # Pass fp16 Q and packed fp16 thresholds directly. Both are prepared
        # outside timing so the timed region only measures the attention path.
        # Amortized timing: runs warmup+iters calls with one sync per batch so
        # per-call sync overhead doesn't dominate fast kernels. last_cluster_pass
        # reflects the final call, which is what we want for pruning stats.
        attend_ours_ms = _time_repeated(
            lambda: index.attend(qn, th, q_head_to_kv=q_head_to_kv)
        )

        # ── Pruning measurement (NOT timed) ──
        # After _time_gpu(attend) the cluster_pass mask is populated and
        # sync'd. Reduce + .item() happens here, outside any timed region.
        cp = index.last_cluster_pass()
        if cp is not None:
            # AND across subspaces: parents that pass every threshold.
            parent_alive = cp.all(dim=0)               # (H_q, K)
            scanned_parent_frac = parent_alive.float().mean().item()
            # With BF=4 block layout every surviving parent scans BF children,
            # and the buffer keys are always scanned, so the fraction of
            # raw keys touched is:
            bf = int(index.state["bf"])
            k = int(index.state["K"])
            n_idx = k * bf                             # indexed-keys count (padded)
            n_buf = int(index.n_buffered)
            scanned_key_frac = (
                (scanned_parent_frac * n_idx + n_buf) / max(n_idx + n_buf, 1)
            )
        else:
            scanned_parent_frac = float("nan")
            scanned_key_frac = float("nan")
        # ── Timed: dense attention baseline ──
        dense_attn_ms = _time_repeated(
            lambda: baseline_attention(
                qn, all_keys_so_far, all_values_so_far, q_head_to_kv
            )
        )
        # ── Timed: SDPA baseline ──
        sdpa_ms = _time_repeated(
            lambda: baseline_sdpa(
                qn, all_keys_so_far, all_values_so_far, q_head_to_kv
            )
        )

        # ── Append new (k, v) to buffer ──
        new_key = keys[:, token_idx : token_idx + 1, :]
        new_val = values[:, token_idx : token_idx + 1, :]
        index.append_decoding_kv(new_key, new_val)

        # ── Timed: update every update_interval steps ──
        update_ms = 0.0
        if index.needs_update(args.update_interval):
            _, update_ms = _time_gpu(index.update)
            last_update_ms = update_ms
        update_costs.append(update_ms)
        amort_update_ms = sum(update_costs) / len(update_costs)
        amort_ours = attend_ours_ms + amort_update_ms

        rows.append({
            "step": step,
            "n_keys": int(all_keys_so_far.shape[1]),
            "attend_ours_ms": round(attend_ours_ms, 4),
            "dense_attn_ms": round(dense_attn_ms, 4),
            "sdpa_ms": round(sdpa_ms, 4),
            "update_ms": round(update_ms, 4),
            "amortized_ours_ms": round(amort_ours, 4),
            "memory_bytes": index.memory_bytes(),
            "scanned_parent_frac": round(scanned_parent_frac, 5),
            "scanned_key_frac": round(scanned_key_frac, 5),
        })

        if (step + 1) % args.flush_every == 0 or step == n_decode - 1:
            with out_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
            rows.clear()
            elapsed = time.perf_counter() - sim_start
            print(
                f"step {step+1}/{n_decode}  "
                f"attend={attend_ours_ms:.3f}ms  dense={dense_attn_ms:.3f}ms  "
                f"sdpa={sdpa_ms:.3f}ms  last_upd={last_update_ms:.2f}ms  "
                f"scan[parents]={scanned_parent_frac:.3f} "
                f"scan[keys]={scanned_key_frac:.3f}  [{elapsed:.1f}s]"
            )

    # ── End-of-run summary ──
    n_upd = sum(1 for u in update_costs if u > 0)
    total_upd_ms = sum(update_costs)
    mean_upd_ms = total_upd_ms / n_upd if n_upd else 0.0
    print(
        f"\nUpdates: {n_upd} fired over {n_decode} steps "
        f"(interval={args.update_interval}, mode={args.update_mode}) "
        f"— total={total_upd_ms:.1f}ms  mean={mean_upd_ms:.2f}ms/update  "
        f"amortized={total_upd_ms / max(n_decode, 1):.3f}ms/step"
    )
    print(f"Done. CSV -> {out_csv}")


if __name__ == "__main__":
    main()
