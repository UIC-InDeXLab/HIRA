"""End-to-end benchmark: decoding simulation with the subspace k-center index.

For each decoding step i:
  1. (excluded from timing) Compute per-subspace thresholds from the true
     top-k set over all keys up to step i.
  2. (timed) index.search(query, thresholds) — returns dot products over
     index survivors + buffer keys.
  3. (timed) baseline_dot(query, all_keys) — brute-force.
  4. Append the new key to the index buffer.
  5. Every `--update-interval` steps: index.update() — timed separately.

Reports (incremental CSV) in kernel_impl/reports/:
  step, n_keys, search_ours_ms, search_baseline_ms, update_ms,
  amortized_ours_ms, memory_bytes, scanned_fraction

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
    baseline_dot,
)
from hira.benchmark_area.quick_pruning.pruning_bench_utils import (
    CaptureState,
    _capture_qkv,
    _q_to_kv_map,
)

DEFAULT_PROMPT = "Benchmark the index over a long decoding trace."


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


def _time_gpu(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000  # ms


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
                   help="Which transformer layer's Q/K to simulate.")

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
                   help='"full": rebuild index on all keys. '
                        '"inc": append a mini-index built from the buffer.')
    p.add_argument("--update-interval", type=int, default=256,
                   help="Flush the decoding buffer into the index every N steps.")

    # ── Kernel selection ──
    p.add_argument("--build-kernel", default="build_v1_0",
                   help="Module name under kernels/ for build (auto-discovered).")
    p.add_argument("--search-kernel", default="search_v1_0",
                   help="Module name under kernels/ for search (auto-discovered).")
    p.add_argument("--update-kernel", default="update_v1_0",
                   help="Module name under kernels/ for update (auto-discovered).")

    # ── Output ──
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Defaults to kernel_impl/reports/bench_<mode>.csv.")
    p.add_argument("--flush-every", type=int, default=50,
                   help="Flush CSV every N decoding steps.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for k-center seeding (affects build).")
    return p.parse_args()


def load_qk(args):
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
    queries_cpu, keys_cpu, _ = cap.to_layer_tensors(layer)
    return queries_cpu, keys_cpu, layer


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(args.seed)

    queries_cpu, keys_cpu, layer = load_qk(args)
    keys = keys_cpu.to(device="cuda", dtype=torch.float32)
    queries = queries_cpu.to(device="cuda", dtype=torch.float32)
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

    # ── Build index on prefill keys ──
    cfg = IndexConfig(
        n_subspaces=args.n_subspaces, bf=args.bf, refine_iter=args.refine_iter,
        update_mode=args.update_mode,
        build_kernel=args.build_kernel,
        search_kernel=args.search_kernel,
        update_kernel=args.update_kernel,
    )
    index = SubspaceKCenterIndex(cfg)
    prefill_keys = keys[:, :n_prefill, :].contiguous()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    index.build(prefill_keys)
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"Build: {build_ms:.1f} ms ({args.build_kernel})")

    # ── Prep output CSV ──
    out_csv = args.output_csv or (
        Path(__file__).parent / "reports" / f"bench_{args.update_mode}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step", "n_keys", "search_ours_ms", "search_baseline_ms",
        "update_ms", "amortized_ours_ms", "memory_bytes", "scanned_fraction",
    ]
    # Clear + header
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    rows: list[dict] = []
    update_costs: list[float] = []  # cumulative update cost, for amortization

    sim_start = time.perf_counter()
    for step in range(n_decode):
        token_idx = n_prefill + step
        q = queries[:, token_idx, :]
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # ── Threshold computation (NOT timed) ──
        all_keys_so_far = keys[:, : token_idx + 1, :]
        keys_eval = all_keys_so_far if q_head_to_kv is None else all_keys_so_far[q_head_to_kv]
        th = subspace_topk_thresholds(qn, keys_eval, args.topk, index.state["dim_slices"])

        # ── Timed: our search ──
        dots_ours, search_ours_ms = _time_gpu(
            lambda: index.search(qn, th, q_head_to_kv=q_head_to_kv)
        )
        # ── Timed: baseline ──
        _, search_base_ms = _time_gpu(
            lambda: baseline_dot(qn, all_keys_so_far, q_head_to_kv)
        )

        # Scanned fraction over index children only (buffer excluded — its
        # entries are always scanned and would inflate the ratio).
        n_index = index.n_children
        scanned_frac = float(
            (dots_ours[:, :n_index] != float("-inf")).float().mean().item()
        )

        # ── Append new key to buffer ──
        new_key = keys[:, token_idx : token_idx + 1, :]
        index.append_decoding_key(new_key)

        # ── Timed: update every update_interval steps ──
        update_ms = 0.0
        if index.needs_update(args.update_interval):
            _, update_ms = _time_gpu(index.update)
        update_costs.append(update_ms)
        amort_update_ms = (sum(update_costs) / len(update_costs))

        amort_ours = search_ours_ms + amort_update_ms

        rows.append({
            "step": step,
            "n_keys": int(all_keys_so_far.shape[1]),
            "search_ours_ms": round(search_ours_ms, 4),
            "search_baseline_ms": round(search_base_ms, 4),
            "update_ms": round(update_ms, 4),
            "amortized_ours_ms": round(amort_ours, 4),
            "memory_bytes": index.memory_bytes(),
            "scanned_fraction": round(scanned_frac, 5),
        })

        if (step + 1) % args.flush_every == 0 or step == n_decode - 1:
            with out_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
            rows.clear()
            elapsed = time.perf_counter() - sim_start
            print(
                f"step {step+1}/{n_decode}  "
                f"search={search_ours_ms:.3f}ms  base={search_base_ms:.3f}ms  "
                f"upd={update_ms:.2f}ms  amort={amort_ours:.3f}ms  "
                f"scanned={scanned_frac:.3f}  [{elapsed:.1f}s]"
            )

    print(f"\nDone. CSV -> {out_csv}")


if __name__ == "__main__":
    main()
