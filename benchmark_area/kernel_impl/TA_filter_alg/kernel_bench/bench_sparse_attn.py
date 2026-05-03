"""Benchmark masked attention baselines across sparse top-k mask fractions.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.sparse_attn._sdpa_cuda_atomic_fp16 import (
    sdpa_cuda_atomic_fp16,
)
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.sparse_attn._sdpa_cuda_sparse_v2_4_fp16 import (
    sdpa_cuda_sparse_v2_4_fp16,
)
from hira.benchmark_area.quick_pruning.pruning_bench_utils import CaptureState, _q_to_kv_map


def time_call(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def mask_to_compact(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert dense int8 mask [Hq, N] to (live_idx [Hq, N] int32, live_count [Hq] int32).

    Done outside the timing loop — compact-input kernels are not meant to do
    this work themselves.
    """
    h_q, n_ctx = mask.shape
    live_count = mask.sum(dim=1).to(torch.int32)
    live_idx = torch.zeros(h_q, n_ctx, dtype=torch.int32, device=mask.device)
    for h in range(h_q):
        idx = mask[h].nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        c = idx.numel()
        if c > 0:
            live_idx[h, :c] = idx
    return live_idx, live_count


def topk_dot_mask(
    q: torch.Tensor,
    keys: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    fraction: float,
) -> torch.Tensor:
    h_q = int(q.shape[0])
    n_ctx = int(keys.shape[1])
    if fraction >= 1.0:
        return torch.ones(h_q, n_ctx, device=q.device, dtype=torch.int8)
    k_eff = max(1, min(n_ctx, int(round(float(fraction) * n_ctx))))
    keys_eff = keys if q_head_to_kv is None else keys.index_select(0, q_head_to_kv)
    scores = torch.einsum("hd,hnd->hn", q.float(), keys_eff.float())
    top_idx = torch.topk(scores, k_eff, dim=-1).indices
    mask = torch.zeros(h_q, n_ctx, device=q.device, dtype=torch.int8)
    mask.scatter_(1, top_idx, 1)
    return mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", "--input", dest="input_qkv", type=Path, required=True)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--fractions", nargs="+", type=float, default=[1.0, 0.2, 0.1, 0.05])
    p.add_argument("--n-queries", type=int, default=50)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=4)
    args = p.parse_args()

    cap = CaptureState.load(args.input_qkv)
    qcpu, kcpu, vcpu = cap.to_layer_tensors(args.layer)
    if vcpu is None:
        raise RuntimeError("Captured values required.")

    keys = kcpu.to(device="cuda", dtype=torch.float32).half().contiguous()
    values = vcpu.to(device="cuda", dtype=torch.float32).half().contiguous()
    h_q = int(qcpu.shape[0])
    h_kv, n_ctx, d = keys.shape
    q_head_to_kv = _q_to_kv_map(h_q, h_kv, "cuda") if h_q != h_kv else None
    scale = 1.0 / math.sqrt(d)

    total_q = qcpu.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride))[
        : args.n_queries
    ]
    queries: list[torch.Tensor] = []
    queries_f32: list[torch.Tensor] = []
    for qi in q_indices:
        q = qcpu[:, qi, :].to(device="cuda", dtype=torch.float32)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        queries_f32.append(q.contiguous())
        queries.append(q.half().contiguous())

    # input_kind: "mask" → fn(q, keys, values, mask, q_head_to_kv, scale, **kwargs)
    #             "compact" → fn(q, keys, values, live_idx, live_count, q_head_to_kv, scale, **kwargs)
    baselines = [
        ("cuda_atomic",   sdpa_cuda_atomic_fp16,   "mask",    {}),
        ("cuda_sparse_v2_4", sdpa_cuda_sparse_v2_4_fp16, "compact", {}),
    ]

    fractions = [float(f) for f in args.fractions]
    if all(abs(f - 1.0) > 1e-8 for f in fractions):
        fractions = [1.0] + fractions
    fractions = sorted(set(fractions), reverse=True)

    print(f"Hq={h_q} Hkv={h_kv} N={n_ctx} D={d} queries={len(queries)}")
    results_by_baseline: dict[str, dict[float, float]] = {name: {} for name, _, _, _ in baselines}
    for frac in fractions:
        masks = [topk_dot_mask(q, keys, q_head_to_kv, frac) for q in queries_f32]
        compacts = [mask_to_compact(m) for m in masks]
        torch.cuda.synchronize()
        actual = sum(float(m.float().mean().item()) for m in masks) / len(masks)
        print(f"\nfraction={frac:g} actual={actual:.4f}")
        for name, fn, input_kind, kwargs in baselines:
            if input_kind == "mask":
                def run() -> None:
                    for q, mask in zip(queries, masks):
                        fn(q, keys, values, mask, q_head_to_kv, scale, **kwargs)
            elif input_kind == "compact":
                def run() -> None:
                    for q, (li, lc) in zip(queries, compacts):
                        fn(q, keys, values, li, lc, q_head_to_kv, scale, **kwargs)
            else:
                raise ValueError(f"unknown input_kind={input_kind}")

            try:
                ms = time_call(run, args.iters, args.warmup) / len(queries)
                results_by_baseline[name][frac] = ms
                print(f"  {name:<20s} {ms:9.6f} ms/query")
            except Exception as exc:
                torch.cuda.synchronize()
                print(f"  {name:<20s} skipped: {type(exc).__name__}: {str(exc).splitlines()[0]}")

    print("\nSpeedup vs fraction=1.0 (speedup = t@1.0 / t@fraction)")
    print("-" * 72)
    header = ["baseline"] + [f"f={f:g}" for f in fractions]
    print("  " + "  ".join(f"{h:<14s}" for h in header))
    for name, _, _, _ in baselines:
        base = results_by_baseline[name].get(1.0)
        row = [f"{name:<14s}"]
        for frac in fractions:
            t = results_by_baseline[name].get(frac)
            if base is None or t is None:
                cell = "n/a"
            else:
                speedup = base / t if t > 0.0 else float("inf")
                cell = f"{speedup:.3f}x"
            row.append(f"{cell:<14s}")
        print("  " + "  ".join(row))


if __name__ == "__main__":
    main()
