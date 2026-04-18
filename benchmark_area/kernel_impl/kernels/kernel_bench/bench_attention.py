"""Micro-benchmark: compare all search_v* kernels.

Requires real keys/queries because search speed depends on distribution.
Either pass --input-qkv path.pt or let it capture from --model.

Usage:
    python -m hira.benchmark_area.kernel_impl.kernels.kernel_bench.bench_search \
        --input-qkv /path/to/capture.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.kernels import attention_kernels, search_kernels
from hira.benchmark_area.kernel_impl.kernels.build_v1_0 import build as build_v1
from hira.benchmark_area.kernel_impl.kernels.build_v2_0 import build as build_v2
from hira.benchmark_area.kernel_impl.kernels.build_v2_1 import build as build_v2_1
from hira.benchmark_area.kernel_impl.kernels.build_v2_1_fp16 import build as build_v2_1_fp16
from hira.benchmark_area.kernel_impl.kernels.build_v2_2 import build as build_v2_2
from hira.benchmark_area.kernel_impl.kernels.build_v2_2_fp16 import build as build_v2_2_fp16
from hira.benchmark_area.kernel_impl.kernels.build_v2_3 import build as build_v2_3
from hira.benchmark_area.kernel_impl.kernels.build_v2_4 import build as build_v2_4
from hira.benchmark_area.quick_pruning.pruning_bench_utils import (
    CaptureState,
    _capture_qkv,
    _q_to_kv_map,
)

SEARCH_BUILD_KERNELS = {
    "search_v10_0": "build_v2_0",
    "search_v11_0": "build_v2_1",
    "search_v11_1": "build_v2_1_fp16",
    "search_v12_0": "build_v2_2",
    "search_v12_1": "build_v2_2",
    "search_v12_2": "build_v2_2_fp16",
    "search_v13_0": "build_v2_1",
    "search_v14_0": "build_v2_3",
    "search_v15_0": "build_v2_1",
    "search_v15_1": "build_v2_1_fp16",
    "search_v16_0": "build_v2_1",
    "search_v16_1": "build_v2_1",
    "search_v17_0": "build_v2_1",
    "search_v17_1": "build_v2_1",
    "search_v18_0": "build_v2_1",
    "search_v18_1": "build_v2_1",
    "search_v18_2": "build_v2_1_fp16",
    "search_v18_3": "build_v2_1",
}

# Only the winners are benched. Non-winners remain on disk but are filtered out
# in the attention loop below (see `name not in ATTENTION_BUILD_KERNELS`).
#   - v1.5         : generic fallback (any BF/S, handles non-empty buffer)
#   - v1.15        : BF=4/S=8  specialist (empty buffer only)
#   - v1.15_s16    : BF=16/S=16 specialist (empty buffer only)
ATTENTION_BUILD_KERNELS = {
    "attention_v1_5": "build_v2_4",
    "attention_v1_15": "build_v2_4",
    "attention_v1_15_s16": "build_v2_4",
    "attention_v1_16": "build_v2_4",
}

BUILD_FNS = {
    "build_v1_0": build_v1,
    "build_v2_0": build_v2,
    "build_v2_1": build_v2_1,
    "build_v2_1_fp16": build_v2_1_fp16,
    "build_v2_2": build_v2_2,
    "build_v2_2_fp16": build_v2_2_fp16,
    "build_v2_3": build_v2_3,
    "build_v2_4": build_v2_4,
}


def subspace_topk_thresholds(q, keys, topk, dim_slices):
    """Per-subspace thresholds derived from the true full-space top-k set."""
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


def time_call(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", type=Path, default=None)
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--n-tokens", type=int, default=1000)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--refine-iter", type=int, default=5)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--n-queries", type=int, default=20)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first kernel error instead of skipping incompatible kernels.",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    if args.input_qkv is not None:
        print(f"Loading capture from {args.input_qkv} ...")
        cap = CaptureState.load(args.input_qkv)
    else:
        print(f"Capturing {args.n_tokens} tokens from {args.model} ...")
        cap = _capture_qkv(
            model_name=args.model, prompt_text="Benchmark.",
            n=args.n_tokens, device="cuda",
            torch_dtype=torch.float16, show_progress=True,
        )

    layer_ids = cap.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, values_cpu = cap.to_layer_tensors(layer)
    keys = keys_cpu.to(device="cuda", dtype=torch.float32)
    values = (
        values_cpu.to(device="cuda", dtype=torch.float32)
        if values_cpu is not None else None
    )
    queries = queries_cpu
    H_q = queries.shape[0]
    H_kv, N, D = keys.shape
    D_v = int(values.shape[-1]) if values is not None else D
    q_head_to_kv = _q_to_kv_map(H_q, H_kv, "cuda") if H_q != H_kv else None

    buffer = torch.empty(H_kv, 0, D, device="cuda", dtype=torch.float32)
    value_buffer = (
        torch.empty(H_kv, 0, D_v, device="cuda", dtype=torch.float32)
        if values is not None else None
    )
    state_cache: dict[str, dict] = {}

    def get_state(build_name: str) -> dict:
        state = state_cache.get(build_name)
        if state is None:
            if build_name == "build_v2_4":
                state = BUILD_FNS[build_name](
                    keys, args.bf, args.S, args.refine_iter, values=values
                )
            else:
                state = BUILD_FNS[build_name](keys, args.bf, args.S, args.refine_iter)
            state_cache[build_name] = state
        return state

    # Pre-compute per-subspace thresholds over a sweep of queries; average across them.
    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride))[: args.n_queries]

    required_builds = {
        SEARCH_BUILD_KERNELS.get(name, "build_v1_0")
        for name in search_kernels()
    } | {
        ATTENTION_BUILD_KERNELS[name]
        for name in attention_kernels()
        if name in ATTENTION_BUILD_KERNELS
    }
    query_pairs_by_build: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    keys_eval = keys if q_head_to_kv is None else keys[q_head_to_kv]

    # Precompute (q, thresholds) pairs per build layout to avoid including them in timing.
    for build_name in sorted(required_builds):
        state = get_state(build_name)
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for qi in q_indices:
            q = queries[:, qi, :].to(device="cuda", dtype=torch.float32)
            qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            th = subspace_topk_thresholds(qn, keys_eval, args.topk, state["dim_slices"])
            pairs.append((qn, th))
        query_pairs_by_build[build_name] = pairs

    print(f"search micro-bench: layer {layer} H_q={H_q} H_kv={H_kv} N={N} D={D} S={args.S}")
    print("-" * 70)

    def bench_fn(fn, state, query_pairs):
        def f():
            for qn, th in query_pairs:
                fn(q=qn, th_per_subspace=th, state=state,
                   buffer_keys=buffer, keys_children=keys,
                   q_head_to_kv=q_head_to_kv)
        return f

    results = []
    skipped: list[str] = []

    def _record_skip(label: str) -> None:
        skipped.append(label)

    def _try_bench(label: str, build_name: str, fn) -> float | None:
        try:
            ms = time_call(fn, iters=args.iters, warmup=3)
            return ms
        except Exception as exc:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            if args.strict:
                raise
            _record_skip(label)
            print(f"  {label:<24s} skipped")
            return None

    for name, info in sorted(search_kernels().items()):
        build_name = SEARCH_BUILD_KERNELS.get(name, "build_v1_0")
        label = f"{name} ({info.version})"
        try:
            state = get_state(build_name)
            query_pairs = query_pairs_by_build[build_name]
        except Exception as exc:
            if args.strict:
                raise
            _record_skip(label)
            print(f"  {label:<24s} skipped")
            continue
        ms = _try_bench(name, build_name, bench_fn(info.fn, state, query_pairs))
        if ms is None:
            continue
        per_q = ms / len(query_pairs)
        results.append((label, per_q))
        print(f"  {name:<24s} {info.version:<6s}  {per_q:8.3f} ms/query  [{build_name}]")

    # Torch baseline: brute-force dot product over all keys
    keys_q = keys if q_head_to_kv is None else keys[q_head_to_kv]
    baseline_pairs = query_pairs_by_build.get("build_v1_0")
    if baseline_pairs is None:
        baseline_pairs = next(iter(query_pairs_by_build.values()))

    def baseline():
        for qn, _ in baseline_pairs:
            _ = torch.einsum("hd,hnd->hn", qn, keys_q)

    ms = time_call(baseline, iters=args.iters, warmup=3)
    per_q = ms / len(baseline_pairs)
    results.append(("torch_baseline (full dot)", per_q))
    print(f"  {'torch_baseline':<24s} {'-':<6s}  {per_q:8.3f} ms/query")

    def matmul_baseline():
        for qn, _ in baseline_pairs:
            _ = torch.matmul(keys_q, qn.unsqueeze(-1)).squeeze(-1)

    ms = time_call(matmul_baseline, iters=args.iters, warmup=3)
    per_q = ms / len(baseline_pairs)
    results.append(("matmul baseline", per_q))
    print(f"  {'matmul baseline':<24s} {'-':<6s}  {per_q:8.3f} ms/query")

    # FP16 baselines for fair comparison with fp16-key search kernels
    keys_q_f16 = keys_q.half()

    def baseline_fp16():
        for qn, _ in baseline_pairs:
            _ = torch.einsum("hd,hnd->hn", qn.half(), keys_q_f16)

    ms = time_call(baseline_fp16, iters=args.iters, warmup=3)
    per_q = ms / len(baseline_pairs)
    results.append(("torch_baseline fp16", per_q))
    print(f"  {'torch_baseline fp16':<24s} {'-':<6s}  {per_q:8.3f} ms/query")

    def matmul_baseline_fp16():
        for qn, _ in baseline_pairs:
            _ = torch.matmul(keys_q_f16, qn.half().unsqueeze(-1)).squeeze(-1)

    ms = time_call(matmul_baseline_fp16, iters=args.iters, warmup=3)
    per_q = ms / len(baseline_pairs)
    results.append(("matmul baseline fp16", per_q))
    print(f"  {'matmul baseline fp16':<24s} {'-':<6s}  {per_q:8.3f} ms/query")

    # ── Fused attention kernels + attention baselines ──
    if values is not None:
        print("-" * 70)
        print("Attention (fused search + softmax + @V → (H_q, D_v))")
        import math
        scale = 1.0 / math.sqrt(D)
        values_q = values if q_head_to_kv is None else values[q_head_to_kv]
        values_q_f16 = values_q.half()
        successful_attention: list[tuple[str, object, str]] = []

        for name, info in sorted(attention_kernels().items()):
            if name not in ATTENTION_BUILD_KERNELS:
                continue
            build_name = ATTENTION_BUILD_KERNELS[name]
            label = f"{name} ({info.version})"
            try:
                state = get_state(build_name)
                query_pairs = query_pairs_by_build.get(build_name) or next(
                    iter(query_pairs_by_build.values())
                )
            except Exception as exc:
                if args.strict:
                    raise
                _record_skip(label)
                print(f"  {label:<24s} skipped")
                continue

            def attend_fn():
                for qn, th in query_pairs:
                    info.fn(
                        q=qn, th_per_subspace=th, state=state,
                        buffer_keys=buffer,
                        buffer_values=value_buffer,
                        keys_children=keys,
                        q_head_to_kv=q_head_to_kv,
                        scale=scale,
                    )

            ms = _try_bench(name, build_name, attend_fn)
            if ms is None:
                continue
            per_q = ms / len(query_pairs)
            results.append((label, per_q))
            successful_attention.append((name, info, build_name))
            print(f"  {name:<24s} {info.version:<6s}  {per_q:8.3f} ms/query  [{build_name}]")

        # Dense attention baseline (fp32 math, matches our fused output dtype).
        def dense_attn_fp32():
            for qn, _ in baseline_pairs:
                scores = torch.einsum("hd,hnd->hn", qn, keys_q) * scale
                probs = torch.softmax(scores, dim=-1)
                _ = torch.einsum("hn,hnd->hd", probs, values_q)

        ms = time_call(dense_attn_fp32, iters=args.iters, warmup=3)
        per_q = ms / len(baseline_pairs)
        results.append(("dense attn fp32", per_q))
        print(f"  {'dense attn fp32':<24s} {'-':<6s}  {per_q:8.3f} ms/query")

        # FP16 dense attention.
        def dense_attn_fp16():
            for qn, _ in baseline_pairs:
                scores = torch.einsum("hd,hnd->hn", qn.half(), keys_q_f16) * scale
                probs = torch.softmax(scores.float(), dim=-1).half()
                _ = torch.einsum("hn,hnd->hd", probs, values_q_f16)

        ms = time_call(dense_attn_fp16, iters=args.iters, warmup=3)
        per_q = ms / len(baseline_pairs)
        results.append(("dense attn fp16", per_q))
        print(f"  {'dense attn fp16':<24s} {'-':<6s}  {per_q:8.3f} ms/query")

        # SDPA (flash backend chosen automatically).
        def sdpa_baseline():
            for qn, _ in baseline_pairs:
                q4 = qn.view(1, H_q, 1, D)
                k4 = keys_q.view(1, H_q, N, D)
                v4 = values_q.view(1, H_q, N, D_v)
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q4, k4, v4, is_causal=False, scale=scale
                )

        ms = time_call(sdpa_baseline, iters=args.iters, warmup=3)
        per_q = ms / len(baseline_pairs)
        results.append(("sdpa fp32", per_q))
        print(f"  {'sdpa fp32':<24s} {'-':<6s}  {per_q:8.3f} ms/query")

        def sdpa_baseline_fp16():
            for qn, _ in baseline_pairs:
                q4 = qn.half().view(1, H_q, 1, D)
                k4 = keys_q_f16.view(1, H_q, N, D)
                v4 = values_q_f16.view(1, H_q, N, D_v)
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q4, k4, v4, is_causal=False, scale=scale
                )

        ms = time_call(sdpa_baseline_fp16, iters=args.iters, warmup=3)
        per_q = ms / len(baseline_pairs)
        results.append(("sdpa fp16", per_q))
        print(f"  {'sdpa fp16':<24s} {'-':<6s}  {per_q:8.3f} ms/query")

        # Correctness checks: compare fused attention to dense attention.
        #   - tight gate (as timed): expected sparse-approximation error.
        #   - loose gate (all parents pass): should match dense to fp16 noise.
        if successful_attention:
            first_attn_name, info, build_name = successful_attention[0]
            state = get_state(build_name)
            qn0, th0 = query_pairs_by_build[build_name][0]
            S_subspaces = len(state["dim_slices"])
            th_loose = torch.full(
                (S_subspaces, H_q), -1e9, device="cuda", dtype=torch.float32
            )
            scores_ref = torch.einsum("hd,hnd->hn", qn0, keys_q) * scale
            probs_ref = torch.softmax(scores_ref, dim=-1)
            out_ref = torch.einsum("hn,hnd->hd", probs_ref, values_q)
            ref_scale = out_ref.float().abs().max().item() + 1e-9

            for tag, th_used in (("tight(pruned)", th0), ("loose(all pass)", th_loose)):
                out_ours = info.fn(
                    q=qn0, th_per_subspace=th_used, state=state,
                    buffer_keys=buffer,
                    buffer_values=value_buffer,
                    keys_children=keys,
                    q_head_to_kv=q_head_to_kv,
                    scale=scale,
                )
                diff = (out_ours.float() - out_ref.float()).abs().max().item()
                print(
                    f"  correctness[{tag:<15s}]: max_abs_diff={diff:.4e}  "
                    f"rel={diff / ref_scale:.4e} ({first_attn_name})"
                )

            # Additional per-kernel correctness using loose gate, for non-first kernels.
            for attn_name, info_k, build_k in successful_attention[1:]:
                state_k = get_state(build_k)
                qn_k, _ = query_pairs_by_build[build_k][0]
                S_k = len(state_k["dim_slices"])
                th_loose_k = torch.full(
                    (S_k, H_q), -1e9, device="cuda", dtype=torch.float32
                )
                try:
                    out_k = info_k.fn(
                        q=qn_k, th_per_subspace=th_loose_k, state=state_k,
                        buffer_keys=buffer,
                        buffer_values=value_buffer,
                        keys_children=keys,
                        q_head_to_kv=q_head_to_kv,
                        scale=scale,
                    )
                    diff_k = (out_k.float() - out_ref.float()).abs().max().item()
                    print(
                        f"  correctness[loose(all pass)]: max_abs_diff={diff_k:.4e}  "
                        f"rel={diff_k / ref_scale:.4e} ({attn_name})"
                    )
                except Exception as exc:
                    print(f"  correctness[{attn_name}] FAILED: {type(exc).__name__}: {exc}")

    print("-" * 70)
    if results:
        best = min(results, key=lambda r: r[1])
        print(f"Fastest: {best[0]} at {best[1]:.3f} ms/query")
    else:
        print("Fastest: none (all kernels failed or were skipped)")
    if skipped:
        print("Skipped kernels:")
        for label in skipped:
            print(f"  {label:<24s} skipped")


if __name__ == "__main__":
    main()
