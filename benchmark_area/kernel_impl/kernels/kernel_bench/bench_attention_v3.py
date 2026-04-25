"""Benchmark v3.x attention kernels against dense and SDPA baselines.

Usage:
    /home/mohsen/venv/bin/python bench_attention_v3.py \
        --input ../../../quick_pruning/capture_qkv_12000_Qwen_Qwen2.5-7B-Instruct.pt \
        --bf 4 --S 8 --buffer 256
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.kernels.build_v2_7 import build as build_v2_7
from hira.benchmark_area.kernel_impl.kernels.attention_v3_0 import attend as attention_v3_0
from hira.benchmark_area.kernel_impl.kernels.attention_v3_1 import attend as attention_v3_1


@dataclass
class CaptureState:
    prompt_length: int | None = None
    prefill_keys: dict[int, torch.Tensor] = field(default_factory=dict)
    prefill_values: dict[int, torch.Tensor] = field(default_factory=dict)
    generated_queries: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    generated_keys: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    generated_values: dict[int, list[torch.Tensor]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "CaptureState":
        data = torch.load(path, map_location="cpu")
        return cls(
            prompt_length=data["prompt_length"],
            prefill_keys={int(k): v.contiguous() for k, v in data["prefill_keys"].items()},
            prefill_values={
                int(k): v.contiguous() for k, v in data["prefill_values"].items()
            },
            generated_queries={
                int(k): [x.contiguous() for x in values]
                for k, values in data["generated_queries"].items()
            },
            generated_keys={
                int(k): [x.contiguous() for x in values]
                for k, values in data["generated_keys"].items()
            },
            generated_values={
                int(k): [x.contiguous() for x in values]
                for k, values in data["generated_values"].items()
            },
        )

    def layer_ids(self) -> list[int]:
        return sorted(self.prefill_keys.keys())

    def to_layer_tensors(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        prefill_k = self.prefill_keys[layer_idx]
        q_list = self.generated_queries[layer_idx]
        k_list = self.generated_keys[layer_idx]
        queries = torch.stack(q_list, dim=1)
        generated_k = torch.stack(k_list, dim=1) if k_list else None
        keys = torch.cat([prefill_k, generated_k], dim=1) if generated_k is not None else prefill_k

        values: torch.Tensor | None = None
        if layer_idx in self.prefill_values:
            prefill_v = self.prefill_values[layer_idx]
            v_list = self.generated_values.get(layer_idx, [])
            if len(v_list) == len(q_list) and v_list:
                generated_v = torch.stack(v_list, dim=1)
                values = torch.cat([prefill_v, generated_v], dim=1)
            else:
                values = prefill_v
        return queries, keys, values


def _q_to_kv_map(num_q_heads: int, num_kv_heads: int, device: str) -> torch.Tensor:
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"GQA mapping requires num_q_heads % num_kv_heads == 0, got "
            f"{num_q_heads} and {num_kv_heads}."
        )
    groups = num_q_heads // num_kv_heads
    return torch.arange(num_q_heads, device=device, dtype=torch.int64) // groups


def subspace_topk_thresholds(q, keys, topk, dim_slices):
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
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "--input-qkv", dest="input_qkv", type=Path, required=True)
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--buffer", "--buffer-len", dest="buffer_len", type=int, default=0)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--n-queries", type=int, default=20)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--layer", type=int, default=15)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bench_attention_v3.py")

    cap = CaptureState.load(args.input_qkv)
    layer_ids = cap.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, values_cpu = cap.to_layer_tensors(layer)
    keys_full = keys_cpu.to(device="cuda", dtype=torch.float32)
    values_full = values_cpu.to(device="cuda", dtype=torch.float32)
    queries = queries_cpu
    H_q = queries.shape[0]
    H_kv, N, D = keys_full.shape
    D_v = int(values_full.shape[-1])
    q_head_to_kv = _q_to_kv_map(H_q, H_kv, "cuda") if H_q != H_kv else None

    if args.buffer_len < 0 or args.buffer_len >= N:
        raise ValueError(f"--buffer must be in [0, {N - 1}], got {args.buffer_len}")
    n_index = N - args.buffer_len
    keys = keys_full[:, :n_index, :].contiguous()
    values = values_full[:, :n_index, :].contiguous()
    if args.buffer_len:
        buffer = keys_full[:, n_index:, :].contiguous()
        value_buffer = values_full[:, n_index:, :].contiguous()
    else:
        buffer = torch.empty(H_kv, 0, D, device="cuda", dtype=torch.float32)
        value_buffer = torch.empty(H_kv, 0, D_v, device="cuda", dtype=torch.float32)

    state = build_v2_7(keys, args.bf, args.S, values=values)
    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride))[
        : args.n_queries
    ]
    keys_eval = keys_full if q_head_to_kv is None else keys_full[q_head_to_kv]
    query_pairs = []
    query_pairs_fp16 = []
    for qi in q_indices:
        q = queries[:, qi, :].to(device="cuda", dtype=torch.float32)
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        th = subspace_topk_thresholds(qn, keys_eval, args.topk, state["dim_slices"])
        q_norms = torch.stack(
            [qn[:, start:end].norm(dim=-1) for start, end in state["dim_slices"]], dim=0
        )
        th_packed_fp16 = torch.cat([th, q_norms], dim=0).to(torch.float16)
        query_pairs.append((qn, th))
        query_pairs_fp16.append((qn.to(torch.float16), th_packed_fp16))

    groups = H_q // H_kv if q_head_to_kv is not None else 1
    scale = 1.0 / math.sqrt(D)
    keys_full_f16 = keys_full.half()
    values_full_f16 = values_full.half()
    buffer_f16 = buffer.half()
    value_buffer_f16 = value_buffer.half()

    print(
        f"attention v3 bench: layer {layer} H_q={H_q} H_kv={H_kv} "
        f"N_idx={n_index} N_buf={args.buffer_len} D={D} S={args.S}"
    )
    print("-" * 70)

    def run_kernel(fn):
        def wrapped():
            for qn, th in query_pairs_fp16:
                fn(
                    q=qn,
                    th_per_subspace=th,
                    state=state,
                    buffer_keys=buffer_f16,
                    buffer_values=value_buffer_f16,
                    keys_children=keys,
                    q_head_to_kv=q_head_to_kv,
                    scale=scale,
                )

        return wrapped

    ms = time_call(run_kernel(attention_v3_0), iters=args.iters, warmup=3)
    print(f"{'attention_v3_0':<20s} {ms / len(query_pairs_fp16):8.3f} ms/query")
    ms = time_call(run_kernel(attention_v3_1), iters=args.iters, warmup=3)
    print(f"{'attention_v3_1':<20s} {ms / len(query_pairs_fp16):8.3f} ms/query")

    def dense_attn_fp16():
        for qn, _ in query_pairs_fp16:
            if groups == 1:
                scores = torch.einsum("hd,hnd->hn", qn, keys_full_f16) * scale
                probs = torch.softmax(scores.float(), dim=-1).half()
                _ = torch.einsum("hn,hnd->hd", probs, values_full_f16)
            else:
                q_hg = qn.view(H_kv, groups, D)
                scores = torch.einsum("hgd,hnd->hgn", q_hg, keys_full_f16) * scale
                probs = torch.softmax(scores.float(), dim=-1).half()
                _ = torch.einsum("hgn,hnd->hgd", probs, values_full_f16)

    ms = time_call(dense_attn_fp16, iters=args.iters, warmup=3)
    print(f"{'dense attn fp16':<20s} {ms / len(query_pairs_fp16):8.3f} ms/query")

    def sdpa_baseline_fp16():
        for qn, _ in query_pairs_fp16:
            q4 = qn.view(1, H_q, 1, D)
            k4 = keys_full_f16.view(1, H_kv, N, D)
            v4 = values_full_f16.view(1, H_kv, N, D_v)
            _ = torch.nn.functional.scaled_dot_product_attention(
                q4,
                k4,
                v4,
                is_causal=False,
                scale=scale,
                enable_gqa=(groups > 1),
            )

    ms = time_call(sdpa_baseline_fp16, iters=args.iters, warmup=3)
    print(f"{'sdpa fp16':<20s} {ms / len(query_pairs_fp16):8.3f} ms/query")


if __name__ == "__main__":
    main()
