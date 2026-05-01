"""Profile each step of TA_attention_v_4_0 on the Llama capture."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path("/home/mohsen/kvcache/hira")
sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.TA_filter_alg import attention_kernels, build_kernels
from hira.benchmark_area.quick_pruning.pruning_bench_utils import CaptureState, _q_to_kv_map
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels._TA_common import _LOG2E, next_pow2
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels._TA_triton_attn_premask import run_ta_attn_premask
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels._TA_triton_centroid_scores import run_ta_centroid_scores
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels._TA_triton_depth import run_ta_stop_depth
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels._TA_triton_masks_v4 import run_ta_mark_candidates_from_children
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels._TA_triton_reduce import run_ta_reduce


CAPTURE_PATH = "/home/mohsen/kvcache/hira/benchmark_area/quick_pruning/capture_qkv_8000_meta-llama_Llama-3.2-3B-Instruct.pt"


def main() -> None:
    torch.set_grad_enabled(False)
    cap = CaptureState.load(CAPTURE_PATH)
    layer = 15
    if layer not in cap.layer_ids():
        layer = cap.layer_ids()[len(cap.layer_ids()) // 2]
    queries_cpu, keys_cpu, values_cpu = cap.to_layer_tensors(layer)
    keys = keys_cpu.to(device="cuda", dtype=torch.float32)
    values = values_cpu.to(device="cuda", dtype=torch.float32)
    queries = queries_cpu.to(device="cuda", dtype=torch.float32)
    h_q = queries.shape[0]
    h_kv = keys.shape[0]
    q_map = _q_to_kv_map(h_q, h_kv, "cuda") if h_q != h_kv else None

    n = keys.shape[1]
    n_buf = 256
    n_idx = n - n_buf
    keys_idx, keys_buf = keys[:, :n_idx], keys[:, n_idx:]
    values_idx, values_buf = values[:, :n_idx], values[:, n_idx:]

    builds = build_kernels()
    build_fn = builds["TA_build_v_1_1"].fn
    state = build_fn(keys_idx, bf=4, n_subspaces=8, values=values_idx)

    # Pick first query
    q = queries[:, 0, :].cuda()  # (H_q, D)
    # Compute threshold T = 20-th largest dot
    keys_full = keys_idx
    keys_eval = keys_full if q_map is None else keys_full.index_select(0, q_map)
    scores_full = torch.einsum("hd,hnd->hn", q.float(), keys_eval.float())
    top_vals, _ = scores_full.topk(20, dim=-1)
    threshold = top_vals[:, 19].contiguous()

    # Setup like v4.0
    import math
    d = q.shape[1]
    d_v = values_idx.shape[-1]
    s_sub = state["n_subspaces"]
    k_clusters = state["K"]
    n_pad = state["N_pad"]
    groups = h_q // h_kv
    scale = 1.0 / math.sqrt(d)
    scale_log2e = scale * _LOG2E
    q_f16 = q.to(torch.float16).contiguous()
    threshold_f32 = threshold.float().contiguous()

    scores = torch.empty(h_q, s_sub, k_clusters, device=q.device, dtype=torch.float16)
    depth = torch.empty(h_q, device=q.device, dtype=torch.int32)
    cand_mask = torch.empty(h_q, n_pad, device=q.device, dtype=torch.int8)
    invalid_i8 = state["invalid_mask"].to(torch.int8).contiguous()
    NUM_SPLITS = 32
    out_m = torch.empty(h_q, NUM_SPLITS, device=q.device, dtype=torch.float32)
    out_l = torch.empty(h_q, NUM_SPLITS, device=q.device, dtype=torch.float32)
    out_o = torch.empty(h_q, NUM_SPLITS, d_v, device=q.device, dtype=torch.float32)
    out = torch.empty(h_q, d_v, device=q.device, dtype=torch.float32)

    # Buffer
    bk = keys_buf.index_select(0, q_map).to(torch.float16).contiguous()
    bv = values_buf.index_select(0, q_map).to(torch.float16).contiguous()
    bk_t = bk.transpose(-1, -2).contiguous()
    b_inv = torch.zeros(bk.shape[0], bk.shape[1], dtype=torch.int8, device=q.device)
    groups_pow = max(next_pow2(groups), 4)

    print(f"H_q={h_q} H_kv={h_kv} K={k_clusters} N_pad={n_pad} S={s_sub} groups={groups}")

    def time_step(name, fn, iters=200):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1e3 / iters
        print(f"  {name:30s} {elapsed:.4f} ms")
        return elapsed

    def step_centroid():
        run_ta_centroid_scores(q_f16, state["centers_padded_f16"], state["dim_offsets"], state["dim_widths"], scores, groups=groups, block_k=64)

    def step_sort():
        return torch.sort(scores, dim=-1, descending=True)

    sorted_scores, order = torch.sort(scores, dim=-1, descending=True)

    def step_depth():
        run_ta_stop_depth(sorted_scores, threshold_f32, depth)

    cand_mask.zero_()
    run_ta_mark_candidates_from_children(order, depth, state["children_padded_i32"], cand_mask, n_pad=n_pad, groups=groups, block_rows=16, num_splits=128, num_warps=8, num_stages=2)

    def step_zero():
        cand_mask.zero_()

    def step_mark():
        run_ta_mark_candidates_from_children(order, depth, state["children_padded_i32"], cand_mask, n_pad=n_pad, groups=groups, block_rows=16, num_splits=128, num_warps=8, num_stages=2)

    def step_attn():
        run_ta_attn_premask(q=q_f16, keys_t_f16=state["keys_padded_t_f16"], values_f16=state["values_padded_f16"], cand_mask_i8=cand_mask, invalid_mask_i8=invalid_i8, threshold_f32=threshold_f32, buf_keys_t_f16=bk_t, buf_values_f16=bv, buf_invalid_i8=b_inv, h_kv_eff=h_kv, n_pad=n_pad, scale=scale, scale_log2e=scale_log2e, groups=groups, groups_pow=groups_pow, block_n=32, num_splits=NUM_SPLITS, out_m=out_m, out_l=out_l, out_o=out_o, num_warps=4, num_stages=3)

    def step_reduce():
        run_ta_reduce(out_m, out_l, out_o, out)

    print("\nPer-step timing (Llama, N=8000-256, S=8, bf=4):")
    time_step("centroid scores", step_centroid)
    time_step("torch.sort", step_sort)
    time_step("stop depth", step_depth)
    time_step("zero cand_mask", step_zero)
    time_step("mark candidates", step_mark)
    time_step("attn premask", step_attn)
    time_step("reduce", step_reduce)

    # Now measure depth
    from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels._TA_common import compute_centroid_scores, build_selected_clusters
    print("\nDepth statistics over queries:")
    depths_list = []
    for q_idx in range(min(20, queries.shape[1])):
        q_t = queries[:, q_idx, :].cuda()
        keys_eval = keys_full if q_map is None else keys_full.index_select(0, q_map)
        scores_full = torch.einsum("hd,hnd->hn", q_t.float(), keys_eval.float())
        top_vals, _ = scores_full.topk(20, dim=-1)
        thr = top_vals[:, 19].contiguous()
        cs_scores = compute_centroid_scores(q_t.to(torch.float16), state["centers_padded_f16"], state["dim_slices"], q_map)
        srt, _ = torch.sort(cs_scores, dim=-1, descending=True)
        from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels._TA_common import stop_depth_per_head
        d_per = stop_depth_per_head(srt, thr.float())
        depths_list.append(d_per)
    all_d = torch.stack(depths_list, dim=0).flatten().cpu().numpy()
    import numpy as np
    print(f"  depth: min={all_d.min()} max={all_d.max()} mean={all_d.mean():.1f} median={np.median(all_d):.1f} p95={np.percentile(all_d, 95):.1f}")
    print(f"  K={k_clusters} (so depth/K mean = {all_d.mean()/k_clusters*100:.1f}%)")


if __name__ == "__main__":
    main()
