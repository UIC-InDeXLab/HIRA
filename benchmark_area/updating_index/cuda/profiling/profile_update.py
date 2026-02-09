#!/usr/bin/env python3

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def _find_repo_root(start: Path) -> Path:
    for parent in start.resolve().parents:
        if (parent / "hira").is_dir() and (parent / "TODO.md").exists():
            return parent
    # Fallback: assume the repo root is 5 levels up from this file.
    return start.resolve().parents[5]


def _import_cuda_indexer():
    here = Path(__file__).resolve()
    root = _find_repo_root(here)
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from hira.index.indexer import CUDAIndexer  # pylint: disable=import-error

    return CUDAIndexer


@dataclass
class Config:
    depth: str
    branching_factor: int
    dim: int
    initial_keys: int
    update_keys: int
    window_size: int
    max_iterations: int
    pad_value: float
    warmup: int
    iters: int
    mode: str
    impl: str
    out_dir: Path
    with_stack: bool
    with_shapes: bool


def _make_keys(n: int, d: int, *, seed: int, device: str) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn((n, d), generator=g, dtype=torch.float32)
    return x.to(device=device).contiguous()


def _update_steps_two_level(index, new_keys: torch.Tensor) -> None:
    # A faithful copy of CUDAIndexer.update() with extra record_function ranges.
    if new_keys.numel() == 0:
        return
    if index.parents is None or index.children is None or index.parent_radii is None:
        raise RuntimeError("CUDAIndexer.update() called before build()")

    new_keys = new_keys.to("cuda").contiguous()
    bf = int(index.branching_factor)

    if index.pad_value is None:
        raise ValueError(
            "CUDAIndexer.update() requires pad_value to detect empty slots"
        )

    device = new_keys.device

    with record_function("update/nearest_parent"):
        valid_parent = ~(torch.all(index.parents == index.pad_value, dim=-1))
        nearest_parent, _nearest_d2 = index._nearest_l2(
            new_keys, index.parents, valid_mask=valid_parent
        )

    with record_function("update/fill_existing_children"):
        placed_mask, _ = index._fill_existing_children(
            x=new_keys, parent_idx=nearest_parent
        )

    with record_function("update/update_parent_radii"):
        if placed_mask.any():
            inserted_keys = new_keys[placed_mask]
            inserted_parents = nearest_parent[placed_mask]
            index._update_parent_radii_for_inserted(
                inserted_keys=inserted_keys,
                inserted_parent_idx=inserted_parents,
            )

    with record_function("update/overflow_build_new_parents"):
        overflow = new_keys[~placed_mask]
        if overflow.numel() == 0:
            return

        new_parents, new_children_flat, new_parent_radii = (
            index._build_new_parents_from_overflow(overflow, seed=1234)
        )
        if new_parents.numel() == 0:
            return

    with record_function("update/overflow_append_blocks"):
        index.parents = torch.cat([index.parents, new_parents], dim=0).contiguous()
        index.parent_radii = torch.cat(
            [index.parent_radii, new_parent_radii.to(index.parent_radii.dtype)], dim=0
        ).contiguous()
        index.children = torch.cat(
            [index.children, new_children_flat], dim=0
        ).contiguous()

        # Extend score buffer
        if index.buffer is None:
            index.buffer = torch.zeros(
                (index.children.shape[0],), device="cuda", dtype=torch.float32
            )
        else:
            add = new_children_flat.shape[0]
            if add > 0:
                index.buffer = torch.cat(
                    [
                        index.buffer,
                        torch.zeros((add,), device="cuda", dtype=torch.float32),
                    ],
                    dim=0,
                )


def _update_steps_three_level(index, new_keys: torch.Tensor) -> None:
    # Copy of CUDAIndexer.update() (THREE_LEVELS) with record_function ranges.
    if new_keys.numel() == 0:
        return
    if index.parents is None or index.children is None or index.parent_radii is None:
        raise RuntimeError("CUDAIndexer.update() called before build()")
    if index.grand_parents is None or index.grand_parent_radii is None:
        raise RuntimeError("THREE_LEVELS build state incomplete")

    new_keys = new_keys.to("cuda").contiguous()
    bf = int(index.branching_factor)

    if index.pad_value is None:
        raise ValueError(
            "CUDAIndexer.update() requires pad_value to detect empty slots"
        )

    device = new_keys.device

    with record_function("update/nearest_parent"):
        valid_parent = ~(torch.all(index.parents == index.pad_value, dim=-1))
        nearest_parent, _nearest_d2 = index._nearest_l2(
            new_keys, index.parents, valid_mask=valid_parent
        )

    with record_function("update/fill_existing_children"):
        placed_mask, _ = index._fill_existing_children(
            x=new_keys, parent_idx=nearest_parent
        )

    with record_function("update/update_parent_radii"):
        if placed_mask.any():
            inserted_keys = new_keys[placed_mask]
            inserted_parents = nearest_parent[placed_mask]
            index._update_parent_radii_for_inserted(
                inserted_keys=inserted_keys,
                inserted_parent_idx=inserted_parents,
            )

    with record_function("update/update_grandparent_radii"):
        if placed_mask.any():
            inserted_parents = nearest_parent[placed_mask]
            g = index.grand_parents.shape[0]
            gp_idx = (inserted_parents // bf).to(torch.int64)
            gp_centers = index.grand_parents.index_select(0, gp_idx)
            dist_gp = torch.linalg.norm(
                index.parents.index_select(0, inserted_parents) - gp_centers, dim=1
            ).float()
            total = (
                dist_gp + index.parent_radii.index_select(0, inserted_parents).float()
            )

            upd_gp = torch.full((g,), float("-inf"), device=device, dtype=torch.float32)
            upd_gp.scatter_reduce_(0, gp_idx, total, reduce="amax", include_self=True)
            upd_gp = torch.where(
                torch.isfinite(upd_gp), upd_gp, torch.zeros_like(upd_gp)
            )
            index.grand_parent_radii = torch.maximum(
                index.grand_parent_radii.float(), upd_gp
            ).to(dtype=index.grand_parent_radii.dtype)

    with record_function("update/overflow_build_new_parents"):
        overflow = new_keys[~placed_mask]
        if overflow.numel() == 0:
            return

        new_parents, new_children_flat, new_parent_radii = (
            index._build_new_parents_from_overflow(overflow, seed=1234)
        )
        if new_parents.numel() == 0:
            return

    with record_function("update/overflow_place_or_append"):
        K = new_parents.shape[0]
        d = new_parents.shape[1]

        G = index.grand_parents.shape[0]
        P_total = index.parents.shape[0]
        if P_total != G * bf:
            raise RuntimeError(
                f"Invalid THREE_LEVELS layout: parents={P_total} but grand_parents={G} and bf={bf}"
            )
        if index.children.shape[0] != P_total * bf:
            raise RuntimeError(
                "Invalid THREE_LEVELS layout: children not aligned with parents blocks"
            )

        parents2 = index.parents.view(G, bf, d)
        parent_empty = torch.all(parents2 == index.pad_value, dim=-1)  # (G,bf)
        parent_avail = parent_empty.sum(dim=1).to(torch.int64)  # (G,)
        slot_ids = torch.arange(bf, device=device, dtype=torch.int64)
        slot_mat = slot_ids.view(1, bf).expand(G, bf)
        empty_first = torch.where(parent_empty, slot_mat, torch.full_like(slot_mat, bf))
        empty_sorted = torch.sort(empty_first, dim=1).values  # (G,bf)

        gp_near, _ = index._nearest_l2(
            new_parents, index.grand_parents, valid_mask=None
        )

        order = torch.argsort(gp_near)
        gp_sorted = gp_near[order]
        src_sorted = order

        counts = torch.bincount(gp_near, minlength=G).to(torch.int64)
        prefix = torch.cumsum(counts, dim=0)
        start = prefix.index_select(0, gp_sorted) - counts.index_select(0, gp_sorted)
        rank = torch.arange(K, device=device, dtype=torch.int64) - start

        can_place = rank < parent_avail.index_select(0, gp_sorted)

        place_src = src_sorted[can_place]
        place_gp = gp_sorted[can_place]
        place_rank = rank[can_place]
        place_slot = empty_sorted[place_gp, place_rank]
        place_parent_global = place_gp * bf + place_slot

        if place_src.numel() > 0:
            index.parents.index_copy_(
                0, place_parent_global, new_parents.index_select(0, place_src)
            )
            index.parent_radii.index_copy_(
                0,
                place_parent_global,
                new_parent_radii.index_select(0, place_src).to(
                    index.parent_radii.dtype
                ),
            )

            children_blocks = new_children_flat.view(K, bf, d)
            dst_children3 = index.children.view(P_total, bf, d)
            dst_children3.index_copy_(
                0, place_parent_global, children_blocks.index_select(0, place_src)
            )

            gp_centers = index.grand_parents.index_select(0, place_gp)
            dist_gp = torch.linalg.norm(
                index.parents.index_select(0, place_parent_global) - gp_centers, dim=1
            ).float()
            total = (
                dist_gp
                + index.parent_radii.index_select(0, place_parent_global).float()
            )

            upd_gp = torch.full((G,), float("-inf"), device=device, dtype=torch.float32)
            upd_gp.scatter_reduce_(0, place_gp, total, reduce="amax", include_self=True)
            upd_gp = torch.where(
                torch.isfinite(upd_gp), upd_gp, torch.zeros_like(upd_gp)
            )
            index.grand_parent_radii = torch.maximum(
                index.grand_parent_radii.float(), upd_gp
            ).to(dtype=index.grand_parent_radii.dtype)

        if can_place.all():
            # Extend buffer only if children grew (it did not in this case)
            return

        remaining_src = src_sorted[~can_place]
        rem_parents = new_parents.index_select(0, remaining_src)
        rem_pr = new_parent_radii.index_select(0, remaining_src)
        rem_children_blocks = new_children_flat.view(K, bf, d).index_select(
            0, remaining_src
        )

        M = rem_parents.shape[0]
        g_new = int((M + bf - 1) // bf)
        g_new = max(1, g_new)
        torch.manual_seed(4321)
        perm = torch.randperm(M, device=device)
        gp_choose = perm[:g_new]
        gp_new = rem_parents.index_select(0, gp_choose).contiguous()

        gp_assign, gp_d2 = index._nearest_l2(rem_parents, gp_new, valid_mask=None)

        order2 = torch.argsort(gp_assign * (M + 1) + torch.argsort(gp_d2))
        ga_sorted = gp_assign[order2]
        rp_sorted = rem_parents[order2]
        rr_sorted = rem_pr[order2]
        rc_sorted = rem_children_blocks[order2]

        counts2 = torch.bincount(ga_sorted, minlength=g_new).to(torch.int64)
        prefix2 = torch.cumsum(counts2, dim=0)
        start2 = prefix2.index_select(0, ga_sorted) - counts2.index_select(0, ga_sorted)
        rank2 = torch.arange(M, device=device, dtype=torch.int64) - start2

        placed2 = rank2 < bf
        parents_block = torch.full(
            (g_new * bf, d), index.pad_value, device=device, dtype=torch.float32
        )
        pr_block = torch.zeros(
            (g_new * bf,), device=device, dtype=index.parent_radii.dtype
        )
        children_block = torch.full(
            (g_new * bf, bf, d), index.pad_value, device=device, dtype=torch.float32
        )

        if placed2.any():
            ga_p = ga_sorted[placed2]
            r_p = rank2[placed2]
            dst = ga_p * bf + r_p
            parents_block.index_copy_(0, dst, rp_sorted[placed2])
            pr_block.index_copy_(
                0, dst, rr_sorted[placed2].to(index.parent_radii.dtype)
            )
            children_block.index_copy_(0, dst, rc_sorted[placed2])

        if (~placed2).any():
            leftovers_p = rp_sorted[~placed2]
            leftovers_r = rr_sorted[~placed2]
            leftovers_c = rc_sorted[~placed2]

            next_slot = (
                ((torch.all(parents_block == index.pad_value, dim=-1)).logical_not())
                .view(g_new, bf)
                .sum(dim=1)
                .to(torch.int64)
            )

            lp = leftovers_p
            lp_norm = (lp * lp).sum(dim=1, keepdim=True)
            gp_norm = (gp_new * gp_new).sum(dim=1)
            d2mat = lp_norm + gp_norm[None, :] - 2.0 * (lp @ gp_new.t())
            d2mat = torch.clamp_min(d2mat, 0.0)

            for j in range(lp.shape[0]):
                cap = next_slot < bf
                if not bool(cap.any().item()):
                    raise RuntimeError(
                        "Invariant broken: no capacity for new grandparent slots"
                    )
                d2j = torch.where(
                    cap, d2mat[j], torch.tensor(float("inf"), device=device)
                )
                gsel = int(torch.argmin(d2j).item())
                s = int(next_slot[gsel].item())
                idx = gsel * bf + s
                parents_block[idx] = leftovers_p[j]
                pr_block[idx] = leftovers_r[j].to(index.parent_radii.dtype)
                children_block[idx] = leftovers_c[j]
                next_slot[gsel] += 1

        children_block_flat = children_block.view(g_new * bf * bf, d)

        gp_f = gp_new.float()
        parents_f = parents_block.float().view(g_new, bf, d)
        pr_f = pr_block.float().view(g_new, bf)
        validp = ~torch.all(parents_f == index.pad_value, dim=-1)
        dists_gp = torch.linalg.norm(parents_f - gp_f[:, None, :], dim=-1)
        totals = dists_gp + pr_f
        totals = torch.where(validp, totals, torch.tensor(float("-inf"), device=device))
        gp_radii_new = torch.max(totals, dim=1).values
        gp_radii_new = torch.where(
            torch.isfinite(gp_radii_new), gp_radii_new, torch.zeros_like(gp_radii_new)
        ).to(index.grand_parent_radii.dtype)

        index.grand_parents = torch.cat(
            [index.grand_parents, gp_new], dim=0
        ).contiguous()
        index.grand_parent_radii = torch.cat(
            [index.grand_parent_radii, gp_radii_new], dim=0
        ).contiguous()
        index.parents = torch.cat([index.parents, parents_block], dim=0).contiguous()
        index.parent_radii = torch.cat(
            [index.parent_radii, pr_block], dim=0
        ).contiguous()
        index.children = torch.cat(
            [index.children, children_block_flat], dim=0
        ).contiguous()

        if index.buffer is None:
            index.buffer = torch.zeros(
                (index.children.shape[0],), device="cuda", dtype=torch.float32
            )
        else:
            add = children_block_flat.shape[0]
            if add > 0:
                index.buffer = torch.cat(
                    [
                        index.buffer,
                        torch.zeros((add,), device="cuda", dtype=torch.float32),
                    ],
                    dim=0,
                )


def _run_profile(cfg: Config) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available (torch.cuda.is_available() is False)")

    CUDAIndexer = _import_cuda_indexer()

    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build initial index
    with record_function("setup/build_index"):
        initial = _make_keys(cfg.initial_keys, cfg.dim, seed=123, device="cuda")

        depth = (
            CUDAIndexer.DEPTH.TWO_LEVELS
            if cfg.depth == "two"
            else CUDAIndexer.DEPTH.THREE_LEVELS
        )
        index = CUDAIndexer(
            depth=depth,
            max_iterations=cfg.max_iterations,
            branching_factor=cfg.branching_factor,
            verbose=False,
            pad_value=cfg.pad_value,
        ).build(initial)

    updates = _make_keys(cfg.update_keys, cfg.dim, seed=999, device="cuda")

    def do_one_pass():
        for s in range(0, updates.shape[0], cfg.window_size):
            chunk = updates[s : s + cfg.window_size]
            if cfg.mode == "full":
                if cfg.impl == "update_v2":
                    index.update_v2(chunk)
                else:
                    index.update(chunk)
            else:
                if cfg.impl == "update_v2":
                    # In steps mode, call update_v2 directly (it has its own internal structure).
                    index.update_v2(chunk)
                else:
                    if cfg.depth == "two":
                        _update_steps_two_level(index, chunk)
                    else:
                        _update_steps_three_level(index, chunk)

    # Warmup (excluded from profiler)
    for _ in range(cfg.warmup):
        do_one_pass()
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(
        activities=activities,
        record_shapes=cfg.with_shapes,
        with_stack=cfg.with_stack,
        profile_memory=True,
    ) as prof:
        for _ in range(cfg.iters):
            with record_function("update/pass"):
                do_one_pass()
        torch.cuda.synchronize()

    # Console summary
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=80,
    )
    print(table)

    # Chrome trace
    trace_path = out_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\nWrote Chrome trace to: {trace_path}")


def main():
    p = argparse.ArgumentParser(description="Profile CUDAIndexer.update()")
    p.add_argument("--depth", choices=["two", "three"], default="two")
    p.add_argument("--branching-factor", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--initial-keys", type=int, default=10000)
    p.add_argument("--update-keys", type=int, default=80000)
    p.add_argument("--window-size", type=int, default=256)
    p.add_argument("--max-iterations", type=int, default=1)
    p.add_argument("--pad-value", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--mode", choices=["full", "steps"], default="steps")
    p.add_argument("--impl", choices=["update", "update_v2"], default="update")
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/hira_update_profile"))
    p.add_argument("--with-stack", action="store_true")
    p.add_argument("--with-shapes", action="store_true")
    args = p.parse_args()

    cfg = Config(
        depth=args.depth,
        branching_factor=args.branching_factor,
        dim=args.dim,
        initial_keys=args.initial_keys,
        update_keys=args.update_keys,
        window_size=args.window_size,
        max_iterations=args.max_iterations,
        pad_value=args.pad_value,
        warmup=args.warmup,
        iters=args.iters,
        mode=args.mode,
        impl=args.impl,
        out_dir=args.out_dir,
        with_stack=args.with_stack,
        with_shapes=args.with_shapes,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    _run_profile(cfg)


if __name__ == "__main__":
    # Avoid excessive CPU threading noise in traces.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
