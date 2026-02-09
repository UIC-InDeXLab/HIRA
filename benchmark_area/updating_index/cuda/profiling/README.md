# CUDA update profiling

This folder contains a small profiler harness for `CUDAIndexer.update()`.

## Quick start

From the repo root:

```bash
python hira/benchmark_area/updating_index/cuda/profiling/profile_update.py \
  --depth two \
  --branching-factor 16 \
  --dim 128 \
  --initial-keys 65536 \
  --update-keys 4096 \
  --window-size 256 \
  --mode steps \
  --out-dir /tmp/hira_update_profile
```

Outputs:
- A console table of the top ops sorted by CUDA time.
- A Chrome trace at `/tmp/hira_update_profile/trace.json` (open in `chrome://tracing`).

## Modes

- `--mode full`: profiles the stock `CUDAIndexer.update()`.
- `--mode steps`: runs an equivalent update implementation but wrapped in `torch.profiler.record_function(...)` blocks so the trace shows high-level phases (nearest-parent, fill, radii update, overflow, etc.).

If you only need a “what kernels dominate” view, use `full`. If you want “what phase dominates”, use `steps`.
