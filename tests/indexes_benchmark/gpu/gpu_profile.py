import torch
import sys
from pathlib import Path
import math
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hira.index import KMeansIndex, KMeansIndexConfig, Index
from hira.search import HalfspaceSearcher


@torch.no_grad()
def bench_cuda_ms(fn, *args, warmup=10, iters=100):
    # warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        output = fn(*args)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, output


def brute_force_search(index: "Index", query: torch.Tensor, threshold: float):
    return brute_force_gpu(index, query, threshold)


def indexed_search(index, query, threshold):
    return search_gpu(query, threshold, index)


"""GPU PROFILING CODE BELOW"""

from torch.profiler import schedule, profile, ProfilerActivity, record_function


# @torch.compile(mode="max-autotune")
def search_gpu(query, threshold, index):
    query_norm = query / torch.norm(query, p=2)

    level0 = index.levels[0]
    level1 = index.levels[1]

    # with record_function("* level1 filter"):
    #     scores = torch.matmul(level1.ball_centers, query_norm)
    #     # parent_mask = (scores + level1.ball_radii) >= threshold
    #     best_parent = torch.argmax(scores)

    with record_function("* level1 filter"):
        scores = torch.matmul(level1.ball_centers, query_norm)
        ub = scores + level1.ball_radii
        # top_parent = torch.argmax(ub)
        _, parents = torch.topk(ub, k=4, largest=True, sorted=False)

    with record_function("* map 1 -> 0"):
        # child_mask = parent_mask[level0.child2parent]
        start = level0.p_pointer[parents[0]]
        end = level0.p_pointer[parents[0] + 1]

    with record_function("* retrieve"):
        keys = index.keys[start:end]
        # keys = torch.cat([index.keys[start_1:end_1], index.keys[start_2:end_2]], dim=0)

    with record_function("* final scoring"):
        scores = torch.matmul(keys, query_norm)
        output = (scores >= threshold).nonzero(as_tuple=True)[0]

    # print(f"  Indexed search found {output.numel()} results")

    return output


def brute_force_gpu(index: "Index", query: torch.Tensor, threshold: float):
    """Brute-force search."""
    query_norm = query / torch.norm(query, p=2)
    with record_function("* brute-force"):
        scores = torch.matmul(index.keys, query_norm)
        result = (scores >= threshold).nonzero(as_tuple=True)[0]

    # print(f"  Brute-force search found {result.numel()} results")

    return result


def index_prep(index):
    level0 = index.levels[0]
    level1 = index.levels[1]
    print(f"level 0 length: {len(level0.ball_centers)}")
    print(f"level 1 length: {len(level1.ball_centers)}")
    inds = torch.arange(len(level1.ball_centers))
    avg_size = level0.p_pointer[inds + 1] - level0.p_pointer[inds]
    print(f"Single cluster size: {torch.mean(avg_size.float()).item():.2f}")
    # breakpoint()
    return index


def profile_gpu(my_fn, *args, warmup=10, iters=100):
    # Warmup (important!)
    for _ in range(warmup):
        my_fn(*args)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=5, warmup=5, active=20, repeat=1),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,  # can be expensive but very helpful
    ) as prof:
        for _ in range(iters):
            my_fn(*args)
            prof.step()

    print(
        prof.key_averages(group_by_input_shape=False).table(
            sort_by="cuda_time_total", row_limit=10
        )
    )


def generate_real_data(
    num_keys: int, dim: int, real_data_path: str, seed: int = 42
) -> torch.Tensor:
    """Load real KV cache data from NPZ file.

    Args:
        num_keys: Number of keys to use (will subsample if needed)
        dim: Expected dimension (for validation)
        real_data_path: Path to .npz file containing 'keys' array
        seed: Random seed for subsampling

    Returns:
        Tensor of shape (num_keys, dim) containing real KV cache keys
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"  Loading real data from: {real_data_path}")
    data = np.load(real_data_path)
    keys = torch.from_numpy(data["keys"]).float()

    # Subsample if needed
    if len(keys) > num_keys:
        indices = torch.randperm(len(keys))[:num_keys]
        keys = keys[indices]

    print(f"  Loaded {len(keys)} real keys (dimension={keys.shape[1]})")

    return keys


DEFAULT_NUM_KEYS = 90000
DEFAULT_DIM = 128
DEFAULT_BRANCHING_FACTOR = 256
DEFAULT_LEVELS = 2  # Auto-calculate if none
DEFAULT_ITERATIONS = 1
DEFAULT_DEVICE = "cuda"
DEFAULT_TARGET_RESULTS = 10
DEFAULT_NUM_RUNS = 50
# DEFAULT_DATA_PATH = "/home/mohsen/kvcache/hira/tests/kv_sampling/kv_data/kv_data_qwen_Qwen2.5-3B-Instruct_layer35_20251227_223030.npz"
DEFAULT_DATA_PATH = "/home/mohsen/kvcache/hira/tests/kv_sampling/kv_data/kv_data_Meta-Llama-3-8B-Instruct_layer31_20251219_004657.npz"


def main():
    print("=" * 80)
    print("LINE-BY-LINE PROFILING")
    print("=" * 80)
    print()

    # ===== CONFIGURATION =====
    num_keys = DEFAULT_NUM_KEYS
    dim = DEFAULT_DIM
    branching_factor = DEFAULT_BRANCHING_FACTOR
    num_levels = (
        DEFAULT_LEVELS
        if DEFAULT_LEVELS is not None
        else math.ceil(math.log(num_keys) / math.log(branching_factor))
    )
    max_iterations = DEFAULT_ITERATIONS
    device = DEFAULT_DEVICE
    target_results = DEFAULT_TARGET_RESULTS
    num_runs = DEFAULT_NUM_RUNS
    # =========================

    print(f"Configuration:")
    print(f"  Keys: {num_keys}")
    print(f"  Dimension: {dim}")
    print(f"  Levels: {num_levels}")
    print(f"  Branching factor: {branching_factor}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Device: {device}")
    print(f"  Target results: {target_results}")
    print()

    print("Generating data...")
    keys = generate_real_data(num_keys, dim, real_data_path=DEFAULT_DATA_PATH)
    keys = keys.to(device)

    print("Building index...")
    config = KMeansIndexConfig(
        num_levels=num_levels,
        branching_factor=branching_factor,
        max_iterations=max_iterations,
        device=device,
    )
    index = KMeansIndex(config)
    index.build(keys)
    index = index_prep(index)

    print("Creating query...")
    query = torch.randn(dim).to(device)

    # Find threshold for target number of results
    query_norm = query / torch.norm(query, p=2)
    all_scores = torch.matmul(keys, query_norm)
    # Sort and find threshold that gives exactly the target number of results
    sorted_scores, _ = torch.sort(all_scores, descending=True)
    threshold = sorted_scores[min(target_results, len(sorted_scores) - 1)].item()
    expected_count = (all_scores >= threshold).sum().item()

    print(
        f"\nRunning profiled searches (threshold={threshold:.4f}, expected ~{expected_count} results)..."
    )

    # Profile indexed search
    print("  Indexed search...")
    t_index, output = bench_cuda_ms(
        indexed_search, index, query, threshold, warmup=10, iters=num_runs
    )

    # Profile brute-force search
    print("  Brute-force search...")
    t_brute, _ = bench_cuda_ms(
        brute_force_search, index, query, threshold, warmup=10, iters=num_runs
    )
    print("\nDone!")

    print(f"CUDA event timing (ms): indexed={t_index:.6f}  brute={t_brute:.6f}")

    # output[1].print_stats()

    print("\nDetailed profiling for indexed search:")
    profile_gpu(indexed_search, index, query, threshold, warmup=10, iters=num_runs)
    print("\nDetailed profiling for brute-force search:")
    profile_gpu(brute_force_search, index, query, threshold, warmup=10, iters=num_runs)


if __name__ == "__main__":
    main()
