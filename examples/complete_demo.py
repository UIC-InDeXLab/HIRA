"""
Complete example demonstrating all Hira components.

This example shows:
1. Building a hierarchical index from scratch
2. Performing range searches
3. Using HiraCache with a model
4. Comparing different configurations
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import Hira components
from hira.index import KMeansIndexBuilder, RebuildUpdater, AllGPUPolicy
from hira.search import HalfspaceRangeSearcher
from hira.cache import HiraCache
from hira.attention import HiraAttention
from hira.utils import FixedThresholdStrategy


def demo_index_building():
    """Demonstrate hierarchical index building."""
    print("=" * 80)
    print("1. Hierarchical Index Building Demo")
    print("=" * 80)
    
    # Create some random key vectors
    num_keys = 1000
    head_dim = 64
    keys = torch.randn(num_keys, head_dim)
    print(f"\nCreated {num_keys} random key vectors of dimension {head_dim}")
    
    # Build a hierarchical index
    builder = KMeansIndexBuilder(max_iterations=50)
    index = builder.build(
        keys=keys,
        num_levels=3,
        branching_factor=10,
        device=torch.device("cpu"),
    )
    
    print(f"\nBuilt hierarchical index:")
    print(f"  - Number of levels: {index.num_levels()}")
    print(f"  - Total keys indexed: {index.num_keys}")
    
    # Show structure of each level
    for level_idx in range(index.num_levels()):
        level = index.get_level(level_idx)
        print(f"\n  Level {level_idx}:")
        print(f"    - Clusters: {level.num_clusters()}")
        print(f"    - Centroids shape: {level.centroids.shape}")
        mem = level.memory_usage()
        print(f"    - Memory: {mem['total_mb']:.2f} MB")
    
    # Show hirarchy path for a sample key
    key_idx = 0
    path = index.get_hirarchy_path(key_idx)
    print(f"\n  Hirarchy path for key {key_idx}: {path}")
    print(f"    (Cluster IDs at each level: coarse → fine)")


def demo_range_search():
    """Demonstrate range searching."""
    print("\n" + "=" * 80)
    print("2. Range Search Demo")
    print("=" * 80)
    
    # Create keys and build index
    num_keys = 500
    head_dim = 32
    keys = torch.randn(num_keys, head_dim)
    
    builder = KMeansIndexBuilder(max_iterations=30)
    index = builder.build(
        keys=keys,
        num_levels=2,
        branching_factor=10,
        device=torch.device("cpu"),
    )
    
    # Create a query
    query = torch.randn(head_dim)
    
    # Perform range search with different thresholds
    searcher = HalfspaceRangeSearcher()
    
    thresholds = [-2.0, 0.0, 2.0]
    
    print(f"\nPerforming range search over {num_keys} keys")
    print(f"Query dimension: {head_dim}")
    
    for threshold in thresholds:
        results = searcher.search(
            query=query,
            threshold=threshold,
            index=index,
            keys=keys,
        )
        
        # Verify correctness
        if len(results) > 0:
            selected_keys = keys[results]
            scores = torch.matmul(selected_keys, query)
            min_score = scores.min().item()
            max_score = scores.max().item()
            
            print(f"\n  Threshold: {threshold:.2f}")
            print(f"    - Keys selected: {len(results)} / {num_keys} ({len(results)/num_keys*100:.1f}%)")
            print(f"    - Score range: [{min_score:.2f}, {max_score:.2f}]")
            print(f"    - All scores >= threshold: {(scores >= threshold - 1e-5).all().item()}")
        else:
            print(f"\n  Threshold: {threshold:.2f}")
            print(f"    - Keys selected: 0 / {num_keys} (0.0%)")


def demo_hira_cache():
    """Demonstrate HiraCache usage."""
    print("\n" + "=" * 80)
    print("3. HiraCache Demo")
    print("=" * 80)
    
    # Create cache with custom configuration
    cache = HiraCache(
        num_levels=3,
        branching_factor=32,
        builder=KMeansIndexBuilder(max_iterations=50),
        updater=RebuildUpdater(update_frequency="every_n", update_interval=64),
        memory_policy=AllGPUPolicy(device="cpu"),
        build_index_every_n=64,
    )
    
    print("\nCreated HiraCache with configuration:")
    print(f"  - Levels: {cache.num_levels}")
    print(f"  - Branching factor: {cache.branching_factor}")
    print(f"  - Index build frequency: every {cache.build_index_every_n} tokens")
    
    # Simulate cache updates (as if from a model)
    batch_size = 1
    num_heads = 8
    head_dim = 64
    num_layers = 4
    
    print(f"\nSimulating generation with:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Attention heads: {num_heads}")
    print(f"  - Head dimension: {head_dim}")
    print(f"  - Layers: {num_layers}")
    
    # Simulate prefill (32 tokens)
    print("\n  Prefill phase (32 tokens)...")
    prefill_len = 32
    for layer_idx in range(num_layers):
        keys = torch.randn(batch_size, num_heads, prefill_len, head_dim)
        values = torch.randn(batch_size, num_heads, prefill_len, head_dim)
        cache.update(keys, values, layer_idx)
    
    # Simulate decode (64 tokens, one at a time)
    print("  Decode phase (64 tokens)...")
    decode_len = 64
    for step in range(decode_len):
        for layer_idx in range(num_layers):
            keys = torch.randn(batch_size, num_heads, 1, head_dim)
            values = torch.randn(batch_size, num_heads, 1, head_dim)
            cache.update(keys, values, layer_idx)
    
    # Show cache statistics
    print("\n  Cache statistics after generation:")
    cache_info = cache.get_cache_info()
    print(f"    - Total tokens: {cache_info['total_tokens']}")
    
    for layer_idx in range(min(2, num_layers)):
        layer_info = cache_info['layers'][layer_idx]
        print(f"\n    Layer {layer_idx}:")
        print(f"      - Sequence length: {layer_info['seq_length']}")
        print(f"      - Has index: {layer_info['has_index']}")
        
        if layer_info['has_index']:
            index_info = layer_info['index_info']
            print(f"      - Index levels: {index_info['num_levels']}")
            print(f"      - Indexed keys: {index_info['num_keys']}")
            mem = index_info['memory_usage']
            print(f"      - Index memory: {mem['total_mb']:.2f} MB")


def demo_configuration_comparison():
    """Compare different Hira configurations."""
    print("\n" + "=" * 80)
    print("4. Configuration Comparison")
    print("=" * 80)
    
    configurations = [
        {
            "name": "Fast (2 levels, sparse rebuild)",
            "num_levels": 2,
            "branching_factor": 16,
            "build_every": 128,
        },
        {
            "name": "Balanced (3 levels, moderate rebuild)",
            "num_levels": 3,
            "branching_factor": 32,
            "build_every": 64,
        },
        {
            "name": "Precise (4 levels, frequent rebuild)",
            "num_levels": 4,
            "branching_factor": 64,
            "build_every": 32,
        },
    ]
    
    print("\nComparing configurations:")
    print(f"\n{'Configuration':<40} {'Levels':<10} {'Branching':<12} {'Build Freq':<12}")
    print("-" * 80)
    
    for config in configurations:
        print(f"{config['name']:<40} {config['num_levels']:<10} "
              f"{config['branching_factor']:<12} {config['build_every']:<12}")
    
    print("\nTrade-offs:")
    print("  - More levels: Better search precision, higher memory cost")
    print("  - Higher branching factor: More clusters, better granularity")
    print("  - More frequent builds: Fresher index, higher computation cost")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("Hira Complete Demo")
    print("=" * 80)
    
    # Run each demo
    demo_index_building()
    demo_range_search()
    demo_hira_cache()
    demo_configuration_comparison()
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
Hira provides:

1. **Hierarchical Indexing**: Multi-level clustering of key vectors
2. **Efficient Range Search**: Find high-score keys without exhaustive search
3. **Integrated Cache**: HF-compatible cache with automatic index maintenance
4. **Flexible Configuration**: Tunable trade-offs between speed, memory, and accuracy

Next steps:
- Try basic_usage.py for model integration
- Try patch_llama.py for full HiraAttention
- Try benchmark.py to measure performance
- Read ARCHITECTURE.md for detailed design documentation
""")


if __name__ == "__main__":
    main()
