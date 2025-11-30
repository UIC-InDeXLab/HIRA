# Hira Quick Start Guide

## Installation

```bash
cd /home/mohsen/kvcache/hira
pip install -e .
```

For development with all dependencies:
```bash
pip install -e ".[dev,examples]"
```

## Quick Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hira import HiraCache

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Create HiraCache
cache = HiraCache(num_levels=3, branching_factor=32)

# Generate
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, past_key_values=cache, max_new_tokens=50)

print(tokenizer.decode(outputs[0]))
```

## Running Examples

### Basic Usage
```bash
cd examples
python basic_usage.py
```

### Complete Demo (no model required)
```bash
python complete_demo.py
```

### Patch Llama with HiraAttention
```bash
python patch_llama.py
```

### Benchmark Performance
```bash
python benchmark.py
```

## Running Tests

```bash
cd tests
pytest test_index.py -v
pytest test_search.py -v
```

## Configuration Options

### HiraCache Configuration
```python
cache = HiraCache(
    num_levels=3,              # Number of hirarchy levels
    branching_factor=32,       # Clusters per level
    build_index_every_n=128,   # Rebuild index every N tokens
)
```

### HiraAttention Configuration
```python
from hira import HiraAttention
from hira.utils import FixedThresholdStrategy
from hira.search import HalfspaceRangeSearcher

attention = HiraAttention(
    threshold_strategy=FixedThresholdStrategy(threshold=0.0),
    range_searcher=HalfspaceRangeSearcher(),
    use_hira_during_prefill=False,
)
```

## Project Structure

```
hira/
├── __init__.py              # Main package
├── README.md                # Project overview
├── ARCHITECTURE.md          # Detailed architecture docs
├── setup.py                 # Installation
├── requirements.txt         # Dependencies
│
├── index/                   # Hierarchical indexing
│   ├── builders.py          # Index construction
│   ├── structure.py         # Index data structures
│   ├── updater.py           # Index updates
│   └── memory_policy.py     # GPU/CPU placement
│
├── search/                  # Range searching
│   └── range_searcher.py    # Halfspace search
│
├── attention/               # Attention mechanisms
│   ├── hira_attention.py   # Core attention
│   └── processor.py         # HF processor
│
├── cache/                   # Cache management
│   └── hira_cache.py       # Custom HF cache
│
├── utils/                   # Utilities
│   └── threshold.py         # Threshold strategies
│
├── kernels/                 # C++/CUDA kernels (future)
│   ├── README.md
│   ├── cpp/
│   └── cuda/
│
├── examples/                # Usage examples
│   ├── basic_usage.py
│   ├── patch_llama.py
│   ├── complete_demo.py
│   └── benchmark.py
│
└── tests/                   # Unit tests
    ├── test_index.py
    └── test_search.py
```

## Key Concepts

### 1. Hierarchical Index
Multi-level clustering of key vectors for efficient range searching.

### 2. Range Search
Find keys where `query · key >= threshold` without checking all keys.

### 3. Sparse Attention
Compute attention only over selected high-score keys.

### 4. HiraCache
HuggingFace-compatible cache that maintains hierarchical indexes.

## Performance Tips

1. **Adjust num_levels**: More levels = finer granularity but higher memory
2. **Tune branching_factor**: Higher = more clusters = better precision
3. **Set build_index_every_n**: Balance freshness vs computation cost
4. **Use during decode only**: Set `use_hira_during_prefill=False`

## Troubleshooting

### Issue: Index not being built
- Check `build_index_every_n` setting
- Ensure enough tokens have been generated
- Verify cache type is `HiraCache`

### Issue: Slow generation
- Reduce `num_levels` or `branching_factor`
- Increase `build_index_every_n`
- Check if using CUDA device

### Issue: High memory usage
- Use `HybridGPUCPUPolicy` for memory tiering
- Reduce `num_levels`
- Lower `branching_factor`

## Next Steps

1. Read `ARCHITECTURE.md` for detailed design
2. Run examples to understand usage patterns
3. Experiment with different configurations
4. Benchmark on your specific use case
5. Consider contributing C++/CUDA kernels

## Support

For issues and questions, please refer to:
- `ARCHITECTURE.md` - Design documentation
- `examples/` - Usage examples
- `tests/` - Test cases
- `kernels/README.md` - Future optimizations
