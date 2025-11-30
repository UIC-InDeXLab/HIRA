# Hira: Hierarchical Range-Searching Attention

WIP

## Overview

Hira implements an attention mechanism that:
- Maintains a hierarchical index over key vectors in the KV cache
- For each query, computes a score threshold and performs halfspace range search
- Returns only "high-score" keys whose dot product with the query exceeds the threshold
- Computes attention only over the selected key subset

## Architecture

### Core Components

1. **Index Building** (`index/builders.py`)

2. **Index Representation** (`index/structure.py`)

3. **Index Updates** (`index/updater.py`)

4. **Memory Tiering** (`index/memory_policy.py`)
   - Manages device placement of index levels (GPU/CPU)
   - Configurable policies for partial spilling

5. **Range Search** (`search/range_searcher.py`)
   - Halfspace range search over hierarchical index
   - Returns candidate key indices

6. **Attention Module** (`attention/hira_attention.py`)
   - HuggingFace-compatible attention layer
   - Integrates all components
   - Drop-in replacement for standard attention

7. **Cache Management** (`cache/hira_cache.py`)
   - Custom HF Cache subclass
   - Maintains hierarchical index alongside KV cache
   - Handles index updates during generation

## Installation

```bash
cd hira
pip install -e .
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from hira.cache import HiraCache
from hira.attention import patch_model_with_hira_attention

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Patch with Hira attention
patch_model_with_hira_attention(model, config={
    "num_levels": 3,
    "branching_factor": 32,
    "threshold_strategy": "top_k",
    "top_k": 256
})

# Use with custom cache
cache = HiraCache()
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, past_key_values=cache, max_new_tokens=50)
```

## Project Structure

```
hira/
├── __init__.py
├── README.md
├── setup.py
├── requirements.txt
├── index/              # Hierarchical index components
│   ├── __init__.py
│   ├── builders.py     # Index building strategies
│   ├── structure.py    # Index data structure
│   ├── updater.py      # Index update mechanisms
│   └── memory_policy.py # Device placement policies
├── search/             # Range search components
│   ├── __init__.py
│   └── range_searcher.py # Halfspace range search
├── attention/          # Attention mechanisms
│   ├── __init__.py
│   ├── hira_attention.py  # Main attention module
│   └── processor.py    # HF attention processor
├── cache/              # Cache management
│   ├── __init__.py
│   └── hira_cache.py  # Custom HF cache
├── kernels/            # Future C++/CUDA optimizations
│   ├── README.md
│   ├── cpp/            # C++ implementations
│   └── cuda/           # CUDA kernels
├── utils/              # Utilities
│   ├── __init__.py
│   ├── threshold.py    # Threshold computation strategies
│   └── metrics.py      # Performance metrics
├── examples/           # Usage examples
│   ├── basic_usage.py
│   ├── patch_llama.py
│   └── benchmark.py
└── tests/              # Unit tests
    ├── __init__.py
    ├── test_index.py
    ├── test_search.py
    └── test_attention.py
```

## Configuration

Key parameters:
- `num_levels`: Number of hirarchy levels (default: 3)
- `branching_factor`: Number of clusters per level (default: 32)
- `threshold_strategy`: "top_k", "percentile", or "fixed"
- `top_k`: Number of keys to retrieve (when using top_k strategy)
- `update_strategy`: "rebuild" or "incremental"
- `memory_policy`: Device placement configuration
