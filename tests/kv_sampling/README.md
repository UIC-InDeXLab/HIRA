# KV Cache Data Collection

This directory contains scripts for collecting real KV cache data from Llama 3.2 for use in pruning experiments.

## Overview

The collection pipeline:
1. Loads Llama 3.2 model
2. Runs inference on RULER benchmark tasks (long-context synthetic tasks)
3. Captures all keys and sampled queries during decoding
4. Saves data to a single CSV file

## Files

- `collect_kv_data.py` - Main script for data collection
- `kv_capture_hook.py` - PyTorch hooks to capture keys/queries from attention layers
- `ruler_loader.py` - RULER benchmark task generator
- `README.md` - This file

## Installation

```bash
# Activate your virtual environment
source /home/mohsen/venv/bin/activate

# Install required packages
pip install transformers torch pandas numpy tqdm
```

## Quick Start

```bash
cd /home/mohsen/kvcache/hira/tests/kv_sampling
source ~/venv/bin/activate

# Collect data with default settings
python collect_kv_data.py --model meta-llama/Llama-3.2-1B --num-samples 2 --num-generate 50

# Or with custom parameters
python collect_kv_data.py \
    --model meta-llama/Llama-3.2-1B \
    --ruler-task passkey \
    --num-samples 5 \
    --context-length 4096 \
    --num-generate 100 \
    --query-sample-rate 10 \
    --device cuda
```

## What Gets Collected

✅ **Keys**: All key vectors from every layer during generation
- Shape per layer: `[seq_len, num_heads, head_dim]`  
- Includes both prefill and decoding keys

✅ **Queries**: Sampled query vectors during decoding
- Sampled every N steps (configurable via `--query-sample-rate`)
- Shape per query: `[num_heads, head_dim]`
- Only from decoding phase (not prefill)

## Output Files

After running, you'll get:
- `kv_data_*.csv` - Main data file with all keys and queries
- `metadata_*.txt` - Collection parameters and statistics

Example output:
```
✓ Saved 3560960 rows
  File size: 1.5 GB
  Total keys: 3,555,840
  Total queries: 5,120
  Layers: 16
  Heads per layer: 32
```

### Parameters

- `--model`: HuggingFace model name (default: `meta-llama/Llama-3.2-1B`)
  - Options: `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B`, etc.

- `--ruler-task`: RULER benchmark task (default: `needle`)
  - `needle`: Needle in a haystack - find hidden fact
  - `variable_tracking`: Track variable assignments
  - `passkey`: Retrieve hidden passkey
  - `number_retrieval`: Find specific numbers
  - `all`: Mix of all tasks

- `--num-samples`: Number of samples to collect (default: 10)

- `--context-length`: Target context length in tokens (default: 4096)
  - Adjust based on your GPU memory

- `--num-generate`: Tokens to generate per sample (default: 100)

- `--query-sample-rate`: Sample queries every N steps (default: 10)
  - Lower N = more queries, larger file

- `--output-dir`: Output directory (default: `./kv_data`)

- `--device`: Device to use (default: `cuda` if available, else `cpu`)

## Output Format

### CSV File

Single CSV with columns:
- `sample_idx`: Sample index (0 to num_samples-1)
- `layer`: Layer index (0 to num_layers-1)
- `head`: Attention head index (0 to num_heads-1)
- `position`: Token position (for keys only, -1 for queries)
- `type`: 'key' or 'query'
- `decoding_step`: Decoding step number (for queries only, -1 for keys)
- `vector`: Comma-separated vector values

### Metadata File

Text file with collection parameters and statistics.

## Example Workflow

1. **Collect data:**
```bash
python collect_kv_data.py \
    --ruler-task passkey \
    --num-samples 5 \
    --context-length 8192 \
    --query-sample-rate 10
```

2. **Load data in pruning experiments:**
```python
import pandas as pd
import torch

# Load CSV
df = pd.read_csv('kv_data/kv_data_Llama-3.2-1B_passkey_8192_20251215_120000.csv')

# Extract keys for a specific layer
layer_0_keys = df[(df['layer'] == 0) & (df['type'] == 'key')]

# Convert vectors
keys_tensor = torch.tensor([
    [float(x) for x in row['vector'].split(',')]
    for _, row in layer_0_keys.iterrows()
])

# Reshape: [num_keys, num_heads, head_dim]
num_heads = layer_0_keys['head'].max() + 1
head_dim = len(layer_0_keys.iloc[0]['vector'].split(','))
keys_tensor = keys_tensor.reshape(-1, num_heads, head_dim)
```

## Memory Considerations

- **GPU Memory**: Llama-3.2-1B requires ~4GB, 3B requires ~12GB
- **CSV Size**: 
  - 4K context, 100 generated tokens, N=10: ~50-100MB per sample
  - 8K context, 100 generated tokens, N=10: ~100-200MB per sample

Reduce file size by:
- Increasing `--query-sample-rate` (e.g., N=50 or N=100)
- Reducing `--num-generate`
- Collecting fewer `--num-samples`

## Troubleshooting

**Out of GPU memory:**
- Use smaller model: `meta-llama/Llama-3.2-1B`
- Reduce `--context-length`
- Use `--device cpu` (slower but no GPU needed)

**File too large:**
- Increase `--query-sample-rate`
- Reduce `--num-samples` or `--num-generate`

**Model not found:**
- Ensure you have HuggingFace access to Llama models
- Login: `huggingface-cli login`

## Loading Data in Your Experiments

Use the provided `load_kv_data.py` helper:

```python
from load_kv_data import load_kv_data, load_kv_data_multihead

# Option 1: Load flattened (for compatibility with existing experiments)
keys, queries = load_kv_data(
    'kv_data/kv_data_*.csv',
    layer_idx=0,
    max_keys=100000,
    max_queries=1000
)
# keys: [num_keys, head_dim]
# queries: [num_queries, head_dim]

# Option 2: Load with multi-head structure preserved  
keys, queries = load_kv_data_multihead(
    'kv_data/kv_data_*.csv',
    layer_idx=0,
    max_positions=10000
)
# keys: [num_positions, num_heads, head_dim]
# queries: List of [num_heads, head_dim]

# Option 3: Single attention head
keys, queries = load_kv_data_multihead(
    'kv_data/kv_data_*.csv',
    layer_idx=0,
    head_idx=0,
    max_positions=5000
)
# keys: [num_positions, head_dim]
# queries: List of [head_dim]
```

## Integration with Pruning Experiments

Replace synthetic data generation in your pruning experiments:

```python
# In your experiment.py or test scripts
import sys
sys.path.append('/home/mohsen/kvcache/hira/tests/kv_sampling')
from load_kv_data import load_kv_data

# Instead of generating synthetic data:
# keys = torch.randn(NUM_KEYS, DIMENSION)

# Use real data:
keys, queries = load_kv_data(
    '/home/mohsen/kvcache/hira/tests/kv_sampling/kv_data/kv_data_*.csv',
    layer_idx=0,
    max_keys=100000
)

# Now run your pruning index tests with real KV distributions!
# from tests.pruning.kmeans_ball_index import KMeansBallIndex
# index = KMeansBallIndex(num_clusters=1000)
# index.build(keys)
# ...
```
