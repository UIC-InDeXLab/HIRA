"""
Helper functions to load collected KV data for use in pruning experiments.
"""

import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


def load_kv_data(
    csv_path: str,
    layer_idx: int = 0,
    max_keys: Optional[int] = None,
    max_queries: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load real KV data from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        layer_idx: Which layer to load (0 to num_layers-1)
        max_keys: Maximum number of keys to load (None = all)
        max_queries: Maximum number of queries to load (None = all)
        
    Returns:
        keys: Tensor of shape [num_keys, head_dim] (flattened across heads)
        queries: Tensor of shape [num_queries, head_dim] (flattened across heads)
    """
    print(f"Loading KV data from {csv_path}")
    print(f"Layer: {layer_idx}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Filter for specific layer
    layer_df = df[df['layer'] == layer_idx]
    
    print(f"Total rows for layer {layer_idx}: {len(layer_df)}")
    
    # Extract keys
    keys_df = layer_df[layer_df['type'] == 'key']
    if max_keys:
        keys_df = keys_df.head(max_keys)
    
    print(f"Loading {len(keys_df)} keys...")
    keys_list = []
    for _, row in keys_df.iterrows():
        vector = np.array([float(x) for x in row['vector'].split(',')])
        keys_list.append(vector)
    
    keys = torch.tensor(np.array(keys_list), dtype=torch.float32)
    
    # Extract queries
    queries_df = layer_df[layer_df['type'] == 'query']
    if max_queries:
        queries_df = queries_df.head(max_queries)
    
    print(f"Loading {len(queries_df)} queries...")
    queries_list = []
    for _, row in queries_df.iterrows():
        vector = np.array([float(x) for x in row['vector'].split(',')])
        queries_list.append(vector)
    
    queries = torch.tensor(np.array(queries_list), dtype=torch.float32)
    
    print(f"Keys shape: {keys.shape}")
    print(f"Queries shape: {queries.shape}")
    
    return keys, queries


def load_kv_data_multihead(
    csv_path: str,
    layer_idx: int = 0,
    head_idx: Optional[int] = None,
    max_positions: Optional[int] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Load KV data preserving multi-head structure.
    
    Args:
        csv_path: Path to the CSV file
        layer_idx: Which layer to load
        head_idx: Specific head to load (None = all heads)
        max_positions: Maximum number of positions/keys to load
        
    Returns:
        keys: Tensor of shape [num_positions, num_heads, head_dim] or [num_positions, head_dim] if head_idx specified
        queries: List of tensors, each of shape [num_heads, head_dim] or [head_dim] if head_idx specified
    """
    df = pd.read_csv(csv_path)
    
    # Filter for layer
    layer_df = df[df['layer'] == layer_idx]
    
    if head_idx is not None:
        layer_df = layer_df[layer_df['head'] == head_idx]
    
    # Get keys
    keys_df = layer_df[layer_df['type'] == 'key'].sort_values(['position', 'head'])
    
    if max_positions:
        max_pos = keys_df['position'].min() + max_positions
        keys_df = keys_df[keys_df['position'] < max_pos]
    
    # Determine dimensions
    num_positions = keys_df['position'].nunique()
    num_heads = keys_df['head'].nunique() if head_idx is None else 1
    
    # Parse first vector to get head_dim
    first_vector = np.array([float(x) for x in keys_df.iloc[0]['vector'].split(',')])
    head_dim = len(first_vector)
    
    # Initialize keys tensor
    if head_idx is None:
        keys = torch.zeros(num_positions, num_heads, head_dim)
    else:
        keys = torch.zeros(num_positions, head_dim)
    
    # Fill keys
    for _, row in keys_df.iterrows():
        pos = row['position']
        head = row['head'] if head_idx is None else 0
        vector = torch.tensor([float(x) for x in row['vector'].split(',')], dtype=torch.float32)
        
        if head_idx is None:
            keys[pos, head, :] = vector
        else:
            keys[pos, :] = vector
    
    # Get queries
    queries_df = layer_df[layer_df['type'] == 'query'].sort_values('decoding_step')
    
    queries = []
    for _, row in queries_df.iterrows():
        vector = torch.tensor([float(x) for x in row['vector'].split(',')], dtype=torch.float32)
        queries.append(vector)
    
    print(f"Loaded keys shape: {keys.shape}")
    print(f"Loaded {len(queries)} queries")
    
    return keys, queries


# Example usage for integration with pruning experiments
if __name__ == "__main__":
    # Example: Load data for layer 0
    csv_file = "kv_data/kv_data_Llama-3.2-1B_needle_4096_20251215_085504.csv"
    
    if Path(csv_file).exists():
        print("=== Example 1: Load flattened keys/queries ===")
        keys, queries = load_kv_data(csv_file, layer_idx=0, max_keys=10000, max_queries=100)
        
        print(f"\n=== Example 2: Load with multi-head structure ===")
        keys_mh, queries_mh = load_kv_data_multihead(csv_file, layer_idx=0, max_positions=1000)
        
        print(f"\n=== Example 3: Single head ===")
        keys_single, queries_single = load_kv_data_multihead(csv_file, layer_idx=0, head_idx=0, max_positions=500)
    else:
        print(f"CSV file not found: {csv_file}")
        print("Run collect_kv_data.py first to generate data.")
