"""
KV Cache Data Collection Script

This script runs Llama 3.2 on RULER benchmark tasks and collects:
- All keys from the KV cache during decoding
- Queries sampled every N steps

The data is saved to a single CSV file for use in pruning experiments.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse

# Add hira to path
sys.path.append('/home/mohsen/kvcache/hira')

from kv_capture_hook import KVCaptureHook
from ruler_loader import RulerDataLoader


class KVDataCollector:
    """Collects KV cache data during model inference."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        query_sample_rate: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 4096,
    ):
        """
        Initialize KV data collector.
        
        Args:
            model_name: HuggingFace model name
            query_sample_rate: Sample queries every N decoding steps
            device: Device to run on
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.query_sample_rate = query_sample_rate
        self.device = device
        self.max_length = max_length
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        self.model.eval()
        
        # Initialize KV capture hooks
        self.kv_hook = KVCaptureHook(
            model=self.model,
            query_sample_rate=query_sample_rate
        )
        
        print(f"Model loaded on {device}")
        print(f"Number of layers: {self.kv_hook.num_layers}")
        print(f"Number of attention heads: {self.kv_hook.num_heads}")
        print(f"Head dimension: {self.kv_hook.head_dim}")
    
    def collect_from_text(
        self,
        input_text: str,
        num_tokens_to_generate: int = 100,
    ) -> Dict:
        """
        Run generation and collect KV data.
        
        Args:
            input_text: Input prompt
            num_tokens_to_generate: Number of tokens to generate
            
        Returns:
            Dictionary with collected data
        """
        print(f"\nProcessing input ({len(input_text)} chars)...")
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - num_tokens_to_generate,
        ).to(self.device)
        
        input_length = inputs.input_ids.shape[1]
        print(f"Input tokens: {input_length}")
        
        # Clear previous data
        self.kv_hook.reset()
        
        # Generate with hooks active
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=num_tokens_to_generate,
                do_sample=False,  # Deterministic
                use_cache=True,
            )
        
        generated_length = outputs.shape[1] - input_length
        print(f"Generated tokens: {generated_length}")
        
        # Extract keys from cache
        self.kv_hook.finalize_collection()
        
        # Get collected data
        data = self.kv_hook.get_collected_data()
        
        return {
            'input_text': input_text,
            'input_length': input_length,
            'generated_length': generated_length,
            'data': data,
        }
    
    def save_to_csv(
        self,
        collected_data: List[Dict],
        output_path: str,
    ):
        """
        Save collected data to a single CSV file.
        
        Args:
            collected_data: List of collection results
            output_path: Path to save CSV
        """
        print(f"\nSaving data to {output_path}...")
        
        rows = []
        
        for sample_idx, sample in enumerate(tqdm(collected_data, desc="Processing samples")):
            data = sample['data']
            
            # Process each layer
            for layer_idx in range(len(data['keys'])):
                keys = data['keys'][layer_idx]  # [seq_len, num_heads, head_dim]
                
                # Add all keys
                for pos_idx in range(keys.shape[0]):
                    for head_idx in range(keys.shape[1]):
                        key_vector = keys[pos_idx, head_idx].cpu().numpy()
                        
                        row = {
                            'sample_idx': sample_idx,
                            'layer': layer_idx,
                            'head': head_idx,
                            'position': pos_idx,
                            'type': 'key',
                            'decoding_step': -1,  # N/A for keys
                            'vector': ','.join(map(str, key_vector)),
                        }
                        rows.append(row)
                
                # Add sampled queries
                if layer_idx in data['queries']:
                    queries = data['queries'][layer_idx]  # List of (step, tensor)
                    
                    for decoding_step, query_tensor in queries:
                        # query_tensor: [num_heads, head_dim]
                        for head_idx in range(query_tensor.shape[0]):
                            query_vector = query_tensor[head_idx].cpu().numpy()
                            
                            row = {
                                'sample_idx': sample_idx,
                                'layer': layer_idx,
                                'head': head_idx,
                                'position': -1,  # N/A for queries
                                'type': 'query',
                                'decoding_step': decoding_step,
                                'vector': ','.join(map(str, query_vector)),
                            }
                            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Saved {len(rows)} rows to {output_path}")
        print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        # Print summary statistics
        print("\nData Summary:")
        print(f"  Total samples: {len(collected_data)}")
        if len(df) > 0:
            print(f"  Total keys: {len(df[df['type'] == 'key'])}")
            print(f"  Total queries: {len(df[df['type'] == 'query'])}")
            print(f"  Layers: {df['layer'].nunique()}")
            print(f"  Heads per layer: {df['head'].nunique()}")
        else:
            print(f"  WARNING: No data was collected!")


def main():
    parser = argparse.ArgumentParser(description='Collect KV cache data from Llama 3.2')
    
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Llama-3.2-1B',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--ruler-task',
        type=str,
        default='needle',
        choices=['needle', 'variable_tracking', 'passkey', 'number_retrieval', 'all'],
        help='RULER task to use'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to collect'
    )
    parser.add_argument(
        '--context-length',
        type=int,
        default=4096,
        help='Context length for RULER tasks'
    )
    parser.add_argument(
        '--num-generate',
        type=int,
        default=100,
        help='Number of tokens to generate per sample'
    )
    parser.add_argument(
        '--query-sample-rate',
        type=int,
        default=10,
        help='Sample queries every N decoding steps'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./kv_data',
        help='Output directory for CSV files'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize collector
    collector = KVDataCollector(
        model_name=args.model,
        query_sample_rate=args.query_sample_rate,
        device=args.device,
        max_length=args.context_length,
    )
    
    # Load RULER data
    print(f"\nLoading RULER benchmark: {args.ruler_task}")
    ruler_loader = RulerDataLoader(
        task=args.ruler_task,
        context_length=args.context_length,
    )
    
    samples = ruler_loader.get_samples(num_samples=args.num_samples)
    print(f"Loaded {len(samples)} samples")
    
    # Collect data from each sample
    collected_data = []
    
    for idx, sample_text in enumerate(tqdm(samples, desc="Collecting data")):
        print(f"\n{'='*60}")
        print(f"Sample {idx + 1}/{len(samples)}")
        print(f"{'='*60}")
        
        result = collector.collect_from_text(
            input_text=sample_text,
            num_tokens_to_generate=args.num_generate,
        )
        
        collected_data.append(result)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_short = args.model.split('/')[-1]
    output_file = output_dir / f"kv_data_{model_short}_{args.ruler_task}_{args.context_length}_{timestamp}.csv"
    
    collector.save_to_csv(collected_data, str(output_file))
    
    # Save metadata
    metadata_file = output_dir / f"metadata_{model_short}_{args.ruler_task}_{args.context_length}_{timestamp}.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"KV Cache Data Collection Metadata\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"RULER Task: {args.ruler_task}\n")
        f.write(f"Context Length: {args.context_length}\n")
        f.write(f"Number of Samples: {args.num_samples}\n")
        f.write(f"Tokens Generated per Sample: {args.num_generate}\n")
        f.write(f"Query Sample Rate: Every {args.query_sample_rate} steps\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Output File: {output_file.name}\n")
    
    print(f"\n✓ Metadata saved to {metadata_file}")
    print("\n" + "="*60)
    print("Collection complete!")
    print("="*60)


if __name__ == "__main__":
    main()
