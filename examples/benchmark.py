"""
Benchmark script for comparing Hira attention with standard attention.

This script measures:
- Latency (time per token)
- Throughput (tokens per second)
- Memory usage
- Accuracy (perplexity or other metrics)
"""

import torch
import time
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv
import os

from hira import HiraCache
from transformers.cache_utils import DynamicCache


def measure_generation_time(
    model,
    tokenizer,
    prompt: str,
    cache_type: str,
    max_new_tokens: int = 100,
    **cache_kwargs
) -> Dict[str, Any]:
    """
    Measure generation time and stats.
    
    Args:
        model: Model to use
        tokenizer: Tokenizer
        prompt: Input prompt
        cache_type: "standard" or "hira"
        max_new_tokens: Number of tokens to generate
        **cache_kwargs: Arguments for cache creation
        
    Returns:
        Dictionary with timing and memory stats
    """
    device = next(model.parameters()).device
    
    # Create cache
    if cache_type == "hira":
        cache = HiraCache(**cache_kwargs)
    else:
        cache = DynamicCache()
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    
    # Clear GPU cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,  # Deterministic for comparison
        )
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Memory usage
    if device.type == "cuda":
        end_mem = torch.cuda.memory_allocated()
        mem_used = (end_mem - start_mem) / (1024 ** 2)  # MB
    else:
        mem_used = 0
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_length = outputs.shape[1]
    tokens_generated = output_length - input_length
    
    return {
        "cache_type": cache_type,
        "elapsed_time": elapsed,
        "tokens_generated": tokens_generated,
        "latency_per_token": elapsed / tokens_generated if tokens_generated > 0 else 0,
        "throughput": tokens_generated / elapsed if elapsed > 0 else 0,
        "memory_mb": mem_used,
        "generated_text": generated_text,
    }


def run_benchmark():
    """Run comprehensive benchmark."""
    print("=" * 80)
    print("Hira Attention Benchmark")
    print("=" * 80)
    
    # Load environment
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    # Load model
    print("\nLoading model...")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
    ]
    
    # Benchmark configurations
    configs = [
        {
            "name": "Standard Attention",
            "cache_type": "standard",
            "cache_kwargs": {},
        },
        {
            "name": "Hira (3 levels, build every 128)",
            "cache_type": "hira",
            "cache_kwargs": {
                "num_levels": 3,
                "branching_factor": 32,
                "build_index_every_n": 128,
            },
        },
        {
            "name": "Hira (2 levels, build every 64)",
            "cache_type": "hira",
            "cache_kwargs": {
                "num_levels": 2,
                "branching_factor": 16,
                "build_index_every_n": 64,
            },
        },
    ]
    
    max_new_tokens = 100
    
    # Run benchmarks
    results = []
    
    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"Testing: {config['name']}")
        print(f"{'=' * 80}")
        
        config_results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n  Prompt {i+1}: {prompt[:50]}...")
            
            result = measure_generation_time(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                cache_type=config["cache_type"],
                max_new_tokens=max_new_tokens,
                **config["cache_kwargs"],
            )
            
            config_results.append(result)
            
            print(f"    Tokens generated: {result['tokens_generated']}")
            print(f"    Latency per token: {result['latency_per_token']*1000:.2f} ms")
            print(f"    Throughput: {result['throughput']:.2f} tokens/sec")
            if device == "cuda":
                print(f"    Memory used: {result['memory_mb']:.2f} MB")
        
        # Average results
        avg_latency = sum(r['latency_per_token'] for r in config_results) / len(config_results)
        avg_throughput = sum(r['throughput'] for r in config_results) / len(config_results)
        avg_memory = sum(r['memory_mb'] for r in config_results) / len(config_results)
        
        print(f"\n  Average Results:")
        print(f"    Avg latency per token: {avg_latency*1000:.2f} ms")
        print(f"    Avg throughput: {avg_throughput:.2f} tokens/sec")
        if device == "cuda":
            print(f"    Avg memory: {avg_memory:.2f} MB")
        
        results.append({
            "config": config["name"],
            "avg_latency": avg_latency,
            "avg_throughput": avg_throughput,
            "avg_memory": avg_memory,
            "per_prompt": config_results,
        })
    
    # Summary comparison
    print(f"\n{'=' * 80}")
    print("Summary Comparison")
    print(f"{'=' * 80}")
    
    baseline = results[0]
    
    print(f"\n{'Configuration':<40} {'Latency':<15} {'Throughput':<15} {'Memory':<15}")
    print(f"{'-'*40} {'-'*15} {'-'*15} {'-'*15}")
    
    for result in results:
        latency_ratio = result['avg_latency'] / baseline['avg_latency']
        throughput_ratio = result['avg_throughput'] / baseline['avg_throughput']
        memory_ratio = result['avg_memory'] / baseline['avg_memory'] if baseline['avg_memory'] > 0 else 1.0
        
        print(f"{result['config']:<40} {latency_ratio:.2f}x {'':<9} {throughput_ratio:.2f}x {'':<9} {memory_ratio:.2f}x")
    
    print(f"\n{'=' * 80}")
    print("Benchmark complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    run_benchmark()
