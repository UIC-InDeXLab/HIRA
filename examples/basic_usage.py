"""
Basic usage example for Hira attention.

This example demonstrates how to use HiraCache and HiraAttention
with a HuggingFace model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv
import os

from hira import HiraCache, HiraAttention
from hira.utils import FixedThresholdStrategy
from hira.search import HalfspaceRangeSearcher


def main():
    """Basic usage example."""
    print("=" * 60)
    print("Hira Basic Usage Example")
    print("=" * 60)

    # Load environment
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Load model and tokenizer
    print("\n1. Loading model...")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create HiraCache
    print("\n2. Creating HiraCache...")
    cache = HiraCache(
        num_levels=3,
        branching_factor=32,
        build_index_every_n=128,  # Build index every 128 tokens
    )
    print(f"   Cache configured with {cache.num_levels} levels")

    # Prepare input
    text = "What is the capital of France?"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(f"\n3. Input: {text}")
    print(f"   Input length: {inputs.input_ids.shape[1]} tokens")

    # Generate with HiraCache
    print("\n4. Generating with HiraCache...")
    print("   Note: Using standard attention (not yet patched with HiraAttention)")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=50,
            use_cache=True,
            do_sample=True,
            temperature=0.7,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n   Generated text:\n   {generated_text}\n")

    # Show cache statistics
    print("\n5. Cache Statistics:")
    print("=" * 60)
    cache_info = cache.get_cache_info()
    print(f"   Total tokens generated: {cache_info['total_tokens']}")
    print(f"   Number of layers: {cache_info['num_layers']}")

    for layer_info in cache_info["layers"][:3]:  # Show first 3 layers
        layer_idx = layer_info["layer_idx"]
        seq_len = layer_info["seq_length"]
        has_index = layer_info["has_index"]
        print(f"\n   Layer {layer_idx}:")
        print(f"     - Sequence length: {seq_len}")
        print(f"     - Has index: {has_index}")

        if has_index:
            index_info = layer_info["index_info"]
            print(f"     - Index levels: {index_info['num_levels']}")
            print(f"     - Indexed keys: {index_info['num_keys']}")
            mem_usage = index_info["memory_usage"]
            print(f"     - Index memory: {mem_usage['total_mb']:.2f} MB")

    print("\n" + "=" * 60)
    print("Done!")
    print("\nNote: This example uses HiraCache but standard attention.")
    print("See patch_llama.py for using HiraAttention with hierarchical search.")


if __name__ == "__main__":
    main()
