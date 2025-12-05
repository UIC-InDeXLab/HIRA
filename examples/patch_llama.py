"""
Example: Patching Llama model with HiraAttention.

This example shows how to manually patch a Llama model's attention layers
to use hierarchical range-searching attention.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from huggingface_hub import login
from dotenv import load_dotenv
import os
import types
from typing import Optional, Tuple

from hira import HiraCache, HiraAttention
from hira.utils import FixedThresholdStrategy
from hira.search import HalfspaceSearcher


def create_hira_forward_method(hira_attention: HiraAttention):
    """
    Create a custom forward method for Llama attention that uses HiraAttention.
    
    Args:
        hira_attention: HiraAttention instance to use
        
    Returns:
        Custom forward method
    """
    def hira_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple]]:
        """
        Custom forward pass using HiraAttention.
        
        This replaces the standard Llama attention forward pass.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Update KV cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        
        # Use HiraAttention for attention computation
        attn_output, attn_weights = hira_attention.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            past_key_values=past_key_value,
            layer_idx=self.layer_idx,
            scaling=self.scaling,
        )
        
        # Reshape and project output
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, past_key_value
    
    return hira_forward


def patch_llama_with_hira(model, hira_attention: HiraAttention):
    """
    Patch all attention layers in a Llama model to use HiraAttention.
    
    Args:
        model: Llama model to patch
        hira_attention: HiraAttention instance
    """
    print(f"Patching {len(model.model.layers)} layers with HiraAttention...")
    
    custom_forward = create_hira_forward_method(hira_attention)
    
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.forward = types.MethodType(custom_forward, layer.self_attn)
    
    print(f"Successfully patched {len(model.model.layers)} layers!")


def main():
    """Example of patching Llama with HiraAttention."""
    print("=" * 60)
    print("Llama + HiraAttention Example")
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
    
    # Create HiraAttention
    print("\n2. Creating HiraAttention...")
    threshold = 0.0  # Fixed threshold
    hira_attention = HiraAttention(
        threshold_strategy=FixedThresholdStrategy(threshold=threshold),
        range_searcher=HalfspaceSearcher(),
        use_hira_during_prefill=False,  # Use standard attention during prefill
    )
    print(f"   Threshold: {threshold}")
    print(f"   Use during prefill: False")
    
    # Patch the model
    print("\n3. Patching model with HiraAttention...")
    patch_llama_with_hira(model, hira_attention)
    
    # Create HiraCache
    print("\n4. Creating HiraCache...")
    cache = HiraCache(
        num_levels=3,
        branching_factor=32,
        build_index_every_n=64,  # Build index every 64 tokens
    )
    
    # Prepare input
    text = "Explain the theory of relativity in simple terms."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(f"\n5. Input: {text}")
    print(f"   Input length: {inputs.input_ids.shape[1]} tokens")
    
    # Generate with HiraAttention
    print("\n6. Generating with HiraAttention...")
    print("   (Using hierarchical search during decode)")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=100,
            use_cache=True,
            do_sample=True,
            temperature=0.7,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n   Generated text:\n   {generated_text}\n")
    
    # Show cache and index statistics
    print("\n7. Statistics:")
    print("=" * 60)
    cache_info = cache.get_cache_info()
    print(f"   Total tokens: {cache_info['total_tokens']}")
    print(f"   Number of layers: {cache_info['num_layers']}")
    
    # Show details for first layer
    layer_0_info = cache_info['layers'][0]
    print(f"\n   Layer 0 details:")
    print(f"     - Sequence length: {layer_0_info['seq_length']}")
    print(f"     - Has index: {layer_0_info['has_index']}")
    
    if layer_0_info['has_index']:
        index_info = layer_0_info['index_info']
        print(f"     - Index levels: {index_info['num_levels']}")
        print(f"     - Indexed keys: {index_info['num_keys']}")
        mem_usage = index_info['memory_usage']
        print(f"     - Index memory: {mem_usage['total_mb']:.2f} MB")
        
        # Show per-level breakdown
        for level_idx in range(index_info['num_levels']):
            level_mem = mem_usage.get(f"level_{level_idx}_mb", 0)
            print(f"       - Level {level_idx}: {level_mem:.2f} MB")
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
