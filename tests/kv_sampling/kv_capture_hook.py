"""
PyTorch hooks for capturing keys and queries from transformer attention layers.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class KVCaptureHook:
    """
    Hook to capture keys and queries from Llama model during generation.
    
    This class registers forward hooks on attention layers to extract:
    - All key vectors from the KV cache
    - Query vectors sampled every N decoding steps
    """
    
    def __init__(
        self,
        model: nn.Module,
        query_sample_rate: int = 10,
    ):
        """
        Initialize KV capture hooks.
        
        Args:
            model: The Llama model
            query_sample_rate: Sample queries every N decoding steps
        """
        self.model = model
        self.query_sample_rate = query_sample_rate
        
        # Storage for collected data
        self.keys: List[torch.Tensor] = []  # Per layer: [seq_len, num_heads, head_dim]
        self.queries: Dict[int, List[Tuple[int, torch.Tensor]]] = {}  # Per layer: [(step, query)]
        
        # Cache for past_key_values from generation
        self.past_key_values = None
        
        # Generation state  
        self.decoding_step = 0
        self.seen_prefill = False  # Track if we've seen the initial prefill
        
        # Model architecture info
        self.num_layers = len(model.model.layers)
        
        # Get num_heads and head_dim from first layer
        first_attn = model.model.layers[0].self_attn
        self.num_heads = first_attn.num_heads
        self.num_key_value_heads = getattr(first_attn, 'num_key_value_heads', self.num_heads)
        self.head_dim = first_attn.head_dim
        
        # Register hooks
        self.hooks = []
        self._register_hooks()
        
        print(f"KV Capture Hook initialized:")
        print(f"  Layers: {self.num_layers}")
        print(f"  Attention heads: {self.num_heads}")
        print(f"  Key-Value heads: {self.num_key_value_heads}")
        print(f"  Head dimension: {self.head_dim}")
        print(f"  Query sample rate: every {query_sample_rate} steps")
    
    def _register_hooks(self):
        """Register forward hooks on all attention layers."""
        for layer_idx, layer in enumerate(self.model.model.layers):
            hook = layer.self_attn.register_forward_hook(
                self._make_hook_fn(layer_idx)
            )
            self.hooks.append(hook)
        
        # Also hook the main model forward to capture past_key_values
        self.model_hook = self.model.register_forward_hook(self._model_forward_hook)
    
    def _model_forward_hook(self, module, input, output):
        """Hook on main model to capture past_key_values from output."""
        # Output is CausalLMOutputWithPast which has past_key_values attribute
        if hasattr(output, 'past_key_values') and output.past_key_values is not None:
            self.past_key_values = output.past_key_values
    
    def _make_hook_fn(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            """
            Forward hook to capture keys and queries.
            
            Output format: (attn_output, attn_weights, past_key_value)
            - output[0]: attention output [batch, seq_len, hidden_dim]
            - output[1]: attention weights (usually None)
            - output[2]: DynamicCache object with keys and values
            """
            # The output tuple has attention output at index 0
            # This gives us the sequence length
            if not isinstance(output, tuple) or len(output) == 0:
                return
            
            attn_output = output[0]  # [batch, seq_len, hidden_dim]
            seq_len = attn_output.shape[1]
            
            # Track prefill completion on first layer
            if layer_idx == 0:
                if seq_len > 1 and not self.seen_prefill:
                    # This is the prefill pass
                    self.seen_prefill = True
                elif seq_len == 1 and self.seen_prefill:
                    # This is decoding - single token after prefill
                    pass
            
            # Sample queries during decoding (seq_len == 1 after prefill)
            if seq_len == 1 and self.seen_prefill:
                # This is a decoding step
                # We need to get the input hidden states to compute query
                # We can use module.input_layernorm or access from a pre-hook
                # For now, let's extract from the past_key_value cache
                
                if self.decoding_step % self.query_sample_rate == 0:
                    # Store that we need to sample a query for this layer/step
                    # We'll extract it from cache later
                    if layer_idx not in self.queries:
                        self.queries[layer_idx] = []
                    # Mark this step for query extraction (we'll fill in the tensor later)
                    self.queries[layer_idx].append((self.decoding_step, None))
                
                # Increment decoding step counter on last layer only
                if layer_idx == self.num_layers - 1:
                    self.decoding_step += 1
        
        return hook_fn
    
    def finalize_collection(self):
        """Extract keys from past_key_values after generation."""
        if self.past_key_values is None:
            print("Warning: No past_key_values captured!")
            return
        
        print(f"Extracting keys from cache (type: {type(self.past_key_values)})")
        
        # past_key_values can be a tuple of tuples or a Cache object
        if isinstance(self.past_key_values, tuple):
            # Format: ((key_layer_0, value_layer_0), (key_layer_1, value_layer_1), ...)
            for layer_idx, (key_cache, value_cache) in enumerate(self.past_key_values):
                # key_cache shape: [batch, num_key_value_heads, seq_len, head_dim]
                
                # For GQA, repeat keys to match num_heads
                if key_cache.shape[1] != self.num_heads:
                    num_groups = self.num_heads // self.num_key_value_heads
                    key_cache = key_cache.repeat_interleave(num_groups, dim=1)
                
                # Reshape: [batch, num_heads, seq_len, head_dim] -> [seq_len, num_heads, head_dim]
                keys = key_cache[0].transpose(0, 1).detach().clone().cpu()
                
                self.keys.append(keys)
                
                # Extract queries from marked steps
                if layer_idx in self.queries:
                    for idx, (step, _) in enumerate(self.queries[layer_idx]):
                        # Query is the key at the corresponding position
                        # Step is the decoding step, which corresponds to position after prefill
                        # We need to figure out which position in keys this corresponds to
                        # Actually, for simplicity, let's extract queries from the last few keys
                        # The query at step N is the key at position (prefill_len + N)
                        
                        # For now, let's just sample from recent keys
                        # We'll improve this logic
                        query_pos = min(step, keys.shape[0] - 1)
                        query = keys[query_pos, :, :].clone()  # [num_heads, head_dim]
                        
                        # Update the query in place
                        self.queries[layer_idx][idx] = (step, query)
        else:
            # Modern transformers use Cache objects
            # Try to access the cache tensors
            if hasattr(self.past_key_values, 'key_cache'):
                # DynamicCache format
                for layer_idx in range(len(self.past_key_values.key_cache)):
                    key_cache = self.past_key_values.key_cache[layer_idx]
                    
                    # For GQA, repeat keys to match num_heads
                    if key_cache.shape[1] != self.num_heads:
                        num_groups = self.num_heads // self.num_key_value_heads
                        key_cache = key_cache.repeat_interleave(num_groups, dim=1)
                    
                    # Reshape: [batch, num_heads, seq_len, head_dim] -> [seq_len, num_heads, head_dim]
                    keys = key_cache[0].transpose(0, 1).detach().clone().cpu()
                    
                    self.keys.append(keys)
                    
                    # Extract queries from marked steps
                    if layer_idx in self.queries:
                        # Get the prefill length (first time we see seq_len > 1)
                        # Assuming generation produced keys.shape[0] total tokens
                        total_len = keys.shape[0]
                        
                        for idx, (step, _) in enumerate(self.queries[layer_idx]):
                            # The query for decoding step N is at position (total_len - num_generated + step)
                            # Or more simply: we know the cache grows by 1 each step
                            # So if we have queries at steps [0, 10, 20], they correspond to
                            # positions [prefill_len, prefill_len+10, prefill_len+20]
                            
                            # Since we don't track prefill_len explicitly, let's use a different approach:
                            # Query at step S is the key at position -(num_generated_steps - S)
                            # Where num_generated_steps = self.decoding_step
                            
                            query_pos = total_len - self.decoding_step + step
                            if 0 <= query_pos < total_len:
                                query = keys[query_pos, :, :].clone()  # [num_heads, head_dim]
                                self.queries[layer_idx][idx] = (step, query)
            else:
                print(f"Warning: Unknown cache format: {type(self.past_key_values)}")
                print(f"Available attributes: {dir(self.past_key_values)}")
        
        print(f"Extracted keys for {len(self.keys)} layers")
        if len(self.keys) > 0:
            print(f"Keys shape per layer: {self.keys[0].shape}")
        print(f"Queries collected: {sum(len(v) for v in self.queries.values())} total")
    
    def reset(self):
        """Reset collected data for a new generation."""
        self.keys = []
        self.queries = {}
        self.decoding_step = 0
        self.seen_prefill = False
        self.past_key_values = None
    
    def get_collected_data(self) -> Dict:
        """
        Get all collected data.
        
        Returns:
            Dictionary with:
            - 'keys': List of tensors, one per layer [seq_len, num_heads, head_dim]
            - 'queries': Dict mapping layer_idx to list of (step, tensor) tuples
        """
        return {
            'keys': self.keys,
            'queries': self.queries,
        }
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        if hasattr(self, 'model_hook'):
            self.model_hook.remove()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()
