# """
# HiraAttention: Hierarchical range-searching attention mechanism.

# This module provides the core attention implementation that uses hierarchical
# indexes for efficient key selection.
# """

# from typing import Optional, Tuple, Callable
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers.cache_utils import Cache

# from ..cache import HiraCache
# from ..search import HalfspaceSearcher
# from ..utils import FixedThresholdStrategy, ThresholdStrategy


# class HiraAttention:
#     """
#     Hierarchical range-searching attention mechanism.
    
#     This class provides attention computation that:
#     1. For each query, computes a score threshold
#     2. Uses hierarchical index to find high-score keys
#     3. Computes sparse attention only over selected keys
    
#     This is designed to be used as a drop-in replacement for standard
#     attention in HuggingFace models.
    
#     Args:
#         threshold_strategy: Strategy for computing score thresholds
#         range_searcher: RangeSearcher for finding qualifying keys
#         scaling: Attention scaling factor (default: 1/sqrt(head_dim))
#         use_hira_during_prefill: Whether to use hierarchical search during prefill
#                                    (default: False, use full attention during prefill)
#     """
    
#     def __init__(
#         self,
#         threshold_strategy: Optional[ThresholdStrategy] = None,
#         range_searcher: Optional[HalfspaceSearcher] = None,
#         scaling: Optional[float] = None,
#         use_hira_during_prefill: bool = False,
#     ):
#         self.threshold_strategy = threshold_strategy or FixedThresholdStrategy(threshold=0.0)
#         self.range_searcher = range_searcher or HalfspaceSearcher()
#         self.scaling = scaling
#         self.use_hira_during_prefill = use_hira_during_prefill
    
#     def forward(
#         self,
#         query_states: torch.Tensor,
#         key_states: torch.Tensor,
#         value_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         past_key_values: Optional[Cache] = None,
#         layer_idx: int = 0,
#         **kwargs
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         """
#         Compute hierarchical range-searching attention.
        
#         Args:
#             query_states: Query tensor [batch_size, num_heads, seq_len, head_dim]
#             key_states: Key tensor [batch_size, num_heads, seq_len, head_dim]
#             value_states: Value tensor [batch_size, num_heads, seq_len, head_dim]
#             attention_mask: Optional attention mask
#             past_key_values: Cache object (should be HiraCache for hierarchical search)
#             layer_idx: Layer index
#             **kwargs: Additional arguments
            
#         Returns:
#             Tuple of (attention_output, attention_weights)
#             Note: attention_weights will be None for sparse attention
#         """
#         batch_size, num_heads, q_len, head_dim = query_states.shape
        
#         # Determine scaling
#         if self.scaling is None:
#             scaling = 1.0 / (head_dim ** 0.5)
#         else:
#             scaling = self.scaling
        
#         # Check if we should use hierarchical search
#         use_hira = self._should_use_hira(
#             q_len=q_len,
#             past_key_values=past_key_values,
#             layer_idx=layer_idx,
#         )
        
#         if use_hira:
#             # Hierarchical sparse attention
#             attn_output = self._hierarchical_attention(
#                 query_states=query_states,
#                 key_states=key_states,
#                 value_states=value_states,
#                 past_key_values=past_key_values,
#                 layer_idx=layer_idx,
#                 scaling=scaling,
#                 attention_mask=attention_mask,
#             )
#             attn_weights = None  # Sparse attention doesn't return full weights
#         else:
#             # Standard full attention (fallback)
#             attn_output, attn_weights = self._standard_attention(
#                 query_states=query_states,
#                 key_states=key_states,
#                 value_states=value_states,
#                 scaling=scaling,
#                 attention_mask=attention_mask,
#             )
        
#         return attn_output, attn_weights
    
#     def _should_use_hira(
#         self,
#         q_len: int,
#         past_key_values: Optional[Cache],
#         layer_idx: int,
#     ) -> bool:
#         """
#         Determine whether to use hierarchical search for this forward pass.
        
#         Args:
#             q_len: Query sequence length
#             past_key_values: Cache object
#             layer_idx: Layer index
            
#         Returns:
#             True if hierarchical search should be used
#         """
#         # Don't use hierarchical search if cache is not HiraCache
#         if not isinstance(past_key_values, HiraCache):
#             return False
        
#         # During prefill (q_len > 1), optionally use full attention
#         if q_len > 1 and not self.use_hira_during_prefill:
#             return False
        
#         # Check if index exists for this layer
#         index = past_key_values.get_index(layer_idx)
#         if index is None:
#             return False
        
#         return True
    
#     def _hierarchical_attention(
#         self,
#         query_states: torch.Tensor,
#         key_states: torch.Tensor,
#         value_states: torch.Tensor,
#         past_key_values: HiraCache,
#         layer_idx: int,
#         scaling: float,
#         attention_mask: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Compute sparse attention using hierarchical range search.
        
#         Args:
#             query_states: [batch_size, num_heads, q_len, head_dim]
#             key_states: [batch_size, num_heads, kv_len, head_dim]
#             value_states: [batch_size, num_heads, kv_len, head_dim]
#             past_key_values: HiraCache instance
#             layer_idx: Layer index
#             scaling: Attention scaling factor
#             attention_mask: Optional mask
            
#         Returns:
#             Attention output [batch_size, num_heads, q_len, head_dim]
#         """
#         batch_size, num_heads, q_len, head_dim = query_states.shape
#         kv_len = key_states.shape[2]
        
#         # Get hierarchical index and flattened keys
#         index = past_key_values.get_index(layer_idx)
#         keys_flat = past_key_values.get_keys_flat(layer_idx)  # [batch*heads*kv_len, head_dim]
        
#         # Reshape keys and values to match flattened index
#         # keys: [batch, heads, kv_len, head_dim] -> [batch*heads, kv_len, head_dim]
#         keys_grouped = key_states.reshape(batch_size * num_heads, kv_len, head_dim)
#         values_grouped = value_states.reshape(batch_size * num_heads, kv_len, head_dim)
        
#         # Process each query position
#         output_list = []
        
#         for q_idx in range(q_len):
#             # Get query for this position: [batch, heads, head_dim]
#             query = query_states[:, :, q_idx, :]  # [batch, heads, head_dim]
#             query_grouped = query.reshape(batch_size * num_heads, head_dim)
            
#             # Compute attention for each batch*head
#             batch_head_outputs = []
            
#             for bh_idx in range(batch_size * num_heads):
#                 q = query_grouped[bh_idx]  # [head_dim]
                
#                 # Compute threshold for this query
#                 # For fixed threshold, this is constant, but interface allows flexibility
#                 threshold = self.threshold_strategy.compute_threshold(q, keys_flat)
                
#                 # Perform hierarchical range search to find qualifying keys
#                 # Note: We search over all keys in the cache (batch*heads*kv_len)
#                 # and then filter to the relevant head
#                 qualifying_indices = self.range_searcher.search(
#                     query=q,
#                     threshold=threshold,
#                     index=index,
#                     keys=keys_flat,
#                 )
                
#                 # Filter indices to only include keys from this batch*head
#                 # keys_flat has shape [batch*heads*kv_len, head_dim]
#                 # We need indices in range [bh_idx * kv_len, (bh_idx+1) * kv_len)
#                 start_idx = bh_idx * kv_len
#                 end_idx = (bh_idx + 1) * kv_len
                
#                 mask = (qualifying_indices >= start_idx) & (qualifying_indices < end_idx)
#                 local_indices = qualifying_indices[mask] - start_idx
                
#                 # Compute sparse attention over selected keys
#                 if len(local_indices) == 0:
#                     # No keys selected, return zero vector (or handle differently)
#                     # TODO: Consider fallback strategies when no keys are selected
#                     attn_out = torch.zeros(head_dim, dtype=query.dtype, device=query.device)
#                 else:
#                     # Get selected keys and values
#                     selected_keys = keys_grouped[bh_idx, local_indices, :]  # [num_selected, head_dim]
#                     selected_values = values_grouped[bh_idx, local_indices, :]  # [num_selected, head_dim]
                    
#                     # Compute attention scores
#                     scores = torch.matmul(selected_keys, q) * scaling  # [num_selected]
                    
#                     # Apply softmax (only over selected keys)
#                     attn_weights = F.softmax(scores, dim=0)  # [num_selected]
                    
#                     # Weighted sum of values
#                     attn_out = torch.matmul(attn_weights, selected_values)  # [head_dim]
                
#                 batch_head_outputs.append(attn_out)
            
#             # Stack outputs for this query position
#             # [batch*heads, head_dim] -> [batch, heads, head_dim]
#             output_tensor = torch.stack(batch_head_outputs, dim=0)
#             output_tensor = output_tensor.reshape(batch_size, num_heads, head_dim)
#             output_list.append(output_tensor)
        
#         # Stack outputs for all query positions
#         # [q_len, batch, heads, head_dim] -> [batch, heads, q_len, head_dim]
#         attn_output = torch.stack(output_list, dim=2)
        
#         return attn_output
    
#     def _standard_attention(
#         self,
#         query_states: torch.Tensor,
#         key_states: torch.Tensor,
#         value_states: torch.Tensor,
#         scaling: float,
#         attention_mask: Optional[torch.Tensor] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Compute standard scaled dot-product attention (fallback).
        
#         Args:
#             query_states: [batch_size, num_heads, q_len, head_dim]
#             key_states: [batch_size, num_heads, kv_len, head_dim]
#             value_states: [batch_size, num_heads, kv_len, head_dim]
#             scaling: Attention scaling factor
#             attention_mask: Optional mask
            
#         Returns:
#             Tuple of (attention_output, attention_weights)
#         """
#         # Compute attention scores
#         attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scaling
        
#         # Apply attention mask if provided
#         if attention_mask is not None:
#             attn_weights = attn_weights + attention_mask
        
#         # Softmax
#         attn_weights = F.softmax(attn_weights, dim=-1)
        
#         # Weighted sum of values
#         attn_output = torch.matmul(attn_weights, value_states)
        
#         return attn_output, attn_weights


# def patch_model_with_hira_attention(
#     model: nn.Module,
#     config: Optional[dict] = None,
# ):
#     """
#     Patch a HuggingFace model to use HiraAttention.
    
#     This function modifies the model's attention layers to use hierarchical
#     range-searching attention instead of standard attention.
    
#     TODO: Implement model-specific patching for different architectures
#     Currently supports: Llama-style models
    
#     Args:
#         model: HuggingFace model to patch
#         config: Configuration dict with keys:
#                - num_levels: Number of hirarchy levels
#                - branching_factor: Clusters per level
#                - threshold: Fixed threshold value
#                - use_hira_during_prefill: Whether to use during prefill
    
#     Example:
#         >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
#         >>> patch_model_with_hira_attention(model, config={"num_levels": 3})
#     """
#     config = config or {}
    
#     # Extract configuration
#     threshold = config.get("threshold", 0.0)
#     use_hira_during_prefill = config.get("use_hira_during_prefill", False)
    
#     # Create threshold strategy and range searcher
#     threshold_strategy = FixedThresholdStrategy(threshold=threshold)
#     range_searcher = HalfspaceSearcher()
    
#     # Create HiraAttention instance
#     hira_attention = HiraAttention(
#         threshold_strategy=threshold_strategy,
#         range_searcher=range_searcher,
#         use_hira_during_prefill=use_hira_during_prefill,
#     )
    
#     # TODO: Implement model-specific patching logic
#     # This would involve:
#     # 1. Detecting the model architecture
#     # 2. Finding attention layers
#     # 3. Wrapping or replacing the attention forward method
#     # 4. Ensuring compatibility with the model's attention interface
    
#     # For now, return the HiraAttention instance for manual patching
#     # See examples/ for how to manually patch models
    
#     print("Note: Automatic patching not yet implemented.")
#     print("Please see examples/patch_llama.py for manual patching instructions.")
#     print(f"Created HiraAttention with threshold={threshold}")
    
#     return hira_attention
