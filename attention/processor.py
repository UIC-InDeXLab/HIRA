"""
HuggingFace-compatible attention processor for Hira.

This module provides attention processors that integrate with HuggingFace's
attention processor API.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .hira_attention import HiraAttention
from ..utils import FixedThresholdStrategy
from ..search import HalfspaceRangeSearcher


class HiraAttentionProcessor:
    """
    Attention processor for HuggingFace models using HiraAttention.
    
    This processor can be set on HuggingFace model attention layers that
    support custom attention processors.
    
    TODO: Implement full HF attention processor API compatibility
    TODO: Add support for different attention variants (flash, sdpa, etc.)
    
    Args:
        threshold: Fixed threshold for key selection
        use_hira_during_prefill: Whether to use hierarchical search during prefill
    
    Example:
        >>> processor = HiraAttentionProcessor(threshold=0.0)
        >>> model.model.layers[0].self_attn.set_attn_processor(processor)
    """
    
    def __init__(
        self,
        threshold: float = 0.0,
        use_hira_during_prefill: bool = False,
    ):
        self.hira_attention = HiraAttention(
            threshold_strategy=FixedThresholdStrategy(threshold=threshold),
            range_searcher=HalfspaceRangeSearcher(),
            use_hira_during_prefill=use_hira_during_prefill,
        )
    
    def __call__(
        self,
        attn_module: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        """
        Process attention using HiraAttention.
        
        This method follows the HuggingFace attention processor API.
        
        Args:
            attn_module: The attention module
            hidden_states: Input hidden states
            attention_mask: Optional attention mask
            position_embeddings: RoPE embeddings (cos, sin)
            past_key_value: Cache object
            output_attentions: Whether to output attention weights
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (attention_output, attention_weights, cache)
        """
        # TODO: Implement full processor logic
        # This requires:
        # 1. Extracting Q, K, V from hidden_states using attn_module's projections
        # 2. Applying position embeddings (e.g., RoPE)
        # 3. Calling HiraAttention.forward()
        # 4. Returning in the expected format
        
        raise NotImplementedError(
            "HiraAttentionProcessor not fully implemented yet. "
            "Please see examples/ for manual attention patching."
        )
