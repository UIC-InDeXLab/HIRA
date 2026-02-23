import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.configuration_utils import PreTrainedConfig
from typing import Any

from hira.cache.hira_config import HiraConfig
from hira.cache.hira_config import DeviceMode
from hira.index.indexer import CPUIndexer, CUDAIndexer


class HiraCacheLayer(CacheLayerMixin):
    """
    Single layer, but different heads (KV heads).

    Definitions:
        - D: head dim (128)
        - B: Batch size (1)
        - L: sequence length
        - H: # of heads
    e.g.
        key_states, value_states: (B, H_kv, L, D)
        query_states: (B, H_q, L, D)
        : H_q = 28
        : H_kv = 8
    """

    def __init__(self, device_mode: DeviceMode, indexer_kwargs: dict[str, Any]):
        super().__init__()
        self.device_mode = device_mode
        self.indexer_kwargs = indexer_kwargs

        self.indexer_cls = None
        if self.device_mode == DeviceMode.CPU_ONLY:
            self.indexer_cls = CPUIndexer
        elif self.device_mode == DeviceMode.CUDA_ONLY:
            self.indexer_cls = CUDAIndexer
        else:
            raise NotImplementedError(
                f"Device mode {self.device_mode} not supported yet"
            )

    def lazy_initialization(self, key_states, value_states):
        # key_states and value_states might be fake and empty
        # this is called once after prefilling
        self.dim = key_states.shape[-1]
        self.H_kv = key_states.shape[-3]

        # create indexers
        self.indexer = self.indexer_cls(**self.indexer_kwargs)

        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.keys = None  # no self.keys, just indexer

        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            # build
            self.indexer.build(key_states)
        else:
            # update
            self.indexer.update(key_states)
        # concat
        self.values = torch.cat([self.values, value_states], dim=1)

        return self.indexer, self.values

    def get_mask_sizes(self, cache_position):
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self):
        """Returns the sequence length of the cached states."""
        if not self.is_initialized or self.values.numel() == 0:
            return 0
        return self.values.shape[-2]

    def get_max_cache_shape(self):
        return -1

    def reset(self):
        """Resets the cache values while preserving the objects"""
        if self.is_initialized:
            self.indexer = self.indexer_cls(**self.indexer_kwargs)
            self.values.zero_()
        # This attribute is set on several Layers
        if hasattr(self, "cumulative_length"):
            self.cumulative_length = 0


class HiraIndex(Cache):
    """
    Implements HuggingFace's Cache interface for HIRA index.
    """

    def __init__(
        self,
        cache_config: PreTrainedConfig,
        hira_config: HiraConfig,
        num_layers: int,
    ):
        self.device_mode = hira_config.device_mode
        self.update_every = hira_config.update_every

        # extract num layers [COPIED CODE]
        config = cache_config.get_text_config(decoder=True)
        layer_types = getattr(config, "layer_types", None)
        # If `layer_types` is not explicitly provided, infer if the model is fully sliding
        if layer_types is None:
            if getattr(config, "sliding_window", None) is not None:
                layer_types = [
                    "sliding_attention" for _ in range(config.num_hidden_layers)
                ]
            elif getattr(config, "attention_chunk_size", None) is not None:
                layer_types = [
                    "chunked_attention" for _ in range(config.num_hidden_layers)
                ]
            else:
                layer_types = [
                    "full_attention" for _ in range(config.num_hidden_layers)
                ]
        # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
        if hasattr(config, "num_kv_shared_layers"):
            layer_types = layer_types[: -config.num_kv_shared_layers]

        # build layers
        layers = []
        for _ in range(num_layers):
            # treating all layer types the same
            layer = HiraCacheLayer(
                device_mode=self.device_mode,
                indexer_kwargs=hira_config.get_indexer_kwargs(),
            )
            layers.append(layer)

        super().__init__(
            layers=layers,
            layer_class_to_replicate=HiraCacheLayer,
            offloading=False,  # handle manually
            offload_only_non_sliding=None,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ):
        indexer, values = self.layers[layer_idx].update(
            key_states, value_states, cache_kwargs
        )
        return indexer, values
