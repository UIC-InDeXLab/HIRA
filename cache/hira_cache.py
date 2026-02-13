import torch
from transformers.cache_utils import Cache

from hira.index.hira_config import HiraConfig
from hira.index.hira_config import DeviceMode
from hira.index.indexer import Indexer, CPUIndexer, CUDAIndexer
from hira.index.searcher import Searcher, CPUSearcher, CUDASearcher
from hira.cache.threshold_finder import THRESHOLD_METHODS


# TODO: implement based on Cache class
class HiraIndex(Cache):
    def __init__(self, config: HiraConfig):
        self.device_mode = config.device_mode
        self.update_every = config.update_every
        self.cuda_memory_limit = config.cuda_memory_limit
        self.sample_size = config.sample_size
        self.threshold_topk = config.threshold_topk
        self.threshold_finder = THRESHOLD_METHODS[config.threshold_method]

        # index state
        self.sample_keys = None  # used for threshold finding
        self.fast_access_keys = None  # TODO: of size self.cuda_memory_limit
        self.num_fast_access_keys = 0

        # assigning indexer and searcher based on device mode
        if self.device_mode == DeviceMode.CPU_ONLY:
            self.indexer: Indexer = CPUIndexer(
                num_levels=config.num_levels,
                branching_factor=config.branching_factor,
                max_iterations=config.max_iterations,
                balance_every=config.balance_every,
            )
            self.searcher: Searcher = CPUSearcher(chunk_size=config.chunk_size)
        elif self.device_mode == DeviceMode.CUDA_ONLY:
            if config.num_levels == 2:
                depth = CUDAIndexer.Depth.TWO_LEVEL
            elif config.num_levels == 3:
                depth = CUDAIndexer.Depth.THREE_LEVEL
            else:
                raise ValueError(
                    f"CUDA indexer only supports 2 or 3 levels, got {config.num_levels}"
                )
            self.indexer: Indexer = CUDAIndexer(
                depth=depth,
                max_iterations=config.max_iterations,
                branching_factor=config.branching_factor,
            )
            self.searcher: Searcher = CUDASearcher(block_c=config.block_c)

    def build(self, keys):
        self.sample_keys = keys[torch.randperm(keys.size(0))[: self.sample_size]]
        self.indexer.build(keys)

    def search(self, query):
        # TODO: other threshold methods
        threshold = self.threshold_finder(self.sample_keys, query, self.threshold_topk)
        return self.searcher.search(query, threshold, self.indexer)

    def update(self, new_keys):
        # TODO: update self.fast_access_keys
        # TODO: Then, every self.update_every updates, call indexer.update()
        pass
