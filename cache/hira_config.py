from dataclasses import dataclass, field
from typing import Optional


# Operation modes
@dataclass
class DeviceMode:
    CPU_ONLY = "cpu_only"
    CUDA_ONLY = "cuda_only"
    CPU_CUDA = "cpu-cuda"


@dataclass
class HiraConfig:
    # global
    device_mode: str = DeviceMode.CPU_ONLY  # "cpu_only" or "cuda_only" or "cpu-cuda"
    cuda_memory_limit: Optional[int] = None  # in MB, only for "cpu-cuda" mode
    update_every: int = 1
    num_levels: int = 5
    max_iterations: int = 1
    branching_factor: int = 10
    # threshold finding
    threshold_method: str = "sample_mean_topk"  # "sample_max" or "sample_mean_topk"
    sample_size: int = 1000  # for threshold finding
    threshold_topk: int = 10  # for threshold finding
    # cpu indexer
    balance_every: int = 1
    # cpu searcher
    chunk_size: int = 8 * 1024
    # cuda searcher
    block_c: int = 128

    @classmethod
    def from_dict(cls, config_dict: dict) -> "HiraConfig":
        raise NotImplementedError("from_dict method is not implemented yet")

    @classmethod
    def from_yaml(cls, config_path: str) -> "HiraConfig":
        raise NotImplementedError("from_yaml method is not implemented yet")
