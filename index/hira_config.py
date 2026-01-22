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
    device_mode: str = DeviceMode.CPU_ONLY  # "cpu_only" or "cuda_only" or "cpu-cuda"

    cuda_memory_limit: Optional[int] = None  # in MB, only for "cpu-cuda" mode

    @dataclass
    class IndexingConfig:
        num_levels: int = 5
        max_iterations: int = 1  # for k-means
        branching_factor: int = 10

    indexing: IndexingConfig = field(default_factory=IndexingConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "HiraConfig":
        pass

    @classmethod
    def from_yaml(cls, config_path: str) -> "HiraConfig":
        pass
