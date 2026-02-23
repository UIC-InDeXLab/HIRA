from dataclasses import dataclass


# Operation modes
@dataclass
class DeviceMode:
    CPU_ONLY = "cpu_only"
    CUDA_ONLY = "cuda_only"
    CPU_CUDA = "cpu-cuda"


@dataclass
class HiraConfig:
    # cache
    device_mode: str = DeviceMode.CPU_ONLY  # "cpu_only" or "cuda_only" or "cpu-cuda"
    update_every: int = 1
    # indexers
    num_levels: int = 5
    max_iterations: int = 1
    branching_factor: int = 10
    # threshold
    threshold_method: str = "sample_mean_topk"  # "sample_max" or "sample_mean_topk"
    sample_size: int = 1000

    @classmethod
    def from_dict(cls, config_dict: dict) -> "HiraConfig":
        raise NotImplementedError("from_dict method is not implemented yet")

    @classmethod
    def from_yaml(cls, config_path: str) -> "HiraConfig":
        raise NotImplementedError("from_yaml method is not implemented yet")

    def get_indexer_kwargs(self) -> dict:
        return {
            "num_levels": self.num_levels,
            "max_iterations": self.max_iterations,
            "branching_factor": self.branching_factor,
        }
