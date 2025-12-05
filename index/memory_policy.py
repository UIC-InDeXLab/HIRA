"""
Memory tiering policies for hierarchical index device placement.

The MemoryTieringPolicy determines which levels of the hirarchy should reside
on GPU vs CPU. This is critical for managing memory as cache size grows.
"""

from abc import ABC, abstractmethod
from typing import Dict, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .index import Index


class MemoryTieringPolicy(ABC):
    """
    Abstract base class for memory tiering policies.
    
    A MemoryTieringPolicy decides which levels of the hierarchical index
    should reside on GPU vs CPU. This allows for memory-performance tradeoffs.
    """
    
    @abstractmethod
    def get_device_assignments(
        self, index: "Index"
    ) -> Dict[int, torch.device]:
        """
        Determine device placement for each level.
        
        Args:
            index: Index to assign devices for
            
        Returns:
            Dictionary mapping level_idx -> device
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, any]:
        """
        Return the configuration of this policy.
        
        Returns:
            Dictionary with policy configuration
        """
        pass


class AllGPUPolicy(MemoryTieringPolicy):
    """
    Simple policy: keep all levels on GPU.
    
    This is the default policy for initial implementation. It provides
    maximum performance but requires all index data to fit in GPU memory.
    
    Args:
        device: GPU device to use (e.g., "cuda:0")
    """
    
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
    
    def get_device_assignments(
        self, index: "Index"
    ) -> Dict[int, torch.device]:
        """
        Assign all levels to GPU.
        
        Args:
            index: Index
            
        Returns:
            Dictionary with all levels assigned to GPU
        """
        return {
            level_idx: self.device
            for level_idx in range(index.num_levels())
        }
    
    def get_config(self) -> Dict[str, any]:
        """Return policy configuration."""
        return {
            "policy_type": "all_gpu",
            "device": str(self.device),
        }


class HybridGPUCPUPolicy(MemoryTieringPolicy):
    """
    Hybrid policy: store some levels on GPU, others on CPU.
    
    This policy enables memory-performance tradeoffs by keeping frequently
    accessed (coarse) levels on GPU and less frequently accessed (fine)
    levels on CPU.
    
    The intuition is:
    - Coarse levels (level 0, 1, ...) are accessed for every query (pruning)
    - Fine levels may only be accessed for a subset of queries
    - By offloading fine levels to CPU, we save GPU memory
    
    TODO: Implement dynamic policy that adjusts based on memory pressure
    TODO: Add support for prefetching fine levels when needed
    
    Args:
        num_gpu_levels: Number of levels to keep on GPU (starting from level 0)
        gpu_device: GPU device (e.g., "cuda:0")
        cpu_device: CPU device (default: "cpu")
    """
    
    def __init__(
        self,
        num_gpu_levels: int = 2,
        gpu_device: str = "cuda:0",
        cpu_device: str = "cpu",
    ):
        self.num_gpu_levels = num_gpu_levels
        self.gpu_device = torch.device(gpu_device)
        self.cpu_device = torch.device(cpu_device)
    
    def get_device_assignments(
        self, index: "Index"
    ) -> Dict[int, torch.device]:
        """
        Assign first N levels to GPU, rest to CPU.
        
        Args:
            index: Index
            
        Returns:
            Dictionary with device assignments
        """
        assignments = {}
        for level_idx in range(index.num_levels()):
            if level_idx < self.num_gpu_levels:
                assignments[level_idx] = self.gpu_device
            else:
                assignments[level_idx] = self.cpu_device
        return assignments
    
    def get_config(self) -> Dict[str, any]:
        """Return policy configuration."""
        return {
            "policy_type": "hybrid_gpu_cpu",
            "num_gpu_levels": self.num_gpu_levels,
            "gpu_device": str(self.gpu_device),
            "cpu_device": str(self.cpu_device),
        }


class AdaptivePolicy(MemoryTieringPolicy):
    """
    Adaptive policy that adjusts device placement based on memory usage.
    
    TODO: Implement adaptive policy
    This policy would:
    - Monitor GPU memory usage
    - Dynamically move levels between GPU and CPU based on availability
    - Use access statistics to prioritize which levels to keep on GPU
    
    Args:
        target_gpu_usage: Target GPU memory usage (0.0 - 1.0)
        gpu_device: GPU device
        cpu_device: CPU device
    """
    
    def __init__(
        self,
        target_gpu_usage: float = 0.8,
        gpu_device: str = "cuda:0",
        cpu_device: str = "cpu",
    ):
        self.target_gpu_usage = target_gpu_usage
        self.gpu_device = torch.device(gpu_device)
        self.cpu_device = torch.device(cpu_device)
    
    def get_device_assignments(
        self, index: "Index"
    ) -> Dict[int, torch.device]:
        """
        Adaptively assign devices based on memory usage.
        
        TODO: Implement adaptive logic
        For now, falls back to putting all on GPU.
        
        Args:
            index: Index
            
        Returns:
            Dictionary with device assignments
        """
        # TODO: Implement adaptive strategy based on:
        # - Current GPU memory usage
        # - Index level sizes
        # - Access patterns
        
        # For now, simple fallback: all on GPU
        return {
            level_idx: self.gpu_device
            for level_idx in range(index.num_levels())
        }
    
    def get_config(self) -> Dict[str, any]:
        """Return policy configuration."""
        return {
            "policy_type": "adaptive",
            "target_gpu_usage": self.target_gpu_usage,
            "gpu_device": str(self.gpu_device),
            "cpu_device": str(self.cpu_device),
        }
