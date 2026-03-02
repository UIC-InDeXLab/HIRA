from abc import ABC
import torch


class BaseIndexer(ABC):
    def build(self, keys: torch.Tensor, values: torch.Tensor):
        raise NotImplementedError

    def update(self, new_keys: torch.Tensor, values: torch.Tensor):
        raise NotImplementedError
