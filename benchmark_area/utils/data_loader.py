import torch
import numpy as np


def get_real_data(
    num_keys: int, dim: int, real_data_path: str, seed: int = 42
) -> torch.Tensor:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"  Loading real data from: {real_data_path}")
    data = np.load(real_data_path)
    keys = torch.from_numpy(data["keys"]).float()

    # Subsample if needed
    if len(keys) > num_keys:
        indices = torch.randperm(len(keys))[:num_keys]
        keys = keys[indices]

    print(f"  Loaded {len(keys)} real keys (dimension={keys.shape[1]})")

    return keys
