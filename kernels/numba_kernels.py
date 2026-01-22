import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=True)
def exact_filter_mask_numba(keys, leaf_idx, query, threshold):
    """
    CPU Kernel for searching. Used in CPUSearcher.
    """
    n = leaf_idx.shape[0]
    d = query.shape[0]
    mask = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        idx = leaf_idx[i]
        s = 0.0
        for j in range(d):
            s += keys[idx, j] * query[j]
        mask[i] = s >= threshold
    return mask
