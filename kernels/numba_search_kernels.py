# import numpy as np
# from numba import njit, prange


# @njit(cache=True, parallel=True, fastmath=True)
# def exact_filter_mask_numba(keys, leaf_idx, query, threshold):
#     """Single-head exact dot-product filter.

#     Args:
#         keys:      (N, D)   float array  – all key vectors for one head.
#         leaf_idx:  (M,)     int array    – candidate key indices.
#         query:     (D,)     float array  – normalised query vector.
#         threshold: float                 – dot-product threshold.

#     Returns:
#         (M,) float64 array – score where ``dot >= threshold``, else 0.0.
#     """
#     n = leaf_idx.shape[0]
#     d = query.shape[0]
#     out = np.zeros(n, dtype=np.float64)
#     for i in prange(n):
#         idx = leaf_idx[i]
#         s = 0.0
#         for j in range(d):
#             s += keys[idx, j] * query[j]
#         if s >= threshold:
#             out[i] = s
#     return out


# @njit(cache=True, parallel=True, fastmath=True)
# def exact_filter_mask_batched_numba(keys, leaf_masks, query, thresholds):
#     """Multi-head batched exact dot-product filter.

#     Parallelised across ``H * N`` elements (heads × keys) with dynamic
#     scheduling, so load imbalance from sparse ``leaf_masks`` is handled well.

#     Args:
#         keys:        (H, N, D)  float array – key vectors per head.
#         leaf_masks:  (H, N)     bool array  – candidate mask per head.
#         query:       (H, D)     float array – normalised query per head.
#         thresholds:  (H,)       float array – per-head thresholds.

#     Returns:
#         (H, N) float32 array – dot-product score where
#         ``leaf_masks[h,i] and dot(keys[h,i], query[h]) >= thresholds[h]``,
#         else 0.0.
#     """
#     H = keys.shape[0]
#     N = keys.shape[1]
#     D = keys.shape[2]
#     out = np.zeros((H, N), dtype=np.float32)
#     total = H * N
#     for flat in prange(total):
#         h = flat // N
#         i = flat % N
#         if not leaf_masks[h, i]:
#             continue
#         s = 0.0
#         th = thresholds[h]
#         for j in range(D):
#             s += keys[h, i, j] * query[h, j]
#         if s >= th:
#             out[h, i] = s
#     return out
