from numba import njit, prange
import numpy as np


@njit(cache=True, parallel=True, fastmath=True)
def two_level_filter_numba(
    K,  # (N, n_cols) float32
    P,  # (n_parents, n_cols) float32
    R,  # (n_parents,) float32
    q,  # (n_cols,) float32
    out,  # (N,) float32
    t,  # float32
    branching_factor,  # int
):
    n_parents = P.shape[0]
    n_cols = q.shape[0]

    for pid in prange(n_parents):
        # --- parent dot
        parent_dot = 0.0
        # manual dot is usually best for Numba + SIMD
        for j in range(n_cols):
            parent_dot += P[pid, j] * q[j]
        # parent_dot = np.dot(P[pid], q)

        passes = (parent_dot + R[pid]) > t
        if not passes:
            continue

        child_base = pid * branching_factor

        for c in range(branching_factor):
            child_id = child_base + c

            # --- child dot
            child_dot = 0.0
            for j in range(n_cols):
                child_dot += K[child_id, j] * q[j]
            # child_dot = np.dot(K[child_id], q)

            out[child_id] = child_dot


# wrapper
def numba_two_level_filter(K, P, R, q, t, branch, out=None):
    n, d = K.shape
    assert n % branch == 0
    m = n // branch
    assert P.shape == (m, d)
    assert R.shape == (m,)

    if out is None:
        out = np.zeros((n,), dtype=K.dtype)

    two_level_filter_numba(K, P, R, q, out, t, branch)
    return out
