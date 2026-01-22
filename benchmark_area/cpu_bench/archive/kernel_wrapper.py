import os
import ctypes
import numpy as np

# Load shared library next to this file
_lib_path = os.path.join(os.path.dirname(__file__), "libtwo_level.so")
_lib = ctypes.CDLL(_lib_path)

# Declare the C function signature
_two_level = _lib.two_level_filter_omp
_two_level.argtypes = [
    ctypes.c_void_p,  # K (float*)
    ctypes.c_void_p,  # P (float*)
    ctypes.c_void_p,  # R (float*)
    ctypes.c_void_p,  # q (float*)
    ctypes.c_void_p,  # out (float*)
    ctypes.c_int64,  # N
    ctypes.c_int64,  # Kp
    ctypes.c_int64,  # D
    ctypes.c_float,  # t
    ctypes.c_int32,  # branching_factor
]
_two_level.restype = None


def two_level_filter(K, P, R, q, t, bf, out=None):
    """
    K: (N, D) float32 C-contiguous
    P: (Kp, D) float32 C-contiguous
    R: (Kp,) float32 C-contiguous
    q: (D,) float32 C-contiguous
    t: float
    bf: int
    out: optional (N,) float32 buffer; if None, allocates (uninitialized)
         NOTE: kernel only writes passing children; you may want to prefill.
    """
    # Ensure correct dtypes + contiguous layout
    K = np.ascontiguousarray(K, dtype=np.float32)
    P = np.ascontiguousarray(P, dtype=np.float32)
    R = np.ascontiguousarray(R, dtype=np.float32)
    q = np.ascontiguousarray(q, dtype=np.float32)

    N, D = K.shape
    Kp = P.shape[0]

    if P.shape[1] != D:
        raise ValueError(f"P.shape[1] ({P.shape[1]}) must equal D ({D})")
    if R.shape[0] != Kp:
        raise ValueError(f"R.shape[0] ({R.shape[0]}) must equal Kp ({Kp})")
    if q.shape[0] != D:
        raise ValueError(f"q.shape[0] ({q.shape[0]}) must equal D ({D})")

    if out is None:
        out = np.empty((N,), dtype=np.float32)
    else:
        out = np.ascontiguousarray(out, dtype=np.float32)
        if out.shape != (N,):
            raise ValueError(f"out.shape {out.shape} must be ({N},)")

    # If you need dense semantics, prefill here:
    # out.fill(-np.inf)

    _two_level(
        K.ctypes.data,
        P.ctypes.data,
        R.ctypes.data,
        q.ctypes.data,
        out.ctypes.data,
        ctypes.c_int64(N),
        ctypes.c_int64(Kp),
        ctypes.c_int64(D),
        ctypes.c_float(t),
        ctypes.c_int32(bf),
    )
    return out
