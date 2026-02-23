"""JIT-compiled PyTorch C++ extensions for HIRA search kernels.

On first import, ``torch.utils.cpp_extension.load()`` compiles
``torch_ext.cpp`` with OpenMP + native-arch optimisations. The compiled
``.so`` is cached in ``~/.cache/torch_extensions/`` and reused on
subsequent imports (recompiles automatically if the source changes).

Usage::

    from hira.kernels.cpp.torch_ext_loader import (
        hira_torch_ext,
        hira_torch_ext_v2,
        hira_torch_ext_v3,
        hira_torch_ext_v4,
    )
    result = hira_torch_ext.exact_filter(keys, leaf_mask, query, thresholds)
    result = hira_torch_ext.fused_tree_search(keys, query, thresholds, ...)
    result = hira_torch_ext_v2.fused_tree_search_v2(keys, query, thresholds, ...)
    result = hira_torch_ext_v3.fused_tree_search_v3(keys, query, thresholds, ...)
    result = hira_torch_ext_v4.fused_tree_search_v4(keys, query, thresholds, ...)
"""

import os
import sys

# Ensure the venv bin dir (where ninja lives) is on PATH so that
# torch.utils.cpp_extension can locate the ninja binary.
_venv_bin = os.path.dirname(sys.executable)
if _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

from torch.utils.cpp_extension import load

_DIR = os.path.dirname(os.path.abspath(__file__))

hira_torch_ext = load(
    name="hira_torch_ext",
    sources=[os.path.join(_DIR, "torch_ext.cpp")],
    extra_cflags=["-O3", "-march=native", "-ffast-math", "-fopenmp"],
    extra_ldflags=["-fopenmp"],
    verbose=False,
)

hira_torch_ext_v2 = load(
    name="hira_torch_ext_v2",
    sources=[os.path.join(_DIR, "torch_ext_v2.cpp")],
    extra_cflags=["-O3", "-march=native", "-ffast-math", "-fopenmp"],
    extra_ldflags=["-fopenmp"],
    verbose=False,
)

hira_torch_ext_v3 = load(
    name="hira_torch_ext_v3",
    sources=[os.path.join(_DIR, "torch_ext_v3.cpp")],
    extra_cflags=["-O3", "-march=native", "-ffast-math", "-fopenmp"],
    extra_ldflags=["-fopenmp"],
    verbose=False,
)

hira_torch_ext_v4 = load(
    name="hira_torch_ext_v4",
    sources=[os.path.join(_DIR, "torch_ext_v4.cpp")],
    extra_cflags=["-O3", "-march=native", "-ffast-math", "-fopenmp"],
    extra_ldflags=["-fopenmp"],
    verbose=False,
)
