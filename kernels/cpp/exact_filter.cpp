/*
 * HIRA – C++/OpenMP exact-filter kernel for multi-head search.
 *
 * Build:  cd hira/kernels/cpp && make
 *
 * This file provides a pybind11 module ``hira_cpp_kernels`` with one function:
 *
 *   exact_filter_mask_batched(keys, leaf_masks, query, thresholds)
 *       keys:        (H, N, D)  float32  contiguous
 *       leaf_masks:  (H, N)     bool     contiguous
 *       query:       (H, D)     float32  contiguous
 *       thresholds:  (H,)       float32  contiguous
 *       -> returns   (H, N)     float32  numpy array  (score or 0.0)
 *
 * The outer loop over (h, i) is parallelised with OpenMP; the inner
 * dot-product loop over D is written so the compiler can auto-vectorise
 * (AVX2 / AVX-512 / NEON depending on -march flags).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

/* ------------------------------------------------------------------ */
/*  exact_filter_mask_batched                                         */
/* ------------------------------------------------------------------ */

py::array_t<float> exact_filter_mask_batched(
    py::array_t<float,  py::array::c_style | py::array::forcecast> keys,
    py::array_t<bool,   py::array::c_style | py::array::forcecast> leaf_masks,
    py::array_t<float,  py::array::c_style | py::array::forcecast> query,
    py::array_t<float,  py::array::c_style | py::array::forcecast> thresholds)
{
    /* ---- Validate shapes ---- */
    if (keys.ndim() != 3)
        throw std::runtime_error("keys must be 3-D (H, N, D)");
    if (leaf_masks.ndim() != 2)
        throw std::runtime_error("leaf_masks must be 2-D (H, N)");
    if (query.ndim() != 2)
        throw std::runtime_error("query must be 2-D (H, D)");
    if (thresholds.ndim() != 1)
        throw std::runtime_error("thresholds must be 1-D (H,)");

    const ssize_t H = keys.shape(0);
    const ssize_t N = keys.shape(1);
    const ssize_t D = keys.shape(2);

    if (leaf_masks.shape(0) != H || leaf_masks.shape(1) != N)
        throw std::runtime_error("leaf_masks shape mismatch");
    if (query.shape(0) != H || query.shape(1) != D)
        throw std::runtime_error("query shape mismatch");
    if (thresholds.shape(0) != H)
        throw std::runtime_error("thresholds shape mismatch");

    /* ---- Raw pointers (no bounds-checking overhead) ---- */
    const float* k_ptr  = keys.data();
    const bool*  m_ptr  = leaf_masks.data();
    const float* q_ptr  = query.data();
    const float* th_ptr = thresholds.data();

    /* ---- Output (float32, zero-initialised) ---- */
    auto result = py::array_t<float>({H, N});
    float* r_ptr = result.mutable_data();
    std::memset(r_ptr, 0, static_cast<size_t>(H * N) * sizeof(float));

    /* ---- Parallel exact filter ---- */
    const ssize_t total = H * N;

    #pragma omp parallel for schedule(dynamic, 256)
    for (ssize_t flat = 0; flat < total; ++flat) {
        if (!m_ptr[flat])           /* skip inactive leaves */
            continue;

        const ssize_t h = flat / N;
        const ssize_t i = flat % N;

        const float* ki = k_ptr + (h * N + i) * D;   /* keys[h, i, :] */
        const float* qh = q_ptr + h * D;              /* query[h, :]   */
        const float  th = th_ptr[h];

        /* dot product – written for auto-vectorisation */
        float s = 0.0f;
        #pragma omp simd reduction(+:s)
        for (ssize_t j = 0; j < D; ++j)
            s += ki[j] * qh[j];

        if (s >= th)
            r_ptr[flat] = s;
        /* else stays 0.0f */
    }

    return result;
}

/* ------------------------------------------------------------------ */
/*  pybind11 module definition                                        */
/* ------------------------------------------------------------------ */

PYBIND11_MODULE(hira_cpp_kernels, m) {
    m.doc() = "HIRA C++/OpenMP exact-filter kernels for multi-head search";
    m.def("exact_filter_mask_batched",
          &exact_filter_mask_batched,
          py::arg("keys"),
          py::arg("leaf_masks"),
          py::arg("query"),
          py::arg("thresholds"),
          R"doc(
Multi-head batched exact dot-product filter.

Parameters
----------
keys : ndarray, shape (H, N, D), float32
    Key vectors for each head.
leaf_masks : ndarray, shape (H, N), bool
    Candidate mask – only True entries are evaluated.
query : ndarray, shape (H, D), float32
    Normalised query vector per head.
thresholds : ndarray, shape (H,), float32
    Per-head dot-product threshold.

Returns
-------
ndarray, shape (H, N), float32
    Dot-product score where leaf_masks[h,i] is True AND
    dot(keys[h,i], query[h]) >= thresholds[h], else 0.0.
)doc");
}
