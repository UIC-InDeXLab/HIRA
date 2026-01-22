// two_level_filter_omp.cpp
// OpenMP fused parent-filter + child scoring kernel (analogous to your Numba).
//
// Assumptions:
// - K is row-major contiguous float32: shape (N, D)
// - P is row-major contiguous float32: shape (Kp, D)
// - R is float32: shape (Kp)
// - q is float32: shape (D)
// - out is float32: shape (N) (caller allocates; only writes passing children)
//
// Compile example (GCC/Clang):
//   g++ -O3 -march=native -ffast-math -fopenmp -shared -fPIC two_level_filter_omp.cpp -o libtwo_level.so
//

#include <cstdint>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" void two_level_filter_omp(
    const float *K,          // [N * D]
    const float *P,          // [Kp * D]
    const float *R,          // [Kp]
    const float *q,          // [D]
    float *out,              // [N]
    int64_t N,               // total keys rows
    int64_t Kp,              // num parents
    int64_t D,               // n_cols (e.g., 128)
    float t,                 // threshold
    int32_t branching_factor // bf
)
{
    if (!K || !P || !R || !q || !out)
        return;
    if (N <= 0 || Kp <= 0 || D <= 0 || branching_factor <= 0)
        return;

    // Only first M children exist for the Kp parents
    const int64_t M = std::min<int64_t>(N, Kp * (int64_t)branching_factor);

// Parallelize over parents. Each parent writes to a disjoint slice [pid*bf, pid*bf+bf).
#pragma omp parallel for schedule(static)
    for (int64_t pid = 0; pid < Kp; ++pid)
    {
        const int64_t child_base = pid * (int64_t)branching_factor;
        if (child_base >= M)
            continue;

        // --- parent dot: dot(P[pid], q)
        const float *p_row = P + pid * D;

        float parent_dot = 0.0f;
// Compiler should auto-vectorize with -O3 -march=native (D=128 helps).
#pragma omp simd reduction(+ : parent_dot)
        for (int64_t j = 0; j < D; ++j)
        {
            parent_dot += p_row[j] * q[j];
        }

        if (parent_dot + R[pid] <= t)
            continue;

        // --- children
        int64_t end = child_base + (int64_t)branching_factor;
        if (end > M)
            end = M;

        for (int64_t child_id = child_base; child_id < end; ++child_id)
        {
            const float *k_row = K + child_id * D;

            float child_dot = 0.0f;
#pragma omp simd reduction(+ : child_dot)
            for (int64_t j = 0; j < D; ++j)
            {
                child_dot += k_row[j] * q[j];
            }
            out[child_id] = child_dot;
        }
    }
}
