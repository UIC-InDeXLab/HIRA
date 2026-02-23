/*
 * HIRA – Zero-copy fused tree-walk + exact-filter kernel.
 *
 * Uses torch/extension.h to accept torch::Tensor directly — no numpy
 * conversion or memory copies.  The kernel reads from and writes to
 * PyTorch's own memory buffers.
 *
 * Built via JIT:  torch.utils.cpp_extension.load()
 *
 * Provides two functions:
 *   - fused_tree_search(keys, query, thresholds, centers, radii, c2p, sizes)
 *   - exact_filter(keys, leaf_mask, query, thresholds)
 */

#include <torch/extension.h>

#include <vector>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/*  Helper: dot product with SIMD hint                                */
/* ------------------------------------------------------------------ */
static inline float dot(const float* __restrict__ a,
                        const float* __restrict__ b,
                        int64_t D)
{
    float s = 0.0f;
    #pragma omp simd reduction(+:s)
    for (int64_t j = 0; j < D; ++j)
        s += a[j] * b[j];
    return s;
}

/* ------------------------------------------------------------------ */
/*  exact_filter                                                      */
/* ------------------------------------------------------------------ */
/*
 * Args:
 *   keys       : (H, N, D)  float32, contiguous
 *   leaf_mask  : (H, N)     bool,    contiguous
 *   query      : (H, D)     float32, contiguous
 *   thresholds : (H,)       float32, contiguous
 *
 * Returns:
 *   (H, N) float32 – score where qualifying, 0.0 elsewhere.
 */
torch::Tensor exact_filter(
    torch::Tensor keys,
    torch::Tensor leaf_mask,
    torch::Tensor query,
    torch::Tensor thresholds)
{
    TORCH_CHECK(keys.dim() == 3,       "keys must be 3-D (H, N, D)");
    TORCH_CHECK(leaf_mask.dim() == 2,  "leaf_mask must be 2-D (H, N)");
    TORCH_CHECK(query.dim() == 2,      "query must be 2-D (H, D)");
    TORCH_CHECK(thresholds.dim() == 1, "thresholds must be 1-D (H,)");

    keys       = keys.contiguous().to(torch::kFloat32);
    leaf_mask  = leaf_mask.contiguous().to(torch::kBool);
    query      = query.contiguous().to(torch::kFloat32);
    thresholds = thresholds.contiguous().to(torch::kFloat32);

    const int64_t H = keys.size(0);
    const int64_t N = keys.size(1);
    const int64_t D = keys.size(2);

    const float* k_ptr  = keys.data_ptr<float>();
    const bool*  m_ptr  = leaf_mask.data_ptr<bool>();
    const float* q_ptr  = query.data_ptr<float>();
    const float* th_ptr = thresholds.data_ptr<float>();

    auto result = torch::zeros({H, N}, keys.options());
    float* r_ptr = result.data_ptr<float>();

    const int64_t total = H * N;

    #pragma omp parallel for schedule(dynamic, 256)
    for (int64_t flat = 0; flat < total; ++flat) {
        if (!m_ptr[flat])
            continue;

        const int64_t h = flat / N;
        const int64_t i = flat % N;

        const float* ki = k_ptr + (h * N + i) * D;
        const float* qh = q_ptr + h * D;
        const float  th = th_ptr[h];

        float s = dot(ki, qh, D);
        if (s >= th)
            r_ptr[flat] = s;
    }

    return result;
}

/* ------------------------------------------------------------------ */
/*  fused_tree_search                                                 */
/* ------------------------------------------------------------------ */
/*
 * Args:
 *   keys         : (H, N, D)    float32
 *   query        : (H, D)       float32
 *   thresholds   : (H,)         float32
 *   level_centers: vector of (H, K_l, D) float32 tensors
 *   level_radii  : vector of (H, K_l)    float32 tensors
 *   level_c2p    : vector of (H, K_l)    int64 tensors  (empty for root)
 *   level_sizes  : vector of int64                       (K_l per level)
 *
 * Returns:
 *   (H, N) float32 – qualifying scores, 0.0 elsewhere.
 */
torch::Tensor fused_tree_search(
    torch::Tensor keys,
    torch::Tensor query,
    torch::Tensor thresholds,
    std::vector<torch::Tensor> level_centers,
    std::vector<torch::Tensor> level_radii,
    std::vector<torch::Tensor> level_c2p,
    std::vector<int64_t> level_sizes)
{
    keys       = keys.contiguous().to(torch::kFloat32);
    query      = query.contiguous().to(torch::kFloat32);
    thresholds = thresholds.contiguous().to(torch::kFloat32);

    const int64_t H = keys.size(0);
    const int64_t N = keys.size(1);
    const int64_t D = keys.size(2);
    const int64_t num_levels = static_cast<int64_t>(level_sizes.size());

    const float* k_ptr  = keys.data_ptr<float>();
    const float* q_ptr  = query.data_ptr<float>();
    const float* th_ptr = thresholds.data_ptr<float>();

    /* Pre-extract raw pointers for each level */
    struct LevelData {
        const float*   centers;
        const float*   radii;
        const int64_t* c2p;
        int64_t        K;
    };

    std::vector<LevelData> levels(num_levels);
    /* Make sure tensors are contiguous and correct dtype */
    for (int64_t l = 0; l < num_levels; ++l) {
        level_centers[l] = level_centers[l].contiguous().to(torch::kFloat32);
        level_radii[l]   = level_radii[l].contiguous().to(torch::kFloat32);

        levels[l].centers = level_centers[l].data_ptr<float>();
        levels[l].radii   = level_radii[l].data_ptr<float>();
        if (l < num_levels - 1 && level_c2p[l].numel() > 0) {
            level_c2p[l] = level_c2p[l].contiguous().to(torch::kInt64);
            levels[l].c2p = level_c2p[l].data_ptr<int64_t>();
        } else {
            levels[l].c2p = nullptr;
        }
        levels[l].K = level_sizes[l];
    }

    auto result = torch::zeros({H, N}, keys.options());
    float* r_ptr = result.data_ptr<float>();

    /* Per-head tree walk (parallelised over H) */
    #pragma omp parallel
    {
        std::vector<char> mask_a, mask_b;

        #pragma omp for schedule(dynamic, 1)
        for (int64_t h = 0; h < H; ++h) {

            const float  th  = th_ptr[h];
            const float* q_h = q_ptr + h * D;

            /* Single level: flat scan */
            if (num_levels == 1) {
                for (int64_t i = 0; i < N; ++i) {
                    float s = dot(k_ptr + (h * N + i) * D, q_h, D);
                    if (s >= th) r_ptr[h * N + i] = s;
                }
                continue;
            }

            /* Root level */
            const int64_t root_idx = num_levels - 1;
            const LevelData& root = levels[root_idx];
            const int64_t K_root = root.K;

            mask_a.assign(static_cast<size_t>(K_root), 0);
            for (int64_t c = 0; c < K_root; ++c) {
                float s = dot(root.centers + (h * K_root + c) * D, q_h, D);
                float r = root.radii[h * K_root + c];
                mask_a[c] = (s + r >= th) ? 1 : 0;
            }

            /* Walk intermediate levels (root-1 … 1) */
            char* parent_mask = mask_a.data();
            char* child_mask_ptr;

            for (int64_t l = root_idx - 1; l >= 1; --l) {
                const LevelData& lev = levels[l];
                const int64_t K_child = lev.K;

                mask_b.assign(static_cast<size_t>(K_child), 0);
                child_mask_ptr = mask_b.data();

                const int64_t* c2p_h = lev.c2p + h * K_child;

                for (int64_t c = 0; c < K_child; ++c) {
                    if (!parent_mask[c2p_h[c]])
                        continue;
                    float s = dot(lev.centers + (h * K_child + c) * D, q_h, D);
                    float r = lev.radii[h * K_child + c];
                    child_mask_ptr[c] = (s + r >= th) ? 1 : 0;
                }

                std::swap(mask_a, mask_b);
                parent_mask = mask_a.data();
            }

            /* Level-0: expand to leaves and exact filter */
            const LevelData& lev0 = levels[0];
            const int64_t* c2p0_h = lev0.c2p + h * N;

            for (int64_t i = 0; i < N; ++i) {
                if (!parent_mask[c2p0_h[i]])
                    continue;
                float s = dot(k_ptr + (h * N + i) * D, q_h, D);
                if (s >= th)
                    r_ptr[h * N + i] = s;
            }
        }
    } /* end omp parallel */

    return result;
}

/* ------------------------------------------------------------------ */
/*  pybind11 module                                                   */
/* ------------------------------------------------------------------ */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "HIRA zero-copy C++/OpenMP kernels (torch::Tensor interface)";
    m.def("exact_filter",       &exact_filter,
          "Batched exact dot-product filter (H,N,D) keys + (H,N) mask -> (H,N) scores");
    m.def("fused_tree_search",  &fused_tree_search,
          "Fused tree-walk + exact filter -> (H,N) scores");
}
