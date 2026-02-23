/*
 * HIRA – Zero-copy fused tree-walk + exact-filter kernel (v2).
 *
 * Goals vs v1:
 *   - keep previous kernel intact for A/B benchmarking
 *   - reduce high-head overhead by:
 *       1) static OpenMP scheduling in traversal
 *       2) reusable per-thread mask buffers (no per-level assign allocations)
 *       3) flattening exact filter over (H * N) work items
 *
 * No brute-force fallback is used here by design.
 */

#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline float dot(const float* __restrict__ a,
                        const float* __restrict__ b,
                        int64_t D) {
    float s = 0.0f;
    #pragma omp simd reduction(+:s)
    for (int64_t j = 0; j < D; ++j) {
        s += a[j] * b[j];
    }
    return s;
}

torch::Tensor fused_tree_search_v2(
    torch::Tensor keys,                    // (H, N, D) float
    torch::Tensor query,                   // (H, D) float
    torch::Tensor thresholds,              // (H,) float
    std::vector<torch::Tensor> level_centers,  // (H, K_l, D)
    std::vector<torch::Tensor> level_radii,    // (H, K_l)
    std::vector<torch::Tensor> level_c2p,      // (H, K_l), empty for root
    std::vector<int64_t> level_sizes) {

    TORCH_CHECK(keys.dim() == 3, "keys must be (H, N, D)");
    TORCH_CHECK(query.dim() == 2, "query must be (H, D)");
    TORCH_CHECK(thresholds.dim() == 1, "thresholds must be (H,)");
    TORCH_CHECK(!level_sizes.empty(), "level_sizes must be non-empty");

    keys = keys.contiguous().to(torch::kFloat32);
    query = query.contiguous().to(torch::kFloat32);
    thresholds = thresholds.contiguous().to(torch::kFloat32);

    const int64_t H = keys.size(0);
    const int64_t N = keys.size(1);
    const int64_t D = keys.size(2);
    const int64_t num_levels = static_cast<int64_t>(level_sizes.size());

    TORCH_CHECK(query.size(0) == H && query.size(1) == D,
                "query shape mismatch vs keys");
    TORCH_CHECK(thresholds.size(0) == H,
                "threshold shape mismatch vs keys");

    struct LevelData {
        const float* centers;   // (H, K, D)
        const float* radii;     // (H, K)
        const int64_t* c2p;     // (H, K) for non-root
        int64_t K;
    };

    std::vector<LevelData> levels(num_levels);
    int64_t max_nonleaf_k = 1;
    for (int64_t l = 0; l < num_levels; ++l) {
        TORCH_CHECK(level_centers[l].dim() == 3, "level_centers[", l, "] must be 3-D");
        TORCH_CHECK(level_radii[l].dim() == 2, "level_radii[", l, "] must be 2-D");

        level_centers[l] = level_centers[l].contiguous().to(torch::kFloat32);
        level_radii[l] = level_radii[l].contiguous().to(torch::kFloat32);

        levels[l].centers = level_centers[l].data_ptr<float>();
        levels[l].radii = level_radii[l].data_ptr<float>();
        levels[l].K = level_sizes[l];
        max_nonleaf_k = std::max(max_nonleaf_k, (l > 0 ? levels[l].K : int64_t(1)));

        if (l < num_levels - 1 && level_c2p[l].numel() > 0) {
            level_c2p[l] = level_c2p[l].contiguous().to(torch::kInt64);
            levels[l].c2p = level_c2p[l].data_ptr<int64_t>();
        } else {
            levels[l].c2p = nullptr;
        }
    }

    const float* k_ptr = keys.data_ptr<float>();
    const float* q_ptr = query.data_ptr<float>();
    const float* th_ptr = thresholds.data_ptr<float>();

    // Stage 1: traversal -> candidate leaf mask.
    auto leaf_mask = torch::zeros({H, N}, torch::TensorOptions().dtype(torch::kBool).device(keys.device()));
    bool* m_ptr = leaf_mask.data_ptr<bool>();

    #pragma omp parallel
    {
        std::vector<uint8_t> mask_a(static_cast<size_t>(max_nonleaf_k), uint8_t(0));
        std::vector<uint8_t> mask_b(static_cast<size_t>(max_nonleaf_k), uint8_t(0));

        #pragma omp for schedule(static)
        for (int64_t h = 0; h < H; ++h) {
            const float th = th_ptr[h];
            const float* q_h = q_ptr + h * D;

            if (num_levels == 1) {
                bool* m_h = m_ptr + h * N;
                for (int64_t i = 0; i < N; ++i) {
                    m_h[i] = true;
                }
                continue;
            }

            const int64_t root_idx = num_levels - 1;
            const LevelData& root = levels[root_idx];
            const int64_t K_root = root.K;

            std::fill(mask_a.begin(), mask_a.begin() + K_root, uint8_t(0));
            for (int64_t c = 0; c < K_root; ++c) {
                const float s = dot(root.centers + (h * K_root + c) * D, q_h, D);
                const float r = root.radii[h * K_root + c];
                mask_a[c] = (s + r >= th) ? uint8_t(1) : uint8_t(0);
            }

            uint8_t* parent_mask = mask_a.data();

            for (int64_t l = root_idx - 1; l >= 1; --l) {
                const LevelData& lev = levels[l];
                const int64_t K_child = lev.K;
                std::fill(mask_b.begin(), mask_b.begin() + K_child, uint8_t(0));

                const int64_t* c2p_h = lev.c2p + h * K_child;
                for (int64_t c = 0; c < K_child; ++c) {
                    if (!parent_mask[c2p_h[c]]) {
                        continue;
                    }
                    const float s = dot(lev.centers + (h * K_child + c) * D, q_h, D);
                    const float r = lev.radii[h * K_child + c];
                    mask_b[c] = (s + r >= th) ? uint8_t(1) : uint8_t(0);
                }

                std::swap(mask_a, mask_b);
                parent_mask = mask_a.data();
            }

            const LevelData& lev0 = levels[0];
            const int64_t* c2p0_h = lev0.c2p + h * N;
            bool* m_h = m_ptr + h * N;
            for (int64_t i = 0; i < N; ++i) {
                m_h[i] = static_cast<bool>(parent_mask[c2p0_h[i]]);
            }
        }
    }

    // Stage 2: exact filter over flat (H * N) space.
    auto result = torch::zeros({H, N}, keys.options());
    float* r_ptr = result.data_ptr<float>();
    const int64_t total = H * N;

    #pragma omp parallel for schedule(static, 256)
    for (int64_t flat = 0; flat < total; ++flat) {
        if (!m_ptr[flat]) {
            continue;
        }
        const int64_t h = flat / N;
        const int64_t i = flat % N;
        const float* ki = k_ptr + (h * N + i) * D;
        const float* qh = q_ptr + h * D;
        const float th = th_ptr[h];
        const float s = dot(ki, qh, D);
        if (s >= th) {
            r_ptr[flat] = s;
        }
    }

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "HIRA zero-copy C++/OpenMP kernels v2 (reduced high-H overhead)";
    m.def("fused_tree_search_v2", &fused_tree_search_v2,
          "Fused tree-walk + exact filter (v2)");
}

