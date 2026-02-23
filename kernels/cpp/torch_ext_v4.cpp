/*
 * HIRA – Zero-copy fused tree-walk + exact-filter kernel (v4).
 *
 * Optimized for fixed head dimension D=128:
 *   1) active-parent traversal (CSR adjacency; same pruning semantics as v3)
 *   2) compact candidate list
 *   3) dot-product specialized for D=128
 *
 * This version intentionally co-exists with v1/v2/v3 for A/B testing.
 */

#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline float dot128(const float* __restrict__ a,
                           const float* __restrict__ b) {
    float s = 0.0f;
    #pragma omp simd reduction(+:s)
    for (int64_t j = 0; j < 128; ++j) {
        s += a[j] * b[j];
    }
    return s;
}

torch::Tensor fused_tree_search_v4(
    torch::Tensor keys,                         // (H, N, D) float
    torch::Tensor query,                        // (H, D) float
    torch::Tensor thresholds,                   // (H,) float
    std::vector<torch::Tensor> level_centers,   // (H, K_l, D)
    std::vector<torch::Tensor> level_radii,     // (H, K_l)
    std::vector<torch::Tensor> level_c2p,       // kept for API parity (unused)
    std::vector<int64_t> level_sizes,           // K_l
    std::vector<torch::Tensor> level_offsets,   // (H, P_l+1), empty for root
    std::vector<torch::Tensor> level_children)  // (H, K_l),   empty for root
{
    TORCH_CHECK(keys.dim() == 3, "keys must be (H, N, D)");
    TORCH_CHECK(query.dim() == 2, "query must be (H, D)");
    TORCH_CHECK(thresholds.dim() == 1, "thresholds must be (H,)");
    TORCH_CHECK(!level_sizes.empty(), "level_sizes must be non-empty");
    TORCH_CHECK(level_offsets.size() == level_sizes.size(),
                "level_offsets must have one entry per level");
    TORCH_CHECK(level_children.size() == level_sizes.size(),
                "level_children must have one entry per level");

    keys = keys.contiguous().to(torch::kFloat32);
    query = query.contiguous().to(torch::kFloat32);
    thresholds = thresholds.contiguous().to(torch::kFloat32);

    const int64_t H = keys.size(0);
    const int64_t N = keys.size(1);
    const int64_t D = keys.size(2);
    const int64_t num_levels = static_cast<int64_t>(level_sizes.size());

    TORCH_CHECK(D == 128, "v4 is specialized for D=128; got D=", D);
    TORCH_CHECK(query.size(0) == H && query.size(1) == D,
                "query shape mismatch vs keys");
    TORCH_CHECK(thresholds.size(0) == H,
                "threshold shape mismatch vs keys");

    (void)level_c2p; // API compatibility

    struct LevelData {
        const float* centers;      // (H, K, D)
        const float* radii;        // (H, K)
        const int64_t* offsets;    // (H, P+1) for non-root
        const int64_t* children;   // (H, K)
        int64_t K;                 // #nodes at this level
        int64_t P;                 // #parents at upper level
    };

    std::vector<LevelData> levels(num_levels);
    for (int64_t l = 0; l < num_levels; ++l) {
        TORCH_CHECK(level_centers[l].dim() == 3, "level_centers[", l, "] must be 3-D");
        TORCH_CHECK(level_radii[l].dim() == 2, "level_radii[", l, "] must be 2-D");

        level_centers[l] = level_centers[l].contiguous().to(torch::kFloat32);
        level_radii[l] = level_radii[l].contiguous().to(torch::kFloat32);

        levels[l].centers = level_centers[l].data_ptr<float>();
        levels[l].radii = level_radii[l].data_ptr<float>();
        levels[l].K = level_sizes[l];
        levels[l].P = (l < num_levels - 1) ? level_sizes[l + 1] : 0;

        if (l < num_levels - 1 && level_offsets[l].numel() > 0 && level_children[l].numel() > 0) {
            level_offsets[l] = level_offsets[l].contiguous().to(torch::kInt64);
            level_children[l] = level_children[l].contiguous().to(torch::kInt64);
            levels[l].offsets = level_offsets[l].data_ptr<int64_t>();
            levels[l].children = level_children[l].data_ptr<int64_t>();
        } else {
            levels[l].offsets = nullptr;
            levels[l].children = nullptr;
        }
    }

    const float* k_ptr = keys.data_ptr<float>();
    const float* q_ptr = query.data_ptr<float>();
    const float* th_ptr = thresholds.data_ptr<float>();

    auto result = torch::zeros({H, N}, keys.options());
    float* r_ptr = result.data_ptr<float>();

    // Stage 1: per-head traversal -> compact candidate keys.
    std::vector<std::vector<int64_t>> candidates(static_cast<size_t>(H));

    #pragma omp parallel
    {
        std::vector<int64_t> active, next_active;
        std::vector<uint8_t> root_mask;
        std::vector<int64_t> local_candidates;

        #pragma omp for schedule(static)
        for (int64_t h = 0; h < H; ++h) {
            const float th = th_ptr[h];
            const float* q_h = q_ptr + h * D;

            local_candidates.clear();

            if (num_levels == 1) {
                local_candidates.reserve(static_cast<size_t>(N));
                for (int64_t i = 0; i < N; ++i) {
                    local_candidates.push_back(i);
                }
                candidates[static_cast<size_t>(h)] = std::move(local_candidates);
                continue;
            }

            const int64_t root_idx = num_levels - 1;
            const LevelData& root = levels[root_idx];
            const int64_t K_root = root.K;

            active.clear();
            root_mask.assign(static_cast<size_t>(K_root), uint8_t(0));
            for (int64_t c = 0; c < K_root; ++c) {
                const float s = dot128(root.centers + (h * K_root + c) * D, q_h);
                const float r = root.radii[h * K_root + c];
                if (s + r >= th) {
                    root_mask[static_cast<size_t>(c)] = uint8_t(1);
                    active.push_back(c);
                }
            }

            for (int64_t l = root_idx - 1; l >= 1; --l) {
                if (active.empty()) {
                    break;
                }
                const LevelData& lev = levels[l];
                const int64_t K_child = lev.K;
                const int64_t P = lev.P;

                TORCH_CHECK(lev.offsets != nullptr && lev.children != nullptr,
                            "v4 requires adjacency tensors for non-root levels");

                const int64_t* off_h = lev.offsets + h * (P + 1);
                const int64_t* child_h = lev.children + h * K_child;

                next_active.clear();
                for (int64_t p : active) {
                    const int64_t begin = off_h[p];
                    const int64_t end = off_h[p + 1];
                    for (int64_t t = begin; t < end; ++t) {
                        const int64_t c = child_h[t];
                        const float s = dot128(lev.centers + (h * K_child + c) * D, q_h);
                        const float r = lev.radii[h * K_child + c];
                        if (s + r >= th) {
                            next_active.push_back(c);
                        }
                    }
                }
                active.swap(next_active);
            }

            if (!active.empty()) {
                const LevelData& lev0 = levels[0];
                const int64_t P0 = lev0.P;
                TORCH_CHECK(lev0.offsets != nullptr && lev0.children != nullptr,
                            "v4 requires adjacency tensors for level 0");

                const int64_t* off0_h = lev0.offsets + h * (P0 + 1);
                const int64_t* child0_h = lev0.children + h * N;

                // Conservative reserve to avoid repeated growth.
                local_candidates.reserve(static_cast<size_t>(N / 4));
                for (int64_t p : active) {
                    const int64_t begin = off0_h[p];
                    const int64_t end = off0_h[p + 1];
                    for (int64_t t = begin; t < end; ++t) {
                        local_candidates.push_back(child0_h[t]);
                    }
                }
            }

            candidates[static_cast<size_t>(h)] = std::move(local_candidates);
        }
    }

    // Stage 2: exact filter per head over compact candidates.
    #pragma omp parallel for schedule(static)
    for (int64_t h = 0; h < H; ++h) {
        const float* q_h = q_ptr + h * D;
        const float th = th_ptr[h];
        const auto& cand = candidates[static_cast<size_t>(h)];
        for (int64_t i : cand) {
            const float s = dot128(k_ptr + (h * N + i) * D, q_h);
            if (s >= th) {
                r_ptr[h * N + i] = s;
            }
        }
    }

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "HIRA zero-copy C++/OpenMP kernels v4 (D=128 specialized)";
    m.def("fused_tree_search_v4", &fused_tree_search_v4,
          "Fused tree-walk + exact filter (v4, D=128 specialized)");
}
