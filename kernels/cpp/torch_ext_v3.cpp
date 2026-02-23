/*
 * HIRA – Zero-copy fused tree-walk + exact-filter kernel (v3).
 *
 * v3 focuses on reducing high-H overhead by avoiding full child scans:
 *   - traversal iterates only children of currently-active parents
 *     via precomputed per-level parent->children adjacency (CSR-like).
 *   - exact stage evaluates only candidate leaves (no H*N mask scan).
 *
 * This file is intentionally separate from v1/v2 for A/B comparisons.
 */

#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

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

torch::Tensor fused_tree_search_v3(
    torch::Tensor keys,                         // (H_kv, N, D) float
    torch::Tensor query,                        // (H_q, D) float
    torch::Tensor thresholds,                   // (H_q,) float
    std::vector<torch::Tensor> level_centers,   // (H, K_l, D)
    std::vector<torch::Tensor> level_radii,     // (H, K_l)
    std::vector<torch::Tensor> level_c2p,       // kept for API parity (unused in v3)
    std::vector<int64_t> level_sizes,           // K_l
    std::vector<torch::Tensor> level_offsets,   // (H, P_l+1), empty for root
    std::vector<torch::Tensor> level_children,  // (H, K_l),   empty for root
    torch::Tensor q_head_to_kv)
{
    TORCH_CHECK(keys.dim() == 3, "keys must be (H_kv, N, D)");
    TORCH_CHECK(query.dim() == 2, "query must be (H_q, D)");
    TORCH_CHECK(thresholds.dim() == 1, "thresholds must be (H_q,)");
    TORCH_CHECK(!level_sizes.empty(), "level_sizes must be non-empty");
    TORCH_CHECK(level_offsets.size() == level_sizes.size(),
                "level_offsets must have one entry per level");
    TORCH_CHECK(level_children.size() == level_sizes.size(),
                "level_children must have one entry per level");

    keys = keys.contiguous().to(torch::kFloat32);
    query = query.contiguous().to(torch::kFloat32);
    thresholds = thresholds.contiguous().to(torch::kFloat32);

    const int64_t H_kv = keys.size(0);
    const int64_t N = keys.size(1);
    const int64_t D = keys.size(2);
    const int64_t H_q = query.size(0);
    const int64_t num_levels = static_cast<int64_t>(level_sizes.size());

    TORCH_CHECK(query.size(1) == D, "query dim mismatch vs keys");
    TORCH_CHECK(thresholds.size(0) == H_q,
                "threshold shape mismatch vs query");

    std::vector<int64_t> q2kv(static_cast<size_t>(H_q), int64_t(0));
    if (q_head_to_kv.defined() && q_head_to_kv.numel() > 0) {
        TORCH_CHECK(q_head_to_kv.dim() == 1, "q_head_to_kv must be 1-D (H_q,)");
        TORCH_CHECK(q_head_to_kv.size(0) == H_q,
                    "q_head_to_kv must have length H_q");
        q_head_to_kv = q_head_to_kv.contiguous().to(torch::kInt64).to(torch::kCPU);
        const int64_t* map_ptr = q_head_to_kv.data_ptr<int64_t>();
        for (int64_t qh = 0; qh < H_q; ++qh) {
            const int64_t kv = map_ptr[qh];
            TORCH_CHECK(kv >= 0 && kv < H_kv,
                        "q_head_to_kv[", qh, "] out of range: ", kv);
            q2kv[static_cast<size_t>(qh)] = kv;
        }
    } else {
        if (H_q == H_kv) {
            for (int64_t qh = 0; qh < H_q; ++qh) {
                q2kv[static_cast<size_t>(qh)] = qh;
            }
        } else {
            TORCH_CHECK((H_q % H_kv) == 0,
                        "H_q must be divisible by H_kv when q_head_to_kv is not provided");
            const int64_t group_size = H_q / H_kv;
            for (int64_t qh = 0; qh < H_q; ++qh) {
                q2kv[static_cast<size_t>(qh)] = qh / group_size;
            }
        }
    }

    struct LevelData {
        const float* centers;      // (H, K, D)
        const float* radii;        // (H, K)
        const int64_t* offsets;    // (H, P+1) for non-root/non-single-level
        const int64_t* children;   // (H, K)
        int64_t K;                 // #nodes at this level
        int64_t P;                 // #parents at upper level (for offsets stride)
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

    auto result = torch::zeros({H_q, N}, keys.options());
    float* r_ptr = result.data_ptr<float>();

    #pragma omp parallel
    {
        std::vector<int64_t> active, next_active;
        std::vector<uint8_t> root_mask;

        #pragma omp for schedule(static)
        for (int64_t qh = 0; qh < H_q; ++qh) {
            const int64_t h = q2kv[static_cast<size_t>(qh)];
            const float th = th_ptr[qh];
            const float* q_h = q_ptr + qh * D;

            // Single-level corner case: flat exact scan.
            if (num_levels == 1) {
                for (int64_t i = 0; i < N; ++i) {
                    const float s = dot(k_ptr + (h * N + i) * D, q_h, D);
                    if (s >= th) {
                        r_ptr[qh * N + i] = s;
                    }
                }
                continue;
            }

            // Root activation.
            const int64_t root_idx = num_levels - 1;
            const LevelData& root = levels[root_idx];
            const int64_t K_root = root.K;

            active.clear();
            root_mask.assign(static_cast<size_t>(K_root), uint8_t(0));
            for (int64_t c = 0; c < K_root; ++c) {
                const float s = dot(root.centers + (h * K_root + c) * D, q_h, D);
                const float r = root.radii[h * K_root + c];
                if (s + r >= th) {
                    root_mask[static_cast<size_t>(c)] = uint8_t(1);
                    active.push_back(c);
                }
            }

            // Intermediate traversal: iterate only children of active parents.
            for (int64_t l = root_idx - 1; l >= 1; --l) {
                if (active.empty()) {
                    break;
                }
                const LevelData& lev = levels[l];
                const int64_t K_child = lev.K;
                const int64_t P = lev.P;  // parents at level l+1

                TORCH_CHECK(lev.offsets != nullptr && lev.children != nullptr,
                            "v3 requires adjacency tensors for non-root levels");

                const int64_t* off_h = lev.offsets + h * (P + 1);
                const int64_t* child_h = lev.children + h * K_child;

                next_active.clear();
                // Each child has exactly one parent => no duplicates across active parents.
                for (int64_t p : active) {
                    const int64_t begin = off_h[p];
                    const int64_t end = off_h[p + 1];
                    for (int64_t t = begin; t < end; ++t) {
                        const int64_t c = child_h[t];
                        const float s = dot(lev.centers + (h * K_child + c) * D, q_h, D);
                        const float r = lev.radii[h * K_child + c];
                        if (s + r >= th) {
                            next_active.push_back(c);
                        }
                    }
                }
                active.swap(next_active);
            }

            if (active.empty()) {
                continue;
            }

            // Leaf expansion + exact filter on candidates only.
            const LevelData& lev0 = levels[0];
            const int64_t P0 = lev0.P;  // level-1 size
            TORCH_CHECK(lev0.offsets != nullptr && lev0.children != nullptr,
                        "v3 requires adjacency tensors for level 0");
            const int64_t* off0_h = lev0.offsets + h * (P0 + 1);
            const int64_t* child0_h = lev0.children + h * N;

            for (int64_t p : active) {
                const int64_t begin = off0_h[p];
                const int64_t end = off0_h[p + 1];
                for (int64_t t = begin; t < end; ++t) {
                    const int64_t i = child0_h[t];
                    const float s = dot(k_ptr + (h * N + i) * D, q_h, D);
                    if (s >= th) {
                        r_ptr[qh * N + i] = s;
                    }
                }
            }
        }
    }

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "HIRA zero-copy C++/OpenMP kernels v3 (active-parent traversal)";
    m.def("fused_tree_search_v3", &fused_tree_search_v3,
          "Fused tree-walk + exact filter (v3, active-parent traversal)",
          py::arg("keys"),
          py::arg("query"),
          py::arg("thresholds"),
          py::arg("level_centers"),
          py::arg("level_radii"),
          py::arg("level_c2p"),
          py::arg("level_sizes"),
          py::arg("level_offsets"),
          py::arg("level_children"),
          py::arg("q_head_to_kv") = torch::Tensor());
}
