#include <torch/extension.h>

void sdpa_cuda_sparse_v1_6_launch(
    torch::Tensor q,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor mask,
    torch::Tensor indices,
    torch::Tensor block_counts,
    torch::Tensor block_offsets,
    torch::Tensor counts,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_o,
    torch::Tensor counters,
    torch::Tensor out,
    double scale,
    int64_t num_splits);

torch::Tensor sdpa_cuda_sparse_v1_6_forward(
    torch::Tensor q,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor mask,
    torch::Tensor indices,
    torch::Tensor block_counts,
    torch::Tensor block_offsets,
    torch::Tensor counts,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_o,
    torch::Tensor counters,
    torch::Tensor out,
    double scale,
    int64_t num_splits) {
  sdpa_cuda_sparse_v1_6_launch(
      q, keys, values, mask, indices, block_counts, block_offsets, counts,
      partial_m, partial_l, partial_o, counters, out, scale, num_splits);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sdpa_cuda_sparse_v1_6_forward, "Masked decode SDPA fp16 CUDA sparse v1.6");
}
