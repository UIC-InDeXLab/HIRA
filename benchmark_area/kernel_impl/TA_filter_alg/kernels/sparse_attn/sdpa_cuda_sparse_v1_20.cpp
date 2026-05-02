#include <torch/extension.h>

void sdpa_cuda_sparse_v1_20_launch(
    torch::Tensor q,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor mask,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_o,
    torch::Tensor counters,
    torch::Tensor out,
    double scale,
    int64_t num_splits);

torch::Tensor sdpa_cuda_sparse_v1_20_forward(
    torch::Tensor q,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor mask,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_o,
    torch::Tensor counters,
    torch::Tensor out,
    double scale,
    int64_t num_splits) {
  sdpa_cuda_sparse_v1_20_launch(
      q, keys, values, mask, partial_m, partial_l, partial_o,
      counters, out, scale, num_splits);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sdpa_cuda_sparse_v1_20_forward, "Masked decode SDPA fp16 CUDA sparse v1.20");
}
