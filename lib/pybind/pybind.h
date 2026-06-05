#pragma once

#include <torch/all.h>
#include <torch/python.h>

#include <cstdint>
#include <optional>

namespace causalflow::petit::rocm::quantization {
struct PetitSolutionHints;
}

namespace causalflow::petit::pybind {

torch::Tensor RepackNvFp4(torch::Tensor &b_q_weight, int64_t size_n,
                          int64_t size_k);

torch::Tensor ProcessNvFp4Scales(torch::Tensor &scales, int64_t size_n,
                                 int64_t size_k);

torch::Tensor ProcessMxFp4Scales(torch::Tensor &scales, int64_t size_n,
                                 int64_t size_k);

torch::Tensor MulNvFp4A16(const torch::Tensor &A, const torch::Tensor &B,
                          const torch::Tensor &s,
                          const torch::Tensor &global_scale, int64_t size_m,
                          int64_t size_n, int64_t size_k, int64_t solution_id);

torch::Tensor MulMxFp4A16(const torch::Tensor &A, const torch::Tensor &B,
                          const torch::Tensor &s,
                          const torch::Tensor &global_scale, int64_t size_m,
                          int64_t size_n, int64_t size_k, int64_t solution_id);

torch::Tensor FusedMoeMatmul1Stage(
    const torch::Tensor &out, const torch::Tensor &input_q,
    const torch::Tensor &w1_q, const torch::Tensor &w2_q,
    const torch::Tensor &sorted_token_ids, const torch::Tensor &sorted_weights,
    const torch::Tensor &sorted_expert_ids, const torch::Tensor &num_valid_ids,
    int64_t topk, const torch::Tensor &input_scale,
    const torch::Tensor &w1_scale, const torch::Tensor &w2_scale,
    uint64_t solution_id, int64_t num_persistent_tgs = 0,
    const std::optional<torch::Tensor> &w13_bias = std::nullopt,
    const std::optional<torch::Tensor> &w2_bias = std::nullopt);

py::list GetNvFp4Solutions(
    const causalflow::petit::rocm::quantization::PetitSolutionHints &hints,
    int64_t size_m, int64_t size_n, int64_t size_k);

} // namespace causalflow::petit::pybind
