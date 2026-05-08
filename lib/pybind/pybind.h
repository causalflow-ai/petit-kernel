#pragma once

#include <torch/all.h>
#include <torch/python.h>

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

torch::Tensor FusedMoeFp8BlockscaleG1u1(
    const torch::Tensor &input_q, const torch::Tensor &w13_q,
    const torch::Tensor &w2_q, const torch::Tensor &sorted_token_ids,
    const torch::Tensor &sorted_weights, const torch::Tensor &sorted_expert_ids,
    const torch::Tensor &num_valid_ids, int64_t topk,
    const torch::Tensor &input_scale, const torch::Tensor &fc1_scale,
    const torch::Tensor &fc2_scale, int64_t num_persistent_tgs = 0,
    const std::optional<torch::Tensor> &out = std::nullopt);

torch::Tensor FusedMoeFp8BlockscaleG1u1MxFp4(
    const torch::Tensor &input_q, const torch::Tensor &w13_q,
    const torch::Tensor &w2_q, const torch::Tensor &sorted_token_ids,
    const torch::Tensor &sorted_weights, const torch::Tensor &sorted_expert_ids,
    const torch::Tensor &num_valid_ids, int64_t topk,
    const torch::Tensor &input_scale, const torch::Tensor &fc1_scale,
    const torch::Tensor &fc2_scale, int64_t num_persistent_tgs = 0,
    const std::optional<torch::Tensor> &out = std::nullopt);

py::list GetNvFp4Solutions(
    const causalflow::petit::rocm::quantization::PetitSolutionHints &hints,
    int64_t size_m, int64_t size_n, int64_t size_k);

} // namespace causalflow::petit::pybind
