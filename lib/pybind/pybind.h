#pragma once

#include <torch/all.h>
#include <torch/python.h>

#include <cstdint>
#include <memory>
#include <optional>

namespace causalflow::petit::rocm::quantization {
struct PetitSolutionHints;
}

namespace causalflow::petit::pybind {

class VmmSymmetricHeap {
  public:
    struct Layout {
        std::uint32_t barrier_record_bytes;
        std::uint32_t rank_sym_buffer_base;
        std::uint32_t rank_slot_bytes;
        std::uint32_t local_offset;
        std::uint32_t local_bytes;
    };

    explicit VmmSymmetricHeap(int world_size);
    VmmSymmetricHeap(const VmmSymmetricHeap &) = delete;
    VmmSymmetricHeap &operator=(const VmmSymmetricHeap &) = delete;
    ~VmmSymmetricHeap();
    bool Allocate(const Layout &layout);
    torch::Tensor LocalTensor() const;
    int world_size() const;
    int rank() const;
    int device_index() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void Cleanup();
};

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

int64_t FusedMoe2StageWorkspaceSize(int64_t max_num_m_blocks, int64_t inter_dim,
                                    uint64_t solution_id);

torch::Tensor FusedMoeMatmul2Stage1(
    const torch::Tensor &intermediate, const torch::Tensor &input_q,
    const torch::Tensor &w1_q, const torch::Tensor &sorted_token_ids,
    const torch::Tensor &sorted_expert_ids, const torch::Tensor &num_valid_ids,
    int64_t topk, const torch::Tensor &input_scale,
    const torch::Tensor &w1_scale, int64_t inter_dim, int64_t num_experts,
    uint64_t solution_id, int64_t num_persistent_tgs = 0,
    const std::optional<torch::Tensor> &w13_bias = std::nullopt);

torch::Tensor FusedMoeMatmul2Stage2(
    const torch::Tensor &out, const torch::Tensor &intermediate,
    const torch::Tensor &w2_q, const torch::Tensor &sorted_token_ids,
    const torch::Tensor &sorted_weights, const torch::Tensor &sorted_expert_ids,
    const torch::Tensor &num_valid_ids, int64_t topk,
    const torch::Tensor &w2_scale, int64_t inter_dim, int64_t num_experts,
    uint64_t solution_id, int64_t num_persistent_tgs = 0,
    const std::optional<torch::Tensor> &w2_bias = std::nullopt);

pybind11::tuple MegaMoeWorkspaceInputViews(
    VmmSymmetricHeap &workspace, int64_t max_tokens, uint64_t solution_id);

torch::Tensor MegaMoe(
    VmmSymmetricHeap &workspace, const torch::Tensor &w13,
    const torch::Tensor &w2, const torch::Tensor &scales_w13,
    const torch::Tensor &scales_w2, int64_t num_tokens, uint64_t solution_id,
    const std::optional<torch::Tensor> &w13_bias = std::nullopt,
    const std::optional<torch::Tensor> &w2_bias = std::nullopt,
    const std::optional<torch::Tensor> &out = std::nullopt);

py::list GetNvFp4Solutions(
    const causalflow::petit::rocm::quantization::PetitSolutionHints &hints,
    int64_t size_m, int64_t size_n, int64_t size_k);

} // namespace causalflow::petit::pybind
