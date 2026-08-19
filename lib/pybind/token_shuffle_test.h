#pragma once

#include "pybind.h"

namespace causalflow::petit::pybind {

pybind11::tuple TokenShuffleMxFp4(
    VmmSymmetricHeap &workspace, const torch::Tensor &input_rows,
    const torch::Tensor &topk_ids, const torch::Tensor &topk_weights);

void PrepareTokenShuffleMxFp4(
    VmmSymmetricHeap &workspace, const torch::Tensor &input_rows,
    const torch::Tensor &topk_ids, const torch::Tensor &topk_weights);

void RunTokenShuffleMxFp4(VmmSymmetricHeap &workspace, int64_t num_tokens,
                          const torch::Tensor &expert_counts,
                          const torch::Tensor &profile_cycles);

pybind11::tuple EpochTokenShuffleMxFp4(
    VmmSymmetricHeap &workspace, const torch::Tensor &input_rows,
    const torch::Tensor &topk_ids, const torch::Tensor &topk_weights);

void PrepareEpochTokenShuffleMxFp4(
    VmmSymmetricHeap &workspace, const torch::Tensor &input_rows,
    const torch::Tensor &topk_ids, const torch::Tensor &topk_weights);

void RunEpochTokenShuffleMxFp4(VmmSymmetricHeap &workspace,
                               int64_t num_tokens,
                               const torch::Tensor &expert_counts,
                               const torch::Tensor &profile_cycles);

void SeedEpochTokenShuffleWrap(VmmSymmetricHeap &workspace);

pybind11::tuple DirectPushTokenShuffleMxFp4(
    VmmSymmetricHeap &workspace, const torch::Tensor &input_rows,
    const torch::Tensor &topk_ids, const torch::Tensor &topk_weights);

void PrepareDirectPushTokenShuffleMxFp4(
    VmmSymmetricHeap &workspace, const torch::Tensor &input_rows,
    const torch::Tensor &topk_ids, const torch::Tensor &topk_weights);

void RunDirectPushTokenShuffleMxFp4(
    VmmSymmetricHeap &workspace, int64_t num_tokens,
    const torch::Tensor &expert_counts,
    const torch::Tensor &profile_cycles);

} // namespace causalflow::petit::pybind
