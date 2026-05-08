#pragma once

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

static constexpr int kFusedMoEErrorInvalidShape = 1;
static constexpr int kFusedMoEErrorInvalidArgument = 2;
static constexpr int kFusedMoEErrorUnsupportedArch = 3;

// Fused-MoE FP8 block-scale kernel (single solution: 32x128x256 thread-group).
//
// Tensor layouts:
// - out:             [m, n] float
// - act:             [m, n] fp8_e4m3
// - w13:             [experts, 2 * k, n] fp8_e4m3
// - w2:              [experts, n, k] fp8_e4m3
// - sorted_token_ids:[max_num_tokens_padded] uint32 token ids
// - sorted_weights:  [max_num_tokens_padded] float route weights
// - sorted_expert_ids:[max_num_m_blocks] uint32 expert ids
// - num_valid_ids:   [2] uint32 = {num_valid_sorted_ids, token_count}
// - scales_act:      [m, n / 128] float
// - scales_w13:      [experts, (2 * k / 128) * (n / 128)] float
// - scales_w2:       [experts, (k / 128) * (n / 128)] float-bitcast-u32
//
// m = token count, n = model dim, k = inter dim.
int FusedMoEBlockScaleFP8(uint4 *__restrict__ out, const uint4 *act,
                          const uint4 *w13, const uint4 *w2,
                          const uint4 *sorted_token_ids,
                          const uint4 *sorted_weights,
                          const uint4 *sorted_expert_ids,
                          const unsigned *num_valid_ids, unsigned topk,
                          const uint4 *scales_act, const uint4 *scales_w13,
                          const unsigned *scales_w2, unsigned max_num_m_blocks,
                          unsigned m, unsigned n, unsigned k,
                          hipStream_t stream, unsigned num_persistent_tgs = 0);

int FusedMoEBlockScaleFP8MXFP4Weight(
    uint4 *__restrict__ out, const uint4 *act, const uint4 *w13,
    const uint4 *w2, const uint4 *sorted_token_ids, const uint4 *sorted_weights,
    const uint4 *sorted_expert_ids, const unsigned *num_valid_ids,
    unsigned topk, const uint4 *scales_act, const uint4 *scales_w13,
    const unsigned *scales_w2, unsigned max_num_m_blocks, unsigned m,
    unsigned n, unsigned k, unsigned num_experts, hipStream_t stream,
    unsigned num_persistent_tgs = 0);

} // namespace causalflow::petit::rocm::moe
