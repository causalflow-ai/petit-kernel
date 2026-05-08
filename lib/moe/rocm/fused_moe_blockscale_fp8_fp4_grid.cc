#include "fused_moe_blockscale_fp8_fp4_grid.cuh"

namespace causalflow::petit::rocm::moe {

int FusedMoEBlockScaleFP8MXFP4Weight(
    uint4 *__restrict__ out, const uint4 *act, const uint4 *w13,
    const uint4 *w2, const uint4 *sorted_token_ids, const uint4 *sorted_weights,
    const uint4 *sorted_expert_ids, const unsigned *num_valid_ids,
    unsigned topk, const uint4 *scales_act, const uint4 *scales_w13,
    const unsigned *scales_w2, unsigned max_num_m_blocks, unsigned m,
    unsigned n, unsigned k, unsigned num_experts, hipStream_t stream,
    unsigned num_persistent_tgs) {
    if (out == nullptr || num_valid_ids == nullptr) {
        return kFusedMoEErrorInvalidArgument;
    }

    if (n == 0 || k == 0 || n % kQuantBlockK != 0 || k % kQuantBlockK != 0) {
        return kFusedMoEErrorInvalidShape;
    }

    if (act == nullptr || w13 == nullptr || w2 == nullptr ||
        sorted_token_ids == nullptr || sorted_weights == nullptr ||
        sorted_expert_ids == nullptr || scales_act == nullptr ||
        scales_w13 == nullptr || scales_w2 == nullptr) {
        return kFusedMoEErrorInvalidArgument;
    }

    LaunchOnestageFusedMoEBlockScaleFP8<
        FusedMoEMxFp4Config,
        FusedMoEBlockScaleFP8Fp4Kernel<FusedMoEMxFp4Config>>(
        out, act, w13, w2, sorted_token_ids, sorted_weights, sorted_expert_ids,
        num_valid_ids, topk, scales_act, scales_w13, scales_w2,
        max_num_m_blocks, m, n, k, num_experts, stream, num_persistent_tgs);

    auto e = hipGetLastError();
    return e == hipSuccess ? 0 : kFusedMoEErrorInvalidArgument;
}

} // namespace causalflow::petit::rocm::moe
