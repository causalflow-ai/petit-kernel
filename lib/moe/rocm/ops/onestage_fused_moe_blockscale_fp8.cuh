#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "moe/rocm/memory_ops.cuh"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <class Config, class KernelTrait>
struct OnestageFusedMoEBlockScaleFP8 {
    using Scalar = typename KernelTrait::Scalar;
    static constexpr unsigned kGroupM = Config::kGroupM;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kThreads = kNumWarps * 64;
    static constexpr unsigned kScaleBlockSize = 128;
    static constexpr unsigned kTokenBatch = KernelTrait::kTokenBatch;
    static constexpr unsigned kSubGroupSize = 16;
    static constexpr unsigned kRoutesPerBlock = kTokenBatch * kNumWarps;
    static constexpr unsigned kRefBufferRange = static_cast<unsigned>(-16);

    using Input = typename KernelTrait::Input;
    using W13 = typename KernelTrait::W13;
    using W2 = typename KernelTrait::W2;
    using QuantizeAndShuffleOp = typename KernelTrait::QuantizeAndShuffleOp;
    using Stage1Trait = typename KernelTrait::Stage1Trait;
    using Stage1Op = typename KernelTrait::Stage1Op;
    using Stage2Trait = typename KernelTrait::Stage2Trait;
    using Stage2Op = typename KernelTrait::Stage2Op;

    struct ShmBuf {
        static constexpr unsigned kStage = Stage1Op::kStage;
        static_assert(Stage1Op::kStage == Stage2Op::kStage);
        typename Stage1Op::Shm x;
        typename QuantizeAndShuffleOp::MaxShm max;
        typename QuantizeAndShuffleOp::Shm q_h;
        typename Stage2Op::Shm ret;
    };

    __device__ static causalflow::petit::rocm::BufferResource
    MakeBufferResource(const void *ptr, unsigned range) {
        causalflow::petit::rocm::BufferResource r;
        r.v = {
            .ptr = reinterpret_cast<uintptr_t>(ptr),
            .range = range,
            .config =
                causalflow::petit::rocm::BufferResource::kDataFormatU32Config,
        };
        return r;
    }

    __device__ float2 LoadSortedWeights(const unsigned *sorted_weights,
                                        unsigned tid,
                                        unsigned route_base) const {
        auto br =
            MakeBufferResource(sorted_weights + route_base, kRefBufferRange);
        uint2 u;
        u.x = br.template LoadU32<BufferResource::kNone>(
            (tid % kSubGroupSize) * sizeof(unsigned), 0);
        u.y = br.template LoadU32<BufferResource::kNone>(
            (tid % kSubGroupSize + kSubGroupSize) * sizeof(unsigned), 0);

        return reinterpret_cast<const float2 &>(u);
    }

    __device__ void QuantizeAndShuffle(
        uint4 out[QuantizeAndShuffleOp::kElementsPerThreadVec4], float4 *dq_act,
        ShmBuf &shm, unsigned tid, unsigned wid, unsigned wtid,
        const float4 h[QuantizeAndShuffleOp::kElementsPerThreadVec4]) const {
        QuantizeAndShuffleOp::Run(out, dq_act, shm.max, shm.q_h, h, tid, wid,
                                  wtid);
    }

    __device__ void Stage1(float4 h[Stage1Trait::kAccumFragments], ShmBuf &shm,
                           unsigned wid, unsigned wtid, unsigned tid,
                           const uint2 token_select,
                           const unsigned tokens[kTokenBatch], unsigned m) {
        Stage1Trait trait{input_, w1_, w3_};
        Stage1Op::Run(h, shm.x, trait, dim_, tid, wid, wtid, token_select,
                      tokens, m);
    }

    __device__ void
    Stage2(uint4 *__restrict__ out, ShmBuf &shm,
           const uint4 quant_h[Stage2Trait::kActivationFragments],
           float4 dq_act, float2 sorted_weights,
           const unsigned tokens[kTokenBatch], unsigned invalid_token_mask,
           unsigned tid, unsigned wid, unsigned wtid) {
        Stage2Trait trait{w2_};
        Stage2Op::Run(out, shm.ret, trait, dim_, quant_h, dq_act,
                      sorted_weights, tokens, invalid_token_mask, tid, wid,
                      wtid);
    }

    __device__ void
    Compute(uint4 *__restrict__ out, const uint4 *act, const uint4 *w13_base,
            const uint4 *w2, const unsigned *sorted_token_ids,
            const unsigned *sorted_weights_ptr,
            const unsigned *sorted_expert_ids, const uint4 *scales_act,
            const unsigned *scales_w13, const unsigned *__restrict__ scales_w2,
            const unsigned *num_valid_ids_ptr, unsigned topk, unsigned m,
            unsigned dim, unsigned inter_dim, unsigned num_experts,
            unsigned persistent_route_step) {
        (void)topk;
        dim_ = dim;
        inter_dim_ = inter_dim;

        const unsigned n_blocks = dim_ / kScaleBlockSize;
        const unsigned k_blocks = inter_dim_ / kScaleBlockSize;

        const unsigned tid = threadIdx.x, tile_k = blockIdx.x;
        const unsigned wid = __builtin_amdgcn_readfirstlane(tid / kWarpSize),
                       wtid = tid % kWarpSize, col_id = tid % kSubGroupSize;

        __shared__ ShmBuf shm;
        const unsigned num_valid_ids = num_valid_ids_ptr[0];
        const unsigned num_tokens = num_valid_ids_ptr[1];
        const unsigned num_valid_m_blocks =
            tal::CeilingDiv<unsigned>(num_valid_ids, kRoutesPerBlock);
        const unsigned route_group_limit = num_valid_m_blocks;

        const unsigned route_group_begin = blockIdx.y;
        const unsigned route_group_step = persistent_route_step;
        const unsigned route_group_end =
            route_group_step ? route_group_limit : route_group_begin + 1;
        for (unsigned route_group = route_group_begin;
             route_group < route_group_end;
             route_group += (route_group_step ? route_group_step : 1)) {
            const unsigned route_base = route_group * kRoutesPerBlock;
            if (route_group < route_group_limit && route_base < num_valid_ids) {
                const unsigned expert_id = sorted_expert_ids[route_group];
                const bool valid_expert =
                    num_experts == 0 || expert_id < num_experts;

                if (valid_expert) {
                    const unsigned *scale_act_ptr =
                        reinterpret_cast<const unsigned *>(scales_act) +
                        (wid / 2) * m;
                    const unsigned input_value_range = m * dim_;
                    const unsigned input_scale_range =
                        (n_blocks - (wid / 2)) * m * sizeof(float);
                    input_.Initialize(act, input_value_range, scale_act_ptr,
                                      input_scale_range, dim_);

                    KernelTrait::InitializeWeights(
                        *this, w13_base, w2, scales_w13, scales_w2, expert_id,
                        tile_k, n_blocks, k_blocks);
                    sorted_token_br_ = MakeBufferResource(
                        sorted_token_ids + route_base, kRefBufferRange);

                    uint2 token_select;
                    token_select.x =
                        sorted_token_ids[route_base + col_id] & 0x00ffffffu;
                    token_select.y =
                        sorted_token_ids[route_base + col_id + kSubGroupSize] &
                        0x00ffffffu;

                    unsigned tokens[kTokenBatch];
                    for (int i = 0; i < kTokenBatch; i++) {
                        const unsigned token_idx = wid + i * 4;
                        tokens[i] =
                            sorted_token_br_
                                .template LoadU32<BufferResource::kNone>(
                                    0, token_idx * sizeof(unsigned)) &
                            0x00ffffffu;
                    }

                    uint2 safe_token_select = token_select;
                    safe_token_select.x = (safe_token_select.x < num_tokens)
                                              ? safe_token_select.x
                                              : 0;
                    safe_token_select.y = (safe_token_select.y < num_tokens)
                                              ? safe_token_select.y
                                              : 0;

                    unsigned safe_tokens[kTokenBatch];
                    for (int i = 0; i < kTokenBatch; i++) {
                        safe_tokens[i] =
                            (tokens[i] < num_tokens) ? tokens[i] : 0;
                    }

                    const float2 sorted_weights =
                        LoadSortedWeights(sorted_weights_ptr, tid, route_base);
                    float4 h[Stage1Trait::kAccumFragments];
                    Stage1(h, shm, wid, wtid, tid, safe_token_select,
                           safe_tokens, m);
                    __syncthreads();

                    unsigned invalid_token_mask = 0;
                    for (int i = 0; i < kTokenBatch; i++) {
                        invalid_token_mask |= (tokens[i] >= num_tokens) << i;
                    }
                    invalid_token_mask =
                        __builtin_amdgcn_readfirstlane(invalid_token_mask);

                    uint4 quant_h[QuantizeAndShuffleOp::kElementsPerThreadVec4];
                    float4 dq_act;
                    QuantizeAndShuffle(quant_h, &dq_act, shm, tid, wid, wtid,
                                       h);

                    Stage2(out, shm, quant_h, dq_act, sorted_weights,
                           safe_tokens, invalid_token_mask, tid, wid, wtid);
                }
            }
            __syncthreads();
        }
    }

    unsigned dim_;
    unsigned inter_dim_;
    Input input_;
    BufferResource sorted_token_br_;
    W2 w2_;
    W13 w1_, w3_;
};

template <class Config, class Kernel>
__global__ static void __launch_bounds__(64 * Config::kNumWarps)
    OnestageFusedMoEBlockScaleFP8Compute(
        uint4 *__restrict__ out, const uint4 *act, const uint4 *w13,
        const uint4 *w2, const uint4 *sorted_token_ids,
        const uint4 *sorted_weights, const uint4 *sorted_expert_ids,
        const unsigned *__restrict__ num_valid_ids, unsigned topk,
        const uint4 *scales_act, const uint4 *scales_w13,
        const unsigned *__restrict__ scales_w2, unsigned m, unsigned n,
        unsigned k, unsigned num_experts, unsigned persistent_route_step) {
    Kernel kernel;
    kernel.Compute(
        out, act, w13, w2, reinterpret_cast<const unsigned *>(sorted_token_ids),
        reinterpret_cast<const unsigned *>(sorted_weights),
        reinterpret_cast<const unsigned *>(sorted_expert_ids), scales_act,
        reinterpret_cast<const unsigned *>(scales_w13), scales_w2,
        num_valid_ids, topk, m, n, k, num_experts, persistent_route_step);
}

template <class Config, class Kernel>
void LaunchOnestageFusedMoEBlockScaleFP8(
    uint4 *__restrict__ out, const uint4 *act, const uint4 *w13,
    const uint4 *w2, const uint4 *sorted_token_ids, const uint4 *sorted_weights,
    const uint4 *sorted_expert_ids, const unsigned *__restrict__ num_valid_ids,
    unsigned topk, const uint4 *scales_act, const uint4 *scales_w13,
    const unsigned *__restrict__ scales_w2, unsigned max_num_m_blocks,
    unsigned m, unsigned n, unsigned k, unsigned num_experts,
    hipStream_t stream, unsigned num_persistent_tgs) {
    if (m == 0 || n == 0 || k == 0 || topk == 0 || max_num_m_blocks == 0) {
        return;
    }
    const unsigned split_k = tal::CeilingDiv<unsigned>(k, Config::kGroupDim);
    unsigned route_groups = max_num_m_blocks;
    unsigned persistent_route_step = 0;
    if (num_persistent_tgs > 0) {
        unsigned persistent_route_groups =
            tal::CeilingDiv<unsigned>(num_persistent_tgs, split_k);
        if (persistent_route_groups == 0) {
            persistent_route_groups = 1;
        }
        route_groups = persistent_route_groups < route_groups
                           ? persistent_route_groups
                           : route_groups;
        persistent_route_step = route_groups;
    }
    dim3 blocks(64 * Config::kNumWarps);
    dim3 grids(split_k, route_groups, 1);
    OnestageFusedMoEBlockScaleFP8Compute<Config, Kernel>
        <<<grids, blocks, 0, stream>>>(
            out, act, w13, w2, sorted_token_ids, sorted_weights,
            sorted_expert_ids, num_valid_ids, topk, scales_act, scales_w13,
            scales_w2, m, n, k, num_experts, persistent_route_step);
}

} // namespace causalflow::petit::rocm::moe
