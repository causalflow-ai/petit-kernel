#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "moe/rocm/memory_ops.cuh"

namespace causalflow::petit::rocm::moe {

template <class Config>
struct OnestageFusedMoEBlockScaleFP8 {
    static constexpr unsigned kGroupM = Config::kGroupM;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kThreads = kNumWarps * 64;
    static constexpr unsigned kScaleBlockSize = 128;
    static constexpr unsigned kTokenBatch = Config::kTokenBatch;
    static constexpr unsigned kSubGroupSize = 16;
    static constexpr unsigned kRoutesPerBlock = kTokenBatch * kNumWarps;
    static constexpr unsigned kRefBufferRange = static_cast<unsigned>(-16);

    using Input = typename Config::Input;
    using W13Weights = typename Config::W13Weights;
    using W2Weights = typename Config::W2Weights;
    using Bias = typename Config::Bias;
    using QuantizeAndShuffleOp = typename Config::QuantizeAndShuffleOp;
    using Stage1Trait = typename Config::Stage1Trait;
    using Stage1Op = typename Config::Stage1Op;
    using Stage2Trait = typename Config::Stage2Trait;
    using Stage2Op = typename Config::Stage2Op;

    struct ShmBuf {
        typename Stage1Op::Shm x;
        typename QuantizeAndShuffleOp::Shm stage2_input;
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

    __device__ void Stage1(float4 h[Stage1Trait::kAccumFragments], ShmBuf &shm,
                           unsigned wid, unsigned wtid, unsigned tid,
                           const uint2 token_select,
                           const unsigned tokens[kTokenBatch], unsigned m,
                           unsigned expert_id, unsigned tile_k,
                           const void *w13_bias) {
        Stage1Trait trait{input_, w13_weights_.w1_, w13_weights_.w3_,
                          w1_bias_, w3_bias_};
        trait.InitializeBias(w13_bias, expert_id, inter_dim_, tile_k);
        Stage1Op::Run(h, shm.x, trait, dim_, tid, wid, wtid, token_select,
                      tokens, m);
    }

    __device__ void
    Stage2(uint4 *__restrict__ out, ShmBuf &shm,
           const typename Stage2Trait::InputRegs &input, float2 sorted_weights,
           const unsigned tokens[kTokenBatch], unsigned invalid_token_mask,
           unsigned tile_k, unsigned tid, unsigned wid, unsigned wtid,
           unsigned expert_id, const void *w2_bias) {
        Stage2Trait trait{w2_weights_.w2_, w2_bias_};
        trait.InitializeBias(w2_bias, expert_id, dim_, tile_k);
        Stage2Op::Run(out, shm.ret, trait, dim_, input, sorted_weights, tokens,
                      invalid_token_mask, tile_k, tid, wid, wtid);
    }

    __device__ void
    Compute(uint4 *__restrict__ out, const uint4 *act, const uint4 *w13_base,
            const uint4 *w2, const unsigned *sorted_token_ids,
            const unsigned *sorted_weights_ptr,
            const unsigned *sorted_expert_ids, const uint4 *scales_act,
            const unsigned *scales_w13, const unsigned *__restrict__ scales_w2,
            const unsigned *num_valid_ids_ptr, unsigned topk, unsigned m,
            unsigned dim, unsigned inter_dim, unsigned num_experts,
            unsigned persistent_route_step, const void *w13_bias,
            const void *w2_bias) {
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
        if (route_group_begin < route_group_limit) {
            for (unsigned route_group = route_group_begin;
                 route_group < route_group_end;
                 route_group += (route_group_step ? route_group_step : 1)) {
                const unsigned route_base = route_group * kRoutesPerBlock;
                const unsigned expert_id = sorted_expert_ids[route_group];
                bool valid_expert = true;
                if constexpr (Config::kValidateExpertIds) {
                    valid_expert = expert_id < num_experts;
                }

                if (valid_expert) {
                    input_.Initialize(act, scales_act, wid, m, n_blocks, dim_);

                    Config::InitializeWeights(
                        *this, w13_base, w2, scales_w13, scales_w2, expert_id,
                        tile_k, n_blocks, k_blocks);

                    uint2 token_select;
                    token_select.x =
                        sorted_token_ids[route_base + col_id] & 0x00ffffffu;
                    token_select.y =
                        sorted_token_ids[route_base + col_id + kSubGroupSize] &
                        0x00ffffffu;

                    unsigned tokens[kTokenBatch];
#pragma unroll
                    for (int i = 0; i < kTokenBatch; i++) {
                        const unsigned token_idx = wid + i * kNumWarps;
                        tokens[i] =
                            sorted_token_ids[route_base + token_idx] &
                            0x00ffffffu;
                    }
                    unsigned invalid_token_mask = 0;
#pragma unroll
                    for (int i = 0; i < kTokenBatch; i++) {
                        invalid_token_mask |=
                            (tokens[i] >= num_tokens) << i;
                    }
                    invalid_token_mask =
                        __builtin_amdgcn_readfirstlane(invalid_token_mask);

                    const float2 sorted_weights =
                        LoadSortedWeights(sorted_weights_ptr, tid, route_base);
                    float4 h[Stage1Trait::kAccumFragments];
                    Stage1(h, shm, wid, wtid, tid, token_select, tokens, m,
                           expert_id, tile_k, w13_bias);
                    __syncthreads();

                    typename Stage2Trait::InputRegs stage2_input;
                    QuantizeAndShuffleOp::Run(stage2_input, shm.stage2_input, h,
                                              tid, wid, wtid);

                    Stage2(out, shm, stage2_input, sorted_weights, tokens,
                           invalid_token_mask, tile_k, tid, wid, wtid,
                           expert_id, w2_bias);
                }
                __syncthreads();
            }
        }
    }

    unsigned dim_;
    unsigned inter_dim_;
    Input input_;
    W13Weights w13_weights_;
    Bias w1_bias_, w3_bias_;
    W2Weights w2_weights_;
    Bias w2_bias_;
};

template <class Kernel>
__global__ static void __launch_bounds__(64 * Kernel::kNumWarps)
    OnestageFusedMoEBlockScaleFP8Compute(
        uint4 *__restrict__ out, const uint4 *act, const uint4 *w13,
        const uint4 *w2, const uint4 *sorted_token_ids,
        const uint4 *sorted_weights, const uint4 *sorted_expert_ids,
        const unsigned *__restrict__ num_valid_ids, unsigned topk,
        const uint4 *scales_act, const uint4 *scales_w13,
        const unsigned *__restrict__ scales_w2, unsigned m, unsigned n,
        unsigned k, unsigned num_experts, unsigned persistent_route_step,
        const void *w13_bias, const void *w2_bias) {
    Kernel kernel;
    kernel.Compute(
        out, act, w13, w2, reinterpret_cast<const unsigned *>(sorted_token_ids),
        reinterpret_cast<const unsigned *>(sorted_weights),
        reinterpret_cast<const unsigned *>(sorted_expert_ids), scales_act,
        reinterpret_cast<const unsigned *>(scales_w13), scales_w2,
        num_valid_ids, topk, m, n, k, num_experts, persistent_route_step,
        w13_bias, w2_bias);
}

} // namespace causalflow::petit::rocm::moe
