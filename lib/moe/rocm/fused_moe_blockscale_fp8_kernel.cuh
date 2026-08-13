#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "moe/rocm/fused_moe.h"
#include "moe/rocm/memory_ops.cuh"

namespace causalflow::petit::rocm::moe {

template <FusedMoEMfmaShape> struct RouteWeightsLayout {
    using Type = float2;

    __device__ static Type Load(BufferResource br, unsigned tid) {
        uint2 u;
        u.x = br.template LoadU32<BufferResource::kNone>(
            (tid % 16) * sizeof(unsigned), 0);
        u.y = br.template LoadU32<BufferResource::kNone>(
            (tid % 16 + 16) * sizeof(unsigned), 0);
        return reinterpret_cast<const float2 &>(u);
    }
};

template <class Config>
struct OnestageFusedMoEBlockScaleFP8 {
    static constexpr unsigned kGroupM = Config::kGroupM;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kThreads = kNumWarps * 64;
    static constexpr unsigned kScaleBlockSize = 128;
    static constexpr unsigned kTokenBatch = Config::kTokenBatch;
    static constexpr unsigned kRoutesPerBlock = kTokenBatch * kNumWarps;
    using Input = typename Config::Input;
    using W13Weights = typename Config::W13Weights;
    using W2Weights = typename Config::W2Weights;
    using Bias = typename Config::Bias;
    using QuantizeAndShuffleOp = typename Config::QuantizeAndShuffleOp;
    using Stage1Tiles = typename Config::Stage1Tiles;
    using Stage1Op = typename Config::Stage1Op;
    using Stage2Tiles = typename Config::Stage2Tiles;
    using Stage2Op = typename Config::Stage2Op;
    using RouteWeightLayout = RouteWeightsLayout<Config::kMfmaShape>;
    using RouteWeights = typename RouteWeightLayout::Type;
    using TokenMetadata = unsigned[kRoutesPerBlock];

    static_assert(kRoutesPerBlock == 32, "token metadata cache expects M32");
    static_assert(sizeof(TokenMetadata) == 32 * sizeof(unsigned));

    struct ShmBuf {
        typename Stage1Op::Shm x;
        typename QuantizeAndShuffleOp::Shm stage2_input;
        typename Stage2Op::Shm ret;
        TokenMetadata row_metadata;
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

    __device__ RouteWeights LoadSortedWeights(const unsigned *sorted_weights,
                                              unsigned tid,
                                              unsigned route_base) const {
        auto br = MakeBufferResource(sorted_weights + route_base,
                                     kRoutesPerBlock * sizeof(unsigned));
        return RouteWeightLayout::Load(br, tid);
    }

    __device__ static void
    PrefetchTokenMetadata(TokenMetadata &row_metadata,
                          const unsigned *sorted_token_ids, unsigned route_base,
                          unsigned wid, unsigned wtid) {
        if (wtid < kTokenBatch) {
            const unsigned row = wid * kTokenBatch + wtid;
            const auto metadata = MakeBufferResource(
                sorted_token_ids + route_base,
                kRoutesPerBlock * sizeof(unsigned));
            row_metadata[row] =
                metadata.template LoadU32<BufferResource::kNone>(
                    row * sizeof(unsigned), 0);
        }
        amdgcn_s_waitcnt<0, -1, 0>();
    }

    __device__ static void ReadTokens(unsigned tokens[kTokenBatch],
                                      const TokenMetadata &row_metadata,
                                      unsigned wid) {
        using SharedUnsigned = __attribute__((address_space(3))) unsigned;
        const auto *metadata = (const SharedUnsigned *)&row_metadata[0];
#pragma unroll
        for (unsigned i = 0; i < kTokenBatch; ++i) {
            tokens[i] = metadata[wid * kTokenBatch + i] & 0x00ffffffu;
        }
    }

    __device__ void Stage1(float4 h[Stage1Tiles::kAccumFragments], ShmBuf &shm,
                           unsigned wid, unsigned wtid, unsigned tid,
                           const unsigned tokens[kTokenBatch], unsigned m,
                           unsigned expert_id, unsigned tile_k,
                           const void *w13_bias) {
        Stage1Tiles tiles{input_, w13_weights_.w1_, w13_weights_.w3_,
                          w1_bias_, w3_bias_};
        tiles.InitializeBias(w13_bias, expert_id, Config::kInterDim, tile_k);
        Stage1Op::Run(h, shm.x, tiles, Config::kDim, tid, wid, wtid, tokens, m);
    }

    __device__ void
    Stage2(const BufferResource &out, ShmBuf &shm,
           const typename Stage2Tiles::InputRegs &input,
           const RouteWeights &sorted_weights,
           const unsigned tokens[kTokenBatch], unsigned tile_k, unsigned tid,
           unsigned wid, unsigned wtid, unsigned expert_id,
           const void *w2_bias) {
        Stage2Tiles tiles{w2_weights_.w2_, w2_bias_};
        tiles.InitializeBias(w2_bias, expert_id, Config::kDim, tile_k);
        Stage2Op::Run(out, shm.ret, tiles, Config::kDim, input, sorted_weights,
                      tokens, tile_k, tid, wid, wtid);
    }

    __device__ void
    Compute(uint4 *__restrict__ out, const uint4 *act, const uint4 *w13_base,
            const uint4 *w2, const unsigned *sorted_token_ids,
            const unsigned *sorted_weights_ptr,
            const unsigned *sorted_expert_ids, const uint4 *scales_act,
            const unsigned *scales_w13, const unsigned *__restrict__ scales_w2,
            const unsigned *num_valid_ids_ptr, unsigned topk, unsigned m,
            unsigned num_experts, unsigned persistent_route_step,
            const void *w13_bias, const void *w2_bias) {
        (void)topk;
        static constexpr unsigned kNBlocks = Config::kDim / kScaleBlockSize;
        static constexpr unsigned kKBlocks =
            Config::kInterDim / kScaleBlockSize;

        const unsigned tid = threadIdx.x, tile_k = blockIdx.x;
        const unsigned wid = __builtin_amdgcn_readfirstlane(tid / kWarpSize),
                       wtid = tid % kWarpSize;

        __shared__ ShmBuf shm;
        const unsigned num_valid_ids = num_valid_ids_ptr[0];
        const unsigned num_tokens = num_valid_ids_ptr[1];
        const BufferResource output = MakeBufferResource(
            out, num_tokens * Config::kDim * sizeof(__hip_bfloat16));
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
                    PrefetchTokenMetadata(shm.row_metadata, sorted_token_ids,
                                          route_base, wid, wtid);

                    if constexpr (Config::kActDType ==
                                  FusedMoEDataType::kMxFp4) {
                        input_.Initialize(act, scales_act, wid, m, kNBlocks,
                                          route_group, route_group_limit);
                    } else {
                        input_.Initialize(act, scales_act, wid, m, kNBlocks);
                    }
                    Config::InitializeWeights(
                        *this, w13_base, w2, scales_w13, scales_w2, expert_id,
                        tile_k, kNBlocks, kKBlocks);

                    unsigned tokens[kTokenBatch];
                    ReadTokens(tokens, shm.row_metadata, wid);
                    float4 h[Stage1Tiles::kAccumFragments];
                    Stage1(h, shm, wid, wtid, tid, tokens, m, expert_id,
                           tile_k, w13_bias);
                    __syncthreads();

                    typename Stage2Tiles::InputRegs stage2_input;
                    QuantizeAndShuffleOp::Run(stage2_input, shm.stage2_input, h,
                                              tid, wid, wtid);

                    const RouteWeights sorted_weights =
                        LoadSortedWeights(sorted_weights_ptr, tid, route_base);
                    Stage2(output, shm, stage2_input, sorted_weights, tokens,
                           tile_k, tid, wid, wtid, expert_id, w2_bias);
                }
                __syncthreads();
            }
        }
    }

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
        const unsigned *__restrict__ scales_w2, unsigned m,
        unsigned num_experts, unsigned persistent_route_step,
        const void *w13_bias, const void *w2_bias) {
    Kernel kernel;
    kernel.Compute(
        out, act, w13, w2, reinterpret_cast<const unsigned *>(sorted_token_ids),
        reinterpret_cast<const unsigned *>(sorted_weights),
        reinterpret_cast<const unsigned *>(sorted_expert_ids), scales_act,
        reinterpret_cast<const unsigned *>(scales_w13), scales_w2,
        num_valid_ids, topk, m, num_experts, persistent_route_step, w13_bias,
        w2_bias);
}

} // namespace causalflow::petit::rocm::moe
