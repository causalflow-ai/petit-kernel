#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "moe/rocm/fused_moe.h"
#include "moe/rocm/memory_ops.cuh"

namespace causalflow::petit::rocm::moe {

struct RouteWeightsLayout {
    using Type = float2;
    static constexpr unsigned kRoutesPerGroup = 32;

    __device__ static Type Load(const unsigned *sorted_weights,
                                unsigned route_base, unsigned tid) {
        const auto weights = MakeBufferResource(
            sorted_weights + route_base,
            kRoutesPerGroup * sizeof(unsigned));
        uint2 u;
        u.x = weights.template LoadU32<BufferResource::kNone>(
            (tid % 16) * sizeof(unsigned), 0);
        u.y = weights.template LoadU32<BufferResource::kNone>(
            (tid % 16 + 16) * sizeof(unsigned), 0);
        return reinterpret_cast<const float2 &>(u);
    }
};

template <class Config> struct OnestageFusedMoEStage1Epilogue {
    using QuantizeAndShuffleOp = typename Config::QuantizeAndShuffleOp;
    using W2Weights = typename Config::W2Weights;
    using Bias = typename Config::Bias;
    using Stage2Tiles = typename Config::Stage2Tiles;
    using Stage2Op = typename Config::Stage2Op;

    struct Shm {
        typename QuantizeAndShuffleOp::Shm stage2_input;
        typename Stage2Op::Shm output;
    };

    struct Context {
        BufferResource output;
        const uint4 *w2;
        const unsigned *scales_w2;
        const unsigned *sorted_weights;
        const void *w2_bias;
        unsigned n_blocks;
        unsigned k_blocks;
    };

    template <class Kernel>
    __device__ void
    Run(Kernel &kernel, Context &context, Shm &shm,
        float4 hidden[Kernel::Stage1Tiles::kAccumFragments],
        const unsigned tokens[Kernel::kTokenBatch],
        const typename Kernel::TokenMetadata &, unsigned route_base,
        unsigned, unsigned expert_id, unsigned tile_k, unsigned tid,
        unsigned wid, unsigned wtid) {
        Config::InitializeW2(*this, context.w2, context.scales_w2, expert_id,
                             tile_k, context.n_blocks, context.k_blocks);
        typename Stage2Tiles::InputRegs stage2_input;
        QuantizeAndShuffleOp::Run(stage2_input, shm.stage2_input, hidden, tid,
                                  wid, wtid);
        const auto route_weights =
            RouteWeightsLayout::Load(context.sorted_weights, route_base, tid);
        Stage2Tiles tiles{w2_weights_.w2_, w2_bias_};
        tiles.InitializeBias(context.w2_bias, expert_id, tile_k);
        Stage2Op::Run(context.output, shm.output, tiles, stage2_input,
                      route_weights, tokens, tile_k, tid, wid, wtid);
    }

    W2Weights w2_weights_;
    Bias w2_bias_;
};

template <class Config_, class Epilogue_> struct FusedMoEStage1 {
    using Config = Config_;
    using Epilogue = Epilogue_;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kThreads = kNumWarps * 64;
    static constexpr unsigned kScaleBlockSize = 128;
    static constexpr unsigned kTokenBatch = Config::kTokenBatch;
    static constexpr unsigned kRoutesPerBlock = kTokenBatch * kNumWarps;

    using Input = typename Config::Input;
    using W13Weights = typename Config::W13Weights;
    using Bias = typename Config::Bias;
    using Stage1Tiles = typename Config::Stage1Tiles;
    using Stage1Op = typename Config::Stage1Op;
    using TokenMetadata = unsigned[kThreads];

    static_assert(kRoutesPerBlock == 32 || kRoutesPerBlock == 64,
                  "token metadata cache expects M32 or M64");

    struct ShmBuf {
        union {
            typename Stage1Op::Shm stage1;
            typename Epilogue::Shm epilogue;
        } data;
        TokenMetadata row_metadata;
    };

    __device__ static void
    PrefetchTokenMetadata(TokenMetadata &row_metadata,
                          const unsigned *sorted_token_ids, unsigned route_base,
                          unsigned wid, unsigned wtid) {
        const unsigned row = (wtid % kTokenBatch) + wid * kTokenBatch +
                             (wtid / kTokenBatch) * kRoutesPerBlock;
        const auto metadata = MakeBufferResource(
            sorted_token_ids + route_base,
            kRoutesPerBlock * sizeof(unsigned));
        row_metadata[row] =
            metadata.template LoadU32<BufferResource::kNone>(
                row * sizeof(unsigned), 0);
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

    __device__ void Stage1(float4 h[Stage1Tiles::kAccumFragments],
                           typename Stage1Op::Shm &stage1_shm,
                           unsigned wid, unsigned wtid, unsigned tid,
                           const unsigned tokens[kTokenBatch], unsigned m,
                           unsigned expert_id, unsigned tile_k,
                           const void *w13_bias) {
        Stage1Tiles tiles{input_, w13_weights_.w1_, w13_bias_};
        tiles.InitializeBias(w13_bias, expert_id, tile_k);
        Stage1Op::Run(h, stage1_shm, tiles, tid, wid, wtid, tokens, m);
    }

    __device__ void
    RunStage1Route(ShmBuf &shm, typename Epilogue::Context &epilogue,
                   const uint4 *act, const uint4 *w13,
                   const unsigned *sorted_token_ids,
                   const uint4 *scales_act, const unsigned *scales_w13,
                   unsigned route_group, unsigned route_group_limit,
                   unsigned expert_id, unsigned tile_k, unsigned m,
                   unsigned num_experts, unsigned tid, unsigned wid,
                   unsigned wtid, const void *w13_bias) {
        bool valid_expert = true;
        if constexpr (Config::kValidateExpertIds)
            valid_expert = expert_id < num_experts;
        if (!valid_expert)
            return;

        const unsigned route_base = route_group * kRoutesPerBlock;
        PrefetchTokenMetadata(shm.row_metadata, sorted_token_ids, route_base,
                              wid, wtid);
        if constexpr (Config::kActDType == FusedMoEDataType::kMxFp4) {
            input_.Initialize(act, scales_act, wid, m,
                              Config::kDim / kScaleBlockSize, route_group,
                              route_group_limit);
        } else {
            input_.Initialize(act, scales_act, wid, m,
                              Config::kDim / kScaleBlockSize);
        }
        Config::InitializeW13(*this, w13, scales_w13, expert_id, tile_k,
                              Config::kDim / kScaleBlockSize,
                              Config::kInterDim / kScaleBlockSize);

        unsigned tokens[kTokenBatch];
        ReadTokens(tokens, shm.row_metadata, wid);
        float4 hidden[Stage1Tiles::kAccumFragments];
        Stage1(hidden, shm.data.stage1, wid, wtid, tid, tokens, m, expert_id,
               tile_k, w13_bias);
        __syncthreads();
        epilogue_.Run(*this, epilogue, shm.data.epilogue, hidden, tokens,
                      shm.row_metadata, route_base, route_group, expert_id,
                      tile_k, tid, wid, wtid);
    }

    __device__ void
    Compute(const uint4 *act, const uint4 *w13,
            const unsigned *sorted_token_ids,
            const unsigned *sorted_expert_ids, const uint4 *scales_act,
            const unsigned *scales_w13,
            const unsigned *num_valid_ids_ptr, unsigned m,
            unsigned num_experts, unsigned persistent_route_step,
            const void *w13_bias, typename Epilogue::Context &epilogue) {
        const unsigned tid = threadIdx.x, tile_k = blockIdx.x;
        const unsigned wid = __builtin_amdgcn_readfirstlane(tid / kWarpSize),
                       wtid = tid % kWarpSize;

        __shared__ ShmBuf shm;
        const unsigned num_valid_ids = num_valid_ids_ptr[0];
        const unsigned route_group_limit =
            tal::CeilingDiv<unsigned>(num_valid_ids, kRoutesPerBlock);

        const unsigned route_group_begin = blockIdx.y;
        const unsigned route_group_step = persistent_route_step;
        const unsigned route_group_end =
            route_group_step ? route_group_limit : route_group_begin + 1;
        if (route_group_begin < route_group_limit) {
            for (unsigned route_group = route_group_begin;
                 route_group < route_group_end;
                 route_group += (route_group_step ? route_group_step : 1)) {
                const unsigned expert_id = sorted_expert_ids[route_group];
                RunStage1Route(shm, epilogue, act, w13, sorted_token_ids,
                               scales_act, scales_w13, route_group,
                               route_group_limit, expert_id, tile_k, m,
                               num_experts, tid, wid, wtid, w13_bias);
                __syncthreads();
            }
        }
    }

    Input input_;
    W13Weights w13_weights_;
    Bias w13_bias_;
    Epilogue epilogue_;
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
    (void)topk;
    using Config = typename Kernel::Config;
#if !defined(__gfx950__)
    if constexpr (Config::kSolution.weight_ordering ==
                  FusedMoEWeightOrdering::kNativeMxFp4)
        return;
#endif
    static constexpr unsigned kNBlocks =
        Config::kDim / Kernel::kScaleBlockSize;
    static constexpr unsigned kKBlocks =
        Config::kInterDim / Kernel::kScaleBlockSize;
    Kernel kernel;
    typename Kernel::Epilogue::Context epilogue{
        MakeBufferResource(
            out, num_valid_ids[1] * Config::kDim * sizeof(__hip_bfloat16)),
        w2,
        scales_w2,
        reinterpret_cast<const unsigned *>(sorted_weights),
        w2_bias,
        kNBlocks,
        kKBlocks,
    };
    kernel.Compute(act, w13,
                   reinterpret_cast<const unsigned *>(sorted_token_ids),
                   reinterpret_cast<const unsigned *>(sorted_expert_ids),
                   scales_act,
                   reinterpret_cast<const unsigned *>(scales_w13),
                   num_valid_ids, m, num_experts, persistent_route_step,
                   w13_bias, epilogue);
}

} // namespace causalflow::petit::rocm::moe
