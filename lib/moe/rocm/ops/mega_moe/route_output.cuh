#pragma once

#include "moe/rocm/mega_moe/workspace.cuh"
#include "moe/rocm/ops/op_stages.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <class TileSchedule>
struct MegaMoETwoStage2Epilogue : TwoStageStage2Epilogue<TileSchedule> {
    using Base = TwoStageStage2Epilogue<TileSchedule>;
    using Config = typename TileSchedule::Config;
    using Workspace = MegaMoEWorkspace<Config>;
    using typename Base::BiasPrefetch;
    using typename Base::Shm;
    using Base::Apply;
    using Base::PrefetchBias;
    using Base::WriteShm;

    struct Context {
        Workspace *workspace = nullptr;
        unsigned pool_base = 0;
        unsigned work_m = 0;
    };

    TAL_DEVICE static float2 LoadRouteWeights(const Context &context,
                                              unsigned wtid) {
        const unsigned offset =
            context.workspace->L1TokenWeightsOffset(context.pool_base);
        const unsigned lane = wtid % 16;
        const uint2 packed{
            context.workspace->br_.template LoadU32<BufferResource::kNone>(
                lane * sizeof(unsigned), offset),
            context.workspace->br_.template LoadU32<BufferResource::kNone>(
                (lane + 16) * sizeof(unsigned), offset),
        };
        return __builtin_bit_cast(float2, packed);
    }

    TAL_DEVICE static void WriteBack(const Context &context, Shm &shm,
                                     unsigned tile_col, unsigned wid,
                                     unsigned wtid) {
        const auto *output = reinterpret_cast<const unsigned *>(shm.output);
#pragma unroll
        for (unsigned row_group = 0; row_group < Base::kTileRows /
                                                        Base::kNumWarps;
             ++row_group) {
            const unsigned row = wid + row_group * Base::kNumWarps;
            const bool valid = row < context.work_m;
            TokenMetadata metadata{};
            if (valid) {
                metadata = __builtin_bit_cast(
                    TokenMetadata,
                    context.workspace->br_
                        .template LoadU64<BufferResource::kNone>(
                            0, context.workspace->TokenMetadataOffset(
                                   context.pool_base + row)));
            }
            const unsigned src_rank =
                __builtin_amdgcn_readfirstlane(metadata.src_rank);
            const unsigned token_topk_idx =
                __builtin_amdgcn_readfirstlane(metadata.token_topk_idx);
            const unsigned row_bytes =
                Config::kHiddenSize * sizeof(__hip_bfloat16);
            BufferResource output_row = context.workspace->br_;
            output_row.v.ptr +=
                context.workspace->RouteOutputBufferOffset(src_rank) +
                token_topk_idx * row_bytes;
            // A zero-sized descriptor discards invalid rows. A valid
            // descriptor is bounded to one logical-hidden output row, so the
            // padded columns in the final compute tile are also discarded.
            output_row.v.range = valid ? row_bytes : 0;
#pragma unroll
            for (unsigned col_half = 0; col_half < 2; ++col_half) {
                const unsigned pair_col = col_half * kWarpSize + wtid;
                const unsigned value =
                    output[row * (Base::kTileCols / 2) + pair_col];
                const unsigned col = tile_col + pair_col * 2;
                output_row.template StoreU32<BufferResource::kSC1Bit>(
                    col * sizeof(__hip_bfloat16), 0, value);
            }
        }
    }
};

// Reduce the source-owned (token, top-k) route rows in FP32, return BF16,
// and clear every row for the next invocation.
template <class Config> struct SourceRouteReducer {
    using Workspace = MegaMoEWorkspace<Config>;

    static constexpr unsigned kNumSMs = Config::kNumSMs;
    static constexpr unsigned kThreads = Config::kThreads;
    static constexpr unsigned kTopK = Config::kTopK;
    static constexpr unsigned kHiddenSize = Config::kHiddenSize;
    static constexpr unsigned kComputeHiddenSize = Config::kComputeHiddenSize;
    static constexpr unsigned kElementsPerVec =
        sizeof(uint4) / sizeof(__hip_bfloat16);
    static constexpr unsigned kVecCols = kHiddenSize / kElementsPerVec;

    static_assert(kThreads % kWarpSize == 0);
    static_assert(kHiddenSize % kElementsPerVec == 0);
    static_assert(kComputeHiddenSize >= kHiddenSize);

    TAL_DEVICE explicit SourceRouteReducer(Workspace *workspace)
        : workspace_(workspace) {}

    TAL_DEVICE void Run(uint4 *__restrict__ output, unsigned num_tokens,
                        unsigned output_row_stride, unsigned sm_id,
                        unsigned wid, unsigned wtid) const {
        static constexpr unsigned kWarpsPerBlock = kThreads / kWarpSize;
        static constexpr unsigned kWavesPerToken =
            (kVecCols + kWarpSize - 1) / kWarpSize;
        static constexpr unsigned kTotalWaves = kNumSMs * kWarpsPerBlock;
        static constexpr unsigned kPackedElements = kElementsPerVec / 2;

        const unsigned global_wave = sm_id * kWarpsPerBlock + wid;
        const unsigned total_wave_tasks = num_tokens * kWavesPerToken;
        const unsigned owner =
            workspace_->RouteOutputBufferOffset(workspace_->Rank());
        for (unsigned wave_task = global_wave;
             wave_task < total_wave_tasks; wave_task += kTotalWaves) {
            const unsigned token = wave_task / kWavesPerToken;
            const unsigned wave_in_token = wave_task % kWavesPerToken;
            for (unsigned vec_col = wave_in_token * kWarpSize + wtid;
                 vec_col < kVecCols;
                 vec_col += kWavesPerToken * kWarpSize) {
                const unsigned col_offset = vec_col * sizeof(uint4);
                const unsigned route_row_offset =
                    owner + token * kTopK * kHiddenSize *
                                sizeof(__hip_bfloat16);
                float2 accum[kPackedElements] = {};
#pragma unroll
                for (unsigned topk = 0; topk < kTopK; ++topk) {
                    const unsigned row_offset =
                        route_row_offset +
                        topk * kHiddenSize * sizeof(__hip_bfloat16);
                    const uint4 value = workspace_->br_.template Load<
                        BufferResource::kNone>(col_offset, row_offset);
                    workspace_->br_.template Store<BufferResource::kNone>(
                        col_offset, row_offset, uint4{});
                    const auto *bf16 =
                        reinterpret_cast<const __hip_bfloat162 *>(&value);
#pragma unroll
                    for (unsigned pair = 0; pair < kPackedElements; ++pair) {
                        accum[pair] = amdgcn_pk_add_f32(
                            accum[pair], __bfloat1622float2(bf16[pair]));
                    }
                }

                uint4 packed_output;
                auto *bf16 =
                    reinterpret_cast<__hip_bfloat162 *>(&packed_output);
#pragma unroll
                for (unsigned pair = 0; pair < kPackedElements; ++pair)
                    bf16[pair] = __float22bfloat162_rn(accum[pair]);
                output[token * (output_row_stride / kElementsPerVec) + vec_col] =
                    packed_output;
            }
        }
    }

  private:
    Workspace *workspace_;
};

} // namespace causalflow::petit::rocm::moe
