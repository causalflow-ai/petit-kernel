#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "moe/rocm/fused_moe_blockscale_fp8_kernel.cuh"
#include "moe/rocm/memory_ops.cuh"
#include "moe/rocm/ops/op_stages.cuh"
#include "moe/rocm/ops/schedule_tiles.cuh"
#include "moe/rocm/ops/stage1_accumulator_lds.cuh"
#include "moe/rocm/quantization.cuh"

namespace causalflow::petit::rocm::moe {

template <class Config> struct TwoStageFusedMoEWorkspace {
    static constexpr unsigned kRoutesPerBlock = Config::kGroupM;

    __host__ __device__ static constexpr unsigned
    PayloadBytes(unsigned route_groups, unsigned inter_dim) {
        return route_groups * kRoutesPerBlock * inter_dim / 2;
    }

    __host__ __device__ static constexpr unsigned
    ScaleRows(unsigned route_groups) {
        return tal::CeilingDiv<unsigned>(route_groups * kRoutesPerBlock, 256) *
               256;
    }

    __host__ __device__ static constexpr unsigned
    ScaleCols(unsigned inter_dim) {
        return tal::CeilingDiv<unsigned>(inter_dim / 32, 8) * 8;
    }

    __host__ __device__ static constexpr unsigned
    ScaleOffset(unsigned route_groups, unsigned inter_dim) {
        return PayloadBytes(route_groups, inter_dim);
    }

    __host__ __device__ static constexpr unsigned
    Bytes(unsigned route_groups, unsigned inter_dim) {
        return ScaleOffset(route_groups, inter_dim) +
               ScaleRows(route_groups) * ScaleCols(inter_dim);
    }
};

template <class Config> struct MxFp4Stage1WorkspaceEpilogue {
    static constexpr unsigned kGroupM = 32;
    static constexpr unsigned kGroupN = 128;
    static constexpr unsigned kInputFragments =
        (kGroupM * kGroupN) / (Config::kNumWarps * kWarpSize) / 4;
    using Shm = float[kGroupM * kGroupN];

    struct Context {
        unsigned short *__restrict__ payload;
        BufferResource scale;
        unsigned num_valid_ids;
        unsigned m;
    };

    template <class Kernel>
    __device__ void
    Run(Kernel &, Context &context, Shm &shm,
        float4 h[kInputFragments],
        const unsigned[Kernel::kTokenBatch],
        const typename Kernel::TokenMetadata &row_metadata,
        unsigned, unsigned route_group, unsigned, unsigned tile_n,
        unsigned tid, unsigned wid, unsigned wtid) {
        StoreStage1AccumulatorLds<Config::kNumWarps, kGroupN>(shm, h, wid,
                                                              wtid);
        __syncthreads();

        const unsigned route_in_slice = tid / 32;
        const unsigned col_lane = tid % 32;
        const auto *metadata =
            reinterpret_cast<const unsigned *>(&row_metadata[0]);
#pragma unroll
        for (unsigned route_slice = 0; route_slice < 4; ++route_slice) {
            const unsigned row = route_slice * 8 + route_in_slice;
            const unsigned sorted_row = route_group * Config::kGroupM + row;
            const unsigned fused = metadata[row];
            const unsigned token = fused & 0x00ffffffu;
            const unsigned slot = fused >> 24;
            const bool valid = sorted_row < context.num_valid_ids &&
                               token < context.m && slot < Config::kTopK;
            if (valid) {
                const unsigned col_local = col_lane * 4;
                const auto *value = reinterpret_cast<const float4 *>(
                    &shm[row * kGroupN + col_local]);
                float max_abs = AiterMxFp4Quantization::MaximumAbs(*value);
                max_abs = ReduceMaximum<0x041f>(max_abs);
                max_abs = ReduceMaximum<0x081f>(max_abs);
                max_abs = ReduceMaximum<0x101f>(max_abs);

                unsigned short packed;
                const unsigned scale_byte =
                    QuantizeMxFp4<AiterMxFp4Quantization, 1>(
                        reinterpret_cast<unsigned char *>(&packed), value,
                        max_abs);
                const unsigned payload_row = token * Config::kTopK + slot;
                const unsigned payload_offset =
                    payload_row * (Config::kInterDim / 2) +
                    tile_n * (kGroupN / 2) + col_local / 2;
                const unsigned partner = __builtin_amdgcn_mov_dpp(
                    static_cast<unsigned>(packed), 0xb1, 0xf, 0xf, false);
                if ((col_lane & 1u) == 0) {
                    const unsigned packed_values =
                        static_cast<unsigned>(packed) | (partner << 16);
                    __builtin_nontemporal_store(
                        packed_values,
                        reinterpret_cast<unsigned *>(context.payload) +
                            payload_offset / sizeof(unsigned));
                }

                const unsigned scale0 = __shfl(scale_byte, 0, 32);
                const unsigned scale1 = __shfl(scale_byte, 8, 32);
                const unsigned scale2 = __shfl(scale_byte, 16, 32);
                const unsigned scale3 = __shfl(scale_byte, 24, 32);
                if (col_lane == 0) {
                    const unsigned packed_scales =
                        scale0 | (scale1 << 8) | (scale2 << 16) |
                        (scale3 << 24);
                    const unsigned scale_cols =
                        TwoStageFusedMoEWorkspace<Config>::ScaleCols(
                            Config::kInterDim);
                    context.scale.template StoreU32<BufferResource::kNone>(
                        sorted_row * scale_cols + tile_n * 4, 0,
                        packed_scales);
                }
            }
        }
    }

  private:
    template <unsigned kPattern>
    __device__ static float ReduceMaximum(float value) {
        const unsigned bits = reinterpret_cast<const unsigned &>(value);
        const unsigned peer_bits = __builtin_amdgcn_ds_swizzle(bits, kPattern);
        const float peer = reinterpret_cast<const float &>(peer_bits);
        return __builtin_elementwise_maximum(value, peer);
    }

};

// Stage-2 input policy for reconstructing one K tile from the workspace.
// The kernel owns scheduling; this type owns only the workspace layout and
// the global-to-LDS-to-register transfer.
template <class Config, class Input> struct MxFp4Stage2Input {
    using Workspace = TwoStageFusedMoEWorkspace<Config>;

    static constexpr unsigned kRowsPerTile = Config::kGroupM;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kVectorsPerRow = (kGroupDim / 2) / sizeof(uint4);
    static constexpr unsigned kScaleCols =
        Workspace::ScaleCols(Config::kInterDim);
    static constexpr unsigned kLdsStages = 2;
    static constexpr unsigned kLdsVectorsPerRow = 16;
    static constexpr unsigned kInterDim = Config::kInterDim;
    static constexpr unsigned kK256Tiles = kInterDim / kGroupDim;

    using Shm = uint4[kLdsStages][kRowsPerTile][kLdsVectorsPerRow];

    struct Prefetch {
        uint4 value;
        unsigned scale;
    };

    struct Address {
        BufferResource values;
        BufferResource scales;
        unsigned value_voffset;
        unsigned scale_voffset;
    };

    __device__ static Address
    Initialize(const void *__restrict__ workspace_ptr,
               const BufferResource &sorted_tokens, unsigned route_group,
               unsigned max_route_groups, unsigned topk, unsigned num_tokens,
               unsigned tid, unsigned wtid) {
        const auto *values = reinterpret_cast<const uint4 *>(workspace_ptr);
        const auto *scales = reinterpret_cast<const unsigned *>(
            reinterpret_cast<const unsigned char *>(workspace_ptr) +
            Workspace::PayloadBytes(max_route_groups, kInterDim));
        const unsigned row_in_group = tid / kVectorsPerRow;
        const unsigned vector_in_row = tid % kVectorsPerRow;
        const unsigned packed_id =
            sorted_tokens.template LoadU32<BufferResource::kNone>(
                row_in_group * sizeof(unsigned), 0);
        const unsigned token = packed_id & 0x00ffffffu;
        const unsigned slot = packed_id >> 24;
        const bool valid = token < num_tokens && slot < topk;
        const unsigned value_row_bytes = kInterDim / 2;
        const unsigned values_bytes = static_cast<unsigned>(
            Workspace::PayloadBytes(max_route_groups, kInterDim));
        const unsigned value_voffset =
            valid ? (token * topk + slot) * value_row_bytes +
                        vector_in_row * sizeof(uint4)
                  : (unsigned)-16;
        const unsigned scale_voffset =
            route_group * Config::kGroupM * kScaleCols;
        return {
            MakeBufferResource(values, values_bytes),
            MakeBufferResource(
                scales,
                static_cast<unsigned>(
                    Workspace::Bytes(max_route_groups, kInterDim) -
                    Workspace::ScaleOffset(max_route_groups, kInterDim))),
            value_voffset,
            scale_voffset,
        };
    }

    template <unsigned kTileK>
    __device__ static Prefetch Load(const Address &address, unsigned wtid) {
        static_assert(kTileK < kK256Tiles, "invalid K256 tile");
        return {
            address.values.template Load<BufferResource::kNone>(
                address.value_voffset + kTileK * kGroupDim / 2, 0),
            LoadScaleWord<kTileK>(address, wtid),
        };
    }

    __device__ static void StoreLds(Shm &shm, const uint4 &value,
                                    unsigned stage, unsigned tid) {
        const unsigned row = tid / kVectorsPerRow;
        const unsigned vector = tid % kVectorsPerRow;
        shm[stage][row][vector ^ (row & 15u)] = value;
    }

    __device__ static Input ReadLds(Shm &shm, unsigned stage, unsigned scale,
                                    unsigned wtid) {
        Input input;
#pragma unroll
        for (unsigned row_group = 0; row_group < 2; ++row_group) {
#pragma unroll
            for (unsigned k128 = 0; k128 < 2; ++k128) {
                const unsigned row = wtid % 16 + row_group * 16;
                const unsigned vector = wtid / 16 + k128 * 4;
                input.x[row_group * 2 + k128] =
                    shm[stage][row][vector ^ (row & 15u)];
            }
        }
        input.scale[0] = scale;
        return input;
    }

  private:
    __device__ static unsigned LoadScaleByte(const Address &address,
                                             unsigned row, unsigned col) {
        const unsigned word =
            address.scales.template LoadU32<BufferResource::kNone>(
                address.scale_voffset + row * kScaleCols + (col & ~3u), 0);
        return (word >> ((col & 3u) * 8)) & 0xffu;
    }

    template <unsigned kTileK>
    __device__ static unsigned LoadScaleWord(const Address &address,
                                             unsigned wtid) {
        const unsigned row16 = wtid & 15u;
        const unsigned scale4 = wtid >> 4;
        const unsigned col0 = kTileK * 8 + scale4;
        const unsigned col1 = col0 + 4;
        return LoadScaleByte(address, row16, col0) |
               (LoadScaleByte(address, row16 + 16, col0) << 8) |
               (LoadScaleByte(address, row16, col1) << 16) |
               (LoadScaleByte(address, row16 + 16, col1) << 24);
    }

};

template <class Config, bool kPersistent>
__global__ static void __launch_bounds__(Config::kThreads)
    TwoStageFusedMoEStage1Compute(
        void *__restrict__ workspace, const uint4 *act, const uint4 *w13,
        const uint4 *sorted_token_ids, const uint4 *sorted_expert_ids,
        const unsigned *__restrict__ num_valid_ids, const uint4 *scales_act,
        const uint4 *scales_w13, unsigned m, unsigned num_experts,
        unsigned max_num_m_blocks, const void *w13_bias) {
    using Workspace = TwoStageFusedMoEWorkspace<Config>;
    using Epilogue = MxFp4Stage1WorkspaceEpilogue<Config>;
    using Kernel = FusedMoEStage1<Config, Epilogue>;

    Kernel kernel;
    const unsigned payload_bytes =
        Workspace::PayloadBytes(max_num_m_blocks, Config::kInterDim);
    auto *workspace_bytes = reinterpret_cast<unsigned char *>(workspace);
    typename Epilogue::Context epilogue{
        reinterpret_cast<unsigned short *>(workspace),
        MakeBufferResource(
            workspace_bytes + payload_bytes,
            Workspace::ScaleRows(max_num_m_blocks) *
                Workspace::ScaleCols(Config::kInterDim)),
        num_valid_ids[0],
        m,
    };
    const unsigned persistent_route_step = kPersistent ? gridDim.y : 0;
    kernel.Compute(act, w13,
                   reinterpret_cast<const unsigned *>(sorted_token_ids),
                   reinterpret_cast<const unsigned *>(sorted_expert_ids),
                   scales_act, reinterpret_cast<const unsigned *>(scales_w13),
                   num_valid_ids, m, num_experts, persistent_route_step,
                   w13_bias, epilogue);
}

template <class Config> struct TwoStageFusedMoEStage2 {
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kTokenBatch = Config::kTokenBatch;
    static constexpr unsigned kRoutesPerBlock = kTokenBatch * kNumWarps;

    using W2Weights = typename Config::W2Weights;
    using Bias = typename Config::Stage2Bias;
    using Stage2Tiles = typename Config::Stage2Tiles;
    using WorkspaceInput = MxFp4Stage2Input<
        Config, typename Stage2Tiles::InputRegs>;
    using Stage2EpilogueOp = TwoStageStage2Epilogue<Stage2Tiles>;
    using Intermediate = typename Stage2Tiles::InputRegs;

    static_assert(Stage2Tiles::kActivationFragments == 4, "");

    static constexpr unsigned kPersistentWorkers = 256;
    static constexpr unsigned kInterDim = Config::kInterDim;
    static constexpr unsigned kK256Tiles = kInterDim / kGroupDim;

    static_assert(kInterDim % kGroupDim == 0,
                  "two-stage MoE stage 2 requires complete K256 tiles");
    static_assert(kK256Tiles > 0,
                  "two-stage MoE stage 2 K loop is empty");

    struct EpiloguePrefetch {
        float2 route_weights;
        typename Stage2EpilogueOp::BiasPrefetch bias;
    };

    struct ShmBuf {
        union {
            typename Stage2EpilogueOp::Shm stage2;
            typename WorkspaceInput::Shm input;
        };
    };

    template <unsigned kTileK>
    __device__ __forceinline__ void
    RunStage2KLoop(float4 accum[Stage2Tiles::kAccumFragments],
                   Stage2Tiles &tiles,
                   const typename WorkspaceInput::Address &address,
                   unsigned wid, unsigned wtid, unsigned tid, ShmBuf &shm,
                   const unsigned *sorted_weights, unsigned route_base,
                   unsigned tile_col, EpiloguePrefetch &epilogue) {
        static_assert(kTileK < kK256Tiles, "invalid K256 tile");
        static constexpr unsigned kStage = kTileK & 1u;
        const typename WorkspaceInput::Prefetch input_global =
            WorkspaceInput::template Load<kTileK>(address, wtid);
        WorkspaceInput::StoreLds(shm.input, input_global.value, kStage, tid);

        static constexpr unsigned kWeightStage = kTileK & 1u;
        tiles.LoadKStage(kWeightStage, tid, wid, wtid);
        __syncthreads();
        const Intermediate input = WorkspaceInput::ReadLds(
            shm.input, kStage, input_global.scale, wtid);
        tiles.Matmul(accum, input, kWeightStage, wtid);

        if constexpr (kTileK + 1 == kK256Tiles) {
            epilogue.route_weights = RouteWeightsLayout::Load(
                sorted_weights, route_base, wtid);
            Stage2EpilogueOp::PrefetchBias(epilogue.bias, tiles, tile_col,
                                           tid);
        } else {
            __syncthreads();
            RunStage2KLoop<kTileK + 1>(
                accum, tiles, address, wid, wtid, tid, shm, sorted_weights,
                route_base, tile_col, epilogue);
        }
    }

    TAL_DEVICE void
    Stage2(uint4 *__restrict__ out, const void *__restrict__ intermediate_ptr,
           unsigned route_group, unsigned max_route_groups, unsigned topk,
           unsigned num_tokens, const unsigned *sorted_weights,
           unsigned route_base, unsigned tile_n, unsigned tid, unsigned wid,
           unsigned wtid, unsigned expert_id, const void *w2_bias,
           ShmBuf &shm) {
        Stage2Tiles tiles{w2_weights_.w2_, w2_bias_};
        tiles.InitializeBias(w2_bias, expert_id, tile_n);

        float4 accum[Stage2Tiles::kAccumFragments];
        ClearMat(accum);
        const typename WorkspaceInput::Address intermediate_address =
            WorkspaceInput::Initialize(intermediate_ptr, sorted_token_br_,
                                       route_group, max_route_groups, topk,
                                       num_tokens, tid, wtid);
        const unsigned output_bytes =
            num_tokens * Config::kDim * sizeof(__hip_bfloat16);
        if (tid < kRoutesPerBlock) {
            const unsigned packed_token =
                sorted_token_br_.template LoadU32<BufferResource::kNone>(
                    0, tid * sizeof(unsigned));
            const unsigned token = packed_token & 0x00ffffffu;
            const unsigned slot = packed_token >> 24;
            // Invalid routes use the first byte past the resource range, so
            // their branchless buffer atomics are discarded by hardware.
            const unsigned output_row_offset =
                token < num_tokens && slot < topk
                    ? token * Config::kDim * sizeof(__hip_bfloat16)
                    : output_bytes;
            Stage2EpilogueOp::StoreOutputRowOffset(shm.stage2, tid,
                                                   output_row_offset);
        }
        __syncthreads();
        EpiloguePrefetch epilogue;
        const unsigned tile_col = tile_n * kGroupDim;
        RunStage2KLoop<0>(accum, tiles, intermediate_address, wid, wtid, tid,
                          shm, sorted_weights, route_base, tile_col, epilogue);

        // This is the C-shuffle's leading barrier.  It replaces the final
        // K-loop barrier while still protecting the LDS union from overwrite.
        __syncthreads();
        amdgcn_s_setprio<0>();
        Stage2EpilogueOp::Apply(accum, epilogue.bias,
                                epilogue.route_weights);
        Stage2EpilogueOp::WriteShm(shm.stage2, accum, wid, wtid);
        __syncthreads();
        const BufferResource output = MakeBufferResource(out, output_bytes);
        Stage2EpilogueOp::WriteBack(output, shm.stage2, tile_col, tid);
    }

    TAL_DEVICE void
    Compute(uint4 *__restrict__ out, const void *intermediate_ptr,
            const uint4 *w2, const unsigned *sorted_token_ids,
            const unsigned *sorted_weights, const unsigned *sorted_expert_ids,
            const unsigned *scales_w2, const unsigned *num_valid_ids_ptr,
            unsigned topk, unsigned num_experts, unsigned max_route_groups,
            const void *w2_bias) {
        const unsigned tid = threadIdx.x;
        const unsigned tile_n = blockIdx.x;
        const unsigned persistent_worker = blockIdx.y;
        const unsigned wid = __builtin_amdgcn_readfirstlane(tid / kWarpSize);
        const unsigned wtid = tid % kWarpSize;

        __shared__ ShmBuf shm;
        const unsigned num_valid_ids = num_valid_ids_ptr[0];
        const unsigned num_tokens = num_valid_ids_ptr[1];
        const unsigned route_group_limit =
            tal::CeilingDiv<unsigned>(num_valid_ids, kRoutesPerBlock);
        const unsigned route_groups_per_worker =
            route_group_limit / kPersistentWorkers;
        const unsigned route_group_remainder =
            route_group_limit % kPersistentWorkers;
        const unsigned worker_route_groups =
            route_groups_per_worker +
            static_cast<unsigned>(persistent_worker < route_group_remainder);
        const unsigned route_group_begin =
            persistent_worker * route_groups_per_worker +
            (persistent_worker < route_group_remainder ? persistent_worker
                                                       : route_group_remainder);

        for (unsigned worker_tile = 0; worker_tile < worker_route_groups;
             ++worker_tile) {
            const unsigned route_group = route_group_begin + worker_tile;
            const unsigned route_base = route_group * kRoutesPerBlock;
            if (route_group < route_group_limit && route_base < num_valid_ids) {
                const unsigned expert_id = sorted_expert_ids[route_group];
                const bool valid_expert =
                    num_experts == 0 || expert_id < num_experts;
                if (valid_expert) {
                    w2_weights_.Initialize(w2, scales_w2, expert_id, tile_n,
                                           0);
                    sorted_token_br_ = MakeBufferResource(
                        sorted_token_ids + route_base,
                        kRoutesPerBlock * sizeof(unsigned));

                    Stage2(out, intermediate_ptr, route_group, max_route_groups,
                           topk, num_tokens, sorted_weights, route_base, tile_n,
                           tid, wid, wtid, expert_id, w2_bias, shm);
                }
            }
            __syncthreads();
        }
    }

    BufferResource sorted_token_br_;
    W2Weights w2_weights_;
    Bias w2_bias_;
};

template <class Kernel>
__global__ static void __launch_bounds__(Kernel::kThreads, 1)
    TwoStageFusedMoEStage2Compute(
        uint4 *__restrict__ out, const void *intermediate, const uint4 *w2,
        const uint4 *sorted_token_ids, const uint4 *sorted_weights,
        const uint4 *sorted_expert_ids,
        const unsigned *__restrict__ num_valid_ids, unsigned topk,
        const unsigned *__restrict__ scales_w2, unsigned num_experts,
        unsigned max_route_groups, const void *w2_bias) {
    Kernel kernel;
    kernel.Compute(out, intermediate, w2,
                   reinterpret_cast<const unsigned *>(sorted_token_ids),
                   reinterpret_cast<const unsigned *>(sorted_weights),
                   reinterpret_cast<const unsigned *>(sorted_expert_ids),
                   scales_w2, num_valid_ids, topk, num_experts,
                   max_route_groups, w2_bias);
}

} // namespace causalflow::petit::rocm::moe
