#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "moe/rocm/fused_moe_blockscale_fp8_kernel.cuh"
#include "moe/rocm/memory_ops.cuh"
#include "moe/rocm/ops/mxfp4_activation.cuh"
#include "moe/rocm/ops/op_stages.cuh"
#include "moe/rocm/ops/schedule_tiles.cuh"

namespace causalflow::petit::rocm::moe {

template <class Config> struct TwoStageFusedMoEWorkspace {
    using Layout = MxFp4ActivationLayout;
    static constexpr unsigned kRoutesPerBlock = Config::kGroupM;

    TAL_HOST_DEVICE static constexpr unsigned
    ValueBytes(unsigned route_groups, unsigned inter_dim) {
        return Layout::ValueBytes(route_groups * kRoutesPerBlock, inter_dim);
    }

    TAL_HOST_DEVICE static constexpr unsigned
    ScaleRows(unsigned route_groups) {
        return Layout::PaddedScaleRows(route_groups * kRoutesPerBlock);
    }

    TAL_HOST_DEVICE static constexpr unsigned
    ScaleCols(unsigned inter_dim) {
        return Layout::ScaleCols(inter_dim);
    }

    TAL_HOST_DEVICE static constexpr unsigned
    ScaleOffset(unsigned route_groups, unsigned inter_dim) {
        return ValueBytes(route_groups, inter_dim);
    }

    TAL_HOST_DEVICE static constexpr unsigned
    Bytes(unsigned route_groups, unsigned inter_dim) {
        return ScaleOffset(route_groups, inter_dim) +
               ScaleRows(route_groups) * ScaleCols(inter_dim);
    }
};

template <class Config> struct MxFp4Stage1WorkspaceEpilogue {
    using Quantizer = MxFp4ActivationQuantizer<Config>;
    static constexpr unsigned kInputFragments = Quantizer::kInputFragments;
    using Shm = typename Quantizer::QuantizeShm;

    struct Context {
        BufferResource workspace;
        unsigned scale_base;
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
        Quantizer::StoreAccumulator(shm, h, wid, wtid);
        __syncthreads();

        const unsigned route_in_slice = tid / 32;
        const unsigned col_lane = tid % 32;
        const auto *metadata =
            reinterpret_cast<const unsigned *>(&row_metadata[0]);
#pragma unroll
        for (unsigned route_slice = 0;
             route_slice < Config::kGroupM / 8; ++route_slice) {
            const unsigned row = route_slice * 8 + route_in_slice;
            const unsigned sorted_row = route_group * Config::kGroupM + row;
            StoreRoute(context, shm, metadata[row], row, sorted_row, tile_n,
                       col_lane);
        }
    }

  private:
    TAL_DEVICE static void StoreRoute(Context &context, const Shm &shm,
                                      unsigned fused, unsigned row,
                                      unsigned sorted_row, unsigned tile_n,
                                      unsigned col_lane) {
        const unsigned token = fused & 0x00ffffffu;
        const unsigned slot = fused >> 24;
        const bool valid = sorted_row < context.num_valid_ids &&
                           token < context.m && slot < Config::kTopK;
        if (!valid)
            return;
        const unsigned value_row = token * Config::kTopK + slot;
#pragma unroll
        for (unsigned col_segment = 0;
             col_segment < Config::kStage1GroupN / 128; ++col_segment) {
            const unsigned quant_col_lane = col_segment * 32 + col_lane;
            const auto quantized =
                Quantizer::Quantize(shm, row, quant_col_lane);
            Quantizer::Store(
                context.workspace, 0, context.scale_base, value_row,
                sorted_row, tile_n, quant_col_lane, Config::kInterDim,
                TwoStageFusedMoEWorkspace<Config>::ScaleCols(
                    Config::kInterDim),
                quantized);
        }
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
#if !defined(__gfx950__)
    if constexpr (Config::kSolution.weight_ordering ==
                  FusedMoEWeightOrdering::kNativeMxFp4)
        return;
#endif
    using Workspace = TwoStageFusedMoEWorkspace<Config>;
    using Epilogue = MxFp4Stage1WorkspaceEpilogue<Config>;
    using Kernel = FusedMoEStage1<Config, Epilogue>;

    Kernel kernel;
    const unsigned value_bytes =
        Workspace::ValueBytes(max_num_m_blocks, Config::kInterDim);
    const unsigned workspace_bytes =
        Workspace::Bytes(max_num_m_blocks, Config::kInterDim);
    typename Epilogue::Context epilogue{
        MakeBufferResource(workspace, workspace_bytes),
        value_bytes,
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
    using ConfigType = Config;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kTokenBatch = Config::kStage2TokenBatch;
    static constexpr unsigned kRoutesPerBlock = kTokenBatch * kNumWarps;

    using W2Weights = typename Config::W2Weights;
    using Bias = typename Config::Stage2Bias;
    using Stage2Tiles = typename Config::Stage2Tiles;
    using Stage2Input = MxFp4Stage2Input<
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

    using Stage2Shm = typename Stage2EpilogueOp::Shm;
    using InputShm = typename Stage2Input::InputShm;
    static_assert(sizeof(InputShm) == 2 * 32 * 16 * sizeof(uint4),
                  "stage2 activation LDS layout must remain two M32xK256 "
                  "buffers");
    static_assert(sizeof(InputShm) == offsetof(Stage2Shm, output_row_offsets),
                  "stage2 row offsets must follow the activation LDS buffers");
    static_assert(offsetof(Stage2Shm, route_weights) ==
                      sizeof(InputShm) + kRoutesPerBlock * sizeof(unsigned),
                  "stage2 route weights must follow the row offsets");
    static_assert(sizeof(Stage2Shm) ==
                      sizeof(InputShm) +
                          2 * kRoutesPerBlock * sizeof(unsigned),
                  "stage2 LDS layout must include row offsets and route "
                  "weights");

    struct ShmBuf {
        union {
            Stage2Shm stage2;
            InputShm input;
        };
    };

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
        using Workspace = TwoStageFusedMoEWorkspace<Config>;
        const unsigned values_bytes =
            Workspace::ValueBytes(max_route_groups, kInterDim);
        const unsigned workspace_size =
            Workspace::Bytes(max_route_groups, kInterDim);
        const auto *workspace_bytes =
            reinterpret_cast<const unsigned char *>(intermediate_ptr);
        const BufferResource workspace =
            MakeBufferResource(workspace_bytes, workspace_size);
        const unsigned row = tid / Stage2Input::kVectorsPerRow;
        const unsigned vector = tid % Stage2Input::kVectorsPerRow;
        const unsigned packed_id =
            sorted_token_br_.template LoadU32<BufferResource::kNone>(
                row * sizeof(unsigned), 0);
        const unsigned token = packed_id & 0x00ffffffu;
        const unsigned slot = packed_id >> 24;
        const bool valid = token < num_tokens && slot < topk;
        const unsigned value_voffset =
            valid ? (token * topk + slot) * (kInterDim / 2) +
                        vector * sizeof(uint4)
                  : 0;
        const unsigned scale_voffset =
            route_group * kRoutesPerBlock * Stage2Input::kScaleCols;
        const unsigned output_bytes =
            num_tokens * Config::kDim * sizeof(__hip_bfloat16);
        const BufferResource route_weights = MakeBufferResource(
            sorted_weights + route_base,
            kRoutesPerBlock * sizeof(unsigned));
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
            const unsigned route_weight =
                route_weights.template LoadU32<BufferResource::kNone>(
                    tid * sizeof(unsigned), 0);
            Stage2EpilogueOp::StoreOutputRowOffset(shm.stage2, tid,
                                                   output_row_offset);
            Stage2EpilogueOp::StoreRouteWeight(shm.stage2, tid, route_weight);
        }
        __syncthreads();
        EpiloguePrefetch epilogue;
        const unsigned tile_col = tile_n * kGroupDim;
#pragma unroll
        for (unsigned tile_k = 0; tile_k < kK256Tiles; ++tile_k) {
            const unsigned stage = tile_k & 1u;
            const auto input_global = Stage2Input::LoadTile(
                workspace, value_voffset, 0, scale_voffset, values_bytes,
                tile_k, valid, wtid);
            Stage2Input::StoreLds(shm.input, input_global.value, stage, tid);
            tiles.LoadKStage(stage, tid, wid, wtid);
            __syncthreads();
            const Intermediate input = Stage2Input::ReadLds(
                shm.input, stage, input_global.scale, wtid);
            tiles.Matmul(accum, input, stage, wtid);

            if (tile_k + 1 == kK256Tiles) {
                epilogue.route_weights =
                    Stage2EpilogueOp::LoadRouteWeights(shm.stage2, wtid);
                Stage2EpilogueOp::PrefetchBias(epilogue.bias, tiles, tile_col,
                                               tid);
            } else {
                __syncthreads();
            }
        }

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
                const unsigned expert_id = sorted_expert_ids[
                    route_group / Config::kStage1ToStage2GroupRatio];
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
#if !defined(__gfx950__)
    using Config = typename Kernel::ConfigType;
    if constexpr (Config::kSolution.weight_ordering ==
                  FusedMoEWeightOrdering::kNativeMxFp4)
        return;
#endif
    Kernel kernel;
    kernel.Compute(out, intermediate, w2,
                   reinterpret_cast<const unsigned *>(sorted_token_ids),
                   reinterpret_cast<const unsigned *>(sorted_weights),
                   reinterpret_cast<const unsigned *>(sorted_expert_ids),
                   scales_w2, num_valid_ids, topk, num_experts,
                   max_route_groups, w2_bias);
}

} // namespace causalflow::petit::rocm::moe
