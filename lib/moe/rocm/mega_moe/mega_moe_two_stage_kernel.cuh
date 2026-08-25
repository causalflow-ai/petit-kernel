#pragma once

#include "moe/rocm/comm/barrier.cuh"
#include "moe/rocm/mega_moe/scheduler.cuh"
#include "moe/rocm/mega_moe/workspace.cuh"
#include "moe/rocm/ops/mega_moe/route_output.cuh"
#include "moe/rocm/ops/mega_moe/token_shuffle.cuh"
#include "moe/rocm/ops/mega_moe/token_shuffle_direct_push.cuh"
#include "moe/rocm/ops/mxfp4_activation.cuh"
#include "moe/rocm/ops/op_stages.cuh"

#include <hip/hip_runtime.h>

#include <type_traits>

namespace causalflow::petit::rocm::moe {

template <class Config_, bool kExternalInputs = false>
struct MegaMoETwoStageCommComputeKernel {
    using Config = Config_;
    using Input = typename Config::Input;
    using W13Weights = typename Config::W13Weights;
    using W2Weights = typename Config::W2Weights;
    using Bias = typename Config::Bias;
    using Stage2Bias = typename Config::Stage2Bias;
    using Stage1Tiles = typename Config::Stage1Tiles;
    using Stage1Op = typename Config::Stage1Op;
    using Stage2Tiles = typename Config::Stage2Tiles;
    using Stage2Epilogue = MegaMoETwoStage2Epilogue<Stage2Tiles>;
    using ActivationQuantizer = MxFp4ActivationQuantizer<Config>;
    using Stage2Input = MxFp4Stage2Input<
        Config, typename Stage2Tiles::InputRegs>;
    using Workspace = MegaMoEWorkspace<Config>;
    // A single-rank invocation needs no remote push protocol. Keep the pull
    // implementation there and use fixed-role direct push for every
    // registered multi-rank configuration.
    using TokenDispatch = std::conditional_t<
        Config::kNumRanks == 1, TokenShuffle<Config>,
        DirectPushTokenShuffle<Config, kExternalInputs>>;
    using XGpuSync = typename Config::XGpuSync;
    using Scheduler = MegaMoETwoStageScheduler<Config>;

    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kThreads = Config::kThreads;
    static constexpr unsigned kNumSMs = Config::kNumSMs;
    static constexpr unsigned kTokenBatch = Config::kTokenBatch;
    static constexpr unsigned kRoutesPerBlock = kTokenBatch * kNumWarps;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kInterDim = Config::kInterDim;
    static constexpr unsigned kComputeHiddenSize = Config::kComputeHiddenSize;
    static constexpr unsigned kK256Tiles = kInterDim / kGroupDim;
    static constexpr unsigned kStage1TileCount =
        kInterDim / Config::kStage1GroupN;
    static constexpr unsigned kCommComputeEntryGridSyncIndex = 2;
    static constexpr unsigned kComputeCompleteGridSyncIndex = 3;
    static constexpr unsigned kOutputHandoffGridSyncIndex = 4;

    static_assert(Config::kSolution.stages == FusedMoEStages::kTwoStage);
    static_assert(Config::kActDType == FusedMoEDataType::kMxFp4);
    static_assert(Config::kStage1GroupN == 128 ||
                  Config::kStage1GroupN == 256);
    static_assert(Config::kGroupN == 256);
    static_assert(kRoutesPerBlock == Config::kGroupM);
    static_assert(kInterDim % 512 == 0 && kK256Tiles >= 2);
    static_assert(Stage1Tiles::kAccumFragments ==
                  ActivationQuantizer::kInputFragments);
    static_assert(kComputeHiddenSize >= Config::kHiddenSize);

    struct Stage1Context {};

    struct EpiloguePrefetch {
        float2 route_weights;
        typename Stage2Epilogue::BiasPrefetch bias;
    };

    struct ShmBuf {
        union {
            union {
                typename Stage1Op::Shm stage1;
                typename ActivationQuantizer::QuantizeShm quantize;
                typename Stage2Epilogue::Shm stage2;
                typename Stage2Input::InputShm input;
            } compute;
            typename TokenDispatch::Shm dispatch;
        };
    };

    TAL_DEVICE void RunStage1(Workspace &workspace, ShmBuf &shm,
                              TokenDispatch &dispatch,
                              unsigned dispatch_epoch,
                              const uint4 *w13, const unsigned *scales_w13,
                              const void *w13_bias,
                              const typename Scheduler::Work &work,
                              unsigned tid, unsigned wid, unsigned wtid) {
        if constexpr (Config::kNumRanks > 1) {
            dispatch.WaitForExpertPayload(work.expert_idx, dispatch_epoch,
                                          tid);
        }
        const unsigned pool_base = work.pool_block * kRoutesPerBlock;
        input_.Initialize(workspace, work.pool_block, work.work_m);
        input_.PrepareScales(shm.compute.stage1.x, wid, wtid);
        Config::InitializeW13(*this, w13, scales_w13, work.expert_idx,
                              work.tile, 0, 0);

        Stage1Tiles tiles{input_, w13_weights_.w1_, w13_bias_};
        tiles.InitializeBias(w13_bias, work.expert_idx, work.tile);
        unsigned tokens[kTokenBatch];
#pragma unroll
        for (unsigned i = 0; i < kTokenBatch; ++i)
            tokens[i] = wid * kTokenBatch + i;
        float4 hidden[Stage1Tiles::kAccumFragments];
        Stage1Op::Run(hidden, shm.compute.stage1, tiles, tid, wid, wtid,
                      tokens, work.work_m);
        __syncthreads();
        ActivationQuantizer::StoreAccumulator(shm.compute.quantize, hidden,
                                              wid, wtid);
        __syncthreads();
        const unsigned route_in_slice = tid / 32;
        const unsigned col_lane = tid % 32;
#pragma unroll
        for (unsigned route_slice = 0;
             route_slice < Config::kGroupM / 8; ++route_slice) {
            const unsigned route = route_slice * 8 + route_in_slice;
            if (route < work.work_m) {
#pragma unroll
                for (unsigned col_segment = 0;
                     col_segment < Config::kStage1GroupN / 128;
                     ++col_segment) {
                    const unsigned quant_col =
                        col_segment * 32 + col_lane;
                    const auto quantized = ActivationQuantizer::Quantize(
                        shm.compute.quantize, route, quant_col);
                    ActivationQuantizer::template Store<
                        BufferResource::kSC1Bit>(
                        workspace.br_, workspace.L2TokenBufferOffset(0),
                        workspace.L2ScaleBufferOffset(), pool_base + route,
                        pool_base + route, work.tile, quant_col, kInterDim,
                        Workspace::kL2ScaleCols, quantized);
                }
            }
        }

        // The protected value/scale stores are device-scoped. Every wave
        // completes its own stores before the CTA joins and publishes the L2
        // tile bit, so no cache-wide agent release is needed.
        complete_scoped_vmem();
        __syncthreads();
        if (tid == 0) {
            for (unsigned subblock = 0;
                 subblock < tal::CeilingDiv(work.work_m, 32u); ++subblock) {
                workspace.br_
                    .template AtomicOrU32<BufferResource::kAtomicScopeAgent>(
                        workspace.L2ArrivalMaskOffset(work.pool_block +
                                                      subblock),
                        0, 1u << work.tile);
            }
        }
        __syncthreads();
    }

    TAL_DEVICE void WaitL2Block(Workspace &workspace, unsigned pool_block,
                                unsigned tid) const {
        static constexpr unsigned kReadyMask =
            (1u << kStage1TileCount) - 1;
        if (tid == 0) {
            unsigned observed;
            do {
                // The arrival mask is read-only here.  Poll both cache scopes
                // coherently instead of invalidating the whole cache on every
                // unsuccessful observation.
                observed = workspace.br_.template LoadU32<
                    BufferResource::kSC0Bit | BufferResource::kSC1Bit>(
                    workspace.L2ArrivalMaskOffset(pool_block), 0);
                if ((observed & kReadyMask) != kReadyMask) {
                    asm volatile("s_sleep 1" ::: "memory");
                }
            } while ((observed & kReadyMask) != kReadyMask);
        }
        __syncthreads();
        // RunStage2 consumes the published payload and scales with coherent
        // loads below, so no whole-cache acquire invalidation is needed here.
    }

    TAL_DEVICE void RunStage2(Workspace &workspace, ShmBuf &shm,
                              const uint4 *w2, const unsigned *scales_w2,
                              const void *w2_bias, unsigned dispatch_epoch,
                              const typename Scheduler::Work &work,
                              unsigned tid, unsigned wid, unsigned wtid) {
        const unsigned pool_base = work.pool_block * kRoutesPerBlock;
        // Unlike the fused one-stage path, two-stage work.tile selects an
        // output (N) tile after the full intermediate has been materialized.
        // Initialize W2 at that N tile and at K tile zero.
        w2_weights_.Initialize(w2, scales_w2, work.expert_idx, work.tile, 0);
        Stage2Tiles tiles{w2_weights_.w2_, w2_bias_};
        tiles.InitializeBias(w2_bias, work.expert_idx, work.tile);
        float4 accum[Stage2Tiles::kAccumFragments];
        ClearMat(accum);
        WaitL2Block(workspace, work.pool_block, tid);

        const unsigned row = tid / Stage2Input::kVectorsPerRow;
        const unsigned vector = tid % Stage2Input::kVectorsPerRow;
        const unsigned value_voffset =
            (pool_base + row) * (kInterDim / 2) + vector * sizeof(uint4);
        const unsigned value_soffset = workspace.L2TokenBufferOffset(0);
        const unsigned scale_voffset = pool_base * Stage2Input::kScaleCols;
        const unsigned scale_soffset = workspace.L2ScaleBufferOffset();

        const unsigned tile_col = work.tile * Config::kGroupN;
        EpiloguePrefetch epilogue;
        typename Stage2Epilogue::Context output_context{
            &workspace, pool_base, work.work_m};
        // Route weights and bias are independent of the W2 contraction. Issue
        // their small loads before the K loop so its twelve K256 tiles hide
        // the latency without keeping another W2 tile live.
        epilogue.route_weights =
            Stage2Epilogue::LoadRouteWeights(output_context, wtid);
        Stage2Epilogue::PrefetchBias(epilogue.bias, tiles, tile_col, tid);

#pragma unroll
        for (unsigned tile_k = 0; tile_k < kK256Tiles; ++tile_k) {
            const unsigned stage = tile_k & 1u;
            // Apply coherence only to the payload and scale lines protected by
            // the arrival mask instead of invalidating the entire cache.
            const auto prefetched = Stage2Input::template LoadTile<
                BufferResource::kSC0Bit | BufferResource::kSC1Bit>(
                workspace.br_, value_voffset, value_soffset, scale_voffset,
                scale_soffset, tile_k, true, wtid);
            Stage2Input::StoreLds(shm.compute.input, prefetched.value, stage,
                                  tid);
            tiles.LoadKStage(stage, tid, wid, wtid);
            __syncthreads();
            const auto input = Stage2Input::ReadLds(
                shm.compute.input, stage, prefetched.scale, wtid);
            tiles.Matmul(accum, input, stage, wtid);
            if (tile_k + 1 < kK256Tiles)
                __syncthreads();
        }

        __syncthreads();
        Stage2Epilogue::Apply(accum, epilogue.bias,
                              epilogue.route_weights);
        Stage2Epilogue::WriteShm(shm.compute.stage2, accum, wid, wtid);
        __syncthreads();
        Stage2Epilogue::WriteBack(output_context, shm.compute.stage2, tile_col,
                                  wid, wtid);
        amdgcn_s_waitcnt<0, -1, 0>();
        if constexpr (Config::kNumRanks > 1)
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
        __syncthreads();

        if constexpr (Config::kNumRanks > 1) {
            // One lane publishes each route after every lane has drained this
            // output tile. The source reducer waits on the token's complete
            // top-k x output-tile dependency count.
            if (wtid == 0) {
#pragma unroll
                for (unsigned row_group = 0;
                     row_group < kRoutesPerBlock / kNumWarps; ++row_group) {
                    const unsigned row = wid + row_group * kNumWarps;
                    if (row < work.work_m) {
                        const TokenMetadata metadata = __builtin_bit_cast(
                            TokenMetadata,
                            workspace.br_.template LoadU64<
                                BufferResource::kNone>(
                                0, workspace.TokenMetadataOffset(
                                       workspace.Rank(), pool_base + row)));
                        workspace.br_.template AtomicAddI32<
                            BufferResource::kAtomicScopeSystem>(
                            workspace.RouteOutputReadyOffset(
                                metadata.src_rank, dispatch_epoch & 1,
                                metadata.token_topk_idx / Config::kTopK),
                            0, 1);
                    }
                }
            }
        }
    }

    TAL_DEVICE void Compute(Workspace &workspace, Scheduler &scheduler,
                            ShmBuf &shm, TokenDispatch &dispatch,
                            unsigned dispatch_epoch, const uint4 *w13,
                            const uint4 *w2,
                            const unsigned *scales_w13,
                            const unsigned *scales_w2, const void *w13_bias,
                            const void *w2_bias, unsigned sm_id, unsigned tid,
                            unsigned wid, unsigned wtid) {
        unsigned logical_id = sm_id;
        typename Scheduler::Work work;

        // Work IDs are ordered by phase. Separate loops keep each GEMM's
        // descriptors and unrolled pipeline state out of the other phase's
        // register-pressure region.
        for (;;) {
            if (!scheduler.GetWork(wtid, logical_id, &work))
                return;
            work.phase =
                static_cast<MegaMoEBlockPhase>(__builtin_amdgcn_readfirstlane(
                    static_cast<unsigned>(work.phase)));
            if (work.phase != MegaMoEBlockPhase::kLinear1)
                break;
            work.expert_idx = __builtin_amdgcn_readfirstlane(work.expert_idx);
            work.pool_block = __builtin_amdgcn_readfirstlane(work.pool_block);
            work.work_m = __builtin_amdgcn_readfirstlane(work.work_m);
            work.tile = __builtin_amdgcn_readfirstlane(work.tile);
            RunStage1(workspace, shm, dispatch, dispatch_epoch, w13,
                      scales_w13, w13_bias, work, tid, wid, wtid);
            logical_id += kNumSMs;
        }

        for (;;) {
            work.expert_idx = __builtin_amdgcn_readfirstlane(work.expert_idx);
            work.pool_block = __builtin_amdgcn_readfirstlane(work.pool_block);
            work.work_m = __builtin_amdgcn_readfirstlane(work.work_m);
            work.tile = __builtin_amdgcn_readfirstlane(work.tile);
            RunStage2(workspace, shm, w2, scales_w2, w2_bias,
                      dispatch_epoch, work, tid, wid, wtid);
            logical_id += kNumSMs;
            if (!scheduler.GetWork(wtid, logical_id, &work))
                break;
        }
    }
    TAL_DEVICE void Run(uint4 *out, const uint4 *w13, const uint4 *w2,
                        const unsigned *scales_w13, const unsigned *scales_w2,
                        unsigned num_tokens, unsigned output_row_stride,
                        const void *w13_bias,
                        const void *w2_bias, void *base, unsigned rank,
                        const uint4 *input_tokens,
                        const unsigned *input_topk_ids,
                        const float *input_topk_weights) {
        __shared__ ShmBuf shm;
        const unsigned sm_id = blockIdx.x;
        const unsigned tid = threadIdx.x;
        const unsigned wid = __builtin_amdgcn_readfirstlane(tid / kWarpSize);
        const unsigned wtid = tid % kWarpSize;
        Workspace workspace(base, rank);

        TokenDispatch dispatch(num_tokens, &workspace, &shm.dispatch,
                               input_tokens, input_topk_ids,
                               input_topk_weights);
        unsigned dispatch_epoch = 0;
        if constexpr (Config::kNumRanks == 1) {
            dispatch.Run(sm_id, tid, wid, wtid);
            // Pull dispatch distributes materialization across the grid, so
            // EP1 retains its local dispatch-to-compute handoff.
            grid_sync<kNumSMs, kCommComputeEntryGridSyncIndex>(
                workspace, sm_id, tid, [] { __syncthreads(); });
        } else {
            // The planner publishes the local schedule and each producer
            // publishes one expert payload; compute waits on those
            // dependencies directly.
            dispatch_epoch = dispatch.Run(sm_id, tid, wid, wtid);
            // Every CTA must observe the destination-owned schedule before it
            // reads scheduler metadata. This is a local plan-ready dependency,
            // not a dispatch-wide barrier.
            dispatch.WaitForLocalPlan(dispatch_epoch, tid);
        }

        Scheduler scheduler(&workspace);
        scheduler.FetchRecvSumPerExpert(wtid);
        Compute(workspace, scheduler, shm, dispatch, dispatch_epoch, w13, w2,
                scales_w13, scales_w2, w13_bias, w2_bias, sm_id, tid, wid,
                wtid);

        SourceRouteReducer<Config> reducer(&workspace);
        if constexpr (Config::kNumRanks == 1) {
            // EP1 retains the pull path's post-compute cleanup and collective
            // handoff. Multi-rank direct push publishes route dependencies
            // from stage 2 and reclaims L2 state at next-epoch admission.
            amdgcn_s_waitcnt<0, -1, 0>();
            grid_sync<kNumSMs, kComputeCompleteGridSyncIndex,
                      /* kAcquirePayload */ false>(
                workspace, sm_id, tid, [] { __syncthreads(); });
            for (unsigned block = sm_id * kThreads + tid;
                 block < Workspace::kMaxPoolBlocks;
                 block += kNumSMs * kThreads) {
                workspace.br_.template StoreU32<BufferResource::kNone>(
                    workspace.L2ArrivalMaskOffset(block), 0, 0);
            }
            dispatch.ResetRoutingCounters(sm_id, tid);
            const auto output_sync_ticket =
                XGpuSync::template Begin<kOutputHandoffGridSyncIndex>(
                    workspace, sm_id, tid, [] { __syncthreads(); });
            XGpuSync::template Finish<kOutputHandoffGridSyncIndex>(
                workspace, sm_id, tid, output_sync_ticket,
                [] { __syncthreads(); });
            reducer.Run(out, num_tokens, output_row_stride, sm_id, wid, wtid);
        } else {
            reducer.Run(out, num_tokens, output_row_stride, sm_id, wid, wtid,
                        dispatch_epoch);
        }
    }
    Input input_;
    W13Weights w13_weights_;
    Bias w13_bias_;
    W2Weights w2_weights_;
    Stage2Bias w2_bias_;
};

template <class Kernel>
__global__ static void __launch_bounds__(Kernel::kThreads)
    MegaMoETwoStage(uint4 *out, const uint4 *w13, const uint4 *w2,
                    const unsigned *scales_w13, const unsigned *scales_w2,
                    unsigned num_tokens, unsigned output_row_stride,
                    const void *w13_bias,
                    const void *w2_bias, void *base, unsigned rank,
                    const uint4 *input_tokens, const unsigned *input_topk_ids,
                    const float *input_topk_weights) {
    Kernel kernel;
    kernel.Run(out, w13, w2, scales_w13, scales_w2, num_tokens,
               output_row_stride, w13_bias, w2_bias, base, rank, input_tokens,
               input_topk_ids, input_topk_weights);
}
} // namespace causalflow::petit::rocm::moe
