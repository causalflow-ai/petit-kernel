#pragma once

#include "moe/rocm/fused_moe_config_selector.cuh"
#include "moe/rocm/mega_moe.h"
#include "moe/rocm/mega_moe/mega_moe_two_stage_kernel.cuh"
#include "moe/rocm/mem/input_mxfp4_packed.cuh"

#include <type_traits>

namespace causalflow::petit::rocm::moe {

template <FusedMoESolutionId Solution> struct MegaMoEConfigSelector {
    using Self = MegaMoEConfigSelector;
    using XGpuSync = LegacyXGpuSync<Self>;
    static constexpr unsigned kGroupM = 32;
    // Mega stage 1 reuses each shuffled activation tile across both N128
    // projection halves. Keep the M32 CTA geometry while computing N256 per
    // projection so persistent tickets do not reload A for adjacent N tiles.
    static constexpr unsigned kStage1GroupN = 256;
    static constexpr unsigned kGroupN =
        Solution.W2TileShape() == MegaMoETileShape::kN128 ? 128 : 256;
    static constexpr unsigned kGroupDim = 256;
    static constexpr unsigned kTokenBatch = 8;
    static constexpr unsigned kStage2GroupM = 32;
    static constexpr unsigned kStage2TokenBatch = 8;
    static constexpr unsigned kNumWarps = 4;
    static constexpr unsigned kStage1WarpsM = 1;
    static constexpr unsigned kStage1WarpsN = kNumWarps;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kNumSMs = 256;
    // Pull dispatch (used by EP1) owns 0/1 and uses 2 for its local handoff.
    // The EP1 fused kernel uses 3 for compute completion. Slot 4 is the xGPU
    // output handoff used by the separate multi-rank combine kernel. Direct
    // push uses per-expert epochs instead of an entry grid barrier.
    static constexpr unsigned kGridSyncSlots = 5;
    static constexpr unsigned kStage2GroupInterDim = kGroupDim;
    static constexpr unsigned kNumRanks = 1u << Solution.NumRanksLog2();
    static constexpr unsigned kNumExperts = Solution.NumExperts();
    static constexpr unsigned kTopK = Solution.TopK();
    static constexpr unsigned kHiddenSize = Solution.HiddenSizeDiv64() * 64;
    static constexpr unsigned kComputeHiddenSize =
        (kHiddenSize + 511) / 512 * 512;
    static constexpr unsigned kDim = kComputeHiddenSize;
    static constexpr unsigned kInterDim = Solution.MegaInterDimDiv64() * 64;
    static constexpr unsigned kProducerBlocks =
        Solution.ProducerBlocks() < kNumExperts ? Solution.ProducerBlocks()
                                                : kNumExperts;
    static constexpr unsigned kMaxTokensPerRank = 1024;
    static constexpr unsigned kSortedTokenBlock = 32;
    // A 7168-wide MXFP4 activation row occupies 238 uint4 vectors. Use all
    // four base-kernel waves so the legacy pull transport remains valid even
    // though multi-rank MegaMoE normally takes the direct-push path.
    static constexpr unsigned kWarpsPerPullToken =
        kHiddenSize > 4096 ? 4 : 2;
    static constexpr unsigned kWeightScaleBlockSize = 128;
    // Persistent MegaMoE schedules multiple row tickets for the same expert.
    // Keep W13 cacheable so later tickets can reuse the expert's weight lines.
    // The generic gfx950 local-MoE policy is non-temporal because its
    // independently scheduled tiles do not have this reuse guarantee.
    static constexpr int kWeightLoadAux = BufferResource::kNone;
    using InputTransport = typename RemoteInputTransportSelector<
        Solution.act_dtype, kHiddenSize, kComputeHiddenSize, kWarpsPerPullToken,
        kNumWarps>::Type;
    static constexpr unsigned kInputTokenBytes =
        InputTransport::kInputTokenBytes;
    static constexpr unsigned kMaxExpertsPerToken =
        kTopK < kNumExperts / kNumRanks ? kTopK : kNumExperts / kNumRanks;
    static constexpr unsigned kRouteOutputBufferBytes =
        kTopK * kHiddenSize * sizeof(__hip_bfloat16);
    static constexpr FusedMoEDataType kActDType = Solution.act_dtype;
    static constexpr FusedMoEDataType kWeightDType = Solution.weight_dtype;
    static constexpr FusedMoEMfmaShape kMfmaShape = Solution.mfma;
    static constexpr bool kValidateExpertIds = false;
    static constexpr MxFp4TileShape kW13TileShape =
        MxFp4TileShape::kN256;
    static constexpr MxFp4TileShape kW2TileShape =
        Solution.W2TileShape() == MegaMoETileShape::kN128
            ? MxFp4TileShape::kN128
            : MxFp4TileShape::kN256;
    static constexpr FusedMoESolutionId kSolution = Solution;

    static_assert(kNumExperts != 0 && kTopK != 0 && kHiddenSize != 0 &&
                  kInterDim != 0);
    static_assert(kSolution.stages == FusedMoEStages::kTwoStage);
    static_assert(kActDType == FusedMoEDataType::kMxFp4);
    static_assert(kNumExperts % kNumRanks == 0);
    static_assert(kTopK <= kNumExperts);
    static_assert(kInterDim % 512 == 0);
    using ActivationOp =
        typename FusedMoEActivationSelector<kSolution.activation>::Type;
    using Input = MxFp4InputPacked<Self>;
    using Weight = FusedMoEWeightSelector<kSolution.weight_dtype,
                                          kSolution.weight_ordering, Self>;
    using W13Weights = typename Weight::W13Weights;
    using W2Weights = typename Weight::W2Weights;
    using W13 = typename W13Weights::W13;
    using W2 = typename W2Weights::W2;
    using Bias =
        typename BiasLayoutSelector<kSolution.bias_dtype,
                                    kSolution.weight_dtype, kSolution.mfma,
                                    kNumWarps, kStage1GroupN>::Type;
    using Stage2Bias =
        typename BiasLayoutSelector<kSolution.bias_dtype,
                                    kSolution.weight_dtype, kSolution.mfma,
                                    kNumWarps, kGroupN>::Type;
    using Stage1Tiles = typename FusedMoEStage1TilesSelector<
        kSolution.weight_dtype, kSolution.weight_ordering, kSolution.mfma,
        Self>::Type;
    using Stage1Op = typename FusedMoEStage1OpSelector<
        kSolution.stage1_buffering, Stage1Tiles>::Type;
    using Stage2Tiles = typename FusedMoEStage2TilesSelector<
        kSolution.weight_dtype, kSolution.weight_ordering, kSolution.mfma,
        Self>::Type;

    template <class Kernel>
    TAL_DEVICE static void
    InitializeW13(Kernel &kernel, const uint4 *w13,
                  const unsigned *scales_w13, unsigned expert,
                  unsigned tile_k, unsigned n_blocks, unsigned k_blocks) {
        Weight::InitializeW13(kernel, w13, scales_w13, expert, tile_k,
                              n_blocks, k_blocks);
    }

    template <class Kernel>
    TAL_DEVICE static void
    InitializeW2(Kernel &kernel, const uint4 *w2,
                 const unsigned *scales_w2, unsigned expert,
                 unsigned tile_k, unsigned n_blocks, unsigned k_blocks) {
        Weight::InitializeW2(kernel, w2, scales_w2, expert, tile_k,
                             n_blocks, k_blocks);
    }
};

// Use a four-wave M64 x N512 stage-1 tile for medium token counts. Keep stage
// 2 on the four-wave M32 base config; the direct-push pool and L2 readiness
// layout remain expressed in M32 units.
template <class Base> struct MegaMoEStage1M64W4Config : Base {
    using Self = MegaMoEStage1M64W4Config;
    static constexpr unsigned kGroupM = 64;
    static constexpr unsigned kStage1GroupN = 256;
    static constexpr unsigned kTokenBatch = 16;
    static constexpr unsigned kNumWarps = 4;
    static constexpr unsigned kStage1WarpsM = 2;
    static constexpr unsigned kStage1WarpsN = 2;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr MxFp4TileShape kW13TileShape =
        MxFp4TileShape::kM64N256;

    using Input = MxFp4InputPacked<Self>;
    using Weight = FusedMoEWeightSelector<
        Base::kSolution.weight_dtype, Base::kSolution.weight_ordering, Self>;
    using W13Weights = typename Weight::W13Weights;
    using W13 = typename W13Weights::W13;
    using Bias = std::conditional_t<
        Base::kSolution.bias_dtype == FusedMoEDataType::kNone,
        NoopBiasLayout<kNumWarps, kStage1GroupN>,
        Bf16BiasLayout<kNumWarps, kStage1GroupN,
                       MxFp4BiasLayoutM64N256, 8>>;
    using Stage1Tiles = typename FusedMoEStage1TilesSelector<
        Base::kSolution.weight_dtype, Base::kSolution.weight_ordering,
        Base::kSolution.mfma, Self>::Type;
    using Stage1Op = typename FusedMoEStage1OpSelector<
        Base::kSolution.stage1_buffering, Stage1Tiles>::Type;

    template <class Kernel>
    TAL_DEVICE static void
    InitializeW13(Kernel &kernel, const uint4 *w13,
                  const unsigned *scales_w13, unsigned expert,
                  unsigned tile_k, unsigned n_blocks, unsigned k_blocks) {
        Weight::InitializeW13(kernel, w13, scales_w13, expert, tile_k,
                              n_blocks, k_blocks);
    }
};

// Use the eight-wave variant at large token counts, where the additional
// waves are fully amortized.
template <class Base> struct MegaMoEStage1M64W8Config : Base {
    using Self = MegaMoEStage1M64W8Config;
    static constexpr unsigned kGroupM = 64;
    static constexpr unsigned kStage1GroupN = 256;
    static constexpr unsigned kTokenBatch = 8;
    static constexpr unsigned kNumWarps = 8;
    static constexpr unsigned kStage1WarpsM = 2;
    static constexpr unsigned kStage1WarpsN = 4;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr bool kStage1WaveM64 = true;
    static constexpr MxFp4TileShape kW13TileShape =
        MxFp4TileShape::kM64N256W8;

    using Input = MxFp4InputPacked<Self>;
    using Weight = FusedMoEWeightSelector<
        Base::kSolution.weight_dtype, Base::kSolution.weight_ordering, Self>;
    using W13Weights = typename Weight::W13Weights;
    using W13 = typename W13Weights::W13;
    using Bias = std::conditional_t<
        Base::kSolution.bias_dtype == FusedMoEDataType::kNone,
        NoopBiasLayout<kNumWarps, kStage1GroupN>,
        Bf16BiasLayout<kNumWarps, kStage1GroupN,
                       MxFp4BiasLayoutM64N256W8, 2, 4>>;
    using Stage1Tiles = typename FusedMoEStage1TilesSelector<
        Base::kSolution.weight_dtype, Base::kSolution.weight_ordering,
        Base::kSolution.mfma, Self>::Type;
    using Stage1Op = typename FusedMoEStage1OpSelector<
        Base::kSolution.stage1_buffering, Stage1Tiles>::Type;

    template <class Kernel>
    TAL_DEVICE static void
    InitializeW13(Kernel &kernel, const uint4 *w13,
                  const unsigned *scales_w13, unsigned expert,
                  unsigned tile_k, unsigned n_blocks, unsigned k_blocks) {
        Weight::InitializeW13(kernel, w13, scales_w13, expert, tile_k,
                              n_blocks, k_blocks);
    }
};

template <unsigned long kRepr>
int MegaMoESolutionAdapter<kRepr>::Invoke(MegaMoEParams params) {
    static constexpr FusedMoESolutionId kSolution =
        FusedMoESolutionId::FromRepr(kRepr);
    using Config = MegaMoEConfigSelector<kSolution>;
    using Kernel = MegaMoETwoStageCommComputeKernel<Config>;
    using ExternalInputKernel =
        MegaMoETwoStageCommComputeKernel<Config, true>;
    using M64Stage1Config = MegaMoEStage1M64W4Config<Config>;
    using M64Stage1Kernel =
        MegaMoETwoStageCommComputeKernel<M64Stage1Config>;
    using M64ExternalInputStage1Kernel =
        MegaMoETwoStageCommComputeKernel<M64Stage1Config, true>;
    using M64W8Stage1Config = MegaMoEStage1M64W8Config<Config>;
    using M64W8Stage1Kernel =
        MegaMoETwoStageCommComputeKernel<M64W8Stage1Config>;
    using M64W8ExternalInputStage1Kernel =
        MegaMoETwoStageCommComputeKernel<M64W8Stage1Config, true>;
    using CombineKernel = MegaMoECombineKernel<Config>;
    // Choose M64 from route density rather than applying the GPT-OSS cutoff
    // to every expert topology. The E256/top-k8 shape fills useful M64 work
    // at M128; the sparser E384/top-k6 and GPT-OSS shapes retain M256.
    static constexpr unsigned kM64MinTokens =
        Config::kNumExperts == 256 ? 128 : 256;
    // GPT-OSS needs the lower-overhead four-wave tile through M512. Preserve
    // the existing eight-wave crossover for the denser E256/E384 shapes.
    static constexpr unsigned kM64W8MinTokens =
        Config::kNumExperts == 128 ? 1024 : kM64MinTokens;

    const unsigned external_input_count =
        static_cast<unsigned>(params.input_tokens != nullptr) +
        static_cast<unsigned>(params.input_topk_ids != nullptr) +
        static_cast<unsigned>(params.input_topk_weights != nullptr);

    if ((params.num_tokens != 0 && params.out == nullptr) ||
        params.w13 == nullptr || params.w2 == nullptr ||
        params.scales_w13 == nullptr || params.scales_w2 == nullptr ||
        params.workspace == nullptr ||
        params.hidden_size != Config::kComputeHiddenSize ||
        (params.output_row_stride != Config::kHiddenSize &&
         params.output_row_stride != Config::kComputeHiddenSize) ||
        params.inter_dim != Config::kInterDim ||
        (external_input_count != 0 && external_input_count != 3)) {
        return kFusedMoEErrorInvalidArgument;
    }

    const auto launch = [&]<class SelectedStage1Kernel,
                            class SelectedKernel>() -> int {
        if constexpr (Config::kNumRanks > 1) {
            hipLaunchKernelGGL(
                (MegaMoEStage1<SelectedStage1Kernel>),
                dim3(Config::kNumSMs),
                dim3(SelectedStage1Kernel::kThreads), 0, params.stream,
                reinterpret_cast<const uint4 *>(params.w13),
                params.scales_w13, params.num_tokens, params.w13_bias,
                params.workspace, params.rank,
                reinterpret_cast<const uint4 *>(params.input_tokens),
                params.input_topk_ids, params.input_topk_weights);
            if (hipGetLastError() != hipSuccess)
                return kFusedMoEErrorInvalidArgument;
            hipLaunchKernelGGL(
                (MegaMoEStage2<SelectedKernel>),
                dim3(SelectedKernel::kStage2GridBlocks),
                dim3(SelectedKernel::kThreads), 0, params.stream,
                reinterpret_cast<const uint4 *>(params.w2),
                params.scales_w2, params.w2_bias, params.workspace,
                params.rank);
        } else {
            hipLaunchKernelGGL(
                (MegaMoETwoStage<SelectedKernel>),
                dim3(Config::kNumSMs), dim3(Config::kThreads), 0,
                params.stream, reinterpret_cast<uint4 *>(params.out),
                reinterpret_cast<const uint4 *>(params.w13),
                reinterpret_cast<const uint4 *>(params.w2),
                params.scales_w13, params.scales_w2, params.num_tokens,
                params.output_row_stride, params.w13_bias, params.w2_bias,
                params.workspace, params.rank,
                reinterpret_cast<const uint4 *>(params.input_tokens),
                params.input_topk_ids, params.input_topk_weights);
        }
        if (hipGetLastError() != hipSuccess)
            return kFusedMoEErrorInvalidArgument;
        if constexpr (Config::kNumRanks > 1) {
            hipLaunchKernelGGL(
                (MegaMoECombine<CombineKernel>),
                dim3(CombineKernel::kNumSMs),
                dim3(CombineKernel::kThreads), 0, params.stream,
                reinterpret_cast<uint4 *>(params.out), params.num_tokens,
                params.output_row_stride, params.workspace, params.rank);
        }
        return hipGetLastError() == hipSuccess ? 0
                                               : kFusedMoEErrorInvalidArgument;
    };
    if constexpr (Config::kNumRanks > 1) {
        if (external_input_count == 3) {
            if (params.num_tokens >= kM64W8MinTokens) {
                return launch.template operator()<
                    M64W8ExternalInputStage1Kernel, ExternalInputKernel>();
            }
            if (params.num_tokens >= kM64MinTokens) {
                return launch.template operator()<
                    M64ExternalInputStage1Kernel, ExternalInputKernel>();
            }
            return launch.template operator()<ExternalInputKernel,
                                              ExternalInputKernel>();
        }
    } else if (external_input_count != 0) {
        return kFusedMoEErrorUnsupported;
    }
    if constexpr (Config::kNumRanks > 1) {
        if (params.num_tokens >= kM64W8MinTokens) {
            return launch.template operator()<M64W8Stage1Kernel, Kernel>();
        }
        if (params.num_tokens >= kM64MinTokens) {
            return launch.template operator()<M64Stage1Kernel, Kernel>();
        }
    }
    return launch.template operator()<Kernel, Kernel>();
}

template <unsigned long kRepr>
int MegaMoESolutionAdapter<kRepr>::GetWorkspaceInfo(
    unsigned rank, MegaMoEWorkspaceInfo *info) {
    static constexpr FusedMoESolutionId kSolution =
        FusedMoESolutionId::FromRepr(kRepr);
    using Config = MegaMoEConfigSelector<kSolution>;
    using Workspace = MegaMoEWorkspace<Config>;

    if (info == nullptr) {
        return kFusedMoEErrorInvalidArgument;
    }
    Workspace offsets(nullptr, rank);
    *info = {
        Workspace::XGpuBarrierRecordBytes(),
        Workspace::RankSymBufferBase(),
        Workspace::RankSymBufferSlotBytes(),
        Workspace::LocalOffsetBase(),
        Workspace::kLocalBytes,
        offsets.InputTokensOffset(rank),
        offsets.InputTokenTopKExpertIDOffset(),
        offsets.InputTokenTopKExpertWeightOffset(rank),
        Config::kInputTokenBytes,
        Config::kMaxTokensPerRank,
        Config::kNumRanks,
        Config::kNumExperts,
        Config::kTopK,
        Config::kHiddenSize,
        Config::kComputeHiddenSize,
        Config::kActDType,
    };
    return 0;
}

inline constexpr FusedMoESolutionId kMegaMoETwoStageMxFp4SolutionId =
    FusedMoESolutionId::MakeMegaBase(
        FusedMoEDataType::kMxFp4, FusedMoEDataType::kMxFp4,
        FusedMoEDataType::kBf16, FusedMoEWeightOrdering::kNativeMxFp4,
        FusedMoEMfmaShape::kMfmaScaleFp4MxFp4, FusedMoEStages::kTwoStage,
        FusedMoEActivationFunction::kOpenAISwiGLU,
        FusedMoEStage1Buffering::kDoubleBuffer);

inline constexpr FusedMoESolutionId kMegaMoETwoStageMxFp4SiluSolutionId =
    FusedMoESolutionId::MakeMegaBase(
        FusedMoEDataType::kMxFp4, FusedMoEDataType::kMxFp4,
        FusedMoEDataType::kNone, FusedMoEWeightOrdering::kNativeMxFp4,
        FusedMoEMfmaShape::kMfmaScaleFp4MxFp4, FusedMoEStages::kTwoStage,
        FusedMoEActivationFunction::kSiluDot,
        FusedMoEStage1Buffering::kDoubleBuffer);

} // namespace causalflow::petit::rocm::moe
