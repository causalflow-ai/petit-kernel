#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "fused_moe_blockscale_fp8_kernel.cuh"
#include "moe/rocm/ops/schedule_tiles.cuh"
#include "gemm/rocm/amd_intrinsics.cuh"
#include "gemm/rocm/quantization/fp4/quantization_utils.cuh"
#include "memory_ops.cuh"
#include "moe/rocm/fused_moe.cuh"
#include "moe/rocm/ops/activation.cuh"
#include "moe/rocm/ops/quantize_and_shuffle.cuh"
#include "moe/rocm/ops/op_stages.cuh"
#include "moe/rocm/quantization.cuh"
#include "moe/rocm/mem/bias.cuh"
#include "moe/rocm/mem/input_bf16.cuh"
#include "moe/rocm/mem/input_channel_scale_fp8.cuh"
#include "moe/rocm/mem/input_mxfp4.cuh"
#include "moe/rocm/mem/weight_blockscale_fp8.cuh"
#include "moe/rocm/mem/weight_mxfp4.cuh"
#include "moe/rocm/fused_moe_2stage_kernel.cuh"

#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <FusedMoEDataType kBiasDType, FusedMoEDataType kWeightDType,
          FusedMoEMfmaShape kMfma, unsigned kNumWarps, unsigned kGroupN,
          unsigned kGroupM = 32>
struct BiasLayoutSelector;

template <FusedMoEDataType kWeightDType, FusedMoEMfmaShape kMfma,
          unsigned kNumWarps, unsigned kGroupN, unsigned kGroupM>
struct BiasLayoutSelector<FusedMoEDataType::kNone, kWeightDType, kMfma,
                          kNumWarps, kGroupN, kGroupM> {
    using Type = NoopBiasLayout<kNumWarps, kGroupN>;
};

template <FusedMoEDataType, FusedMoEMfmaShape, unsigned kGroupM,
          unsigned kGroupN>
struct BiasMemoryLayoutSelector;

template <unsigned kGroupM, unsigned kGroupN>
struct BiasMemoryLayoutSelector<FusedMoEDataType::kMxFp4,
                                FusedMoEMfmaShape::kMfmaFp816x16x32,
                                kGroupM, kGroupN> {
    using Type = typename MxFp4BiasLayout<kGroupM, kGroupN>::Type;
};

template <unsigned kGroupM, unsigned kGroupN>
struct BiasMemoryLayoutSelector<FusedMoEDataType::kMxFp4,
                                FusedMoEMfmaShape::kMfmaBf16MxFp4,
                                kGroupM, kGroupN> {
    using Type = typename MxFp4BiasLayout<kGroupM, kGroupN>::Type;
};

template <unsigned kGroupM, unsigned kGroupN>
struct BiasMemoryLayoutSelector<FusedMoEDataType::kMxFp4,
                                FusedMoEMfmaShape::kMfmaScaleFp4MxFp4,
                                kGroupM, kGroupN> {
    using Type = typename MxFp4BiasLayout<kGroupM, kGroupN>::Type;
};

template <FusedMoEDataType kWeightDType, FusedMoEMfmaShape kMfma,
          unsigned kNumWarps, unsigned kGroupN, unsigned kGroupM>
struct BiasLayoutSelector<FusedMoEDataType::kBf16, kWeightDType, kMfma,
                          kNumWarps, kGroupN, kGroupM> {
    static constexpr unsigned kWarpsM = kGroupM / 32;
    static constexpr unsigned kWarpsN = kNumWarps / kWarpsM;
    static constexpr unsigned kLoadGlobal = kGroupN / kWarpsN / 16;
    using Type = Bf16BiasLayout<
        kNumWarps, kGroupN,
        typename BiasMemoryLayoutSelector<kWeightDType, kMfma, kGroupM,
                                          kGroupN>::Type,
        kLoadGlobal>;
};

template <unsigned kGroupM, unsigned kGroupN>
struct MxFp4TileShapeSelector;

template <> struct MxFp4TileShapeSelector<32, 128> {
    static constexpr MxFp4TileShape value = MxFp4TileShape::kN128;
};

template <> struct MxFp4TileShapeSelector<32, 256> {
    static constexpr MxFp4TileShape value = MxFp4TileShape::kN256;
};

template <> struct MxFp4TileShapeSelector<64, 256> {
    static constexpr MxFp4TileShape value = MxFp4TileShape::kM64N256;
};

template <FusedMoEDataType kWeightDType, FusedMoEWeightOrdering kWeightOrdering,
          class Config>
struct FusedMoEWeightSelector;

template <class Config>
struct FusedMoEWeightSelector<FusedMoEDataType::kBlockScaleFp8,
                              FusedMoEWeightOrdering::kPetitFp8, Config> {
    using Weights = BlockScaleFp8Weights<Config>;
    using W13Weights = typename Weights::W13Weights;
    using W2Weights = typename Weights::W2Weights;
    using W13 = typename W13Weights::W13;
    using W2 = typename W2Weights::W2;

    template <class Kernel>
    __device__ static void
    InitializeW13(Kernel &kernel, const uint4 *w13_base,
                  const unsigned *scales_w13, unsigned expert_id,
                  unsigned tile_k, unsigned n_blocks, unsigned k_blocks) {
        kernel.w13_weights_.Initialize(w13_base, scales_w13, expert_id, tile_k,
                                       n_blocks, k_blocks, Config::kDim,
                                       Config::kInterDim);
    }

    template <class Kernel>
    __device__ static void
    InitializeW2(Kernel &kernel, const uint4 *w2,
                 const unsigned *scales_w2, unsigned expert_id,
                 unsigned tile_k, unsigned n_blocks, unsigned k_blocks) {
        kernel.w2_weights_.Initialize(w2, scales_w2, expert_id, tile_k,
                                      n_blocks, k_blocks, Config::kDim,
                                      Config::kInterDim);
    }
};

template <class Config>
struct FusedMoEWeightSelector<FusedMoEDataType::kMxFp4,
                              FusedMoEWeightOrdering::kPetitMxFp4, Config> {
    using Weights = MxFp4Weights<Config>;
    using W13Weights = typename Weights::W13Weights;
    using W2Weights = typename Weights::W2Weights;
    using W13 = typename W13Weights::W13;
    using W2 = typename W2Weights::W2;

    template <class Kernel>
    __device__ static void
    InitializeW13(Kernel &kernel, const uint4 *w13_base,
                  const unsigned *scales_w13, unsigned expert_id,
                  unsigned tile_k, unsigned, unsigned) {
        kernel.w13_weights_.Initialize(w13_base, scales_w13, expert_id, tile_k,
                                       Config::kDim, Config::kInterDim);
    }

    template <class Kernel>
    __device__ static void
    InitializeW2(Kernel &kernel, const uint4 *w2,
                 const unsigned *scales_w2, unsigned expert_id,
                 unsigned tile_k, unsigned, unsigned) {
        kernel.w2_weights_.Initialize(w2, scales_w2, expert_id, 0, tile_k);
    }
};

template <class Config>
struct FusedMoEWeightSelector<FusedMoEDataType::kMxFp4,
                              FusedMoEWeightOrdering::kNativeMxFp4, Config> {
    using Weights = MxFp4Weights<Config>;
    using W13Weights = typename Weights::W13Weights;
    using W2Weights = typename Weights::W2Weights;
    using W13 = typename W13Weights::W13;
    using W2 = typename W2Weights::W2;

    template <class Kernel>
    __device__ static void
    InitializeW13(Kernel &kernel, const uint4 *w13_base,
                  const unsigned *scales_w13, unsigned expert_id,
                  unsigned tile_k, unsigned, unsigned) {
        kernel.w13_weights_.Initialize(w13_base, scales_w13, expert_id, tile_k,
                                       Config::kDim, Config::kInterDim);
    }

    template <class Kernel>
    __device__ static void
    InitializeW2(Kernel &kernel, const uint4 *w2,
                 const unsigned *scales_w2, unsigned expert_id,
                 unsigned tile_k, unsigned, unsigned) {
        kernel.w2_weights_.Initialize(w2, scales_w2, expert_id, 0, tile_k);
    }
};

template <FusedMoEDataType kWeightDType,
          FusedMoEWeightOrdering kWeightOrdering, FusedMoEMfmaShape kMfma,
          class Config>
struct FusedMoEStage1TilesSelector;

template <class Config>
struct FusedMoEStage1TilesSelector<
    FusedMoEDataType::kBlockScaleFp8, FusedMoEWeightOrdering::kPetitFp8,
    FusedMoEMfmaShape::kMfmaFp816x16x32, Config> {
    using Type =
        W13TileSchedule<BlockScaleFp8TileOps<Config, typename Config::W13>>;
};

template <class Config>
struct FusedMoEStage1TilesSelector<
    FusedMoEDataType::kMxFp4, FusedMoEWeightOrdering::kPetitMxFp4,
    FusedMoEMfmaShape::kMfmaFp816x16x32, Config> {
    using Type =
        W13TileSchedule<PetitMxFp4TileOps<Config, typename Config::W13>>;
};

template <class Config>
struct FusedMoEStage1TilesSelector<
    FusedMoEDataType::kMxFp4, FusedMoEWeightOrdering::kNativeMxFp4,
    FusedMoEMfmaShape::kMfmaBf16MxFp4, Config> {
    using Type =
        W13TileSchedule<Bf16MxFp4TileOps<Config, typename Config::W13>>;
};

template <class Config>
struct FusedMoEStage1TilesSelector<
    FusedMoEDataType::kMxFp4, FusedMoEWeightOrdering::kNativeMxFp4,
    FusedMoEMfmaShape::kMfmaScaleFp4MxFp4, Config> {
    using Type =
        W13TileSchedule<NativeMxFp4TileOps<Config, typename Config::W13>>;
};

template <FusedMoEDataType kWeightDType,
          FusedMoEWeightOrdering kWeightOrdering, FusedMoEMfmaShape kMfma,
          class Config>
struct FusedMoEStage2TilesSelector;

template <class Config>
struct FusedMoEStage2TilesSelector<
    FusedMoEDataType::kBlockScaleFp8, FusedMoEWeightOrdering::kPetitFp8,
    FusedMoEMfmaShape::kMfmaFp816x16x32, Config> {
    using Type =
        W2TileSchedule<BlockScaleFp8TileOps<Config, typename Config::W2>>;
};

template <class Config>
struct FusedMoEStage2TilesSelector<
    FusedMoEDataType::kMxFp4, FusedMoEWeightOrdering::kPetitMxFp4,
    FusedMoEMfmaShape::kMfmaFp816x16x32, Config> {
    using Type = W2TileSchedule<PetitMxFp4TileOps<Config, typename Config::W2>>;
};

template <class Config>
struct FusedMoEStage2TilesSelector<
    FusedMoEDataType::kMxFp4, FusedMoEWeightOrdering::kNativeMxFp4,
    FusedMoEMfmaShape::kMfmaBf16MxFp4, Config> {
    using Type = W2TileSchedule<Bf16MxFp4TileOps<Config, typename Config::W2>>;
};

template <class Config>
struct FusedMoEStage2TilesSelector<
    FusedMoEDataType::kMxFp4, FusedMoEWeightOrdering::kNativeMxFp4,
    FusedMoEMfmaShape::kMfmaScaleFp4MxFp4, Config> {
    using Type = W2TileSchedule<NativeMxFp4TileOps<Config, typename Config::W2>>;
};

template <FusedMoEActivationFunction kActivation>
struct FusedMoEActivationSelector {};

template <>
struct FusedMoEActivationSelector<FusedMoEActivationFunction::kSiluDot> {
    using Type = SiluDotOp;
};

template <>
struct FusedMoEActivationSelector<FusedMoEActivationFunction::kOpenAISwiGLU> {
    using Type = OpenAISwiGLUOp;
};

template <FusedMoEStage1Buffering kBuffering, class Stage1Tiles>
struct FusedMoEStage1OpSelector {
};

template <class Stage1Tiles>
struct FusedMoEStage1OpSelector<FusedMoEStage1Buffering::kSingleBuffer,
                                Stage1Tiles> {
    using Type = OnestageFusedMoEStage1SingleBufferOp<Stage1Tiles>;
};

template <class Stage1Tiles>
struct FusedMoEStage1OpSelector<FusedMoEStage1Buffering::kDoubleBuffer,
                                Stage1Tiles> {
    using Type = OnestageFusedMoEStage1DoubleBufferOp<Stage1Tiles>;
};

template <FusedMoEDataType kActDType, class Config>
struct FusedMoEInputSelector;

template <class Config>
struct FusedMoEInputSelector<FusedMoEDataType::kChannelScaleFp8, Config> {
    using Type = ChannelScaleFp8Input<Config>;
};

template <class Config>
struct FusedMoEInputSelector<FusedMoEDataType::kBf16, Config> {
    using Type = Bf16Input<Config>;
};

template <class Config>
struct FusedMoEInputSelector<FusedMoEDataType::kMxFp4, Config> {
    using Type = MxFp4Input<Config>;
};

template <FusedMoEDataType kWeightDType>
struct Fp8QuantizeShufflePolicySelector;

template <>
struct Fp8QuantizeShufflePolicySelector<
    FusedMoEDataType::kBlockScaleFp8> {
    using Type = BlockScaleFp8QuantizeShufflePolicy;
};

template <>
struct Fp8QuantizeShufflePolicySelector<FusedMoEDataType::kMxFp4> {
    using Type = PetitMxFp4QuantizeShufflePolicy;
};

template <FusedMoEDataType kActDType, class Config>
struct HiddenShuffleSelector {
    using Type = QuantizeAndShuffleFp8<
        Config, typename Fp8QuantizeShufflePolicySelector<
                    Config::kWeightDType>::Type>;
};

template <class Config>
struct HiddenShuffleSelector<FusedMoEDataType::kBf16, Config> {
    using Type =
        PackAndShuffleBf16<Config::kNumWarps, Config::kGroupN,
                           typename Config::Stage2Tiles::InputRegs>;
};

template <class Config>
struct HiddenShuffleSelector<FusedMoEDataType::kMxFp4, Config> {
    using Type =
        QuantizeAndShuffleMxFp4<Config::kNumWarps, Config::kGroupN,
                                typename Config::Stage2Tiles::InputRegs>;
};

template <FusedMoESolutionId id, unsigned kTopK_ = 4>
struct ConfigSelector {
    using Self = ConfigSelector<id, kTopK_>;
    static constexpr unsigned kDim = id.Dim();
    static constexpr unsigned kInterDim = id.InterDim();
    static constexpr unsigned kTopK = kTopK_;
    static_assert(FusedMoESolutionId::IsShapeEncodable(kDim, kInterDim));
    static constexpr unsigned kGroupM = id.Stage1TileM();
    static constexpr unsigned kGroupN = 256;
    static constexpr unsigned kStage1GroupN =
        id.stages == FusedMoEStages::kTwoStage ? id.Stage1TileN() / 2
                                               : kGroupN;
    static constexpr unsigned kGroupDim = 256;
    static constexpr unsigned kNumWarps = 4;
    static constexpr unsigned kTokenBatch = kGroupM / kNumWarps;
    static constexpr unsigned kStage1WarpsM = kGroupM / 32;
    static constexpr unsigned kStage1WarpsN = kNumWarps / kStage1WarpsM;
    static constexpr unsigned kStage2GroupM = 32;
    static constexpr unsigned kStage2TokenBatch = 8;
    static constexpr unsigned kStage1ToStage2GroupRatio =
        kGroupM / kStage2GroupM;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kStage2GroupInterDim = kGroupDim;
    static constexpr FusedMoEDataType kActDType = id.act_dtype;
    static constexpr FusedMoEDataType kWeightDType = id.weight_dtype;
    static constexpr FusedMoEMfmaShape kMfmaShape = id.mfma;
    static constexpr bool kValidateExpertIds = false;
    static constexpr MxFp4TileShape kW13TileShape =
        MxFp4TileShapeSelector<kGroupM, kStage1GroupN>::value;
    static constexpr MxFp4TileShape kW2TileShape = MxFp4TileShape::kN256;

    static_assert(kGroupM == 32 ||
                  (id.stages == FusedMoEStages::kTwoStage &&
                   id.act_dtype == FusedMoEDataType::kMxFp4 &&
                   id.weight_dtype == FusedMoEDataType::kMxFp4 &&
                   id.weight_ordering == FusedMoEWeightOrdering::kNativeMxFp4 &&
                   id.mfma == FusedMoEMfmaShape::kMfmaScaleFp4MxFp4 &&
                   id.bias_dtype == FusedMoEDataType::kBf16));

    using ActivationOp =
        typename FusedMoEActivationSelector<id.activation>::Type;
    using Input = typename FusedMoEInputSelector<id.act_dtype, Self>::Type;
    using Weight =
        FusedMoEWeightSelector<id.weight_dtype, id.weight_ordering, Self>;
    using W13Weights = typename Weight::W13Weights;
    using W2Weights = typename Weight::W2Weights;
    using W13 = typename W13Weights::W13;
    using W2 = typename W2Weights::W2;
    using Bias = typename BiasLayoutSelector<
        id.bias_dtype, id.weight_dtype, id.mfma, kNumWarps, kStage1GroupN,
        kGroupM>::Type;
    using Stage2Bias =
        typename BiasLayoutSelector<id.bias_dtype, id.weight_dtype, id.mfma,
                                    kNumWarps, kGroupN>::Type;
    using Stage1Tiles = typename FusedMoEStage1TilesSelector<
        id.weight_dtype, id.weight_ordering, id.mfma, Self>::Type;
    using Stage1Op =
        typename FusedMoEStage1OpSelector<id.stage1_buffering,
                                          Stage1Tiles>::Type;
    using Stage2Tiles = typename FusedMoEStage2TilesSelector<
        id.weight_dtype, id.weight_ordering, id.mfma, Self>::Type;
    using Stage2Op = OnestageFusedMoEStage2Op<Stage2Tiles>;
    using Weights = typename Weight::Weights;
    using QuantizeAndShuffleOp =
        typename HiddenShuffleSelector<id.act_dtype, Self>::Type;

    template <class Kernel>
    __device__ static void
    InitializeW13(Kernel &kernel, const uint4 *w13_base,
                  const unsigned *scales_w13, unsigned expert_id,
                  unsigned tile_k, unsigned n_blocks, unsigned k_blocks) {
        Weight::InitializeW13(kernel, w13_base, scales_w13, expert_id, tile_k,
                              n_blocks, k_blocks);
    }

    template <class Kernel>
    __device__ static void
    InitializeW2(Kernel &kernel, const uint4 *w2,
                 const unsigned *scales_w2, unsigned expert_id,
                 unsigned tile_k, unsigned n_blocks, unsigned k_blocks) {
        Weight::InitializeW2(kernel, w2, scales_w2, expert_id, tile_k,
                             n_blocks, k_blocks);
    }

    static int Invoke(FusedMoE1StageParams params) {
        static_assert(kDim % kQuantBlockK == 0);
        static_assert(kInterDim % kGroupDim == 0);
        static constexpr unsigned kSplitK = kInterDim / kGroupDim;
        using Epilogue = OnestageFusedMoEStage1Epilogue<Self>;
        using Kernel = FusedMoEStage1<Self, Epilogue>;

        if (params.out == nullptr || params.num_valid_ids == nullptr) {
            return kFusedMoEErrorInvalidArgument;
        }

        if (params.n != kDim || params.k != kInterDim) {
            return kFusedMoEErrorInvalidArgument;
        }

        if (params.act == nullptr || params.w13 == nullptr ||
            params.w2 == nullptr || params.sorted_token_ids == nullptr ||
            params.sorted_weights == nullptr ||
            params.sorted_expert_ids == nullptr ||
            (params.scales_act == nullptr &&
             id.act_dtype != FusedMoEDataType::kBf16) ||
            params.scales_w13 == nullptr || params.scales_w2 == nullptr) {
            return kFusedMoEErrorInvalidArgument;
        }
        if (params.m == 0 || params.topk == 0 || params.max_num_m_blocks == 0) {
            return 0;
        }

        unsigned route_groups = params.max_num_m_blocks;
        unsigned persistent_route_step = 0;
        if (params.num_persistent_tgs > 0) {
            unsigned persistent_route_groups =
                tal::CeilingDiv<unsigned>(params.num_persistent_tgs, kSplitK);
            if (persistent_route_groups == 0) {
                persistent_route_groups = 1;
            }
            route_groups = persistent_route_groups < route_groups
                               ? persistent_route_groups
                               : route_groups;
            persistent_route_step = route_groups;
        }

        const dim3 blocks(64 * kNumWarps);
        const dim3 grids(kSplitK, route_groups, 1);
        const unsigned num_experts =
            kValidateExpertIds ? params.num_experts : 0;
        OnestageFusedMoEBlockScaleFP8Compute<Kernel>
            <<<grids, blocks, 0, params.stream>>>(
                reinterpret_cast<uint4 *>(params.out),
                reinterpret_cast<const uint4 *>(params.act),
                reinterpret_cast<const uint4 *>(params.w13),
                reinterpret_cast<const uint4 *>(params.w2),
                reinterpret_cast<const uint4 *>(params.sorted_token_ids),
                reinterpret_cast<const uint4 *>(params.sorted_weights),
                reinterpret_cast<const uint4 *>(params.sorted_expert_ids),
                params.num_valid_ids, params.topk,
                reinterpret_cast<const uint4 *>(params.scales_act),
                reinterpret_cast<const uint4 *>(params.scales_w13),
                params.scales_w2, params.m, num_experts, persistent_route_step,
                params.w13_bias, params.w2_bias);

        const auto e = hipGetLastError();
        return e == hipSuccess ? 0 : kFusedMoEErrorInvalidArgument;
    }
};

template <unsigned long kRepr>
int FusedMoESolutionAdapter<kRepr>::Invoke(FusedMoE1StageParams params) {
    static constexpr FusedMoESolutionId kSolId =
        FusedMoESolutionId::FromRepr(kRepr);
    using Impl = ConfigSelector<kSolId>;
    return Impl::Invoke(params);
}

static constexpr FusedMoESolutionId kFusedMoEBlockScaleFp8SolutionId =
    FusedMoESolutionId::MakeBase(
        FusedMoEDataType::kChannelScaleFp8, FusedMoEDataType::kBlockScaleFp8,
        FusedMoEDataType::kNone, FusedMoEWeightOrdering::kPetitFp8,
        FusedMoEMfmaShape::kMfmaFp816x16x32, FusedMoEStages::kOneStage,
        FusedMoEActivationFunction::kSiluDot,
        FusedMoEStage1Buffering::kDoubleBuffer);

static constexpr FusedMoESolutionId kFusedMoEFp8PetitMxFp4SolutionId =
    FusedMoESolutionId::MakeBase(
        FusedMoEDataType::kChannelScaleFp8, FusedMoEDataType::kMxFp4,
        FusedMoEDataType::kNone, FusedMoEWeightOrdering::kPetitMxFp4,
        FusedMoEMfmaShape::kMfmaFp816x16x32, FusedMoEStages::kOneStage,
        FusedMoEActivationFunction::kSiluDot,
        FusedMoEStage1Buffering::kSingleBuffer);

static constexpr FusedMoESolutionId kFusedMoEFp8PetitMxFp4BiasSolutionId =
    FusedMoESolutionId::MakeBase(
        FusedMoEDataType::kChannelScaleFp8, FusedMoEDataType::kMxFp4,
        FusedMoEDataType::kBf16, FusedMoEWeightOrdering::kPetitMxFp4,
        FusedMoEMfmaShape::kMfmaFp816x16x32, FusedMoEStages::kOneStage,
        FusedMoEActivationFunction::kOpenAISwiGLU,
        FusedMoEStage1Buffering::kDoubleBuffer);

static constexpr FusedMoESolutionId kFusedMoEBf16NativeMxFp4BiasSolutionId =
    FusedMoESolutionId::MakeBase(
        FusedMoEDataType::kBf16, FusedMoEDataType::kMxFp4,
        FusedMoEDataType::kBf16, FusedMoEWeightOrdering::kNativeMxFp4,
        FusedMoEMfmaShape::kMfmaBf16MxFp4, FusedMoEStages::kOneStage,
        FusedMoEActivationFunction::kOpenAISwiGLU,
        FusedMoEStage1Buffering::kDoubleBuffer);

static constexpr FusedMoESolutionId kFusedMoEMxFp4NativeMxFp4BiasSolutionId =
    FusedMoESolutionId::MakeBase(
        FusedMoEDataType::kMxFp4, FusedMoEDataType::kMxFp4,
        FusedMoEDataType::kBf16, FusedMoEWeightOrdering::kNativeMxFp4,
        FusedMoEMfmaShape::kMfmaScaleFp4MxFp4, FusedMoEStages::kOneStage,
        FusedMoEActivationFunction::kOpenAISwiGLU,
        FusedMoEStage1Buffering::kDoubleBuffer);

static constexpr FusedMoESolutionId kFusedMoETwoStageMxFp4BiasSolutionId =
    FusedMoESolutionId::Make(
        FusedMoEDataType::kMxFp4, FusedMoEDataType::kMxFp4,
        FusedMoEDataType::kBf16, FusedMoEWeightOrdering::kNativeMxFp4,
        FusedMoEMfmaShape::kMfmaScaleFp4MxFp4, FusedMoEStages::kTwoStage,
        FusedMoEActivationFunction::kOpenAISwiGLU,
        FusedMoEStage1Buffering::kDoubleBuffer, 3072, 3072);

static constexpr FusedMoESolutionId kFusedMoETwoStageMxFp4SiluSolutionId =
    FusedMoESolutionId::MakeBase(
        FusedMoEDataType::kMxFp4, FusedMoEDataType::kMxFp4,
        FusedMoEDataType::kNone, FusedMoEWeightOrdering::kNativeMxFp4,
        FusedMoEMfmaShape::kMfmaScaleFp4MxFp4, FusedMoEStages::kTwoStage,
        FusedMoEActivationFunction::kSiluDot,
        FusedMoEStage1Buffering::kDoubleBuffer);

} // namespace causalflow::petit::rocm::moe
