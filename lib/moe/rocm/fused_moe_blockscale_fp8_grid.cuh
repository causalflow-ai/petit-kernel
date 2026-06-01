#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/amd_intrinsics.cuh"
#include "memory_ops.cuh"
#include "moe/rocm/fused_moe.cuh"
#include "moe/rocm/ops/op_stage1.cuh"
#include "moe/rocm/ops/op_stage2.cuh"
#include "moe/rocm/ops/onestage_fused_moe_fp8_quantize_shuffle.cuh"
#include "fused_moe_blockscale_fp8_kernel.cuh"
#include "moe/rocm/ops/schedule_stage1.cuh"
#include "moe/rocm/ops/schedule_stage2.cuh"
#include "moe/rocm/quantization.cuh"
#include "moe/rocm/mem/input_channel_scale_fp8.cuh"
#include "moe/rocm/mem/weight_blockscale_fp8.cuh"

#include <cmath>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <class Config_> struct FusedMoEBlockScaleFP8KernelTrait {
    using Config = Config_;
    using Scalar = __hip_fp8_e4m3;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kTokenBatch = 8;
    using Weights = BlockScaleFp8Weights<Config>;
    using W13Weights = typename Weights::W13Weights;
    using W2Weights = typename Weights::W2Weights;
    using Input = ChannelScaleFp8Input<Config>;
    using W2 = typename W2Weights::W2;
    using W13 = typename W13Weights::W13;
    using Stage1Trait = BlockScaleFp8Stage1Schedule<Config, kTokenBatch>;
    using Stage1Op =
        OnestageFusedMoEStage1DoubleBufferOp<Stage1Trait,
                                             Config::kGroupDim, kTokenBatch>;
    using Stage2Trait = BlockScaleFp8Stage2Schedule<Config>;
    using Stage2Op =
        OnestageFusedMoEStage2Op<Stage2Trait, Config::kGroupDim,
                                 kTokenBatch>;

    static constexpr unsigned kElementsPerThread =
        (Config::kGroupM * Config::kGroupN) / kThreads;
    static constexpr unsigned kElementsPerThreadVec4 = kElementsPerThread / 4;

    using QuantizationShuffleReadLayout =
        tal::Layout<tal::Shape<tal::C<kElementsPerThreadVec4>, tal::_2,
                               tal::Shape<tal::C<16>, tal::C<kNumWarps>>>,
                    tal::Stride<tal::C<2 * kWarpSize>, tal::_16,
                                tal::Stride<tal::_1, tal::C<32>>>>;
    using QuantizeAndShuffleOp =
        QuantizeAndShuffleFp8<kNumWarps, Config::kGroupN,
                              QuantizationShuffleReadLayout>;

    template <class Kernel>
    __device__ static void
    InitializeWeights(Kernel &kernel, const uint4 *w13_base, const uint4 *w2,
                      const unsigned *scales_w13, const unsigned *scales_w2,
                      unsigned expert_id, unsigned tile_k, unsigned n_blocks,
                      unsigned k_blocks) {
        kernel.w13_weights_.Initialize(w13_base, scales_w13, expert_id,
                                       tile_k, n_blocks, k_blocks,
                                       kernel.dim_, kernel.inter_dim_);
        kernel.w2_weights_.Initialize(w2, scales_w2, expert_id, tile_k,
                                      n_blocks, k_blocks, kernel.dim_,
                                      kernel.inter_dim_);
    }
};

template <class Config>
using FusedMoEBlockScaleFP8Kernel =
    OnestageFusedMoEBlockScaleFP8<Config,
                                  FusedMoEBlockScaleFP8KernelTrait<Config>>;

struct FusedMoEConfig {
    static constexpr unsigned kGroupM = 32;
    static constexpr unsigned kGroupN = 256;
    static constexpr unsigned kGroupDim = 256;
    static constexpr unsigned kNumWarps = 4;
    static constexpr unsigned kStage2GroupInterDim = kGroupDim;
};

} // namespace causalflow::petit::rocm::moe
