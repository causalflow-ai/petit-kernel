#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/amd_intrinsics.cuh"
#include "gemm/rocm/quantization/fp4/quantization_utils.cuh"
#include "memory_ops.cuh"
#include "moe/rocm/fused_moe.cuh"
#include "moe/rocm/ops/onestage_fused_moe_stage1.cuh"
#include "moe/rocm/ops/onestage_fused_moe_stage2.cuh"
#include "moe/rocm/ops/onestage_fused_moe_fp8_quantize_shuffle.cuh"
#include "fused_moe_blockscale_fp8_kernel.cuh"
#include "moe/rocm/quantization.cuh"
#include "moe/rocm/warp_schedule.cuh"

#include <cmath>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <class Config, unsigned kTokenBatch>
struct FusedMoEBlockScaleFP8Fp4Stage1Trait;
template <class Config> struct FusedMoEBlockScaleFP8Fp4Stage2Trait;

__device__ static inline float4 Fma4(float4 a, float4 s, float4 c) {
    const auto *a2 = reinterpret_cast<const float2 *>(&a);
    const auto *c2 = reinterpret_cast<const float2 *>(&c);
    const auto *s2 = reinterpret_cast<const float2 *>(&s);
    float4 r;
    auto *r2 = reinterpret_cast<float2 *>(&r);
    r2[0] = amdgcn_pk_fma_f32(s2[0], c2[0], a2[0]);
    r2[1] = amdgcn_pk_fma_f32(s2[1], c2[1], a2[1]);
    return r;
}

__device__ static inline float4 LoadFp8E8m0Scale(unsigned packed) {
    union {
        v4f f;
        unsigned u[4];
    };
    for (unsigned i = 0; i < 4; i++) {
        unsigned char b = (packed >> (i * 8)) & 0xFF;
        u[i] = (unsigned)b << 23;
    }
    // asm("v_lshlrev_b32_sdwa %0, 23, %4 dst_sel:DWORD dst_unused:UNUSED_PAD "
    //     "src0_sel:DWORD src1_sel:BYTE_0\r\n"
    //     "v_lshlrev_b32_sdwa %1, 23, %4 dst_sel:DWORD dst_unused:UNUSED_PAD "
    //     "src0_sel:DWORD src1_sel:BYTE_1\r\n"
    //     "v_lshlrev_b32_sdwa %2, 23, %4 dst_sel:DWORD dst_unused:UNUSED_PAD "
    //     "src0_sel:DWORD src1_sel:BYTE_2\r\n"
    //     "v_lshlrev_b32_sdwa %3, 23, %4 dst_sel:DWORD dst_unused:UNUSED_PAD "
    //     "src0_sel:DWORD src1_sel:BYTE_3\r\n"
    //     : "=v"(u[0]), "=v"(u[1]), "=v"(u[2]), "=v"(u[3])
    //     : "v"(packed));
    // BF8 conversion of FP4 data needs 2^15 compensation
    static const v4f kExpBias = {32768.0f, 32768.0f, 32768.0f, 32768.0f};
    f *= kExpBias;

    return reinterpret_cast<const float4 &>(f);
}

template <unsigned kLoadGlobal, unsigned kActivationFragments>
__device__ static inline void
MatmulBlockScaleFp4(float4 t[2 * kLoadGlobal], const uint4 w[kLoadGlobal],
                    const uint4 x[kActivationFragments], float4 x_scale,
                    unsigned packed_scale, unsigned stage) {
    const unsigned tid = threadIdx.x;
    const unsigned wtid = tid % kWarpSize;
    static constexpr unsigned kFragmentsPerRow = kActivationFragments / 2;
    for (unsigned i = 0; i < kLoadGlobal; i++) {
        static_assert(kLoadGlobal == 4, "");

        for (int j = 0; j < 4; j++) {
            const unsigned s = __shfl(packed_scale, (wtid & 48) + i * 4 + j);
            const float4 w_scale = LoadFp8E8m0Scale(s);
            const unsigned qw = reinterpret_cast<const unsigned *>(&w[i])[j];
            uint2 bf8;
            causalflow::petit::rocm::quantization::detail::Fp4ToBf8(
                reinterpret_cast<unsigned *>(&bf8), qw);

            for (int row = 0; row < 2; row++) {
                const uint2 *x_2 = reinterpret_cast<const uint2 *>(
                    x + row * kFragmentsPerRow + stage * 2);
                const float x_s =
                    reinterpret_cast<const float *>(&x_scale)[stage * 2 + row];

                const v4f x_s4 = {x_s, x_s, x_s, x_s};
                v4f fs;
                fs = reinterpret_cast<const v4f &>(w_scale) * x_s4;

                float4 m_acc{0, 0, 0, 0};
                m_acc = mma_m16n16k32_bf8_fp8_f32(bf8, x_2[j], m_acc);
                t[i * 2 + row] =
                    Fma4(t[i * 2 + row], reinterpret_cast<const float4 &>(fs),
                         m_acc);
            }
        }
    }
}

template <class Config, unsigned kTokenBatch>
struct FusedMoEBlockScaleFP8Fp4Stage1Trait {
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kActivationFragments = Config::kGroupDim / 32;
    static constexpr unsigned kKStages = Config::kGroupDim / 128;
    using Input = InputLayout<kTokenBatch, kNumWarps, Config::kGroupDim>;
    using W13 = MxFp4WeightLayout<kNumWarps, Config::kGroupDim>;

    Input &input;
    W13 &w1, &w3;
    uint4 w1_tile[kKStages][W13::kLoadGlobal];
    unsigned scale_w1[kKStages];
    static constexpr unsigned kAccumFragments = 2 * W13::kLoadGlobal;

    static_assert(kAccumFragments == Config::kGroupDim / 32, "");

    __device__ explicit FusedMoEBlockScaleFP8Fp4Stage1Trait(Input &input,
                                                            W13 &w1, W13 &w3)
        : input(input), w1(w1), w3(w3) {}

    __device__ void PrefetchInput(unsigned *shm_act, float *shm_scale,
                                  unsigned wid, unsigned wtid,
                                  const uint2 token_select,
                                  const unsigned tokens[Input::kTokenBatch],
                                  unsigned m) {
        input.FetchAsync(shm_act, wid, wtid, tokens);
        input.FetchScaleAsync(shm_scale, wid, wtid, token_select, m);
    }

    __device__ void LoadInitial(unsigned tid, unsigned wid, unsigned wtid) {
        for (unsigned j = 0; j < kKStages; ++j) {
            w1.LoadTile(w1_tile[j], j, wid, wtid);
            scale_w1[j] = w1.LoadScale(tid);
            w1.template AdvanceStep<0, 1>();
        }
    }

    __device__ void ReadInput(uint4 x[kActivationFragments], float4 &scale_x,
                              const unsigned *shm_act, const float *shm_scale,
                              unsigned wtid) const {
        input.FetchToRegsFP4(x, shm_act, wtid);
        scale_x = input.FetchScaleToReg(shm_scale, wtid);
    }

    __device__ void Matmul(float4 t_gate[kAccumFragments],
                           float4 t_up[kAccumFragments],
                           const uint4 x[kActivationFragments], float4 scale_x,
                           unsigned tid, unsigned wid, unsigned wtid) {
        uint4 w3_tile[kKStages][W13::kLoadGlobal];
        unsigned scale_w3[kKStages];
        for (unsigned j = 0; j < kKStages; j++) {
            w3.LoadTile(w3_tile[j], j, wid, wtid);
            scale_w3[j] = w3.LoadScale(tid);
            w3.template AdvanceStep<0, 1>();
            MatmulBlockScaleFp4<W13::kLoadGlobal, kActivationFragments>(
                t_gate, w1_tile[j], x, scale_x, scale_w1[j], j);
        }
        for (unsigned j = 0; j < kKStages; j++) {
            w1.LoadTile(w1_tile[j], j, wid, wtid);
            scale_w1[j] = w1.LoadScale(tid);
            w1.template AdvanceStep<0, 1>();
            MatmulBlockScaleFp4<W13::kLoadGlobal, kActivationFragments>(
                t_up, w3_tile[j], x, scale_x, scale_w3[j], j);
        }
    }
};

template <class Config> struct FusedMoEBlockScaleFP8Fp4KernelTrait {
    using Scalar = __hip_fp8_e4m3;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kTokenBatch = 8;
    using Input = InputLayout<kTokenBatch, kNumWarps, Config::kGroupDim>;
    using W13 = MxFp4WeightLayout<kNumWarps, Config::kGroupDim>;
    using W2 = MxFp4WeightLayout<kNumWarps, Config::kGroupN>;
    using Stage1Trait =
        FusedMoEBlockScaleFP8Fp4Stage1Trait<Config, kTokenBatch>;
    using Stage1Op =
        OnestageFusedMoEStage1SingleBufferOp<Stage1Trait,
                                                  Config::kGroupDim,
                                                  kTokenBatch>;

    using Stage2Trait = FusedMoEBlockScaleFP8Fp4Stage2Trait<Config>;
    using Stage2Op =
        OnestageFusedMoEStage2Op<Stage2Trait, Config::kGroupDim,
                                      kTokenBatch>;

    static constexpr unsigned kElementsPerThread =
        (Config::kGroupM * Config::kGroupN) / kThreads;
    static constexpr unsigned kElementsPerThreadVec4 = kElementsPerThread / 4;

    using QuantizationShuffleReadLayout =
        tal::Layout<tal::Shape<tal::C<kElementsPerThreadVec4>, tal::_2,
                               tal::Shape<tal::C<16>, tal::_4>>,
                    tal::Stride<tal::C<2 * kWarpSize>, tal::C<kWarpSize>,
                                tal::Stride<tal::_1, tal::_16>>>;
    using QuantizeAndShuffleOp =
        QuantizeAndShuffleFp8<kNumWarps, Config::kGroupN,
                              QuantizationShuffleReadLayout>;

    template <class Kernel>
    __device__ static void
    InitializeWeights(Kernel &kernel, const uint4 *w13_base, const uint4 *w2,
                      const unsigned *scales_w13, const unsigned *scales_w2,
                      unsigned expert_id, unsigned tile_k,
                      unsigned unused_input_scale_blocks,
                      unsigned unused_inter_dim_scale_blocks) {
        (void)unused_input_scale_blocks;
        (void)unused_inter_dim_scale_blocks;
        static_assert(Config::kGroupN == 256,
                      "The scale requires 256 elements per group dim");
        // 1 warp loads 4x64x128 blocks of scales (256B), 4 warps load
        // collectively
        static constexpr unsigned kScaleGroupK = 128;
        static constexpr unsigned kScaleGroupN = 64;
        static constexpr unsigned kLayoutN = 16;
        static constexpr unsigned kWeightVecSize = sizeof(uint4) * 2;
        static constexpr unsigned kRowGroupSize = W2::kRowGroupSize;
        const uint4 *w1_ptr =
            w13_base +
            expert_id * (2 * kernel.inter_dim_ * kernel.dim_) / kWeightVecSize +
            tile_k * Config::kGroupDim * kernel.dim_ / kWeightVecSize;
        const unsigned w13_scale_words_per_expert =
            (2 * kernel.dim_ * kernel.inter_dim_) / kRowGroupSize /
            (sizeof(unsigned) / sizeof(unsigned char));
        const unsigned w13_scale_words_per_col =
            (kernel.dim_ / kScaleGroupK) * (Config::kGroupN / kScaleGroupN) *
            kWarpSize;
        const unsigned *scale_w1_ptr = scales_w13 +
                                       expert_id * w13_scale_words_per_expert +
                                       tile_k * w13_scale_words_per_col;
        const unsigned w13_value_range = Config::kGroupDim * kernel.dim_ / 2;
        const unsigned w13_scale_range =
            w13_scale_words_per_col * sizeof(unsigned);
        kernel.w1_.Initialize(w1_ptr, w13_value_range, scale_w1_ptr,
                              w13_scale_range, kernel.dim_);

        kernel.w3_.Initialize(
            w1_ptr + kernel.inter_dim_ * kernel.dim_ / kWeightVecSize,
            w13_value_range, scale_w1_ptr + w13_scale_words_per_expert / 2,
            w13_scale_range, kernel.dim_);

        const unsigned w2_value_k_tile_offset =
            tile_k * Config::kStage2GroupInterDim * kLayoutN / kWeightVecSize;
        const uint4 *w2_ptr =
            w2 +
            expert_id * (kernel.dim_ * kernel.inter_dim_) / kWeightVecSize +
            w2_value_k_tile_offset;
        const unsigned w2_scale_words_per_expert =
            (kernel.dim_ * kernel.inter_dim_) / W2::kRowGroupSize /
            (sizeof(unsigned) / sizeof(unsigned char));
        const unsigned w2_scale_k_tile_offset =
            tile_k * (Config::kStage2GroupInterDim / kScaleGroupK) *
            (Config::kGroupN / kScaleGroupN) * kWarpSize;
        const unsigned *scale_w2_ptr = scales_w2 +
                                       expert_id * w2_scale_words_per_expert +
                                       w2_scale_k_tile_offset;
        const unsigned w2_value_range = (kernel.inter_dim_ * kernel.dim_) / 2 -
                                        w2_value_k_tile_offset * sizeof(uint4);
        const unsigned w2_scale_range =
            w2_scale_words_per_expert * sizeof(unsigned) -
            w2_scale_k_tile_offset * sizeof(unsigned);

        kernel.w2_.Initialize(w2_ptr, w2_value_range, scale_w2_ptr,
                              w2_scale_range, kernel.inter_dim_);
    }
};

template <class Config>
using FusedMoEBlockScaleFP8Fp4Kernel =
    OnestageFusedMoEBlockScaleFP8<Config,
                                  FusedMoEBlockScaleFP8Fp4KernelTrait<Config>>;

template <class Config> struct FusedMoEBlockScaleFP8Fp4Stage2Trait {
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    using Schedule = WarpSchedule<Config>;
    static constexpr int kKStages = Config::kGroupN / 128;
    using W2 = MxFp4WeightLayout<kNumWarps, Config::kGroupN>;
    W2 &w2;
    uint4 w2_tile[2][kKStages][W2::kLoadGlobal];
    unsigned scale_w2[2][kKStages];
    static constexpr unsigned kAccumFragments = 2 * W2::kLoadGlobal;
    static constexpr unsigned kActivationFragments = Config::kGroupDim / 32;
    static constexpr unsigned kOutputPacksPerToken = Config::kGroupN / 128;

    static_assert(kAccumFragments == Schedule::kAccumulatorFragments, "");
    static_assert(kActivationFragments == Config::kGroupDim / 32, "");
    static_assert(kOutputPacksPerToken > 0, "");

    __device__ explicit FusedMoEBlockScaleFP8Fp4Stage2Trait(W2 &w2) : w2(w2) {}

    __device__ void LoadStage(unsigned stage, unsigned tid, unsigned wid,
                              unsigned wtid) {
        for (unsigned j = 0; j < kKStages; ++j) {
            w2.LoadTile(w2_tile[stage][j], j, wid, wtid);
            scale_w2[stage][j] = w2.LoadScale(tid);
            w2.template AdvanceStep<0, 1>();
        }
        w2.template AdvanceStep<Config::kGroupN / 128, -kKStages>();
    }

    __device__ void Matmul(float4 t[kAccumFragments],
                           const uint4 quant_h[kActivationFragments],
                           float4 dq_act, unsigned stage, unsigned wtid) {
        (void)wtid;
        for (unsigned j = 0; j < kKStages; ++j) {
            MatmulBlockScaleFp4<W2::kLoadGlobal, kActivationFragments>(
                t, w2_tile[stage][j], quant_h, dq_act, scale_w2[stage][j], j);
        }
    }
};

struct FusedMoEMxFp4Config {
    static constexpr unsigned kGroupM = 32;
    static constexpr unsigned kGroupN = 256;
    // For stage1 how many elements are processed over N for all 4 warps
    static constexpr unsigned kGroupDim = 256;
    static constexpr unsigned kNumWarps = 4;

    // Stage2 consumes the same inter-dim slice selected by blockIdx.x as
    // stage1. The two stage2 pipeline stages advance over output dim tiles,
    // not over additional inter-dim tiles.
    static constexpr unsigned kStage2GroupInterDim = kGroupDim;
};

} // namespace causalflow::petit::rocm::moe
