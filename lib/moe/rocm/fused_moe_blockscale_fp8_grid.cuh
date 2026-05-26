#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/amd_intrinsics.cuh"
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

template <class T> __device__ static inline T GetDppValue(T src, int ctrl) {
    static constexpr unsigned kDppRowNewBcastBase = 0x150;
    static_assert(sizeof(T) == sizeof(int), "");

    auto bits = reinterpret_cast<const int &>(src);
    int dst;
    if (ctrl == 0) {
        dst = __builtin_amdgcn_mov_dpp(bits, kDppRowNewBcastBase, 0xf, 0xf, 0);
    } else if (ctrl == 1) {
        dst = __builtin_amdgcn_mov_dpp(bits, kDppRowNewBcastBase + 1, 0xf, 0xf,
                                       0);
    } else if (ctrl == 2) {
        dst = __builtin_amdgcn_mov_dpp(bits, kDppRowNewBcastBase + 2, 0xf, 0xf,
                                       0);
    } else {
        dst = __builtin_amdgcn_mov_dpp(bits, kDppRowNewBcastBase + 3, 0xf, 0xf,
                                       0);
    }
    return reinterpret_cast<const T &>(dst);
}

__host__ __device__ static constexpr int Stage1ScaleDppCtrl(int stage,
                                                            int col) {
    return (stage << 1) + (col >> 1);
}

__host__ __device__ static constexpr int Stage2ScaleDppCtrl(int stage,
                                                            int col) {
    return (stage << 1) + (col >> 1);
}

__host__ __device__ static constexpr int Stage2DqIdx(int i, int k) {
    return k * 2 + i;
}

template <auto ScaleDppCtrl>
__device__ static inline void
MatmulBlockScaleFp8(float4 t[8], const uint4 w[8], const uint4 x[8],
                    float4 x_scale, float w_scale, unsigned stage) {
    for (int i = 0; i < 4; i++) {
        const uint2 *w_2 = reinterpret_cast<const uint2 *>(w) + i * 4;
        for (int row = 0; row < 2; ++row) {
            const int tg = row * 4;
            const uint2 *x_2 =
                reinterpret_cast<const uint2 *>(x + tg + stage * 2);
            float4 m_acc{0, 0, 0, 0};
            for (int j = 0; j < 4; j++) {
                m_acc = mma_m16n16k32_fp8_fp8_f32(w_2[j], x_2[j], m_acc);
            }
            const float x_s = stage == 0 ? (row == 0 ? x_scale.x : x_scale.y)
                                         : (row == 0 ? x_scale.z : x_scale.w);
            const float w_sdpp = GetDppValue(w_scale, ScaleDppCtrl(stage, i));
            t[i * 2 + row] = Fma4(t[i * 2 + row], w_sdpp * x_s, m_acc);
        }
    }
}

template <class Config, unsigned kTokenBatch>
struct FusedMoEBlockScaleFP8Stage1Trait {
    using Scalar = __hip_fp8_e4m3;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kActivationFragments = 8;
    using Input = InputLayout<kTokenBatch, kNumWarps, Config::kGroupDim>;
    using W13 = W13Layout<Scalar, kNumWarps, Config::kGroupN>;

    struct Shm {
        unsigned act[Input::kShmInputElements];
        float scale[Input::kThreads];
    };

    struct InputRegs {
        uint4 x[kActivationFragments];
        float4 scale_x;
    };

    Input &input;
    W13 &w1, &w3;
    uint4 w1_tile[2][W13::kTileLoads];
    float scale_w1;
    static constexpr unsigned kAccumFragments = 8;

    static_assert(W13::kTileLoads == 8, "");

    __device__ explicit FusedMoEBlockScaleFP8Stage1Trait(Input &input, W13 &w1,
                                                         W13 &w3)
        : input(input), w1(w1), w3(w3) {}

    __device__ void PrefetchInput(Shm *shm, unsigned wid, unsigned wtid,
                                  const uint2 token_select,
                                  const unsigned tokens[Input::kTokenBatch],
                                  unsigned m) {
        input.FetchAsync(shm->act, wid, wtid, tokens);
        input.FetchScaleAsync(shm->scale, wid, wtid, token_select, m);
    }

    __device__ void LoadInitial(unsigned tid, unsigned wid, unsigned wtid) {
        w1.LoadTile(w1_tile[0], 0, wid, wtid);
        w1.LoadTile(w1_tile[1], 1, wid, wtid);
        scale_w1 = w1.LoadScale(tid);
    }

    __device__ void ReadInput(InputRegs &regs, const Shm *shm,
                              unsigned wtid) const {
        input.FetchToRegs(regs.x, shm->act, wtid);
        regs.scale_x = input.FetchScaleToReg(shm->scale, wtid);
    }

    __device__ void Matmul(float4 t_gate[8], float4 t_up[8],
                           const InputRegs &regs, unsigned tid, unsigned wid,
                           unsigned wtid) {
        uint4 w3_tile[2][W13::kTileLoads];
        float scale_w3;
        for (int j = 0; j < 2; j++) {
            w3.LoadTile(w3_tile[j], j, wid, wtid);
            if (j == 0) {
                scale_w3 = w3.LoadScale(tid);
            }
            MatmulBlockScaleFp8<Stage1ScaleDppCtrl>(
                t_gate, w1_tile[j], regs.x, regs.scale_x, scale_w1, j);
        }
        for (int j = 0; j < 2; j++) {
            w1.LoadTile(w1_tile[j], j, wid, wtid);
            if (j == 0) {
                scale_w1 = w1.LoadScale(tid);
            }
            MatmulBlockScaleFp8<Stage1ScaleDppCtrl>(
                t_up, w3_tile[j], regs.x, regs.scale_x, scale_w3, j);
        }
    }

};

template <class Config> struct FusedMoEBlockScaleFP8Stage2Trait {
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    using W2 = W2Layout<__hip_fp8_e4m3, kNumWarps, Config::kGroupN>;
    W2 &w2;
    uint4 w2_tile[2][2][W2::kLoadGlobal];
    float scale_w2[2];
    static constexpr unsigned kAccumFragments = 8;
    static constexpr unsigned kActivationFragments = 8;
    static constexpr unsigned kOutputPacksPerToken = 2;
    struct InputRegs {
        uint4 x[kActivationFragments];
        float4 scale;
    };

    static_assert(W2::kTileLoads == 8, "");

    __device__ explicit FusedMoEBlockScaleFP8Stage2Trait(W2 &w2) : w2(w2) {}

    __device__ void LoadStage(unsigned stage, unsigned tid, unsigned wid,
                              unsigned wtid) {
        w2.LoadTile(w2_tile[stage][0], 0, wid, wtid);
        w2.LoadTile(w2_tile[stage][1], 1, wid, wtid);
        scale_w2[stage] = w2.LoadScale(tid);
    }

    __device__ void Matmul(float4 t[8], const InputRegs &input,
                           unsigned stage, unsigned wtid,
                           bool dbg = false) const {
        (void)wtid;
        MatmulBlockScaleFp8<Stage2ScaleDppCtrl>(
            t, w2_tile[stage][0], input.x, input.scale, scale_w2[stage], 0);
        MatmulBlockScaleFp8<Stage2ScaleDppCtrl>(
            t, w2_tile[stage][1], input.x, input.scale, scale_w2[stage], 1);
    }

};

template <class Config> struct FusedMoEBlockScaleFP8KernelTrait {
    using Scalar = __hip_fp8_e4m3;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kTokenBatch = 8;
    using Input = InputLayout<kTokenBatch, kNumWarps, Config::kGroupDim>;
    using W2 = W2Layout<Scalar, kNumWarps, Config::kGroupN>;
    using W13 = W13Layout<Scalar, kNumWarps, Config::kGroupN>;
    using Stage1Trait = FusedMoEBlockScaleFP8Stage1Trait<Config, kTokenBatch>;
    using Stage1Op =
        OnestageFusedMoEStage1DoubleBufferOp<Stage1Trait,
                                             Config::kGroupDim, kTokenBatch>;
    using Stage2Trait = FusedMoEBlockScaleFP8Stage2Trait<Config>;
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
        static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(Scalar);
        const uint4 *w1_ptr =
            w13_base +
            expert_id * (2 * kernel.inter_dim_ * kernel.dim_) / kVecSize +
            tile_k * Config::kGroupDim * kernel.dim_ / kVecSize;
        const unsigned *scale_w1_ptr =
            scales_w13 + expert_id * (2 * k_blocks * n_blocks) +
            tile_k * (Config::kGroupDim / Kernel::kScaleBlockSize) *
                (kernel.dim_ / Kernel::kScaleBlockSize);
        const unsigned w13_value_range = Config::kGroupDim * kernel.dim_;
        const unsigned w13_scale_range =
            (Config::kGroupDim / Kernel::kScaleBlockSize) * n_blocks *
            sizeof(float);
        kernel.w1_.Initialize(w1_ptr, w13_value_range, scale_w1_ptr,
                              w13_scale_range, kernel.dim_);
        kernel.w3_.Initialize(
            w1_ptr + kernel.inter_dim_ * kernel.dim_ / kVecSize,
            w13_value_range, scale_w1_ptr + k_blocks * n_blocks,
            w13_scale_range, kernel.dim_);

        // w2 is pre-shuffled as [rbi][cbi][kki][bni][kpi] with
        // blockN=16/blockK=32. Advancing logical K by 256 (= 8 * 32)
        // means advancing cbi by 8 blocks, i.e. 8 * (2 * 16 * 16) =
        // 4096 fp8 elements.
        const uint4 *w2_ptr =
            w2 + expert_id * (kernel.dim_ * kernel.inter_dim_) / kVecSize +
            tile_k * (Config::kGroupDim * 16) / kVecSize;

        const unsigned *scale_w2_ptr =
            scales_w2 + expert_id * (n_blocks * k_blocks) +
            tile_k * (Config::kGroupDim / Kernel::kScaleBlockSize);
        const unsigned w2_value_range =
            kernel.dim_ * kernel.inter_dim_ - tile_k * Config::kGroupDim * 16;
        const unsigned w2_scale_range =
            n_blocks * k_blocks * sizeof(unsigned) -
            tile_k * (Config::kGroupDim / Kernel::kScaleBlockSize) *
                sizeof(unsigned);
        kernel.w2_.Initialize(w2_ptr, w2_value_range, scale_w2_ptr,
                              w2_scale_range, kernel.inter_dim_);
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
};

} // namespace causalflow::petit::rocm::moe
