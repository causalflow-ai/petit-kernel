#pragma once

#include "moe/rocm/mem/bias.cuh"
#include "moe/rocm/ops/schedule_stage1.cuh"
#include "moe/rocm/memory_ops.cuh"

#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

__host__ __device__ static constexpr int Stage2ScaleDppCtrl(int stage,
                                                            int col) {
    return (stage << 1) + (col >> 1);
}

template <class Config> struct BlockScaleFp8Stage2Schedule {
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    using W2 = W2Layout<__hip_fp8_e4m3, kNumWarps, Config::kGroupN>;
    using Bias = NoopBiasLayout<Config::kNumWarps, Config::kGroupN>;
    W2 &w2;
    Bias &w2_bias;
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

    __device__ explicit BlockScaleFp8Stage2Schedule(W2 &w2, Bias &w2_bias)
        : w2(w2), w2_bias(w2_bias) {}

    __device__ void InitializeBias(const void *w2_bias_ptr, unsigned expert_id,
                                   unsigned dim, unsigned tile_k) {
        const unsigned expert_stride = Bias::PackedStride(dim);
        w2_bias.Initialize(w2_bias_ptr, expert_id, dim, tile_k, expert_stride);
    }

    __device__ void LoadStage(unsigned stage, unsigned tid, unsigned wid,
                              unsigned wtid) {
        w2.LoadTile(w2_tile[stage][0], 0, wid, wtid);
        w2.LoadTile(w2_tile[stage][1], 1, wid, wtid);
        scale_w2[stage] = w2.LoadScale(tid);
    }

    __device__ void Matmul(float4 t[8], const InputRegs &input,
                           unsigned stage, unsigned wtid,
                           bool dbg = false) const {
        MatmulBlockScaleFp8<Stage2ScaleDppCtrl>(
            t, w2_tile[stage][0], input.x, input.scale, scale_w2[stage], 0);
        MatmulBlockScaleFp8<Stage2ScaleDppCtrl>(
            t, w2_tile[stage][1], input.x, input.scale, scale_w2[stage], 1);
    }

};

template <class Config> struct PetitMxFp4Stage2Schedule {
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr int kKStages = Config::kGroupN / 128;
    using W2 = MxFp4WeightLayout<kNumWarps, Config::kGroupN>;
    using Bias = NoopBiasLayout<Config::kNumWarps, Config::kGroupN>;
    W2 &w2;
    Bias &w2_bias;
    uint4 w2_tile[2][kKStages][W2::kLoadGlobal];
    unsigned scale_w2[2][kKStages];
    static constexpr unsigned kAccumFragments = 2 * W2::kLoadGlobal;
    static constexpr unsigned kExpectedAccumFragments =
        2 * (Config::kGroupN / Config::kNumWarps / 16);
    static constexpr unsigned kActivationFragments = Config::kGroupDim / 32;
    static constexpr unsigned kOutputPacksPerToken = Config::kGroupN / 128;
    struct InputRegs {
        uint4 x[kActivationFragments];
        float4 scale;
    };

    static_assert(kAccumFragments == kExpectedAccumFragments, "");
    static_assert(kActivationFragments == Config::kGroupDim / 32, "");
    static_assert(kOutputPacksPerToken > 0, "");

    __device__ explicit PetitMxFp4Stage2Schedule(W2 &w2, Bias &w2_bias)
        : w2(w2), w2_bias(w2_bias) {}

    __device__ void InitializeBias(const void *w2_bias_ptr, unsigned expert_id,
                                   unsigned dim, unsigned tile_k) {
        const unsigned expert_stride = Bias::PackedStride(dim);
        w2_bias.Initialize(w2_bias_ptr, expert_id, dim, tile_k, expert_stride);
    }

    __device__ void LoadStage(unsigned stage, unsigned tid, unsigned wid,
                              unsigned wtid) {
        for (unsigned j = 0; j < kKStages; ++j) {
            w2.LoadTile(w2_tile[stage][j], j, wid, wtid);
            scale_w2[stage][j] = w2.LoadScale(tid);
            w2.template AdvanceStep<0, 1>();
        }
        w2.template AdvanceStep<Config::kGroupN / 128, -kKStages>();
    }

    __device__ void Matmul(float4 t[kAccumFragments], const InputRegs &input,
                           unsigned stage, unsigned wtid) {
        for (unsigned j = 0; j < kKStages; ++j) {
            MatmulBlockScaleFp4<W2::kLoadGlobal, kActivationFragments>(
                t, w2_tile[stage][j], input.x, input.scale, scale_w2[stage][j],
                j);
        }
    }

};

} // namespace causalflow::petit::rocm::moe
