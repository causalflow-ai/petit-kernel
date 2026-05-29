#pragma once

#include "gemm/rocm/amd_intrinsics.cuh"
#include "gemm/rocm/quantization/fp4/quantization_utils.cuh"
#include "moe/rocm/fused_moe.cuh"
#include "moe/rocm/mem/bias.cuh"
#include "moe/rocm/mem/input_channel_scale_fp8.cuh"
#include "moe/rocm/memory_ops.cuh"

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

__device__ static inline float4 LoadFp8E8m0Scale(unsigned packed) {
    union {
        v4f f;
        unsigned u[4];
    };
    for (unsigned i = 0; i < 4; i++) {
        unsigned char b = (packed >> (i * 8)) & 0xFF;
        u[i] = (unsigned)b << 23;
    }
    // FP4 payloads are embedded into BF8 lanes before MFMA. CDNA3 BF8 uses
    // FNUZ (bias 16), while gfx950 uses OCP BF8 (bias 15).
#if defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__)
    static const v4f kExpBias = {16384.0f, 16384.0f, 16384.0f, 16384.0f};
#else
    static const v4f kExpBias = {32768.0f, 32768.0f, 32768.0f, 32768.0f};
#endif
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

template <class Config_>
struct BlockScaleFp8Stage1Schedule {
    using Config = Config_;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kActivationFragments = 8;
    using Input = typename Config::Input;
    using W13 = typename Config::W13;
    using Bias = typename Config::Bias;

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
    Bias &w1_bias, &w3_bias;
    uint4 w1_tile[2][W13::kTileLoads];
    float scale_w1;
    static constexpr unsigned kAccumFragments = 8;

    static_assert(W13::kTileLoads == 8, "");

    __device__ explicit BlockScaleFp8Stage1Schedule(Input &input, W13 &w1,
                                                    W13 &w3, Bias &w1_bias,
                                                    Bias &w3_bias)
        : input(input), w1(w1), w3(w3), w1_bias(w1_bias),
          w3_bias(w3_bias) {}

    __device__ void InitializeBias(const void *w13_bias, unsigned expert_id,
                                   unsigned inter_dim, unsigned tile_k) {
        const unsigned projection_stride = Bias::PackedStride(inter_dim);
        const unsigned expert_stride = 2 * projection_stride;
        const void *w3_bias_ptr =
            w13_bias == nullptr
                ? nullptr
                : reinterpret_cast<const char *>(w13_bias) +
                      projection_stride * Bias::kElementBytes;
        w1_bias.Initialize(w13_bias, expert_id, inter_dim, tile_k,
                           expert_stride);
        w3_bias.Initialize(w3_bias_ptr, expert_id, inter_dim, tile_k,
                           expert_stride);
    }

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

    __device__ void AddBias(float4 t_gate[kAccumFragments],
                            float4 t_up[kAccumFragments], unsigned tid) const {
        w1_bias.AddToAccumulator(t_gate, 0, tid);
        w3_bias.AddToAccumulator(t_up, 0, tid);
    }
};

template <class Config_>
struct PetitMxFp4Stage1Schedule {
    using Config = Config_;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kActivationFragments = Config::kGroupDim / 32;
    static constexpr unsigned kKStages = Config::kGroupDim / 128;
    using Input = typename Config::Input;
    using W13 = typename Config::W13;
    using Bias = typename Config::Bias;

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
    Bias &w1_bias, &w3_bias;
    uint4 w1_tile[kKStages][W13::kLoadGlobal];
    unsigned scale_w1[kKStages];
    static constexpr unsigned kAccumFragments = 2 * W13::kLoadGlobal;

    static_assert(kAccumFragments == Config::kGroupDim / 32, "");

    __device__ explicit PetitMxFp4Stage1Schedule(Input &input, W13 &w1, W13 &w3,
                                            Bias &w1_bias, Bias &w3_bias)
        : input(input), w1(w1), w3(w3), w1_bias(w1_bias),
          w3_bias(w3_bias) {}

    __device__ void InitializeBias(const void *w13_bias, unsigned expert_id,
                                   unsigned inter_dim, unsigned tile_k) {
        const unsigned projection_stride = Bias::PackedStride(inter_dim);
        const unsigned expert_stride = 2 * projection_stride;
        const void *w3_bias_ptr =
            w13_bias == nullptr
                ? nullptr
                : reinterpret_cast<const char *>(w13_bias) +
                      projection_stride * Bias::kElementBytes;
        w1_bias.Initialize(w13_bias, expert_id, inter_dim, tile_k,
                           expert_stride);
        w3_bias.Initialize(w3_bias_ptr, expert_id, inter_dim, tile_k,
                           expert_stride);
    }

    __device__ void PrefetchInput(Shm *shm, unsigned wid, unsigned wtid,
                                  const uint2 token_select,
                                  const unsigned tokens[Input::kTokenBatch],
                                  unsigned m) {
        input.FetchAsync(shm->act, wid, wtid, tokens);
        input.FetchScaleAsync(shm->scale, wid, wtid, token_select, m);
    }

    __device__ void LoadInitial(unsigned tid, unsigned wid, unsigned wtid) {
        for (unsigned j = 0; j < kKStages; ++j) {
            w1.LoadTile(w1_tile[j], j, wid, wtid);
            scale_w1[j] = w1.LoadScale(tid);
            w1.template AdvanceStep<0, 1>();
        }
    }

    __device__ void ReadInput(InputRegs &regs, const Shm *shm,
                              unsigned wtid) const {
        input.FetchToRegsFP4(regs.x, shm->act, wtid);
        regs.scale_x = input.FetchScaleToReg(shm->scale, wtid);
    }

    __device__ void Matmul(float4 t_gate[kAccumFragments],
                           float4 t_up[kAccumFragments],
                           const InputRegs &regs, unsigned tid, unsigned wid,
                           unsigned wtid) {
        uint4 w3_tile[kKStages][W13::kLoadGlobal];
        unsigned scale_w3[kKStages];
        for (unsigned j = 0; j < kKStages; j++) {
            w3.LoadTile(w3_tile[j], j, wid, wtid);
            scale_w3[j] = w3.LoadScale(tid);
            w3.template AdvanceStep<0, 1>();
            MatmulBlockScaleFp4<W13::kLoadGlobal, kActivationFragments>(
                t_gate, w1_tile[j], regs.x, regs.scale_x, scale_w1[j], j);
        }
        for (unsigned j = 0; j < kKStages; j++) {
            w1.LoadTile(w1_tile[j], j, wid, wtid);
            scale_w1[j] = w1.LoadScale(tid);
            w1.template AdvanceStep<0, 1>();
            MatmulBlockScaleFp4<W13::kLoadGlobal, kActivationFragments>(
                t_up, w3_tile[j], regs.x, regs.scale_x, scale_w3[j], j);
        }
    }

    __device__ void AddBias(float4 t_gate[kAccumFragments],
                            float4 t_up[kAccumFragments], unsigned tid) const {
        w1_bias.AddToAccumulator(t_gate, 0, tid);
        w3_bias.AddToAccumulator(t_up, 0, tid);
    }
};

} // namespace causalflow::petit::rocm::moe
