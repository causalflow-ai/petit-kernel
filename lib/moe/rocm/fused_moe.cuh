#pragma once

#include "gemm/rocm/amd_intrinsics.cuh"
#include "moe/rocm/fused_moe.h"

namespace causalflow::petit::rocm::moe {

__device__ static inline float4 Fma4(float4 a, float s, float4 c) {
    const auto *a2 = reinterpret_cast<const float2 *>(&a);
    const auto *c2 = reinterpret_cast<const float2 *>(&c);
    float2 s2 = {s, s};
    float4 r;
    auto *r2 = reinterpret_cast<float2 *>(&r);
    r2[0] = amdgcn_pk_fma_f32(s2, c2[0], a2[0]);
    r2[1] = amdgcn_pk_fma_f32(s2, c2[1], a2[1]);
    return r;
}

struct SiluDotOp {
    __device__ static inline float4 Apply(float4 gate, float4 up) {
        static constexpr float kMinusLog2e = -1.4426950408889634f;
        static constexpr float2 kMinusLog2e2{kMinusLog2e, kMinusLog2e};
        static constexpr float2 kOne2{1.0f, 1.0f};
        auto *g2 = reinterpret_cast<const float2 *>(&gate);
        auto *u2 = reinterpret_cast<const float2 *>(&up);
        float2 i2[2];

        i2[0] = g2[0] * kMinusLog2e2;
        i2[1] = g2[1] * kMinusLog2e2;
        i2[0][0] = amdgcn_exp2f(i2[0][0]);
        i2[0][1] = amdgcn_exp2f(i2[0][1]);
        i2[1][0] = amdgcn_exp2f(i2[1][0]);
        i2[1][1] = amdgcn_exp2f(i2[1][1]);
        i2[0] += kOne2;
        i2[1] += kOne2;
        i2[0][0] = __builtin_amdgcn_rcpf(i2[0][0]);
        i2[0][1] = __builtin_amdgcn_rcpf(i2[0][1]);
        i2[1][0] = __builtin_amdgcn_rcpf(i2[1][0]);
        i2[1][1] = __builtin_amdgcn_rcpf(i2[1][1]);

        float4 r;
        reinterpret_cast<float2 &>(r) = amdgcn_pk_mul_f32(
            amdgcn_pk_mul_f32(reinterpret_cast<const float2 &>(gate), i2[0]),
            u2[0]);
        reinterpret_cast<float2 *>(&r)[1] = amdgcn_pk_mul_f32(
            amdgcn_pk_mul_f32(reinterpret_cast<const float2 *>(&gate)[1], i2[1]),
            u2[1]);
        return r;
    }
};

struct NoBiasOp {
    template <class Kernel>
    __device__ void Initialize(Kernel &kernel) {}

    template <unsigned kAccumFragments>
    __device__ void AddBiasStage1(float4 t_gate[kAccumFragments],
                                  float4 t_up[kAccumFragments],
                                  unsigned wid, unsigned wtid) {}

    template <unsigned kAccumFragments>
    __device__ void AddBiasStage2(float4 t_gate[kAccumFragments],
                                  unsigned tile_d,
                                  unsigned wid, unsigned wtid) {}
};

template <unsigned kInstMFMA, unsigned kInstVmemRead = 0,
          unsigned kInstDsRead = 0, unsigned kInstDsWrite = 0,
          unsigned kInstVALU = 0>
__device__ static inline void HotLoopScheduler() {
#if HAS_AMD_SCHED_GROUP_BARRIER && HAS_AMD_SCHED_BARRIER
    static constexpr unsigned kSchedGroupId = 0;
    static constexpr unsigned kInstIssue =
        kInstVmemRead + kInstDsRead + kInstDsWrite + kInstVALU;

    if constexpr (kInstMFMA > 0 && kInstIssue > 0) {
        static constexpr unsigned kInstMFMAPerIssue =
            kInstMFMA / kInstIssue > 12 ? 4
                                        : (kInstMFMA / kInstIssue > 6 ? 2 : 1);

        for (unsigned i = 0; i < kInstDsWrite; ++i) {
            amdgcn_sched_group_barrier<0x200, 1, kSchedGroupId>();
            amdgcn_sched_group_barrier<0x8, kInstMFMAPerIssue,
                                       kSchedGroupId>();
        }
        for (unsigned i = 0; i < kInstVmemRead; ++i) {
            amdgcn_sched_group_barrier<0x20, 1, kSchedGroupId>();
            amdgcn_sched_group_barrier<0x8, kInstMFMAPerIssue,
                                       kSchedGroupId>();
        }
        for (unsigned i = 0; i < kInstDsRead; ++i) {
            amdgcn_sched_group_barrier<0x100, 1, kSchedGroupId>();
            amdgcn_sched_group_barrier<0x8, kInstMFMAPerIssue,
                                       kSchedGroupId>();
        }
        for (unsigned i = 0; i < kInstVALU; ++i) {
            amdgcn_sched_group_barrier<0x1, 1, kSchedGroupId>();
            amdgcn_sched_group_barrier<0x8, kInstMFMAPerIssue,
                                       kSchedGroupId>();
        }
    }
    amdgcn_sched_barrier<0>();
#endif
}

template <typename Mat> __device__ static inline void ClearMat(Mat &mat) {
    for (int i = 0; i < sizeof(Mat) / sizeof(float); i++) {
        reinterpret_cast<float *>(&mat)[i] = 0.f;
    }
}

} // namespace causalflow::petit::rocm::moe
