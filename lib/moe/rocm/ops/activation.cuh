#pragma once

#include "gemm/rocm/amd_intrinsics.cuh"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

struct SiluDotOp {
    __device__ static inline float4 Apply(float4 gate, float4 up) {
        static constexpr float kMinusLog2e = -1.4426950408889634f;
        static constexpr v2f kMinusLog2e2{kMinusLog2e, kMinusLog2e};
        static constexpr v2f kOne2{1.0f, 1.0f};
        auto *g2 = reinterpret_cast<const v2f *>(&gate);
        auto *u2 = reinterpret_cast<const float2 *>(&up);
        v2f i2[2];

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
        auto *i2f = reinterpret_cast<const float2 *>(i2);
        reinterpret_cast<float2 &>(r) = amdgcn_pk_mul_f32(
            amdgcn_pk_mul_f32(reinterpret_cast<const float2 &>(gate), i2f[0]),
            u2[0]);
        reinterpret_cast<float2 *>(&r)[1] = amdgcn_pk_mul_f32(
            amdgcn_pk_mul_f32(reinterpret_cast<const float2 *>(&gate)[1],
                              i2f[1]),
            u2[1]);
        return r;
    }
};

struct OpenAISwiGLUOp {
    __device__ static inline float4 Apply(float4 gate, float4 up) {
        static constexpr float kMinusAlphaLog2e = -2.455307455790015f;
        static constexpr float kLimit = 7.0f;
        float4 r;
        const float *g = reinterpret_cast<const float *>(&gate);
        const float *u = reinterpret_cast<const float *>(&up);
        float *o = reinterpret_cast<float *>(&r);
        for (int i = 0; i < 4; ++i) {
            const float gc = fminf(g[i], kLimit);
            const float uc = fminf(fmaxf(u[i], -kLimit), kLimit);
            const float sig =
                __builtin_amdgcn_rcpf(1.0f +
                                      amdgcn_exp2f(kMinusAlphaLog2e * gc));
            o[i] = gc * sig * (uc + 1.0f);
        }
        return r;
    }
};

} // namespace causalflow::petit::rocm::moe
