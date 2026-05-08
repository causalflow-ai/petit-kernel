#pragma once

#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/amd_intrinsics.cuh"

#include <cmath>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <type_traits>

namespace causalflow::petit::rocm::moe {

static constexpr unsigned kQuantBlockK = 128;
static constexpr float kFp8E4m3Max = 240.0f;
static constexpr float kQuantFloor = 1e-6f;

__host__ __device__ static inline float ClampQuantAbsmax(float absmax) {
    if (!__builtin_isfinite(absmax)) {
        return kQuantFloor;
    }
    return fmaxf(absmax, kQuantFloor);
}

__host__ __device__ static inline float QuantScaleFromAbsmax(float absmax) {
    const float clamped = ClampQuantAbsmax(absmax);
#if defined(__HIP_DEVICE_COMPILE__)
    return kFp8E4m3Max * __builtin_amdgcn_rcpf(clamped);
#else
    return kFp8E4m3Max / clamped;
#endif
}

__host__ __device__ static inline float DequantScaleFromAbsmax(float absmax) {
    const float clamped = ClampQuantAbsmax(absmax);
#if defined(__HIP_DEVICE_COMPILE__)
    return clamped * __builtin_amdgcn_rcpf(kFp8E4m3Max);
#else
    return clamped / kFp8E4m3Max;
#endif
}

template <unsigned kBlockK = kQuantBlockK>
__host__ __device__ static inline float OnlineQuantize1x128(unsigned char *out,
                                                            const float *in) {
    static_assert((kBlockK % 4) == 0,
                  "OnlineQuantize1x128 expects kBlockK divisible by 4.");
    float absmax = 0.0f;
    for (unsigned i = 0; i < kBlockK; ++i) {
        absmax = fmaxf(absmax, fabsf(in[i]));
    }
    const float q_scale = QuantScaleFromAbsmax(absmax);
    const float dq_scale = DequantScaleFromAbsmax(absmax);
    auto *const packed_out = reinterpret_cast<unsigned *>(out);

    for (unsigned i = 0; i < kBlockK; i += 4) {
        const __hip_fp8x4_e4m3 packed(float4{
            in[i + 0] * q_scale,
            in[i + 1] * q_scale,
            in[i + 2] * q_scale,
            in[i + 3] * q_scale,
        });
        packed_out[i / 4] = packed.__x;
    }
    return dq_scale;
}

template <unsigned kRows = 2, unsigned kCols = kQuantBlockK>
__device__ static inline void OnlineQuantize2x128(uint4 out[kRows],
                                                  const float4 in[kCols / 16],
                                                  const float4 quant_scale) {
    static_assert(kRows == 2, "OnlineQuantize2x128 expects exactly 2 rows.");
    static_assert(kCols == kQuantBlockK,
                  "OnlineQuantize2x128 expects kCols == 128.");
    int dummy = 0;
    for (int i = 0; i < 2; ++i) {
        unsigned *q = reinterpret_cast<unsigned *>(out + i);
        for (int k = 0; k < 2; ++k) {
            const float s =
                reinterpret_cast<const float *>(&quant_scale)[k * 2 + i];
            for (int j = 0; j < 2; ++j) {
                // Stage1 accumulators are laid out as:
                // [A0-row0, A0-row1, A1-row0, A1-row1, A2-row0, ...].
                const float4 v = in[k * 4 + j * 2 + i];
                const unsigned u0 =
                    amdgcn_cvt_pk_fp8_f32<false>(v.x * s, v.y * s, dummy);
                const unsigned u1 =
                    amdgcn_cvt_pk_fp8_f32<false>(v.z * s, v.w * s, dummy);
                q[k * 2 + j] = (u1 << 16) | u0;
            }
        }
    }
}

} // namespace causalflow::petit::rocm::moe
