#pragma once

#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/amd_intrinsics.cuh"

#include <cmath>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <type_traits>

namespace causalflow::petit::rocm::moe {

static constexpr unsigned kQuantBlockK = 128;
#if defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__)
static constexpr float kFp8E4m3Max = 448.0f;
#else
static constexpr float kFp8E4m3Max = 240.0f;
#endif
static constexpr float kQuantFloor = 1e-6f;

__device__ static inline unsigned QuantizeFp8E4m3x4(float4 value,
                                                    float scale) {
    const float2 scale2{scale, scale};
    const float2 xy = amdgcn_pk_mul_f32(
        reinterpret_cast<const float2 &>(value), scale2);
    const float2 zw = amdgcn_pk_mul_f32(
        reinterpret_cast<const float2 *>(&value)[1], scale2);
    unsigned packed = 0;
    packed = amdgcn_cvt_pk_fp8_f32<false>(xy.x, xy.y, packed);
    return amdgcn_cvt_pk_fp8_f32<true>(zw.x, zw.y, packed);
}

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
    for (int i = 0; i < 2; ++i) {
        unsigned *q = reinterpret_cast<unsigned *>(out + i);
        for (int k = 0; k < 2; ++k) {
            const float s =
                reinterpret_cast<const float *>(&quant_scale)[k * 2 + i];
            for (int j = 0; j < 2; ++j) {
                // Stage1 accumulators are laid out as:
                // [A0-row0, A0-row1, A1-row0, A1-row1, A2-row0, ...].
                const float4 v = in[k * 4 + j * 2 + i];
                q[k * 2 + j] = QuantizeFp8E4m3x4(v, s);
            }
        }
    }
}

struct MxFp4Scale {
    unsigned byte;
    float packing_scale;
};

struct NativeMxFp4Quantization {
    __device__ static float MaximumAbs(float4 value) {
        return fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                     fmaxf(fabsf(value.z), fabsf(value.w)));
    }

    __device__ static MxFp4Scale EncodeScale(float max_abs) {
        if (max_abs < 1.0e-12f)
            return {127u, 1.0f};
        const float required = max_abs * (1.0f / 6.0f);
        const unsigned required_bits =
            reinterpret_cast<const unsigned &>(required);
        unsigned scale_byte = (required_bits >> 23) & 0xffu;
        if (scale_byte < 0xffu && (required_bits & 0x7fffffu))
            ++scale_byte;
        const unsigned bits = scale_byte << 23;
        return {scale_byte, reinterpret_cast<const float &>(bits)};
    }

    template <unsigned kVectors>
    __device__ static void Pack(unsigned char *dst,
                                const float4 values[kVectors],
                                const MxFp4Scale &scale) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__) &&                  \
    __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_fp4_f32)
        static_assert(kVectors % 2 == 0,
                      "native packing consumes pairs of float4");
        auto *packed = reinterpret_cast<unsigned *>(dst);
#pragma unroll
        for (unsigned vector = 0; vector < kVectors; vector += 2) {
            const float4 a = values[vector];
            const float4 b = values[vector + 1];
            unsigned word = 0;
            word = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                word, a.x, a.y, scale.packing_scale, 0);
            word = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                word, a.z, a.w, scale.packing_scale, 1);
            word = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                word, b.x, b.y, scale.packing_scale, 2);
            word = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                word, b.z, b.w, scale.packing_scale, 3);
            packed[vector / 2] = word;
        }
#else
        (void)dst;
        (void)values;
        (void)scale;
#endif
    }
};

struct AiterMxFp4Quantization {
    using f32x4 = float __attribute__((ext_vector_type(4)));

    __device__ static float MaximumAbs(float4 value) {
        const f32x4 elements{value.x, value.y, value.z, value.w};
        return __builtin_reduce_maximum(
            __builtin_elementwise_abs(elements));
    }

    __device__ static MxFp4Scale EncodeScale(float max_abs) {
        unsigned bits = reinterpret_cast<const unsigned &>(max_abs);
        bits = (bits + 0x00400000u) & 0xff800000u;
        const unsigned exponent = max(bits >> 23, 2u);
        const unsigned scale_byte = exponent - 2;
        const unsigned scale_bits = scale_byte << 23;
        return {scale_byte, reinterpret_cast<const float &>(scale_bits)};
    }

    template <unsigned kVectors>
    __device__ static void Pack(unsigned char *dst,
                                const float4 values[kVectors],
                                const MxFp4Scale &scale) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__) &&                  \
    __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_fp4_f32)
        auto *packed = reinterpret_cast<unsigned short *>(dst);
#pragma unroll
        for (unsigned vector = 0; vector < kVectors; ++vector) {
            const float4 value = values[vector];
            unsigned word = 0;
            word = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                word, value.x, value.y, scale.packing_scale, 0);
            word = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                word, value.z, value.w, scale.packing_scale, 1);
            packed[vector] = static_cast<unsigned short>(word);
        }
#else
        (void)dst;
        (void)values;
        (void)scale;
#endif
    }
};

template <class Quantization, unsigned kVectors>
__device__ static inline unsigned
QuantizeMxFp4(unsigned char *dst, const float4 values[kVectors],
              float max_abs) {
    const MxFp4Scale scale = Quantization::EncodeScale(max_abs);
    Quantization::template Pack<kVectors>(dst, values, scale);
    return scale.byte;
}

} // namespace causalflow::petit::rocm::moe
