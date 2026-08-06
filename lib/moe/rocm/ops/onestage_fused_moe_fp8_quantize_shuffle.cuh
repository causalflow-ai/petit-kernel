#pragma once

#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/amd_intrinsics.cuh"

#include <cmath>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <unsigned kNumWarps, unsigned kGroupN, class ShmReadLayout>
struct QuantizeAndShuffleFp8 {
    static constexpr unsigned kGroupM = 32;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kSubGroupSize = 16;
    static constexpr unsigned kKStages = kGroupN / 128;
    static constexpr unsigned kElementsPerThread =
        (kGroupM * kGroupN) / kThreads;
    static constexpr unsigned kElementsPerThreadVec4 = kElementsPerThread / 4;
    static_assert(kGroupN % 128 == 0, "");

    using FragH = float4[kElementsPerThreadVec4];
    using Quantized = uint4[kElementsPerThreadVec4];
    using FragPacked = unsigned[kElementsPerThread / 4];
    using MaxShm = float2[kThreads];
    using QHiddenShm = unsigned[kThreads * kElementsPerThreadVec4];
    struct Shm {
        MaxShm max;
        QHiddenShm q_h;
    };

    template <class InputRegs>
    __device__ static void Run(InputRegs &out, Shm &shm, const FragH h,
                               unsigned tid, unsigned wid, unsigned wtid) {
        float4 quant_scale;
        ComputeRowMax(shm.max, h, tid, quant_scale, out.scale);
        FragPacked q;
        Quantize(q, h, quant_scale);
        WriteShm(shm.q_h, q, wid, wtid);
        __syncthreads();
        ReadShm(out.x, shm.q_h, wid, wtid);
        __syncthreads();
    }

  private:
    __device__ static void ComputeRowMax(MaxShm &shm_max, const FragH h,
                                         unsigned tid, float4 &quant_scale,
                                         float4 &dequant_scale) {
        static constexpr float kLocalMaxFloor = 1e-6;
#if defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__)
        static constexpr float kFp8e4m3Max = 448;
#else
        static constexpr float kFp8e4m3Max = 240;
#endif
        static constexpr float kFp8e4m3MaxInv = 1.0f / kFp8e4m3Max;
        auto qs2 = reinterpret_cast<float2 *>(&quant_scale);
        auto ds2 = reinterpret_cast<float2 *>(&dequant_scale);
        quant_scale = {0, 0, 0, 0};
        dequant_scale = {0, 0, 0, 0};

        auto max4 = [](float4 v) {
            return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                         fmaxf(fabsf(v.z), fabsf(v.w)));
        };

        for (unsigned c = 0; c < kKStages; c++) {
            const int base = c * 4;
            float2 lm;
            lm.x = fmaxf(kLocalMaxFloor,
                         fmaxf(max4(h[base + 0]), max4(h[base + 2])));
            lm.y = fmaxf(kLocalMaxFloor,
                         fmaxf(max4(h[base + 1]), max4(h[base + 3])));
            shm_max[tid] = lm;
            __syncthreads();

            for (int i = 0; i < 16; i++) {
                float2 v = shm_max[tid % kSubGroupSize + 16 * i];
                lm.x = fmaxf(lm.x, v.x);
                lm.y = fmaxf(lm.y, v.y);
            }
            qs2[c].x = kFp8e4m3Max * __builtin_amdgcn_rcpf(lm.x);
            qs2[c].y = kFp8e4m3Max * __builtin_amdgcn_rcpf(lm.y);
            ds2[c].x = lm.x * kFp8e4m3MaxInv;
            ds2[c].y = lm.y * kFp8e4m3MaxInv;
            __syncthreads();
        }
    }

    __device__ static void Quantize(FragPacked q, const FragH h,
                                    float4 quant_scale) {
        for (unsigned i = 0; i < kElementsPerThreadVec4; i++) {
            const unsigned row = i % 2;
            const unsigned half = i / 4;
            const float s =
                reinterpret_cast<const float *>(&quant_scale)[half * 2 + row];
            const float4 v = h[i];
            const float2 s2 = {s, s};
            const float2 xy =
                amdgcn_pk_mul_f32(reinterpret_cast<const float2 &>(v), s2);
            const float2 zw =
                amdgcn_pk_mul_f32(reinterpret_cast<const float2 *>(&v)[1], s2);
            unsigned qi = 0;
            qi = amdgcn_cvt_pk_fp8_f32<false>(xy.x, xy.y, qi);
            qi = amdgcn_cvt_pk_fp8_f32<true>(zw.x, zw.y, qi);
            q[i] = qi;
        }
    }

    __device__ static void WriteShm(QHiddenShm &shm_q_h, const FragPacked q,
                                    unsigned wid, unsigned wtid) {
        using namespace causalflow::tal;
        using ShmShape = Shape<Shape<C<2>, C<kElementsPerThreadVec4 / 2>>,
                               C<kNumWarps>, Shape<C<16>, C<2>, C<2>>>;
        using ShmStride = Stride<
            Stride<C<(kElementsPerThreadVec4 / 2) * kNumWarps * kWarpSize>,
                   C<kNumWarps * kWarpSize>>,
            C<kWarpSize>, Stride<C<2>, _1, C<32>>>;
        Layout<ShmShape, ShmStride> layout;
        for (unsigned i = 0; i < kElementsPerThreadVec4; i++) {
            const unsigned idx = layout(make_coord(i, wid, wtid));
            shm_q_h[idx] = q[i];
        }
    }

    __device__ static void ReadShm(Quantized out, QHiddenShm &shm_q_h,
                                   unsigned wid, unsigned wtid) {
        using namespace causalflow::tal;
        ShmReadLayout layout;
        const uint2 *s = reinterpret_cast<const uint2 *>(shm_q_h);
        for (unsigned i = 0; i < kElementsPerThreadVec4; i++) {
            uint2 *o = reinterpret_cast<uint2 *>(out + i);
            o[0] = s[layout(make_coord(i, 0, wtid))];
            o[1] = s[layout(make_coord(i, 1, wtid))];
        }
    }
};

} // namespace causalflow::petit::rocm::moe
