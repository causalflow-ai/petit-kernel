#pragma once

#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/amd_intrinsics.cuh"

#include <cmath>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

struct BlockScaleFp8Handoff {
    template <unsigned kFragments>
    __device__ static void WriteCanonicalPacked(unsigned char *dst,
                                                 const unsigned *q,
                                                 unsigned wid,
                                                 unsigned wtid,
                                                 unsigned cols) {
        static_assert(kFragments == 8);
        const unsigned lane = wtid % 16;
#pragma unroll
        for (unsigned i = 0; i < kFragments; ++i) {
            const unsigned row = (i & 1u) * 16 + lane;
            const unsigned col =
                (i / 2) * 64 + wid * 16 + (wtid / 16) * 4;
            *reinterpret_cast<unsigned *>(dst + row * cols + col) = q[i];
        }
    }

    template <unsigned kKStages, unsigned kFragments, class MaxShm>
    __device__ static void ReduceRowMax(float2 (&row_max)[kKStages],
                                        MaxShm &shm_max, const float4 *h,
                                        unsigned tid) {
        static constexpr float kLocalMaxFloor = 1e-6;
        auto max4 = [](float4 v) {
            return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                         fmaxf(fabsf(v.z), fabsf(v.w)));
        };

        static_assert(kFragments == 4 * kKStages, "");
        for (unsigned c = 0; c < kKStages; ++c) {
            const unsigned base = c * 4;
            float2 lm;
            lm.x = fmaxf(kLocalMaxFloor,
                         fmaxf(max4(h[base]), max4(h[base + 2])));
            lm.y = fmaxf(kLocalMaxFloor,
                         fmaxf(max4(h[base + 1]), max4(h[base + 3])));
            shm_max[tid] = lm;
            __syncthreads();

#pragma unroll
            for (unsigned i = 0; i < 16; ++i) {
                const float2 v = shm_max[tid % 16 + 16 * i];
                lm.x = fmaxf(lm.x, v.x);
                lm.y = fmaxf(lm.y, v.y);
            }
            row_max[c] = lm;
            __syncthreads();
        }
    }

    __device__ static unsigned ScaleHalf(unsigned fragment, unsigned) {
        return fragment / 4;
    }

    template <unsigned kFragments>
    __device__ static void ReadCanonicalPacked(uint4 (&out)[kFragments],
                                                const unsigned char *src,
                                                unsigned wtid,
                                                unsigned cols) {
        static_assert(kFragments == 8);
        const unsigned row_lane = wtid % 16;
        const unsigned col_quadrant = wtid / 16;
#pragma unroll
        for (unsigned row_half = 0; row_half < 2; ++row_half) {
            const unsigned row = row_half * 16 + row_lane;
#pragma unroll
            for (unsigned fragment = 0; fragment < 4; ++fragment) {
                // The interleaved FP8 schedule consumes 16 adjacent K bytes.
                const unsigned col = fragment * 64 + col_quadrant * 16;
                out[row_half * 4 + fragment] =
                    *reinterpret_cast<const uint4 *>(src + row * cols + col);
            }
        }
    }

};

struct PetitMxFp4Handoff {
    template <unsigned kFragments>
    __device__ static void WriteCanonicalPacked(unsigned char *dst,
                                                 const unsigned *q,
                                                 unsigned wid,
                                                 unsigned wtid,
                                                 unsigned cols) {
        static_assert(kFragments == 8);
        const unsigned lane = wtid % 16;
#pragma unroll
        for (unsigned i = 0; i < kFragments; ++i) {
            const unsigned row = (i & 1u) * 16 + lane;
            const unsigned col =
                wid * 64 + (i / 2) * 16 + (wtid / 16) * 4;
            *reinterpret_cast<unsigned *>(dst + row * cols + col) = q[i];
        }
    }

    template <unsigned kKStages, unsigned kFragments, class MaxShm>
    __device__ static void ReduceRowMax(float2 (&row_max)[kKStages],
                                        MaxShm &shm_max, const float4 *h,
                                        unsigned tid) {
        static constexpr float kLocalMaxFloor = 1e-6;
        static_assert(kKStages == 2, "MXFP4 handoff expects K256");
        static_assert(kFragments == 8,
                      "MXFP4 handoff expects eight fragments");

        auto max4 = [](float4 v) {
            return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                         fmaxf(fabsf(v.z), fabsf(v.w)));
        };

        float2 lm{kLocalMaxFloor, kLocalMaxFloor};
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment) {
            lm.x = fmaxf(lm.x, max4(h[2 * fragment]));
            lm.y = fmaxf(lm.y, max4(h[2 * fragment + 1]));
        }
        shm_max[tid] = lm;
        __syncthreads();

#pragma unroll
        for (unsigned c = 0; c < kKStages; ++c) {
            float2 reduced{kLocalMaxFloor, kLocalMaxFloor};
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                // Eight lane-16 groups are the two waves that own one N128.
                const float2 v = shm_max[tid % 16 + 16 * (c * 8 + i)];
                reduced.x = fmaxf(reduced.x, v.x);
                reduced.y = fmaxf(reduced.y, v.y);
            }
            row_max[c] = reduced;
        }
    }

    __device__ static unsigned ScaleHalf(unsigned, unsigned tid) {
        return (tid / kWarpSize) / 2;
    }

    template <unsigned kFragments>
    __device__ static void ReadCanonicalPacked(uint4 (&out)[kFragments],
                                                const unsigned char *src,
                                                unsigned wtid,
                                                unsigned cols) {
        static_assert(kFragments == 8);
        const unsigned row_lane = wtid % 16;
        const unsigned k32_quadrant = wtid / 16;
#pragma unroll
        for (unsigned row_half = 0; row_half < 2; ++row_half) {
            const unsigned row = row_half * 16 + row_lane;
#pragma unroll
            for (unsigned fragment = 0; fragment < 4; ++fragment) {
                // For operand B of the Petit BF8 MFMA, one uint4 consists of
                // two K8 vectors from the two K32 halves of a K64 fragment.
                const unsigned col = fragment * 64 + k32_quadrant * 8;
                const uint2 lo = *reinterpret_cast<const uint2 *>(
                    src + row * cols + col);
                const uint2 hi = *reinterpret_cast<const uint2 *>(
                    src + row * cols + col + 32);
                out[row_half * 4 + fragment] =
                    uint4{lo.x, lo.y, hi.x, hi.y};
            }
        }
    }

};

template <class Config, class Handoff> struct QuantizeAndShuffleFp8 {
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kGroupN = Config::kGroupN;
    static constexpr unsigned kGroupM = 32;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kKStages = kGroupN / 128;
    static constexpr unsigned kElementsPerThread =
        (kGroupM * kGroupN) / kThreads;
    static constexpr unsigned kElementsPerThreadVec4 = kElementsPerThread / 4;
    static_assert(kGroupN % 128 == 0, "");
    using OutputRegs = typename Config::Stage2Tiles::InputRegs;

    using FragH = float4[kElementsPerThreadVec4];
    static constexpr unsigned kScaleBlocks = kGroupN / 128;
#if defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__)
    static constexpr float kFp8e4m3Max = 448;
#else
    static constexpr float kFp8e4m3Max = 240;
#endif
    static constexpr float kFp8e4m3MaxInv = 1.0f / kFp8e4m3Max;
    using MaxShm = float2[kThreads];
    using FragPacked = unsigned[kElementsPerThreadVec4];
    struct Shm {
        union {
            MaxShm max;
            unsigned char value[kGroupM * kGroupN];
        } scratch;
        float scale[kGroupM * kScaleBlocks];
    };

    __device__ static void Run(OutputRegs &out, Shm &shm, const FragH h,
                               unsigned tid, unsigned wid, unsigned wtid) {
        float4 quant_scale;
        float4 dequant_scale;
        ComputeRowMax(shm.scratch.max, h, tid, quant_scale, dequant_scale);
        FragPacked q;
        Quantize(q, h, quant_scale);
        __syncthreads();
        Handoff::template WriteCanonicalPacked<kElementsPerThreadVec4>(
            shm.scratch.value, q, wid, wtid, kGroupN);
        if (tid < 16) {
            const float *s = reinterpret_cast<const float *>(&dequant_scale);
            shm.scale[tid * kScaleBlocks] = s[0];
            shm.scale[(16 + tid) * kScaleBlocks] = s[1];
            shm.scale[tid * kScaleBlocks + 1] = s[2];
            shm.scale[(16 + tid) * kScaleBlocks + 1] = s[3];
        }
        __syncthreads();
        Handoff::ReadCanonicalPacked(out.x, shm.scratch.value, wtid,
                                     kGroupN);
        const unsigned row_lane = wtid % 16;
        out.scale = {shm.scale[row_lane * kScaleBlocks],
                     shm.scale[(16 + row_lane) * kScaleBlocks],
                     shm.scale[row_lane * kScaleBlocks + 1],
                     shm.scale[(16 + row_lane) * kScaleBlocks + 1]};
        __syncthreads();
    }

  private:
    __device__ static void ComputeRowMax(MaxShm &shm_max, const FragH h,
                                         unsigned tid, float4 &quant_scale,
                                         float4 &dequant_scale) {
        auto *qs = reinterpret_cast<float *>(&quant_scale);
        auto *ds = reinterpret_cast<float *>(&dequant_scale);

        float2 row_max[kKStages];
        Handoff::template ReduceRowMax<kKStages, kElementsPerThreadVec4>(
            row_max, shm_max, h, tid);
#pragma unroll
        for (unsigned block = 0; block < kKStages; ++block) {
            qs[block * 2] =
                kFp8e4m3Max * __builtin_amdgcn_rcpf(row_max[block].x);
            qs[block * 2 + 1] =
                kFp8e4m3Max * __builtin_amdgcn_rcpf(row_max[block].y);
            ds[block * 2] = row_max[block].x * kFp8e4m3MaxInv;
            ds[block * 2 + 1] = row_max[block].y * kFp8e4m3MaxInv;
        }
    }

    __device__ static void Quantize(FragPacked q, const FragH h,
                                    const float4 &quant_scale) {
        const auto *scale = reinterpret_cast<const float *>(&quant_scale);
#pragma unroll
        for (unsigned i = 0; i < kElementsPerThreadVec4; ++i) {
            const unsigned row_half = i & 1u;
            const unsigned block = Handoff::ScaleHalf(i, threadIdx.x);
            const float2 s2{scale[block * 2 + row_half],
                            scale[block * 2 + row_half]};
            const float4 v = h[i];
            const float2 xy =
                amdgcn_pk_mul_f32(reinterpret_cast<const float2 &>(v), s2);
            const float2 zw =
                amdgcn_pk_mul_f32(reinterpret_cast<const float2 *>(&v)[1], s2);
            unsigned packed = 0;
            packed = amdgcn_cvt_pk_fp8_f32<false>(xy.x, xy.y, packed);
            packed = amdgcn_cvt_pk_fp8_f32<true>(zw.x, zw.y, packed);
            q[i] = packed;
        }
    }

};

template <unsigned kNumWarps, unsigned kGroupN, class ShmReadLayout,
          class OutputRegs>
struct PackAndShuffleBf16 {
    static constexpr unsigned kGroupM = 32;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kInputFragments =
        (kGroupM * kGroupN) / kThreads / 4;
    static constexpr unsigned kOutputFragments = 2 * kInputFragments;
    static constexpr unsigned kElementsPerThreadVec4 = kOutputFragments;
    using Scratch = unsigned[1];
    using Shm = float[kGroupM * kGroupN];

    __device__ static void Run(OutputRegs &out, Shm &shm_h,
                               const float4 h[kInputFragments], unsigned tid,
                               unsigned wid, unsigned wtid) {
        (void)tid;
        WriteCanonical(shm_h, h, wid, wtid);
        __syncthreads();
        ReadCanonical(out.x, shm_h, wtid);
        __syncthreads();
    }

  private:
    __device__ static void WriteCanonical(
        Shm &shm_h, const float4 h[kInputFragments], unsigned wid,
        unsigned wtid) {
        const unsigned lane = wtid % 16;
#pragma unroll
        for (unsigned i = 0; i < kInputFragments; ++i) {
            const unsigned row = (i & 1u) * 16 + lane;
            const unsigned col =
                wid * 64 + (i / 2) * 16 + (wtid / 16) * 4;
#pragma unroll
            for (unsigned component = 0; component < 4; ++component) {
                shm_h[row * kGroupN + col + component] =
                    reinterpret_cast<const float *>(&h[i])[component];
            }
        }
    }

    __device__ static uint4 Pack2(uint2 a, uint2 b) {
        return uint4{a.x, a.y, b.x, b.y};
    }

    __device__ static uint2 ToBf16RnLocal(float4 m) {
        uint4 m_bits;
        for (int j = 0; j < 4; j++) {
            const float f = reinterpret_cast<const float *>(&m)[j];
            const uint u = reinterpret_cast<const uint *>(&m)[j];
            const uint rounded = u + 0x7fffu + ((u >> 16) & 1u);
            reinterpret_cast<uint *>(&m_bits)[j] =
                isnan(f) ? 0x7fff0000u : rounded;
        }
        uint2 o;
        o.x = amdgcn_perm_b32(m_bits.y, m_bits.x, 0x07060302);
        o.y = amdgcn_perm_b32(m_bits.w, m_bits.z, 0x07060302);
        return o;
    }

    __device__ static void ReadCanonical(uint4 out[kOutputFragments],
                                         const Shm &shm_h, unsigned wtid) {
        static constexpr unsigned kFragmentsPerRow = 8;
        static_assert(kFragmentsPerRow == 8, "");
        const unsigned lane = wtid % 16;
        const unsigned k32 = wtid / 16;
#pragma unroll
        for (unsigned row_half = 0; row_half < 2; ++row_half) {
#pragma unroll
            for (unsigned f = 0; f < kFragmentsPerRow; ++f) {
                const unsigned row = row_half * 16 + lane;
                const unsigned col =
                    (f / 4) * 128 + k32 * 32 + (f % 4) * 8;
                const auto *src = &shm_h[row * kGroupN + col];
                out[row_half * kFragmentsPerRow + f] =
                    Pack2(ToBf16RnLocal(*reinterpret_cast<const float4 *>(src)),
                          ToBf16RnLocal(
                              *reinterpret_cast<const float4 *>(src + 4)));
            }
        }
    }
};

template <unsigned kNumWarps, unsigned kGroupN, class ShmReadLayout,
          class OutputRegs>
struct QuantizeAndShuffleMxFp4 {
    static constexpr unsigned kGroupM = 32;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kK128Tiles = kGroupN / 128;
    static constexpr unsigned kK256Tiles = kK128Tiles / 2;
    static constexpr unsigned kInputFragments =
        (kGroupM * kGroupN) / kThreads / 4;
    static constexpr unsigned kOutputFragments = 2 * kK128Tiles;
    static constexpr unsigned kScaleFragments = kK256Tiles;

    using Shm = float4[kThreads * kInputFragments];
    static_assert(kK128Tiles % 2 == 0, "");

    __device__ static void Run(OutputRegs &out, Shm &shm_h,
                               const float4 h[kInputFragments], unsigned tid,
                               unsigned wid, unsigned wtid) {
        (void)tid;
        WriteShm(shm_h, h, wid, wtid);
        __syncthreads();
        ReadShm(out.x, out.scale, shm_h, wtid);
        __syncthreads();
    }

  private:
    __device__ static void WriteShm(Shm &shm_h,
                                    const float4 h[kInputFragments],
                                    unsigned wid, unsigned wtid) {
        static_assert(kGroupN == 256, "native handoff requires K256");
        float *const shm = reinterpret_cast<float *>(&shm_h);
        const unsigned row_lane = wtid % 16;
        const unsigned col_quadrant = wtid / 16;
        for (unsigned i = 0; i < kInputFragments; ++i) {
            const unsigned fragment = i / 2;
            const unsigned m16 = i % 2;
            const unsigned m = m16 * 16 + row_lane;
            const unsigned k = wid * 64 + fragment * 16 +
                               col_quadrant * 4;
#pragma unroll
            for (unsigned component = 0; component < 4; ++component) {
                shm[m * kGroupN + k + component] =
                    reinterpret_cast<const float *>(&h[i])[component];
            }
        }
    }

    __device__ static unsigned PackFloat8(float4 a, float4 b, float scale) {
        unsigned packed = 0;
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
            packed, a.x, a.y, scale, 0);
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
            packed, a.z, a.w, scale, 1);
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
            packed, b.x, b.y, scale, 2);
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
            packed, b.z, b.w, scale, 3);
        return packed;
    }

    __device__ static float MaxAbs(float4 v) {
        return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                     fmaxf(fabsf(v.z), fabsf(v.w)));
    }

    __device__ static float ScaleFloat(float max_abs, unsigned &scale_byte) {
        if (max_abs < 1.0e-12f) {
            scale_byte = 127u;
            return 1.0f;
        }
        // Native FP4 uses an E8M0 scale per 32 values. Pick the smallest
        // power-of-two scale that keeps the block inside the FP4 max of 6.
        const float required = max_abs * (1.0f / 6.0f);
        const unsigned required_bits =
            reinterpret_cast<const unsigned &>(required);
        scale_byte = (required_bits >> 23) & 0xffu;
        if (scale_byte < 0xffu && (required_bits & 0x7fffffu)) {
            ++scale_byte;
        }
        const unsigned bits = scale_byte << 23;
        return reinterpret_cast<const float &>(bits);
    }

    __device__ static void ReadShm(uint4 out[kOutputFragments],
                                   unsigned scales[kScaleFragments],
                                   Shm &shm_h, unsigned wtid) {
        static_assert(kGroupN == 256, "native handoff requires K256");
        const float *const shm = reinterpret_cast<const float *>(&shm_h);
        const unsigned m_lane = wtid % 16;
        const unsigned k32_lane = wtid / 16;
        unsigned packed_scales[kScaleFragments] = {};
        for (unsigned m16 = 0; m16 < 2; ++m16) {
            for (unsigned stage = 0; stage < kK128Tiles; ++stage) {
                const unsigned m = m16 * 16 + m_lane;
                const unsigned k = stage * 128 + k32_lane * 32;
                const float4 *const values =
                    reinterpret_cast<const float4 *>(shm + m * kGroupN + k);
                float max_abs = 0.0f;
#pragma unroll
                for (unsigned vector = 0; vector < 8; ++vector) {
                    max_abs = fmaxf(max_abs, MaxAbs(values[vector]));
                }
                unsigned scale_byte;
                const float scale = ScaleFloat(max_abs, scale_byte);
                uint4 packed;
                packed.x = PackFloat8(values[0], values[1], scale);
                packed.y = PackFloat8(values[2], values[3], scale);
                packed.z = PackFloat8(values[4], values[5], scale);
                packed.w = PackFloat8(values[6], values[7], scale);
                const unsigned out_idx = m16 * kK128Tiles + stage;
                out[out_idx] = packed;
                const unsigned scale_idx = stage / 2;
                const unsigned byte_idx = (stage & 1u) * 2u + m16;
                packed_scales[scale_idx] |= scale_byte << (8 * byte_idx);
            }
        }
        for (unsigned i = 0; i < kScaleFragments; ++i) {
            scales[i] = packed_scales[i];
        }
    }
};

} // namespace causalflow::petit::rocm::moe
