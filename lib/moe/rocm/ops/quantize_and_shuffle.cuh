#pragma once

#include "gemm/rocm/amd_intrinsics.cuh"
#include "moe/rocm/ops/stage1_accumulator_lds.cuh"
#include "moe/rocm/quantization.cuh"

#include <cmath>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

struct BlockScaleFp8QuantizeShufflePolicy {
    template <unsigned kFragments>
    __device__ static void StorePackedLds(unsigned char *dst,
                                          const unsigned *q, unsigned wid,
                                          unsigned wtid, unsigned cols) {
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
    __device__ static void LoadStage2InputRegs(uint4 (&out)[kFragments],
                                               const unsigned char *src,
                                               unsigned wtid, unsigned cols) {
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

struct PetitMxFp4QuantizeShufflePolicy {
    template <unsigned kFragments>
    __device__ static void StorePackedLds(unsigned char *dst,
                                          const unsigned *q, unsigned wid,
                                          unsigned wtid, unsigned cols) {
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
        static_assert(kKStages == 2,
                      "MXFP4 quantize-and-shuffle expects K256");
        static_assert(kFragments == 8,
                      "MXFP4 quantize-and-shuffle expects eight fragments");

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
    __device__ static void LoadStage2InputRegs(uint4 (&out)[kFragments],
                                               const unsigned char *src,
                                               unsigned wtid, unsigned cols) {
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

template <class Config, class Policy> struct QuantizeAndShuffleFp8 {
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
    static constexpr float kFp8e4m3MaxInv = 1.0f / kFp8E4m3Max;
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
        Policy::template StorePackedLds<kElementsPerThreadVec4>(
            shm.scratch.value, q, wid, wtid, kGroupN);
        if (tid < 16) {
            const float *s = reinterpret_cast<const float *>(&dequant_scale);
            shm.scale[tid * kScaleBlocks] = s[0];
            shm.scale[(16 + tid) * kScaleBlocks] = s[1];
            shm.scale[tid * kScaleBlocks + 1] = s[2];
            shm.scale[(16 + tid) * kScaleBlocks + 1] = s[3];
        }
        __syncthreads();
        Policy::LoadStage2InputRegs(out.x, shm.scratch.value, wtid, kGroupN);
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
        Policy::template ReduceRowMax<kKStages, kElementsPerThreadVec4>(
            row_max, shm_max, h, tid);
#pragma unroll
        for (unsigned block = 0; block < kKStages; ++block) {
            qs[block * 2] =
                kFp8E4m3Max * __builtin_amdgcn_rcpf(row_max[block].x);
            qs[block * 2 + 1] =
                kFp8E4m3Max * __builtin_amdgcn_rcpf(row_max[block].y);
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
            const unsigned block = Policy::ScaleHalf(i, threadIdx.x);
            const float s = scale[block * 2 + row_half];
            const float4 v = h[i];
            q[i] = QuantizeFp8E4m3x4(v, s);
        }
    }
};

template <unsigned kNumWarps, unsigned kGroupN, class OutputRegs_>
struct PackAndShuffleBf16 {
    static constexpr unsigned kGroupM = 32;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kInputFragments =
        (kGroupM * kGroupN) / kThreads / 4;
    static constexpr unsigned kOutputFragments = 2 * kInputFragments;
    static constexpr unsigned kElementsPerThreadVec4 = kOutputFragments;
    using OutputRegs = OutputRegs_;
    using Shm = float[kGroupM * kGroupN];

    __device__ static void Run(OutputRegs &out, Shm &shm_h,
                               const float4 h[kInputFragments], unsigned tid,
                               unsigned wid, unsigned wtid) {
        (void)tid;
        StoreStage1AccumulatorLds<kNumWarps, kGroupN>(shm_h, h, wid, wtid);
        __syncthreads();
        LoadBf16Stage2InputRegs(out.x, shm_h, wtid);
        __syncthreads();
    }

  private:
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

    __device__ static void
    LoadBf16Stage2InputRegs(uint4 out[kOutputFragments], const Shm &shm_h,
                            unsigned wtid) {
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

template <unsigned kNumWarps, unsigned kGroupN, class OutputRegs_>
struct QuantizeAndShuffleMxFp4 {
    static constexpr unsigned kGroupM = 32;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kK128Tiles = kGroupN / 128;
    static constexpr unsigned kInputFragments =
        (kGroupM * kGroupN) / kThreads / 4;
    using OutputRegs = OutputRegs_;
    using Shm = float[kGroupM * kGroupN];

    static_assert(kGroupN == 256, "native MXFP4 shuffle expects N256");

    __device__ static void Run(OutputRegs &out, Shm &shm_h,
                               const float4 h[kInputFragments],
                               unsigned tid, unsigned wid, unsigned wtid) {
        (void)tid;
        StoreStage1AccumulatorLds<kNumWarps, kGroupN>(shm_h, h, wid, wtid);
        __syncthreads();

#pragma unroll
        for (unsigned scale = 0; scale < kK128Tiles / 2; ++scale)
            out.scale[scale] = 0;

        const unsigned m_lane = wtid % 16;
        const unsigned k32_lane = wtid / 16;
#pragma unroll
        for (unsigned m16 = 0; m16 < 2; ++m16) {
#pragma unroll
            for (unsigned k128 = 0; k128 < kK128Tiles; ++k128) {
                const unsigned row = m16 * 16 + m_lane;
                const unsigned col = k128 * 128 + k32_lane * 32;
                const auto *values = reinterpret_cast<const float4 *>(
                    &shm_h[row * kGroupN + col]);
                float max_abs = 0.0f;
#pragma unroll
                for (unsigned vector = 0; vector < 8; ++vector) {
                    max_abs = fmaxf(
                        max_abs,
                        NativeMxFp4Quantization::MaximumAbs(values[vector]));
                }

                alignas(uint4) unsigned char packed[sizeof(uint4)];
                const unsigned scale_byte =
                    QuantizeMxFp4<NativeMxFp4Quantization, 8>(
                        packed, values, max_abs);
                out.x[m16 * kK128Tiles + k128] =
                    *reinterpret_cast<const uint4 *>(packed);
                const unsigned scale_idx = k128 / 2;
                const unsigned byte_idx = (k128 & 1u) * 2u + m16;
                out.scale[scale_idx] |= scale_byte << (8 * byte_idx);
            }
        }
        __syncthreads();
    }
};

} // namespace causalflow::petit::rocm::moe
