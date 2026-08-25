#pragma once

#include "gemm/rocm/amd_intrinsics.cuh"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <unsigned kNumWarps, unsigned kGroupN>
__device__ void
StoreStage1AccumulatorLds(float (&shm)[32 * kGroupN], const float4 *h,
                          unsigned wid, unsigned wtid) {
    static_assert(kGroupN % kNumWarps == 0, "invalid accumulator layout");
    static constexpr unsigned kInputFragments =
        (32 * kGroupN) / (kNumWarps * kWarpSize) / 4;
    static constexpr unsigned kColsPerWarp = kGroupN / kNumWarps;
    const unsigned row_lane = wtid % 16;
    const unsigned col_quadrant = wtid / 16;
#pragma unroll
    for (unsigned fragment = 0; fragment < kInputFragments; ++fragment) {
        const unsigned row = (fragment & 1u) * 16 + row_lane;
        const unsigned col = wid * kColsPerWarp +
                             (fragment / 2) * 16 + col_quadrant * 4;
        *reinterpret_cast<float4 *>(&shm[row * kGroupN + col]) = h[fragment];
    }
}

template <unsigned kGroupM, unsigned kGroupN, unsigned kWarpsM,
          unsigned kWarpsN>
__device__ void StoreStage1AccumulatorLds2D(
    float (&shm)[kGroupM * kGroupN], const float4 *h, unsigned wid,
    unsigned wtid) {
    static_assert(kGroupM == 32 * kWarpsM, "each M wave covers M32");
    static_assert(kGroupN % kWarpsN == 0, "invalid N wave partition");
    static constexpr unsigned kWaveN = kGroupN / kWarpsN;
    static constexpr unsigned kInputFragments = kWaveN / 8;
    const unsigned wave_m = wid / kWarpsN;
    const unsigned wave_n = wid % kWarpsN;
    const unsigned row_lane = wtid % 16;
    const unsigned col_quadrant = wtid / 16;
#pragma unroll
    for (unsigned fragment = 0; fragment < kInputFragments; ++fragment) {
        const unsigned row =
            wave_m * 32 + (fragment & 1u) * 16 + row_lane;
        const unsigned col = wave_n * kWaveN + (fragment / 2) * 16 +
                             col_quadrant * 4;
        *reinterpret_cast<float4 *>(&shm[row * kGroupN + col]) = h[fragment];
    }
}

template <unsigned kGroupN, unsigned kNumWarps>
__device__ void StoreStage1AccumulatorLdsM64(
    float (&shm)[64 * kGroupN], const float4 *h, unsigned wid,
    unsigned wtid) {
    static_assert(kGroupN == 32 * kNumWarps,
                  "each wave owns one N32 slice");
    const unsigned row_lane = wtid % 16;
    const unsigned col_quadrant = wtid / 16;
#pragma unroll
    for (unsigned n16 = 0; n16 < 2; ++n16) {
#pragma unroll
        for (unsigned m16 = 0; m16 < 4; ++m16) {
            const unsigned row = m16 * 16 + row_lane;
            const unsigned col =
                wid * 32 + n16 * 16 + col_quadrant * 4;
            *reinterpret_cast<float4 *>(&shm[row * kGroupN + col]) =
                h[n16 * 4 + m16];
        }
    }
}

} // namespace causalflow::petit::rocm::moe
