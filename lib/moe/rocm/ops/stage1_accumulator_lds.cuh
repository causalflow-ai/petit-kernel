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

} // namespace causalflow::petit::rocm::moe
