#pragma once

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <unsigned, unsigned> struct NoopBiasLayout {
    static constexpr unsigned kElementBytes = 1;
    static constexpr unsigned kPackedTileElements = 1;

    __host__ __device__ static constexpr unsigned PackedStride(unsigned dim) {
        return dim;
    }

    __device__ void Initialize(const void *, unsigned, unsigned, unsigned,
                               unsigned) {}
    __device__ void AddToAccumulator(float4 *, unsigned, unsigned) const {}
};

} // namespace causalflow::petit::rocm::moe
