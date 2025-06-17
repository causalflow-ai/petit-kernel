#pragma once

#include "gemm/rocm/amd_intrinsics.cuh"

namespace causalflow::petit::rocm::quantization {

// The shape of tile stored in the shared memory
template <class ElementA_, unsigned kGroupSize_, unsigned kNumTileM_,
          unsigned kNumTileN_, unsigned kNumTileK_>
struct TileShape {
    using ElementA = ElementA_;
    static constexpr unsigned kTile = 16;

    static constexpr unsigned kNumTileM = kNumTileM_;
    static constexpr unsigned kNumTileN = kNumTileN_;
    static constexpr unsigned kNumTileK = kNumTileK_;
    static constexpr unsigned kGroupSize = kGroupSize_;

    static constexpr unsigned kGroupM = kNumTileM * kTile;
    static constexpr unsigned kGroupN = kNumTileN * kTile;
    static constexpr unsigned kGroupK = kNumTileK * kTile;
};

// Describe how warps are partitioned for the matmul operation.
template <unsigned kNumWarps_, unsigned kPartitionN_, unsigned kPartitionK_>
struct WarpPartition {
    static constexpr unsigned kPartitionN = kPartitionN_;
    static constexpr unsigned kPartitionK = kPartitionK_;
    static constexpr unsigned kNumWarps = kNumWarps_;
    static constexpr unsigned kThreads = kNumWarps_ * kWarpSize;

    static_assert(kNumWarps_ == kPartitionN * kPartitionK,
                  "The number of warps must be equal to the product of the "
                  "partition dimensions");

    // Arrange the wid in (k, m, n) order. for now we fix m to 1.

    // The index in the k dimension where the accumulation is performed.
    __device__ static constexpr unsigned AccRowK(unsigned wid) {
        return wid / kPartitionN;
    }

    // Index of the independent dimension of n.
    __device__ static constexpr unsigned WarpColN(unsigned wid) {
        return wid % kPartitionN;
    }
};

} // namespace causalflow::petit::rocm::quantization