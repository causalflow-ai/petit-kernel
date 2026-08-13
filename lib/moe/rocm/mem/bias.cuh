#pragma once

#include "moe/rocm/memory_ops.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <unsigned kNumWarps_, unsigned kGroupN_> struct Bf16BiasLayout {
    static constexpr unsigned kGroupN = kGroupN_;
    static constexpr unsigned kNumWarps = kNumWarps_;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kLoadGlobal = kGroupN / 64;
    static constexpr unsigned kElementBytes = sizeof(__hip_bfloat16);
    static constexpr unsigned kPackedTileElements = kGroupN;

    static_assert(kGroupN % 64 == 0, "");
    static_assert(kGroupN == 256, "block bias layout expects 256-col tiles");
    static_assert(kNumWarps == 4, "block bias layout expects four warps");
    static_assert(kThreads == 256, "block bias layout expects 256 threads");

    __host__ __device__ static constexpr unsigned PackedStride(unsigned dim) {
        return tal::CeilingDiv<unsigned>(dim, kPackedTileElements) *
               kPackedTileElements;
    }

    __device__ void Initialize(const void *value_ptr, unsigned expert_id,
                               unsigned dim, unsigned tile_k,
                               unsigned expert_stride) {
        const unsigned packed_stride = PackedStride(dim);
        const unsigned tile_col = tile_k * kGroupN;
        if (value_ptr == nullptr || expert_stride == 0 ||
            tile_col >= packed_stride) {
            v_.v = {
                .ptr = 0,
                .range = 0,
                .config = BufferResource::kDataFormatU32Config,
            };
            return;
        }
        const unsigned value_offset =
            expert_id * expert_stride + tile_col;
        const auto *ptr =
            reinterpret_cast<const __hip_bfloat16 *>(value_ptr) + value_offset;
        v_.v = {
            .ptr = reinterpret_cast<uintptr_t>(ptr),
            .range = (packed_stride - tile_col) * kElementBytes,
            .config = BufferResource::kDataFormatU32Config,
        };
    }

    __device__ void AddToAccumulator(float4 t[2 * kLoadGlobal],
                                     unsigned tile_col, unsigned tid) const {
        const unsigned wid = tid / kWarpSize;
        const unsigned q = (tid % kWarpSize) / 16;
#pragma unroll
        for (unsigned fragment = 0; fragment < kLoadGlobal; ++fragment) {
            // Packed bias is [wave][q][fragment][component], matching
            // C[16*m16+r, 64*wave+16*fragment+4*q+component].
            const unsigned col = tile_col + wid * 64 + q * 16 + fragment * 4;
            const uint2 packed = v_.template LoadU64<BufferResource::kNone>(
                col * kElementBytes, 0);
            const auto *bf16 =
                reinterpret_cast<const __hip_bfloat162 *>(&packed);
            const float2 lo = __bfloat1622float2(bf16[0]);
            const float2 hi = __bfloat1622float2(bf16[1]);
            const float4 add{lo.x, lo.y, hi.x, hi.y};
            reinterpret_cast<v4f &>(t[2 * fragment]) +=
                reinterpret_cast<const v4f &>(add);
            reinterpret_cast<v4f &>(t[2 * fragment + 1]) +=
                reinterpret_cast<const v4f &>(add);
        }
    }

    BufferResource v_;
};

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
