#pragma once

#include "moe/rocm/memory_ops.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

// MXFP4 bias access differs only in the mapping from [fragment, lane quarter,
// wave] to the packed BF16 vector. Keep that mapping as a TAL layout rather
// than encoding it in separate access classes.
template <unsigned kGroupM, unsigned kGroupN> struct MxFp4BiasLayout;

template <> struct MxFp4BiasLayout<32, 128> {
    // An N128 tile assigns N32 to each wave. Decomposing the wave coordinate
    // maps the two waves in a pair into one packed N64 slice.
    using Type = tal::Layout<
        tal::Shape<tal::_2, tal::_4, tal::Shape<tal::_2, tal::_2>>,
        tal::Stride<tal::_4, tal::_16,
                    tal::Stride<tal::_8, tal::C<64>>>>;
};

template <> struct MxFp4BiasLayout<32, 256> {
    using Type = tal::Layout<tal::Shape<tal::_4, tal::_4, tal::_4>,
                             tal::Stride<tal::_4, tal::_16, tal::C<64>>>;
};

// The M64/N512 W13 tile uses a 2x2 wave grid.  The two M waves load the same
// N256 bias tile, while each N wave owns N128 (eight N16 fragments).
template <> struct MxFp4BiasLayout<64, 256> {
    using Type =
        tal::Layout<tal::Shape<tal::Shape<tal::_4, tal::_2>, tal::_4,
                               tal::Shape<tal::_2, tal::_2>>,
                    tal::Stride<tal::Stride<tal::_4, tal::C<64>>, tal::_16,
                                tal::Stride<tal::C<128>, tal::_0>>>;
};

using MxFp4BiasLayoutM64N256 = typename MxFp4BiasLayout<64, 256>::Type;

// Eight-wave M64 assigns one N32 slice to each wave. Every wave computes all
// four M16 rows, so the wave coordinate only advances through N.
using MxFp4BiasLayoutM64N256W8 =
    tal::Layout<tal::Shape<tal::_2, tal::_4,
                           tal::Shape<tal::_2, tal::_4>>,
                tal::Stride<tal::_4, tal::_16,
                            tal::Stride<tal::_8, tal::C<64>>>>;

template <class Layout_> struct Bf16BiasAccess {
    using Layout = Layout_;

    template <class Bias>
    __device__ static uint2 LoadFragment(const Bias &bias, unsigned fragment,
                                         unsigned tile_col, unsigned tid) {
        const unsigned wid = tid / kWarpSize;
        const unsigned q = (tid % kWarpSize) / 16;
        const unsigned col =
            tile_col + Layout{}(tal::make_coord(fragment, q, wid));
        return bias.v_.template LoadU64<BufferResource::kNone>(
            col * Bias::kElementBytes, 0);
    }

    template <class Bias>
    __device__ static void LoadFragments(const Bias &bias,
                                         uint2 fragments[Bias::kLoadGlobal],
                                         unsigned tile_col, unsigned tid) {
#pragma unroll
        for (unsigned fragment = 0; fragment < Bias::kLoadGlobal; ++fragment) {
            fragments[fragment] =
                LoadFragment(bias, fragment, tile_col, tid);
        }
    }
};

__device__ inline float4 Bf16BiasToFloat(uint2 packed) {
    const auto *bf16 = reinterpret_cast<const __hip_bfloat162 *>(&packed);
    const float2 lo = __bfloat1622float2(bf16[0]);
    const float2 hi = __bfloat1622float2(bf16[1]);
    return float4{lo.x, lo.y, hi.x, hi.y};
}

template <unsigned kNumWarps_, unsigned kGroupN_, class MemoryLayout_,
          unsigned kLoadGlobal_ = kGroupN_ / 64,
          unsigned kMRepeats_ = 2>
struct Bf16BiasLayout {
    static constexpr unsigned kGroupN = kGroupN_;
    static constexpr unsigned kNumWarps = kNumWarps_;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kLoadGlobal = kLoadGlobal_;
    static constexpr unsigned kMRepeats = kMRepeats_;
    static constexpr unsigned kElementBytes = sizeof(__hip_bfloat16);
    static constexpr unsigned kPackedTileElements = 256;
    using Access = Bf16BiasAccess<MemoryLayout_>;

    struct Prefetch {
        uint2 fragments[kLoadGlobal];
    };

    static_assert(kGroupN == 128 || kGroupN == 256,
                  "bias layout expects N128 or N256 tiles");
    static_assert(kNumWarps == 4 || kNumWarps == 8,
                  "bias layout expects four or eight warps");
    static_assert(kThreads == 256 || kThreads == 512,
                  "bias layout expects 256 or 512 threads");

    __host__ __device__ static constexpr unsigned PackedStride(unsigned dim) {
        return tal::CeilingDiv<unsigned>(dim, kPackedTileElements) *
               kPackedTileElements;
    }

    __device__ void Initialize(const void *value_ptr, unsigned expert_id,
                               unsigned, unsigned tile_k,
                               unsigned expert_stride) {
        const unsigned tile_col = tile_k * kGroupN;
        const unsigned value_offset = expert_id * expert_stride + tile_col;
        // Schedulers only issue in-range tiles. A zero descriptor range makes
        // every optional null-bias load return zero in hardware.
        v_.v = {
            .ptr = reinterpret_cast<uintptr_t>(value_ptr) +
                   value_offset * kElementBytes,
            .range = value_ptr == nullptr
                         ? 0
                         : (expert_stride - tile_col) * kElementBytes,
            .config = BufferResource::kDataFormatU32Config,
        };
    }

    __device__ void AddToAccumulator(float4 t[kMRepeats * kLoadGlobal],
                                     unsigned tile_col, unsigned tid) const {
        Prefetch prefetch;
        PrefetchFragments(prefetch, tile_col, tid);
        Apply(t, prefetch);
    }

    __device__ void PrefetchFragments(Prefetch &prefetch, unsigned tile_col,
                                      unsigned tid) const {
        Access::LoadFragments(*this, prefetch.fragments, tile_col, tid);
    }

    __device__ static void Apply(float4 t[kMRepeats * kLoadGlobal],
                                 const Prefetch &prefetch) {
#pragma unroll
        for (unsigned fragment = 0; fragment < kLoadGlobal; ++fragment) {
            const float4 value =
                Bf16BiasToFloat(prefetch.fragments[fragment]);
#pragma unroll
            for (unsigned m16 = 0; m16 < kMRepeats; ++m16)
                reinterpret_cast<v4f &>(
                    t[kMRepeats * fragment + m16]) +=
                    reinterpret_cast<const v4f &>(value);
        }
    }

    BufferResource v_;
};

template <unsigned kNumWarps_, unsigned kGroupN_> struct NoopBiasLayout {
    static constexpr unsigned kNumWarps = kNumWarps_;
    static constexpr unsigned kGroupN = kGroupN_;
    static constexpr unsigned kElementBytes = 1;
    static constexpr unsigned kPackedTileElements = 1;

    struct Prefetch {};

    __host__ __device__ static constexpr unsigned PackedStride(unsigned dim) {
        return dim;
    }

    __device__ void Initialize(const void *, unsigned, unsigned, unsigned,
                               unsigned) {}
    __device__ void AddToAccumulator(float4 *, unsigned, unsigned) const {}
    __device__ void PrefetchFragments(Prefetch &, unsigned, unsigned) const {}

    template <class BiasPrefetch>
    __device__ static void Apply(float4 *, const BiasPrefetch &) {}
};

} // namespace causalflow::petit::rocm::moe
