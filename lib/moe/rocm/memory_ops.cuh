#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/amd_intrinsics.cuh"
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

__device__ inline BufferResource MakeBufferResource(const void *ptr,
                                                    unsigned range) {
    BufferResource resource;
    resource.v = {
        .ptr = reinterpret_cast<uintptr_t>(ptr),
        .range = range,
        .config = BufferResource::kDataFormatU32Config,
    };
    return resource;
}

enum class MoeArchitecture {
    kCdna3,
    kCdna4,
};

template <MoeArchitecture> struct WeightLoadPolicySelector;

template <>
struct WeightLoadPolicySelector<MoeArchitecture::kCdna3> {
    static constexpr int kAux = BufferResource::kNone;
};

template <>
struct WeightLoadPolicySelector<MoeArchitecture::kCdna4> {
    static constexpr int kAux = BufferResource::kNTBit;
};

#if defined(__gfx950__)
using TargetWeightLoadPolicy =
    WeightLoadPolicySelector<MoeArchitecture::kCdna4>;
#else
using TargetWeightLoadPolicy =
    WeightLoadPolicySelector<MoeArchitecture::kCdna3>;
#endif

template <class Scalar_, unsigned kGroupM_, unsigned kGroupN_,
          unsigned kNumWarps_>
struct MatrixLayout {
    using Scalar = Scalar_;
    static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(Scalar);
    static constexpr unsigned kGroupM = kGroupM_;
    static constexpr unsigned kGroupN = kGroupN_;
    static constexpr unsigned kNumWarps = kNumWarps_;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;

    static constexpr unsigned kRefBufferRange = (unsigned)-16;
    static constexpr unsigned kScaleBlockSize = 128;

    static constexpr unsigned kLoadGlobal = tal::CeilingDiv<unsigned>(
        kGroupM * kGroupN * sizeof(Scalar) / sizeof(uint4), kThreads);

    __device__ void Initialize(const void *value_ptr, unsigned value_range,
                               const void *scale_ptr, unsigned scale_range,
                               unsigned stride_n);

    __device__ float FetchScale(unsigned tid) const;
    BufferResource v_;
    BufferResource scales_;
    unsigned stride_n_;
};

template <class Scalar_, unsigned kGroupM_, unsigned kGroupN_,
          unsigned kNumWarps_>
__device__ void
MatrixLayout<Scalar_, kGroupM_, kGroupN_, kNumWarps_>::Initialize(
    const void *value_ptr, unsigned value_range, const void *scale_ptr,
    unsigned scale_range, unsigned stride_n) {
    v_.v = {
        .ptr = reinterpret_cast<uintptr_t>(value_ptr),
        .range = value_range,
        .config = BufferResource::kDataFormatU32Config,
    };
    scales_.v = {
        .ptr = reinterpret_cast<uintptr_t>(scale_ptr),
        .range = scale_range,
        .config = BufferResource::kDataFormatU32Config,
    };
    stride_n_ = stride_n;
}

template <class Scalar_, unsigned kGroupM_, unsigned kGroupN_,
          unsigned kNumWarps_>
__device__ float
MatrixLayout<Scalar_, kGroupM_, kGroupN_, kNumWarps_>::FetchScale(
    unsigned tid) const {
    unsigned off =
        ((tid & 1) * stride_n_ / kScaleBlockSize) * sizeof(unsigned) +
        (tid & 2) * 2;

    const unsigned u = scales_.template LoadU32<BufferResource::kNone>(off, 0);
    return reinterpret_cast<const float &>(u);
}

template <class Scalar, unsigned kNumWarps_, unsigned kGroupN_>
struct W13Layout {
    static constexpr unsigned kGroupMPerWarp = 64;
    static constexpr unsigned kGroupM = kNumWarps_ * kGroupMPerWarp;
    static constexpr unsigned kGroupN = kGroupN_;
    static constexpr unsigned kScaleBlockSize = 128;
    static constexpr unsigned kTileLoads = kGroupN / 32;

    static_assert(kGroupN % 32 == 0, "");

    __device__ void Initialize(const void *value_ptr, unsigned value_range,
                               const void *scale_ptr, unsigned scale_range,
                               unsigned dim);

    template <int kAux = TargetWeightLoadPolicy::kAux>
    __device__ void LoadTile(uint4 reg[kTileLoads], unsigned stage,
                             unsigned wid, unsigned wtid);
    __device__ float LoadScale(unsigned tid);
    template <int kTileN, int kTileK> __device__ void AdvanceStep();

  private:
    BufferResource v_, scales_;
    unsigned v_offset_, s_offset_;
    unsigned stride_;
};

template <class Scalar, unsigned kNumWarps_, unsigned kGroupN_>
__device__ void W13Layout<Scalar, kNumWarps_, kGroupN_>::Initialize(
    const void *value_ptr, unsigned value_range, const void *scale_ptr,
    unsigned scale_range, unsigned dim) {
    v_.v = {
        .ptr = reinterpret_cast<uintptr_t>(value_ptr),
        .range = value_range,
        .config = BufferResource::kDataFormatU32Config,
    };
    scales_.v = {
        .ptr = reinterpret_cast<uintptr_t>(scale_ptr),
        .range = scale_range,
        .config = BufferResource::kDataFormatU32Config,
    };
    v_offset_ = 0;
    s_offset_ = 0;
    stride_ = dim;
}

template <class Scalar, unsigned kNumWarps_, unsigned kGroupN_>
template <int kAux>
__device__ void W13Layout<Scalar, kNumWarps_, kGroupN_>::LoadTile(
    uint4 reg[kTileLoads], unsigned stage, unsigned wid, unsigned wtid) {
    // Match reference assembly:
    //   v44 = (wid * 16 * stride_bytes) + (lane_id * 16)
    //   v45/v46/v47 = v44 + {64,128,192} * stride_bytes
    const unsigned voffset =
        wid * 16 * stride_ * sizeof(Scalar) + wtid * sizeof(uint4);
    const unsigned row_stride_bytes = kWarpSize * stride_ * sizeof(Scalar);
    for (unsigned i = 0; i < kTileLoads / 2; i++) {
        for (int j = 0; j < 2; j++) {
            reg[i * 2 + j] = v_.template Load<kAux>(
                voffset + i * row_stride_bytes +
                    (stage * 2 + j) * kWarpSize * sizeof(uint4),
                v_offset_);
        }
    }
}

template <class Scalar, unsigned kNumWarps_, unsigned kGroupN_>
__device__ float
W13Layout<Scalar, kNumWarps_, kGroupN_>::LoadScale(unsigned tid) {
    unsigned off = ((tid & 1) * stride_ / kScaleBlockSize) * sizeof(unsigned) +
                   (tid & 2) * 2;
    const unsigned u =
        scales_.template LoadU32<BufferResource::kNone>(off, s_offset_);
    return reinterpret_cast<const float &>(u);
}

template <class Scalar, unsigned kNumWarps_, unsigned kGroupN_>
template <int kTileN, int kTileK>
__device__ void W13Layout<Scalar, kNumWarps_, kGroupN_>::AdvanceStep() {
    static_assert(kTileN == 0 && kTileK == 2,
                  "W13 advances one K256 tile");
    v_offset_ += (kTileLoads / 2) * kWarpSize * sizeof(uint4);
    s_offset_ += 8;
}

template <class Scalar, unsigned kNumWarps_, unsigned kGroupN_>
struct W2Layout
    : public MatrixLayout<Scalar, 64 * kNumWarps_, kGroupN_, kNumWarps_> {
    using Base = MatrixLayout<Scalar, 64 * kNumWarps_, kGroupN_, kNumWarps_>;
    using Base::Base;
    static constexpr unsigned kTileLoads = Base::kLoadGlobal / 2;
    static_assert(Base::kLoadGlobal % 2 == 0, "");

    template <int kAux = TargetWeightLoadPolicy::kAux>
    __device__ void LoadTile(uint4 reg[kTileLoads], unsigned stage,
                             unsigned wid, unsigned wtid);
    __device__ float LoadScale(unsigned tid) const {
        const unsigned off =
            ((tid & 1) * Base::stride_n_ / Base::kScaleBlockSize) *
                sizeof(unsigned) +
            (tid & 2) * 2;
        const unsigned u =
            Base::scales_.template LoadU32<BufferResource::kNone>(off, 0);
        return reinterpret_cast<const float &>(u);
    }
    template <int kTileN, int kTileK> __device__ void AdvanceStep() {
        static_assert(kTileN == 1 && kTileK == 0,
                      "W2 advances one N256 tile");
        v_offset_ += Base::stride_n_ * Base::kGroupN * sizeof(Scalar);
        Base::scales_.v.ptr +=
            Base::stride_n_ / Base::kScaleBlockSize * sizeof(unsigned) * 2;
    }

    unsigned v_offset_ = 0;
};

enum class MxFp4TileShape : unsigned {
    kN256,
    kN128,
    // Four-wave M64 x N256 (per projection): two M waves by two N waves.
    // W13 therefore covers N512 when gate and up are counted together.
    kM64N256,
};

template <MxFp4TileShape kLayout, unsigned kNumWarps>
struct MxFp4WeightLayoutSelector;

template <unsigned kNumWarps>
struct MxFp4WeightLayoutSelector<MxFp4TileShape::kN256, kNumWarps> {
    static constexpr unsigned kTileM = 32;
    static constexpr unsigned kGroupN = 256;
    static constexpr unsigned kTileK = 256;
    static constexpr unsigned kLoadGlobal = 4;
    static constexpr unsigned kWaveTileN = 64;
    __host__ __device__ static constexpr unsigned
    ValueOffsetBytes(unsigned wid, unsigned fragment, unsigned stride_n) {
        return (wid * 64 + fragment * 16) * stride_n / 2;
    }

    __host__ __device__ static constexpr unsigned
    K128OffsetBytes(unsigned stage) {
        return stage * kWarpSize * sizeof(uint4);
    }

    __host__ __device__ static constexpr unsigned
    N32Offset(unsigned wid, unsigned n32_pair) {
        return 2 * wid + n32_pair;
    }
};

template <unsigned kNumWarps>
struct MxFp4WeightLayoutSelector<MxFp4TileShape::kN128, kNumWarps> {
    static constexpr unsigned kTileM = 32;
    static constexpr unsigned kGroupN = 128;
    static constexpr unsigned kTileK = 256;
    static constexpr unsigned kLoadGlobal = 2;
    static constexpr unsigned kWaveTileN = 32;
    __host__ __device__ static constexpr unsigned
    ValueOffsetBytes(unsigned wid, unsigned fragment, unsigned stride_n) {
        return (wid * 32 + fragment * 16) * stride_n / 2;
    }

    __host__ __device__ static constexpr unsigned
    K128OffsetBytes(unsigned stage) {
        return stage * kWarpSize * sizeof(uint4);
    }
};

template <unsigned kNumWarps>
struct MxFp4WeightLayoutSelector<MxFp4TileShape::kM64N256, kNumWarps> {
    static constexpr unsigned kTileM = 64;
    static constexpr unsigned kGroupN = 256;
    static constexpr unsigned kTileK = 256;
    static constexpr unsigned kLoadGlobal = 8;
    static constexpr unsigned kWaveTileN = 128;
    static constexpr unsigned kWarpsN = 2;
    static_assert(kNumWarps == 4);

    __host__ __device__ static constexpr unsigned
    ValueOffsetBytes(unsigned wid, unsigned fragment, unsigned stride_n) {
        const unsigned wave_n = wid % kWarpsN;
        return (wave_n * kWaveTileN + fragment * 16) * stride_n / 2;
    }

    __host__ __device__ static constexpr unsigned
    K128OffsetBytes(unsigned stage) {
        return stage * kWarpSize * sizeof(uint4);
    }

    __host__ __device__ static constexpr unsigned
    N32Offset(unsigned wid, unsigned n32_pair) {
        return 4 * (wid % kWarpsN) + n32_pair;
    }
};

template <unsigned kNumWarps_, MxFp4TileShape kLayout_>
struct MxFp4WeightLayout {
    using LayoutSelector = MxFp4WeightLayoutSelector<kLayout_, kNumWarps_>;
    static constexpr unsigned kGroupM = 128;
    static constexpr unsigned kGroupN = LayoutSelector::kGroupN;
    static constexpr unsigned kWaveTileN = LayoutSelector::kWaveTileN;
    static constexpr unsigned kNumWarps = kNumWarps_;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kRowGroupSize = 32;

    static constexpr unsigned kRefBufferRange = (unsigned)-16;
    static constexpr unsigned kScaleBlockSize = 128;
    static constexpr unsigned kLoadGlobal = LayoutSelector::kLoadGlobal;

    static_assert(kGroupN % 64 == 0, "");

    __device__ void Initialize(const void *value_ptr, unsigned value_range,
                               const void *scale_ptr, unsigned scale_range,
                               unsigned stride_n);

    template <int kAux = TargetWeightLoadPolicy::kAux>
    __device__ void LoadTile(uint4 reg[kLoadGlobal], unsigned stage,
                             unsigned wid, unsigned wtid);
    __device__ unsigned LoadScale(unsigned wid, unsigned wtid,
                                  unsigned n32_pair);

    template <int kTileN, int kTileK> __device__ void AdvanceStep();

    BufferResource v_;
    BufferResource scales_;
    unsigned stride_n_;
    int v_offset_ = 0;
    int s_offset_ = 0;
};

template <unsigned kNumWarps_, MxFp4TileShape kLayout_>
template <int kTileN, int kTileK>
__device__ inline void MxFp4WeightLayout<kNumWarps_, kLayout_>::AdvanceStep() {
    static_assert(kTileK % 2 == 0, "block scales advance in K256 units");
    v_offset_ +=
        kTileN * 256 * stride_n_ / 2 + kTileK * kWarpSize * sizeof(uint4);
    s_offset_ += kTileN * 256 * stride_n_ / kRowGroupSize;
    s_offset_ += (kTileK / 2) * kWarpSize * sizeof(unsigned);
}

template <unsigned kNumWarps_, MxFp4TileShape kLayout_>
__device__ inline void MxFp4WeightLayout<kNumWarps_, kLayout_>::Initialize(
    const void *value_ptr, unsigned value_range, const void *scale_ptr,
    unsigned scale_range, unsigned stride_n) {
    v_.v = {
        .ptr = reinterpret_cast<uintptr_t>(value_ptr),
        .range = value_range,
        .config = BufferResource::kDataFormatU32Config,
    };
    scales_.v = {
        .ptr = reinterpret_cast<uintptr_t>(scale_ptr),
        .range = scale_range,
        .config = BufferResource::kDataFormatU32Config,
    };
    stride_n_ = stride_n;
    v_offset_ = 0;
    s_offset_ = 0;
}

template <unsigned kNumWarps_, MxFp4TileShape kLayout_>
template <int kAux>
__device__ inline void MxFp4WeightLayout<kNumWarps_, kLayout_>::LoadTile(
    uint4 reg[kLoadGlobal], unsigned stage, unsigned wid, unsigned wtid) {
    const unsigned lane_k_offset = wtid * sizeof(uint4);
    const unsigned k_offset = LayoutSelector::K128OffsetBytes(stage);
#pragma unroll
    for (unsigned fragment = 0; fragment < kLoadGlobal; ++fragment) {
        const unsigned voffset =
            LayoutSelector::ValueOffsetBytes(wid, fragment, stride_n_) +
            lane_k_offset + k_offset;
        reg[fragment] = v_.template Load<kAux>(voffset, v_offset_);
    }
}

template <unsigned kNumWarps_, MxFp4TileShape kLayout_>
__device__ inline unsigned
MxFp4WeightLayout<kNumWarps_, kLayout_>::LoadScale(unsigned wid, unsigned wtid,
                                                   unsigned n32_pair) {
    if constexpr (kLayout_ == MxFp4TileShape::kN128) {
        const unsigned off = wid * stride_n_ + wtid * sizeof(unsigned);
        return scales_.template LoadU32<BufferResource::kNone>(off, s_offset_);
    }
    const unsigned k256_blocks = stride_n_ / 256;
    // scale_word = ([N32] * K256_blocks + K256) * 64 + lane. The K256
    // component is carried by s_offset_ or folded into the W2 resource base.
    const unsigned n32 = [&] {
        if constexpr (kLayout_ == MxFp4TileShape::kM64N256)
            return LayoutSelector::N32Offset(wid, n32_pair);
        return 2 * wid + n32_pair;
    }();
    const unsigned word = n32 * k256_blocks * kWarpSize + wtid;
    return scales_.template LoadU32<BufferResource::kNone>(
        word * sizeof(unsigned), s_offset_);
}

template <class Scalar, unsigned kNumWarps_, unsigned kGroupN_>
template <int kAux>
__device__ void W2Layout<Scalar, kNumWarps_, kGroupN_>::LoadTile(
    uint4 reg[kTileLoads], unsigned stage, unsigned wid, unsigned wtid) {
    static constexpr unsigned kInnerStep = kWarpSize * sizeof(uint4);
    const unsigned voffset =
        wid * 16 * Base::stride_n_ * sizeof(Scalar) + wtid * sizeof(uint4);
    for (int i = 0; i < 2; i++) {
        for (unsigned j = 0; j < kTileLoads / 2; j++) {
            reg[j * 2 + i] = Base::v_.template Load<kAux>(
                voffset + 64 * j * Base::stride_n_,
                v_offset_ + (stage * 2 + i) * kInnerStep);
        }
    }
}

} // namespace causalflow::petit::rocm::moe
