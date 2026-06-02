#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/amd_intrinsics.cuh"
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

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

    __device__ void LoadTile(uint4 reg[kTileLoads], unsigned stage,
                             unsigned wid, unsigned wtid);
    __device__ float LoadScale(unsigned tid);

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
            reg[i * 2 + j] = v_.template Load<BufferResource::kNone>(
                voffset + i * row_stride_bytes +
                    (stage * 2 + j) * kWarpSize * sizeof(uint4),
                v_offset_);
        }
    }
    if (stage == 1) {
        v_offset_ += (kTileLoads / 2) * kWarpSize * sizeof(uint4);
    }
}

template <class Scalar, unsigned kNumWarps_, unsigned kGroupN_>
__device__ float
W13Layout<Scalar, kNumWarps_, kGroupN_>::LoadScale(unsigned tid) {
    unsigned off = ((tid & 1) * stride_ / kScaleBlockSize) * sizeof(unsigned) +
                   (tid & 2) * 2;
    const unsigned u =
        scales_.template LoadU32<BufferResource::kNone>(off, s_offset_);
    s_offset_ += 8;
    return reinterpret_cast<const float &>(u);
}

template <class Scalar, unsigned kNumWarps_, unsigned kGroupN_>
struct W2Layout
    : public MatrixLayout<Scalar, 64 * kNumWarps_, kGroupN_, kNumWarps_> {
    using Base = MatrixLayout<Scalar, 64 * kNumWarps_, kGroupN_, kNumWarps_>;
    using Base::Base;
    static constexpr unsigned kTileLoads = Base::kLoadGlobal / 2;
    static_assert(Base::kLoadGlobal % 2 == 0, "");

    __device__ void LoadTile(uint4 reg[kTileLoads], unsigned stage,
                             unsigned wid, unsigned wtid);
    __device__ float LoadScale(unsigned tid) {
        auto v = Base::FetchScale(tid);
        Base::scales_.v.ptr +=
            Base::stride_n_ / Base::kScaleBlockSize * sizeof(unsigned) * 2;
        return v;
    }

    unsigned v_offset_ = 0;
};

template <unsigned kNumWarps_, unsigned kGroupN_> struct MxFp4WeightLayout {
    static constexpr unsigned kGroupM = 128;
    static constexpr unsigned kGroupN = kGroupN_;
    static constexpr unsigned kNumWarps = kNumWarps_;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kRowGroupSize = 32;

    static constexpr unsigned kRefBufferRange = (unsigned)-16;
    static constexpr unsigned kScaleBlockSize = 128;

    static constexpr unsigned kLoadGlobal = tal::CeilingDiv<unsigned>(
        kGroupM * kGroupN / sizeof(uint4) / 2, kThreads);

    static_assert(kGroupN % 64 == 0, "");

    __device__ void Initialize(const void *value_ptr, unsigned value_range,
                               const void *scale_ptr, unsigned scale_range,
                               unsigned stride_n);

    __device__ void LoadTile(uint4 reg[kLoadGlobal], unsigned stage,
                             unsigned wid, unsigned wtid);
    __device__ unsigned LoadScale(unsigned tid);

    template <int kTileN, int kTileK> __device__ void AdvanceStep();

    BufferResource v_;
    BufferResource scales_;
    unsigned stride_n_;
    int v_offset_ = 0;
    int s_offset_ = 0;
};

template <unsigned kNumWarps_, unsigned kGroupN_>
template <int kTileN, int kTileK>
__device__ inline void MxFp4WeightLayout<kNumWarps_, kGroupN_>::AdvanceStep() {
    constexpr int kTileNStep = 128;
    constexpr int kTileKStep = 4 * 16;
    constexpr int kValueTileKStepBytes = kTileKStep * sizeof(uint4);
    constexpr int kScaleTileKStepBytes = kThreads * sizeof(unsigned);
    v_offset_ +=
        kTileN * kTileNStep * stride_n_ / 2 + kTileK * kValueTileKStepBytes;
    s_offset_ += kTileN * kTileNStep * stride_n_ / kRowGroupSize +
                 kTileK * kScaleTileKStepBytes;
}

template <unsigned kNumWarps_, unsigned kGroupN_>
__device__ inline void MxFp4WeightLayout<kNumWarps_, kGroupN_>::Initialize(
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

template <unsigned kNumWarps_, unsigned kGroupN_>
__device__ inline void MxFp4WeightLayout<kNumWarps_, kGroupN_>::LoadTile(
    uint4 reg[kLoadGlobal], unsigned stage, unsigned wid, unsigned wtid) {
    (void)stage;
    const unsigned voffset = wid * 16 * stride_n_ / 2 + wtid * sizeof(uint4);
    for (int i = 0; i < kLoadGlobal; i++) {
        reg[i] = v_.template Load<BufferResource::kNone>(
            voffset + 64 * i * stride_n_ / 2, v_offset_);
    }
}

template <unsigned kNumWarps_, unsigned kGroupN_>
__device__ inline unsigned
MxFp4WeightLayout<kNumWarps_, kGroupN_>::LoadScale(unsigned tid) {
    unsigned off = tid * sizeof(unsigned);
    return scales_.template LoadU32<BufferResource::kNone>(off, s_offset_);
}

template <class Scalar, unsigned kNumWarps_, unsigned kGroupN_>
__device__ void W2Layout<Scalar, kNumWarps_, kGroupN_>::LoadTile(
    uint4 reg[kTileLoads], unsigned stage, unsigned wid, unsigned wtid) {
    static constexpr unsigned kInnerStep = kWarpSize * sizeof(uint4);
    const unsigned voffset =
        wid * 16 * Base::stride_n_ * sizeof(Scalar) + wtid * sizeof(uint4);
    for (int i = 0; i < 2; i++) {
        for (unsigned j = 0; j < kTileLoads / 2; j++) {
            reg[j * 2 + i] = Base::v_.template Load<BufferResource::kNone>(
                voffset + 64 * j * Base::stride_n_,
                v_offset_ + (stage * 2 + i) * kInnerStep);
        }
    }
    if (stage == 1) {
        v_offset_ += Base::stride_n_ * Base::kGroupN * sizeof(Scalar);
    }
}

} // namespace causalflow::petit::rocm::moe
