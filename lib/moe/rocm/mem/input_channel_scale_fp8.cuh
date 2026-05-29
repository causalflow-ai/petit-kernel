#pragma once

#include "moe/rocm/memory_ops.cuh"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

// Load input activation into LDS asynchronously and then to registres.  All
// warps have identical activations. A warp collectively uses x_u128[i*4..i*4+4]
// to a swizzled matrix of 16 tokens x (2x32) weights. The token ids are
// [0,8,16,24,1,9,...]. The lower and the upper half of the registers represent
// two 16-token groups.
template <class Config> struct ChannelScaleFp8Input {
    static constexpr unsigned kTokenBatch = Config::kTokenBatch;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kGroupK = kGroupDim;
    static constexpr unsigned kSubGroupSize = 16;
    static constexpr unsigned kActivationFragments = kGroupDim / 32;
    static constexpr unsigned kScaleBlockSize = 128;
    static constexpr unsigned kRefBufferRange = (unsigned)-16;
    static constexpr unsigned kShmInputPaddingBytes = 32 * kNumWarps;
    static constexpr unsigned kShmInputElements =
        kTokenBatch * kThreads + (kShmInputPaddingBytes / sizeof(unsigned));
    static constexpr unsigned kShmInputElementsPerWarp =
        kShmInputElements / kNumWarps;
    static constexpr unsigned kShmInputVec4PerWarp =
        kShmInputElementsPerWarp / (sizeof(uint4) / sizeof(unsigned));

    static_assert(kShmInputElements % kNumWarps == 0, "");
    static_assert(
        kShmInputElementsPerWarp % (sizeof(uint4) / sizeof(unsigned)) == 0, "");

    __device__ void Initialize(const void *value_ptr, const void *scale_ptr,
                               unsigned wid, unsigned m, unsigned n_blocks,
                               unsigned dim) {
        const unsigned *scale_act_ptr =
            reinterpret_cast<const unsigned *>(scale_ptr) + (wid / 2) * m;
        const unsigned value_range = m * dim;
        const unsigned scale_range = (n_blocks - (wid / 2)) * m * sizeof(float);
        Initialize(value_ptr, value_range, scale_act_ptr, scale_range, dim);
    }

    __device__ void Initialize(const void *value_ptr, unsigned value_range,
                               const void *scale_ptr, unsigned scale_range,
                               unsigned dim) {
        values_.v = {
            .ptr = reinterpret_cast<uintptr_t>(value_ptr),
            .range = value_range,
            .config = BufferResource::kDataFormatU32Config,
        };
        scales_.v = {
            .ptr = reinterpret_cast<uintptr_t>(scale_ptr),
            .range = scale_range,
            .config = BufferResource::kDataFormatU32Config,
        };
        dim_ = dim;
        values_offset_bytes_ = 0;
        scales_offset_bytes_ = 0;
    }

    __device__ void FetchAsync(unsigned *shm_x, unsigned wid, unsigned wtid,
                               const unsigned tokens[kTokenBatch]) {
        auto lds_ptr = (__attribute__((address_space(3))) unsigned *)shm_x +
                       wid * kShmInputElementsPerWarp;
        for (int i = 0; i < kTokenBatch; i++) {
            const unsigned src_off = tokens[i] * dim_ + values_offset_bytes_ +
                                     wtid * sizeof(unsigned);
            values_.LoadLds<BufferResource::kNone, sizeof(unsigned), 0>(
                lds_ptr, src_off, 0);
            lds_ptr += kWarpSize;
        }
        values_offset_bytes_ += kGroupK;
    }

    __device__ void FetchToRegs(uint4 regs[kActivationFragments],
                                const unsigned *shm_x, unsigned wtid) const {
        auto x = reinterpret_cast<const uint4 *>(shm_x) +
                 kShmInputVec4PerWarp * (wtid & 3) +
                 ((wtid >> 2) & 3) * (kGroupK / sizeof(uint4)) +
                 wtid / kSubGroupSize;
        static constexpr unsigned kFragmentsPerRow = kActivationFragments / 2;
        for (int i = 0; i < 2; i++) {
            for (unsigned j = 0; j < kFragmentsPerRow; j++) {
                regs[i * kFragmentsPerRow + j] = x[i * kWarpSize + j * 4];
            }
        }
    }

    // The regs are strided by 8 elements (i.e., [0,8), [32, 40), ...) instead
    // of 16 elements. This is for the FP4 layout.
    __device__ void FetchToRegsFP4(uint4 regs[kActivationFragments],
                                   const unsigned *shm_x,
                                   unsigned wtid) const {
        auto x = reinterpret_cast<const uint2 *>(shm_x) +
                 kShmInputVec4PerWarp * 2 * (wtid & 3) +
                 ((wtid >> 2) & 3) * (kGroupK / sizeof(uint2)) +
                 wtid / kSubGroupSize;
        auto r = reinterpret_cast<uint2 *>(regs);
        for (int i = 0; i < 2; i++) {
            for (unsigned j = 0; j < kActivationFragments; j++) {
                r[i * kActivationFragments + j] =
                    x[i * kWarpSize * 2 + j * 4];
            }
        }
    }

    __device__ void FetchScaleAsync(float *shm_scale_x, unsigned wid,
                                    unsigned wtid, const uint2 token_select,
                                    unsigned m) {
        (void)wtid;
        auto lds_ptr =
            (__attribute__((address_space(3))) unsigned *)shm_scale_x +
            wid * kWarpSize;
        const unsigned token_off =
            ((wid & 1) ? token_select.y : token_select.x) * sizeof(unsigned);
        scales_.LoadLds<BufferResource::kNone, sizeof(unsigned), 0>(
            lds_ptr, token_off, scales_offset_bytes_);
        scales_offset_bytes_ +=
            m * (kGroupDim / kScaleBlockSize) * sizeof(float);
    }

    __device__ float4 FetchScaleToReg(const float *shm_scale_x,
                                      unsigned tid) const {
        float4 r{0, 0, 0, 0};
        float *u = reinterpret_cast<float *>(&r);
        for (unsigned i = 0; i < 4; i++) {
            u[i] = shm_scale_x[tid + i * kWarpSize];
        }
        return r;
    }

    BufferResource values_;
    BufferResource scales_;
    unsigned dim_;
    unsigned values_offset_bytes_ = 0;
    unsigned scales_offset_bytes_ = 0;
};

} // namespace causalflow::petit::rocm::moe
