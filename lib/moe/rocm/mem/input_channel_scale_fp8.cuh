#pragma once

#include "moe/rocm/memory_ops.cuh"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

// Load input activation into row-major LDS and then adapt it to the hardware
// MFMA operand layout. Wave w owns rows [w*8, w*8+8).
template <class Config> struct ChannelScaleFp8Input {
    static constexpr unsigned kDim = Config::kDim;
    static constexpr unsigned kTokenBatch = Config::kTokenBatch;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kGroupK = kGroupDim;
    static constexpr unsigned kSubGroupSize = 16;
    static constexpr unsigned kActivationFragments = kGroupDim / 32;
    static constexpr unsigned kScaleBlockSize = 128;
    static constexpr unsigned kRefBufferRange = (unsigned)-16;
    static constexpr unsigned kWordsPerRow = kGroupK / sizeof(unsigned);
    static constexpr unsigned kShmInputElements =
        kTokenBatch * kNumWarps * kWordsPerRow;
    static constexpr unsigned kScaleBlocks = kGroupK / kScaleBlockSize;
    static constexpr unsigned kShmScaleElements =
        kTokenBatch * kNumWarps * kScaleBlocks;

    __device__ void Initialize(const void *value_ptr, const void *scale_ptr,
                               unsigned wid, unsigned m, unsigned n_blocks) {
        (void)wid;
        const unsigned *scale_act_ptr =
            reinterpret_cast<const unsigned *>(scale_ptr);
        const unsigned value_range = m * kDim;
        const unsigned scale_range = n_blocks * m * sizeof(float);
        Initialize(value_ptr, value_range, scale_act_ptr, scale_range);
    }

    __device__ void Initialize(const void *value_ptr, unsigned value_range,
                               const void *scale_ptr, unsigned scale_range) {
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
        values_offset_bytes_ = 0;
        scales_offset_bytes_ = 0;
    }

    __device__ void FetchAsync(unsigned *shm_x, unsigned wid, unsigned wtid,
                               const unsigned tokens[kTokenBatch]) {
        for (int i = 0; i < kTokenBatch; i++) {
            const unsigned row = wid * kTokenBatch + i;
            auto lds_ptr =
                (__attribute__((address_space(3))) unsigned *)shm_x +
                row * kWordsPerRow;
            const unsigned src_off = tokens[i] * kDim + values_offset_bytes_ +
                                     wtid * sizeof(unsigned);
            values_.LoadLds<BufferResource::kNone, sizeof(unsigned), 0>(
                lds_ptr, src_off, 0);
        }
        values_offset_bytes_ += kGroupK;
    }

    __device__ void FetchToRegs(uint4 regs[kActivationFragments],
                                const unsigned *shm_x, unsigned wtid) const {
        const auto *x = reinterpret_cast<const uint4 *>(shm_x);
        const unsigned m4 = (wtid >> 2) & 3;
        const unsigned m1 = wtid & 3;
        const unsigned k32 = wtid / kSubGroupSize;
        static constexpr unsigned kFragmentsPerRow = kActivationFragments / 2;
        for (int i = 0; i < 2; i++) {
            const unsigned row = i * 16 + m4 * 4 + m1;
            for (unsigned j = 0; j < kFragmentsPerRow; j++) {
                const unsigned word_col = k32 * 4 + j * 16;
                regs[i * kFragmentsPerRow + j] =
                    x[(row * kWordsPerRow + word_col) / 4];
            }
        }
    }

    // The regs are strided by 8 elements (i.e., [0,8), [32, 40), ...) instead
    // of 16 elements. This is for the FP4 layout.
    __device__ void FetchToRegsFP4(uint4 regs[kActivationFragments],
                                   const unsigned *shm_x,
                                   unsigned wtid) const {
        const auto *x = reinterpret_cast<const uint2 *>(shm_x);
        const unsigned m4 = (wtid >> 2) & 3;
        const unsigned m1 = wtid & 3;
        const unsigned k32 = wtid / kSubGroupSize;
        static constexpr unsigned kUint2PerRow = kGroupK / sizeof(uint2);
        auto r = reinterpret_cast<uint2 *>(regs);
        for (int i = 0; i < 2; i++) {
            const unsigned row = i * 16 + m4 * 4 + m1;
            for (unsigned j = 0; j < kActivationFragments; j++) {
                r[i * kActivationFragments + j] =
                    x[row * kUint2PerRow + k32 + j * 4];
            }
        }
    }

    __device__ void FetchScaleAsync(float *shm_scale_x, unsigned wid,
                                    unsigned wtid,
                                    const unsigned tokens[kTokenBatch],
                                    unsigned m) {
        static constexpr unsigned kScaleLoadLanes =
            kTokenBatch * kScaleBlocks;
        static_assert(kScaleLoadLanes == 16, "scale load uses one lane row");
        if (wtid < kScaleLoadLanes) {
            const unsigned local_row = wtid / kScaleBlocks;
            const unsigned scale_block = wtid % kScaleBlocks;
            auto lds_ptr =
                (__attribute__((address_space(3))) unsigned *)shm_scale_x +
                wid * kTokenBatch * kScaleBlocks;
            const unsigned token_off =
                (scale_block * m + tokens[local_row]) * sizeof(unsigned);
            scales_.LoadLds<BufferResource::kNone, sizeof(unsigned), 0>(
                lds_ptr, token_off, scales_offset_bytes_);
        }
        scales_offset_bytes_ +=
            m * (kGroupDim / kScaleBlockSize) * sizeof(float);
    }

    __device__ float4 FetchScaleToReg(const float *shm_scale_x,
                                      unsigned tid) const {
        const unsigned m4 = (tid >> 2) & 3;
        const unsigned m1 = tid & 3;
        const unsigned row0 = m4 * 4 + m1;
        const unsigned row1 = 16 + row0;
        return float4{shm_scale_x[row0 * kScaleBlocks],
                      shm_scale_x[row1 * kScaleBlocks],
                      shm_scale_x[row0 * kScaleBlocks + 1],
                      shm_scale_x[row1 * kScaleBlocks + 1]};
    }

    BufferResource values_;
    BufferResource scales_;
    unsigned values_offset_bytes_ = 0;
    unsigned scales_offset_bytes_ = 0;
};

} // namespace causalflow::petit::rocm::moe
