#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "moe/rocm/memory_ops.cuh"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <class Config> struct MxFp4Input {
    static constexpr unsigned kDim = Config::kDim;
    static constexpr unsigned kTokenBatch = Config::kTokenBatch;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kWarpsM = Config::kStage1WarpsM;
    static constexpr unsigned kWarpsN = Config::kStage1WarpsN;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kGroupK = kGroupDim;
    static constexpr unsigned kK128Tiles = kGroupK / 128;
    static constexpr unsigned kK256Tiles = kGroupK / 256;
    static constexpr unsigned kK32PerTile = 4;
    static constexpr unsigned kRowVecsPerTile = kK128Tiles * kK32PerTile;
    static constexpr unsigned kMmaRows = 16;
    static constexpr unsigned kAsyncVecsPerWarp =
        kTokenBatch * kRowVecsPerTile;
    static constexpr unsigned kLoadIterations =
        tal::CeilingDiv<unsigned>(kAsyncVecsPerWarp, kWarpSize);
    static constexpr unsigned kActivationFragments = 2 * kK128Tiles;
    static constexpr unsigned kScaleFragments = kK256Tiles;
    static constexpr unsigned kScaleBlockSize = 32;
    static constexpr unsigned kRowVecs = kDim / kScaleBlockSize;
    static constexpr unsigned kScaleBlocksPerRouteGroup = kDim / 256;
    static constexpr unsigned kScaleWordsPerM32 = kWarpSize;

    static_assert(kGroupK % 128 == 0, "");
    static_assert(kGroupK % 256 == 0, "");
    static_assert(kGroupK / kScaleBlockSize == kRowVecsPerTile, "");
    static_assert(kNumWarps == 4, "");
    static_assert(kWarpSize == kMmaRows * kK32PerTile, "");
    static_assert(kLoadIterations == 1 || kLoadIterations == 2, "");
    static_assert(kScaleFragments == 1, "");
    static_assert(kWarpsM * kWarpsN == kNumWarps, "");
    static_assert(kTokenBatch * kNumWarps == 32 * kWarpsM, "");

    struct Shm {
        uint4 act[kLoadIterations * kThreads];
        unsigned scale[kScaleFragments * kScaleWordsPerM32 * kWarpsM];
    };

    __device__ static auto MakeRowVecLayout() {
        return tal::make_layout(
            tal::make_shape(tal::C<kK128Tiles>{}, tal::C<kK32PerTile>{}),
            tal::make_stride(tal::C<kK32PerTile>{}, tal::_1{}));
    }

    TAL_HOST_DEVICE static constexpr unsigned
    SwizzledVector(unsigned row, unsigned vector) {
        return vector ^ (row & (kRowVecsPerTile - 1));
    }

    __device__ void Initialize(const void *value_ptr, const void *scale_ptr,
                               unsigned, unsigned m, unsigned,
                               unsigned route_group,
                               unsigned route_group_limit) {
        values_.v = {
            .ptr = reinterpret_cast<uintptr_t>(value_ptr),
            .range = m * kDim / 2,
            .config = BufferResource::kDataFormatU32Config,
        };
        scales_.v = {
            .ptr = reinterpret_cast<uintptr_t>(scale_ptr),
            .range = route_group_limit * kWarpsM * kDim,
            .config = BufferResource::kDataFormatU32Config,
        };
        route_group_ = route_group;
        values_offset_vec_ = 0;
        scales_offset_vec_ = 0;
    }

    __device__ void FetchAsync(uint4 *shm_x, unsigned wid, unsigned wtid,
                               const unsigned tokens[kTokenBatch]) {
        const auto row_vec_layout = MakeRowVecLayout();
        const unsigned offset = values_offset_vec_;
        values_offset_vec_ += kGroupK / kScaleBlockSize;
        for (unsigned load = 0; load < kLoadIterations; ++load) {
            const unsigned linear = load * kWarpSize + wtid;
            if (linear >= kAsyncVecsPerWarp) {
                continue;
            }
            const unsigned token_idx = linear / kRowVecsPerTile;
            const unsigned row_vec = linear - token_idx * kRowVecsPerTile;
            const unsigned source_row_vec =
                SwizzledVector(token_idx, row_vec);
            const unsigned k128 = source_row_vec / kK32PerTile;
            const unsigned k32 = source_row_vec - k128 * kK32PerTile;
            const unsigned dst_idx = wid * kAsyncVecsPerWarp +
                                     load * kWarpSize;
            auto lds_ptr =
                (__attribute__((address_space(3))) unsigned *)(shm_x +
                                                               dst_idx);
            const unsigned byte_offset =
                (tokens[token_idx] * kRowVecs + offset +
                 row_vec_layout(tal::make_coord(k128, k32))) *
                sizeof(uint4);
            values_.LoadLds<BufferResource::kNone, sizeof(uint4), 0>(
                lds_ptr, byte_offset, 0);
        }
    }

    __device__ void FetchScaleAsync(unsigned *shm_scale, unsigned wid,
                                    unsigned wtid, const unsigned *,
                                    unsigned) {
        if (wid >= kWarpsM) {
            return;
        }
        const unsigned offset = scales_offset_vec_;
        auto lds_ptr =
            (__attribute__((address_space(3))) unsigned *)shm_scale;
        const unsigned m32_group = route_group_ * kWarpsM + wid;
        const unsigned src_word =
            (m32_group * kScaleBlocksPerRouteGroup + offset) * kWarpSize +
            wtid;
        lds_ptr += wid * kScaleWordsPerM32;
        scales_.LoadLds<BufferResource::kNone, sizeof(unsigned), 0>(
            lds_ptr, src_word * sizeof(unsigned), 0);
    }

    __device__ void FetchToRegs(uint4 regs[kActivationFragments],
                                const uint4 *shm_x, unsigned wtid) const {
        const unsigned wid = threadIdx.x / kWarpSize;
        const unsigned wave_m = wid / kWarpsN;
        const unsigned row = wtid & (kMmaRows - 1);
        const unsigned vector = wtid / kMmaRows;
        const unsigned row_base =
            (wave_m * 32 + row) * kRowVecsPerTile;
        const uint4 *const k0 =
            shm_x + row_base + SwizzledVector(row, vector);
        const uint4 *const k1 =
            shm_x + row_base + SwizzledVector(row, vector + kK32PerTile);
        static constexpr unsigned kNextRow = kMmaRows * kRowVecsPerTile;
        regs[0] = k0[0];
        regs[1] = k1[0];
        regs[2] = k0[kNextRow];
        regs[3] = k1[kNextRow];
    }

    __device__ unsigned FetchScaleToReg(const unsigned *shm_scale,
                                        unsigned wtid) const {
        const unsigned wid = threadIdx.x / kWarpSize;
        const unsigned wave_m = wid / kWarpsN;
        return shm_scale[wave_m * kScaleWordsPerM32 + wtid];
    }

    __device__ void AdvanceScaleStep() {
        scales_offset_vec_ += kScaleFragments;
    }

    BufferResource values_;
    BufferResource scales_;
    unsigned route_group_ = 0;
    unsigned values_offset_vec_ = 0;
    unsigned scales_offset_vec_ = 0;
};

} // namespace causalflow::petit::rocm::moe
