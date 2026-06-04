#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "moe/rocm/memory_ops.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <class Config> struct Bf16Input {
    using Scalar = __hip_bfloat16;
    static constexpr unsigned kTokenBatch = Config::kTokenBatch;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kGroupK = kGroupDim;
    static constexpr unsigned kGroupM = kTokenBatch * kNumWarps;
    static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(Scalar);
    static constexpr unsigned kLoadGlobal =
        tal::CeilingDiv<unsigned>(kGroupK * kGroupM, kThreads * kVecSize);
    static constexpr unsigned kVecPerRow = kGroupK / kVecSize;
    static constexpr unsigned kReadRowsPerWave = kWarpSize / kVecPerRow;
    static constexpr unsigned kMmaK = 32;
    static constexpr unsigned kMmaM = 16;
    static constexpr unsigned kActivationFragments = kGroupDim / 16;

    using Shm = uint4[kGroupM * kGroupK / kVecSize];
    using GlobalReadShape =
        tal::Shape<tal::C<kLoadGlobal>,
                   tal::Shape<tal::C<kReadRowsPerWave>,
                              tal::C<kVecPerRow>>>;

    __device__ void Initialize(const void *value_ptr, const void *, unsigned,
                               unsigned m, unsigned, unsigned dim) {
        Initialize(value_ptr, m * dim * sizeof(Scalar), dim);
    }

    __device__ void Initialize(const void *value_ptr, unsigned value_range,
                               unsigned dim) {
        values_.v = {
            .ptr = reinterpret_cast<uintptr_t>(value_ptr),
            .range = value_range,
            .config = BufferResource::kDataFormatU32Config,
        };
        dim_vec_ = dim / kVecSize;
        values_offset_bytes_ = 0;
    }

    __device__ auto MakeGlobalReadLayout() const {
        return tal::make_layout(
            GlobalReadShape{},
            tal::make_stride(kReadRowsPerWave * dim_vec_,
                             tal::make_stride(dim_vec_, tal::_1{})));
    }

    __device__ static auto MakeStoreSharedLayout() {
        return tal::make_layout(
            GlobalReadShape{},
            tal::make_stride(kReadRowsPerWave * kNumWarps * kVecPerRow,
                             tal::make_stride(kNumWarps * kVecPerRow,
                                              tal::_1{})));
    }

    __device__ void FetchGlobal(uint4 regs[kLoadGlobal], unsigned wtid,
                                const unsigned tokens[kTokenBatch]) const {
        const auto layout = MakeGlobalReadLayout();
        for (unsigned i = 0; i < kLoadGlobal; ++i) {
            const unsigned src_vec = layout(tal::make_coord(i, wtid));
            const unsigned token_idx = src_vec / dim_vec_;
            const unsigned col = src_vec - token_idx * dim_vec_;
            regs[i] = values_.template Load<BufferResource::kNone>(
                (tokens[token_idx] * dim_vec_ + col) * sizeof(uint4),
                values_offset_bytes_);
        }
    }

    __device__ void StoreShared(const uint4 regs[kLoadGlobal], unsigned wid,
                                unsigned wtid, Shm *shm_x) const {
        auto *shm = &(*shm_x)[0];
        const auto layout = MakeStoreSharedLayout();
        for (unsigned i = 0; i < kLoadGlobal; ++i) {
            shm[layout(tal::make_coord(i, wtid)) + wid * kVecPerRow] =
                regs[i];
        }
    }

    __device__ void AdvanceStep() {
        values_offset_bytes_ += kGroupK * sizeof(Scalar);
    }

    __device__ void FetchToRegs(uint4 regs[kActivationFragments],
                                const Shm &__restrict__ shm_x,
                                unsigned wtid) const {
        static_assert(kGroupM / 2 == kMmaM, "");
        static_assert(kMmaK == 32, "");
        static_assert(kVecSize == 8, "");

        constexpr unsigned kK128Tiles = kGroupK / 128;
        constexpr unsigned kFragmentsPerRow = kActivationFragments / 2;
        static_assert(kFragmentsPerRow == kK128Tiles * (kMmaK / kVecSize),
                      "");

        const unsigned n16 = wtid % kMmaM;
        const unsigned k32_lane = wtid / kMmaM;
        for (unsigned row = 0; row < 2; ++row) {
            const unsigned shm_row = row * kMmaM + n16;
            for (unsigned k128_tile = 0; k128_tile < kK128Tiles;
                 ++k128_tile) {
                for (unsigned j = 0; j < kMmaK / kVecSize; ++j) {
                    regs[row * kFragmentsPerRow +
                         k128_tile * (kMmaK / kVecSize) + j] =
                        shm_x[shm_row * kVecPerRow +
                              k128_tile * (128 / kVecSize) +
                              k32_lane * (kMmaK / kVecSize) + j];
                }
            }
        }
    }

    BufferResource values_;
    unsigned dim_vec_;
    unsigned values_offset_bytes_ = 0;
};

} // namespace causalflow::petit::rocm::moe
