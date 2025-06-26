#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/amd_intrinsics.cuh"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::quantization {

static constexpr unsigned kBits = 4;
static constexpr unsigned kPackFactor = 32 / kBits;
static constexpr unsigned kQuantVecSize = sizeof(uint4) / sizeof(unsigned);

// Memory operations related to matrix A and B. A / B have multiple levels of
// hierarchy:
//
// * Global memory. A is continguous and B is shuffled offline into blocks of
// LayoutMxLayoutN tiles.
// * Shared memory. A / B occupy GroupM * GroupK * sizeof(ElementA), GroupN *
// GroupK / kPackFactor * sizeof(uint) bytes respectively.
// * Matmul loop. A / B are loaded into registers using a single 16-byte loads
// across the warps. Each warp exepcts A has kMmaM rows, and B is a
// LayoutM * LayoutN matrix.
template <class Config> struct MatrixALayout {
    static constexpr unsigned kVecSize = Config::kVecSize;
    static constexpr unsigned kGroupM = Config::kGroupM;
    static constexpr unsigned kGroupK = Config::kGroupK;
    static constexpr unsigned kThreads = Config::kThreads;

    using Shm = uint4[kGroupM * kGroupK / kVecSize];

    static constexpr unsigned kLoadGlobalA =
        tal::CeilingDiv<unsigned>(sizeof(Shm) / sizeof(uint4), kThreads);

    __device__ static inline BufferResource
    InitializeBufferResource(unsigned row, unsigned k, const void *a_ptr);

    __device__ static void FetchGlobal(const BufferResource &r_a, unsigned n,
                                       unsigned k, unsigned tid,
                                       uint4 reg_a[kLoadGlobalA]);
    __device__ static void StoreShared(const uint4 reg_a[kLoadGlobalA],
                                       unsigned tid, Shm *shm);

    template <unsigned kReadBatchA>
    __device__ static void
    FetchRegisters(uint4 va[Config::kWarpTileM][kReadBatchA],
                   const Shm &__restrict__ shm, unsigned tile_idx_m,
                   unsigned tile_idx_k, unsigned wtid);
};

template <class Config> struct MatrixBLayout {
    static constexpr unsigned kGroupK = Config::kGroupK;
    static constexpr unsigned kGroupN = Config::kGroupN;
    static constexpr unsigned kThreads = Config::kThreads;
    static constexpr unsigned kLayoutM = Config::kLayoutM;
    static constexpr unsigned kLayoutN = Config::kLayoutN;

    using Shm = uint4[kGroupK * kGroupN / kPackFactor / kQuantVecSize];

    static constexpr unsigned kLoadGlobalB =
        tal::CeilingDiv<unsigned>(sizeof(Shm) / sizeof(uint4), kThreads);

    __device__ static inline BufferResource
    InitializeBufferResource(unsigned n, const void *b_ptr);

    __device__ static void FetchGlobal(const BufferResource &r_b, unsigned n,
                                       unsigned k, unsigned tid,
                                       uint4 reg_b[kLoadGlobalB]);
    __device__ static void StoreShared(const uint4 reg_b[kLoadGlobalB],
                                       unsigned tid, Shm *shm_b);
    // One tile contains kLayoutM * kLayoutN elements
    __device__ static void FetchRegisters(uint4 *__restrict__ qw,
                                          const Shm &__restrict__ shm,
                                          unsigned tile_idx_n,
                                          unsigned tile_idx_k, unsigned wtid);
};

template <class Config> struct ScaleLayout {
    static constexpr unsigned kScaleVecSize = Config::kScaleVecSize;
    static constexpr unsigned kGroupK = Config::kGroupK;
    static constexpr unsigned kGroupN = Config::kGroupN;
    static constexpr unsigned kGroupSize = Config::kGroupSize;
    static constexpr unsigned kThreads = Config::kThreads;
    static constexpr unsigned kScaleZpBatch =
        tal::CeilingDiv(kGroupK, Config::kGroupSize);

    using Shm = uint4[kScaleZpBatch * (kGroupN / kScaleVecSize)];

    static constexpr unsigned kLoadScale = tal::CeilingDiv<unsigned>(
        kGroupN * kScaleZpBatch / kScaleVecSize, kThreads);

    __device__ static inline BufferResource
    InitializeBufferResource(unsigned n, const void *scale_ptr);

    __device__ static void FetchScale(const BufferResource &r_scale, unsigned n,
                                      unsigned tid,
                                      uint4 reg_scale[kLoadScale]);

    __device__ static void StoreScaleShm(const uint4 reg_scale[kLoadScale],
                                         unsigned tid, uint4 *shm_scales);
};

template <class Config> struct ShmBuf {
    static constexpr unsigned kMaxShmSize = 64 * 1024;
    static constexpr unsigned kGroupM = Config::kGroupM;
    static constexpr unsigned kGroupK = Config::kGroupK;
    static constexpr unsigned kGroupN = Config::kGroupN;
    static constexpr unsigned kScaleZpBatch =
        tal::CeilingDiv(kGroupK, Config::kGroupSize);
    static constexpr unsigned kVecSize = Config::kVecSize;

    using LayoutA = MatrixALayout<Config>;
    using LayoutB = MatrixBLayout<Config>;
    using LayoutScale = ScaleLayout<Config>;

    struct ReductionStorage {
        float4 acc[Config::kThreadAccumTileNRegs][Config::kNumWarps / 2]
                  [kWarpSize];
    };

    struct DataWithoutZP {
        LayoutA::Shm a;
        LayoutB::Shm b;
        LayoutScale::Shm scales;
    };

    struct DataWithZP : public DataWithoutZP {
        uint4 zeros[kScaleZpBatch * (kGroupN / kPackFactor / kQuantVecSize)];
    };

    static_assert(std::is_trivial<DataWithZP>::value, "");

    using Data = std::conditional_t<Config::kUseZeroPoints && Config::kZpInShm,
                                    DataWithZP, DataWithoutZP>;

    using Layout = union {
        Data data[Config::kStages];
        uint2 result[Config::kGroupM * Config::kGroupN / 4];
        ReductionStorage red;
    };
};

template <class Config>
__device__ inline BufferResource
MatrixALayout<Config>::InitializeBufferResource(unsigned tile_m, unsigned k,
                                                const void *a_ptr) {
    BufferResource r;
    r.v = {
        .ptr = reinterpret_cast<uintptr_t>(a_ptr),
        .range = static_cast<unsigned>(((tile_m - 1) * k + kGroupK) *
                                       sizeof(typename Config::ElementA)),
        .config = BufferResource::kDataFormatU32Config,
    };
    return r;
}

template <class Config>
__device__ void MatrixALayout<Config>::FetchGlobal(const BufferResource &r_a,
                                                   unsigned n, unsigned k,
                                                   unsigned tid,
                                                   uint4 reg_a[kLoadGlobalA]) {
    // Rely on the range of the buffer to prevent OOB accesses, thus simplifying
    // the control flows.

    for (unsigned i = 0, idx = tid;
         i < tal::CeilingDiv(kGroupM * kGroupK / kVecSize, kThreads);
         ++i, idx += kThreads) {
        unsigned row = idx / (kGroupK / kVecSize),
                 col = idx % (kGroupK / kVecSize);
        reg_a[i] = r_a.Load((row * k / kVecSize + col) * sizeof(uint4), 0,
                            BufferResource::kNone);
    }
}

template <class Config>
__device__ void
MatrixALayout<Config>::StoreShared(const uint4 reg_a[kLoadGlobalA],
                                   unsigned tid, Shm *shm) {
    static constexpr bool kAlignedShmA =
        sizeof(Shm) / sizeof(uint4) % kThreads == 0;

    for (unsigned i = 0, idx = tid; i < kLoadGlobalA; ++i, idx += kThreads) {
        auto shm_coord = Config::WriteShmCoordA(idx);
        auto shm_ptr = GetConditionShmPtr(
            reinterpret_cast<uint4 *>(shm) + shm_coord,
            kAlignedShmA || idx < kGroupM * kGroupK / kVecSize);
        *shm_ptr = reg_a[i];
    }
}

template <class Config>
template <unsigned kReadBatchA>
__device__ void MatrixALayout<Config>::FetchRegisters(
    uint4 va[Config::kWarpTileM][kReadBatchA], const Shm &__restrict__ shm,
    unsigned tile_idx_m, unsigned tile_idx_k, unsigned wtid) {
    for (int m = 0; m < Config::kWarpTileM; m++) {
        for (int j = 0; j < kReadBatchA; j++) {
            auto shm_coord =
                Config::ReadShmCoordA(tile_idx_m + m, tile_idx_k, j, wtid);
            va[m][j] = shm[shm_coord];
        }
    }
}

template <class Config>
__device__ inline BufferResource
MatrixBLayout<Config>::InitializeBufferResource(unsigned n, const void *b_ptr) {
    BufferResource r;
    r.v = {
        .ptr = reinterpret_cast<uintptr_t>(b_ptr),
        .range =
            static_cast<unsigned>((((kGroupK / kLayoutM - 1) * n + kGroupN) *
                                   kLayoutM / kPackFactor) *
                                  sizeof(unsigned)),
        .config = BufferResource::kDataFormatU32Config,
    };
    return r;
}

template <class Config>
__device__ void MatrixBLayout<Config>::FetchGlobal(const BufferResource &r_b,
                                                   unsigned n, unsigned k,
                                                   unsigned tid,
                                                   uint4 reg_b[kLoadGlobalB]) {

    // Rely on the range of the buffer to prevent OOB accesses, thus simplifying
    // the control flows.
    for (unsigned i = 0, idx = tid;
         i < tal::CeilingDiv(kGroupK * kGroupN / kPackFactor / kQuantVecSize,
                             kThreads);
         ++i, idx += kThreads) {
        unsigned row = idx / (kLayoutM * kGroupN / kPackFactor / kQuantVecSize),
                 col = idx % (kLayoutM * kGroupN / kPackFactor / kQuantVecSize);
        reg_b[i] =
            r_b.Load((row * n * kLayoutM / kPackFactor / kQuantVecSize + col) *
                         sizeof(uint4),
                     0, BufferResource::kNone);
    }
}

template <class Config>
__device__ void
MatrixBLayout<Config>::StoreShared(const uint4 reg_b[kLoadGlobalB],
                                   unsigned tid, Shm *shm) {
    static constexpr bool kAlignedShmB =
        sizeof(Shm) / sizeof(uint4) % kThreads == 0;

    for (unsigned i = 0, idx = tid; i < kLoadGlobalB; ++i, idx += kThreads) {
        auto shm_ptr = GetConditionShmPtr(
            reinterpret_cast<uint4 *>(shm) + idx,
            kAlignedShmB ||
                idx < kGroupK * kGroupN / kPackFactor / kQuantVecSize);
        *shm_ptr = reg_b[i];
    }
}

template <class Config>
__device__ void MatrixBLayout<Config>::FetchRegisters(
    uint4 *__restrict__ qw, const Shm &__restrict__ shm, unsigned tile_idx_n,
    unsigned tile_idx_k, unsigned wtid) {
    *qw = shm[tile_idx_k * (kLayoutM * kGroupN / kPackFactor / kQuantVecSize) +
              tile_idx_n * (kLayoutM * kLayoutN / kPackFactor / kQuantVecSize) +
              wtid];
}

template <class Config>
__device__ inline BufferResource
ScaleLayout<Config>::InitializeBufferResource(unsigned n,
                                              const void *scale_ptr) {
    BufferResource r;
    r.v = {
        .ptr = reinterpret_cast<uintptr_t>(scale_ptr),
        .range = ((kScaleZpBatch - 1) * n + kGroupN) *
                 static_cast<unsigned>(sizeof(uint4) / kScaleVecSize),
        .config = BufferResource::kDataFormatU32Config,
    };
    return r;
}

template <class Config>
__device__ void ScaleLayout<Config>::FetchScale(const BufferResource &r_scale,
                                                unsigned n, unsigned tid,
                                                uint4 reg_scale[kLoadScale]) {
    static constexpr unsigned kLayoutM = Config::kLayoutM;
    for (unsigned i = 0, idx = tid; i < kLoadScale; ++i, idx += kThreads) {
        unsigned row = idx / (kLayoutM / kGroupSize * kGroupN / kScaleVecSize),
                 col = idx % (kLayoutM / kGroupSize * kGroupN / kScaleVecSize);
        reg_scale[i] = r_scale.Load(
            (row * n * kLayoutM / kGroupSize / kScaleVecSize + col) *
                sizeof(uint4),
            0, BufferResource::kNone);
    }
}

template <class Config>
__device__ void
ScaleLayout<Config>::StoreScaleShm(const uint4 reg_scale[kLoadScale],
                                   unsigned tid, uint4 *shm_scales) {
    for (unsigned i = 0, idx = tid; i < kLoadScale; ++i, idx += kThreads) {
        auto shm_ptr = GetConditionShmPtr(
            shm_scales + idx, idx < (kGroupN * kScaleZpBatch / kScaleVecSize));
        *shm_ptr = reg_scale[i];
    }
}

} // namespace causalflow::petit::rocm::quantization
