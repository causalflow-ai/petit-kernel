#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "memory_ops.cuh"
#include <type_traits>

#include <hip/hip_bf16.h>

namespace causalflow::petit::rocm::quantization {

namespace detail {

template <class Config> struct WriteResultLayoutBase {
    static constexpr unsigned kTile = Config::kTile;
    static constexpr unsigned kGroupN = Config::kGroupN;
    static constexpr unsigned kThreads = Config::kThreads;
    static constexpr unsigned kResultVecSize = sizeof(uint4) / sizeof(half);
    static constexpr unsigned kPartitionM = Config::WP::kPartitionM;
    static constexpr unsigned kPartitionN = Config::WP::kPartitionN;
    static constexpr unsigned kShmVecSize = sizeof(uint2) / sizeof(half);
    static constexpr unsigned kResultMTilePerPartition =
        ShmBuf<Config>::kResultMTilePerPartition;

    using ShmShape =
        tal::Shape<tal::C<kResultMTilePerPartition>,
                   tal::C<Config::kThreadAccumTileNRegs>, tal::C<kPartitionM>,
                   tal::C<kPartitionN>, tal::Shape<tal::_16, tal::_4>>;
};

template <class Config>
struct WriteResultLayoutEven : public WriteResultLayoutBase<Config> {
    using T = WriteResultLayoutBase<Config>;
    __device__ static auto GetShmLayout() {
        using namespace tal;
        using ShmStride =
            Stride<C<T::kTile * T::kGroupN / T::kShmVecSize>, _4,
                   C<T::kResultMTilePerPartition * T::kTile * T::kGroupN /
                     T::kShmVecSize>,
                   C<T::kGroupN / T::kPartitionN / T::kShmVecSize>,
                   Stride<C<T::kGroupN / T::kShmVecSize>, _1>>;
        using ShmLayout = Layout<typename T::ShmShape, ShmStride>;
        return ShmLayout{};
    }

    __device__ static void WriteBackIdxToCoord(unsigned idx,
                                               unsigned m_tile_start,
                                               unsigned *row, unsigned *col) {
        static constexpr unsigned kResultGroupM =
            T::kResultMTilePerPartition * T::kPartitionM * T::kTile;
        unsigned partition = idx / (kResultGroupM / T::kPartitionM *
                                    T::kGroupN / T::kResultVecSize);
        unsigned partition_idx = idx % (kResultGroupM / T::kPartitionM *
                                        T::kGroupN / T::kResultVecSize);
        *row = partition_idx / (T::kGroupN / T::kResultVecSize) +
               (partition * Config::kWarpTileM + m_tile_start) * T::kTile;
        *col = partition_idx % (T::kGroupN / T::kResultVecSize);
    }
};

template <class Config>
struct WriteResultLayoutUneven : public WriteResultLayoutBase<Config> {
    using T = WriteResultLayoutBase<Config>;
    __device__ static auto GetShmLayout() {
        using namespace tal;
        using ShmStride =
            Stride<C<T::kPartitionM * T::kTile * T::kGroupN / T::kShmVecSize>,
                   _4, C<T::kTile * T::kGroupN / T::kShmVecSize>,
                   C<T::kGroupN / T::kPartitionN / T::kShmVecSize>,
                   Stride<C<T::kGroupN / T::kShmVecSize>, _1>>;
        using ShmLayout = Layout<typename T::ShmShape, ShmStride>;
        return ShmLayout{};
    }

    __device__ static void WriteBackIdxToCoord(unsigned idx,
                                               unsigned m_tile_start,
                                               unsigned *row, unsigned *col) {
        unsigned stripe =
            idx / (T::kPartitionM * T::kTile * T::kGroupN / T::kResultVecSize);
        unsigned stripe_idx =
            idx % (T::kPartitionM * T::kTile * T::kGroupN / T::kResultVecSize);
        unsigned partition =
            stripe_idx / (T::kTile * T::kGroupN / T::kResultVecSize);
        unsigned partition_idx =
            stripe_idx % (T::kTile * T::kGroupN / T::kResultVecSize);
        *row =
            partition_idx / (T::kGroupN / T::kResultVecSize) +
            (partition * Config::kWarpTileM + m_tile_start + stripe) * T::kTile;
        *col = partition_idx % (T::kGroupN / T::kResultVecSize);
    }
};

} // namespace detail

template <class Config, bool kHasAlpha>
__device__ static void
WriteResult(float alpha, typename ShmBuf<Config>::Layout *__restrict__ shm_buf,
            typename Config::ThreadAccum &__restrict__ acc,
            uint4 *__restrict__ c_ptr, unsigned tid, unsigned m, unsigned n) {
    using namespace tal;
    static constexpr bool kIsHalf =
        std::is_same<typename Config::ElementA, half>::value;
    using VectorType = std::conditional_t<kIsHalf, half2, __hip_bfloat162>;
    static constexpr unsigned kGroupN = Config::kGroupN;
    static constexpr unsigned kThreads = Config::kThreads;
    static constexpr unsigned kResultVecSize = sizeof(uint4) / sizeof(half);
    static constexpr unsigned kPartitionM = Config::WP::kPartitionM;
    static constexpr unsigned kResultMTilePerPartition =
        ShmBuf<Config>::kResultMTilePerPartition;
    static constexpr unsigned kWriteLoop =
        tal::CeilingDiv(Config::kWarpTileM, kResultMTilePerPartition);

    static constexpr bool kIsEvenWriteBack =
        Config::kWarpTileM % ShmBuf<Config>::kResultMTilePerPartition == 0;
    using Op = std::conditional_t<kIsEvenWriteBack,
                                  detail::WriteResultLayoutEven<Config>,
                                  detail::WriteResultLayoutUneven<Config>>;

    const unsigned wid = tid / kWarpSize, wtid = tid % kWarpSize,
                   acc_row = Config::WP::WarpK(wid);

    auto shm_layout = Op::GetShmLayout();

    BufferResource br = {
        .v =
            {
                .ptr = reinterpret_cast<uintptr_t>(c_ptr),
                .range = ((m - 1) * n + kGroupN) *
                         static_cast<unsigned>(sizeof(half)),
                .config = BufferResource::kDataFormatU32Config,
            },
    };

#pragma unroll
    for (int q = 0, m_tile_base = 0; q < kWriteLoop;
         q++, m_tile_base += kResultMTilePerPartition) {
        if (acc_row == 0) {
#pragma unroll
            for (int loc = 0, m_tile = m_tile_base;
                 loc < kResultMTilePerPartition && m_tile < Config::kWarpTileM;
                 loc++, m_tile++) {
#pragma unroll
                for (int j = 0; j < Config::kThreadAccumTileNRegs; j++) {
                    VectorType v[2];
                    for (int l = 0; l < 2; l++) {
                        float2 f2 = reinterpret_cast<const float2 *>(
                            &acc[m_tile][j])[l];
                        if constexpr (kHasAlpha) {
                            float2 alpha2{alpha, alpha};
                            f2 = amdgcn_pk_mul_f32(f2, alpha2);
                        }
                        if constexpr (kIsHalf) {
                            v[l] = __float22half2_rn(f2);
                        } else {
                            v[l] = __float22bfloat162_rn(f2);
                        }
                    }
                    auto coord =
                        shm_layout(make_coord(loc, j, Config::WP::WarpM(wid),
                                              Config::WP::WarpN(wid), wtid));
                    shm_buf->result[coord] = reinterpret_cast<const uint2 &>(v);
                }
            }
        }
        __syncthreads();

        unsigned pending_items = std::min(kResultMTilePerPartition,
                                          Config::kWarpTileM - m_tile_base) *
                                 kPartitionM * Config::kTile * kGroupN /
                                 kResultVecSize;

#pragma unroll
        for (unsigned i = 0, idx = tid;
             i < tal::CeilingDiv(kResultMTilePerPartition * kPartitionM *
                                     Config::kTile * kGroupN / kResultVecSize,
                                 kThreads);
             i++, idx += kThreads) {
            if (idx < pending_items) {
                unsigned row, col;
                Op::WriteBackIdxToCoord(idx, m_tile_base, &row, &col);

                uint4 v = reinterpret_cast<const uint4 *>(shm_buf->result)[idx];
                br.Store<BufferResource::kNone>(
                    (row * n / kResultVecSize + col) * sizeof(uint4), 0, v);
            }
        }

        if (q + 1 < kWriteLoop) {
            __syncthreads();
        }
    }
}

} // namespace causalflow::petit::rocm::quantization
