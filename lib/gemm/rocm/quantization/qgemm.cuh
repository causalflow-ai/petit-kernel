#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "memory_ops.cuh"
#include <type_traits>

#include <hip/hip_bf16.h>

namespace causalflow::petit::rocm::quantization {

template <class Config, bool kHasAlpha>
__device__ static void
WriteResult(float alpha, typename ShmBuf<Config>::Layout *__restrict__ shm_buf,
            typename Config::ThreadAccum &__restrict__ acc,
            uint4 *__restrict__ c_ptr, unsigned tid, unsigned m, unsigned n) {
    using namespace tal;
    static constexpr bool kIsHalf =
        std::is_same<typename Config::ElementA, half>::value;
    using VectorType = std::conditional_t<kIsHalf, half2, __hip_bfloat162>;
    static constexpr unsigned kGroupM = Config::kGroupM;
    static constexpr unsigned kGroupN = Config::kGroupN;
    static constexpr unsigned kThreads = Config::kThreads;
    static constexpr unsigned kResultVecSize = sizeof(uint4) / sizeof(half);
    static constexpr unsigned kPartitionM = Config::WP::kPartitionM;
    static constexpr unsigned kPartitionN = Config::WP::kPartitionN;
    static constexpr unsigned kShmVecSize = sizeof(uint2) / sizeof(half);

    const unsigned wid = tid / kWarpSize, wtid = tid % kWarpSize,
                   acc_row = Config::WP::WarpK(wid);

    using ShmShape =
        Shape<C<Config::kWarpTileM>, C<Config::kThreadAccumTileNRegs>,
              C<kPartitionM>, C<kPartitionN>, Shape<_16, _4>>;
    using ShmStride = Stride<C<Config::kTile * kGroupN / kShmVecSize>, _4,
                             C<kGroupM / kPartitionM * kGroupN / kShmVecSize>,
                             C<kGroupN / kPartitionN / kShmVecSize>,
                             Stride<C<kGroupN / kShmVecSize>, _1>>;
    using ShmLayout = Layout<ShmShape, ShmStride>;
    ShmLayout shm_layout;

    BufferResource br = {
        .v =
            {
                .ptr = reinterpret_cast<uintptr_t>(c_ptr),
                .range = ((m - 1) * n + kGroupN) *
                         static_cast<unsigned>(sizeof(half)),
                .config = BufferResource::kDataFormatU32Config,
            },
    };

    if (acc_row == 0) {
#pragma unroll
        for (int i = 0; i < Config::kWarpTileM; i++) {
#pragma unroll
            for (int j = 0; j < Config::kThreadAccumTileNRegs; j++) {
                VectorType v[2];
                for (int l = 0; l < 2; l++) {
                    float2 f2 = reinterpret_cast<const float2 *>(&acc[i][j])[l];
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
                    shm_layout(make_coord(i, j, Config::WP::WarpM(wid),
                                          Config::WP::WarpN(wid), wtid));
                shm_buf->result[coord] = reinterpret_cast<const uint2 &>(v);
            }
        }
    }
    __syncthreads();

    for (unsigned i = 0, idx = tid;
         i < tal::CeilingDiv(kGroupM * kGroupN / kResultVecSize, kThreads) &&
         idx < kGroupM * kGroupN / kResultVecSize;
         i++, idx += kThreads) {
        unsigned row = idx / (kGroupN / kResultVecSize),
                 col = idx % (kGroupN / kResultVecSize);
        uint4 v = reinterpret_cast<const uint4 *>(shm_buf->result)[idx];
        br.Store((row * n / kResultVecSize + col) * sizeof(uint4), 0,
                 BufferResource::kNone, v);
    }
}

} // namespace causalflow::petit::rocm::quantization
