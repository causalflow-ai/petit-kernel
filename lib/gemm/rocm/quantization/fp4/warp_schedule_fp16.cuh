#pragma once

#include "causalflow/petit/tal/tensor/tensor.h"
#include "gemm/rocm/amd_fastmath.cuh"
#include "gemm/rocm/quantization/dequant.cuh"
#include "gemm/rocm/quantization/qgemm.cuh"

namespace causalflow::petit::rocm::quantization::fp4 {

template <class ElementA, bool kHighPrecision> struct MmaSelector;

template <bool kHighPrecision> struct MmaSelector<__half, kHighPrecision> {
    using DQ = Dequantizer<half2, kDataTypeFp4e2m1>;
    using DS = DequantizerForFp8Scale<half2, !kHighPrecision>;
    __device__ static inline float4 Mma(uint2 fa, uint2 fb, float4 c) {
        return mma_m16n16k16_fp16(fa, fb, c);
    }
};

template <bool kHighPrecision>
struct MmaSelector<__hip_bfloat16, kHighPrecision> {
    using DQ = Dequantizer<__hip_bfloat162, kDataTypeFp4e2m1>;
    using DS = DequantizerForFp8Scale<__hip_bfloat162, !kHighPrecision>;
    __device__ static inline float4 Mma(uint2 fa, uint2 fb, float4 c) {
        return mma_m16n16k16_bf16(fa, fb, c);
    }
};

template <class Config>
__device__ static void Matmul(const uint4 va[Config::kWarpTileM][4],
                              const uint *qw, unsigned short packed_scale,
                              typename Config::ThreadAccum &__restrict__ acc,
                              unsigned warp_idx_n) {
    using ArchMma =
        MmaSelector<typename Config::ElementA, Config::kHighPrecision>;
    using DQ = ArchMma::DQ;
    using DS = ArchMma::DS;
    using VectorType = DQ::VectorType;

    VectorType ds;
    DS::DequantFullScale(&ds, packed_scale);

    const auto bias = DQ::Bias(Config::kHighPrecision);
    const VectorType bias2{bias, bias};

    for (int m = 0; m < Config::kWarpTileM; m++) {
        // Compute C^T = B^T * A so that the actual accumulation results
        // are in row-major.
        for (int j = 0; j < 4; j++) {
            const uint2 *va_ptr = reinterpret_cast<const uint2 *>(&va[m][j]);
            uint q = qw[j];
            VectorType dq[4], old_dq[4];
            DQ::Dequant(dq, q);
            DQ::Dequant(dq + 2, q << 8);

            VectorType scale;
            scale.x = j < 2 ? ds.x : ds.y;
            scale.y = scale.x;

            for (int i = 0; i < 4; i++) {
                old_dq[i] = dq[i];
                if constexpr (Config::kHighPrecision) {
                    dq[i] = fastmath::hmul2(dq[i], bias2);
                }
                dq[i] = fastmath::hmul2(dq[i], scale);
            }
            static_assert(sizeof(dq) == sizeof(uint4), "");
            const uint2 *frag_b = reinterpret_cast<const uint2 *>(&dq[0]);

            for (int l = 0; l < 2; l++) {
                acc[m][warp_idx_n] =
                    ArchMma::Mma(frag_b[l], va_ptr[l], acc[m][warp_idx_n]);
            }
        }
    }
}

template <class Config>
__device__ static unsigned short
FetchScalesRegs(const typename ShmBuf<Config>::Layout &__restrict__ shm,
                unsigned stage, unsigned group_k, unsigned group_n,
                unsigned wtid) {
    using namespace causalflow::tal;
    static constexpr unsigned kLayoutN = Config::kLayoutN;
    static constexpr unsigned kGroupN = Config::kGroupN;

    using ScaleShape = Shape<_1, _1, C<kWarpSize>>;
    using ScaleStride =
        Stride<C<kGroupN / kLayoutN * kWarpSize>, C<kWarpSize>, _1>;
    using ScaleLayout = Layout<ScaleShape, ScaleStride>;

    ScaleLayout layout;
    auto coord = layout(make_coord(group_k, group_n, wtid));
    return reinterpret_cast<const unsigned short *>(
        shm.data[stage].scales)[coord];
}

// Each warp works on a kLayoutM x kLayoutN tile.
template <class Config> struct WarpPartitionMatmul {
    using LayoutA = MatrixALayout<Config>;
    using LayoutB = MatrixBLayout<Config>;
    using Partition = typename Config::WP;
    static constexpr unsigned kBatchA =
        kPackFactor * (sizeof(uint4) / sizeof(unsigned)) /
        sizeof(typename Config::ElementA) / sizeof(unsigned);

    __device__ static inline void
    Compute(typename ShmBuf<Config>::Layout *__restrict__ shm_buf,
            typename Config::ThreadAccum *__restrict__ acc, unsigned stage,
            unsigned wid, unsigned wtid) {
        const unsigned warp_row_k = Partition::WarpK(wid),
                       warp_col_n = Partition::WarpN(wid),
                       warp_row_m = Partition::WarpM(wid);

        uint4 qw;
        unsigned short packed_scales;
        uint4 va[Config::kWarpTileM][kBatchA];

        static_assert(
            Config::kGroupK / Config::kLayoutM / Partition::kPartitionK > 0,
            "");
        // Each warp works on a kLayoutM x kLayoutN tile.
        static constexpr unsigned kWarpTileK =
            Config::kGroupK / Config::kLayoutM / Partition::kPartitionK;
        static constexpr unsigned kWarpTileN =
            Config::kGroupN / Config::kLayoutN / Partition::kPartitionN;
        for (int i = 0; i < kWarpTileK; i++) {
            for (int j = 0; j < kWarpTileN; j++) {
                packed_scales = FetchScalesRegs<Config>(
                    *shm_buf, stage, warp_row_k * kWarpTileK + i,
                    warp_col_n * kWarpTileN + j, wtid);
                LayoutA::FetchRegisters(va, shm_buf->data[stage].a,
                                        warp_row_m * Config::kWarpTileM,
                                        warp_row_k * kWarpTileK + i, wtid);
                LayoutB::FetchRegisters(&qw, shm_buf->data[stage].b,
                                        warp_col_n * kWarpTileN + j,
                                        warp_row_k * kWarpTileK + i, wtid);
                Matmul<Config>(va, reinterpret_cast<const uint *>(&qw),
                               packed_scales, *acc, j);
            }
        }
    }
};

} // namespace causalflow::petit::rocm::quantization::fp4
