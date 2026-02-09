#pragma once

#include "causalflow/petit/tal/tensor/tensor.h"
#include "gemm/rocm/amd_fastmath.cuh"
#include "gemm/rocm/quantization/dequant.cuh"
#include "gemm/rocm/quantization/qgemm.cuh"

namespace causalflow::petit::rocm::quantization::fp4 {

template <class ElementA, bool kHighPrecision> struct MmaSelector;

template <bool kHighPrecision> struct MmaSelector<__half, kHighPrecision> {
    using UDQ = UnifiedDequantizerForFp4Fp16<kHighPrecision>;

    __device__ static inline float4 Mma(uint2 fa, uint2 fb, float4 c) {
        return mma_m16n16k16_fp16(fa, fb, c);
    }
};

template <bool kHighPrecision>
struct MmaSelector<__hip_bfloat16, kHighPrecision> {
    using UDQ = UnifiedDequantizerForNvFp4Bf16<kHighPrecision>;

    __device__ static inline float4 Mma(uint2 fa, uint2 fb, float4 c) {
        return mma_m16n16k16_bf16(fa, fb, c);
    }
};

template <class Config>
__device__ static unsigned short
FetchScalesRegs(const typename ShmBuf<Config>::Data &__restrict__ shm,
                unsigned group_k, unsigned group_n, unsigned wtid) {
    using namespace causalflow::tal;
    static constexpr unsigned kLayoutN = Config::kLayoutN;
    static constexpr unsigned kGroupN = Config::kGroupN;

    using ScaleShape = Shape<_1, _1, C<kWarpSize>>;
    using ScaleStride =
        Stride<C<kGroupN / kLayoutN * kWarpSize>, C<kWarpSize>, _1>;
    using ScaleLayout = Layout<ScaleShape, ScaleStride>;

    ScaleLayout layout;
    auto coord = layout(make_coord(group_k, group_n, wtid));
    return reinterpret_cast<const unsigned short *>(shm.scales)[coord];
}

// Each warp works on a kLayoutM x kLayoutN tile.
template <class Config> struct WarpPartitionMatmul {
    using LayoutA = MatrixALayout<Config>;
    using LayoutB = MatrixBLayout<Config>;
    using Partition = typename Config::WP;
    static constexpr unsigned kBatchA = Config::WarpMatmulLayout::kReadBatchA;

    static_assert(Config::kGroupK / Config::kLayoutM / Partition::kPartitionK >
                      0,
                  "");
    // Each warp works on a kLayoutM x kLayoutN tile.
    static constexpr unsigned kWarpAtomK =
        Config::kGroupK / Config::kLayoutM / Partition::kPartitionK;
    static constexpr unsigned kWarpAtomN =
        Config::kGroupN / Config::kLayoutN / Partition::kPartitionN;
    static_assert(kWarpAtomK > 0, "");
    static_assert(kWarpAtomN > 0, "");

    using DataA = uint4[Config::kWarpTileM][kBatchA];
    struct DataB {
        uint4 qw;
        unsigned short packed_scales;
    };

    __device__ inline explicit WarpPartitionMatmul(
        typename ShmBuf<Config>::Layout *__restrict__ shm_buf, unsigned stage,
        unsigned wid, unsigned wtid)
        : data_(&shm_buf->data[stage]), warp_row_m_(Partition::WarpM(wid)),
          warp_col_n_(Partition::WarpN(wid)),
          warp_row_k_(Partition::WarpK(wid)), wtid_(wtid) {}

    __device__ inline void ReadShmA(DataA &va, unsigned tile_idx_k) {
        LayoutA::FetchRegisters(va, data_->a, warp_row_m_ * Config::kWarpTileM,
                                warp_row_k_ * kWarpAtomK + tile_idx_k, wtid_);
    }

    __device__ inline void ReadShmB(DataB &b, unsigned tile_idx_k,
                                    unsigned tile_idx_n) {
        b.packed_scales = FetchScalesRegs<Config>(
            *data_, warp_row_k_ * kWarpAtomK + tile_idx_k,
            warp_col_n_ * kWarpAtomN + tile_idx_n, wtid_);
        LayoutB::FetchRegisters(&b.qw, data_->b,
                                warp_col_n_ * kWarpAtomN + tile_idx_n,
                                warp_row_k_ * kWarpAtomK + tile_idx_k, wtid_);
    }

    __device__ void Prefetch(DataA *a, DataB *b) {
        ReadShmA(*a, 0);
        ReadShmB(*b, 0, 0);
    }

    __device__ void
    PipelineCompute(DataA *a, DataB *b,
                    typename Config::ThreadAccum *__restrict__ acc) {
        for (int i = 0; i < kWarpAtomK; i++) {
            for (int j = 0; j < kWarpAtomN; j++) {
                Matmul(*a, *b, *acc, j);
                if (j + 1 < kWarpAtomN) {
                    ReadShmB(*b, i, j + 1);
                }
            }
            if (i + 1 < kWarpAtomK) {
                ReadShmA(*a, i + 1);
                ReadShmB(*b, i + 1, 0);
            }
        }
    }

    __device__ inline void
    Compute(typename Config::ThreadAccum *__restrict__ acc) {
        DataA va;
        DataB b;

        for (int i = 0; i < kWarpAtomK; i++) {
            ReadShmA(va, i);
            for (int j = 0; j < kWarpAtomN; j++) {
                ReadShmB(b, i, j);
                Matmul(va, b, *acc, j);
            }
        }
    }

    __device__ void Matmul(const DataA &va, const DataB &b,
                           typename Config::ThreadAccum &__restrict__ acc,
                           unsigned warp_idx_n) {
        using causalflow::tal::make_coord;
        using ArchMma =
            MmaSelector<typename Config::ElementA, Config::kHighPrecision>;
        using UDQ = ArchMma::UDQ;
        typename Config::WarpAccumLayout accum_layout;
        typename Config::WarpMatmulRegALayout reg_a_layout;

        auto ds = UDQ::DequantScales(b.packed_scales);
        const uint *qw = reinterpret_cast<const uint *>(&b.qw);

        // Compute C^T = B^T * A so that the actual accumulation results
        // are in row-major.
        for (int j = 0; j < 4; j++) {
            auto acc_idx = accum_layout(make_coord(warp_idx_n, j));
            auto va_idx = reg_a_layout(make_coord(j));
            uint q = qw[j];
            typename UDQ::UnpackedData dq;
            UDQ::DequantWithScale(dq, q, j < 2 ? ds.x : ds.y);

            static_assert(sizeof(dq) == sizeof(uint4), "");
            const uint2 *frag_b = reinterpret_cast<const uint2 *>(&dq[0]);

            for (int m = 0; m < Config::kWarpTileM; m++) {
                const uint2 *va_ptr =
                    reinterpret_cast<const uint2 *>(&va[m][va_idx]);
                for (int l = 0; l < 2; l++) {
                    acc[m][acc_idx] =
                        ArchMma::Mma(frag_b[l], va_ptr[l], acc[m][acc_idx]);
                }
            }
        }
    }

    typename ShmBuf<Config>::Data *data_;
    const unsigned warp_row_m_, warp_col_n_, warp_row_k_, wtid_;
};

} // namespace causalflow::petit::rocm::quantization::fp4
