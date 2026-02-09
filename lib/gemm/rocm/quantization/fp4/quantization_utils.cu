#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/amd_intrinsics.cuh"
#include "gemm/rocm/quantization/dequant.cuh"
#include "gemm/rocm/quantization/gemm.h"
#include "gemm/rocm/quantization/types.h"
#include "gemm_fp4.h"

#include <hip/hip_fp8.h>
#include <type_traits>

namespace causalflow::petit::rocm::quantization::fp4 {

static constexpr unsigned kBits = 4;
static constexpr unsigned kPackFactor = 32 / kBits;
static constexpr unsigned kQuantVecSize = sizeof(uint4) / sizeof(unsigned);
static constexpr unsigned kRowGroupSize = 16;
static constexpr unsigned kMxRowGroupSize = 32;

template <unsigned kLayoutM_, unsigned kLayoutN_, unsigned kTileM_,
          unsigned kTileN_, unsigned kPackSize_, unsigned kOutputVecBatch_,
          unsigned kBlockGroupM_, unsigned kBlockGroupN_>
struct TileShmLayout {
    static constexpr unsigned kLayoutM = kLayoutM_;
    static constexpr unsigned kLayoutN = kLayoutN_;
    static constexpr unsigned kGroupM = kBlockGroupM_ * kLayoutM_;
    static constexpr unsigned kGroupN = kBlockGroupN_ * kLayoutN_;

    __device__ auto GetShmLayout() const {
        using namespace causalflow::tal;
        using ShmShape =
            Shape<Shape<C<kTileM_>, C<kTileN_>>,
                  Shape<C<kBlockGroupM_>, C<kBlockGroupN_>>, Shape<_16, _4>>;
        using ShmStride = Stride<Stride<_1, C<16 * kGroupM / kPackSize_>>,
                                 Stride<C<kLayoutM / kPackSize_>,
                                        C<kGroupM * kLayoutN / kPackSize_>>,
                                 Stride<C<kGroupM / kPackSize_>, C<kTileM_>>>;
        using ShmLayout = Layout<ShmShape, ShmStride>;
        return ShmLayout{};
    }

    __device__ auto GetOnDiskLayout() const {
        using namespace causalflow::tal;

        using OutputShape = Shape<_1, _1, C<kWarpSize>,
                                  Shape<C<kBlockGroupM_>, C<kBlockGroupN_>>>;
        auto output_stride = make_stride(
            n_ * kGroupM / kPackSize_ / kOutputVecBatch_,
            C<kLayoutM * kGroupN / kPackSize_ / kOutputVecBatch_>{}, _1{},
            make_stride(n_ * kLayoutM / kPackSize_ / kOutputVecBatch_,
                        C<kWarpSize>{}));
        auto output_layout = make_layout(OutputShape{}, output_stride);
        return output_layout;
    }

    unsigned n_;
};

template <unsigned kLayoutM_, unsigned kLayoutN_, unsigned kTileM_,
          unsigned kTileN_, unsigned kBlockGroupM_, unsigned kBlockGroupN_>
struct RepackQWeightLayout
    : public TileShmLayout<kLayoutM_, kLayoutN_, kTileM_, kTileN_, kPackFactor,
                           kQuantVecSize, kBlockGroupM_, kBlockGroupN_> {
    using Base =
        TileShmLayout<kLayoutM_, kLayoutN_, kTileM_, kTileN_, kPackFactor,
                      kQuantVecSize, kBlockGroupM_, kBlockGroupN_>;
    static constexpr unsigned kBlockGroupM = kBlockGroupM_;
    static constexpr unsigned kBlockGroupN = kBlockGroupN_;
    static constexpr unsigned kNumWarps = kBlockGroupM_ * kBlockGroupN_;
    static constexpr unsigned kDequantOutputBatch =
        kPackFactor * (sizeof(uint4) / sizeof(uint)) * sizeof(half) /
        sizeof(uint4);
    static constexpr unsigned kLayoutM = Base::kLayoutM;
    static constexpr unsigned kLayoutN = Base::kLayoutN;
    static constexpr unsigned kGroupM = Base::kGroupM;
    static constexpr unsigned kGroupN = Base::kGroupN;
    static constexpr unsigned kTileM = kTileM_;
    static constexpr unsigned kTileN = kTileN_;

    // Define how the 4 uints are packed across the (m, n) order
    static_assert(kTileM_ * kTileN_ == 4, "The weight tile must be 4");

    explicit __device__ RepackQWeightLayout(unsigned n, unsigned k)
        : Base(n), k_(k) {}

    unsigned k_;
};

template <unsigned kLayoutM_, unsigned kLayoutN_, unsigned kPackSize_,
          unsigned kBlockGroupM_, unsigned kBlockGroupN_>
struct RepackScaleLayoutBase {
    static constexpr unsigned kLayoutM = kLayoutM_;
    static constexpr unsigned kLayoutN = kLayoutN_;
    static constexpr unsigned kPackSize = kPackSize_;
    static constexpr unsigned kSubWarpSize =
        kLayoutM * kLayoutN / kPackSize_ / sizeof(unsigned);
    static constexpr unsigned kOutputVecBatch = sizeof(unsigned) / sizeof(char);
    static constexpr unsigned kGroupM = kBlockGroupM_ * kLayoutM;
    static constexpr unsigned kGroupN = kBlockGroupN_ * kLayoutN;
    static constexpr unsigned kNumWarps = 1;

    // The scale is organized as [s[x][k], s[x][k+1], s[x+1][k], s[x+1][k+1]]
    __device__ auto GetShmLayout() const {
        using namespace causalflow::tal;
        using ShmShape =
            Shape<Shape<C<2>, C<2>>,
                  Shape<Shape<_8, C<kSubWarpSize / 8>>,
                        Shape<C<kBlockGroupM_>, C<kBlockGroupN_>>>>;
        using ShmStride = Stride<
            Stride<C<16 * kGroupM / kPackSize_>, C<kGroupM / kPackSize_>>,
            Stride<Stride<C<2 * kGroupM / kPackSize_>, _1>,
                   Stride<C<kLayoutM / kPackSize_>,
                          C<kGroupM * kLayoutN / kPackSize_>>>>;
        using ShmLayout = Layout<ShmShape, ShmStride>;
        return ShmLayout{};
    }

    __device__ auto GetOnDiskLayout() const {
        using namespace causalflow::tal;

        using OutputShape = Shape<
            _1, _1,
            Shape<C<kSubWarpSize>, Shape<C<kBlockGroupM_>, C<kBlockGroupN_>>>>;

        auto output_stride = make_stride(
            n_ * kGroupM / kPackSize_ / kOutputVecBatch,
            C<kLayoutM * kGroupN / kPackSize_ / kOutputVecBatch>{},
            make_stride(
                _1{}, make_stride(n_ * kLayoutM / kPackSize_ / kOutputVecBatch,
                                  C<kSubWarpSize>{})));
        auto output_layout = make_layout(OutputShape{}, output_stride);
        return output_layout;
    }
    unsigned n_;
};

template <unsigned kExpBias_, unsigned kLayoutM, unsigned kLayoutN,
          unsigned kBlockGroupM_, unsigned kBlockGroupN_>
struct RepackScaleLayout
    : public RepackScaleLayoutBase<kLayoutM, kLayoutN, kRowGroupSize,
                                   kBlockGroupM_, kBlockGroupN_> {

    __device__ static unsigned Transform(const unsigned char (&v)[4]) {
        if constexpr (kExpBias_ == 0) {
            return *reinterpret_cast<const unsigned *>(v);
        }

        // Convert the half2 scale to the E5M3 format.
        unsigned r = 0;
        for (int i = 0; i < 2; i++) {
            half2 h2;
            const half kMultiple = half(1 << kExpBias_);
            h2.x = __hip_cvt_fp8_to_halfraw(v[i * 2], __HIP_E4M3);
            h2.y = __hip_cvt_fp8_to_halfraw(v[i * 2 + 1], __HIP_E4M3);
            auto scaled = __hmul2(h2, half2{kMultiple, kMultiple});
            unsigned scaled_u32 = reinterpret_cast<const unsigned &>(scaled);
            unsigned s =
                ((scaled_u32 & 0xffff) >> 7) | ((scaled_u32 >> 23) << 8);
            r |= s << (i * 16);
        }
        return r;
    }
};

// Same shared-memory layout pattern as RepackScaleLayout.
template <unsigned kLayoutM, unsigned kLayoutN, unsigned kBlockGroupM_,
          unsigned kBlockGroupN_>
struct RepackMxScaleLayout
    : public RepackScaleLayoutBase<kLayoutM, kLayoutN, kMxRowGroupSize,
                                   kBlockGroupM_, kBlockGroupN_> {

    __device__ static unsigned Transform(const unsigned char (&v)[4]) {
        return *reinterpret_cast<const unsigned *>(v);
    }
};

using RepackQWeightLayout128x16 = RepackQWeightLayout<128, 16, 4, 1, 2, 2>;
using RepackQWeightLayout64x32 = RepackQWeightLayout<64, 32, 2, 2, 4, 1>;

using RepackScaleLayout64x32 = RepackScaleLayout<kFp8ScaleBias, 64, 32, 1, 2>;
using RepackMxScaleLayout64x32 = RepackMxScaleLayout<64, 32, 4, 1>;

__device__ static unsigned PetitFormat(unsigned v) {
    unsigned r = 0;
    for (int i = 0; i < 8; i++) {
        unsigned off_s = 15 - (i % 4 / 2) * 8 + (i % 2) * 16;
        unsigned off_d = off_s - 6;
        if (i >= 4) {
            off_s = 31 - off_s;
            off_d = off_s + 4;
        }
        unsigned u = (v >> (i * 4)) & 0xf;
        unsigned val = u & 0x7;

        // Change negative zero to positive zero as in MI300x natively
        // supports e5m2fnuz where 0x80 will be incorrectly dequantized to NaN
        // in Petit.
        unsigned sgn = val == 0 ? 0 : u >> 3;

        if (i >= 4) {
            val = __builtin_bitreverse32(val) >> 29;
        }
        r |= (sgn << off_s) | (val << off_d);
    }
    return r;
}

template <class QWLayout, class ProcessWeightOp>
__global__ void RepackNvFp4ToPetitFp4WeightsKernel(
    ProcessWeightOp process, uint4 *__restrict__ output,
    const uint4 *__restrict__ input, unsigned in_chan, unsigned out_chan) {
    using namespace causalflow::tal;
    static constexpr unsigned kGroupM = QWLayout::kGroupM;
    static constexpr unsigned kGroupN = QWLayout::kGroupN;
    static constexpr unsigned kGroupInts = kGroupM * kGroupN / kPackFactor;
    static constexpr unsigned kThreads = QWLayout::kNumWarps * kWarpSize;

    QWLayout layout(out_chan, in_chan);

    const unsigned tid = threadIdx.x, id_m = blockIdx.x, id_n = blockIdx.y;
    const unsigned wid = tid / kWarpSize, wtid = tid % kWarpSize;
    const uint4 *in_ptr =
        input + id_m * kGroupM / kPackFactor / kQuantVecSize +
        id_n * in_chan * kGroupN / kPackFactor / kQuantVecSize;

    __shared__ uint4 shm_qw[kGroupInts / kQuantVecSize];

    [[assume(tid < kThreads)]];
    for (unsigned i = 0, idx = tid;
         i < CeilingDiv(kGroupInts / kQuantVecSize, kThreads) &&
         idx < kGroupInts / kQuantVecSize;
         i++, idx += kThreads) {
        unsigned row = idx / (kGroupM / kPackFactor / kQuantVecSize),
                 col = idx % (kGroupM / kPackFactor / kQuantVecSize);
        shm_qw[idx] = in_ptr[row * in_chan / kPackFactor / kQuantVecSize + col];
    }
    __syncthreads();

    auto shm_layout = layout.GetShmLayout();
    auto output_layout = layout.GetOnDiskLayout();

    const unsigned *sqw = reinterpret_cast<const unsigned *>(shm_qw);

    unsigned ret[4];
    for (int i = 0; i < 4; i++) {
        auto coord = shm_layout(make_coord(i, wid, wtid));
        unsigned qv = sqw[coord];
        ret[i] = process(qv);
    }

    auto output_coord = output_layout(make_coord(id_m, id_n, wtid, wid));
    output[output_coord] = *reinterpret_cast<const uint4 *>(ret);
}

template <class ScaleLayout>
__global__ void RepackFp4ScalesKernel(uint4 *__restrict__ out_scales,
                                      const uint4 *__restrict__ scales,
                                      unsigned in_chan, unsigned out_chan) {
    using namespace causalflow::tal;

    static constexpr unsigned kGroupM = ScaleLayout::kGroupM;
    static constexpr unsigned kGroupN = ScaleLayout::kGroupN;
    static constexpr unsigned kScaleRowGroupSize = ScaleLayout::kPackSize;
    static constexpr unsigned kInVecSize = sizeof(unsigned);
    static constexpr unsigned kThreads = ScaleLayout::kNumWarps * kWarpSize;
    static constexpr unsigned kTotalScaleVec =
        kGroupM * kGroupN / kScaleRowGroupSize / kInVecSize;

    const unsigned tid = threadIdx.x, id_m = blockIdx.x, id_n = blockIdx.y;

    const auto *in_scales =
        reinterpret_cast<const unsigned *>(scales) +
        id_m * kGroupM / kScaleRowGroupSize / kInVecSize +
        id_n * in_chan * kGroupN / kScaleRowGroupSize / kInVecSize;

    __shared__ unsigned shm_scales_u32[kTotalScaleVec];
    auto *shm_scales = reinterpret_cast<unsigned char *>(shm_scales_u32);

    for (unsigned i = 0, idx = tid;
         i < CeilingDiv(kTotalScaleVec, kThreads) && idx < kTotalScaleVec;
         i++, idx += kThreads) {
        static constexpr unsigned kScalesPerCol = kGroupM / kScaleRowGroupSize;
        const unsigned row = idx / (kScalesPerCol / kInVecSize);
        const unsigned col = idx % (kScalesPerCol / kInVecSize);
        shm_scales_u32[idx] =
            in_scales[row * in_chan / kScaleRowGroupSize / kInVecSize + col];
    }

    __syncthreads();

    ScaleLayout scale_layout{out_chan};
    auto shm_layout = scale_layout.GetShmLayout();
    auto output_layout = scale_layout.GetOnDiskLayout();

    auto out = reinterpret_cast<unsigned *>(out_scales);
    for (unsigned idx = tid; idx < kTotalScaleVec; idx += kThreads) {
        unsigned char s[4];
        for (int i = 0; i < 4; i++) {
            s[i] = shm_scales[shm_layout(make_coord(i, idx))];
        }
        const unsigned out_idx = output_layout(make_coord(id_m, id_n, idx));
        out[out_idx] = ScaleLayout::Transform(s);
    }
}

template <unsigned kBlockGroupK_, unsigned kBlockGroupN_,
          unsigned kRowGroupSize_>
struct DequantTraitNativeMixIn {
    static constexpr unsigned kLayoutM = 64;
    static constexpr unsigned kLayoutN = 32;
    static constexpr unsigned kBlockGroupK = kBlockGroupK_;
    static constexpr unsigned kBlockGroupN = kBlockGroupN_;
    static constexpr unsigned kNumWarps = kBlockGroupK_ * kBlockGroupN_;
    static constexpr unsigned kGroupK = kBlockGroupK * kLayoutM;
    static constexpr unsigned kGroupN = kBlockGroupN * kLayoutN;
    static constexpr unsigned kRowGroupSize = kRowGroupSize_;
    static constexpr unsigned kScaleVecSize = sizeof(unsigned);
    static constexpr unsigned kThreadPerRow =
        kGroupK / kQuantVecSize / kPackFactor;

    __device__ static auto GetFetchQWLayout(unsigned size_k, unsigned size_n) {
        using namespace causalflow::tal;
        auto stride_in = make_stride(
            C<kGroupK / kPackFactor / kQuantVecSize>{},
            size_k * kGroupN / kPackFactor / kQuantVecSize,
            make_stride(
                make_stride(_1{}, size_k / kPackFactor / kQuantVecSize)),
            size_k / kPackFactor / kQuantVecSize * (kWarpSize / kThreadPerRow));
        auto layout_in = make_layout(
            Shape<_1, _1, Shape<Shape<C<kThreadPerRow>, _16>>, _1>{},
            stride_in);
        return layout_in;
    }

    __device__ static auto GetFetchScaleLayout(unsigned size_k,
                                               unsigned size_n) {
        using namespace causalflow::tal;
        using ShmShape = Shape<
            _1, _1,
            Shape<C<kGroupK / kRowGroupSize / kScaleVecSize>, C<kGroupN>>>;
        auto stride = make_stride(
            kGroupN * size_k / kRowGroupSize / kScaleVecSize,
            C<kGroupK / kRowGroupSize / kScaleVecSize>{},
            make_stride(_1{}, size_k / kRowGroupSize / kScaleVecSize));
        return make_layout(ShmShape{}, stride);
    }

    __device__ static auto GetOutputLayout(unsigned size_k, unsigned size_n) {
        using namespace causalflow::tal;
        static constexpr unsigned kOutU128 = (sizeof(uint4) / sizeof(uint)) *
                                             kPackFactor * sizeof(half) /
                                             sizeof(uint4);
        static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(half);
        auto stride_out =
            make_stride(C<kGroupK / kVecSize>{}, size_k * kGroupN / kVecSize,
                        make_stride(C<kOutU128>{}, size_k / kVecSize), _1{});
        auto layout_out = make_layout(
            Shape<_1, _1, Shape<C<kThreadPerRow>, _1>, C<kOutU128>>{},
            stride_out);
        return layout_out;
    }
};

template <DataType kOutputType>
struct DequantTraitNvFp4 : public DequantTraitNativeMixIn<2, 1, 16> {
    using Base = DequantTraitNativeMixIn<2, 1, 16>;

    static constexpr unsigned kBlockGroupK = Base::kBlockGroupK;
    static constexpr unsigned kBlockGroupN = Base::kBlockGroupN;
    static constexpr unsigned kNumWarps = Base::kNumWarps;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr bool kIsNativeQWFormat = true;

    using UDQ = std::conditional_t<kOutputType == DataType::kDataTypeFp16,
                                   UnifiedDequantizerForFp4Fp16<true>,
                                   UnifiedDequantizerForNvFp4Bf16<true>>;
    using Scale = half;

    __device__ static uint2 GetScale(unsigned *shm_scales, unsigned tid) {
        using namespace causalflow::tal;
        using ShmScaleLayout =
            Layout<Shape<Shape<C<Base::kThreadPerRow>,
                               C<kThreads / Base::kThreadPerRow>>>,
                   Stride<Stride<_1, C<Base::kGroupK / Base::kRowGroupSize /
                                       sizeof(unsigned short)>>>>;
        ShmScaleLayout shm_scale_layout;
        auto packed_scale = reinterpret_cast<const unsigned short *>(
            shm_scales)[shm_scale_layout(make_coord(tid))];

        uint2 v;
        half2 *u = reinterpret_cast<half2 *>(&v);
        half2 ds;
        ds.x = __hip_cvt_fp8_to_halfraw(packed_scale & 0xff, __HIP_E4M3);
        ds.y = __hip_cvt_fp8_to_halfraw((packed_scale >> 8), __HIP_E4M3);
        ds = __hmul2(ds, half2{1 << kFp8ScaleBias, 1 << kFp8ScaleBias});

        u[0].x = ds.x;
        u[0].y = ds.x;
        u[1].x = ds.y;
        u[1].y = ds.y;
        return v;
    }
};

struct DequantTraitMxFp4 : public DequantTraitNativeMixIn<2, 1, 32> {
    using Base = DequantTraitNativeMixIn<2, 1, 32>;

    static constexpr unsigned kBlockGroupK = Base::kBlockGroupK;
    static constexpr unsigned kBlockGroupN = Base::kBlockGroupN;
    static constexpr unsigned kNumWarps = Base::kNumWarps;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr bool kIsNativeQWFormat = true;

    using UDQ = UnifiedDequantizerForMxFp4Bf16<true>;
    using Scale = __hip_bfloat16;

    __device__ static uint2 GetScale(unsigned *shm_scales, unsigned tid) {
        using namespace causalflow::tal;
        using ShmScaleLayout =
            Layout<Shape<Shape<C<Base::kThreadPerRow>,
                               C<kThreads / Base::kThreadPerRow>>>,
                   Stride<Stride<_1, C<Base::kGroupK / Base::kRowGroupSize>>>>;
        ShmScaleLayout shm_scale_layout;
        auto s = reinterpret_cast<const unsigned char *>(
            shm_scales)[shm_scale_layout(make_coord(tid))];
        uint2 v;
        auto u = reinterpret_cast<__hip_bfloat162 *>(&v);
        u[0] = UDQ::DequantScales(s | (s << 8));
        u[1] = u[0];
        return v;
    }
};

template <class QWLayout, unsigned kRowGroupSize_>
struct DequantTraitPetitMixIn {
    static constexpr unsigned kRowGroupSize = kRowGroupSize_;

    static constexpr unsigned kBlockGroupK = QWLayout::kBlockGroupM;
    static constexpr unsigned kBlockGroupN = QWLayout::kBlockGroupN;
    static constexpr unsigned kLayoutM = QWLayout::kLayoutM;
    static constexpr unsigned kLayoutN = QWLayout::kLayoutN;
    static constexpr unsigned kGroupK = kLayoutM * kBlockGroupK;
    static constexpr unsigned kGroupN = kLayoutN * kBlockGroupN;
    static constexpr unsigned kNumWarps = kBlockGroupK * kBlockGroupN;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kThreadPerRow =
        kGroupK / kQuantVecSize / kPackFactor;
    static constexpr bool kIsNativeQWFormat = false;

    __device__ static auto GetFetchQWLayout(unsigned size_k, unsigned size_n) {
        QWLayout layout(size_n, size_k);
        auto layout_in = layout.GetOnDiskLayout();
        return layout_in;
    }

    __device__ static auto GetFetchScaleLayout(unsigned size_k,
                                               unsigned size_n) {
        static constexpr unsigned kScaleVecSize = sizeof(unsigned);
        using namespace causalflow::tal;
        using ShmShape =
            Shape<_1, _1,
                  Shape<C<kLayoutM * kLayoutN / kRowGroupSize / kScaleVecSize>,
                        C<kNumWarps>>>;
        auto stride =
            make_stride(C<kLayoutM * kGroupN / kRowGroupSize / kScaleVecSize>{},
                        size_n * kGroupK / kRowGroupSize / kScaleVecSize,
                        make_stride(_1{}, size_n * kLayoutM / kRowGroupSize /
                                              kScaleVecSize));
        return make_layout(ShmShape{}, stride);
    }

    __device__ static auto GetOutputLayout(unsigned size_k, unsigned size_n) {
        using namespace causalflow::tal;
        static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(half);
        static constexpr unsigned kTileM_ = QWLayout::kTileM;
        static constexpr unsigned kTileN_ = QWLayout::kTileN;

        using OutputShape = Shape<
            _1, _1,
            Shape<Shape<_16, _4>, Shape<C<kBlockGroupK>, C<kBlockGroupN>>>,
            Shape<C<kTileM_>, C<kTileN_>>>;
        auto stride_out = make_stride(
            C<kGroupK / kVecSize>{}, size_k * kGroupN / kVecSize,
            make_stride(make_stride(size_k / kVecSize, C<kTileM_>{}),
                        make_stride(C<kLayoutM / kVecSize>{},
                                    size_k * kLayoutN / kVecSize)),
            make_stride(_1{}, size_k * 16 / kVecSize));
        auto layout_out = make_layout(OutputShape{}, stride_out);
        return layout_out;
    }
};

template <class QWLayout, DataType kOutputType>
struct DequantTraitPetitNvFp4 : public DequantTraitPetitMixIn<QWLayout, 16> {
    using UDQ = std::conditional_t<kOutputType == DataType::kDataTypeFp16,
                                   UnifiedDequantizerForFp4Fp16<true>,
                                   UnifiedDequantizerForNvFp4Bf16<true>>;
    using Scale = half;

    __device__ static uint2 GetScale(unsigned *shm_scales, unsigned tid) {
        auto packed_scale =
            reinterpret_cast<const unsigned short *>(shm_scales)[tid];

        uint2 v;
        half2 *u = reinterpret_cast<half2 *>(&v);
        half2 ds = UDQ::DequantScales(packed_scale);
        u[0].x = ds.x;
        u[0].y = ds.x;
        u[1].x = ds.y;
        u[1].y = ds.y;
        return v;
    }
};

template <class QWLayout>
struct DequantTraitPetitMxFp4 : public DequantTraitPetitMixIn<QWLayout, 32> {

    using UDQ = UnifiedDequantizerForMxFp4Bf16<true>;
    using Scale = __hip_bfloat16;

    __device__ static uint2 GetScale(unsigned *shm_scales, unsigned tid) {
        static_assert(QWLayout::kLayoutM == 64 && QWLayout::kLayoutN == 32,
                      "DequantTraitPetitMxFp4 only supports 64x32 layout");
        const unsigned wid = tid / kWarpSize;
        const unsigned wtid = tid % kWarpSize;
        const unsigned t =
            wid * (kWarpSize / 2) + (wtid / (kWarpSize / 2)) * 16 + (wtid % 16);
        auto packed_scale =
            reinterpret_cast<const unsigned short *>(shm_scales)[t];

        uint2 v;
        auto *u = reinterpret_cast<__hip_bfloat162 *>(&v);
        auto ds = UDQ::DequantScales(packed_scale);
        u[0].x = ds.x;
        u[0].y = ds.x;
        u[1].x = ds.y;
        u[1].y = ds.y;
        return v;
    }
};

template <class Trait>
__global__ void DequantizeFp4Kernel(uint4 *output, const uint4 *input,
                                    const unsigned char *scales,
                                    float global_scale, unsigned size_k,
                                    unsigned size_n) {
    using namespace causalflow::tal;

    using UDQ = typename Trait::UDQ;
    using Dequantizer = typename UDQ::DQ;
    using Element = typename Dequantizer::Element;
    using VectorType = typename Dequantizer::VectorType;

    static constexpr unsigned kGroupK = Trait::kGroupK;
    static constexpr unsigned kGroupN = Trait::kGroupN;
    static constexpr unsigned kNumWarps = Trait::kNumWarps;
    static constexpr unsigned kScaleVecSize = sizeof(unsigned);
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kScaleU32 =
        kGroupK * kGroupN / Trait::kRowGroupSize / kScaleVecSize;

    static constexpr unsigned kDequantOutputBatch =
        kPackFactor * (sizeof(uint4) / sizeof(uint)) * sizeof(half) /
        sizeof(uint4);

    __shared__ unsigned shm_scales[kScaleU32];

    const unsigned tid = threadIdx.x, id_k = blockIdx.x, id_n = blockIdx.y;
    const unsigned wid = tid / kWarpSize, wtid = tid % kWarpSize;

    auto fetch_scale_layout = Trait::GetFetchScaleLayout(size_k, size_n);
    for (unsigned i = 0, idx = tid; i < CeilingDiv(kScaleU32, kThreads);
         i++, idx += kThreads) {
        if (idx < kScaleU32) {
            shm_scales[idx] = reinterpret_cast<const unsigned *>(
                scales)[fetch_scale_layout(make_coord(id_n, id_k, idx))];
        }
    }
    __syncthreads();

    auto layout_in = Trait::GetFetchQWLayout(size_k, size_n);
    uint4 qw = input[layout_in(make_coord(id_k, id_n, wtid, wid))];
    uint2 scale = Trait::GetScale(shm_scales, tid);

    const auto global_scale_f16 = Element(global_scale);
    VectorType gs2{global_scale_f16, global_scale_f16};

    VectorType ret[16];
    typename UDQ::UnpackedData dq;
    for (int i = 0; i < 4; i++) {
        const uint q = reinterpret_cast<const uint *>(&qw)[i];
        const typename Trait::Scale s =
            reinterpret_cast<const typename Trait::Scale *>(&scale)[i];
        unsigned q_shifted;
        if constexpr (Trait::kIsNativeQWFormat) {
            q_shifted = PetitFormat(q);
        } else {
            q_shifted = q;
        }
        UDQ::DequantWithScale(dq, q_shifted, s);

        for (int j = 0; j < 4; j++) {
            ret[i * 4 + j] = __hmul2(dq[j], gs2);
        }
    }

    auto layout_out = Trait::GetOutputLayout(size_k, size_n);
    for (int i = 0; i < kDequantOutputBatch; i++) {
        auto idx = layout_out(make_coord(id_k, id_n, tid, i));
        output[idx] = reinterpret_cast<const uint4 *>(ret)[i];
    }
}

int DequantNvFp4(unsigned *output, const unsigned *input,
                 const unsigned *scales, float global_scale, DataType out_type,
                 unsigned k, unsigned n) {
    if (out_type == kDataTypeFp16) {
        using Trait = DequantTraitNvFp4<DataType::kDataTypeFp16>;
        if (k % Trait::kGroupK != 0 || n % Trait::kGroupN != 0) {
            return -1;
        }
        dim3 grid(k / Trait::kGroupK, n / Trait::kGroupN);
        dim3 block(Trait::kThreads);
        DequantizeFp4Kernel<Trait>
            <<<grid, block>>>(reinterpret_cast<uint4 *>(output),
                              reinterpret_cast<const uint4 *>(input),
                              reinterpret_cast<const unsigned char *>(scales),
                              global_scale, k, n);
    } else if (out_type == kDataTypeBf16) {
        using Trait = DequantTraitNvFp4<DataType::kDataTypeBf16>;
        if (k % Trait::kGroupK != 0 || n % Trait::kGroupN != 0) {
            return -1;
        }
        dim3 grid(k / Trait::kGroupK, n / Trait::kGroupN);
        dim3 block(Trait::kThreads);
        DequantizeFp4Kernel<Trait>
            <<<grid, block>>>(reinterpret_cast<uint4 *>(output),
                              reinterpret_cast<const uint4 *>(input),
                              reinterpret_cast<const unsigned char *>(scales),
                              global_scale, k, n);
    } else {
        return -1;
    }
    return 0;
}

int DequantMxFp4(unsigned *output, const unsigned *input,
                 const unsigned *scales, float global_scale, DataType out_type,
                 unsigned k, unsigned n) {
    if (out_type != kDataTypeBf16) {
        return -1;
    }
    using Trait = DequantTraitMxFp4;

    if (k % Trait::kGroupK != 0 || n % Trait::kGroupN != 0) {
        return -1;
    }

    dim3 grid(k / Trait::kGroupK, n / Trait::kGroupN);
    dim3 block(Trait::kThreads);
    global_scale *= Trait::UDQ::GlobalScaleFactor();
    DequantizeFp4Kernel<Trait><<<grid, block>>>(
        reinterpret_cast<uint4 *>(output),
        reinterpret_cast<const uint4 *>(input),
        reinterpret_cast<const unsigned char *>(scales), global_scale, k, n);
    return 0;
}

int DequantPetitFp4(unsigned *output, const unsigned *input,
                    const unsigned *scales, float global_scale,
                    DataType out_type, unsigned k, unsigned n) {
    // using Layout = RepackQWeightLayout128x16;
    using Layout = RepackQWeightLayout64x32;
    dim3 grid(k / Layout::kGroupM, n / Layout::kGroupN);
    dim3 block(Layout::kNumWarps * kWarpSize);
    if (k % Layout::kGroupM != 0 || n % Layout::kGroupN != 0) {
        return -1;
    }

    if (out_type == kDataTypeFp16) {
        using Trait = DequantTraitPetitNvFp4<Layout, DataType::kDataTypeFp16>;
        using UDQ = typename Trait::UDQ;
        global_scale *= UDQ::GlobalScaleFactor();
        DequantizeFp4Kernel<Trait>
            <<<grid, block>>>(reinterpret_cast<uint4 *>(output),
                              reinterpret_cast<const uint4 *>(input),
                              reinterpret_cast<const unsigned char *>(scales),
                              global_scale, k, n);
    } else if (out_type == kDataTypeBf16) {
        using Trait = DequantTraitPetitNvFp4<Layout, DataType::kDataTypeBf16>;
        using UDQ = typename Trait::UDQ;
        global_scale *= UDQ::GlobalScaleFactor();
        DequantizeFp4Kernel<Trait>
            <<<grid, block>>>(reinterpret_cast<uint4 *>(output),
                              reinterpret_cast<const uint4 *>(input),
                              reinterpret_cast<const unsigned char *>(scales),
                              global_scale, k, n);
    } else {
        return -1;
    }
    return 0;
}

int DequantPetitMxFp4(unsigned *output, const unsigned *input,
                      const unsigned *scales, float global_scale,
                      DataType out_type, unsigned k, unsigned n) {
    using Layout = RepackQWeightLayout64x32;
    dim3 grid(k / Layout::kGroupM, n / Layout::kGroupN);
    dim3 block(Layout::kNumWarps * kWarpSize);
    if (k % Layout::kGroupM != 0 || n % Layout::kGroupN != 0) {
        return -1;
    }

    if (out_type == kDataTypeBf16) {
        using Trait = DequantTraitPetitMxFp4<Layout>;
        using UDQ = typename Trait::UDQ;
        global_scale *= UDQ::GlobalScaleFactor();
        DequantizeFp4Kernel<Trait>
            <<<grid, block>>>(reinterpret_cast<uint4 *>(output),
                              reinterpret_cast<const uint4 *>(input),
                              reinterpret_cast<const unsigned char *>(scales),
                              global_scale, k, n);
    } else {
        return -1;
    }
    return 0;
}

void RepackNvFp4ToPetitFp4Weights(unsigned *output, const unsigned *input,
                                  unsigned in_chan, unsigned out_chan,
                                  hipStream_t stream) {
    // using Layout = RepackQWeightLayout128x16;
    using Layout = RepackQWeightLayout64x32;
    dim3 grid(in_chan / Layout::kGroupM, out_chan / Layout::kGroupN);
    dim3 block(Layout::kNumWarps * kWarpSize);

    struct ProcessWeightOp {
        __device__ uint operator()(uint qv) const { return PetitFormat(qv); }
    };
    ProcessWeightOp op;

    RepackNvFp4ToPetitFp4WeightsKernel<Layout, ProcessWeightOp>
        <<<grid, block, 0, stream>>>(op, reinterpret_cast<uint4 *>(output),
                                     reinterpret_cast<const uint4 *>(input),
                                     in_chan, out_chan);
}

void RepackNvFp4ToPetitFp4Scales(unsigned *out_scales, const unsigned *scales,
                                 unsigned in_chan, unsigned out_chan,
                                 hipStream_t stream) {
    // using ScaleLayout = RepackScaleLayout128x16;
    using ScaleLayout = RepackScaleLayout64x32;
    static constexpr unsigned kGroupM = ScaleLayout::kGroupM;
    static constexpr unsigned kGroupN = ScaleLayout::kGroupN;
    dim3 scale_grid(in_chan / kGroupM, out_chan / kGroupN);
    dim3 block(ScaleLayout::kNumWarps * kWarpSize);
    RepackFp4ScalesKernel<ScaleLayout><<<scale_grid, block, 0, stream>>>(
        reinterpret_cast<uint4 *>(out_scales),
        reinterpret_cast<const uint4 *>(scales), in_chan, out_chan);
}

void RepackMxFp4ToPetitFp4Scales(unsigned *out_scales, const unsigned *scales,
                                 unsigned in_chan, unsigned out_chan,
                                 hipStream_t stream) {
    using ScaleLayout = RepackMxScaleLayout64x32;
    static constexpr unsigned kGroupM = ScaleLayout::kGroupM;
    static constexpr unsigned kGroupN = ScaleLayout::kGroupN;
    dim3 scale_grid(in_chan / kGroupM, out_chan / kGroupN);
    dim3 block(ScaleLayout::kNumWarps * kWarpSize);
    RepackFp4ScalesKernel<ScaleLayout><<<scale_grid, block, 0, stream>>>(
        reinterpret_cast<uint4 *>(out_scales),
        reinterpret_cast<const uint4 *>(scales), in_chan, out_chan);
}

} // namespace causalflow::petit::rocm::quantization::fp4
