#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/quantization/dequant.cuh"
#include "gemm/rocm/quantization/gemm.h"
#include "gemm_fp4.h"

#include <hip/hip_fp8.h>

namespace causalflow::petit::rocm::quantization::fp4 {

static constexpr unsigned kBits = 4;
static constexpr unsigned kPackFactor = 32 / kBits;
static constexpr unsigned kQuantVecSize = sizeof(uint4) / sizeof(unsigned);

// Only support row group size 16 for now
static constexpr unsigned kRowGroupSize = 16;

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

        using OutputShape =
            Shape<_1, _1, Shape<C<kBlockGroupM_>, C<kBlockGroupN_>>,
                  C<kWarpSize>>;
        auto output_stride = make_stride(
            n_ * kGroupM / kPackSize_ / kOutputVecBatch_,
            C<kLayoutM * kGroupN / kPackSize_ / kOutputVecBatch_>{},
            make_stride(n_ * kLayoutM / kPackSize_ / kOutputVecBatch_,
                        C<kWarpSize>{}),
            _1{});
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
    static constexpr unsigned kNumWarps = kBlockGroupM_ * kBlockGroupN_;
    static constexpr unsigned kDequantOutputBatch =
        kPackFactor * (sizeof(uint4) / sizeof(uint)) * sizeof(half) /
        sizeof(uint4);
    static constexpr unsigned kLayoutM = Base::kLayoutM;
    static constexpr unsigned kLayoutN = Base::kLayoutN;
    static constexpr unsigned kGroupM = Base::kGroupM;
    static constexpr unsigned kGroupN = Base::kGroupN;

    // Define how the 4 uints are packed across the (m, n) order
    static_assert(kTileM_ * kTileN_ == 4, "The weight tile must be 4");

    explicit __device__ RepackQWeightLayout(unsigned n, unsigned k)
        : Base(n), k_(k) {}

    __device__ auto GetDequantOutputLayout() const {
        using namespace causalflow::tal;
        static constexpr unsigned kOutVecSize = sizeof(uint4) / sizeof(half);
        // One quantized uint stores 8 halfs so that we can use kTileM / kTileN
        // directly in the layout
        static_assert(kPackFactor * sizeof(half) == sizeof(uint4),
                      "One quantized uint stores 8 halfs");

        using OutputShape =
            Shape<_1, _1, Shape<C<kBlockGroupM_>, C<kBlockGroupN_>>,
                  Shape<_16, _4>, Shape<C<kTileM_>, C<kTileN_>>>;
        auto stride_out =
            make_stride(C<kGroupM / kOutVecSize>{}, k_ * kGroupN / kOutVecSize,
                        make_stride(C<kLayoutM / kOutVecSize>{},
                                    k_ * kLayoutN / kOutVecSize),
                        make_stride(k_ / kOutVecSize, C<kTileM_>{}),
                        make_stride(_1{}, k_ * 16 / kOutVecSize));
        auto layout_out = make_layout(OutputShape{}, stride_out);
        return layout_out;
    }

    unsigned k_;
};

template <unsigned kLayoutM_, unsigned kLayoutN_, unsigned kTileM_,
          unsigned kTileN_, unsigned kBlockGroupM_, unsigned kBlockGroupN_>
struct RepackScaleLayout
    : public TileShmLayout<kLayoutM_, kLayoutN_, kTileM_, kTileN_,
                           kRowGroupSize, sizeof(uchar2) / sizeof(char),
                           kBlockGroupM_, kBlockGroupN_> {
    static constexpr unsigned kNumWarps = kBlockGroupM_ * kBlockGroupN_;
    using Base = TileShmLayout<kLayoutM_, kLayoutN_, kTileM_, kTileN_,
                               kRowGroupSize, sizeof(uchar2) / sizeof(char),
                               kBlockGroupM_, kBlockGroupN_>;

    // Define how the 2 u8 are packed across the (m, n) order
    static_assert(kTileM_ * kTileN_ == 2, "The scale tile must be 2");
};

using RepackQWeightLayout128x16 = RepackQWeightLayout<128, 16, 4, 1, 2, 2>;
using RepackScaleLayout128x16 = RepackScaleLayout<128, 16, 2, 1, 2, 1>;

using RepackQWeightLayout64x32 = RepackQWeightLayout<64, 32, 2, 2, 4, 1>;
using RepackScaleLayout64x32 = RepackScaleLayout<64, 32, 1, 2, 4, 1>;

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
         i < tal::CeilingDiv(kGroupInts / kQuantVecSize, kThreads) &&
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

    auto output_coord = output_layout(make_coord(id_m, id_n, wid, wtid));
    output[output_coord] = *reinterpret_cast<const uint4 *>(ret);
}

template <class ScaleLayout, unsigned kExpBias = 0>
__global__ void RepackNvFp4ScalesKernel(uint4 *__restrict__ out_scales,
                                        const uint4 *__restrict__ scales,
                                        unsigned in_chan, unsigned out_chan) {
    using namespace causalflow::tal;
    using Element = unsigned char;

    static constexpr unsigned kGroupM = ScaleLayout::kGroupM;
    static constexpr unsigned kGroupN = ScaleLayout::kGroupN;
    static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(Element);
    static constexpr unsigned kThreads = kWarpSize;

    const unsigned tid = threadIdx.x, id_m = blockIdx.x, id_n = blockIdx.y;
    const unsigned wid = tid / kWarpSize, wtid = tid % kWarpSize;

    const uint4 *in_scales =
        scales + id_m * kGroupM / kRowGroupSize / kVecSize +
        id_n * in_chan / kRowGroupSize * kGroupN / kVecSize;

    const half kMultiple = half(1 << kExpBias);

    ScaleLayout scale_layout{out_chan};

    auto output_layout = scale_layout.GetOnDiskLayout();

    __shared__ uint4 shm_scales[kGroupM / kRowGroupSize * kGroupN / kVecSize];
    static_assert((kGroupM / kRowGroupSize) % kVecSize == 0,
                  "Failed to read all scales in a single uint4");

    for (unsigned i = 0, idx = tid;
         i < tal::CeilingDiv(kGroupM / kRowGroupSize * kGroupN / kVecSize,
                             kThreads) &&
         idx < kGroupM / kRowGroupSize * kGroupN / kVecSize;
         i++, idx += kThreads) {
        unsigned row = idx / (kGroupM / kRowGroupSize / kVecSize),
                 col = idx % (kGroupM / kRowGroupSize / kVecSize);
        shm_scales[idx] =
            in_scales[row * in_chan / kRowGroupSize / kVecSize + col];
    }

    __syncthreads();

    const __hip_fp8_storage_t *shm =
        reinterpret_cast<const __hip_fp8_storage_t *>(shm_scales);

    auto shm_layout = scale_layout.GetShmLayout();
    unsigned short data;
    auto v = reinterpret_cast<__hip_fp8_storage_t *>(&data);
    v[0] = shm[shm_layout(make_coord(0, wid, wtid))];
    v[1] = shm[shm_layout(make_coord(1, wid, wtid))];
    unsigned short ret;
    if constexpr (kExpBias != 0) {
        half2 h2;
        h2.x = __hip_cvt_fp8_to_halfraw(v[0], __HIP_E4M3);
        h2.y = __hip_cvt_fp8_to_halfraw(v[1], __HIP_E4M3);
        auto scaled = __hmul2(h2, half2{kMultiple, kMultiple});
        unsigned scaled_u32 = reinterpret_cast<const unsigned &>(scaled);
        // Convert the half2 scale to the E5M3 format.
        ret = ((scaled_u32 & 0xffff) >> 7) | ((scaled_u32 >> 23) << 8);
    } else {
        ret = data;
    }
    auto output_coord = output_layout(make_coord(id_m, id_n, wid, wtid));
    auto out_s2 = reinterpret_cast<unsigned short *>(out_scales);
    out_s2[output_coord] = ret;
}

template <class QWLayout, class UDQ>
__global__ void DequantizePetitFp4Kernel(uint4 *__restrict__ output,
                                         const uint4 *__restrict__ input,
                                         const uint4 *__restrict__ scales,
                                         float global_scale, unsigned size_k,
                                         unsigned size_n) {
    using namespace causalflow::tal;
    using DQ = typename UDQ::DQ;
    using Element = typename DQ::Element;
    using VectorType = typename DQ::VectorType;
    static constexpr unsigned kLayoutM = QWLayout::kLayoutM;
    static constexpr unsigned kLayoutN = QWLayout::kLayoutN;
    static constexpr unsigned kGroupM = QWLayout::kGroupM;
    static constexpr unsigned kGroupN = QWLayout::kGroupN;
    static constexpr unsigned kScaleVecSize = sizeof(uchar2) / sizeof(char);

    const unsigned tid = threadIdx.x, id_k = blockIdx.x, id_n = blockIdx.y;
    const unsigned wid = tid / kWarpSize, wtid = tid % kWarpSize;

    QWLayout layout(size_n, size_k);
    auto layout_in = layout.GetOnDiskLayout();

    using ScaleShape =
        Shape<_1, _1, Shape<C<kGroupM / kLayoutM>, C<kGroupN / kLayoutN>>,
              C<kWarpSize>>;
    auto stride_scale = make_stride(
        size_n * kGroupM / kRowGroupSize / kScaleVecSize,
        C<kLayoutM * kGroupN / kRowGroupSize / kScaleVecSize>{},
        make_stride(size_n * kLayoutM / kRowGroupSize / kScaleVecSize,
                    C<kLayoutM * kLayoutN / kRowGroupSize / kScaleVecSize>{}),
        _1{});
    auto layout_scale = make_layout(ScaleShape{}, stride_scale);

    auto layout_out = layout.GetDequantOutputLayout();

    uint4 qw = input[layout_in(make_coord(id_k, id_n, wid, wtid))];
    unsigned short packed_scale = reinterpret_cast<const unsigned short *>(
        scales)[layout_scale(make_coord(id_k, id_n, wid, wtid))];

    auto ds = UDQ::DequantScales(packed_scale);
    const auto global_scale_f16 = Element(global_scale);
    VectorType gs2{global_scale_f16, global_scale_f16};

    VectorType ret[16];

    for (int i = 0; i < 4; i++) {
        const uint q = reinterpret_cast<const uint *>(&qw)[i];
        const auto s = i < 2 ? ds.x : ds.y;
        typename UDQ::UnpackedData dq;
        UDQ::DequantWithScale(dq, q, s);
        for (int j = 0; j < 4; j++) {
            ret[i * 4 + j] = __hmul2(dq[j], gs2);
        }
    }

    const uint4 *ret_ptr = reinterpret_cast<const uint4 *>(ret);
    for (int i = 0; i < QWLayout::kDequantOutputBatch; i++) {
        auto idx = layout_out(make_coord(id_k, id_n, wid, wtid, i));
        output[idx] = ret_ptr[i];
    }
}

template <class UDQ>
__global__ void DequantizeNvFp4Kernel(uint4 *output, const uint4 *input,
                                      const uchar2 *scales, float global_scale,
                                      unsigned size_k, unsigned size_n) {
    using namespace causalflow::tal;
    using Dequantizer = typename UDQ::DQ;
    // using DequantizerForScale = typename UDQ::DS;
    using Element = typename Dequantizer::Element;
    using VectorType = typename Dequantizer::VectorType;
    static constexpr unsigned kGroupK = 128;
    static constexpr unsigned kGroupN = 16;
    static constexpr unsigned kRowGroupSize = 16;

    using Content __attribute__((
        ext_vector_type(32 / (sizeof(uint) / sizeof(Element))))) = uint;
    static constexpr unsigned kOutVecSize = sizeof(Content) / sizeof(Element);
    static constexpr unsigned kScaleVecSize = sizeof(uchar2) / sizeof(char);
    const unsigned tid = threadIdx.x, id_k = blockIdx.x, id_n = blockIdx.y;

    auto stride_in =
        make_stride(C<kGroupK / kPackFactor / kQuantVecSize>{},
                    size_k * kGroupN / kPackFactor / kQuantVecSize,
                    make_stride(_1{}, size_k / kPackFactor / kQuantVecSize));
    auto layout_in = make_layout(Shape<_1, _1, Shape<_4, _16>>{}, stride_in);
    auto stride_out =
        make_stride(C<kGroupK / kOutVecSize>{}, size_k * kGroupN / kOutVecSize,
                    make_stride(_1{}, size_k / kOutVecSize));
    auto layout_out = make_layout(Shape<_1, _1, Shape<_4, _16>>{}, stride_out);

    auto stride_scale =
        make_stride(C<kGroupK / kRowGroupSize / kScaleVecSize>{},
                    size_k / kRowGroupSize * kGroupN / kScaleVecSize,
                    make_stride(_1{}, size_k / kRowGroupSize / kScaleVecSize));
    auto layout_scale =
        make_layout(Shape<_1, _1, Shape<_4, _16>>{}, stride_scale);

    uint4 qw = input[layout_in(make_coord(id_k, id_n, tid))];
    unsigned short packed_scale = reinterpret_cast<const unsigned short *>(
        scales)[layout_scale(make_coord(id_k, id_n, tid))];

    const auto bias = Dequantizer::Bias(false);
    const VectorType bias2{bias, bias};

    // The scale is in the E4M3 format, we need to convert it to the E5M3 format
    // and times 2**7 to reuse the dequantizer.
    half2 ds;
    ds.x = __hip_cvt_fp8_to_halfraw(packed_scale & 0xff, __HIP_E4M3);
    ds.y = __hip_cvt_fp8_to_halfraw((packed_scale >> 8), __HIP_E4M3);
    ds = __hmul2(ds, half2{1 << kFp8ScaleBias, 1 << kFp8ScaleBias});

    VectorType ret[16];

    const auto global_scale_f16 = Element(global_scale);
    VectorType gs2{global_scale_f16, global_scale_f16};

    typename UDQ::UnpackedData dq;
    for (int i = 0; i < 4; i++) {
        const uint q = reinterpret_cast<const uint *>(&qw)[i];
        unsigned q_shifted = PetitFormat(q);
        const auto s = i < 2 ? ds.x : ds.y;
        UDQ::DequantWithScale(dq, q_shifted, s);

        for (int j = 0; j < 4; j++) {
            ret[i * 4 + j] = __hmul2(dq[j], gs2);
        }
    }

    reinterpret_cast<Content *>(
        output)[layout_out(make_coord(id_k, id_n, tid))] =
        *reinterpret_cast<Content *>(ret);
}

int DequantNvFp4(unsigned *output, const unsigned *input,
                 const unsigned *scales, float global_scale, DataType out_type,
                 unsigned k, unsigned n) {
    static constexpr unsigned kGroupM = 128;
    static constexpr unsigned kGroupN = 16;
    dim3 grid(k / kGroupM, n / kGroupN);
    dim3 block(kWarpSize);
    // We do not need to update the global scale as the kernel compute the bias
    // directly
    if (out_type == kDataTypeFp16) {
        using UDQ = UnifiedDequantizerForFp4Fp16<true>;
        DequantizeNvFp4Kernel<UDQ><<<grid, block>>>(
            reinterpret_cast<uint4 *>(output),
            reinterpret_cast<const uint4 *>(input),
            reinterpret_cast<const uchar2 *>(scales), global_scale, k, n);
    } else if (out_type == kDataTypeBf16) {
        using UDQ = UnifiedDequantizerForFp4Bf16<true>;
        DequantizeNvFp4Kernel<UDQ><<<grid, block>>>(
            reinterpret_cast<uint4 *>(output),
            reinterpret_cast<const uint4 *>(input),
            reinterpret_cast<const uchar2 *>(scales), global_scale, k, n);
    } else {
        return -1;
    }
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
        using UDQ = UnifiedDequantizerForFp4Fp16<true>;
        global_scale *= UDQ::DS::GlobalScaleFactor();
        DequantizePetitFp4Kernel<Layout, UDQ><<<grid, block>>>(
            reinterpret_cast<uint4 *>(output),
            reinterpret_cast<const uint4 *>(input),
            reinterpret_cast<const uint4 *>(scales), global_scale, k, n);
    } else if (out_type == kDataTypeBf16) {
        using UDQ = UnifiedDequantizerForFp4Bf16<true>;
        global_scale *= UDQ::DS::GlobalScaleFactor();
        DequantizePetitFp4Kernel<Layout, UDQ><<<grid, block>>>(
            reinterpret_cast<uint4 *>(output),
            reinterpret_cast<const uint4 *>(input),
            reinterpret_cast<const uint4 *>(scales), global_scale, k, n);
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
    RepackNvFp4ScalesKernel<ScaleLayout, kFp8ScaleBias>
        <<<scale_grid, block, 0, stream>>>(
            reinterpret_cast<uint4 *>(out_scales),
            reinterpret_cast<const uint4 *>(scales), in_chan, out_chan);
}

} // namespace causalflow::petit::rocm::quantization::fp4