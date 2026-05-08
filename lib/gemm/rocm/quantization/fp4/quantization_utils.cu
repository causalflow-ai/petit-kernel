#include "quantization_utils.cuh"

namespace causalflow::petit::rocm::quantization::fp4 {

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
