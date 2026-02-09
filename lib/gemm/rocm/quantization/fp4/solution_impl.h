#pragma once

#include "gemm_fp4_fp16_grid.cuh"

namespace causalflow::petit::rocm::quantization::fp4 {

template <unsigned long kRepr>
int SolutionAdapter<kRepr>::Invoke(unsigned *c, const unsigned *a,
                                   const unsigned *b, const unsigned *scales,
                                   const float *global_scale, const unsigned m,
                                   const unsigned n, const unsigned k,
                                   hipStream_t stream) {
    static constexpr SolutionId kSolId = SolutionId::FromRepr(kRepr);
    if constexpr (kSolId.element_b == MatmulElementB::kMatmulTypeBMxFp4) {
        if constexpr (kSolId.mfma_type != MatmulMfmaType::kMatmulMfmaTypeBf16) {
            return kErrorKernelShape;
        }
    }
    using Impl = ConfigSelector<kSolId>;
    return Impl::Invoke(c, a, b, scales, global_scale, m, n, k, stream);
}

} // namespace causalflow::petit::rocm::quantization::fp4

#define PETIT_KERNEL_IMPL(s)                                                   \
    namespace causalflow::petit::rocm::quantization::fp4 {                     \
    template struct SolutionAdapter<0x##s##ul>;                                \
    }
