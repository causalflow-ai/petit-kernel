#pragma once

#include "gemm/rocm/quantization/types.h"

#include <functional>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::quantization::fp4 {

int DequantPetitFp4(unsigned *output, const unsigned *input,
                    const unsigned *scales, float global_scale,
                    DataType out_type, unsigned k, unsigned n);

int DequantPetitMxFp4(unsigned *output, const unsigned *input,
                      const unsigned *scales, float global_scale,
                      DataType out_type, unsigned k, unsigned n);

template <unsigned long kRepr> struct SolutionAdapter {
    static int Invoke(unsigned *c, const unsigned *a, const unsigned *b,
                      const unsigned *scales, const float *global_scale,
                      const unsigned m, const unsigned n, const unsigned k,
                      hipStream_t stream);
};

struct SolutionMap {
    using Call =
        std::function<int(unsigned *, const unsigned *, const unsigned *,
                          const unsigned *, const float *, const unsigned,
                          const unsigned, const unsigned, hipStream_t)>;
    static const std::unordered_map<unsigned long, Call> &GetDispatchEntries();
};
} // namespace causalflow::petit::rocm::quantization::fp4
