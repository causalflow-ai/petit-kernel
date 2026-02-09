#include "gemm_fp4.h"

#include <unordered_map>

#define PETIT_KERNEL_IMPL(s) {0x##s##ul, SolutionAdapter<0x##s##ul>::Invoke},

namespace causalflow::petit::rocm::quantization::fp4 {

std::unordered_map<unsigned long, SolutionMap::Call> kSolutions = {
#include "solutions.inl"
};

const std::unordered_map<unsigned long, SolutionMap::Call> &
SolutionMap::GetDispatchEntries() {
    return kSolutions;
}

} // namespace causalflow::petit::rocm::quantization::fp4
