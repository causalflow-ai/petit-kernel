#include "moe/rocm/fused_moe.h"
#include "pybind.h"

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>

#include <cstring>

namespace causalflow::petit::pybind {

using causalflow::petit::rocm::moe::FusedMoE1StageParams;
using causalflow::petit::rocm::moe::FusedMoEDataType;
using causalflow::petit::rocm::moe::FusedMoESolutionId;
using causalflow::petit::rocm::moe::kFusedMoEErrorInvalidArgument;
using causalflow::petit::rocm::moe::kFusedMoEErrorInvalidSolution;
using causalflow::petit::rocm::moe::kFusedMoEErrorUnsupported;

namespace {

void CheckKernelStatus(int err, const char *kernel_name) {
    if (err == kFusedMoEErrorInvalidSolution) {
        TORCH_CHECK(false, kernel_name, " invalid solution");
    }
    if (err == kFusedMoEErrorInvalidArgument) {
        TORCH_CHECK(false, kernel_name, " invalid argument");
    }
    if (err == kFusedMoEErrorUnsupported) {
        TORCH_CHECK(false, kernel_name, " unsupported");
    }
    TORCH_CHECK(err == 0, kernel_name, " failed with code ", err);
}

bool IsFusedMoEArchSupported(int dev) {
    hipDeviceProp_t props;
    hipError_t error = hipGetDeviceProperties(&props, dev);
    if (error != hipSuccess) {
        TORCH_CHECK(false, "Failed to get HIP device properties for device ",
                    dev);
    }

    const char *arch = props.gcnArchName;
    return std::strncmp(arch, "gfx940", 6) == 0 ||
           std::strncmp(arch, "gfx941", 6) == 0 ||
           std::strncmp(arch, "gfx942", 6) == 0 ||
           std::strncmp(arch, "gfx950", 6) == 0;
}

void CheckFusedMoEArchSupported(int dev, const char *kernel_name) {
    TORCH_CHECK(IsFusedMoEArchSupported(dev), kernel_name,
                " is only supported on gfx940, gfx941, gfx942, and gfx950");
}

struct FusedMoeShape {
    int64_t tokens;
    int64_t dim;
    int64_t inter_dim;
    int64_t experts;
    int64_t max_num_m_blocks;
};

FusedMoeShape MakeFusedMoEShape(const torch::Tensor &input_q,
                                const torch::Tensor &w2_q,
                                const torch::Tensor &sorted_expert_ids,
                                uint64_t solution_id) {
    const int64_t tokens = input_q.size(0);
    const int64_t dim = input_q.size(1);
    const int64_t experts = w2_q.size(0);
    const auto solution = FusedMoESolutionId::FromRepr(solution_id);
    const int64_t inter_dim =
        (solution.weight_dtype == FusedMoEDataType::kMxFp4 ||
         solution.weight_dtype == FusedMoEDataType::kNvFp4)
            ? w2_q.size(2) * 2
            : w2_q.size(2);
    return {tokens, dim, inter_dim, experts, sorted_expert_ids.numel()};
}

void RunFusedMoEMatmul(
    const torch::Tensor &out, const torch::Tensor &input_q,
    const torch::Tensor &w1_q, const torch::Tensor &w2_q,
    const torch::Tensor &sorted_token_ids, const torch::Tensor &sorted_weights,
    const torch::Tensor &sorted_expert_ids, const torch::Tensor &num_valid_ids,
    int64_t topk, const torch::Tensor &input_scale,
    const torch::Tensor &w1_scale, const torch::Tensor &w2_scale,
    uint64_t solution_id, const FusedMoeShape &shape,
    int64_t num_persistent_tgs) {
    if (shape.max_num_m_blocks == 0) {
        return;
    }

    const int dev = input_q.get_device();
    const auto current_stream = c10::hip::getCurrentHIPStream(dev);
    hipStream_t stream = current_stream.stream();
    FusedMoE1StageParams args{
        reinterpret_cast<unsigned *>(out.data_ptr()),
        reinterpret_cast<const unsigned *>(input_q.data_ptr()),
        reinterpret_cast<const unsigned *>(w1_q.data_ptr()),
        reinterpret_cast<const unsigned *>(w2_q.data_ptr()),
        reinterpret_cast<const unsigned *>(sorted_token_ids.data_ptr()),
        reinterpret_cast<const unsigned *>(sorted_weights.data_ptr()),
        reinterpret_cast<const unsigned *>(sorted_expert_ids.data_ptr()),
        reinterpret_cast<const unsigned *>(num_valid_ids.data_ptr()),
        static_cast<unsigned>(topk),
        reinterpret_cast<const unsigned *>(input_scale.data_ptr()),
        reinterpret_cast<const unsigned *>(w1_scale.data_ptr()),
        reinterpret_cast<const unsigned *>(w2_scale.data_ptr()),
        static_cast<unsigned>(shape.max_num_m_blocks),
        static_cast<unsigned>(shape.tokens),
        static_cast<unsigned>(shape.dim),
        static_cast<unsigned>(shape.inter_dim),
        static_cast<unsigned>(shape.experts),
        stream,
        static_cast<unsigned>(num_persistent_tgs),
    };
    const int err =
        causalflow::petit::rocm::moe::FusedMoEMatmul1Stage(args, solution_id);
    CheckKernelStatus(err, "FusedMoEMatmul1Stage");
}

} // namespace

torch::Tensor FusedMoeMatmul1Stage(
    const torch::Tensor &out, const torch::Tensor &input_q,
    const torch::Tensor &w1_q, const torch::Tensor &w2_q,
    const torch::Tensor &sorted_token_ids, const torch::Tensor &sorted_weights,
    const torch::Tensor &sorted_expert_ids, const torch::Tensor &num_valid_ids,
    int64_t topk, const torch::Tensor &input_scale,
    const torch::Tensor &w1_scale, const torch::Tensor &w2_scale,
    uint64_t solution_id, int64_t num_persistent_tgs) {
    TORCH_CHECK(input_q.device().is_cuda(), "input_q must be on GPU");
    const int dev = input_q.get_device();
    CheckFusedMoEArchSupported(dev, "FusedMoE");

    const FusedMoeShape shape =
        MakeFusedMoEShape(input_q, w2_q, sorted_expert_ids, solution_id);
    RunFusedMoEMatmul(out, input_q, w1_q, w2_q, sorted_token_ids,
                      sorted_weights, sorted_expert_ids, num_valid_ids, topk,
                      input_scale, w1_scale, w2_scale, solution_id, shape,
                      num_persistent_tgs);
    return out;
}

} // namespace causalflow::petit::pybind
