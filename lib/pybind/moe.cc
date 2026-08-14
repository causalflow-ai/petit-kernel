#include "moe/rocm/fused_moe.h"
#include "pybind.h"

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>

#include <cstring>

namespace causalflow::petit::pybind {

using causalflow::petit::rocm::moe::FusedMoE1StageParams;
using causalflow::petit::rocm::moe::FusedMoE2Stage1Params;
using causalflow::petit::rocm::moe::FusedMoE2Stage2Params;
using causalflow::petit::rocm::moe::FusedMoE2StageCommonParams;
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

void CheckIntermediate(const torch::Tensor &intermediate,
                       const torch::Device &device,
                       std::size_t required_bytes) {
    TORCH_CHECK(intermediate.device() == device,
                "intermediate device mismatch");
    TORCH_CHECK(intermediate.dtype() == torch::kUInt8,
                "intermediate must be uint8");
    TORCH_CHECK(intermediate.is_contiguous(),
                "intermediate must be contiguous");
    TORCH_CHECK(static_cast<std::size_t>(intermediate.numel()) >=
                    required_bytes,
                "intermediate is too small: need ", required_bytes,
                " bytes, got ", intermediate.numel());
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
    const auto solution = FusedMoESolutionId::FromRepr(solution_id);
    const int64_t dim = solution.act_dtype == FusedMoEDataType::kMxFp4
                            ? input_q.size(1) * 2
                            : input_q.size(1);
    const int64_t experts = w2_q.size(0);
    const int64_t inter_dim =
        (solution.weight_dtype == FusedMoEDataType::kMxFp4 ||
         solution.weight_dtype == FusedMoEDataType::kNvFp4)
            ? w2_q.size(2) * 2
            : w2_q.size(2);
    return {tokens, dim, inter_dim, experts, sorted_expert_ids.numel()};
}

void RunFusedMoEMatmul(const torch::Tensor &out, const torch::Tensor &input_q,
                       const torch::Tensor &w1_q, const torch::Tensor &w2_q,
                       const torch::Tensor &sorted_token_ids,
                       const torch::Tensor &sorted_weights,
                       const torch::Tensor &sorted_expert_ids,
                       const torch::Tensor &num_valid_ids, int64_t topk,
                       const torch::Tensor &input_scale,
                       const torch::Tensor &w1_scale,
                       const torch::Tensor &w2_scale, uint64_t solution_id,
                       const FusedMoeShape &shape, int64_t num_persistent_tgs,
                       const std::optional<torch::Tensor> &w13_bias,
                       const std::optional<torch::Tensor> &w2_bias) {
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
        w13_bias.has_value() ? w13_bias->data_ptr() : nullptr,
        w2_bias.has_value() ? w2_bias->data_ptr() : nullptr,
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
    uint64_t solution_id, int64_t num_persistent_tgs,
    const std::optional<torch::Tensor> &w13_bias,
    const std::optional<torch::Tensor> &w2_bias) {
    TORCH_CHECK(input_q.device().is_cuda(), "input_q must be on GPU");
    const int dev = input_q.get_device();
    CheckFusedMoEArchSupported(dev, "FusedMoE");
    if (w13_bias.has_value()) {
        TORCH_CHECK(w13_bias->device() == input_q.device(),
                    "w13_bias device mismatch");
        TORCH_CHECK(w13_bias->dtype() == torch::kBFloat16,
                    "w13_bias must be bfloat16");
        TORCH_CHECK(w13_bias->is_contiguous(), "w13_bias must be contiguous");
    }
    if (w2_bias.has_value()) {
        TORCH_CHECK(w2_bias->device() == input_q.device(),
                    "w2_bias device mismatch");
        TORCH_CHECK(w2_bias->dtype() == torch::kBFloat16,
                    "w2_bias must be bfloat16");
        TORCH_CHECK(w2_bias->is_contiguous(), "w2_bias must be contiguous");
    }

    const FusedMoeShape shape =
        MakeFusedMoEShape(input_q, w2_q, sorted_expert_ids, solution_id);
    RunFusedMoEMatmul(out, input_q, w1_q, w2_q, sorted_token_ids,
                      sorted_weights, sorted_expert_ids, num_valid_ids, topk,
                      input_scale, w1_scale, w2_scale, solution_id, shape,
                      num_persistent_tgs, w13_bias, w2_bias);
    return out;
}

int64_t FusedMoe2StageWorkspaceSize(int64_t max_num_m_blocks, int64_t inter_dim,
                                    uint64_t solution_id) {
    TORCH_CHECK(max_num_m_blocks >= 0, "max_num_m_blocks must be non-negative");
    TORCH_CHECK(inter_dim > 0, "inter_dim must be positive");
    const std::size_t bytes =
        causalflow::petit::rocm::moe::FusedMoE2StageWorkspaceSize(
            static_cast<unsigned>(max_num_m_blocks),
            static_cast<unsigned>(inter_dim), solution_id);
    TORCH_CHECK(bytes > 0, "get_2stage_cfgs unsupported configuration");
    return static_cast<int64_t>(bytes);
}

torch::Tensor FusedMoeMatmul2Stage1(
    const torch::Tensor &intermediate, const torch::Tensor &input_q,
    const torch::Tensor &w1_q, const torch::Tensor &sorted_token_ids,
    const torch::Tensor &sorted_expert_ids, const torch::Tensor &num_valid_ids,
    int64_t topk, const torch::Tensor &input_scale,
    const torch::Tensor &w1_scale, int64_t inter_dim, int64_t num_experts,
    uint64_t solution_id, int64_t num_persistent_tgs,
    const std::optional<torch::Tensor> &w13_bias) {
    TORCH_CHECK(input_q.device().is_cuda(), "input_q must be on GPU");
    TORCH_CHECK(inter_dim > 0, "inter_dim must be positive");
    TORCH_CHECK(num_experts > 0, "num_experts must be positive");
    const int dev = input_q.get_device();
    CheckFusedMoEArchSupported(dev, "FusedMoE stage1");
    const auto solution = FusedMoESolutionId::FromRepr(solution_id);
    const int64_t dim = solution.act_dtype == FusedMoEDataType::kMxFp4
                            ? input_q.size(1) * 2
                            : input_q.size(1);
    const int64_t max_num_m_blocks = sorted_expert_ids.numel();
    const std::size_t required_bytes =
        causalflow::petit::rocm::moe::FusedMoE2StageWorkspaceSize(
            static_cast<unsigned>(max_num_m_blocks),
            static_cast<unsigned>(inter_dim), solution_id);
    TORCH_CHECK(required_bytes > 0,
                "FusedMoEMatmul2Stage1 invalid configuration");
    CheckIntermediate(intermediate, input_q.device(), required_bytes);
    if (w13_bias.has_value()) {
        TORCH_CHECK(w13_bias->device() == input_q.device(),
                    "w13_bias device mismatch");
        TORCH_CHECK(solution.bias_dtype == FusedMoEDataType::kBf16,
                    "two-stage MoE only supports bfloat16 bias");
        TORCH_CHECK(w13_bias->dtype() == torch::kBFloat16,
                    "w13_bias dtype does not match solution");
        TORCH_CHECK(w13_bias->is_contiguous(), "w13_bias must be contiguous");
    }

    const auto stream = c10::hip::getCurrentHIPStream(dev).stream();
    FusedMoE2StageCommonParams common{};
    common.intermediate = intermediate.data_ptr();
    common.intermediate_bytes = static_cast<std::size_t>(intermediate.numel());
    common.sorted_token_ids =
        reinterpret_cast<const unsigned *>(sorted_token_ids.data_ptr());
    common.sorted_expert_ids =
        reinterpret_cast<const unsigned *>(sorted_expert_ids.data_ptr());
    common.num_valid_ids =
        reinterpret_cast<const unsigned *>(num_valid_ids.data_ptr());
    common.topk = static_cast<unsigned>(topk);
    common.max_num_m_blocks = static_cast<unsigned>(max_num_m_blocks);
    common.m = static_cast<unsigned>(input_q.size(0));
    common.n = static_cast<unsigned>(dim);
    common.k = static_cast<unsigned>(inter_dim);
    common.num_experts = static_cast<unsigned>(num_experts);
    common.stream = stream;
    common.num_persistent_tgs = static_cast<unsigned>(num_persistent_tgs);
    const FusedMoE2Stage1Params params{
        common,
        reinterpret_cast<const unsigned *>(input_q.data_ptr()),
        reinterpret_cast<const unsigned *>(w1_q.data_ptr()),
        reinterpret_cast<const unsigned *>(input_scale.data_ptr()),
        reinterpret_cast<const unsigned *>(w1_scale.data_ptr()),
        w13_bias.has_value() ? w13_bias->data_ptr() : nullptr,
    };
    const int err = causalflow::petit::rocm::moe::FusedMoEMatmul2Stage1(
        params, solution_id);
    CheckKernelStatus(err, "FusedMoEMatmul2Stage1");
    return intermediate;
}

torch::Tensor FusedMoeMatmul2Stage2(
    const torch::Tensor &out, const torch::Tensor &intermediate,
    const torch::Tensor &w2_q, const torch::Tensor &sorted_token_ids,
    const torch::Tensor &sorted_weights, const torch::Tensor &sorted_expert_ids,
    const torch::Tensor &num_valid_ids, int64_t topk,
    const torch::Tensor &w2_scale, int64_t inter_dim, int64_t num_experts,
    uint64_t solution_id, int64_t num_persistent_tgs,
    const std::optional<torch::Tensor> &w2_bias) {
    TORCH_CHECK(out.device().is_cuda(), "out must be on GPU");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be bfloat16");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(inter_dim > 0, "inter_dim must be positive");
    TORCH_CHECK(num_experts > 0, "num_experts must be positive");
    const int dev = out.get_device();
    CheckFusedMoEArchSupported(dev, "FusedMoE stage2");
    const int64_t max_num_m_blocks = sorted_expert_ids.numel();
    const std::size_t required_bytes =
        causalflow::petit::rocm::moe::FusedMoE2StageWorkspaceSize(
            static_cast<unsigned>(max_num_m_blocks),
            static_cast<unsigned>(inter_dim), solution_id);
    TORCH_CHECK(required_bytes > 0,
                "FusedMoEMatmul2Stage2 invalid configuration");
    CheckIntermediate(intermediate, out.device(), required_bytes);
    if (w2_bias.has_value()) {
        TORCH_CHECK(w2_bias->device() == out.device(),
                    "w2_bias device mismatch");
        const auto solution = FusedMoESolutionId::FromRepr(solution_id);
        TORCH_CHECK(solution.bias_dtype == FusedMoEDataType::kBf16,
                    "two-stage MoE only supports bfloat16 bias");
        TORCH_CHECK(w2_bias->dtype() == torch::kBFloat16,
                    "w2_bias dtype does not match solution");
        TORCH_CHECK(w2_bias->is_contiguous(), "w2_bias must be contiguous");
    }

    const auto stream = c10::hip::getCurrentHIPStream(dev).stream();
    FusedMoE2StageCommonParams common{};
    common.intermediate = intermediate.data_ptr();
    common.intermediate_bytes = static_cast<std::size_t>(intermediate.numel());
    common.sorted_token_ids =
        reinterpret_cast<const unsigned *>(sorted_token_ids.data_ptr());
    common.sorted_expert_ids =
        reinterpret_cast<const unsigned *>(sorted_expert_ids.data_ptr());
    common.num_valid_ids =
        reinterpret_cast<const unsigned *>(num_valid_ids.data_ptr());
    common.topk = static_cast<unsigned>(topk);
    common.max_num_m_blocks = static_cast<unsigned>(max_num_m_blocks);
    common.m = static_cast<unsigned>(out.size(0));
    common.n = static_cast<unsigned>(out.size(1));
    common.k = static_cast<unsigned>(inter_dim);
    common.num_experts = static_cast<unsigned>(num_experts);
    common.stream = stream;
    common.num_persistent_tgs = static_cast<unsigned>(num_persistent_tgs);
    const FusedMoE2Stage2Params params{
        common,
        reinterpret_cast<unsigned *>(out.data_ptr()),
        reinterpret_cast<const unsigned *>(w2_q.data_ptr()),
        reinterpret_cast<const unsigned *>(sorted_weights.data_ptr()),
        reinterpret_cast<const unsigned *>(w2_scale.data_ptr()),
        w2_bias.has_value() ? w2_bias->data_ptr() : nullptr,
    };
    const int err = causalflow::petit::rocm::moe::FusedMoEMatmul2Stage2(
        params, solution_id);
    CheckKernelStatus(err, "FusedMoEMatmul2Stage2");
    return out;
}

} // namespace causalflow::petit::pybind
