#include "moe/rocm/fused_moe.h"
#include "pybind.h"

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>

#include <cstring>

namespace causalflow::petit::pybind {

using causalflow::petit::rocm::moe::FusedMoEBlockScaleFP8;
using causalflow::petit::rocm::moe::FusedMoEBlockScaleFP8MXFP4Weight;
using causalflow::petit::rocm::moe::kFusedMoEErrorInvalidArgument;
using causalflow::petit::rocm::moe::kFusedMoEErrorInvalidShape;
using causalflow::petit::rocm::moe::kFusedMoEErrorUnsupportedArch;

namespace {

void CheckKernelStatus(int err, const char *kernel_name) {
    if (err == kFusedMoEErrorInvalidShape) {
        TORCH_CHECK(false, kernel_name, " invalid shape");
    }
    if (err == kFusedMoEErrorInvalidArgument) {
        TORCH_CHECK(false, kernel_name, " invalid argument");
    }
    if (err == kFusedMoEErrorUnsupportedArch) {
        TORCH_CHECK(false, kernel_name,
                    " is only supported on gfx940, gfx941, gfx942, and gfx950");
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

void CheckCommonMoEInputs(const torch::Tensor &input_q,
                          const torch::Tensor &w13_q, const torch::Tensor &w2_q,
                          const torch::Tensor &sorted_token_ids,
                          const torch::Tensor &sorted_weights,
                          const torch::Tensor &sorted_expert_ids,
                          const torch::Tensor &num_valid_ids,
                          const torch::Tensor &input_scale, int64_t topk,
                          int64_t num_persistent_tgs) {
    TORCH_CHECK(input_q.device().is_cuda(), "input_q must be on GPU");
    TORCH_CHECK(w13_q.device() == input_q.device(), "w13_q device mismatch");
    TORCH_CHECK(w2_q.device() == input_q.device(), "w2_q device mismatch");
    TORCH_CHECK(sorted_token_ids.device() == input_q.device(),
                "sorted_token_ids device mismatch");
    TORCH_CHECK(sorted_weights.device() == input_q.device(),
                "sorted_weights device mismatch");
    TORCH_CHECK(sorted_expert_ids.device() == input_q.device(),
                "sorted_expert_ids device mismatch");
    TORCH_CHECK(num_valid_ids.device() == input_q.device(),
                "num_valid_ids device mismatch");
    TORCH_CHECK(input_scale.device() == input_q.device(),
                "input_scale device mismatch");

    TORCH_CHECK(input_q.dim() == 2, "input_q must be [tokens, dim]");
    TORCH_CHECK(w13_q.dim() == 3, "w13_q must be rank 3");
    TORCH_CHECK(w2_q.dim() == 3, "w2_q must be rank 3");
    TORCH_CHECK(input_q.is_contiguous(), "input_q must be contiguous");
    TORCH_CHECK(w13_q.is_contiguous(), "w13_q must be contiguous");
    TORCH_CHECK(w2_q.is_contiguous(), "w2_q must be contiguous");
    TORCH_CHECK(sorted_token_ids.is_contiguous(),
                "sorted_token_ids must be contiguous");
    TORCH_CHECK(sorted_weights.is_contiguous(),
                "sorted_weights must be contiguous");
    TORCH_CHECK(sorted_expert_ids.is_contiguous(),
                "sorted_expert_ids must be contiguous");
    TORCH_CHECK(num_valid_ids.is_contiguous(),
                "num_valid_ids must be contiguous");
    TORCH_CHECK(sorted_token_ids.dtype() == torch::kInt32,
                "sorted_token_ids must be int32");
    TORCH_CHECK(sorted_weights.dtype() == torch::kFloat32,
                "sorted_weights must be float32");
    TORCH_CHECK(sorted_expert_ids.dtype() == torch::kInt32,
                "sorted_expert_ids must be int32");
    TORCH_CHECK(num_valid_ids.dtype() == torch::kInt32,
                "num_valid_ids must be int32");
    TORCH_CHECK(num_valid_ids.numel() == 2,
                "num_valid_ids must have 2 elements");
    TORCH_CHECK(input_scale.dtype() == torch::kFloat32,
                "input_scale must be float32");
    TORCH_CHECK(topk > 0, "topk must be > 0");
    TORCH_CHECK(num_persistent_tgs >= 0, "num_persistent_tgs must be >= 0");
}

void CheckInputScale(const torch::Tensor &input_scale, int64_t tokens,
                     int64_t dim) {
    TORCH_CHECK(dim % 128 == 0, "dim must be divisible by 128");
    const int64_t n_blocks = dim / 128;
    TORCH_CHECK(input_scale.dim() == 2, "input_scale must be rank 2");
    TORCH_CHECK(input_scale.size(0) == n_blocks &&
                    input_scale.size(1) == tokens,
                "input_scale must be [dim/128, tokens]");
    TORCH_CHECK(input_scale.is_contiguous(), "input_scale must be contiguous");
}

} // namespace

torch::Tensor FusedMoeFp8BlockscaleG1u1(
    const torch::Tensor &input_q, const torch::Tensor &w13_q,
    const torch::Tensor &w2_q, const torch::Tensor &sorted_token_ids,
    const torch::Tensor &sorted_weights, const torch::Tensor &sorted_expert_ids,
    const torch::Tensor &num_valid_ids, int64_t topk,
    const torch::Tensor &input_scale, const torch::Tensor &fc1_scale,
    const torch::Tensor &fc2_scale, int64_t num_persistent_tgs,
    const std::optional<torch::Tensor> &out_opt) {
    TORCH_CHECK(fc1_scale.dtype() == torch::kFloat32,
                "fc1_scale must be float32");
    TORCH_CHECK(fc2_scale.dtype() == torch::kFloat32,
                "fc2_scale must be float32");
    TORCH_CHECK(w13_q.dim() == 3, "w13_q must be [experts, 2*inter, dim]");
    TORCH_CHECK(w2_q.dim() == 3, "w2_q must be [experts, dim, inter]");

    const int64_t tokens = input_q.size(0);
    const int64_t dim = input_q.size(1);
    const int64_t experts = w2_q.size(0);
    const int64_t inter_dim = w2_q.size(2);
    const int64_t max_num_m_blocks = sorted_expert_ids.numel();
    TORCH_CHECK(w2_q.size(1) == dim, "w2_q dim mismatch with input_q");
    TORCH_CHECK(w13_q.size(0) == experts, "w13_q experts mismatch with w2_q");
    TORCH_CHECK(w13_q.size(1) == 2 * inter_dim,
                "w13_q second dim must be 2*inter_dim");
    TORCH_CHECK(w13_q.size(2) == dim, "w13_q dim mismatch with input_q");
    TORCH_CHECK(dim % 128 == 0, "dim must be divisible by 128");
    TORCH_CHECK(inter_dim % 128 == 0, "inter_dim must be divisible by 128");
    const int64_t n_blocks = dim / 128;
    TORCH_CHECK(input_q.is_contiguous(), "input_q must be contiguous");
    TORCH_CHECK(w13_q.is_contiguous(), "w13_q must be contiguous");
    TORCH_CHECK(w2_q.is_contiguous(), "w2_q must be contiguous");
    TORCH_CHECK(sorted_token_ids.is_contiguous(),
                "sorted_token_ids must be contiguous");
    TORCH_CHECK(sorted_weights.is_contiguous(),
                "sorted_weights must be contiguous");
    TORCH_CHECK(sorted_expert_ids.is_contiguous(),
                "sorted_expert_ids must be contiguous");
    TORCH_CHECK(num_valid_ids.is_contiguous(),
                "num_valid_ids must be contiguous");
    TORCH_CHECK(input_scale.dim() == 2 && input_scale.size(0) == tokens &&
                    input_scale.size(1) == n_blocks,
                "input_scale must be contiguous with shape [tokens, dim/128]");
    TORCH_CHECK(input_scale.is_contiguous(), "input_scale must be contiguous");
    TORCH_CHECK(fc1_scale.is_contiguous(), "fc1_scale must be contiguous");
    TORCH_CHECK(fc2_scale.is_contiguous(), "fc2_scale must be contiguous");
    torch::Tensor out;
    if (out_opt.has_value()) {
        out = *out_opt;
        TORCH_CHECK(out.device() == input_q.device(), "out device mismatch");
        TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be bfloat16");
        TORCH_CHECK(out.dim() == 2, "out must be [tokens, dim]");
        TORCH_CHECK(out.size(0) == tokens && out.size(1) == dim,
                    "out shape mismatch");
        TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
        out.zero_();
    } else {
        auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(input_q.device());
        out = torch::zeros({tokens, dim}, options);
    }

    const int dev = input_q.get_device();
    CheckFusedMoEArchSupported(dev, "FusedMoEBlockScaleFP8");
    const auto current_stream = c10::hip::getCurrentHIPStream(dev);
    hipStream_t stream = current_stream.stream();
    if (max_num_m_blocks == 0) {
        return out;
    }

    const int err = FusedMoEBlockScaleFP8(
        reinterpret_cast<uint4 *>(out.data_ptr()),
        reinterpret_cast<const uint4 *>(input_q.data_ptr()),
        reinterpret_cast<const uint4 *>(w13_q.data_ptr()),
        reinterpret_cast<const uint4 *>(w2_q.data_ptr()),
        reinterpret_cast<const uint4 *>(sorted_token_ids.data_ptr()),
        reinterpret_cast<const uint4 *>(sorted_weights.data_ptr()),
        reinterpret_cast<const uint4 *>(sorted_expert_ids.data_ptr()),
        reinterpret_cast<const unsigned *>(num_valid_ids.data_ptr()),
        static_cast<unsigned>(topk),
        reinterpret_cast<const uint4 *>(input_scale.data_ptr()),
        reinterpret_cast<const uint4 *>(fc1_scale.data_ptr()),
        reinterpret_cast<const unsigned *>(fc2_scale.data_ptr()),
        static_cast<unsigned>(max_num_m_blocks), static_cast<unsigned>(tokens),
        static_cast<unsigned>(dim), static_cast<unsigned>(inter_dim), stream,
        static_cast<unsigned>(num_persistent_tgs));

    CheckKernelStatus(err, "FusedMoEBlockScaleFP8");
    return out;
}

torch::Tensor FusedMoeFp8BlockscaleG1u1MxFp4(
    const torch::Tensor &input_q, const torch::Tensor &w13_q,
    const torch::Tensor &w2_q, const torch::Tensor &sorted_token_ids,
    const torch::Tensor &sorted_weights, const torch::Tensor &sorted_expert_ids,
    const torch::Tensor &num_valid_ids, int64_t topk,
    const torch::Tensor &input_scale, const torch::Tensor &fc1_scale,
    const torch::Tensor &fc2_scale, int64_t num_persistent_tgs,
    const std::optional<torch::Tensor> &out_opt) {
    CheckCommonMoEInputs(input_q, w13_q, w2_q, sorted_token_ids, sorted_weights,
                         sorted_expert_ids, num_valid_ids, input_scale, topk,
                         num_persistent_tgs);
    TORCH_CHECK(input_q.scalar_type() == at::kByte ||
                    input_q.scalar_type() == torch::kFloat8_e4m3fn ||
                    input_q.scalar_type() == torch::kFloat8_e4m3fnuz,
                "input_q must be uint8, float8_e4m3fn, or float8_e4m3fnuz");
    TORCH_CHECK(w13_q.dtype() == at::kByte, "w13_q must be uint8");
    TORCH_CHECK(w2_q.dtype() == at::kByte, "w2_q must be uint8");
    TORCH_CHECK(fc1_scale.dtype() == at::kByte, "fc1_scale must be uint8");
    TORCH_CHECK(fc2_scale.dtype() == at::kByte, "fc2_scale must be uint8");

    const int64_t tokens = input_q.size(0);
    const int64_t dim = input_q.size(1);
    const int64_t experts = w2_q.size(0);
    const int64_t inter_dim = w2_q.size(2) * 2;
    const int64_t max_num_m_blocks = sorted_expert_ids.numel();

    TORCH_CHECK(w2_q.size(1) == dim, "w2_q dim mismatch with input_q");
    TORCH_CHECK(w13_q.size(0) == experts, "w13_q experts mismatch with w2_q");
    TORCH_CHECK(w13_q.size(1) == 2 * inter_dim,
                "w13_q second dim must be 2*inter_dim");
    TORCH_CHECK(w13_q.size(2) * 2 == dim, "w13_q third dim must be dim/2");
    TORCH_CHECK(inter_dim % 128 == 0, "inter_dim must be divisible by 128");
    TORCH_CHECK(dim % 128 == 0, "dim must be divisible by 128");

    const int64_t w13_scale_cols = dim / 32;
    const int64_t w2_scale_cols = inter_dim / 32;
    if (fc1_scale.dim() == 3) {
        TORCH_CHECK(fc1_scale.size(0) == experts,
                    "fc1_scale experts mismatch with w13_q");
        TORCH_CHECK(fc1_scale.size(1) == 2 * inter_dim,
                    "fc1_scale second dim must be 2*inter_dim");
        TORCH_CHECK(fc1_scale.size(2) == w13_scale_cols,
                    "fc1_scale third dim must be dim/32");
    } else if (fc1_scale.dim() == 2) {
        TORCH_CHECK(fc1_scale.size(0) == experts,
                    "fc1_scale experts mismatch with w13_q");
        TORCH_CHECK(fc1_scale.size(1) == (2 * inter_dim) * w13_scale_cols,
                    "fc1_scale second dim must be (2*inter_dim)*(dim/32)");
    } else {
        TORCH_CHECK(false, "fc1_scale must be rank-2 or rank-3");
    }

    if (fc2_scale.dim() == 3) {
        TORCH_CHECK(fc2_scale.size(0) == experts,
                    "fc2_scale experts mismatch with w2_q");
        TORCH_CHECK(fc2_scale.size(1) == dim,
                    "fc2_scale second dim must be dim");
        TORCH_CHECK(fc2_scale.size(2) == w2_scale_cols,
                    "fc2_scale third dim must be inter_dim/32");
    } else if (fc2_scale.dim() == 2) {
        TORCH_CHECK(fc2_scale.size(0) == experts,
                    "fc2_scale experts mismatch with w2_q");
        TORCH_CHECK(fc2_scale.size(1) == dim * w2_scale_cols,
                    "fc2_scale second dim must be dim*(inter_dim/32)");
    } else {
        TORCH_CHECK(false, "fc2_scale must be rank-2 or rank-3");
    }

    CheckInputScale(input_scale, tokens, dim);
    TORCH_CHECK(fc1_scale.is_contiguous(), "fc1_scale must be contiguous");
    TORCH_CHECK(fc2_scale.is_contiguous(), "fc2_scale must be contiguous");
    torch::Tensor out;
    if (out_opt.has_value()) {
        out = *out_opt;
        TORCH_CHECK(out.device() == input_q.device(), "out device mismatch");
        TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be bfloat16");
        TORCH_CHECK(out.dim() == 2, "out must be [tokens, dim]");
        TORCH_CHECK(out.size(0) == tokens && out.size(1) == dim,
                    "out shape mismatch");
        TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
        out.zero_();
    } else {
        auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(input_q.device());
        out = torch::zeros({tokens, dim}, options);
    }

    const int dev = input_q.get_device();
    CheckFusedMoEArchSupported(dev, "FusedMoEBlockScaleFP8MXFP4Weight");
    const auto current_stream = c10::hip::getCurrentHIPStream(dev);
    hipStream_t stream = current_stream.stream();
    if (max_num_m_blocks == 0) {
        return out;
    }

    const int err = FusedMoEBlockScaleFP8MXFP4Weight(
        reinterpret_cast<uint4 *>(out.data_ptr()),
        reinterpret_cast<const uint4 *>(input_q.data_ptr()),
        reinterpret_cast<const uint4 *>(w13_q.data_ptr()),
        reinterpret_cast<const uint4 *>(w2_q.data_ptr()),
        reinterpret_cast<const uint4 *>(sorted_token_ids.data_ptr()),
        reinterpret_cast<const uint4 *>(sorted_weights.data_ptr()),
        reinterpret_cast<const uint4 *>(sorted_expert_ids.data_ptr()),
        reinterpret_cast<const unsigned *>(num_valid_ids.data_ptr()),
        static_cast<unsigned>(topk),
        reinterpret_cast<const uint4 *>(input_scale.data_ptr()),
        reinterpret_cast<const uint4 *>(fc1_scale.data_ptr()),
        reinterpret_cast<const unsigned *>(fc2_scale.data_ptr()),
        static_cast<unsigned>(max_num_m_blocks), static_cast<unsigned>(tokens),
        static_cast<unsigned>(dim), static_cast<unsigned>(inter_dim),
        static_cast<unsigned>(experts), stream,
        static_cast<unsigned>(num_persistent_tgs));

    CheckKernelStatus(err, "FusedMoEBlockScaleFP8MXFP4Weight");
    return out;
}

} // namespace causalflow::petit::pybind
