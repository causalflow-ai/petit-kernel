#include "pybind.h"

#include "moe/rocm/mega_moe.h"

#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>

namespace causalflow::petit::pybind {
namespace {

using namespace causalflow::petit::rocm;
using namespace causalflow::petit::rocm::moe;

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

void CheckTensorDeviceAndType(const torch::Tensor &tensor,
                              c10::ScalarType dtype, int device,
                              const char *name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.get_device() == device, name,
                " must be on the workspace HIP device");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has invalid dtype");
}

void CheckTensor(const torch::Tensor &tensor, c10::ScalarType dtype,
                 int device, const char *name) {
    CheckTensorDeviceAndType(tensor, dtype, device, name);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

VmmSymmetricHeap::Layout
WorkspaceLayout(const MegaMoEWorkspaceInfo &info) {
    return {info.barrier_record_bytes, info.rank_sym_buffer_base,
            info.rank_slot_bytes, info.local_offset, info.local_bytes};
}

pybind11::tuple AllocateInputViews(VmmSymmetricHeap &heap,
                                   int64_t max_tokens,
                                   const MegaMoEWorkspaceInfo &info) {
    TORCH_CHECK(max_tokens > 0 && max_tokens <= info.max_tokens_per_rank,
                "invalid MegaMoE token capacity");
    (void)heap.Allocate(WorkspaceLayout(info));
    auto heap_tensor = heap.LocalTensor();
    auto *base = static_cast<unsigned char *>(heap_tensor.data_ptr());
    auto byte_options = heap_tensor.options().dtype(torch::kUInt8);
    auto id_options = heap_tensor.options().dtype(torch::kInt32);
    auto weight_options = heap_tensor.options().dtype(torch::kFloat32);
    torch::Tensor tokens;
    pybind11::object scales = pybind11::none();
    if (info.act_dtype == FusedMoEDataType::kBf16) {
        tokens = torch::from_blob(
            base + info.input_tokens_offset,
            {max_tokens, static_cast<int64_t>(info.hidden_size)},
            {static_cast<int64_t>(info.compute_hidden_size), 1},
            heap_tensor.options().dtype(torch::kBFloat16));
    } else {
        tokens = torch::from_blob(
            base + info.input_tokens_offset,
            {max_tokens, static_cast<int64_t>(info.hidden_size / 2)},
            {static_cast<int64_t>(info.input_token_bytes), 1}, byte_options);
        scales = pybind11::cast(torch::from_blob(
            base + info.input_tokens_offset + info.hidden_size / 2,
            {max_tokens, static_cast<int64_t>(info.hidden_size / 32)},
            {static_cast<int64_t>(info.input_token_bytes), 1}, byte_options));
    }
    auto topk_ids = torch::from_blob(
        base + info.input_topk_expert_id_offset,
        {max_tokens, static_cast<int64_t>(info.topk)}, id_options);
    auto topk_weights = torch::from_blob(
        base + info.input_topk_expert_weight_offset,
        {max_tokens, static_cast<int64_t>(info.topk)}, weight_options);
    return pybind11::make_tuple(tokens, scales, topk_ids, topk_weights);
}

torch::Tensor Launch(VmmSymmetricHeap &heap, int64_t num_tokens,
                     const torch::Tensor &w13, const torch::Tensor &w2,
                     const torch::Tensor &scales_w13,
                     const torch::Tensor &scales_w2,
                     uint64_t solution_id,
                     const MegaMoEWorkspaceInfo &info,
                     const std::optional<torch::Tensor> &w13_bias,
                     const std::optional<torch::Tensor> &w2_bias,
                     const std::optional<torch::Tensor> &output,
                     const std::optional<torch::Tensor> &input_tokens,
                     const std::optional<torch::Tensor> &input_topk_ids,
                     const std::optional<torch::Tensor> &input_topk_weights) {
    const int device = heap.device_index();
    CheckTensor(w13, torch::kUInt8, device, "w13");
    CheckTensor(w2, torch::kUInt8, device, "w2");
    CheckTensor(scales_w13, torch::kUInt8, device, "fc1_scale");
    CheckTensor(scales_w2, torch::kUInt8, device, "fc2_scale");
    TORCH_CHECK(num_tokens >= 0 && num_tokens <= info.max_tokens_per_rank,
                "invalid MegaMoE token count");
    const unsigned local_experts = info.num_experts / info.num_ranks;
    TORCH_CHECK(w13.dim() == 3 && w2.dim() == 3 &&
                    w13.size(0) == local_experts &&
                    w2.size(0) == local_experts,
                "weights must contain the rank-local experts");
    const unsigned inter_dim = static_cast<unsigned>(w2.size(2) * 2);
    TORCH_CHECK(inter_dim % 512 == 0,
                "intermediate dimension must be 512-aligned");
    TORCH_CHECK(w2.size(1) == info.compute_hidden_size &&
                    w13.size(1) == 2 * inter_dim &&
                    w13.size(2) * 2 == info.compute_hidden_size,
                "invalid MegaMoE weight shapes");
    TORCH_CHECK(!heap.Allocate(WorkspaceLayout(info)),
                "request MegaMoE input views before launch");

    c10::DeviceGuard guard(c10::Device(
        c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device)));
    const hipStream_t stream = at::hip::getCurrentHIPStream(device);
    auto heap_tensor = heap.LocalTensor();
    void *base = heap_tensor.data_ptr();
    const unsigned rank = static_cast<unsigned>(heap.rank());
    torch::Tensor out = output ? *output : torch::empty(
        {num_tokens, static_cast<int64_t>(info.compute_hidden_size)},
        w13.options().dtype(torch::kBFloat16));
    CheckTensor(out, torch::kBFloat16, device, "out");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == num_tokens &&
                    (out.size(1) == info.hidden_size ||
                     out.size(1) == info.compute_hidden_size),
                "out must be [num_tokens, hidden_size] or "
                "[num_tokens, compute_hidden_size]");
    if (w13_bias) {
        CheckTensor(*w13_bias, torch::kBFloat16, device, "w13_bias");
    }
    if (w2_bias) {
        CheckTensor(*w2_bias, torch::kBFloat16, device, "w2_bias");
    }
    const void *bias13 = w13_bias ? w13_bias->data_ptr() : nullptr;
    const void *bias2 = w2_bias ? w2_bias->data_ptr() : nullptr;
    const unsigned external_count =
        static_cast<unsigned>(input_tokens.has_value()) +
        static_cast<unsigned>(input_topk_ids.has_value()) +
        static_cast<unsigned>(input_topk_weights.has_value());
    TORCH_CHECK(external_count == 0 || external_count == 3,
                "external MegaMoE inputs must be provided together");
    if (external_count == 3) {
        CheckTensorDeviceAndType(*input_tokens, torch::kUInt8, device,
                                 "input_tokens");
        CheckTensor(*input_topk_ids, torch::kInt32, device, "input_topk_ids");
        CheckTensor(*input_topk_weights, torch::kFloat32, device,
                    "input_topk_weights");
        TORCH_CHECK(input_tokens->dim() == 2 &&
                        input_tokens->size(0) == num_tokens &&
                        input_tokens->size(1) == info.hidden_size / 2,
                    "input_tokens must be [num_tokens, hidden_size / 2]");
        TORCH_CHECK(input_tokens->stride(1) == 1 &&
                        input_tokens->stride(0) == info.input_token_bytes,
                    "input_tokens rows must include aligned scale storage");
        TORCH_CHECK(input_topk_ids->dim() == 2 &&
                        input_topk_ids->size(0) == num_tokens &&
                        input_topk_ids->size(1) == info.topk,
                    "input_topk_ids must be [num_tokens, topk]");
        TORCH_CHECK(input_topk_weights->sizes() == input_topk_ids->sizes(),
                    "input_topk_weights must match input_topk_ids");
    }

    MegaMoEParams params{
        reinterpret_cast<unsigned *>(out.data_ptr()),
        static_cast<unsigned>(out.stride(0)),
        reinterpret_cast<const unsigned *>(w13.data_ptr()),
        reinterpret_cast<const unsigned *>(w2.data_ptr()),
        reinterpret_cast<const unsigned *>(scales_w13.data_ptr()),
        reinterpret_cast<const unsigned *>(scales_w2.data_ptr()),
        input_tokens
            ? reinterpret_cast<const unsigned *>(input_tokens->data_ptr())
            : nullptr,
        input_topk_ids
            ? reinterpret_cast<const unsigned *>(input_topk_ids->data_ptr())
            : nullptr,
        input_topk_weights
            ? reinterpret_cast<const float *>(input_topk_weights->data_ptr())
            : nullptr,
        static_cast<unsigned>(num_tokens),
        info.compute_hidden_size,
        inter_dim,
        bias13,
        bias2,
        base,
        rank,
        stream,
    };
    CheckKernelStatus(MegaMoECompute(params, solution_id), "MegaMoE");
    return out;
}

MegaMoEWorkspaceInfo LookupSolution(VmmSymmetricHeap &heap,
                                    uint64_t solution_id) {
    MegaMoEWorkspaceInfo info{};
    const int err = GetMegaMoEWorkspaceInfo(
        static_cast<unsigned>(heap.rank()), solution_id, &info);
    TORCH_CHECK(err != kFusedMoEErrorInvalidSolution,
                "unsupported MegaMoE solution_id");
    CheckKernelStatus(err, "MegaMoE workspace");
    TORCH_CHECK(static_cast<unsigned>(heap.world_size()) == info.num_ranks,
                "MegaMoE solution rank count does not match the workspace");
    return info;
}

} // namespace

pybind11::tuple MegaMoeWorkspaceInputViews(
    VmmSymmetricHeap &heap, int64_t max_tokens, uint64_t solution_id) {
    return AllocateInputViews(heap, max_tokens,
                              LookupSolution(heap, solution_id));
}

pybind11::tuple MegaMoeQuantizeMxFp4(
    const torch::Tensor &input,
    const std::optional<torch::Tensor> &output_arg,
    const std::optional<torch::Tensor> &output_scales_arg) {
    TORCH_CHECK(input.is_cuda(), "input must be on a HIP device");
    const int device = input.get_device();
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
                "input has invalid dtype");
    TORCH_CHECK(input.dim() == 2 && input.size(1) > 0 &&
                    input.size(1) % 32 == 0,
                "input must be [num_tokens, hidden_size] with a 32-aligned "
                "hidden size");
    TORCH_CHECK(input.stride(1) == 1 && input.stride(0) % 8 == 0,
                "input rows must be contiguous and 16-byte aligned");

    c10::DeviceGuard guard(c10::Device(
        c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device)));
    const auto rows = input.size(0);
    const auto cols = input.size(1);
    const auto value_row_bytes = cols / 2;
    const auto scale_cols = cols / 32;
    const auto scale_row_bytes = (scale_cols + 15) & ~int64_t{15};
    const auto output_row_bytes = value_row_bytes + scale_row_bytes;
    TORCH_CHECK(output_arg.has_value() == output_scales_arg.has_value(),
                "output and output_scales must be provided together");

    torch::Tensor output;
    torch::Tensor scales;
    if (output_arg) {
        output = *output_arg;
        scales = *output_scales_arg;
        CheckTensorDeviceAndType(output, torch::kUInt8, device, "output");
        CheckTensorDeviceAndType(scales, torch::kUInt8, device,
                                 "output_scales");
        TORCH_CHECK(output.dim() == 2 && output.size(0) == rows &&
                        output.size(1) == value_row_bytes,
                    "output must be [num_tokens, hidden_size / 2]");
        TORCH_CHECK(scales.dim() == 2 && scales.size(0) == rows &&
                        scales.size(1) == scale_cols,
                    "output_scales must be [num_tokens, hidden_size / 32]");
        TORCH_CHECK(output.stride(1) == 1 && scales.stride(1) == 1 &&
                        output.stride(0) == output_row_bytes &&
                        scales.stride(0) == output_row_bytes,
                    "output rows must include aligned scale storage");
        TORCH_CHECK(rows == 0 ||
                        scales.data_ptr() ==
                            static_cast<unsigned char *>(output.data_ptr()) +
                                value_row_bytes,
                    "output_scales must be a view into the output rows");
    } else {
        auto storage = torch::empty({rows, output_row_bytes},
                                    input.options().dtype(torch::kUInt8));
        output = storage.narrow(1, 0, value_row_bytes);
        scales = storage.narrow(1, value_row_bytes, scale_cols);
    }
    CheckKernelStatus(
        moe::MegaMoEQuantizeMxFp4(
            input.data_ptr(), static_cast<unsigned char *>(output.data_ptr()),
            static_cast<unsigned>(rows), static_cast<unsigned>(cols),
            static_cast<unsigned>(input.stride(0)),
            at::hip::getCurrentHIPStream(device)),
        "MegaMoE MXFP4 quantizer");
    return pybind11::make_tuple(output, scales);
}

torch::Tensor MegaMoe(
    VmmSymmetricHeap &heap, const torch::Tensor &w13,
    const torch::Tensor &w2, const torch::Tensor &scales_w13,
    const torch::Tensor &scales_w2, int64_t num_tokens, uint64_t solution_id,
    const std::optional<torch::Tensor> &w13_bias,
    const std::optional<torch::Tensor> &w2_bias,
    const std::optional<torch::Tensor> &out,
    const std::optional<torch::Tensor> &input_tokens,
    const std::optional<torch::Tensor> &input_topk_ids,
    const std::optional<torch::Tensor> &input_topk_weights) {
    return Launch(heap, num_tokens, w13, w2, scales_w13, scales_w2,
                  solution_id, LookupSolution(heap, solution_id), w13_bias,
                  w2_bias, out, input_tokens, input_topk_ids,
                  input_topk_weights);
}

} // namespace causalflow::petit::pybind
