#include "moe/rocm/mega_moe_config_selector.cuh"
#include "moe/rocm/quantization.cuh"

#include <hip/hip_bfloat16.h>
#include <cstring>
#include <unordered_map>

namespace causalflow::petit::rocm::moe {
namespace {

using WorkspaceInfoCall = int (*)(unsigned, MegaMoEWorkspaceInfo *);
using LaunchCall = int (*)(MegaMoEParams);

struct MegaMoESolutionOperations {
    WorkspaceInfoCall workspace_info;
    LaunchCall launch;
};

bool IsGfx950(hipStream_t stream) {
    hipDevice_t device = 0;
    if (hipStreamGetDevice(stream, &device) != hipSuccess)
        return false;
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device) != hipSuccess)
        return false;
    return std::strncmp(props.gcnArchName, "gfx950", 6) == 0;
}

#define MEGA_MOE_SOLUTION(BASE, RANKS, EXPERTS, TOPK, HIDDEN, INTER, PRODUCERS, \
                          W2_TILE)                                              \
    BASE.WithMegaMoEConfig(RANKS, EXPERTS, TOPK, HIDDEN, INTER, PRODUCERS,      \
                           W2_TILE)                                             \
        .Repr()
#define MEGA_MOE_REGISTER(BASE, RANKS, EXPERTS, TOPK, HIDDEN, INTER, PRODUCERS, \
                          W2_TILE)                                              \
    {                                                                           \
        MEGA_MOE_SOLUTION(BASE, RANKS, EXPERTS, TOPK, HIDDEN, INTER, PRODUCERS, \
                          W2_TILE),                                             \
        {                                                                       \
            MegaMoESolutionAdapter<MEGA_MOE_SOLUTION(                           \
                BASE, RANKS, EXPERTS, TOPK, HIDDEN, INTER, PRODUCERS,           \
                W2_TILE)>::GetWorkspaceInfo,                                    \
                MegaMoESolutionAdapter<MEGA_MOE_SOLUTION(                       \
                    BASE, RANKS, EXPERTS, TOPK, HIDDEN, INTER, PRODUCERS,       \
                    W2_TILE)>::Invoke                                           \
        }                                                                       \
    }

#define MEGA_MOE_N256 MegaMoETileShape::kN256
#define MEGA_MOE_P56 MegaMoEProducerGeometry::kCta56
#define MEGA_MOE_P64 MegaMoEProducerGeometry::kCta64
#define MEGA_MOE_P128 MegaMoEProducerGeometry::kCta128
#define MEGA_MOE_P192 MegaMoEProducerGeometry::kCta192

const std::unordered_map<unsigned long, MegaMoESolutionOperations>
    kMegaMoESolutions = {
#if PETIT_COMPILE_FUSED_MOE_KERNELS
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SolutionId, 1, 32, 4, 2880,
                          3072, MEGA_MOE_P56, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SolutionId, 2, 32, 4, 2880,
                          3072, MEGA_MOE_P56, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SolutionId, 4, 32, 4, 2880,
                          3072, MEGA_MOE_P56, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SolutionId, 8, 32, 4, 2880,
                          3072, MEGA_MOE_P56, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SolutionId, 8, 128, 4, 2880,
                          3072, MEGA_MOE_P56, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SolutionId, 8, 128, 4, 2880,
                          3072, MEGA_MOE_P64, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SolutionId, 8, 128, 4, 2880,
                          3072, MEGA_MOE_P128, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SiluSolutionId, 8, 256, 8,
                          7168, 2048, MEGA_MOE_P56, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SiluSolutionId, 8, 256, 8,
                          7168, 2048, MEGA_MOE_P128, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SiluSolutionId, 8, 256, 8,
                          7168, 2048, MEGA_MOE_P192, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SiluSolutionId, 8, 384, 6,
                          7168, 3072, MEGA_MOE_P56, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SiluSolutionId, 8, 384, 6,
                          7168, 3072, MEGA_MOE_P192, MEGA_MOE_N256),
#endif
};

#undef MEGA_MOE_N256
#undef MEGA_MOE_P56
#undef MEGA_MOE_P64
#undef MEGA_MOE_P128
#undef MEGA_MOE_P192
#undef MEGA_MOE_REGISTER
#undef MEGA_MOE_SOLUTION

} // namespace

// Assign one thread to each 1x32 MX block for small token counts. This avoids
// the mostly-idle 128x32 tile used by generic quantizers when there are only a
// handful of rows.
#if PETIT_COMPILE_FUSED_MOE_KERNELS
__global__ static void
MegaMoEQuantizeMxFp4Kernel(const __hip_bfloat16 *__restrict__ input,
                           unsigned char *__restrict__ output,
                           unsigned groups_per_row, unsigned input_row_stride) {
#if defined(__gfx950__)
    const unsigned group_col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned scale_row_stride = (groups_per_row + 15) & ~15u;
    const unsigned value_row_bytes = groups_per_row * sizeof(uint4);
    const unsigned output_row_stride = value_row_bytes + scale_row_stride;
    if (group_col >= scale_row_stride)
        return;
    const unsigned row = blockIdx.y;
    auto *row_output = output + row * output_row_stride;
    auto *row_values = reinterpret_cast<uint4 *>(row_output);
    auto *row_scales = row_output + value_row_bytes;
    if (group_col >= groups_per_row) {
        row_scales[group_col] = 0;
        return;
    }
    const auto *row_input = input + row * input_row_stride;
    const auto *src = reinterpret_cast<const uint4 *>(row_input) +
                      group_col * 4;
    float4 values[8];
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
        const uint4 packed = src[chunk];
        const auto *pairs = reinterpret_cast<const __hip_bfloat162 *>(&packed);
        const float2 a = __bfloat1622float2(pairs[0]);
        const float2 b = __bfloat1622float2(pairs[1]);
        const float2 c = __bfloat1622float2(pairs[2]);
        const float2 d = __bfloat1622float2(pairs[3]);
        values[chunk * 2] = {a.x, a.y, b.x, b.y};
        values[chunk * 2 + 1] = {c.x, c.y, d.x, d.y};
    }

    float max_abs = 0.0f;
#pragma unroll
    for (unsigned vector = 0; vector < 8; ++vector) {
        max_abs = fmaxf(
            max_abs,
            NativeMxFp4Quantization::MaximumAbs(values[vector]));
    }
    if (max_abs == 0.0f) {
        row_values[group_col] = {};
        row_scales[group_col] = 0;
        return;
    }
    const float required = max_abs * (1.0f / 6.0f);
    const unsigned required_bits =
        reinterpret_cast<const unsigned &>(required);
    unsigned scale_byte = (required_bits >> 23) & 0xffu;
    if (scale_byte < 0xffu && (required_bits & 0x7fffffu))
        ++scale_byte;
    const unsigned scale_bits = scale_byte << 23;
    const MxFp4Scale scale{scale_byte,
                           reinterpret_cast<const float &>(scale_bits)};
    NativeMxFp4Quantization::Pack<8>(
        reinterpret_cast<unsigned char *>(&row_values[group_col]), values,
        scale);
    row_scales[group_col] = scale_byte;
#endif
}
#endif

int MegaMoEQuantizeMxFp4(const void *input, unsigned char *output,
                         unsigned rows, unsigned cols,
                         unsigned input_row_stride, hipStream_t stream) {
#if PETIT_COMPILE_FUSED_MOE_KERNELS
    if (cols == 0 || cols % 32 != 0 || input_row_stride < cols ||
        input_row_stride % 8 != 0 ||
        (rows != 0 && (input == nullptr || output == nullptr))) {
        return kFusedMoEErrorInvalidArgument;
    }
    if (rows == 0)
        return 0;
    if (!IsGfx950(stream))
        return kFusedMoEErrorUnsupported;
    const unsigned groups_per_row = cols / 32;
    constexpr unsigned kThreads = 64;
    const unsigned blocks = (groups_per_row + kThreads - 1) / kThreads;
    hipLaunchKernelGGL(MegaMoEQuantizeMxFp4Kernel, dim3(blocks, rows),
                       dim3(kThreads), 0, stream,
                       static_cast<const __hip_bfloat16 *>(input), output,
                       groups_per_row, input_row_stride);
    return hipGetLastError() == hipSuccess ? 0
                                           : kFusedMoEErrorInvalidArgument;
#else
    (void)input;
    (void)output;
    (void)rows;
    (void)cols;
    (void)input_row_stride;
    (void)stream;
    return kFusedMoEErrorUnsupported;
#endif
}

int MegaMoECompute(MegaMoEParams params, unsigned long solution_id) {
    const auto it = kMegaMoESolutions.find(solution_id);
    if (it != kMegaMoESolutions.end()) {
        if (!IsGfx950(params.stream))
            return kFusedMoEErrorUnsupported;
        return it->second.launch(params);
    }
    return kFusedMoEErrorInvalidSolution;
}

int GetMegaMoEWorkspaceInfo(unsigned rank, unsigned long solution_id,
                            MegaMoEWorkspaceInfo *info) {
    const auto it = kMegaMoESolutions.find(solution_id);
    if (it != kMegaMoESolutions.end()) {
        return it->second.workspace_info(rank, info);
    }
    return kFusedMoEErrorInvalidSolution;
}

} // namespace causalflow::petit::rocm::moe
