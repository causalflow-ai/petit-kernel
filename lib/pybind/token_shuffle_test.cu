#include "token_shuffle_test.h"

#include "moe/rocm/comm/barrier.cuh"
#include "moe/rocm/mega_moe/workspace.cuh"
#include "moe/rocm/ops/mega_moe/token_shuffle.cuh"
#include "moe/rocm/ops/mega_moe/token_shuffle_direct_push.cuh"

#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

namespace causalflow::petit::pybind {
namespace {

constexpr unsigned kNumSms = 256;
constexpr unsigned kNumWarps = 4;
constexpr unsigned kTopK = 4;
constexpr unsigned kNumExperts = 128;
constexpr unsigned kHiddenSize = 2880;
constexpr unsigned kComputeHiddenSize = 3072;
constexpr unsigned kInterDim = 3072;
constexpr unsigned kMaxTokens = 1024;
constexpr unsigned kWarpsPerPullToken = 2;
constexpr unsigned kGridSyncSlots = 3;
constexpr unsigned kRouteOutputBufferBytes =
    kTopK * kHiddenSize * sizeof(__hip_bfloat16);

template <unsigned NumRanks, bool UseEpochSync = false>
struct TokenShuffleTestConfig {
    using Self = TokenShuffleTestConfig;
    using XGpuSync = std::conditional_t<
        UseEpochSync,
        causalflow::petit::rocm::moe::EpochXGpuSync<Self>,
        causalflow::petit::rocm::moe::LegacyXGpuSync<Self>>;
    static constexpr unsigned kNumSMs = kNumSms;
    static constexpr unsigned kNumWarps =
        ::causalflow::petit::pybind::kNumWarps;
    static constexpr unsigned kThreads =
        kNumWarps * causalflow::petit::rocm::kWarpSize;
    static constexpr unsigned kTopK = ::causalflow::petit::pybind::kTopK;
    static constexpr unsigned kNumExperts =
        ::causalflow::petit::pybind::kNumExperts;
    static constexpr unsigned kProducerBlocks = kNumExperts;
    static constexpr unsigned kNumRanks = NumRanks;
    static constexpr unsigned kHiddenSize =
        ::causalflow::petit::pybind::kHiddenSize;
    static constexpr unsigned kComputeHiddenSize =
        ::causalflow::petit::pybind::kComputeHiddenSize;
    static constexpr unsigned kInterDim =
        ::causalflow::petit::pybind::kInterDim;
    static constexpr unsigned kMaxTokensPerRank =
        ::causalflow::petit::pybind::kMaxTokens;
    static constexpr unsigned kWarpsPerPullToken =
        ::causalflow::petit::pybind::kWarpsPerPullToken;
    static constexpr unsigned kGridSyncSlots =
        ::causalflow::petit::pybind::kGridSyncSlots;
    static constexpr unsigned kInputTokenBytes =
        causalflow::petit::rocm::moe::RemoteMxFp4Transport<
            kHiddenSize, kWarpsPerPullToken, kNumWarps>::kInputTokenBytes;
    static constexpr unsigned kRouteOutputBufferBytes =
        ::causalflow::petit::pybind::kRouteOutputBufferBytes;
    static constexpr auto kActDType =
        causalflow::petit::rocm::moe::FusedMoEDataType::kMxFp4;
};

template <unsigned NumRanks, bool UseEpochSync = false>
using Workspace = causalflow::petit::rocm::moe::MegaMoEWorkspace<
    TokenShuffleTestConfig<NumRanks, UseEpochSync>>;

template <unsigned NumRanks, bool UseEpochSync = false>
__global__ void
TokenShuffleKernel(void *base, unsigned rank, unsigned num_tokens,
                   std::int32_t *expert_counts, std::uint64_t *profile_cycles) {
    using Config = TokenShuffleTestConfig<NumRanks, UseEpochSync>;
    using Shuffle = causalflow::petit::rocm::moe::TokenShuffle<Config>;
    constexpr unsigned kLocalExperts = Config::kNumExperts / NumRanks;
    __shared__ typename Shuffle::Shm shared;
    Workspace<NumRanks, UseEpochSync> workspace(base, rank);
    Shuffle shuffle(num_tokens, &workspace, &shared);
    const unsigned tid = threadIdx.x;
    const unsigned wid = __builtin_amdgcn_readfirstlane(
        tid / causalflow::petit::rocm::kWarpSize);
    const unsigned wtid = tid % causalflow::petit::rocm::kWarpSize;
    unsigned long profile_start = 0;
    if (blockIdx.x == 0 && tid == 0)
        profile_start = clock64();
    shuffle.Run(blockIdx.x, tid, wid, wtid);
    if (blockIdx.x == 0 && tid == 0)
        profile_cycles[0] = clock64() - profile_start;

    // The production fused kernel performs an equivalent handoff before it
    // clears routing state. The standalone benchmark needs its own slot so a
    // following invocation cannot race a late puller from this invocation.
    causalflow::petit::rocm::moe::grid_sync<Config::kNumSMs, 2,
                                            /* kAcquirePayload */ false>(
        workspace, blockIdx.x, tid, [] { __syncthreads(); });
    if (blockIdx.x == 0 && tid < kLocalExperts) {
        expert_counts[tid] = workspace.br_.template LoadU32<
            causalflow::petit::rocm::BufferResource::kSC1Bit>(
            tid * sizeof(unsigned long),
            workspace.RecvSumCounterOffset(rank, 0));
    }
    shuffle.ResetRoutingCounters(blockIdx.x, tid);
}

template <unsigned NumRanks>
__global__ void DirectPushTokenShuffleKernel(void *base, unsigned rank,
                                             unsigned num_tokens,
                                             std::int32_t *expert_counts,
                                             std::uint64_t *profile_cycles) {
    using Config = TokenShuffleTestConfig<NumRanks>;
    using Shuffle =
        causalflow::petit::rocm::moe::DirectPushTokenShuffle<Config>;
    using Ws = Workspace<NumRanks>;
    constexpr unsigned kLocalExperts = Config::kNumExperts / NumRanks;
    __shared__ typename Shuffle::Shm shared;
    Ws workspace(base, rank);
    Shuffle shuffle(num_tokens, &workspace, &shared);
    const unsigned tid = threadIdx.x;
    unsigned long profile_start = 0;
    if (tid == 0)
        profile_start = clock64();
    const unsigned epoch =
        shuffle.Run(blockIdx.x, tid,
                    tid / causalflow::petit::rocm::kWarpSize,
                    tid % causalflow::petit::rocm::kWarpSize);
    const bool owner = blockIdx.x == 0;

    if (owner)
        shuffle.WaitForAllPayloads(epoch, tid);

    // Standalone dispatch has no compute work whose dependency waits can
    // provide completion. Join the fixed roles only for result inspection and
    // cleanup; fused MegaMoE admits compute per expert without this barrier.
    causalflow::petit::rocm::moe::grid_sync<Config::kNumSMs, 2>(
        workspace, blockIdx.x, tid, [] { __syncthreads(); });
    // Inspection after kernel completion may run on any CU. Broadcast the
    // payload acquire across the whole fixed-role grid so no CU can carry a
    // stale peer-written cache line into the inspecting kernel.
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    __syncthreads();
    if (owner && tid == 0)
        profile_cycles[0] = clock64() - profile_start;
    if (owner && tid < kLocalExperts) {
        expert_counts[tid] = workspace.br_.template LoadU32<
            causalflow::petit::rocm::BufferResource::kSC1Bit>(
            workspace.RecvSumCounterOffset(rank, tid), 0);
    }
    // All producers have now quiesced, so the owner can clear shared routing
    // state after capturing the result. This lets a following pull or
    // direct-push benchmark invocation reuse the same workspace.
    if (owner) {
        __syncthreads();
        shuffle.ResetRoutingCounters(blockIdx.x, tid);
    }
}

void HipCheck(hipError_t status, const char *what) {
    TORCH_CHECK(status == hipSuccess, what, ": ", hipGetErrorString(status));
}

void DistributedBarrier() {
    namespace py = pybind11;
    py::module_::import("torch.distributed").attr("barrier")();
}

void CheckTensor(const torch::Tensor &tensor, c10::ScalarType dtype, int device,
                 const char *name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be on a HIP device");
    TORCH_CHECK(tensor.get_device() == device, name,
                " must be on the workspace device");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

template <unsigned NumRanks> VmmSymmetricHeap::Layout TokenShuffleLayout() {
    using Ws = Workspace<NumRanks>;
    return {Ws::XGpuBarrierRecordBytes(), Ws::RankSymBufferBase(),
            Ws::RankSymBufferSlotBytes(), Ws::LocalOffsetBase(),
            Ws::kLocalBytes};
}

template <unsigned NumRanks>
void Prepare(VmmSymmetricHeap &heap, const torch::Tensor &input_rows,
             const torch::Tensor &topk_ids, const torch::Tensor &topk_weights) {
    using Config = TokenShuffleTestConfig<NumRanks>;
    using Ws = Workspace<NumRanks>;
    (void)heap.Allocate(TokenShuffleLayout<NumRanks>());

    const int device = heap.device_index();
    const auto tokens = input_rows.size(0);
    CheckTensor(input_rows, torch::kUInt8, device, "input_rows");
    CheckTensor(topk_ids, torch::kInt32, device, "topk_ids");
    CheckTensor(topk_weights, torch::kFloat32, device, "topk_weights");
    TORCH_CHECK(
        input_rows.dim() == 2 && input_rows.size(1) == Config::kInputTokenBytes,
        "input_rows must be [tokens, ", Config::kInputTokenBytes, "] uint8");
    TORCH_CHECK(tokens > 0 && tokens <= kMaxTokens,
                "token count must be in [1, ", kMaxTokens, "]");
    TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.size(0) == tokens &&
                    topk_ids.size(1) == kTopK,
                "topk_ids must be [tokens, ", kTopK, "]");
    TORCH_CHECK(topk_weights.dim() == 2 && topk_weights.size(0) == tokens &&
                    topk_weights.size(1) == kTopK,
                "topk_weights must be [tokens, ", kTopK, "]");

    c10::DeviceGuard guard(c10::Device(
        c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device)));
    const hipStream_t stream = at::hip::getCurrentHIPStream(device);
    auto heap_tensor = heap.LocalTensor();
    auto *base = static_cast<std::uint8_t *>(heap_tensor.data_ptr());
    Ws offsets(nullptr, static_cast<unsigned>(heap.rank()));
    const auto rank = static_cast<unsigned>(heap.rank());

    // Input setup is intentionally outside the run-only microbenchmark.
    HipCheck(hipStreamSynchronize(stream),
             "hipStreamSynchronize(token shuffle input)");
    DistributedBarrier();
    const auto row_bytes =
        static_cast<size_t>(tokens) * Config::kInputTokenBytes;
    const auto route_count = static_cast<size_t>(tokens) * kTopK;
    HipCheck(hipMemcpyAsync(base + offsets.InputTokensOffset(rank),
                            input_rows.data_ptr(), row_bytes,
                            hipMemcpyDeviceToDevice, stream),
             "hipMemcpyAsync(token shuffle rows)");
    HipCheck(hipMemcpyAsync(base + offsets.InputTokenTopKExpertIDOffset(),
                            topk_ids.data_ptr(),
                            route_count * sizeof(std::int32_t),
                            hipMemcpyDeviceToDevice, stream),
             "hipMemcpyAsync(token shuffle expert ids)");
    HipCheck(
        hipMemcpyAsync(base + offsets.InputTokenTopKExpertWeightOffset(rank),
                       topk_weights.data_ptr(), route_count * sizeof(float),
                       hipMemcpyDeviceToDevice, stream),
        "hipMemcpyAsync(token shuffle route weights)");
    HipCheck(hipStreamSynchronize(stream),
             "hipStreamSynchronize(token shuffle copies)");
    DistributedBarrier();
}

template <unsigned NumRanks>
void PrepareDirectPush(VmmSymmetricHeap &heap, const torch::Tensor &input_rows,
                       const torch::Tensor &topk_ids,
                       const torch::Tensor &topk_weights) {
    using Config = TokenShuffleTestConfig<NumRanks>;
    using Ws = Workspace<NumRanks>;
    (void)heap.Allocate(TokenShuffleLayout<NumRanks>());

    const int device = heap.device_index();
    const auto tokens = input_rows.size(0);
    CheckTensor(input_rows, torch::kUInt8, device, "input_rows");
    CheckTensor(topk_ids, torch::kInt32, device, "topk_ids");
    CheckTensor(topk_weights, torch::kFloat32, device, "topk_weights");
    TORCH_CHECK(
        input_rows.dim() == 2 && input_rows.size(1) == Config::kInputTokenBytes,
        "input_rows must be [tokens, ", Config::kInputTokenBytes, "] uint8");
    TORCH_CHECK(tokens > 0 && tokens <= kMaxTokens,
                "token count must be in [1, ", kMaxTokens, "]");
    TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.size(0) == tokens &&
                    topk_ids.size(1) == kTopK,
                "topk_ids must be [tokens, ", kTopK, "]");
    TORCH_CHECK(topk_weights.dim() == 2 && topk_weights.size(0) == tokens &&
                    topk_weights.size(1) == kTopK,
                "topk_weights must be [tokens, ", kTopK, "]");

    c10::DeviceGuard guard(c10::Device(
        c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device)));
    const hipStream_t stream = at::hip::getCurrentHIPStream(device);
    auto heap_tensor = heap.LocalTensor();
    auto *base = static_cast<std::uint8_t *>(heap_tensor.data_ptr());
    Ws offsets(nullptr, static_cast<unsigned>(heap.rank()));
    const auto rank = static_cast<unsigned>(heap.rank());

    HipCheck(hipStreamSynchronize(stream),
             "hipStreamSynchronize(direct push input)");
    DistributedBarrier();
    HipCheck(hipMemcpyAsync(
                 base + offsets.InputTokensOffset(rank), input_rows.data_ptr(),
                 static_cast<size_t>(tokens) * Config::kInputTokenBytes,
                 hipMemcpyDeviceToDevice, stream),
             "hipMemcpyAsync(direct push rows)");
    HipCheck(hipMemcpyAsync(base + offsets.InputTokenTopKExpertIDOffset(),
                            topk_ids.data_ptr(),
                            static_cast<size_t>(tokens) * kTopK *
                                sizeof(std::int32_t),
                            hipMemcpyDeviceToDevice, stream),
             "hipMemcpyAsync(direct push expert ids)");
    HipCheck(
        hipMemcpyAsync(base + offsets.InputTokenTopKExpertWeightOffset(rank),
                       topk_weights.data_ptr(),
                       static_cast<size_t>(tokens) * kTopK * sizeof(float),
                       hipMemcpyDeviceToDevice, stream),
        "hipMemcpyAsync(direct push route weights)");
    HipCheck(hipStreamSynchronize(stream),
             "hipStreamSynchronize(direct push copies)");
    DistributedBarrier();
}

template <unsigned NumRanks, bool UseEpochSync = false>
void Launch(VmmSymmetricHeap &heap, int64_t tokens,
            const torch::Tensor &expert_counts,
            const torch::Tensor &profile_cycles) {
    using Config = TokenShuffleTestConfig<NumRanks, UseEpochSync>;
    constexpr unsigned kLocalExperts = kNumExperts / NumRanks;
    TORCH_CHECK(tokens > 0 && tokens <= kMaxTokens,
                "token count must be in [1, ", kMaxTokens, "]");
    CheckTensor(expert_counts, torch::kInt32, heap.device_index(),
                "expert_counts");
    TORCH_CHECK(expert_counts.numel() == kLocalExperts,
                "expert_counts has the wrong size");
    CheckTensor(profile_cycles, torch::kInt64, heap.device_index(),
                "profile_cycles");
    TORCH_CHECK(profile_cycles.numel() == 1,
                "profile_cycles must have one element");
    c10::DeviceGuard guard(c10::Device(
        c10::DeviceType::CUDA,
        static_cast<c10::DeviceIndex>(heap.device_index())));
    const hipStream_t stream =
        at::hip::getCurrentHIPStream(heap.device_index());
    auto heap_tensor = heap.LocalTensor();
    hipLaunchKernelGGL(
        (TokenShuffleKernel<NumRanks, UseEpochSync>), dim3(kNumSms),
        dim3(Config::kThreads), 0, stream, heap_tensor.data_ptr(),
        static_cast<unsigned>(heap.rank()), static_cast<unsigned>(tokens),
        expert_counts.data_ptr<std::int32_t>(),
        reinterpret_cast<std::uint64_t *>(
            profile_cycles.data_ptr<std::int64_t>()));
    HipCheck(hipGetLastError(), "TokenShuffleKernel launch");
}

template <unsigned NumRanks>
void LaunchDirectPush(VmmSymmetricHeap &heap, int64_t tokens,
                      const torch::Tensor &expert_counts,
                      const torch::Tensor &profile_cycles) {
    using Config = TokenShuffleTestConfig<NumRanks>;
    constexpr unsigned kLocalExperts = kNumExperts / NumRanks;
    TORCH_CHECK(tokens > 0 && tokens <= kMaxTokens,
                "token count must be in [1, ", kMaxTokens, "]");
    CheckTensor(expert_counts, torch::kInt32, heap.device_index(),
                "expert_counts");
    TORCH_CHECK(expert_counts.numel() == kLocalExperts,
                "expert_counts has the wrong size");
    CheckTensor(profile_cycles, torch::kInt64, heap.device_index(),
                "profile_cycles");
    TORCH_CHECK(profile_cycles.numel() == 1,
                "profile_cycles must have one element");
    c10::DeviceGuard guard(c10::Device(
        c10::DeviceType::CUDA,
        static_cast<c10::DeviceIndex>(heap.device_index())));
    const hipStream_t stream =
        at::hip::getCurrentHIPStream(heap.device_index());
    auto heap_tensor = heap.LocalTensor();
    hipLaunchKernelGGL(
        (DirectPushTokenShuffleKernel<NumRanks>), dim3(kNumSms),
        dim3(Config::kThreads), 0, stream, heap_tensor.data_ptr(),
        static_cast<unsigned>(heap.rank()), static_cast<unsigned>(tokens),
        expert_counts.data_ptr<std::int32_t>(),
        reinterpret_cast<std::uint64_t *>(
            profile_cycles.data_ptr<std::int64_t>()));
    HipCheck(hipGetLastError(), "DirectPushTokenShuffleKernel launch");
}

template <unsigned NumRanks, bool UseEpochSync = false>
pybind11::tuple LaunchChecked(VmmSymmetricHeap &heap, int64_t tokens) {
    using Config = TokenShuffleTestConfig<NumRanks, UseEpochSync>;
    using Ws = Workspace<NumRanks, UseEpochSync>;
    constexpr unsigned kLocalExperts = kNumExperts / NumRanks;
    constexpr int64_t kMaxPoolTokens =
        static_cast<int64_t>(Ws::kMaxPoolBlocks) * 32;
    auto options = torch::TensorOptions()
                       .device(torch::Device(torch::kCUDA, heap.device_index()))
                       .dtype(torch::kInt32);
    auto expert_counts = torch::empty({kLocalExperts}, options);
    auto profile_cycles = torch::empty({1}, options.dtype(torch::kInt64));
    Launch<NumRanks, UseEpochSync>(heap, tokens, expert_counts,
                                   profile_cycles);

    auto heap_tensor = heap.LocalTensor();
    auto *base = static_cast<std::uint8_t *>(heap_tensor.data_ptr());
    Ws offsets(nullptr, static_cast<unsigned>(heap.rank()));
    const auto rank = static_cast<unsigned>(heap.rank());
    auto device_options = torch::TensorOptions().device(
        torch::Device(torch::kCUDA, heap.device_index()));
    auto rows = torch::from_blob(
        base + offsets.L1TokenBufferOffset(rank, 0),
        {kMaxPoolTokens, static_cast<int64_t>(Config::kInputTokenBytes)},
        device_options.dtype(torch::kUInt8));
    auto weights = torch::from_blob(
        base + offsets.L1TokenWeightsOffset(rank, 0), {kMaxPoolTokens},
        device_options.dtype(torch::kFloat32));
    auto metadata =
        torch::from_blob(base + offsets.TokenMetadataOffset(rank, 0),
                         {kMaxPoolTokens}, device_options.dtype(torch::kInt64));
    return pybind11::make_tuple(rows, weights, metadata, expert_counts);
}

template <unsigned NumRanks>
pybind11::tuple LaunchCheckedDirectPush(VmmSymmetricHeap &heap,
                                        int64_t tokens) {
    using Config = TokenShuffleTestConfig<NumRanks>;
    using Ws = Workspace<NumRanks>;
    constexpr unsigned kLocalExperts = kNumExperts / NumRanks;
    constexpr int64_t kMaxPoolTokens =
        static_cast<int64_t>(Ws::kMaxPoolBlocks) * 32;
    auto options = torch::TensorOptions()
                       .device(torch::Device(torch::kCUDA, heap.device_index()))
                       .dtype(torch::kInt32);
    auto expert_counts = torch::empty({kLocalExperts}, options);
    auto profile_cycles = torch::empty({1}, options.dtype(torch::kInt64));
    LaunchDirectPush<NumRanks>(heap, tokens, expert_counts, profile_cycles);

    auto heap_tensor = heap.LocalTensor();
    auto *base = static_cast<std::uint8_t *>(heap_tensor.data_ptr());
    Ws offsets(nullptr, static_cast<unsigned>(heap.rank()));
    const auto rank = static_cast<unsigned>(heap.rank());
    auto device_options = torch::TensorOptions().device(
        torch::Device(torch::kCUDA, heap.device_index()));
    auto rows = torch::from_blob(
        base + offsets.L1TokenBufferOffset(rank, 0),
        {kMaxPoolTokens, static_cast<int64_t>(Config::kInputTokenBytes)},
        device_options.dtype(torch::kUInt8));
    auto weights = torch::from_blob(
        base + offsets.L1TokenWeightsOffset(rank, 0), {kMaxPoolTokens},
        device_options.dtype(torch::kFloat32));
    auto metadata =
        torch::from_blob(base + offsets.TokenMetadataOffset(rank, 0),
                         {kMaxPoolTokens}, device_options.dtype(torch::kInt64));
    return pybind11::make_tuple(rows, weights, metadata, expert_counts);
}

} // namespace

void PrepareTokenShuffleMxFp4(VmmSymmetricHeap &workspace,
                              const torch::Tensor &input_rows,
                              const torch::Tensor &topk_ids,
                              const torch::Tensor &topk_weights) {
    switch (workspace.world_size()) {
    case 2:
        return Prepare<2>(workspace, input_rows, topk_ids, topk_weights);
    case 4:
        return Prepare<4>(workspace, input_rows, topk_ids, topk_weights);
    case 8:
        return Prepare<8>(workspace, input_rows, topk_ids, topk_weights);
    default:
        TORCH_CHECK(false, "token shuffle requires EP=2, 4, or 8");
    }
}

void RunTokenShuffleMxFp4(VmmSymmetricHeap &workspace, int64_t num_tokens,
                          const torch::Tensor &expert_counts,
                          const torch::Tensor &profile_cycles) {
    switch (workspace.world_size()) {
    case 2:
        return Launch<2>(workspace, num_tokens, expert_counts, profile_cycles);
    case 4:
        return Launch<4>(workspace, num_tokens, expert_counts, profile_cycles);
    case 8:
        return Launch<8>(workspace, num_tokens, expert_counts, profile_cycles);
    default:
        TORCH_CHECK(false, "token shuffle requires EP=2, 4, or 8");
    }
}

pybind11::tuple TokenShuffleMxFp4(VmmSymmetricHeap &workspace,
                                  const torch::Tensor &input_rows,
                                  const torch::Tensor &topk_ids,
                                  const torch::Tensor &topk_weights) {
    PrepareTokenShuffleMxFp4(workspace, input_rows, topk_ids, topk_weights);
    switch (workspace.world_size()) {
    case 2:
        return LaunchChecked<2>(workspace, input_rows.size(0));
    case 4:
        return LaunchChecked<4>(workspace, input_rows.size(0));
    case 8:
        return LaunchChecked<8>(workspace, input_rows.size(0));
    default:
        TORCH_CHECK(false, "token shuffle requires EP=2, 4, or 8");
    }
}

void PrepareEpochTokenShuffleMxFp4(VmmSymmetricHeap &workspace,
                                   const torch::Tensor &input_rows,
                                   const torch::Tensor &topk_ids,
                                   const torch::Tensor &topk_weights) {
    PrepareTokenShuffleMxFp4(workspace, input_rows, topk_ids, topk_weights);
}

void RunEpochTokenShuffleMxFp4(VmmSymmetricHeap &workspace,
                               int64_t num_tokens,
                               const torch::Tensor &expert_counts,
                               const torch::Tensor &profile_cycles) {
    switch (workspace.world_size()) {
    case 2:
        return Launch<2, true>(workspace, num_tokens, expert_counts,
                               profile_cycles);
    case 4:
        return Launch<4, true>(workspace, num_tokens, expert_counts,
                               profile_cycles);
    case 8:
        return Launch<8, true>(workspace, num_tokens, expert_counts,
                               profile_cycles);
    default:
        TORCH_CHECK(false, "epoch token shuffle requires EP=2, 4, or 8");
    }
}

pybind11::tuple EpochTokenShuffleMxFp4(VmmSymmetricHeap &workspace,
                                       const torch::Tensor &input_rows,
                                       const torch::Tensor &topk_ids,
                                       const torch::Tensor &topk_weights) {
    PrepareEpochTokenShuffleMxFp4(workspace, input_rows, topk_ids,
                                  topk_weights);
    switch (workspace.world_size()) {
    case 2:
        return LaunchChecked<2, true>(workspace, input_rows.size(0));
    case 4:
        return LaunchChecked<4, true>(workspace, input_rows.size(0));
    case 8:
        return LaunchChecked<8, true>(workspace, input_rows.size(0));
    default:
        TORCH_CHECK(false, "epoch token shuffle requires EP=2, 4, or 8");
    }
}

template <unsigned NumRanks>
void SeedEpochWrap(VmmSymmetricHeap &workspace) {
    using Ws = Workspace<NumRanks, true>;
    c10::DeviceGuard guard(c10::Device(
        c10::DeviceType::CUDA,
        static_cast<c10::DeviceIndex>(workspace.device_index())));
    const hipStream_t stream =
        at::hip::getCurrentHIPStream(workspace.device_index());
    auto heap_tensor = workspace.LocalTensor();
    auto *base = static_cast<std::uint8_t *>(heap_tensor.data_ptr());
    const auto rank = static_cast<unsigned>(workspace.rank());
    HipCheck(hipMemsetAsync(base + Ws::XGpuEpochCounterOffset(rank), 0xff,
                            sizeof(unsigned), stream),
             "hipMemsetAsync(epoch counter wrap seed)");
    HipCheck(hipMemsetAsync(base + Ws::XGpuEpochSignalOffset(rank, 0), 0xff,
                            NumRanks * sizeof(unsigned), stream),
             "hipMemsetAsync(epoch signals wrap seed)");
    HipCheck(hipStreamSynchronize(stream),
             "hipStreamSynchronize(epoch wrap seed)");
    DistributedBarrier();
}

void SeedEpochTokenShuffleWrap(VmmSymmetricHeap &workspace) {
    switch (workspace.world_size()) {
    case 2:
        return SeedEpochWrap<2>(workspace);
    case 4:
        return SeedEpochWrap<4>(workspace);
    case 8:
        return SeedEpochWrap<8>(workspace);
    default:
        TORCH_CHECK(false, "epoch token shuffle requires EP=2, 4, or 8");
    }
}

void PrepareDirectPushTokenShuffleMxFp4(VmmSymmetricHeap &workspace,
                                        const torch::Tensor &input_rows,
                                        const torch::Tensor &topk_ids,
                                        const torch::Tensor &topk_weights) {
    switch (workspace.world_size()) {
    case 2:
        return PrepareDirectPush<2>(workspace, input_rows, topk_ids,
                                    topk_weights);
    case 4:
        return PrepareDirectPush<4>(workspace, input_rows, topk_ids,
                                    topk_weights);
    case 8:
        return PrepareDirectPush<8>(workspace, input_rows, topk_ids,
                                    topk_weights);
    default:
        TORCH_CHECK(false, "direct-push token shuffle requires EP=2, 4, or 8");
    }
}

void RunDirectPushTokenShuffleMxFp4(VmmSymmetricHeap &workspace,
                                    int64_t num_tokens,
                                    const torch::Tensor &expert_counts,
                                    const torch::Tensor &profile_cycles) {
    switch (workspace.world_size()) {
    case 2:
        return LaunchDirectPush<2>(workspace, num_tokens, expert_counts,
                                   profile_cycles);
    case 4:
        return LaunchDirectPush<4>(workspace, num_tokens, expert_counts,
                                   profile_cycles);
    case 8:
        return LaunchDirectPush<8>(workspace, num_tokens, expert_counts,
                                   profile_cycles);
    default:
        TORCH_CHECK(false, "direct-push token shuffle requires EP=2, 4, or 8");
    }
}

pybind11::tuple DirectPushTokenShuffleMxFp4(VmmSymmetricHeap &workspace,
                                            const torch::Tensor &input_rows,
                                            const torch::Tensor &topk_ids,
                                            const torch::Tensor &topk_weights) {
    PrepareDirectPushTokenShuffleMxFp4(workspace, input_rows, topk_ids,
                                       topk_weights);
    switch (workspace.world_size()) {
    case 2:
        return LaunchCheckedDirectPush<2>(workspace, input_rows.size(0));
    case 4:
        return LaunchCheckedDirectPush<4>(workspace, input_rows.size(0));
    case 8:
        return LaunchCheckedDirectPush<8>(workspace, input_rows.size(0));
    default:
        TORCH_CHECK(false, "direct-push token shuffle requires EP=2, 4, or 8");
    }
}

} // namespace causalflow::petit::pybind

PYBIND11_MODULE(_test_ops, module) {
    module.def("token_shuffle_mxfp4",
               &causalflow::petit::pybind::TokenShuffleMxFp4,
               pybind11::arg("workspace"), pybind11::arg("input_rows"),
               pybind11::arg("topk_ids"), pybind11::arg("topk_weights"));
    module.def("token_shuffle_mxfp4_prepare",
               &causalflow::petit::pybind::PrepareTokenShuffleMxFp4,
               pybind11::arg("workspace"), pybind11::arg("input_rows"),
               pybind11::arg("topk_ids"), pybind11::arg("topk_weights"));
    module.def("token_shuffle_mxfp4_run",
               &causalflow::petit::pybind::RunTokenShuffleMxFp4,
               pybind11::arg("workspace"), pybind11::arg("num_tokens"),
               pybind11::arg("expert_counts"), pybind11::arg("profile_cycles"));
    module.def("token_shuffle_epoch_mxfp4",
               &causalflow::petit::pybind::EpochTokenShuffleMxFp4,
               pybind11::arg("workspace"), pybind11::arg("input_rows"),
               pybind11::arg("topk_ids"), pybind11::arg("topk_weights"));
    module.def("token_shuffle_epoch_mxfp4_prepare",
               &causalflow::petit::pybind::PrepareEpochTokenShuffleMxFp4,
               pybind11::arg("workspace"), pybind11::arg("input_rows"),
               pybind11::arg("topk_ids"), pybind11::arg("topk_weights"));
    module.def("token_shuffle_epoch_mxfp4_run",
               &causalflow::petit::pybind::RunEpochTokenShuffleMxFp4,
               pybind11::arg("workspace"), pybind11::arg("num_tokens"),
               pybind11::arg("expert_counts"), pybind11::arg("profile_cycles"));
    module.def("token_shuffle_epoch_seed_wrap",
               &causalflow::petit::pybind::SeedEpochTokenShuffleWrap,
               pybind11::arg("workspace"));
    module.def("token_shuffle_direct_push_mxfp4",
               &causalflow::petit::pybind::DirectPushTokenShuffleMxFp4,
               pybind11::arg("workspace"), pybind11::arg("input_rows"),
               pybind11::arg("topk_ids"), pybind11::arg("topk_weights"));
    module.def("token_shuffle_direct_push_mxfp4_prepare",
               &causalflow::petit::pybind::PrepareDirectPushTokenShuffleMxFp4,
               pybind11::arg("workspace"), pybind11::arg("input_rows"),
               pybind11::arg("topk_ids"), pybind11::arg("topk_weights"));
    module.def("token_shuffle_direct_push_mxfp4_run",
               &causalflow::petit::pybind::RunDirectPushTokenShuffleMxFp4,
               pybind11::arg("workspace"), pybind11::arg("num_tokens"),
               pybind11::arg("expert_counts"), pybind11::arg("profile_cycles"));
}
