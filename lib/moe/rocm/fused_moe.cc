#include "moe/rocm/fused_moe_config_selector.cuh"

#include <unordered_map>

namespace causalflow::petit::rocm::moe {

using Call = int (*)(FusedMoE1StageParams params);
using WorkspaceSizeCall = std::size_t (*)(unsigned, unsigned);
using TwoStage1Call = int (*)(FusedMoE2Stage1Params);
using TwoStage2Call = int (*)(FusedMoE2Stage2Params);

struct TwoStageRegistration {
    WorkspaceSizeCall workspace_size;
    TwoStage1Call stage1;
    TwoStage2Call stage2;
};

template <class Config>
std::size_t TwoStageWorkspaceSize(unsigned max_num_m_blocks,
                                  unsigned inter_dim) {
    if (inter_dim != Config::kInterDim)
        return 0;
    return TwoStageFusedMoEWorkspace<Config>::Bytes(max_num_m_blocks,
                                                    Config::kInterDim);
}

template <bool kPersistent, class Config>
void LaunchTwoStage1(FusedMoE2Stage1Params params, dim3 grid,
                     unsigned num_experts) {
    const auto &common = params.common;
    TwoStageFusedMoEStage1Compute<Config, kPersistent>
        <<<grid, dim3(Config::kThreads), 0, common.stream>>>(
            common.intermediate, reinterpret_cast<const uint4 *>(params.act),
            reinterpret_cast<const uint4 *>(params.w13),
            reinterpret_cast<const uint4 *>(common.sorted_token_ids),
            reinterpret_cast<const uint4 *>(common.sorted_expert_ids),
            common.num_valid_ids,
            reinterpret_cast<const uint4 *>(params.scales_act),
            reinterpret_cast<const uint4 *>(params.scales_w13), common.m,
            num_experts, common.max_num_m_blocks, params.w13_bias);
}

template <class Config> int InvokeTwoStage1(FusedMoE2Stage1Params params) {
    const auto &common = params.common;
    if (common.intermediate == nullptr || common.num_valid_ids == nullptr ||
        params.act == nullptr || params.w13 == nullptr ||
        common.sorted_token_ids == nullptr ||
        common.sorted_expert_ids == nullptr || params.scales_act == nullptr ||
        params.scales_w13 == nullptr) {
        return kFusedMoEErrorInvalidArgument;
    }
    if (common.topk != Config::kTopK || common.n != Config::kDim ||
        common.k != Config::kInterDim)
        return kFusedMoEErrorInvalidArgument;
    const std::size_t required =
        TwoStageWorkspaceSize<Config>(common.max_num_m_blocks, common.k);
    if (common.intermediate_bytes < required)
        return kFusedMoEErrorInvalidArgument;
    if (common.m == 0 || common.max_num_m_blocks == 0)
        return 0;

    constexpr unsigned kNTiles = Config::kInterDim / Config::kStage1GroupN;
    const bool persistent = common.num_persistent_tgs > 0;
    unsigned route_groups = common.max_num_m_blocks;
    if (persistent) {
        unsigned workers =
            tal::CeilingDiv<unsigned>(common.num_persistent_tgs, kNTiles);
        if (workers == 0)
            workers = 1;
        route_groups = workers < route_groups ? workers : route_groups;
    }
    const unsigned num_experts =
        Config::kValidateExpertIds ? common.num_experts : 0;
    const dim3 grid(kNTiles, route_groups, 1);
    if (persistent)
        LaunchTwoStage1<true, Config>(params, grid, num_experts);
    else
        LaunchTwoStage1<false, Config>(params, grid, num_experts);
    return hipGetLastError() == hipSuccess ? 0 : kFusedMoEErrorInvalidArgument;
}

template <class Config> int InvokeTwoStage2(FusedMoE2Stage2Params params) {
    const auto &common = params.common;
    if (common.intermediate == nullptr || params.out == nullptr ||
        params.w2 == nullptr || common.sorted_token_ids == nullptr ||
        params.sorted_weights == nullptr ||
        common.sorted_expert_ids == nullptr ||
        common.num_valid_ids == nullptr || params.scales_w2 == nullptr) {
        return kFusedMoEErrorInvalidArgument;
    }
    if (common.topk != Config::kTopK || common.n != Config::kDim ||
        common.k != Config::kInterDim || common.n % Config::kGroupN != 0)
        return kFusedMoEErrorInvalidArgument;
    const std::size_t required =
        TwoStageWorkspaceSize<Config>(common.max_num_m_blocks, common.k);
    if (common.intermediate_bytes < required)
        return kFusedMoEErrorInvalidArgument;
    if (common.m == 0 || common.max_num_m_blocks == 0)
        return 0;

    using Kernel = TwoStageFusedMoEStage2<Config>;
    const dim3 grid(common.n / Config::kGroupN, Kernel::kPersistentWorkers, 1);
    const unsigned num_experts =
        Config::kValidateExpertIds ? common.num_experts : 0;
    TwoStageFusedMoEStage2Compute<Kernel>
        <<<grid, dim3(Config::kThreads), 0, common.stream>>>(
            reinterpret_cast<uint4 *>(params.out), common.intermediate,
            reinterpret_cast<const uint4 *>(params.w2),
            reinterpret_cast<const uint4 *>(common.sorted_token_ids),
            reinterpret_cast<const uint4 *>(params.sorted_weights),
            reinterpret_cast<const uint4 *>(common.sorted_expert_ids),
            common.num_valid_ids, common.topk, params.scales_w2, num_experts,
            common.max_num_m_blocks, params.w2_bias);
    return hipGetLastError() == hipSuccess ? 0 : kFusedMoEErrorInvalidArgument;
}

template <class ConfigA, class ConfigB>
std::size_t TwoStageWorkspaceSizeEither(unsigned max_num_m_blocks,
                                        unsigned inter_dim) {
    const std::size_t size_a =
        TwoStageWorkspaceSize<ConfigA>(max_num_m_blocks, inter_dim);
    const std::size_t size_b =
        TwoStageWorkspaceSize<ConfigB>(max_num_m_blocks, inter_dim);
    return size_a > size_b ? size_a : size_b;
}

template <class ConfigA, class ConfigB>
int InvokeTwoStage1Either(FusedMoE2Stage1Params params) {
    if (params.common.topk == ConfigA::kTopK)
        return InvokeTwoStage1<ConfigA>(params);
    if (params.common.topk == ConfigB::kTopK)
        return InvokeTwoStage1<ConfigB>(params);
    return kFusedMoEErrorInvalidArgument;
}

template <class ConfigA, class ConfigB>
int InvokeTwoStage2Either(FusedMoE2Stage2Params params) {
    if (params.common.topk == ConfigA::kTopK)
        return InvokeTwoStage2<ConfigA>(params);
    if (params.common.topk == ConfigB::kTopK)
        return InvokeTwoStage2<ConfigB>(params);
    return kFusedMoEErrorInvalidArgument;
}

template <FusedMoESolutionId kBaseId, unsigned kDim, unsigned kInterDim>
void RegisterOneStageShape(std::unordered_map<unsigned long, Call> &calls) {
    static_assert(FusedMoESolutionId::IsShapeEncodable(kDim, kInterDim));
    static constexpr auto kId = kBaseId.WithShape(kDim, kInterDim);
    calls.emplace(kId.Repr(), FusedMoESolutionAdapter<kId.Repr()>::Invoke);
}

template <FusedMoESolutionId kBaseId, unsigned kDim, unsigned kInterDim,
          unsigned kTopK>
void RegisterTwoStageShape(
    std::unordered_map<unsigned long, TwoStageRegistration> &calls) {
    static_assert(FusedMoESolutionId::IsShapeEncodable(kDim, kInterDim));
    static constexpr auto kId = kBaseId.WithShape(kDim, kInterDim);
    using Config = ConfigSelector<kId, kTopK>;
    calls.emplace(kId.Repr(),
                  TwoStageRegistration{TwoStageWorkspaceSize<Config>,
                                       InvokeTwoStage1<Config>,
                                       InvokeTwoStage2<Config>});
}

template <FusedMoESolutionId kBaseId, unsigned kDim, unsigned kInterDim,
          unsigned kTopKA, unsigned kTopKB>
void RegisterTwoStageShapeEitherTopK(
    std::unordered_map<unsigned long, TwoStageRegistration> &calls) {
    static_assert(FusedMoESolutionId::IsShapeEncodable(kDim, kInterDim));
    static_assert(kTopKA != kTopKB);
    static constexpr auto kCachedId = kBaseId.WithShape(kDim, kInterDim);
    static constexpr auto kNonTemporalId = kCachedId.WithWeightLoadPolicy(
        FusedMoEWeightLoadPolicy::kNonTemporal);
    using CachedConfigA = ConfigSelector<kCachedId, kTopKA>;
    using CachedConfigB = ConfigSelector<kCachedId, kTopKB>;
    using NonTemporalConfigA = ConfigSelector<kNonTemporalId, kTopKA>;
    using NonTemporalConfigB = ConfigSelector<kNonTemporalId, kTopKB>;
    calls.emplace(
        kCachedId.Repr(),
        TwoStageRegistration{
            TwoStageWorkspaceSizeEither<CachedConfigA, CachedConfigB>,
            InvokeTwoStage1Either<CachedConfigA, CachedConfigB>,
            InvokeTwoStage2Either<CachedConfigA, CachedConfigB>});
    calls.emplace(
        kNonTemporalId.Repr(),
        TwoStageRegistration{
            TwoStageWorkspaceSizeEither<NonTemporalConfigA,
                                        NonTemporalConfigB>,
            InvokeTwoStage1Either<NonTemporalConfigA, NonTemporalConfigB>,
            InvokeTwoStage2Either<NonTemporalConfigA, NonTemporalConfigB>});
}

template <FusedMoESolutionId kBaseId>
void RegisterOneStageShapes(std::unordered_map<unsigned long, Call> &calls) {
    RegisterOneStageShape<kBaseId, 256, 256>(calls);
    RegisterOneStageShape<kBaseId, 256, 512>(calls);
    RegisterOneStageShape<kBaseId, 512, 512>(calls);
    RegisterOneStageShape<kBaseId, 4096, 1024>(calls);
    RegisterOneStageShape<kBaseId, 7168, 2048>(calls);
    RegisterOneStageShape<kBaseId, 3072, 4096>(calls);
    RegisterOneStageShape<kBaseId, 3072, 256>(calls);
    RegisterOneStageShape<kBaseId, 3072, 3072>(calls);
}

static const std::unordered_map<unsigned long, Call> kCallMap = [] {
    std::unordered_map<unsigned long, Call> calls;
    RegisterOneStageShapes<kFusedMoEBlockScaleFp8SolutionId>(calls);
    RegisterOneStageShapes<kFusedMoEFp8PetitMxFp4SolutionId>(calls);
    RegisterOneStageShapes<kFusedMoEFp8PetitMxFp4BiasSolutionId>(calls);
    RegisterOneStageShapes<kFusedMoEBf16NativeMxFp4BiasSolutionId>(calls);
    RegisterOneStageShapes<kFusedMoEMxFp4NativeMxFp4BiasSolutionId>(calls);
    return calls;
}();

static const std::unordered_map<unsigned long, TwoStageRegistration>
    kTwoStageCallMap = [] {
        std::unordered_map<unsigned long, TwoStageRegistration> calls;
        RegisterTwoStageShape<kFusedMoETwoStageMxFp4BiasSolutionId, 3072,
                              3072, 4>(calls);
        RegisterTwoStageShapeEitherTopK<
            kFusedMoETwoStageMxFp4SiluSolutionId, 7168, 2048, 8, 9>(calls);
        RegisterTwoStageShape<kFusedMoETwoStageMxFp4SiluSolutionId, 7168, 3072,
                              7>(calls);
        return calls;
    }();

int FusedMoEMatmul1Stage(FusedMoE1StageParams params,
                         unsigned long solution_id) {
    const auto it = kCallMap.find(solution_id);
    if (it != kCallMap.end()) {
        return it->second(params);
    }
    return kFusedMoEErrorInvalidSolution;
}

std::size_t FusedMoE2StageWorkspaceSize(unsigned max_num_m_blocks,
                                        unsigned inter_dim,
                                        unsigned long solution_id) {
    const auto it = kTwoStageCallMap.find(solution_id);
    return it == kTwoStageCallMap.end()
               ? 0
               : it->second.workspace_size(max_num_m_blocks, inter_dim);
}

int FusedMoEMatmul2Stage1(FusedMoE2Stage1Params params,
                          unsigned long solution_id) {
    const auto it = kTwoStageCallMap.find(solution_id);
    return it == kTwoStageCallMap.end() ? kFusedMoEErrorInvalidSolution
                                        : it->second.stage1(params);
}

int FusedMoEMatmul2Stage2(FusedMoE2Stage2Params params,
                          unsigned long solution_id) {
    const auto it = kTwoStageCallMap.find(solution_id);
    return it == kTwoStageCallMap.end() ? kFusedMoEErrorInvalidSolution
                                        : it->second.stage2(params);
}

} // namespace causalflow::petit::rocm::moe
