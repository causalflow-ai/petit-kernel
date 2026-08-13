#include "moe/rocm/fused_moe_config_selector.cuh"

#include <unordered_map>

namespace causalflow::petit::rocm::moe {

using Call = int (*)(FusedMoE1StageParams params);

template <FusedMoESolutionId kBaseId, unsigned kDim, unsigned kInterDim>
void RegisterOneStageShape(std::unordered_map<unsigned long, Call> &calls) {
    static_assert(FusedMoESolutionId::IsShapeEncodable(kDim, kInterDim));
    static constexpr auto kId = kBaseId.WithShape(kDim, kInterDim);
    calls.emplace(kId.Repr(), FusedMoESolutionAdapter<kId.Repr()>::Invoke);
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

int FusedMoEMatmul1Stage(FusedMoE1StageParams params,
                         unsigned long solution_id) {
    const auto it = kCallMap.find(solution_id);
    if (it != kCallMap.end()) {
        return it->second(params);
    }
    return kFusedMoEErrorInvalidSolution;
}

} // namespace causalflow::petit::rocm::moe
