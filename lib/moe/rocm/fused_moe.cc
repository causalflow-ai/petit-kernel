#include "moe/rocm/fused_moe_config_selector.cuh"

#include <unordered_map>

namespace causalflow::petit::rocm::moe {

using Call = int (*)(FusedMoE1StageParams params);

static const std::unordered_map<unsigned long, Call> kCallMap = {
    {kFusedMoEBlockScaleFp8SolutionId.Repr(),
     FusedMoESolutionAdapter<kFusedMoEBlockScaleFp8SolutionId.Repr()>::Invoke},
    {kFusedMoEFp8PetitMxFp4SolutionId.Repr(),
     FusedMoESolutionAdapter<kFusedMoEFp8PetitMxFp4SolutionId.Repr()>::Invoke},
};

int FusedMoEMatmul1Stage(FusedMoE1StageParams params,
                         unsigned long solution_id) {
    const auto it = kCallMap.find(solution_id);
    if (it != kCallMap.end()) {
        return it->second(params);
    }
    return kFusedMoEErrorInvalidSolution;
}

} // namespace causalflow::petit::rocm::moe
