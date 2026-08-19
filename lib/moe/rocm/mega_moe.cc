#include "moe/rocm/mega_moe_config_selector.cuh"

#include <unordered_map>

namespace causalflow::petit::rocm::moe {
namespace {

using WorkspaceInfoCall = int (*)(unsigned, MegaMoEWorkspaceInfo *);
using LaunchCall = int (*)(MegaMoEParams);

struct MegaMoESolutionOperations {
    WorkspaceInfoCall workspace_info;
    LaunchCall launch;
};

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
#define MEGA_MOE_P64 MegaMoEProducerGeometry::kCta64
#define MEGA_MOE_P128 MegaMoEProducerGeometry::kCta128

const std::unordered_map<unsigned long, MegaMoESolutionOperations>
    kMegaMoESolutions = {
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SolutionId, 1, 32, 4, 2880,
                          3072, MEGA_MOE_P128, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SolutionId, 2, 32, 4, 2880,
                          3072, MEGA_MOE_P128, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SolutionId, 4, 32, 4, 2880,
                          3072, MEGA_MOE_P128, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SolutionId, 8, 32, 4, 2880,
                          3072, MEGA_MOE_P128, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SolutionId, 8, 128, 4, 2880,
                          3072, MEGA_MOE_P64, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SolutionId, 8, 128, 4, 2880,
                          3072, MEGA_MOE_P128, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SiluSolutionId, 8, 256, 8,
                          7168, 2048, MEGA_MOE_P64, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SiluSolutionId, 8, 256, 8,
                          7168, 2048, MEGA_MOE_P128, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SiluSolutionId, 8, 384, 6,
                          7168, 3072, MEGA_MOE_P64, MEGA_MOE_N256),
        MEGA_MOE_REGISTER(kMegaMoETwoStageMxFp4SiluSolutionId, 8, 384, 6,
                          7168, 3072, MEGA_MOE_P128, MEGA_MOE_N256),
};

#undef MEGA_MOE_N256
#undef MEGA_MOE_P64
#undef MEGA_MOE_P128
#undef MEGA_MOE_REGISTER
#undef MEGA_MOE_SOLUTION

} // namespace

int MegaMoECompute(MegaMoEParams params, unsigned long solution_id) {
    const auto it = kMegaMoESolutions.find(solution_id);
    if (it != kMegaMoESolutions.end()) {
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
