#pragma once

#include "gemm/rocm/amd_intrinsics.cuh"
#include "moe/rocm/mega_moe/workspace.cuh"

namespace causalflow::petit::rocm::moe {

// Operations whose implementation is identical for pull and direct-push
// token shuffle. Protocol-specific counter values, route placement, and
// synchronization remain in the two shuffle implementations.
template <class Config> struct TokenShuffleCommon {
    using Workspace = MegaMoEWorkspace<Config>;

    static constexpr unsigned kThreads = Config::kThreads;
    static constexpr unsigned kTopK = Config::kTopK;
    static constexpr unsigned kNumExperts = Config::kNumExperts;
    static constexpr unsigned kNumRanks = Config::kNumRanks;
    static constexpr unsigned kExpertsPerRank = kNumExperts / kNumRanks;

    TAL_DEVICE __forceinline__ static void
    ClearExpertCounts(unsigned *expert_count, unsigned tid) {
        constexpr unsigned kIterations =
            tal::CeilingDiv(kNumExperts, kThreads);
#pragma unroll
        for (unsigned i = 0; i < kIterations; ++i) {
            const unsigned expert = tid + i * kThreads;
            if (expert < kNumExperts)
                expert_count[expert] = 0;
        }
        __syncthreads();
    }

    TAL_DEVICE __forceinline__ static void
    PublishRecvCounter(Workspace &workspace, unsigned expert,
                       unsigned count) {
        const unsigned destination = expert / kExpertsPerRank;
        const unsigned local_expert = expert % kExpertsPerRank;
        // RecvCounter consumers use only the low count word. The complete
        // publication state is carried separately by RecvSumCounter or the
        // direct-push epoch fields.
        workspace.br_.template StoreU32<BufferResource::kAtomicScopeSystem>(
            workspace.RecvCounterOffset(destination, workspace.Rank(),
                                        local_expert),
            0, count);
    }

    TAL_DEVICE __forceinline__ static unsigned
    LoadLocalNumTokens(Workspace &workspace, unsigned lane) {
        // Every locally sent route produces one result pushed back by its
        // remote expert rank. Keep the expected receive count in the low word
        // of the source-owned send row until output reduction consumes it.
        unsigned routes = 0;
        for (unsigned expert = lane; expert < kNumExperts;
             expert += kWarpSize) {
            routes += workspace.br_
                          .template LoadU64<BufferResource::kNone>(
                              workspace.SendCounterOffset(workspace.Rank(),
                                                          expert),
                              0)
                          .x;
        }
        return __reduce_add_sync(~0ull, routes) / kTopK;
    }

    TAL_DEVICE __forceinline__ static void
    ResetRoutingCounters(Workspace &workspace, unsigned sm_id, unsigned tid) {
        // CTA 0 owns the reset after compute has quiesced.
        if (sm_id != 0)
            return;
        constexpr unsigned kExpertIterations =
            tal::CeilingDiv(kNumExperts, kThreads);
#pragma unroll
        for (unsigned i = 0; i < kExpertIterations; ++i) {
            const unsigned expert = tid + i * kThreads;
            if (expert < kNumExperts) {
                workspace.br_.template StoreU64<BufferResource::kNone>(
                    workspace.SendCounterOffset(workspace.Rank(), expert), 0,
                    {0, 0});
            }
        }
        for (unsigned i = tid; i < kNumRanks * kExpertsPerRank;
             i += kThreads) {
            workspace.br_
                .template StoreU64<BufferResource::kAtomicScopeSystem>(
                    workspace.RecvCounterOffset(workspace.Rank(),
                                                i / kExpertsPerRank,
                                                i % kExpertsPerRank),
                    0, {0, 0});
        }
        for (unsigned local_expert = tid;
             local_expert < kExpertsPerRank; local_expert += kThreads) {
            workspace.br_
                .template StoreU64<BufferResource::kAtomicScopeSystem>(
                    workspace.RecvSumCounterOffset(workspace.Rank(),
                                                   local_expert),
                    0, {0, 0});
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    }
};

} // namespace causalflow::petit::rocm::moe
