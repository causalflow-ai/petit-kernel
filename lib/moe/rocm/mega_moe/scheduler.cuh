#pragma once

#include "moe/rocm/mega_moe/workspace.cuh"

namespace causalflow::petit::rocm::moe {

template <class Config_> struct MegaMoEScheduler {
    using Config = Config_;
    using Workspace = MegaMoEWorkspace<Config>;
    static constexpr unsigned kSortedTokenBlock = Config::kSortedTokenBlock;
    static constexpr unsigned kNumExpertsPerRank =
        Config::kNumExperts / Config::kNumRanks;
    static constexpr unsigned kNumExpertsPerLane =
        tal::CeilingDiv(kNumExpertsPerRank, kWarpSize);

    TAL_DEVICE explicit MegaMoEScheduler(Workspace *workspace)
        : workspace_(workspace) {}

    TAL_DEVICE void FetchRecvSumPerExpert(unsigned wtid) {
#pragma unroll
        for (unsigned i = 0; i < kNumExpertsPerLane; ++i) {
            uint2 value{0, 0};
            const unsigned expert = i * kWarpSize + wtid;
            if (expert < kNumExpertsPerRank) {
                do {
                    value = workspace_->br_
                                .template LoadU64<BufferResource::kSC1Bit>(
                                    expert * sizeof(unsigned long),
                                    workspace_->RecvSumCounterOffset(
                                        workspace_->Rank(), 0));
                } while (value.y != Config::kNumSMs * Config::kNumRanks);
            }
            tokens_per_expert_[i] = value.x;
        }
    }

    TAL_DEVICE bool GetWork(unsigned wtid, unsigned work_id,
                            unsigned *expert_idx, unsigned *work_size) const {
        static_assert(kNumExpertsPerLane == 1,
                      "MegaMoE scheduler supports at most one expert per lane");
        unsigned next = 0;
#pragma unroll
        for (unsigned expert = 0; expert < kNumExpertsPerRank; ++expert) {
            const unsigned tokens = __shfl(tokens_per_expert_[0], expert);
            const unsigned prev = next;
            next += tal::CeilingDiv(tokens, kSortedTokenBlock);
            if (prev <= work_id && work_id < next) {
                *expert_idx = expert;
                *work_size = min(kSortedTokenBlock,
                                 tokens - (work_id - prev) * kSortedTokenBlock);
                return true;
            }
        }
        return false;
    }

  private:
    Workspace *workspace_;
    unsigned tokens_per_expert_[kNumExpertsPerLane];
};

enum class MegaMoEBlockPhase : unsigned { kLinear1, kLinear2 };

// Assign all rank-local Linear 1 blocks before Linear 2 blocks.  This is the
// one-wave form of DeepGEMM's MegaMoE scheduler: if Linear 2 work is assigned
// while Linear 1 is still running, every producer block has already been
// assigned, so polling an L2 arrival mask cannot prevent forward progress.
template <class Config_> struct MegaMoETwoStageScheduler {
    using Config = Config_;
    using Workspace = MegaMoEWorkspace<Config>;
    static constexpr unsigned kSortedTokenBlock = Config::kSortedTokenBlock;
    static constexpr unsigned kNumExpertsPerRank =
        Config::kNumExperts / Config::kNumRanks;
    static constexpr unsigned kNumExpertsPerLane =
        tal::CeilingDiv(kNumExpertsPerRank, kWarpSize);
    static constexpr unsigned kLinear1Tiles =
        Config::kInterDim / Config::kStage1GroupN;
    static constexpr unsigned kLinear2Tiles =
        Config::kComputeHiddenSize / Config::kGroupN;

    struct Work {
        MegaMoEBlockPhase phase;
        unsigned expert_idx;
        unsigned pool_block;
        unsigned work_m;
        unsigned tile;
    };

    TAL_DEVICE explicit MegaMoETwoStageScheduler(Workspace *workspace)
        : workspace_(workspace) {}

    TAL_DEVICE void FetchRecvSumPerExpert(unsigned wtid) {
#pragma unroll
        for (unsigned i = 0; i < kNumExpertsPerLane; ++i) {
            uint2 value{0, 0};
            const unsigned expert = i * kWarpSize + wtid;
            if (expert < kNumExpertsPerRank) {
                do {
                    value = workspace_->br_
                                .template LoadU64<BufferResource::kSC1Bit>(
                                    expert * sizeof(unsigned long),
                                    workspace_->RecvSumCounterOffset(
                                        workspace_->Rank(), 0));
                } while (value.y != Config::kNumSMs * Config::kNumRanks);
            }
            tokens_per_expert_[i] = value.x;
        }
    }

    TAL_DEVICE bool GetWork(unsigned wtid, unsigned logical_id,
                            Work *work) const {
        static_assert(kNumExpertsPerLane == 1,
                      "MegaMoE supports at most one local expert per lane");
        static constexpr unsigned long long kExpertLaneMask =
            kNumExpertsPerRank == kWarpSize
                ? ~0ull
                : (1ull << kNumExpertsPerRank) - 1;
        unsigned reduced_blocks = 0;
        if (wtid < kNumExpertsPerRank) {
            const unsigned lane_blocks =
                tal::CeilingDiv(tokens_per_expert_[0], kSortedTokenBlock);
            reduced_blocks =
                __reduce_add_sync(kExpertLaneMask, lane_blocks);
        }
        const unsigned total_blocks = __shfl(reduced_blocks, 0);

        const unsigned linear1_work = total_blocks * kLinear1Tiles;
        MegaMoEBlockPhase phase;
        unsigned phase_id;
        unsigned tiles_per_block;
        unsigned target_block;
        if (logical_id < linear1_work) {
            phase = MegaMoEBlockPhase::kLinear1;
            phase_id = logical_id;
            tiles_per_block = kLinear1Tiles;
            target_block = phase_id / tiles_per_block;
        } else {
            phase = MegaMoEBlockPhase::kLinear2;
            phase_id = logical_id - linear1_work;
            tiles_per_block = kLinear2Tiles;
            target_block = phase_id / tiles_per_block;
            if (phase_id >= total_blocks * tiles_per_block)
                return false;
        }

        unsigned block_base = 0;
#pragma unroll 1
        for (unsigned expert = 0; expert < kNumExpertsPerRank; ++expert) {
            const unsigned tokens = __shfl(tokens_per_expert_[0], expert);
            const unsigned expert_blocks =
                tal::CeilingDiv(tokens, kSortedTokenBlock);
            if (block_base <= target_block &&
                target_block < block_base + expert_blocks) {
                const unsigned block_in_expert = target_block - block_base;
                *work = {
                    phase,
                    expert,
                    target_block,
                    min(kSortedTokenBlock,
                        tokens - block_in_expert * kSortedTokenBlock),
                    phase_id - target_block * tiles_per_block,
                };
                return true;
            }
            block_base += expert_blocks;
        }
        return false;
    }

  private:
    Workspace *workspace_;
    unsigned tokens_per_expert_[kNumExpertsPerLane];
};

} // namespace causalflow::petit::rocm::moe
