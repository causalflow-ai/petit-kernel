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
        unsigned pool_row;
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
            const unsigned blocks =
                tal::CeilingDiv(value.x, kSortedTokenBlock);
            const unsigned inclusive =
                amdgcn_wave_inclusive_add(blocks, wtid);
            const unsigned block_base = inclusive - blocks;
            expert_metadata_[i] =
                value.x | (block_base << kTokenCountBits);
        }

    }

    TAL_DEVICE bool GetWork(unsigned wtid, unsigned logical_id,
                            Work *work) const {
        static_assert(kNumExpertsPerLane == 1,
                      "MegaMoE supports at most one local expert per lane");
        const unsigned last_metadata =
            __shfl(expert_metadata_[0], kNumExpertsPerRank - 1);
        const unsigned last_tokens = last_metadata & kTokenCountMask;
        const unsigned total_blocks =
            (last_metadata >> kTokenCountBits) +
            tal::CeilingDiv(last_tokens, kSortedTokenBlock);

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

        const unsigned lane_metadata = expert_metadata_[0];
        const unsigned lane_tokens = lane_metadata & kTokenCountMask;
        const unsigned lane_base = lane_metadata >> kTokenCountBits;
        const unsigned lane_blocks =
            tal::CeilingDiv(lane_tokens, kSortedTokenBlock);
        const unsigned long long matches =
            __ballot(wtid < kNumExpertsPerRank && lane_base <= target_block &&
                     target_block < lane_base + lane_blocks) &
            kExpertLaneMask;
        if (matches == 0)
            return false;
        const unsigned expert =
            static_cast<unsigned>(__builtin_ctzll(matches));
        const unsigned metadata = __shfl(lane_metadata, expert);
        const unsigned tokens = metadata & kTokenCountMask;
        const unsigned block_base = metadata >> kTokenCountBits;
        const unsigned block_in_expert = target_block - block_base;
        *work = {
            phase,
            expert,
            target_block,
            target_block * kSortedTokenBlock,
            min(kSortedTokenBlock,
                tokens - block_in_expert * kSortedTokenBlock),
            phase_id - target_block * tiles_per_block,
        };
        return true;
    }

    // Stage 1 may consume two adjacent M32 pool blocks at once.  The direct
    // push layout remains padded per expert in M32 units so stage 2 can still
    // consume each half independently; reconstruct both the coarser stage-1
    // ticket and its physical row base here.
    TAL_DEVICE bool GetStage1Work(unsigned wtid, unsigned logical_id,
                                  Work *work) const {
        static constexpr unsigned kStage1M = Config::kGroupM;
        static_assert(kStage1M == 32 || kStage1M == 64);
        const unsigned target_stage1_block = logical_id / kLinear1Tiles;
        const unsigned tile = logical_id % kLinear1Tiles;
        if constexpr (kStage1M == kSortedTokenBlock) {
            const unsigned lane_metadata = expert_metadata_[0];
            const unsigned lane_tokens = lane_metadata & kTokenCountMask;
            const unsigned lane_base = lane_metadata >> kTokenCountBits;
            const unsigned lane_blocks =
                tal::CeilingDiv(lane_tokens, kSortedTokenBlock);
            const unsigned long long matches =
                __ballot(wtid < kNumExpertsPerRank &&
                         lane_base <= target_stage1_block &&
                         target_stage1_block < lane_base + lane_blocks) &
                kExpertLaneMask;
            if (matches == 0)
                return false;
            const unsigned expert =
                static_cast<unsigned>(__builtin_ctzll(matches));
            const unsigned metadata = __shfl(lane_metadata, expert);
            const unsigned tokens = metadata & kTokenCountMask;
            const unsigned block_base = metadata >> kTokenCountBits;
            const unsigned block_in_expert =
                target_stage1_block - block_base;
            *work = {
                MegaMoEBlockPhase::kLinear1,
                expert,
                target_stage1_block,
                target_stage1_block * kSortedTokenBlock,
                min(kStage1M,
                    tokens - block_in_expert * kStage1M),
                tile,
            };
            return true;
        }
        unsigned stage1_block_base = 0;
        unsigned physical_block_base = 0;
#pragma unroll 1
        for (unsigned expert = 0; expert < kNumExpertsPerRank; ++expert) {
            const unsigned metadata = __shfl(expert_metadata_[0], expert);
            const unsigned tokens = metadata & kTokenCountMask;
            const unsigned stage1_blocks =
                tal::CeilingDiv(tokens, kStage1M);
            const unsigned physical_blocks =
                tal::CeilingDiv(tokens, kSortedTokenBlock);
            if (stage1_block_base <= target_stage1_block &&
                target_stage1_block < stage1_block_base + stage1_blocks) {
                const unsigned block_in_expert =
                    target_stage1_block - stage1_block_base;
                const unsigned pool_block =
                    physical_block_base +
                    block_in_expert * (kStage1M / kSortedTokenBlock);
                *work = {
                    MegaMoEBlockPhase::kLinear1,
                    expert,
                    pool_block,
                    pool_block * kSortedTokenBlock,
                    min(kStage1M, tokens - block_in_expert * kStage1M),
                    tile,
                };
                return true;
            }
            stage1_block_base += stage1_blocks;
            physical_block_base += physical_blocks;
        }
        return false;
    }

    TAL_DEVICE bool GetStage2Work(unsigned wtid, unsigned stage2_id,
                                  Work *work) const {
        const unsigned last_metadata =
            __shfl(expert_metadata_[0], kNumExpertsPerRank - 1);
        const unsigned last_tokens = last_metadata & kTokenCountMask;
        const unsigned total_blocks =
            (last_metadata >> kTokenCountBits) +
            tal::CeilingDiv(last_tokens, kSortedTokenBlock);
        return GetWork(wtid, total_blocks * kLinear1Tiles + stage2_id, work);
    }

  private:
    static constexpr unsigned kTokenCountBits = 17;
    static constexpr unsigned kTokenCountMask =
        (1u << kTokenCountBits) - 1u;
    static constexpr unsigned long long kExpertLaneMask =
        kNumExpertsPerRank == kWarpSize
            ? ~0ull
            : (1ull << kNumExpertsPerRank) - 1;
    Workspace *workspace_;
    unsigned expert_metadata_[kNumExpertsPerLane];
};

} // namespace causalflow::petit::rocm::moe
