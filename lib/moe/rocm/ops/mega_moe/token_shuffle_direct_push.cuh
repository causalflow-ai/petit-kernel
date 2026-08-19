#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "gemm/rocm/amd_intrinsics.cuh"
#include "moe/rocm/comm/barrier.cuh"
#include "moe/rocm/fused_moe.cuh"
#include "moe/rocm/mega_moe/workspace.cuh"
#include "moe/rocm/ops/mega_moe/token_shuffle_common.cuh"

#include <cstdint>

namespace causalflow::petit::rocm::moe {

// Compact planner + source-push dispatch using fixed block roles and a
// per-phase publication protocol. There is no full-grid dispatch-to-compute
// barrier: the owner publishes the local plan, producers publish each expert's
// payload epoch, and compute waits only for the expert it is about to consume.
template <class Config> struct DirectPushTokenShuffle {
    using Workspace = MegaMoEWorkspace<Config>;
    using Common = TokenShuffleCommon<Config>;
    static constexpr unsigned kNumSMs = Config::kNumSMs;
    static constexpr unsigned kThreads = Config::kThreads;
    static constexpr unsigned kTopK = Config::kTopK;
    static constexpr unsigned kNumExperts = Config::kNumExperts;
    static constexpr unsigned kNumRanks = Config::kNumRanks;
    static constexpr unsigned kExpertsPerRank = kNumExperts / kNumRanks;
    static constexpr unsigned kMaxTokens = Config::kMaxTokensPerRank;
    static constexpr unsigned kSortedTokenBlock = 32;
    static constexpr unsigned kRowVecs =
        Config::kInputTokenBytes / sizeof(uint4);
    // The caller selects a producer geometry before launch. Capping it by the
    // number of payload tasks preserves small-expert configurations.
    static constexpr unsigned kProducerBlocks = Config::kProducerBlocks;
    static constexpr unsigned kPeerCoherent =
        BufferResource::kSC0Bit | BufferResource::kSC1Bit;
    static_assert(kTopK > 0 && kTopK <= kWarpSize);
    static_assert(kNumRanks == 2 || kNumRanks == 4 || kNumRanks == 8,
                  "direct-push token shuffle supports EP2/4/8");
    static_assert(kNumExperts % kNumRanks == 0);
    static_assert(kExpertsPerRank <= kWarpSize);
    static_assert(kThreads % kWarpSize == 0);
    static_assert(kProducerBlocks % kNumRanks == 0);
    static_assert(Config::kInputTokenBytes % sizeof(uint4) == 0);
    static_assert(kNumSMs > kProducerBlocks,
                  "direct push requires one planner CTA plus its producer "
                  "grid");
    struct Shm {
        unsigned expert_count[kNumExperts];
        unsigned source_count[kNumExperts];
        unsigned generation;
        uint2 payload_plan;
    };

    TAL_DEVICE DirectPushTokenShuffle(unsigned num_tokens, Workspace *workspace,
                                      Shm *shm)
        : num_tokens_(num_tokens), workspace_(workspace), shm_(shm) {}

    TAL_DEVICE unsigned Run(unsigned block, unsigned tid, unsigned wid,
                            unsigned wtid) {
        if (tid == 0) {
            shm_->generation = static_cast<unsigned>(
                workspace_->br_
                    .template AtomicAddI32<BufferResource::kAtomicScopeAgent>(
                        workspace_->DirectPushEntryCountOffset(
                            workspace_->Rank(), block),
                        0, 1));
        }
        __syncthreads();
        const unsigned epoch = static_cast<unsigned>(shm_->generation + 1);
        current_epoch_ = epoch;
        const unsigned parity = epoch & 1;
        const unsigned expected = Expected(epoch);
        const bool owner = block == 0;
        if (owner) {
            AdmitLaunch(tid, wid, wtid, epoch);
            PopulateSendCounters(tid, wid, wtid, parity, expected);
            BuildDestinationPlan(tid, wid, wtid, parity, expected);
        } else if (block <= kProducerBlocks) {
            WaitForOwnerAdmission(epoch, tid);
            PushPayload<kProducerBlocks>(block - 1, tid, wid, wtid, parity,
                                         expected);
        }
        return epoch;
    }

    TAL_DEVICE unsigned LoadLocalNumTokens(unsigned lane) const {
        unsigned routes = 0;
        const unsigned parity = current_epoch_ & 1;
        for (unsigned expert = lane; expert < kNumExperts;
             expert += kWarpSize) {
            routes += workspace_->br_.template LoadU32<BufferResource::kNone>(
                workspace_->SendCounterOffset(workspace_->Rank(), expert) +
                    parity * sizeof(unsigned),
                0);
        }
        return __reduce_add_sync(~0ull, routes) / kTopK;
    }

    TAL_DEVICE void ResetRoutingCounters(unsigned sm_id, unsigned tid) {
        Common::ResetRoutingCounters(*workspace_, sm_id, tid);
    }

    TAL_DEVICE void WaitForExpertPayload(unsigned local_expert,
                                         unsigned epoch, unsigned tid) const {
        if (tid == 0) {
            wait_xgpu_signal_relaxed(
                *workspace_,
                workspace_->DirectPushPayloadReadyOffset(workspace_->Rank(),
                                                         epoch & 1,
                                                         local_expert),
                static_cast<std::int32_t>(Expected(epoch)));
        }
        __syncthreads();
        // Every compute wave can consume the remote payload. Acquire on each
        // wave after the owner observes readiness so no wave retains a stale
        // vector-cache line.
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
        __syncthreads();
    }

    TAL_DEVICE void WaitForLocalPlan(unsigned epoch, unsigned tid) const {
        if (tid == 0) {
            wait_xgpu_signal_relaxed(
                *workspace_,
                workspace_->DirectPushPlanReadyOffset(workspace_->Rank(),
                                                      epoch & 1,
                                                      workspace_->Rank()),
                static_cast<std::int32_t>(Expected(epoch)));
            agent_fence_acquire();
        }
        __syncthreads();
    }

    TAL_DEVICE void WaitForAllPayloads(unsigned epoch, unsigned tid) const {
        if (tid == 0) {
            for (unsigned expert = 0; expert < kExpertsPerRank; ++expert) {
                wait_xgpu_signal_relaxed(
                    *workspace_,
                    workspace_->DirectPushPayloadReadyOffset(
                        workspace_->Rank(), epoch & 1, expert),
                    static_cast<std::int32_t>(Expected(epoch)));
            }
        }
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    }

  private:
    TAL_DEVICE static unsigned Expected(unsigned epoch) {
        return ((epoch + 1) / 2) * kNumRanks;
    }

    TAL_DEVICE void StoreParityValue(unsigned offset, unsigned parity,
                                     unsigned value) const {
        workspace_->br_.template StoreU32<BufferResource::kNone>(
            offset + parity * sizeof(unsigned), 0, value);
    }

    TAL_DEVICE unsigned LoadParityValue(unsigned offset,
                                        unsigned parity) const {
        return workspace_->br_.template LoadU32<BufferResource::kNone>(
            offset + parity * sizeof(unsigned), 0);
    }

    TAL_DEVICE void AdmitLaunch(unsigned tid, unsigned wid, unsigned wtid,
                                unsigned epoch) {
        // A rank may enter the next invocation while a peer is still leaving
        // the previous one.  Admit every peer before reusing the selected
        // parity through a launch-ready handshake.
        if (wid == 0) {
            if (wtid < kNumRanks) {
                const unsigned peer =
                    (workspace_->Rank() + wtid) % kNumRanks;
                store_xgpu_epoch_release(
                    *workspace_,
                    workspace_->DirectPushLaunchReadyOffset(
                        peer, workspace_->Rank()),
                    epoch);
                wait_xgpu_epoch_relaxed(
                    *workspace_,
                    workspace_->DirectPushLaunchReadyOffset(
                        workspace_->Rank(), peer),
                    epoch);
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
            }
        }
        __syncthreads();
    }

    TAL_DEVICE void WaitForOwnerAdmission(unsigned epoch, unsigned tid) {
        if (tid == 0) {
            wait_xgpu_signal_relaxed(
                *workspace_,
                workspace_->DirectPushEpochGateOffset(workspace_->Rank()),
                static_cast<std::int32_t>(epoch));
            agent_fence_acquire();
        }
        __syncthreads();
    }

    TAL_DEVICE void CopyPayloadRow(unsigned destination, unsigned pool_index,
                                   unsigned route, unsigned vec_lane,
                                   unsigned vec_stride,
                                   bool header_owner) const {
        const unsigned source_token = route / kTopK;
        const unsigned source_row =
            workspace_->InputTokensOffset(workspace_->Rank()) +
            source_token * Config::kInputTokenBytes;
        const unsigned destination_row =
            workspace_->L1TokenBufferOffset(destination, pool_index);

        for (unsigned vec = vec_lane; vec < kRowVecs; vec += vec_stride) {
            const uint4 value =
                workspace_->br_.template Load<BufferResource::kNone>(
                    source_row + vec * sizeof(uint4), 0);
            workspace_->br_.template Store<BufferResource::kNone>(
                destination_row + vec * sizeof(uint4), 0, value);
        }
        if (header_owner) {
            const unsigned weight =
                workspace_->br_.template LoadU32<BufferResource::kNone>(
                    workspace_->InputTokenTopKExpertWeightOffset(
                        workspace_->Rank()) +
                        route * sizeof(float),
                    0);
            workspace_->br_.template StoreU32<BufferResource::kNone>(
                workspace_->L1TokenWeightsOffset(destination, pool_index), 0,
                weight);
            const TokenMetadata metadata{route, workspace_->Rank()};
            workspace_->br_.template StoreU64<BufferResource::kNone>(
                workspace_->TokenMetadataOffset(destination, pool_index), 0,
                __builtin_bit_cast(uint2, metadata));
        }
    }

    TAL_DEVICE void PopulateSendCounters(unsigned tid, unsigned wid,
                                         unsigned wtid, unsigned parity,
                                         unsigned expected) {
        Common::ClearExpertCounts(shm_->expert_count, tid);

        const unsigned route_count = num_tokens_ * kTopK;
        for (unsigned route = tid; route < route_count; route += kThreads) {
            const unsigned expert =
                workspace_->br_.template LoadU32<BufferResource::kNone>(
                    workspace_->InputTokenTopKExpertIDOffset() +
                        route * sizeof(unsigned),
                    0);
            if (expert < kNumExperts)
                atomicAdd(shm_->expert_count + expert, 1);
        }
        __syncthreads();

        for (unsigned expert = tid; expert < kNumExperts; expert += kThreads) {
            const unsigned count = shm_->expert_count[expert];
            StoreParityValue(
                workspace_->SendCounterOffset(workspace_->Rank(), expert),
                parity, count);
            StoreParityValue(
                workspace_->RecvCounterOffset(
                    expert / kExpertsPerRank, workspace_->Rank(),
                    expert % kExpertsPerRank),
                parity, count);
            shm_->expert_count[expert] = 0;
        }
        amdgcn_s_waitcnt<0, -1, 0>();
        // Publish the transposed counts before grouping. This lets wave 0
        // consume peer counts and build the destination plan while the other
        // waves build the source route order in the same CTA.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
        __syncthreads();
        if (wid == 0 && wtid < kNumRanks) {
            const unsigned destination =
                (workspace_->Rank() + wtid) % kNumRanks;
            store_xgpu_epoch_release(
                *workspace_,
                workspace_->DirectPushCountDoneOffset(
                    destination, parity, workspace_->Rank()),
                expected);
        }
        amdgcn_s_waitcnt<0, -1, 0>();
    }

    TAL_DEVICE void BuildDestinationPlan(unsigned tid, unsigned wid,
                                         unsigned wtid, unsigned parity,
                                         unsigned expected) {

        // Wave 0 owns the destination plan while the remaining waves
        // concurrently build the source route order.
        if (wid == 0) {
            if (wtid < kNumRanks) {
                wait_xgpu_signal_relaxed(
                    *workspace_,
                    workspace_->DirectPushCountDoneOffset(
                        workspace_->Rank(), parity, wtid),
                    static_cast<std::int32_t>(expected));
            }
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");

            const bool valid_expert = wtid < kExpertsPerRank;
            unsigned total = 0;
            if (valid_expert) {
                for (unsigned source = 0; source < kNumRanks; ++source) {
                    const unsigned count = LoadParityValue(
                        workspace_->RecvCounterOffset(workspace_->Rank(),
                                                      source, wtid),
                        parity);
                    shm_->source_count[source * kExpertsPerRank + wtid] =
                        count;
                    total += count;
                }
            }

            const unsigned padded =
                valid_expert ? tal::AlignUp(total, 32u) : 0u;
            const unsigned inclusive =
                amdgcn_wave_inclusive_add(padded, wtid);
            const unsigned pool_base = inclusive - padded;

            if (valid_expert) {
                workspace_->br_.template StoreU64<BufferResource::kNone>(
                    workspace_->RecvSumCounterOffset(workspace_->Rank(),
                                                     wtid),
                    0, {total, kNumSMs * kNumRanks});

                unsigned source_prefix = 0;
                for (unsigned source = 0; source < kNumRanks; ++source) {
                    StoreParityValue(
                        workspace_->DirectPushPlanBaseOffset(
                            source, workspace_->Rank(), wtid),
                        parity, pool_base + source_prefix);
                    source_prefix += shm_->source_count[source *
                                                            kExpertsPerRank +
                                                        wtid];
                }
            }

            // Admission proves the previous kernel has completed. Clear only
            // the blocks in the plan being admitted, rather than making the
            // owner CTA sweep the full worst-case workspace for an 8-token
            // launch. Lanes after the last valid expert carry the
            // wave-wide padded-row total from the inclusive scan.
            const unsigned pool_rows = __shfl(inclusive, kWarpSize - 1);
            for (unsigned block = wtid;
                 block < pool_rows / kSortedTokenBlock;
                 block += kWarpSize) {
                workspace_->br_.template StoreU32<BufferResource::kNone>(
                    workspace_->L2ArrivalMaskOffset(block), 0, 0);
            }
        } else {
            const unsigned group_tid =
                (wid - 1) * kWarpSize + wtid;
            const unsigned group_threads =
                (kThreads / kWarpSize - 1) * kWarpSize;
            const unsigned route_count = num_tokens_ * kTopK;
            for (unsigned route = group_tid; route < route_count;
                 route += group_threads) {
                const unsigned expert =
                    workspace_->br_.template LoadU32<BufferResource::kNone>(
                        workspace_->InputTokenTopKExpertIDOffset() +
                            route * sizeof(unsigned),
                        0);
                if (expert < kNumExperts) {
                    const unsigned ordinal =
                        atomicAdd(shm_->expert_count + expert, 1);
                    workspace_->br_.template StoreU32<BufferResource::kNone>(
                        workspace_->RecvTokenOffset(
                            workspace_->Rank(), expert / kExpertsPerRank,
                            expert % kExpertsPerRank, ordinal),
                        0, route);
                }
            }
        }
        amdgcn_s_waitcnt<0, -1, 0>();
        // Every planner lane releases its plan, route-list, or cleanup stores
        // before wave 0 publishes the dependencies that admit producers and
        // compute CTAs.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
        __syncthreads();
        if (wid == 0 && wtid < kNumRanks) {
            store_xgpu_epoch_release(
                *workspace_,
                workspace_->DirectPushPlanReadyOffset(
                    wtid, parity, workspace_->Rank()),
                expected);
        }
        if (tid == 0) {
            store_xgpu_epoch_release(
                *workspace_,
                workspace_->DirectPushEpochGateOffset(workspace_->Rank()),
                current_epoch_);
        }
        amdgcn_s_waitcnt<0, -1, 0>();
        __syncthreads();
    }

    template <unsigned kDispatchBlocks>
    TAL_DEVICE void PushPayload(unsigned producer_slot, unsigned tid,
                                unsigned wid, unsigned wtid, unsigned parity,
                                unsigned expected) {
        // Pin every producer to one destination. With no payload chunking,
        // task_index advances by the dispatch grid size and maps back to a
        // local expert by dividing by the destination count.
        const unsigned destination = producer_slot % kNumRanks;
        if (tid == 0) {
            wait_xgpu_signal_relaxed(
                *workspace_,
                workspace_->DirectPushPlanReadyOffset(
                    workspace_->Rank(), parity, destination),
                static_cast<std::int32_t>(expected));
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
        }
        __syncthreads();

        for (unsigned task_index = producer_slot; task_index < kNumExperts;
             task_index += kDispatchBlocks) {
            const unsigned local_expert = task_index / kNumRanks;
            const unsigned expert =
                destination * kExpertsPerRank + local_expert;
            if (tid == 0) {
                shm_->payload_plan = {
                    LoadParityValue(
                        workspace_->SendCounterOffset(workspace_->Rank(),
                                                      expert),
                        parity),
                    LoadParityValue(
                        workspace_->DirectPushPlanBaseOffset(
                            workspace_->Rank(), destination, local_expert),
                        parity)};
            }
            __syncthreads();
            const uint2 plan = shm_->payload_plan;

            // Assign one row to each wave only once all waves have enough
            // rows; otherwise the whole CTA cooperates on one row.
            if (plan.x >= kThreads / kWarpSize * 2) {
                for (unsigned ordinal = wid; ordinal < plan.x;
                     ordinal += kThreads / kWarpSize) {
                    unsigned route = 0;
                    if (wtid == 0) {
                        route =
                            workspace_->br_.template LoadU32<kPeerCoherent>(
                                workspace_->RecvTokenOffset(
                                    workspace_->Rank(), destination,
                                    local_expert, ordinal),
                                0);
                    }
                    route = __builtin_amdgcn_readfirstlane(route);
                    CopyPayloadRow(destination, plan.y + ordinal, route, wtid,
                                   kWarpSize, wtid == 0);
                }
            } else {
                for (unsigned ordinal = 0; ordinal < plan.x; ++ordinal) {
                    const unsigned route =
                        workspace_->br_.template LoadU32<kPeerCoherent>(
                            workspace_->RecvTokenOffset(
                                workspace_->Rank(), destination, local_expert,
                                ordinal),
                            0);
                    CopyPayloadRow(destination, plan.y + ordinal, route, tid,
                                   kThreads, tid == 0);
                }
            }
            amdgcn_s_waitcnt<0, -1, 0>();
            // Publish each task independently only after every wave's rows
            // and header stores have reached the release point.
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
            __syncthreads();
            if (tid == 0) {
                workspace_->br_
                    .template AtomicAddI32<
                        BufferResource::kAtomicScopeSystem>(
                        workspace_->DirectPushPayloadReadyOffset(
                            destination, parity, local_expert),
                        0, 1);
            }
            amdgcn_s_waitcnt<0, -1, 0>();
            __syncthreads();
        }
    }

    unsigned num_tokens_;
    Workspace *workspace_;
    Shm *shm_;
    unsigned current_epoch_ = 0;
};

} // namespace causalflow::petit::rocm::moe
