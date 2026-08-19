#pragma once

#include "moe/rocm/comm/barrier.cuh"
#include "moe/rocm/mega_moe/workspace.cuh"
#include "moe/rocm/ops/mega_moe/token_shuffle_common.cuh"
#include "dataformat_bf16.cuh"
#include "dataformat_mxfp4.cuh"

namespace causalflow::petit::rocm::moe {

template <FusedMoEDataType kDType, unsigned kLogicalHiddenSize,
          unsigned kComputeHiddenSize,
          unsigned kWarpsForPull, unsigned kNumWarps>
struct RemoteInputTransportSelector;

template <unsigned kLogicalHiddenSize, unsigned kComputeHiddenSize,
          unsigned kWarpsForPull, unsigned kNumWarps>
struct RemoteInputTransportSelector<
    FusedMoEDataType::kBf16, kLogicalHiddenSize, kComputeHiddenSize,
    kWarpsForPull, kNumWarps> {
    using Type = RemoteBf16Transport<kLogicalHiddenSize, kComputeHiddenSize,
                                     kWarpsForPull, kNumWarps>;
};

template <unsigned kLogicalHiddenSize, unsigned kComputeHiddenSize,
          unsigned kWarpsForPull, unsigned kNumWarps>
struct RemoteInputTransportSelector<
    FusedMoEDataType::kMxFp4, kLogicalHiddenSize, kComputeHiddenSize,
    kWarpsForPull, kNumWarps> {
    using Type = RemoteMxFp4Transport<kLogicalHiddenSize, kWarpsForPull,
                                      kNumWarps>;
};

///
// Shuffle tokens for EP MoE via: (1) sending the token_topk_index to remote
// ranks, and (2) pull tokens and route weight from remote ranks.
template <class Config> struct TokenShuffle {
  public:
    using Workspace = MegaMoEWorkspace<Config>;
    using Common = TokenShuffleCommon<Config>;
    using XGpuSync = typename Config::XGpuSync;

    static constexpr unsigned kNumSMs = Config::kNumSMs;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kThreads = Config::kThreads;
    static constexpr unsigned kNumTopK = Config::kTopK;
    static constexpr unsigned kNumExperts = Config::kNumExperts;
    static constexpr unsigned kNumRanks = Config::kNumRanks;
    static constexpr unsigned kNumExpertsPerRank = kNumExperts / kNumRanks;
    static constexpr unsigned kMaxTokensPerRank = Config::kMaxTokensPerRank;
    static constexpr unsigned kHiddenSize = Config::kHiddenSize;

    // L1 computes kSortedTokenBlock tokens at a time.
    static constexpr unsigned kSortedTokenBlock = 32;

    // How many warps are used to pull a token from remote ranks.
    static constexpr unsigned kWarpsPerPullToken = Config::kWarpsPerPullToken;

    using InputTransport = typename RemoteInputTransportSelector<
        Config::kActDType, kHiddenSize, Config::kComputeHiddenSize,
        kWarpsPerPullToken, kNumWarps>::Type;

    // Legacy
    static constexpr unsigned kNumDispatchThreads = kThreads;

    struct Shm {
        unsigned expert_count[Config::kNumExperts];
        typename InputTransport::Shm inputs;
    };

    TAL_DEVICE TokenShuffle(unsigned num_tokens, Workspace *ws, Shm *shm)
        : ws_(ws), shm_(shm), num_tokens_(num_tokens) {}

    TAL_DEVICE void Run(unsigned sm_id, unsigned tid, unsigned wid, unsigned wtid) {
        PushTokenTopKToRemote(sm_id, tid, wid, wtid);
        const auto sync_ticket =
            XGpuSync::template Begin<kDispatchGridSyncIndex>(
                *ws_, sm_id, tid, [=]() { __syncthreads(); });
        PopulateRemoveRecvCounter(sm_id, tid, wid, wtid);
        __syncthreads();

        // Only SM0 writes the receive counters. Publish their completion to
        // peers, then acquire all peer publications before pulling tokens.
        XGpuSync::template Finish<kEpilogueGridSyncIndex>(
            *ws_, sm_id, tid, sync_ticket, [=]() { __syncthreads(); });

        PullTokens(sm_id, tid, wid, wtid);
    }

  public:
    TAL_DEVICE unsigned LoadLocalNumTokens(unsigned wtid) const {
        return Common::LoadLocalNumTokens(*ws_, wtid);
    }

    TAL_DEVICE void ResetRoutingCounters(unsigned sm_id, unsigned tid) {
        Common::ResetRoutingCounters(*ws_, sm_id, tid);
    }

    TAL_DEVICE void ResetLocalCombineSlot(unsigned sm_id, unsigned tid) {
        // The publish handoff guarantees that every wave has consumed the old
        // slot count before CTA 0 clears it for the next invocation.
        if (sm_id == 0 && tid == 0) {
            ws_->br_.template StoreU32<BufferResource::kNone>(
                0, ws_->LocalCombineSlotOffset(), 0);
        }
    }

  private:
    TAL_DEVICE void PullTokens(unsigned sm_id, unsigned tid, unsigned wid,
                               unsigned wtid) {
        static_assert (kNumSMs % kNumExpertsPerRank == 0, "");
        static_assert(kNumExpertsPerRank <= kWarpSize,
                      "A single warp can handle all experts in a rank");
        // The low word is the token count; every rank contributes kNumSMs to
        // the high word.  Wait for the counter's own completion state rather
        // than using the cross-rank signal as a proxy for its visibility.
        uint2 recv_sum_status{0, 0};
        if (wtid < kNumExpertsPerRank) {
            do {
                recv_sum_status = ws_->br_.template LoadU64<
                    BufferResource::kSC1Bit>(
                    wtid * sizeof(unsigned long),
                    ws_->RecvSumCounterOffset(ws_->Rank(), 0));
            } while (recv_sum_status.y != kNumSMs * kNumRanks);
        }

        static constexpr unsigned long long kExpertLaneMask =
            kNumExpertsPerRank == kWarpSize
                ? ~0ull
                : (1ull << kNumExpertsPerRank) - 1;
        unsigned reduced_routes = 0;
        if (wtid < kNumExpertsPerRank) {
            reduced_routes =
                __reduce_add_sync(kExpertLaneMask, recv_sum_status.x);
        }
        const unsigned local_routes = __shfl(reduced_routes, 0);

        static_assert(kNumWarps % kWarpsPerPullToken == 0, "");
        static constexpr unsigned kPullGroupsPerSM =
            kNumWarps / kWarpsPerPullToken;
        static constexpr unsigned kNumPullGroups =
            kNumSMs * kPullGroupsPerSM;
        const unsigned pull_group =
            sm_id + (wid / kWarpsPerPullToken) * kNumSMs;

        static_assert(kNumRanks <= kWarpSize, "");
        for (unsigned route_idx = pull_group; route_idx < local_routes;
             route_idx += kNumPullGroups) {
            unsigned expert = 0;
            unsigned expert_route_start = 0;
            unsigned pool_token_start = 0;

            for (unsigned candidate = 0;
                 candidate < kNumExpertsPerRank; ++candidate) {
                const unsigned expert_tokens =
                    __shfl(recv_sum_status.x, candidate);
                if (route_idx < expert_route_start + expert_tokens) {
                    expert = candidate;
                    break;
                }
                expert_route_start += expert_tokens;
                pool_token_start +=
                    tal::AlignUp(expert_tokens, kSortedTokenBlock);
            }

            const unsigned route_in_expert =
                route_idx - expert_route_start;
            const unsigned recv_counter_base =
                ws_->RecvCounterOffset(ws_->Rank(), 0, 0);
            const unsigned recv_counter_offset =
                wtid < kNumRanks ? (wtid * kNumExpertsPerRank + expert) *
                                       sizeof(unsigned long)
                                 : Workspace::WorkspaceBytes();
            const unsigned tokens_per_rank =
                ws_->br_.template LoadU32<BufferResource::kSC1Bit>(
                    recv_counter_offset, recv_counter_base);

            unsigned src_rank = 0;
            unsigned src_route_start = 0;
#pragma unroll
            for (unsigned candidate = 0; candidate < kNumRanks;
                 ++candidate) {
                const unsigned source_tokens =
                    __shfl(tokens_per_rank, candidate);
                if (route_in_expert < src_route_start + source_tokens) {
                    src_rank = candidate;
                    break;
                }
                src_route_start += source_tokens;
            }

            const unsigned route_in_source =
                route_in_expert - src_route_start;
            const unsigned pool_token_idx =
                pool_token_start + route_in_expert;
            const unsigned recv_token_base =
                ws_->RecvTokenOffset(ws_->Rank(), 0, 0, 0);
            const unsigned token_topk_idx =
                ws_->br_.template LoadU32<BufferResource::kSC1Bit>(
                    ((src_rank * kNumExpertsPerRank + expert) *
                         kMaxTokensPerRank +
                     route_in_source) *
                        sizeof(unsigned),
                    recv_token_base);
            if (wtid == 0 && wid % kWarpsPerPullToken == 0) {
                TokenMetadata d{};
                d.token_topk_idx = token_topk_idx;
                d.src_rank = src_rank;
                if constexpr (!Workspace::kUsesDirectRemoteCombine) {
                    ws_->br_.template AtomicAddI32<
                        BufferResource::kAtomicScopeAgent>(
                        ws_->LocalCombineSlotOffset(), 0, 1);
                    d.local_combine_slot = route_idx;
                }
                ws_->br_.template StoreU64<BufferResource::kNone>(
                    ws_->TokenMetadataOffset(pool_token_idx), 0,
                    __builtin_bit_cast(uint2, d));
                if constexpr (!Workspace::kUsesDirectRemoteCombine) {
                    ws_->br_.template StoreU64<BufferResource::kNone>(
                        ws_->LocalCombinePublishMetadataOffset(route_idx), 0,
                        __builtin_bit_cast(uint2, d));
                }
            }
            TransferTokenToLocalAsync(wid, wtid, src_rank, token_topk_idx,
                                      pool_token_idx);
        }
    }

    TAL_DEVICE void TransferTokenToLocalAsync(unsigned wid, unsigned wtid,
                                              unsigned src_rank,
                                              unsigned token_topk_idx,
                                              unsigned pool_token_idx) {
        unsigned src_token = token_topk_idx / kNumTopK;
        unsigned src_offset = ws_->InputTokensOffset(src_rank) +
                              src_token * InputTransport::kInputTokenBytes;
        unsigned dst_offset = ws_->L1TokenBufferOffset(pool_token_idx);
        if (wid % kWarpsPerPullToken == 0 && wtid == 0) {
            const unsigned weight =
                ws_->br_.template LoadU32<BufferResource::kSC1Bit>(
                    ws_->InputTokenTopKExpertWeightOffset(src_rank) +
                        token_topk_idx * sizeof(float),
                    0);
            // TODO: Try coalesing with TokenMetadata
            ws_->br_.template StoreU32<BufferResource::kNone>(
                ws_->L1TokenWeightsOffset(pool_token_idx), 0, weight);
        }
        using LdsInputShm =
            __attribute__((address_space(3))) typename InputTransport::Shm;
        InputTransport::Copy(*ws_, (LdsInputShm *)&shm_->inputs,
                             wid, wtid, src_offset, dst_offset);
    }

    TAL_DEVICE void PushTokenTopKToRemote(unsigned sm_id, unsigned tid,
                                          unsigned wid, unsigned wtid) {
        [[assume(tid < kThreads)]];
        Common::ClearExpertCounts(shm_->expert_count, tid);

        // Count experts' tokens
        ForeachLocalTokenTopK(
            sm_id, wid, wtid,
            [&](unsigned token_topk_idx, unsigned expert_idx) {
                atomicAdd(shm_->expert_count + expert_idx, 1);
            });
        __syncthreads();

        // Get the start offset of each expert's token_topk_idx in the remote
        // rank
#pragma unroll
        for (unsigned i = tid; i < kNumExperts; i += kThreads) {
            const unsigned long send_value =
                (1ull << 32) | shm_->expert_count[i];
            auto off = ws_->br_.template AtomicAddU64<BufferResource::kAtomicScopeAgent>(
                i * sizeof(unsigned long),
                ws_->SendCounterOffset(ws_->Rank(), 0), send_value);
            shm_->expert_count[i] = (unsigned)off;
        }
        __syncthreads();

        // Write TokenTopK to remote rank's buffer
        ForeachLocalTokenTopK(
            sm_id, wid, wtid,
            [&](unsigned token_topk_idx, unsigned expert_idx) {
                auto dst_rank = expert_idx / kNumExpertsPerRank;
                auto dst_expert = expert_idx % kNumExpertsPerRank;
                auto dst_slot_idx =
                    atomicAdd(shm_->expert_count + expert_idx, 1);
                auto dst_offset = ws_->RecvTokenOffset(
                    dst_rank, ws_->Rank(), dst_expert, dst_slot_idx);
                ws_->br_.template StoreU32<BufferResource::kAtomicScopeSystem>(
                    dst_offset, 0, token_topk_idx);
            });
        __syncthreads();
    }

    TAL_DEVICE void PopulateRemoveRecvCounter(unsigned sm_id, unsigned tid,
                                              unsigned wid, unsigned wtid) {
        // Synchronization ensures that local send counters have been populated
        // and visible.
        if (sm_id) {
            return;
        }

#pragma unroll
        for (unsigned i = tid; i < kNumExperts; i += kThreads) {
            const auto dst_rank = i / kNumExpertsPerRank;
            const auto dst_expert = i % kNumExpertsPerRank;
            const unsigned long expert_status = __builtin_bit_cast(
                unsigned long,
                ws_->br_.template LoadU64<BufferResource::kSC0Bit |
                                           BufferResource::kSC1Bit>(
                    i * sizeof(unsigned long),
                    ws_->SendCounterOffset(ws_->Rank(), 0)));
            Common::PublishRecvCounter(*ws_, i, (unsigned)expert_status);
            // Why system scope?
            // The high 32 bit is supposeed to be kNumSM * kNumRanks
            ws_->br_.template AtomicAddU64<BufferResource::kAtomicScopeSystem>(
                ws_->RecvSumCounterOffset(dst_rank, dst_expert), 0,
                expert_status);
        }
    }

    template <class Process>
    TAL_DEVICE void ForeachLocalTokenTopK(unsigned sm_id, unsigned wid,
                                          unsigned wtid,
                                          const Process &process) {
        static constexpr unsigned kNumTokensPerWarp = kWarpSize / kNumTopK;
        static constexpr unsigned kNumActivateLanes =
            kNumTokensPerWarp * kNumTopK;
        for (unsigned i = (sm_id * kNumWarps + wid) * kNumTokensPerWarp;
             i + (wtid / kNumTopK) < num_tokens_;
             i += kNumSMs * kNumWarps * kNumTokensPerWarp) {
            if (wtid < kNumActivateLanes) {
                const auto token_topk_idx = i * kNumTopK + wtid;
                const auto expert_id =
                    ws_->br_.template LoadU32<BufferResource::kNone>(
                        wtid * sizeof(int),
                        ws_->InputTokenTopKExpertIDOffset() +
                            i * kNumTopK * sizeof(unsigned));
                if (expert_id < kNumExperts) {
                    process(token_topk_idx, expert_id);
                }
            }
        }
    }

    static constexpr unsigned kDispatchGridSyncIndex = 0;
    static constexpr unsigned kEpilogueGridSyncIndex = 1;

    Workspace *ws_;
    Shm *shm_;
    unsigned num_tokens_;
};
} // namespace causalflow::petit::rocm::moe
