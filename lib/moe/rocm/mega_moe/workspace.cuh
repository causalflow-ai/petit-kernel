#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "moe/rocm/fused_moe.cuh"
#include "moe/rocm/ops/mxfp4_activation.cuh"

#include <type_traits>

namespace causalflow::petit::rocm::moe {

template <class Layout, class = void> struct GridSyncSlotCount {
    static constexpr unsigned value = 2;
};

template <class Layout>
struct GridSyncSlotCount<Layout,
                         std::void_t<decltype(Layout::kGridSyncSlots)>> {
    static constexpr unsigned value = Layout::kGridSyncSlots;
};

///
// MegaMoEWorkspace provides convienent APIs to access the data across multi
// ranks. The layout is the following:
// - Cross GPU barriers: one VMM page per rank, visible to every rank
//     - Counter: u32, local phase/status word
//     - Signal: (2,), i32, written by peers and polled by the owner
// Aligned to 4KB page boundary
// - Per-GPU shared data (kNumRanks instances each of which has slot_stride
// bytes, globally visible across all ranks)
//     - SendCounter: (kNumExperts,), i64, how many tokens will be dispatched
//     to each expert
//     - RecvCounter: (kRank, kNumExpertsPerRank,), i64, how many tokens will be
//     received from each local expert of a source rank
//     - RecvSumCounter: (kNumExpertsPerRank,), i64, total tokens will be
//     received from each local expert
//     - RecvToken: (kNumRanks, kNumExpertsPerRank, max_tokens_per_rank), i32
//     received [src_rank][local_experts][slots] = src_token_topk_idx
//     - Input TokenTopK Expert Weight: (max_tokens_per_rank, kTopK), f32
//     - Input Tokens: max_tokens_per_rank tokens (in input format)
//     - Route output: source-owned (token, top-k) BF16 rows
//     - Route output ready: (2, max_tokens_per_rank), i32 parity-buffered
//       epoch-local stage-2 contribution readiness
//     - L1 payload arrival masks: one u32 row mask per M32 routed-token block
//       published by direct-push sources and consumed by stage 1
//     - Token metadata: worst-case routed tokens, u64
//     - L1 token buffer: worst-case routed tokens * Layout::kInputTokenBytes
//     - L1 token weights: worst-case routed tokens * f32
//     - Aligned to 4KB page boundary (slot_stride_)
// Aligned to 2MB page boundary
//
// Local data: (only visible to the local rank)
// - Grid sync barrier (aligned to 64-byte cache line size)
// - Direct-push work heads: two sets of eight independently cached scheduler
//   shards, one set per GEMM stage
// - Input TokenTopK Expert ID: (max_tokens_per_rank, kTopK), i32
// - L2 arrival masks: one u32 bit mask per M32 routed-token block
// - L2 token buffer: stage-2 MXFP4 activation values for all routed tokens
// - L2 scale buffer: padded E8M0 scales for the L2 token buffer
//
// We require the total VMA address space to be within 4GB so that we can use
// buffer_load instructions for efficient memory access.
//
// Epoch synchronization and direct push use additional non-overlapping fields
// in the unused portion of each rank's cross-GPU barrier page:
//     - EpochCounter: i32, the locally owned publication generation
//     - EpochSignal: (kNumRanks,), i32, one publication per source rank
//     - EntryCount: (kNumSMs,), i32, monotonic invocation epoch owned by each
//     fixed-role CTA
//     - PlanBase: (2, kNumRanks, kNumExpertsPerRank), i32, parity-buffered
//     destination-owned row bases pushed back to each source
//     - CountDone: (2, kNumRanks,), i32, parity-buffered count readiness by
//     source, stored on the destination rank
//     - PlanReady: (2, kNumRanks,), i32, parity-buffered destination-plan
//     readiness stored on the source rank
//     - PayloadReady: (2, kNumExpertsPerRank,), i32, cumulative completed
//     source payloads for each destination-local expert
//     - EpochGate: i32, rank-local owner-to-producer admission gate
//     - LaunchReady: (kNumRanks,), i32, peer launch-admission epochs
// Direct push and pull shuffle are mutually exclusive within an invocation and
// use the same SendCounter, RecvCounter, RecvSumCounter, RecvToken, token
// metadata, L1 token buffer, and L1 token weights.
template <class Layout> class MegaMoEWorkspace {
    using ActivationLayout = MxFp4ActivationLayout;
    static constexpr unsigned kXGpuBarrierCounterOffset = 0;

    static constexpr unsigned kCacheLineBytes = 128;
    static constexpr unsigned kPageBytes = 4096;
    static constexpr unsigned kLargePageBytes = 2 * 1024 * 1024;
    static constexpr unsigned kXGpuEpochCounterOffset = kCacheLineBytes;
    static constexpr unsigned kXGpuEpochSignalBaseOffset =
        2 * kCacheLineBytes;
    // VMM mappings are page-granular.  Keep each rank's xGPU signal record
    // on an independently owned page, matching the legacy symmetric-buffer
    // layout.  Packing records 64 bytes apart puts every rank's system-scope
    // atomic traffic on one GPU-owned backing page at world size eight.
    // DeepSeek EP8 has up to 48 local experts, so its direct-push plan no
    // longer fits in the otherwise-unused tail of a single 4 KiB VMM page.
    // VMM mappings are page-granular and the public workspace description
    // already carries this stride, so reserve a second page for the larger
    // expert configurations while preserving the GPT-OSS layout.
    static constexpr unsigned kXGpuBarrierRecordBytes =
        Layout::kNumExperts > 128 ? 2 * kPageBytes : kPageBytes;
    static constexpr unsigned kMaxGridSyncSlots =
        GridSyncSlotCount<Layout>::value;
    static_assert(kMaxGridSyncSlots * sizeof(unsigned) + sizeof(unsigned) <
                      kCacheLineBytes,
                  "Too many grid sync slots");

    static constexpr unsigned kNumRanks = Layout::kNumRanks;
    static constexpr unsigned kNumExperts = Layout::kNumExperts;
    static constexpr unsigned kNumExpertsPerRank = kNumExperts / kNumRanks;
    static constexpr unsigned kMaxTokensPerRank = Layout::kMaxTokensPerRank;
    static constexpr unsigned kTopK = Layout::kTopK;
    static constexpr unsigned kDirectWorkShards = 8;
    static constexpr unsigned kDirectWorkHeadSets = 2;
    static constexpr unsigned kDirectWorkShardStride = 64;
    static constexpr unsigned kDirectWorkHeadBytes =
        kDirectWorkHeadSets * kDirectWorkShards * kDirectWorkShardStride;
    static constexpr unsigned kRankSymBufferBase =
        kNumRanks * kXGpuBarrierRecordBytes;
    static constexpr unsigned kSortedTokenBlock = 32;
    static constexpr unsigned kMaxExpertsPerToken =
        kTopK < kNumExpertsPerRank ? kTopK : kNumExpertsPerRank;
    // Both shuffles lay out routed entries per local expert and pad every
    // expert's run to kSortedTokenBlock. Account for the worst-case top-k
    // fanout plus every per-expert tail.
    static constexpr unsigned kMaxPoolTokens = tal::AlignUp<unsigned>(
        kNumRanks * kMaxTokensPerRank * kMaxExpertsPerToken +
            kNumExpertsPerRank * (kSortedTokenBlock - 1),
        kSortedTokenBlock);

  public:
    static constexpr unsigned kMaxPoolBlocks =
        kMaxPoolTokens / kSortedTokenBlock;
    static constexpr unsigned kL2ScaleRows =
        ActivationLayout::PaddedScaleRows(kMaxPoolTokens);
    static constexpr unsigned kL2ScaleCols =
        ActivationLayout::ScaleCols(Layout::kInterDim);

  private:
    static constexpr unsigned long kL2ArrivalMaskBytes =
        static_cast<unsigned long>(kMaxPoolBlocks) * sizeof(unsigned);
    static constexpr unsigned long kL2TokenBufferBytes =
        static_cast<unsigned long>(kMaxPoolTokens) * Layout::kInterDim / 2;
    static constexpr unsigned long kL2ScaleBufferBytes =
        static_cast<unsigned long>(kL2ScaleRows) * kL2ScaleCols;
    static_assert(kXGpuEpochSignalBaseOffset +
                          kNumRanks * sizeof(unsigned) <=
                      3 * kCacheLineBytes,
                  "xGPU epoch slots exceed their reserved cache line");
    static constexpr unsigned kDirectControlOffset = 3 * kCacheLineBytes;
    static constexpr unsigned kDirectEntryCountBytes =
        Layout::kNumSMs * sizeof(unsigned);
    static constexpr unsigned kDirectPlanBaseBytes =
        kNumExperts * sizeof(unsigned long);
    static constexpr unsigned kDirectCountDoneBytes =
        2 * kNumRanks * sizeof(unsigned);
    static constexpr unsigned kDirectPlanReadyBytes =
        2 * kNumRanks * sizeof(unsigned);
    static constexpr unsigned kDirectPayloadReadyBytes =
        2 * kNumExpertsPerRank * sizeof(unsigned);
    static constexpr unsigned kDirectEpochGateBytes = sizeof(unsigned);
    static constexpr unsigned kDirectLaunchReadyBytes =
        kNumRanks * sizeof(unsigned);
    static constexpr unsigned kDirectControlBytes =
        kDirectEntryCountBytes + kDirectPlanBaseBytes + kDirectCountDoneBytes +
        kDirectPlanReadyBytes + kDirectPayloadReadyBytes +
        kDirectEpochGateBytes + kDirectLaunchReadyBytes;
    static_assert(kDirectControlOffset + kDirectControlBytes <=
                      kXGpuBarrierRecordBytes,
                  "Direct-push controls exceed the cross-GPU barrier page");
    static constexpr unsigned long kTokenMetadataBytes =
        static_cast<unsigned long>(kMaxPoolTokens) * sizeof(TokenMetadata);
    static constexpr unsigned long kRouteOutputReadyBytes =
        2ul * kMaxTokensPerRank * sizeof(unsigned);
    static constexpr unsigned long kL1PayloadArrivalMaskBytes =
        static_cast<unsigned long>(kMaxPoolBlocks) * sizeof(unsigned);
    static constexpr unsigned long kL1TokenBufferBytes =
        static_cast<unsigned long>(kMaxPoolTokens) * Layout::kInputTokenBytes;
    static constexpr unsigned long kL1TokenWeightBytes =
        static_cast<unsigned long>(kMaxPoolTokens) * sizeof(float);
    static constexpr unsigned long kRankSlotRawBytes =
        kNumExperts * sizeof(unsigned long) +
        kNumRanks * kNumExpertsPerRank * sizeof(unsigned long) +
        kNumExpertsPerRank * sizeof(unsigned long) +
        static_cast<unsigned long>(kNumRanks) * kNumExpertsPerRank *
            kMaxTokensPerRank * sizeof(unsigned) +
        static_cast<unsigned long>(kMaxTokensPerRank) * kTopK * sizeof(float) +
        static_cast<unsigned long>(kMaxTokensPerRank) *
            Layout::kInputTokenBytes +
        static_cast<unsigned long>(kMaxTokensPerRank) *
            Layout::kRouteOutputBufferBytes +
        kRouteOutputReadyBytes + kL1PayloadArrivalMaskBytes +
        kTokenMetadataBytes + kL1TokenBufferBytes + kL1TokenWeightBytes;
    static_assert(kRankSlotRawBytes <= 0xffffffffull,
                  "MegaMoE rank slot exceeds 32-bit offsets");
    static constexpr unsigned kSlotStride = tal::AlignUp<unsigned>(
        static_cast<unsigned>(kRankSlotRawBytes), kRankSymBufferBase);
    static constexpr unsigned kLocalOffsetBase = tal::AlignUp(
        kRankSymBufferBase + kNumRanks * kSlotStride, kLargePageBytes);

    static_assert(kNumExperts % kNumRanks == 0,
                  "the expert count must divide evenly across ranks");
    static_assert(sizeof(TokenMetadata) == sizeof(unsigned long) &&
                      kNumRanks < 256,
                  "");

  public:
    static constexpr unsigned long kLocalDataBytes64 =
        kCacheLineBytes + kDirectWorkHeadBytes +
        static_cast<unsigned long>(kMaxTokensPerRank) * kTopK *
            sizeof(unsigned) +
        kL2ArrivalMaskBytes + kL2TokenBufferBytes + kL2ScaleBufferBytes;
    static constexpr unsigned long kLocalBytes64 = kLocalDataBytes64;
    static_assert(kLocalBytes64 <= (1ull << 32),
                  "MegaMoE local workspace exceeds 32-bit offsets");
    static constexpr unsigned kLocalBytes =
        static_cast<unsigned>(kLocalBytes64);
    static constexpr unsigned long kWorkspaceBytes64 =
        static_cast<unsigned long>(kLocalOffsetBase) + kLocalBytes64;
    static_assert(kWorkspaceBytes64 < (1ull << 32),
                  "MegaMoE workspace must fit in the 4 GiB buffer VMA range");
    static constexpr unsigned kWorkspaceBytes = (unsigned)kWorkspaceBytes64;

    TAL_HOST_DEVICE static constexpr unsigned WorkspaceBytes() {
        return kWorkspaceBytes;
    }

    TAL_HOST_DEVICE static constexpr unsigned XGpuBarrierRecordBytes() {
        return kXGpuBarrierRecordBytes;
    }

    TAL_HOST_DEVICE static constexpr unsigned RankSymBufferBase() {
        return kRankSymBufferBase;
    }

    TAL_HOST_DEVICE static constexpr unsigned RankSymBufferSlotBytes() {
        return kSlotStride;
    }

    TAL_HOST_DEVICE static constexpr unsigned LocalOffsetBase() {
        return kLocalOffsetBase;
    }

    TAL_HOST_DEVICE
    explicit MegaMoEWorkspace(void *buf_base, unsigned rank_id)
        : rank_id_(rank_id) {
        br_.v = {
            .ptr = reinterpret_cast<std::uintptr_t>(buf_base),
            .range = kWorkspaceBytes,
            .config = BufferResource::kDataFormatU32Config,
        };
    }

    TAL_HOST_DEVICE static inline unsigned
    XGpuBarrierCounterOffset(unsigned rank) {
        [[assume(rank < kNumRanks)]];
        return kXGpuBarrierCounterOffset + rank * kXGpuBarrierRecordBytes;
    }

    TAL_HOST_DEVICE static inline unsigned
    XGpuBarrierSignalOffset(unsigned rank, unsigned phase) {
        [[assume(rank < kNumRanks)]];
        [[assume(phase < 2)]];
        return XGpuBarrierCounterOffset(rank) + sizeof(unsigned) +
               phase * sizeof(unsigned);
    }

    TAL_HOST_DEVICE static inline unsigned
    XGpuEpochCounterOffset(unsigned rank) {
        [[assume(rank < kNumRanks)]];
        return rank * kXGpuBarrierRecordBytes + kXGpuEpochCounterOffset;
    }

    TAL_HOST_DEVICE static inline unsigned
    XGpuEpochSignalOffset(unsigned rank, unsigned source_rank) {
        [[assume(rank < kNumRanks)]];
        [[assume(source_rank < kNumRanks)]];
        return rank * kXGpuBarrierRecordBytes + kXGpuEpochSignalBaseOffset +
               source_rank * sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned
    SendCounterOffset(unsigned rank, unsigned expert_idx) const {
        [[assume(rank < kNumRanks)]];
        [[assume(expert_idx < kNumExperts)]];
        return RankOffsetBase(rank) + expert_idx * sizeof(unsigned long);
    }

    TAL_HOST_DEVICE inline unsigned
    RecvCounterOffset(unsigned rank, unsigned src_rank,
                      unsigned local_expert_idx) const {
        [[assume(rank < kNumRanks)]];
        [[assume(src_rank < kNumRanks)]];
        [[assume(local_expert_idx < kNumExpertsPerRank)]];
        return SendCounterOffset(rank, kNumExperts - 1) +
               sizeof(unsigned long) +
               (src_rank * kNumExpertsPerRank + local_expert_idx) *
                   sizeof(unsigned long);
    }

    TAL_HOST_DEVICE inline unsigned
    RecvSumCounterOffset(unsigned rank, unsigned local_expert_idx) const {
        [[assume(rank < kNumRanks)]];
        [[assume(local_expert_idx < kNumExpertsPerRank)]];
        return RecvCounterOffset(rank, kNumRanks - 1, kNumExpertsPerRank - 1) +
               sizeof(unsigned long) + local_expert_idx * sizeof(unsigned long);
    }

    TAL_HOST_DEVICE inline unsigned RecvTokenOffset(unsigned rank,
                                                    unsigned src_rank,
                                                    unsigned local_expert_idx,
                                                    unsigned slot) const {
        [[assume(rank < kNumRanks)]];
        [[assume(src_rank < kNumRanks)]];
        [[assume(local_expert_idx < kNumExpertsPerRank)]];
        [[assume(slot < kMaxTokensPerRank)]];
        return RecvSumCounterOffset(rank, kNumExpertsPerRank - 1) +
               sizeof(unsigned long) +
               ((src_rank * kNumExpertsPerRank + local_expert_idx) *
                    kMaxTokensPerRank +
                slot) *
                   sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned
    InputTokenTopKExpertWeightOffset(unsigned rank) const {
        [[assume(rank < kNumRanks)]];
        return RecvTokenOffset(rank, kNumRanks - 1, kNumExpertsPerRank - 1,
                               kMaxTokensPerRank - 1) +
               sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned InputTokensOffset(unsigned rank) const {
        [[assume(rank < kNumRanks)]];
        return InputTokenTopKExpertWeightOffset(rank) +
               kMaxTokensPerRank * kTopK * sizeof(float);
    }

    TAL_HOST_DEVICE inline unsigned
    RouteOutputBufferOffset(unsigned rank) const {
        [[assume(rank < kNumRanks)]];
        return InputTokensOffset(rank) +
               kMaxTokensPerRank * Layout::kInputTokenBytes;
    }

    TAL_HOST_DEVICE inline unsigned RouteOutputReadyOffset(
        unsigned rank, unsigned parity, unsigned token) const {
        [[assume(rank < kNumRanks)]];
        [[assume(parity < 2)]];
        [[assume(token < kMaxTokensPerRank)]];
        return RouteOutputBufferOffset(rank) +
               kMaxTokensPerRank * Layout::kRouteOutputBufferBytes +
               (parity * kMaxTokensPerRank + token) * sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned
    L1PayloadArrivalMaskOffset(unsigned rank,
                               unsigned pool_block_index) const {
        [[assume(rank < kNumRanks)]];
        [[assume(pool_block_index < kMaxPoolBlocks)]];
        return RouteOutputReadyOffset(rank, 1, kMaxTokensPerRank - 1) +
               sizeof(unsigned) + pool_block_index * sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned
    TokenMetadataOffset(unsigned rank, unsigned pool_token_index) const {
        [[assume(rank < kNumRanks)]];
        [[assume(pool_token_index < kMaxPoolTokens)]];
        return L1PayloadArrivalMaskOffset(rank, kMaxPoolBlocks - 1) +
               sizeof(unsigned) +
               pool_token_index * sizeof(TokenMetadata);
    }

    TAL_HOST_DEVICE inline unsigned
    L1TokenBufferOffset(unsigned rank, unsigned pool_token_index) const {
        [[assume(rank < kNumRanks)]];
        [[assume(pool_token_index < kMaxPoolTokens)]];
        return TokenMetadataOffset(rank, kMaxPoolTokens - 1) +
               sizeof(TokenMetadata) +
               pool_token_index * Layout::kInputTokenBytes;
    }

    TAL_HOST_DEVICE inline unsigned
    L1TokenWeightsOffset(unsigned rank, unsigned pool_token_index) const {
        [[assume(rank < kNumRanks)]];
        [[assume(pool_token_index < kMaxPoolTokens)]];
        return L1TokenBufferOffset(rank, kMaxPoolTokens - 1) +
               Layout::kInputTokenBytes + pool_token_index * sizeof(float);
    }

    // Direct-push planning metadata and publication state lives in otherwise
    // unused space in each rank's cross-GPU barrier page.
    TAL_HOST_DEVICE inline unsigned
    DirectPushEntryCountOffset(unsigned rank, unsigned block) const {
        [[assume(rank < kNumRanks)]];
        [[assume(block < Layout::kNumSMs)]];
        return XGpuBarrierCounterOffset(rank) + kDirectControlOffset +
               block * sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned DirectPushPlanBaseOffset(
        unsigned rank, unsigned source_rank,
        unsigned local_expert_idx) const {
        [[assume(rank < kNumRanks)]];
        [[assume(source_rank < kNumRanks)]];
        [[assume(local_expert_idx < kNumExpertsPerRank)]];
        return DirectPushEntryCountOffset(rank, Layout::kNumSMs - 1) +
               sizeof(unsigned) +
               (source_rank * kNumExpertsPerRank + local_expert_idx) *
                   sizeof(unsigned long);
    }

    TAL_HOST_DEVICE inline unsigned DirectPushPlanBaseOffset(
        unsigned rank, unsigned source_rank, unsigned local_expert_idx,
        unsigned parity) const {
        [[assume(parity < 2)]];
        return DirectPushPlanBaseOffset(rank, source_rank, local_expert_idx) +
               parity * sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned
    DirectPushCountDoneOffset(unsigned rank, unsigned parity,
                              unsigned source_rank) const {
        [[assume(rank < kNumRanks)]];
        [[assume(parity < 2)]];
        [[assume(source_rank < kNumRanks)]];
        return DirectPushPlanBaseOffset(rank, kNumRanks - 1,
                                        kNumExpertsPerRank - 1) +
               sizeof(unsigned long) +
               (parity * kNumRanks + source_rank) * sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned
    DirectPushPlanReadyOffset(unsigned rank, unsigned parity,
                              unsigned destination_rank) const {
        [[assume(rank < kNumRanks)]];
        [[assume(parity < 2)]];
        [[assume(destination_rank < kNumRanks)]];
        return DirectPushCountDoneOffset(rank, 1, kNumRanks - 1) +
               sizeof(unsigned) +
               (parity * kNumRanks + destination_rank) * sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned
    DirectPushPayloadReadyOffset(unsigned rank, unsigned parity,
                                 unsigned local_expert_idx) const {
        [[assume(rank < kNumRanks)]];
        [[assume(parity < 2)]];
        [[assume(local_expert_idx < kNumExpertsPerRank)]];
        return DirectPushPlanReadyOffset(rank, 1, kNumRanks - 1) +
               sizeof(unsigned) +
               (parity * kNumExpertsPerRank + local_expert_idx) *
                   sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned
    DirectPushEpochGateOffset(unsigned rank) const {
        [[assume(rank < kNumRanks)]];
        return DirectPushPayloadReadyOffset(rank, 1,
                                            kNumExpertsPerRank - 1) +
               sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned
    DirectPushLaunchReadyOffset(unsigned rank, unsigned source_rank) const {
        [[assume(rank < kNumRanks)]];
        [[assume(source_rank < kNumRanks)]];
        return DirectPushEpochGateOffset(rank) + sizeof(unsigned) +
               source_rank * sizeof(unsigned);
    }

    //
    // Private data mapped only on this rank.
    //
    TAL_HOST_DEVICE inline unsigned GridSyncBarrierOffset() const {
        return kLocalOffsetBase;
    }

    // Keep the eight destination-local work-head atomics on separate 64-byte
    // lines.
    TAL_HOST_DEVICE inline unsigned
    DirectPushWorkHeadOffset(unsigned shard, unsigned set = 0) const {
        [[assume(shard < kDirectWorkShards)]];
        [[assume(set < kDirectWorkHeadSets)]];
        return GridSyncBarrierOffset() + kCacheLineBytes +
               (set * kDirectWorkShards + shard) * kDirectWorkShardStride;
    }

    TAL_HOST_DEVICE inline unsigned InputTokenTopKExpertIDOffset() const {
        return GridSyncBarrierOffset() + kCacheLineBytes +
               kDirectWorkHeadBytes;
    }

    TAL_HOST_DEVICE inline unsigned
    L2ArrivalMaskOffset(unsigned pool_block_index) const {
        [[assume(pool_block_index < kMaxPoolBlocks)]];
        return InputTokenTopKExpertIDOffset() +
               kMaxTokensPerRank * kTopK * sizeof(unsigned) +
               pool_block_index * sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned
    L2TokenBufferOffset(unsigned pool_token_index) const {
        [[assume(pool_token_index < kMaxPoolTokens)]];
        return L2ArrivalMaskOffset(kMaxPoolBlocks - 1) + sizeof(unsigned) +
               pool_token_index * Layout::kInterDim / 2;
    }

    TAL_HOST_DEVICE inline unsigned L2ScaleBufferOffset() const {
        return L2TokenBufferOffset(kMaxPoolTokens - 1) + Layout::kInterDim / 2;
    }

    TAL_HOST_DEVICE inline unsigned Rank() const { return rank_id_; }

    BufferResource br_;

  private:
    unsigned rank_id_;

    TAL_HOST_DEVICE inline unsigned RankOffsetBase(unsigned rank) const {
        [[assume(rank < kNumRanks)]];
        return kRankSymBufferBase + rank * kSlotStride;
    }
};
} // namespace causalflow::petit::rocm::moe
