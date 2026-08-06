#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "moe/rocm/fused_moe.cuh"

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

template <class Layout, class = void> struct UsesDirectRemoteCombine {
    static constexpr bool value = false;
};

template <class Layout>
struct UsesDirectRemoteCombine<
    Layout, std::void_t<decltype(Layout::kSolution)>> {
    static constexpr bool value =
        Layout::kSolution.stages == FusedMoEStages::kTwoStage;
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
//     - RecvSumCounter: (kNumExpertsPerRank,), i32, total tokens will be
//     received from each local expert
//     - RecvToken: (kNumRanks, kNumExpertsPerRank, max_tokens_per_rank), i32
//     received [src_rank][local_experts][slots] = src_token_topk_idx
//     - Input TokenTopK Expert Weight: (max_tokens_per_rank, kTopK), f32
//     - Input Tokens: max_tokens_per_rank tokens (in input format)
//     - Combine Buffer: max_tokens_per_rank tokens fragments (in combine
//     format)
//     - Aligned to 4KB page boundary (slot_stride_)
// Aligned to 2MB page boundary
//
// Local data: (only visible to the local rank)
// - Grid sync barrier (aligned to 64-byte cache line size)
// - Input TokenTopK Expert ID: (max_tokens_per_rank, kTopK), i32
// - Token metadata: worst-case routed tokens, u64
// - Compact local-combine publication metadata: valid routed tokens, u64
//   (one-stage only)
// - L1 token buffer: worst-case routed tokens * Layout::kInputTokenBytes
// - L1 token weights: worst-case routed tokens * f32
// - L2 arrival masks: one u32 bit mask per M32 routed-token block
// - L2 token buffer: two-stage MXFP4 intermediates for all routed tokens
// - L2 scale buffer: padded E8M0 scales for the L2 token buffer
// - L1 combine buffer: private route rows (one-stage only)
//
// We require the total VMA address space to be within 4GB so that we can use
// buffer_load instructions for efficient memory access.
template <class Layout> class MegaMoEWorkspace {
    static constexpr unsigned kXGpuBarrierCounterOffset = 0;

    static constexpr unsigned kCacheLineBytes = 64;
    static constexpr unsigned kPageBytes = 4096;
    static constexpr unsigned kLargePageBytes = 2 * 1024 * 1024;
    // VMM mappings are page-granular.  Keep each rank's xGPU signal record
    // on an independently owned page, matching the legacy symmetric-buffer
    // layout.  Packing records 64 bytes apart puts every rank's system-scope
    // atomic traffic on one GPU-owned backing page at world size eight.
    static constexpr unsigned kXGpuBarrierRecordBytes = kPageBytes;
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
    static constexpr unsigned kRankSymBufferBase =
        kNumRanks * kXGpuBarrierRecordBytes;
    static constexpr unsigned kSortedTokenBlock = 32;
    static constexpr unsigned kMaxExpertsPerToken =
        kTopK < kNumExpertsPerRank ? kTopK : kNumExpertsPerRank;
    static constexpr unsigned kMaxLocalCombineSlots =
        kNumRanks * kMaxTokensPerRank * kMaxExpertsPerToken;
    static constexpr unsigned long kLocalCombinePublishMetadataCapacityBytes =
        static_cast<unsigned long>(kMaxLocalCombineSlots) *
        sizeof(TokenMetadata);
    static constexpr unsigned long kLocalCombineBufferCapacityBytes =
        static_cast<unsigned long>(kMaxTokensPerRank) *
        Layout::kCombineBufferBytes;
    // PullTokens lays out routed entries per local expert and pads every
    // expert's run to kSortedTokenBlock.  Account for the worst-case top-k
    // fanout plus all per-expert tails, as the legacy dispatcher does.
    static constexpr unsigned kMaxPoolTokens = tal::AlignUp<unsigned>(
        kNumRanks * kMaxTokensPerRank * kMaxExpertsPerToken +
            kNumExpertsPerRank * (kSortedTokenBlock - 1),
        kSortedTokenBlock);

  public:
    static constexpr bool kUsesDirectRemoteCombine =
        UsesDirectRemoteCombine<Layout>::value;
    static constexpr unsigned kMaxPoolBlocks =
        kMaxPoolTokens / kSortedTokenBlock;
    static constexpr unsigned kL2ScaleRows =
        tal::CeilingDiv<unsigned>(kMaxPoolTokens, 256) * 256;
    static constexpr unsigned kL2ScaleCols =
        tal::CeilingDiv<unsigned>(Layout::kInterDim / 32, 8) * 8;
    static constexpr unsigned long kLocalCombinePublishMetadataBytes =
        kUsesDirectRemoteCombine
            ? 0
            : kLocalCombinePublishMetadataCapacityBytes;
    static constexpr unsigned long kLocalCombineBufferBytes =
        kUsesDirectRemoteCombine
            ? 0
            : kLocalCombineBufferCapacityBytes;

  private:
    static constexpr unsigned long kL2ArrivalMaskBytes =
        static_cast<unsigned long>(kMaxPoolBlocks) * sizeof(unsigned);
    static constexpr unsigned long kL2TokenBufferBytes =
        static_cast<unsigned long>(kMaxPoolTokens) * Layout::kInterDim / 2;
    static constexpr unsigned long kL2ScaleBufferBytes =
        static_cast<unsigned long>(kL2ScaleRows) * kL2ScaleCols;
    static constexpr unsigned kSlotStride = tal::AlignUp<unsigned>(
        kNumExperts * sizeof(unsigned long) +
            kNumRanks * kNumExpertsPerRank * sizeof(unsigned long) +
            kNumExpertsPerRank * sizeof(unsigned long) +
            kNumRanks * kNumExpertsPerRank * kMaxTokensPerRank *
                sizeof(unsigned) +
            kMaxTokensPerRank * kTopK * sizeof(float) +
            kMaxTokensPerRank * Layout::kInputTokenBytes +
            kMaxTokensPerRank * Layout::kCombineBufferBytes,
        kRankSymBufferBase);
    static constexpr unsigned kLocalOffsetBase = tal::AlignUp(
        kRankSymBufferBase + kNumRanks * kSlotStride, kLargePageBytes);

    static_assert(kNumExperts % kNumRanks == 0,
                  "the expert count must divide evenly across ranks");
    static_assert(sizeof(TokenMetadata) == sizeof(unsigned long) &&
                      kNumRanks < 256,
                  "");

  public:
    static constexpr unsigned long kLocalDataBytes64 =
        kCacheLineBytes +
        static_cast<unsigned long>(kMaxTokensPerRank) * kTopK *
            sizeof(unsigned) +
        kMaxPoolTokens * sizeof(TokenMetadata) +
        static_cast<unsigned long>(kMaxPoolTokens) * Layout::kInputTokenBytes +
        kMaxPoolTokens * sizeof(float) + kL2ArrivalMaskBytes +
        kL2TokenBufferBytes + kL2ScaleBufferBytes;
    static constexpr unsigned long kLegacyLocalBytes64 =
        kLocalDataBytes64 + kLocalCombinePublishMetadataCapacityBytes +
        kLocalCombineBufferCapacityBytes;
    static constexpr unsigned long kLocalBytes64 =
        kLocalDataBytes64 + kLocalCombinePublishMetadataBytes +
        kLocalCombineBufferBytes;
    static_assert(
        !kUsesDirectRemoteCombine || kLocalBytes64 < kLegacyLocalBytes64,
        "two-stage workspace must omit local-combine publication storage");
    static_assert(
        kUsesDirectRemoteCombine
            ? kLocalBytes64 == kLocalDataBytes64
            : kLocalBytes64 == kLegacyLocalBytes64,
        "invalid stage-specific local-combine workspace layout");
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

    TAL_HOST_DEVICE inline unsigned SendCounterOffset(unsigned rank,
                                                      unsigned expert_idx) {
        [[assume(rank < kNumRanks)]];
        [[assume(expert_idx < kNumExperts)]];
        return RankOffsetBase(rank) + expert_idx * sizeof(unsigned long);
    }

    TAL_HOST_DEVICE inline unsigned
    RecvCounterOffset(unsigned rank, unsigned src_rank,
                      unsigned local_expert_idx) {
        [[assume(rank < kNumRanks)]];
        [[assume(src_rank < kNumRanks)]];
        [[assume(local_expert_idx < kNumExpertsPerRank)]];
        return SendCounterOffset(rank, kNumExperts - 1) +
               sizeof(unsigned long) +
               (src_rank * kNumExpertsPerRank + local_expert_idx) *
                   sizeof(unsigned long);
    }

    TAL_HOST_DEVICE inline unsigned
    RecvSumCounterOffset(unsigned rank, unsigned local_expert_idx) {
        [[assume(rank < kNumRanks)]];
        [[assume(local_expert_idx < kNumExpertsPerRank)]];
        return RecvCounterOffset(rank, kNumRanks - 1, kNumExpertsPerRank - 1) +
               sizeof(unsigned long) + local_expert_idx * sizeof(unsigned long);
    }

    TAL_HOST_DEVICE inline unsigned RecvTokenOffset(unsigned rank,
                                                    unsigned src_rank,
                                                    unsigned local_expert_idx,
                                                    unsigned slot) {
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
    InputTokenTopKExpertWeightOffset(unsigned rank) {
        [[assume(rank < kNumRanks)]];
        return RecvTokenOffset(rank, kNumRanks - 1, kNumExpertsPerRank - 1,
                               kMaxTokensPerRank - 1) +
               sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned InputTokensOffset(unsigned rank) {
        [[assume(rank < kNumRanks)]];
        return InputTokenTopKExpertWeightOffset(rank) +
               kMaxTokensPerRank * kTopK * sizeof(float);
    }

    TAL_HOST_DEVICE inline unsigned CombineBufferOffset(unsigned rank) {
        [[assume(rank < kNumRanks)]];
        return InputTokensOffset(rank) +
               kMaxTokensPerRank * Layout::kInputTokenBytes;
    }

    //
    // Local data per ranks
    //
    TAL_HOST_DEVICE inline unsigned GridSyncBarrierOffset() const {
        return kLocalOffsetBase;
    }

    TAL_HOST_DEVICE inline unsigned LocalCombineSlotOffset() const {
        return GridSyncBarrierOffset() + kMaxGridSyncSlots * sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned InputTokenTopKExpertIDOffset() const {
        return GridSyncBarrierOffset() + kCacheLineBytes;
    }

    TAL_HOST_DEVICE inline unsigned
    TokenMetadataOffset(unsigned pool_token_index) const {
        [[assume(pool_token_index < kMaxPoolTokens)]];
        return InputTokenTopKExpertIDOffset() +
               kMaxTokensPerRank * kTopK * sizeof(unsigned) +
               pool_token_index * sizeof(TokenMetadata);
    }

    TAL_HOST_DEVICE inline unsigned
    LocalCombinePublishMetadataOffset(unsigned slot) const {
        [[assume(slot < kMaxLocalCombineSlots)]];
        return TokenMetadataOffset(kMaxPoolTokens - 1) + sizeof(TokenMetadata) +
               slot * sizeof(TokenMetadata);
    }

    TAL_HOST_DEVICE inline unsigned
    L1TokenBufferOffset(unsigned pool_token_index) const {
        [[assume(pool_token_index < kMaxPoolTokens)]];
        return InputTokenTopKExpertIDOffset() +
               kMaxTokensPerRank * kTopK * sizeof(unsigned) +
               kMaxPoolTokens * sizeof(TokenMetadata) +
               static_cast<unsigned>(kLocalCombinePublishMetadataBytes) +
               pool_token_index * Layout::kInputTokenBytes;
    }

    TAL_HOST_DEVICE inline unsigned
    L1TokenWeightsOffset(unsigned pool_token_index) const {
        [[assume(pool_token_index < kMaxPoolTokens)]];
        return InputTokenTopKExpertIDOffset() +
               kMaxTokensPerRank * kTopK * sizeof(unsigned) +
               kMaxPoolTokens * sizeof(TokenMetadata) +
               static_cast<unsigned>(kLocalCombinePublishMetadataBytes) +
               kMaxPoolTokens * Layout::kInputTokenBytes +
               pool_token_index * sizeof(float);
    }

    TAL_HOST_DEVICE inline unsigned
    L2ArrivalMaskOffset(unsigned pool_block_index) const {
        [[assume(pool_block_index < kMaxPoolBlocks)]];
        return L1TokenWeightsOffset(kMaxPoolTokens - 1) + sizeof(float) +
               pool_block_index * sizeof(unsigned);
    }

    TAL_HOST_DEVICE inline unsigned
    L2TokenBufferOffset(unsigned pool_token_index) const {
        [[assume(pool_token_index < kMaxPoolTokens)]];
        return L1TokenWeightsOffset(kMaxPoolTokens - 1) + sizeof(float) +
               static_cast<unsigned>(kL2ArrivalMaskBytes) +
               pool_token_index * Layout::kInterDim / 2;
    }

    TAL_HOST_DEVICE inline unsigned L2ScaleBufferOffset() const {
        return L1TokenWeightsOffset(kMaxPoolTokens - 1) + sizeof(float) +
               static_cast<unsigned>(kL2ArrivalMaskBytes + kL2TokenBufferBytes);
    }

    TAL_HOST_DEVICE inline unsigned CombineBufferOffset() const {
        return L1TokenWeightsOffset(kMaxPoolTokens - 1) + sizeof(float) +
               static_cast<unsigned>(kL2ArrivalMaskBytes + kL2TokenBufferBytes +
                                     kL2ScaleBufferBytes);
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
