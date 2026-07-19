#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/amd_intrinsics.cuh"
#include "moe/rocm/memory_ops.cuh"

namespace causalflow::petit::rocm::moe {

// Adapter for token-shuffle rows (packed E2M1 payload followed by row-major
// E8M0 scales). It presents the exact input interface consumed by the local
// NativeMxFp4TileOps stage-1 pipeline.
template <class Config> struct MxFp4InputPacked {
    static constexpr unsigned kDim = Config::kDim;
    static constexpr unsigned kHiddenSize = Config::kHiddenSize;
    static constexpr unsigned kTokenBatch = Config::kTokenBatch;
    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kGroupM = kTokenBatch * kNumWarps;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kGroupK = kGroupDim;
    static constexpr unsigned kK128Tiles = kGroupK / 128;
    static constexpr unsigned kK32PerTile = 4;
    static constexpr unsigned kRowVecsPerTile = kK128Tiles * kK32PerTile;
    static constexpr unsigned kActivationFragments = 2 * kK128Tiles;
    static constexpr unsigned kScaleBlockSize = 32;
    static constexpr unsigned kValueBytes = kHiddenSize / 2;
    static constexpr unsigned kScaleBytes = kHiddenSize / kScaleBlockSize;
    static constexpr unsigned kRowStride = tal::AlignUp<unsigned>(
        kValueBytes + kScaleBytes, sizeof(uint4));
    static constexpr unsigned kPaddedScaleBytes = kRowStride - kValueBytes;
    static constexpr unsigned kScaleVectorsPerRow =
        kPaddedScaleBytes / sizeof(uint4);
    static constexpr unsigned kScaleWordsPerRow =
        kPaddedScaleBytes / sizeof(unsigned);
    static constexpr unsigned kScaleTiles = Config::kComputeHiddenSize /
                                             kGroupDim;
    static constexpr unsigned kScaleWords = kScaleTiles * kWarpSize;
    static constexpr unsigned kScaleStages = 2;
    static constexpr unsigned kScaleWordsPerStage =
        kScaleWords / kScaleStages;
    static constexpr unsigned kScaleVectors =
        kScaleWords * sizeof(unsigned) / sizeof(uint4);
    static constexpr unsigned kScaleVectorsPerStage =
        kScaleVectors / kScaleStages;
    static constexpr unsigned kWarpsPerScaleStage =
        kNumWarps / kScaleStages;

    struct Shm {
        uint4 act[kThreads];
        unsigned scale[kScaleWordsPerStage];
    };

    static_assert(kNumWarps == 4 && kTokenBatch == 8);
    static_assert(kGroupDim == 256);
    static_assert(kRowVecsPerTile * kTokenBatch == kWarpSize);
    static_assert(Config::kComputeHiddenSize % kGroupDim == 0);
    static_assert(kScaleTiles % kScaleStages == 0);
    static_assert(kNumWarps % kScaleStages == 0);
    static_assert(kGroupM * kPaddedScaleBytes ==
                  kScaleWords * sizeof(unsigned));
    static_assert(kPaddedScaleBytes == kScaleTiles * 8);
    static_assert(kScaleVectors <= kNumWarps * kWarpSize);

    template <class Workspace>
    TAL_DEVICE void Initialize(Workspace &workspace, unsigned work_id,
                               unsigned m) {
        workspace_ = workspace.br_;
        activation_offset_ =
            workspace.L1TokenBufferOffset(work_id * kTokenBatch * kNumWarps);
        m_ = m;
        values_offset_vec_ = 0;
        scale_tile_ = 0;
    }

    TAL_DEVICE void FetchAsync(uint4 *shm_x, unsigned wid, unsigned wtid,
                               const unsigned tokens[kTokenBatch]) {
        const unsigned token_idx = wtid / kRowVecsPerTile;
        const unsigned row_vec =
            wtid - token_idx * kRowVecsPerTile;
        const unsigned source_row_vec =
            row_vec ^ (token_idx & (kRowVecsPerTile - 1));
        const unsigned dst_idx = wid * kWarpSize;
        auto *lds = (__attribute__((address_space(3))) unsigned *)(
            shm_x + dst_idx);
        const unsigned row = tokens[token_idx];
        const unsigned first_element =
            values_offset_vec_ * kScaleBlockSize +
            source_row_vec * kScaleBlockSize;
        const unsigned actual =
            activation_offset_ + row * kRowStride +
            (values_offset_vec_ + source_row_vec) * sizeof(uint4);
        const unsigned offset =
            row < m_ && first_element < kHiddenSize ? actual : ~0u;
        workspace_.template LoadLds<BufferResource::kNone, sizeof(uint4), 0>(
            lds, offset, 0);
        values_offset_vec_ += kGroupDim / kScaleBlockSize;
    }

    TAL_DEVICE void FetchScaleAsync(unsigned *shm_scale, unsigned wid,
                                    unsigned wtid,
                                    const unsigned *,
                                    unsigned) const {}

    template <unsigned kStages>
    TAL_DEVICE void PrepareScales(Shm (&shm)[kStages], unsigned wid,
                                  unsigned wtid) {
        static_assert(kStages == kScaleStages);
        LoadScalesAsync(shm, wid, wtid);
        amdgcn_s_waitcnt_barrier<0>();
        RepackScales(shm, wid * kWarpSize + wtid);
    }

    TAL_DEVICE void FetchToRegs(uint4 regs[kActivationFragments],
                                const uint4 *shm_x, unsigned wtid) const {
        const unsigned row = wtid & 15u;
        const unsigned vector = wtid / 16;
        const unsigned row_base = row * kRowVecsPerTile;
        const auto swizzle = [=](unsigned value) {
            return value ^ (row & (kRowVecsPerTile - 1));
        };
        const uint4 *k0 = shm_x + row_base + swizzle(vector);
        const uint4 *k1 =
            shm_x + row_base + swizzle(vector + kK32PerTile);
        static constexpr unsigned kNextRow = 16 * kRowVecsPerTile;
        regs[0] = k0[0];
        regs[1] = k1[0];
        regs[2] = k0[kNextRow];
        regs[3] = k1[kNextRow];
    }

    TAL_DEVICE unsigned FetchScaleToReg(const unsigned *shm_scale,
                                        unsigned wtid) const {
        return shm_scale[scale_tile_ / kScaleStages * kWarpSize + wtid];
    }

    TAL_DEVICE void AdvanceScaleStep() {
        ++scale_tile_;
    }

  private:
    TAL_DEVICE void LoadScalesAsync(Shm (&shm)[kScaleStages], unsigned wid,
                                    unsigned wtid) {
        const unsigned stage = wid / kWarpsPerScaleStage;
        const unsigned wave_in_stage = wid % kWarpsPerScaleStage;
        const unsigned local_vector = wave_in_stage * kWarpSize + wtid;
        const unsigned vector =
            stage * kScaleVectorsPerStage + local_vector;
        const unsigned stage_word =
            wave_in_stage * kWarpSize < kScaleVectorsPerStage
                ? wave_in_stage * kWarpSize *
                      (sizeof(uint4) / sizeof(unsigned))
                : 0;
        auto *lds = (__attribute__((address_space(3))) unsigned *)(
            shm[stage].scale + stage_word);
        const unsigned row = vector / kScaleVectorsPerRow;
        const unsigned row_vector = vector - row * kScaleVectorsPerRow;
        BufferResource scales = workspace_;
        scales.v.range = activation_offset_ + kGroupM * kRowStride;
        const unsigned src =
            local_vector < kScaleVectorsPerStage
                ? activation_offset_ + kValueBytes + row * kRowStride +
                      row_vector * sizeof(uint4)
                : ~0u;
        scales.template LoadLds<BufferResource::kNone, sizeof(uint4), 0>(
            lds, src, 0);
    }

    TAL_DEVICE void RepackScales(Shm (&shm)[kScaleStages], unsigned tid) {
        static constexpr unsigned kScaleTasks = kScaleWords / 4;
        static constexpr unsigned kNumQuads = kThreads / 4;
        static constexpr unsigned kTasksPerThread =
            tal::CeilingDiv<unsigned>(kScaleTasks, kNumQuads);
        const unsigned quad = tid / 4;
        const unsigned quad_lane = tid % 4;
        unsigned packed[kTasksPerThread];
#pragma unroll
        for (unsigned i = 0; i < kTasksPerThread; ++i) {
            const unsigned task = (quad + i * kNumQuads) % kScaleTasks;
            const unsigned tile = task / 16;
            const unsigned row16 = task % 16;
            const unsigned row = row16 + 16 * (quad_lane & 1);
            const unsigned half = quad_lane >> 1;
            const unsigned raw_word =
                row * kScaleWordsPerRow + tile * 2 + half;
            const unsigned raw_stage = raw_word / kScaleWordsPerStage;
            const unsigned raw =
                shm[raw_stage]
                    .scale[raw_word - raw_stage * kScaleWordsPerStage];
            const unsigned v0 =
                __builtin_amdgcn_mov_dpp(raw, 0x00, 0xf, 0xf, false);
            const unsigned v1 =
                __builtin_amdgcn_mov_dpp(raw, 0x55, 0xf, 0xf, false);
            const unsigned v2 =
                __builtin_amdgcn_mov_dpp(raw, 0xaa, 0xf, 0xf, false);
            const unsigned v3 =
                __builtin_amdgcn_mov_dpp(raw, 0xff, 0xf, 0xf, false);
            const unsigned select_pair =
                0x0c0c0400 + quad_lane * 0x00000101;
            const unsigned pair01 =
                amdgcn_perm_b32(v1, v0, select_pair);
            const unsigned pair23 =
                amdgcn_perm_b32(v3, v2, select_pair);
            packed[i] = amdgcn_perm_b32(pair23, pair01, 0x05040100);
        }

        __syncthreads();
#pragma unroll
        for (unsigned i = 0; i < kTasksPerThread; ++i) {
            const unsigned task = (quad + i * kNumQuads) % kScaleTasks;
            const unsigned tile = task / 16;
            const unsigned row16 = task % 16;
            const unsigned stage = tile % kScaleStages;
            const unsigned stage_tile = tile / kScaleStages;
            shm[stage].scale[stage_tile * kWarpSize + quad_lane * 16 +
                             row16] = packed[i];
        }
        __syncthreads();
    }

  public:
    BufferResource workspace_;
    unsigned activation_offset_ = 0;
    unsigned m_ = 0;
    unsigned values_offset_vec_ = 0;
    unsigned scale_tile_ = 0;
};

} // namespace causalflow::petit::rocm::moe
