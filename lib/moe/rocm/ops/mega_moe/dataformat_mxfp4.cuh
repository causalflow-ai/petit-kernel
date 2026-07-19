#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "gemm/rocm/amd_intrinsics.cuh"

namespace causalflow::petit::rocm::moe {

template <unsigned kHiddenSize, unsigned kWarpsForPull, unsigned kNumWarps>
struct RemoteMxFp4Transport {
    static constexpr unsigned kScaleBytes = kHiddenSize / 32;
    // Workspace rows are 16-byte aligned so each lane can transfer one uint4.
    static constexpr unsigned kInputTokenBytes =
        tal::AlignUp(kHiddenSize / 2 + kScaleBytes, 16u);
    static constexpr unsigned kRowBytes = kInputTokenBytes;
    static constexpr unsigned kRowVecs = kRowBytes / sizeof(uint4);
    static_assert(kWarpsForPull > 0 && kWarpsForPull <= kNumWarps);
    static_assert(kNumWarps % kWarpsForPull == 0);
    // A pull group owns one token.  Its consecutive waves cooperatively copy
    // the row, so wide MXFP4 rows (for example GPT-OSS's padded 3072 hidden
    // size) can use a two-wave pull.  Out-of-range lanes use the hardware's
    // buffer OOB behaviour for the aligned tail.
    static_assert(kRowVecs <= kWarpSize * kWarpsForPull);

    struct Shm {
        // buffer_load_lds writes a contiguous vector for every lane in every
        // participating wave.  Each pull group reserves a wave footprint per
        // warp, which totals one footprint per block warp.
        uint4 rows[kWarpSize * kNumWarps];
    };

    template <class Workspace>
    TAL_DEVICE static void LoadLdsAsync(
        Workspace &ws, __attribute__((address_space(3))) Shm *shm,
        unsigned wid, unsigned wtid, unsigned src_offset) {
        const unsigned group = wid / kWarpsForPull;
        const unsigned wave_in_group = wid % kWarpsForPull;
        const unsigned vec = wave_in_group * kWarpSize + wtid;
        using lds_u32 = unsigned __attribute__((address_space(3)));
        auto ptr = (lds_u32 *)shm +
                   (group * kWarpsForPull + wave_in_group) * kWarpSize *
                       (sizeof(uint4) / sizeof(unsigned));
        // buffer_load_lds takes a warp-uniform LDS base and lays the lane
        // results out consecutively.  Keep the workspace descriptor uniform
        // and put the complete address in voffset; ~0u is OOB because soffset
        // is zero and the workspace is smaller than 4 GiB.
        const unsigned voffset =
            vec < kRowVecs ? src_offset + vec * sizeof(uint4) : ~0u;
        // The pre-pull grid epilogue is control-only.  Read peer-published
        // rows coherently instead of invalidating every CTA's payload cache.
        ws.br_.template LoadLds<BufferResource::kSC1Bit, sizeof(uint4), 0>(
            ptr, voffset, 0);
    }

    template <class Workspace>
    TAL_DEVICE static void StoreLds(
        Workspace &ws, __attribute__((address_space(3))) Shm *shm,
        unsigned wid, unsigned wtid, unsigned dst_offset) {
        const unsigned group = wid / kWarpsForPull;
        const unsigned wave_in_group = wid % kWarpsForPull;
        const unsigned vec = wave_in_group * kWarpSize + wtid;
        const bool in_bounds = vec < kRowVecs;
        using lds_u4 = uint4 __attribute__((address_space(3)));
        auto ptr = reinterpret_cast<lds_u4 *>(shm) +
                   (group * kWarpsForPull + wave_in_group) * kWarpSize;
        // HIP's uint4 copy constructor cannot bind an LDS reference.  This is
        // a representation-preserving register copy; the conditional pointer
        // itself remains in LDS address space.
        const uint4 v = __builtin_bit_cast(
            uint4, *GetConditionShmPtr(
                       ptr + wtid, in_bounds));
        const unsigned voffset =
            in_bounds ? dst_offset + vec * sizeof(uint4) : ~0u;
        ws.br_.template Store<BufferResource::kNone>(voffset, 0, v);
    }

    template <class Workspace>
    TAL_DEVICE static void Copy(
        Workspace &ws, __attribute__((address_space(3))) Shm *shm,
        unsigned wid, unsigned wtid, unsigned src_offset, unsigned dst_offset) {
        LoadLdsAsync(ws, shm, wid, wtid, src_offset);
        amdgcn_s_waitcnt<0, -1, 0>();
        // Each pull wave owns a disjoint LDS row.  A block-wide barrier here
        // is unsafe because persistent blocks may have different numbers of
        // routed tokens to pull.
        __syncwarp();
        StoreLds(ws, shm, wid, wtid, dst_offset);
    }
};

} // namespace causalflow::petit::rocm::moe
