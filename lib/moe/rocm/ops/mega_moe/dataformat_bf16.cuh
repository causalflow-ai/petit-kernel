#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "gemm/rocm/amd_intrinsics.cuh"

#include <hip/hip_bf16.h>

namespace causalflow::petit::rocm::moe {

template <unsigned kLogicalHiddenSize, unsigned kComputeHiddenSize,
          unsigned kWarpsForPull, unsigned kNumWarps>
struct RemoteBf16Transport {
    static constexpr unsigned kInputTokenBytes =
        kComputeHiddenSize * sizeof(__hip_bfloat16);
    static constexpr unsigned kCopyBytes =
        kLogicalHiddenSize * sizeof(__hip_bfloat16);
    static constexpr unsigned kRowVecs = kCopyBytes / sizeof(uint4);
    static constexpr unsigned kVecsPerPass = kWarpSize * kWarpsForPull;
    static constexpr unsigned kPasses =
        tal::CeilingDiv<unsigned>(kRowVecs, kVecsPerPass);

    static_assert(kWarpsForPull == 2,
                  "BF16 activation pulls use exactly two waves");
    static_assert(kNumWarps % kWarpsForPull == 0);
    static_assert(kComputeHiddenSize >= kLogicalHiddenSize);
    static_assert(kInputTokenBytes % sizeof(uint4) == 0);
    static_assert(kCopyBytes % sizeof(uint4) == 0,
                  "BF16 activation rows must be uint4 aligned");

    struct Shm {
        uint4 rows[kWarpSize * kNumWarps];
    };

    template <class Workspace>
    TAL_DEVICE static void LoadLdsAsync(
        Workspace &ws, __attribute__((address_space(3))) Shm *shm,
        unsigned wid, unsigned wtid, unsigned src_offset,
        unsigned pass) {
        BufferResource br = ws.br_;
        br.v.range = src_offset + kCopyBytes;
        const unsigned group = wid / kWarpsForPull;
        const unsigned wave_in_group = wid % kWarpsForPull;
        using lds_u32 = unsigned __attribute__((address_space(3)));
        auto ptr = reinterpret_cast<lds_u32 *>(shm) +
                   (group * kWarpsForPull + wave_in_group) * kWarpSize *
                       (sizeof(uint4) / sizeof(unsigned));
        const unsigned vec =
            pass * kVecsPerPass + wave_in_group * kWarpSize + wtid;
        // Pair with the control-only pre-pull grid epilogue.
        br.LoadLds<BufferResource::kSC1Bit, sizeof(uint4), 0>(
            ptr, vec * sizeof(uint4), src_offset);
    }

    template <class Workspace>
    TAL_DEVICE static void StoreLds(
        Workspace &ws, __attribute__((address_space(3))) Shm *shm,
        unsigned wid, unsigned wtid, unsigned dst_offset,
        unsigned pass) {
        const unsigned group = wid / kWarpsForPull;
        const unsigned wave_in_group = wid % kWarpsForPull;
        const unsigned vec =
            pass * kVecsPerPass + wave_in_group * kWarpSize + wtid;
        using lds_u4 = uint4 __attribute__((address_space(3)));
        auto ptr = reinterpret_cast<lds_u4 *>(shm) +
                   (group * kWarpsForPull + wave_in_group) * kWarpSize;
        const uint4 v = __builtin_bit_cast(
            uint4, *GetConditionShmPtr(ptr + wtid, vec < kRowVecs));
        BufferResource br = ws.br_;
        br.v.range = dst_offset + kCopyBytes;
        br.template Store<BufferResource::kNone>(vec * sizeof(uint4),
                                                  dst_offset, v);
    }

    template <class Workspace>
    TAL_DEVICE static void Copy(
        Workspace &ws, __attribute__((address_space(3))) Shm *shm,
        unsigned wid, unsigned wtid, unsigned src_offset,
        unsigned dst_offset) {
#pragma unroll
        for (unsigned pass = 0; pass < kPasses; ++pass) {
            LoadLdsAsync(ws, shm, wid, wtid, src_offset, pass);
            amdgcn_s_waitcnt<0, -1, 0>();
            __syncwarp();
            StoreLds(ws, shm, wid, wtid, dst_offset, pass);
        }
    }
};

} // namespace causalflow::petit::rocm::moe
