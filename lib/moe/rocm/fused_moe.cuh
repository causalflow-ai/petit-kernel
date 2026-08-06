#pragma once

#include "gemm/rocm/amd_intrinsics.cuh"
#include "moe/rocm/fused_moe.h"

namespace causalflow::petit::rocm::moe {

struct TokenMetadata {
    unsigned token_topk_idx;
    unsigned src_rank : 8;
    unsigned local_combine_slot : 24;
};
static_assert(sizeof(TokenMetadata) == sizeof(unsigned long), "");

__device__ static inline float4 Fma4(float4 a, float s, float4 c) {
    const auto *a2 = reinterpret_cast<const float2 *>(&a);
    const auto *c2 = reinterpret_cast<const float2 *>(&c);
    float2 s2 = {s, s};
    float4 r;
    auto *r2 = reinterpret_cast<float2 *>(&r);
    r2[0] = amdgcn_pk_fma_f32(s2, c2[0], a2[0]);
    r2[1] = amdgcn_pk_fma_f32(s2, c2[1], a2[1]);
    return r;
}

template <unsigned kInstMFMA, unsigned kInstVmemRead = 0,
          unsigned kInstDsRead = 0, unsigned kInstDsWrite = 0,
          unsigned kInstVALU = 0>
__device__ static inline void HotLoopScheduler() {
#if HAS_AMD_SCHED_GROUP_BARRIER && HAS_AMD_SCHED_BARRIER
    static constexpr unsigned kSchedGroupId = 0;
    static constexpr unsigned kInstIssue =
        kInstVmemRead + kInstDsRead + kInstDsWrite + kInstVALU;

    if constexpr (kInstMFMA > 0 && kInstIssue > 0) {
        static constexpr unsigned kInstMFMAPerIssue =
            kInstMFMA / kInstIssue > 12 ? 4
                                        : (kInstMFMA / kInstIssue > 6 ? 2 : 1);

        for (unsigned i = 0; i < kInstDsWrite; ++i) {
            amdgcn_sched_group_barrier<0x200, 1, kSchedGroupId>();
            amdgcn_sched_group_barrier<0x8, kInstMFMAPerIssue,
                                       kSchedGroupId>();
        }
        for (unsigned i = 0; i < kInstVmemRead; ++i) {
            amdgcn_sched_group_barrier<0x20, 1, kSchedGroupId>();
            amdgcn_sched_group_barrier<0x8, kInstMFMAPerIssue,
                                       kSchedGroupId>();
        }
        for (unsigned i = 0; i < kInstDsRead; ++i) {
            amdgcn_sched_group_barrier<0x100, 1, kSchedGroupId>();
            amdgcn_sched_group_barrier<0x8, kInstMFMAPerIssue,
                                       kSchedGroupId>();
        }
        for (unsigned i = 0; i < kInstVALU; ++i) {
            amdgcn_sched_group_barrier<0x1, 1, kSchedGroupId>();
            amdgcn_sched_group_barrier<0x8, kInstMFMAPerIssue,
                                       kSchedGroupId>();
        }
    }
    amdgcn_sched_barrier<0>();
#endif
}

template <typename Mat> __device__ static inline void ClearMat(Mat &mat) {
    for (int i = 0; i < sizeof(Mat) / sizeof(float); i++) {
        reinterpret_cast<float *>(&mat)[i] = 0.f;
    }
}

} // namespace causalflow::petit::rocm::moe
