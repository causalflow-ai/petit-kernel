#pragma once

#include "moe/rocm/fused_moe.cuh"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <class Trait, unsigned kGroupDim, unsigned kTokenBatch>
struct FusedMoEBlockScaleFP8Stage1Op {
    static constexpr unsigned kStage = 2;
    using Input = typename Trait::Input;
    static constexpr unsigned kAccumFragments = Trait::kAccumFragments;
    static constexpr unsigned kActivationFragments =
        Trait::kActivationFragments;

    struct Shm {
        struct {
            unsigned act[Input::kShmInputElements];
            float scale[Input::kThreads];
        } x[kStage];
    };

    __device__ static void Run(float4 h[kAccumFragments], Shm &shm,
                               Trait &trait, unsigned dim, unsigned tid,
                               unsigned wid, unsigned wtid,
                               const uint2 token_select,
                               const unsigned tokens[kTokenBatch], unsigned m) {
        float4 t_gate[kAccumFragments], t_up[kAccumFragments];
        uint4 x[kStage][kActivationFragments];
        float4 scale_x[kStage];
        ClearMat(t_gate);
        ClearMat(t_up);

        unsigned curr = 0;
        trait.PrefetchInput(shm.x[curr].act, shm.x[curr].scale, wid, wtid,
                            token_select, tokens, m);
        trait.LoadInitial(tid, wid, wtid);

        amdgcn_s_waitcnt_barrier<0>();
        trait.ReadInput(x[curr], scale_x[curr], shm.x[curr].act,
                        shm.x[curr].scale, wtid);

        for (unsigned d = 0; d < dim; d += 2 * kGroupDim) {
            __syncthreads();
#pragma unroll
            for (unsigned curr = 0, next = 1; curr < 2;
                 curr++, next = 1 - curr) {
                // The up/gate projection loop is dominated by W13/input global
                // loads and 128 MFMA instructions.
                HotLoopScheduler<128, 6, 0, 0, 2>();
                trait.PrefetchInput(shm.x[next].act, shm.x[next].scale, wid,
                                    wtid, token_select, tokens, m);
                trait.Matmul(t_gate, t_up, x[curr], scale_x[curr], tid, wid,
                             wtid);

                if (d + kGroupDim >= dim) {
                    break;
                }

                amdgcn_s_waitcnt_barrier<0>();
                trait.ReadInput(x[next], scale_x[next], shm.x[next].act,
                                shm.x[next].scale, wtid);
            }
        }

        for (unsigned i = 0; i < kAccumFragments; i++) {
            h[i] = SiluDot(t_gate[i], t_up[i]);
        }
    }
};

} // namespace causalflow::petit::rocm::moe
