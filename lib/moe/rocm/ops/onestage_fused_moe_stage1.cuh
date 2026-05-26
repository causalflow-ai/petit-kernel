#pragma once

#include "moe/rocm/fused_moe.cuh"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <class Trait, unsigned kGroupDim, unsigned kTokenBatch>
struct OnestageFusedMoEStage1DoubleBufferOp {
    static constexpr unsigned kStage = 2;
    using Input = typename Trait::Input;
    static constexpr unsigned kAccumFragments = Trait::kAccumFragments;

    struct Shm {
        typename Trait::Shm x[kStage];
    };

    __device__ static void Run(float4 h[kAccumFragments], Shm &shm,
                               Trait &trait, unsigned dim, unsigned tid,
                               unsigned wid, unsigned wtid,
                               const uint2 token_select,
                               const unsigned tokens[kTokenBatch], unsigned m) {
        float4 t_gate[kAccumFragments], t_up[kAccumFragments];
        typename Trait::InputRegs x[kStage];
        ClearMat(t_gate);
        ClearMat(t_up);

        unsigned curr = 0;
        trait.PrefetchInput(&shm.x[curr], wid, wtid, token_select, tokens, m);
        trait.LoadInitial(tid, wid, wtid);

        amdgcn_s_waitcnt_barrier<0>();
        trait.ReadInput(x[curr], &shm.x[curr], wtid);

        for (unsigned d = 0; d < dim; d += 2 * kGroupDim) {
            __syncthreads();
#pragma unroll
            for (unsigned curr = 0, next = 1; curr < 2;
                 curr++, next = 1 - curr) {
                // The up/gate projection loop is dominated by W13/input global
                // loads and 128 MFMA instructions.
                HotLoopScheduler<128, 6, 0, 0, 2>();
                trait.PrefetchInput(&shm.x[next], wid, wtid, token_select,
                                    tokens, m);
                trait.Matmul(t_gate, t_up, x[curr], tid, wid, wtid);

                if (d + kGroupDim >= dim) {
                    break;
                }

                amdgcn_s_waitcnt_barrier<0>();
                trait.ReadInput(x[next], &shm.x[next], wtid);
            }
        }

        for (unsigned i = 0; i < kAccumFragments; i++) {
            h[i] = SiluDot(t_gate[i], t_up[i]);
        }
    }
};

template <class Trait, unsigned kGroupDim, unsigned kTokenBatch>
struct OnestageFusedMoEStage1SingleBufferOp {
    static constexpr unsigned kStage = 1;
    using Input = typename Trait::Input;
    static constexpr unsigned kAccumFragments = Trait::kAccumFragments;

    struct Shm {
        typename Trait::Shm x[kStage];
    };

    __device__ static void Run(float4 h[kAccumFragments], Shm &shm, Trait &trait,
                               unsigned dim, unsigned tid, unsigned wid,
                               unsigned wtid, const uint2 token_select,
                               const unsigned tokens[kTokenBatch], unsigned m) {
        float4 t_gate[kAccumFragments], t_up[kAccumFragments];
        typename Trait::InputRegs x;
        ClearMat(t_gate);
        ClearMat(t_up);

        trait.LoadInitial(tid, wid, wtid);

        for (unsigned d = 0; d < dim; d += kGroupDim) {
            trait.PrefetchInput(&shm.x[0], wid, wtid, token_select, tokens, m);
            amdgcn_s_waitcnt_barrier<0>();
            trait.ReadInput(x, &shm.x[0], wtid);
            trait.Matmul(t_gate, t_up, x, tid, wid, wtid);
        }

        for (unsigned i = 0; i < kAccumFragments; i++) {
            h[i] = SiluDot(t_gate[i], t_up[i]);
        }
    }
};

} // namespace causalflow::petit::rocm::moe
