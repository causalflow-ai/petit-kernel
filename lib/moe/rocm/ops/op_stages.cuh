#pragma once

#include "causalflow/petit/tal/tensor/layout.h"
#include "moe/rocm/fused_moe.cuh"
#include "moe/rocm/ops/activation.cuh"
#include "moe/rocm/ops/schedule_tiles.cuh"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <class TileOps_> struct W13TileSchedule {
    using TileOps = TileOps_;
    using Config = typename TileOps::Config;
    using Input = typename Config::Input;
    using Weight = typename Config::W13;
    using Bias = typename Config::Bias;
    using Shm = typename TileOps::Shm;
    using InputRegs = typename TileOps::InputRegs;

    static constexpr unsigned kNumWarps = TileOps::kNumWarps;
    static constexpr unsigned kAccumFragments = TileOps::kAccumFragments;
    static constexpr unsigned kActivationFragments =
        TileOps::kActivationFragments;

    Input &input;
    Weight &w1, &w3;
    Bias &w1_bias, &w3_bias;
    typename TileOps::Tile w1_tile, w3_tile;

    __device__ W13TileSchedule(Input &input, Weight &w1, Weight &w3,
                               Bias &w1_bias, Bias &w3_bias)
        : input(input), w1(w1), w3(w3), w1_bias(w1_bias), w3_bias(w3_bias) {}

    __device__ void InitializeBias(const void *w13_bias, unsigned expert_id,
                                   unsigned inter_dim, unsigned tile_k) {
        const unsigned projection_stride = Bias::PackedStride(inter_dim);
        const unsigned expert_stride = 2 * projection_stride;
        const void *w3_bias_ptr =
            w13_bias == nullptr ? nullptr
                                : reinterpret_cast<const char *>(w13_bias) +
                                      projection_stride * Bias::kElementBytes;
        w1_bias.Initialize(w13_bias, expert_id, inter_dim, tile_k,
                           expert_stride);
        w3_bias.Initialize(w3_bias_ptr, expert_id, inter_dim, tile_k,
                           expert_stride);
    }

    __device__ void PrefetchInput(Shm *shm, unsigned wid, unsigned wtid,
                                  const unsigned tokens[Input::kTokenBatch],
                                  unsigned m) {
        TileOps::PrefetchInput(input, shm, wid, wtid, tokens, m);
    }

    __device__ void ReadInput(InputRegs &regs, const Shm *shm,
                              unsigned wtid) {
        TileOps::ReadInput(input, regs, shm, wtid);
    }

    __device__ void LoadInitial(unsigned tid, unsigned wid, unsigned wtid) {
        TileOps::Load(w1, w1_tile, tid, wid, wtid);
        TileOps::Load(w3, w3_tile, tid, wid, wtid);
        w1.template AdvanceStep<0, TileOps::kKStages>();
        w3.template AdvanceStep<0, TileOps::kKStages>();
    }

    __device__ void Matmul(float4 gate[kAccumFragments],
                           float4 up[kAccumFragments],
                           const InputRegs &input_regs, unsigned tid,
                           unsigned wid, unsigned wtid) {
        TileOps::Matmul(gate, w1_tile, input_regs, wtid);
        TileOps::Matmul(up, w3_tile, input_regs, wtid);
        TileOps::Load(w1, w1_tile, tid, wid, wtid);
        TileOps::Load(w3, w3_tile, tid, wid, wtid);
        w1.template AdvanceStep<0, TileOps::kKStages>();
        w3.template AdvanceStep<0, TileOps::kKStages>();
    }

    __device__ void AddBias(float4 gate[kAccumFragments],
                            float4 up[kAccumFragments], unsigned tid) const {
        w1_bias.AddToAccumulator(gate, 0, tid);
        w3_bias.AddToAccumulator(up, 0, tid);
    }
};

template <class TileOps_> struct W2TileSchedule {
    using TileOps = TileOps_;
    using Config = typename TileOps::Config;
    using Weight = typename Config::W2;
    using Bias = typename Config::Bias;
    using CShuffle = typename TileOps::CShuffle;
    using InputRegs = typename TileOps::InputRegs;

    static constexpr unsigned kNumWarps = TileOps::kNumWarps;
    static constexpr unsigned kAccumFragments = TileOps::kAccumFragments;
    static constexpr unsigned kActivationFragments =
        TileOps::kActivationFragments;
    static constexpr unsigned kOutputPacksPerToken =
        TileOps::kOutputPacksPerToken;

    Weight &weight;
    Bias &bias;
    typename TileOps::Tile stages[2];

    __device__ W2TileSchedule(Weight &weight, Bias &bias)
        : weight(weight), bias(bias) {}

    __device__ void InitializeBias(const void *bias_ptr, unsigned expert_id,
                                   unsigned dim, unsigned tile_k) {
        const unsigned expert_stride = Bias::PackedStride(dim);
        const unsigned bias_tile = TileOps::kStage2BiasUsesTileK ? tile_k : 0;
        bias.Initialize(bias_ptr, expert_id, dim, bias_tile, expert_stride);
    }

    __device__ void LoadStage(unsigned stage, unsigned tid, unsigned wid,
                              unsigned wtid) {
        TileOps::Load(weight, stages[stage], tid, wid, wtid);
        weight.template AdvanceStep<1, 0>();
    }

    __device__ void Matmul(float4 t[kAccumFragments], const InputRegs &input,
                           unsigned stage, unsigned wtid) const {
        TileOps::Matmul(t, stages[stage], input, wtid);
    }

    __device__ void AddBias(float4 t[kAccumFragments], unsigned tile_d,
                            unsigned tid) const {
        bias.AddToAccumulator(t, tile_d, tid);
    }
};

template <class TileSchedule> struct OnestageFusedMoEStage1DoubleBufferOp {
    using Config = typename TileSchedule::Config;
    using ActivationOp = typename Config::ActivationOp;
    static constexpr unsigned kStage = 2;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kTokenBatch = Config::kTokenBatch;
    using Input = typename TileSchedule::Input;
    static constexpr unsigned kAccumFragments = TileSchedule::kAccumFragments;

    struct Shm {
        typename TileSchedule::Shm x[kStage];
    };

    __device__ static void Run(float4 h[kAccumFragments], Shm &shm,
                               TileSchedule &tiles, unsigned dim, unsigned tid,
                               unsigned wid, unsigned wtid,
                               const unsigned tokens[kTokenBatch], unsigned m) {
        float4 t_gate[kAccumFragments], t_up[kAccumFragments];
        typename TileSchedule::InputRegs x[kStage];
        ClearMat(t_gate);
        ClearMat(t_up);

        unsigned curr = 0;
        tiles.PrefetchInput(&shm.x[curr], wid, wtid, tokens, m);
        tiles.LoadInitial(tid, wid, wtid);

        amdgcn_s_waitcnt_barrier<0>();
        tiles.ReadInput(x[curr], &shm.x[curr], wtid);

        for (unsigned d = 0; d < dim; d += 2 * kGroupDim) {
            __syncthreads();
#pragma unroll
            for (unsigned curr = 0, next = 1; curr < 2;
                 curr++, next = 1 - curr) {
                // The up/gate projection loop is dominated by W13/input global
                // loads and 128 MFMA instructions.
                HotLoopScheduler<128, 6, 0, 0, 2>();
                tiles.PrefetchInput(&shm.x[next], wid, wtid, tokens, m);
                tiles.Matmul(t_gate, t_up, x[curr], tid, wid, wtid);

                if (d + kGroupDim >= dim) {
                    break;
                }

                amdgcn_s_waitcnt_barrier<0>();
                tiles.ReadInput(x[next], &shm.x[next], wtid);
            }
        }

        tiles.AddBias(t_gate, t_up, tid);
        for (unsigned i = 0; i < kAccumFragments; i++) {
            h[i] = ActivationOp::Apply(t_gate[i], t_up[i]);
        }
    }
};

template <class TileSchedule> struct OnestageFusedMoEStage1SingleBufferOp {
    using Config = typename TileSchedule::Config;
    using ActivationOp = typename Config::ActivationOp;
    static constexpr unsigned kStage = 1;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kTokenBatch = Config::kTokenBatch;
    using Input = typename TileSchedule::Input;
    static constexpr unsigned kAccumFragments = TileSchedule::kAccumFragments;

    struct Shm {
        typename TileSchedule::Shm x[kStage];
    };

    __device__ static void Run(float4 h[kAccumFragments], Shm &shm,
                               TileSchedule &tiles, unsigned dim, unsigned tid,
                               unsigned wid, unsigned wtid,
                               const unsigned tokens[kTokenBatch], unsigned m) {
        float4 t_gate[kAccumFragments], t_up[kAccumFragments];
        typename TileSchedule::InputRegs x;
        ClearMat(t_gate);
        ClearMat(t_up);

        tiles.LoadInitial(tid, wid, wtid);

        for (unsigned d = 0; d < dim; d += kGroupDim) {
            tiles.PrefetchInput(&shm.x[0], wid, wtid, tokens, m);
            amdgcn_s_waitcnt_barrier<0>();
            tiles.ReadInput(x, &shm.x[0], wtid);
            tiles.Matmul(t_gate, t_up, x, tid, wid, wtid);
        }

        tiles.AddBias(t_gate, t_up, tid);
        for (unsigned i = 0; i < kAccumFragments; i++) {
            h[i] = ActivationOp::Apply(t_gate[i], t_up[i]);
        }
    }
};


__device__ static inline void
BufferAtomicWriteBf16x2(const BufferResource &base, unsigned vo,
                        unsigned value) {
    asm volatile("buffer_atomic_pk_add_bf16 %2, %1, %0, 0 offen\n\t"
                 :
                 : "s"(base.content), "v"(vo), "v"(value)
                 : "memory");
}

template <unsigned kAccumFragments>
__device__ static inline void MultRouteWeights(float4 t[kAccumFragments],
                                               float2 rw2) {
    static_assert(kAccumFragments % 2 == 0, "");
    for (int i = 0; i < 2; i++) {
        float rw = reinterpret_cast<const float *>(&rw2)[i];
        float2 rw_pk = {rw, rw};
        for (unsigned j = 0; j < kAccumFragments / 2; j++) {
            float4 &f = t[j * 2 + i];
            auto xy =
                amdgcn_pk_mul_f32(reinterpret_cast<const float2 &>(f), rw_pk);
            auto zw = amdgcn_pk_mul_f32(reinterpret_cast<const float2 *>(&f)[1],
                                        rw_pk);
            reinterpret_cast<float2 &>(f) = xy;
            reinterpret_cast<float2 *>(&f)[1] = zw;
        }
    }
}

__device__ static inline uint2 ToBf16Rn(float4 m) {
    return uint2{amdgcn_cvt_pk_bf16_f32(m.x, m.y),
                 amdgcn_cvt_pk_bf16_f32(m.z, m.w)};
}

template <class TileSchedule>
struct OnestageFusedMoEStage2Op {
    using Config = typename TileSchedule::Config;
    using CShuffle = typename TileSchedule::CShuffle;
    static constexpr unsigned kNumWarps = 4;
    static constexpr unsigned kStage = 2;
    static constexpr unsigned kSubGroupSize = 16;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kTokenBatch = Config::kTokenBatch;
    static constexpr unsigned kSubGroupPadding = 2;
    static constexpr unsigned kSubGroupRowWords =
        2 * kSubGroupSize + kSubGroupPadding;
    static constexpr unsigned kAccumFragments = TileSchedule::kAccumFragments;
    static constexpr unsigned kActivationFragments =
        TileSchedule::kActivationFragments;
    static constexpr unsigned kTokenPairs = kTokenBatch / 2;

    static_assert(TileSchedule::kNumWarps == kNumWarps, "");
    static_assert(kTokenBatch % 2 == 0, "");
    static_assert(kAccumFragments % 2 == 0, "");

    using Shm =
        unsigned[kStage][kAccumFragments * kSubGroupSize * kSubGroupRowWords];

    __device__ static void WriteShm(Shm &shm, unsigned stage,
                                    const uint2 o[kAccumFragments],
                                    unsigned wid, unsigned wtid) {
        CShuffle::template Write<kGroupDim, kAccumFragments, kTokenPairs,
                                 kNumWarps, kSubGroupRowWords>(shm, stage, o,
                                                              wid, wtid);
    }

    __device__ static void ReadShm(Shm &shm, unsigned stage,
                                   uint2 o[kTokenBatch], unsigned wid,
                                   unsigned wtid) {
        CShuffle::template Read<kGroupDim, kTokenBatch, kNumWarps,
                                kSubGroupRowWords>(shm, stage, o, wid, wtid);
    }

    template <unsigned kId>
    __device__ static void WriteOne(const BufferResource &out, uint2 value,
                                    const unsigned tokens[kTokenBatch],
                                    unsigned dim, unsigned d, unsigned wtid) {
        const unsigned vo =
            (tokens[kId] * dim + d + wtid * 2) * sizeof(__hip_bfloat16);
        BufferAtomicWriteBf16x2(out, vo, value.x);
        BufferAtomicWriteBf16x2(out, vo + 256, value.y);
    }

    __device__ static void WriteBack(const BufferResource &out,
                                     const uint2 o[kTokenBatch],
                                     const unsigned tokens[kTokenBatch],
                                     unsigned dim, unsigned d, unsigned wtid) {
        WriteOne<0>(out, o[0], tokens, dim, d, wtid);
        WriteOne<1>(out, o[1], tokens, dim, d, wtid);
        WriteOne<2>(out, o[2], tokens, dim, d, wtid);
        WriteOne<3>(out, o[3], tokens, dim, d, wtid);
        WriteOne<4>(out, o[4], tokens, dim, d, wtid);
        WriteOne<5>(out, o[5], tokens, dim, d, wtid);
        WriteOne<6>(out, o[6], tokens, dim, d, wtid);
        WriteOne<7>(out, o[7], tokens, dim, d, wtid);
    }

    template <class RouteWeights>
    __device__ static void Run(const BufferResource &out, Shm &shm,
                               TileSchedule &tiles,
                               unsigned dim,
                               const typename TileSchedule::InputRegs &input,
                               const RouteWeights &sorted_weights,
                               const unsigned tokens[kTokenBatch],
                               unsigned tile_k, unsigned tid, unsigned wid,
                               unsigned wtid) {
        unsigned curr = 0, next = 1;
        tiles.LoadStage(curr, tid, wid, wtid);
        uint2 zeroes[kAccumFragments] = {
            {0, 0},
        };
        WriteShm(shm, next, zeroes, wid, wtid);

        for (unsigned d = 0; d < dim; d += 2 * kGroupDim) {
#pragma unroll
            for (unsigned curr = 0, next = 1; curr < 2;
                 curr++, next = 1 - curr) {
                const unsigned tile_d = d + curr * kGroupDim;
                if (tile_d >= dim) {
                    break;
                }
                // TODO: Optimize the syncthreads
                __syncthreads();
                float4 t[kAccumFragments];
                ClearMat(t);
                // The down-projection loop mixes W2 global loads, LDS shuffle
                // traffic, and a shorter 64-MFMA body.
                HotLoopScheduler<64, 3, 1, 1, 2>();
                tiles.LoadStage(next, tid, wid, wtid);
                uint2 ret[kTokenBatch];
                ReadShm(shm, next, ret, wid, wtid);
                tiles.Matmul(t, input, curr, wtid);
                if (tile_k == 0) {
                    tiles.AddBias(t, tile_d, tid);
                }
                MultRouteWeights<kAccumFragments>(t, sorted_weights);
                uint2 o[kAccumFragments];
                for (unsigned i = 0; i < kAccumFragments; i++) {
                    o[i] = ToBf16Rn(t[i]);
                }

                WriteShm(shm, curr, o, wid, wtid);
                if (tile_d != 0) {
                    WriteBack(out, ret, tokens, dim, tile_d - kGroupDim, wtid);
                }
                __syncthreads();
            }
        }

        __syncthreads();
        uint2 ret[kTokenBatch];
        ReadShm(shm, (dim / kGroupDim - 1) % kStage, ret, wid, wtid);
        WriteBack(out, ret, tokens, dim, dim - kGroupDim, wtid);
    }
};

} // namespace causalflow::petit::rocm::moe
