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
    Weight &w1;
    Bias &w13_bias;
    typename TileOps::Tile w1_tile, w3_tile;

    __device__ W13TileSchedule(Input &input, Weight &w1, Bias &bias)
        : input(input), w1(w1), w13_bias(bias) {}

    __device__ void InitializeBias(const void *bias_ptr, unsigned expert_id,
                                   unsigned tile_k) {
        const unsigned projection_stride =
            Bias::PackedStride(Config::kInterDim);
        const unsigned expert_stride = 2 * projection_stride;
        w13_bias.Initialize(bias_ptr, expert_id, Config::kInterDim, tile_k,
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
        LoadW3Tile(tid, wid, wtid);
        w1.template AdvanceStep<0, TileOps::kKStages>();
    }

    __device__ void Matmul(float4 gate[kAccumFragments],
                           float4 up[kAccumFragments],
                           const InputRegs &input_regs, unsigned tid,
                           unsigned wid, unsigned wtid) {
        TileOps::Matmul(gate, w1_tile, input_regs, wtid);
        TileOps::Matmul(up, w3_tile, input_regs, wtid);
        TileOps::Load(w1, w1_tile, tid, wid, wtid);
        LoadW3Tile(tid, wid, wtid);
        w1.template AdvanceStep<0, TileOps::kKStages>();
    }

    __device__ void AddBias(float4 gate[kAccumFragments],
                            float4 up[kAccumFragments], unsigned tid) const {
        w13_bias.AddToAccumulator(gate, 0, tid);
        w13_bias.AddToAccumulator(up, Bias::PackedStride(Config::kInterDim),
                                  tid);
    }

  private:
    __device__ void LoadW3Tile(unsigned tid, unsigned wid, unsigned wtid) {
        static constexpr unsigned kValueOffset =
            Weight::ValueProjectionOffsetBytes(Config::kDim,
                                               Config::kInterDim);
        static constexpr unsigned kScaleOffset =
            Weight::ScaleProjectionOffsetBytes(Config::kDim,
                                               Config::kInterDim);
        TileOps::LoadProjection(w1, w3_tile, tid, wid, wtid, kValueOffset,
                                kScaleOffset);
    }
};

template <class TileOps_> struct W2TileSchedule {
    using TileOps = TileOps_;
    using Config = typename TileOps::Config;
    using Weight = typename Config::W2;
    using Bias = typename Config::Stage2Bias;
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
                                   unsigned tile_k) {
        const unsigned expert_stride = Bias::PackedStride(Config::kDim);
        const unsigned bias_tile = TileOps::kStage2BiasUsesTileK ? tile_k : 0;
        bias.Initialize(bias_ptr, expert_id, Config::kDim, bias_tile,
                        expert_stride);
    }

    __device__ void LoadStage(unsigned stage, unsigned tid, unsigned wid,
                              unsigned wtid) {
        TileOps::Load(weight, stages[stage], tid, wid, wtid);
        weight.template AdvanceStep<1, 0>();
    }

    __device__ void LoadKStage(unsigned stage, unsigned tid, unsigned wid,
                               unsigned wtid) {
        TileOps::Load(weight, stages[stage], tid, wid, wtid);
        weight.template AdvanceStep<0, TileOps::kKStages>();
    }

    __device__ void Matmul(float4 t[kAccumFragments], const InputRegs &input,
                           unsigned stage, unsigned wtid) const {
        TileOps::Matmul(t, stages[stage], input, wtid);
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
                               TileSchedule &tiles, unsigned tid,
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

#pragma unroll
        for (unsigned d = 0; d < Config::kDim; d += 2 * kGroupDim) {
#pragma unroll
            for (unsigned curr = 0, next = 1; curr < 2;
                 curr++, next = 1 - curr) {
                // The up/gate projection loop is dominated by W13/input global
                // loads and 128 MFMA instructions.
                HotLoopScheduler<128, 6, 0, 0, 2>();
                tiles.PrefetchInput(&shm.x[next], wid, wtid, tokens, m);
                tiles.Matmul(t_gate, t_up, x[curr], tid, wid, wtid);

                if (d + kGroupDim >= Config::kDim) {
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
                               TileSchedule &tiles, unsigned tid,
                               unsigned wid, unsigned wtid,
                               const unsigned tokens[kTokenBatch], unsigned m) {
        float4 t_gate[kAccumFragments], t_up[kAccumFragments];
        typename TileSchedule::InputRegs x;
        ClearMat(t_gate);
        ClearMat(t_up);

        tiles.LoadInitial(tid, wid, wtid);

#pragma unroll
        for (unsigned d = 0; d < Config::kDim; d += kGroupDim) {
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

template <class TileSchedule> struct W2AccumulatorEpilogue {
    static constexpr unsigned kAccumFragments = TileSchedule::kAccumFragments;
    using Bias = typename TileSchedule::Bias;
    using BiasPrefetch = typename Bias::Prefetch;
    using NoopBias = NoopBiasLayout<Bias::kNumWarps, Bias::kGroupN>;

    __device__ static void PrefetchBias(BiasPrefetch &prefetch,
                                        const TileSchedule &tiles,
                                        unsigned tile_col, unsigned tid) {
        tiles.bias.PrefetchFragments(prefetch, tile_col, tid);
    }

    template <class SelectedBias = Bias, class RouteWeights>
    __device__ static void Apply(float4 accum[kAccumFragments],
                                 const BiasPrefetch &bias,
                                 const RouteWeights &route_weights) {
        SelectedBias::Apply(accum, bias);
        MultRouteWeights<kAccumFragments>(accum, route_weights);
    }
};

template <class TileSchedule> struct TwoStageStage2Epilogue {
    using Config = typename TileSchedule::Config;
    using AccumulatorEpilogue = W2AccumulatorEpilogue<TileSchedule>;
    using BiasPrefetch = typename AccumulatorEpilogue::BiasPrefetch;
    static constexpr unsigned kAccumFragments = TileSchedule::kAccumFragments;
    static constexpr unsigned kNumWarps = 4;
    static constexpr unsigned kTokenBatch = Config::kStage2TokenBatch;
    static constexpr unsigned kTileRows = kTokenBatch * kNumWarps;
    static constexpr unsigned kTileCols = Config::kGroupN;

    struct Shm {
        unsigned short output[kTileRows * kTileCols];
        unsigned output_row_offsets[kTileRows];
        float route_weights[kTileRows];
    };

    static_assert(kTileRows == Config::kStage2GroupM,
                  "two-stage C-shuffle row geometry mismatch");
    static_assert(kTileRows == 32, "two-stage C-shuffle expects M32");
    static_assert(kTileCols == 256, "two-stage C-shuffle expects N256");
    static_assert(kAccumFragments == 8,
                  "two-stage C-shuffle expects N64 waves");

    __device__ static void PrefetchBias(BiasPrefetch &prefetch,
                                        const TileSchedule &tiles,
                                        unsigned tile_col, unsigned tid) {
        AccumulatorEpilogue::PrefetchBias(prefetch, tiles, tile_col, tid);
    }

    template <class RouteWeights>
    __device__ static void Apply(float4 accum[kAccumFragments],
                                 const BiasPrefetch &bias,
                                 const RouteWeights &route_weights) {
        AccumulatorEpilogue::Apply(accum, bias, route_weights);
    }

    __device__ static void StoreOutputRowOffset(Shm &shm, unsigned row,
                                                unsigned offset) {
        if (row < kTileRows)
            shm.output_row_offsets[row] = offset;
    }

    __device__ static void StoreRouteWeight(Shm &shm, unsigned row,
                                            unsigned weight) {
        if (row < kTileRows)
            shm.route_weights[row] = reinterpret_cast<const float &>(weight);
    }

    __device__ static float2 LoadRouteWeights(const Shm &shm, unsigned wtid) {
        const unsigned row = wtid % 16;
        return {shm.route_weights[row], shm.route_weights[row + 16]};
    }

    __device__ static void WriteShm(Shm &shm,
                                    const float4 accum[kAccumFragments],
                                    unsigned wid, unsigned wtid) {
        const unsigned q = wtid / 16;
        const unsigned r = wtid % 16;
#pragma unroll
        for (unsigned mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (unsigned ni = 0; ni < kAccumFragments / 2; ++ni) {
                const unsigned fragment = 2 * ni + mi;
#pragma unroll
                for (unsigned component = 0; component < 4; ++component) {
                    const float value = reinterpret_cast<const float *>(
                        &accum[fragment])[component];
                    const unsigned packed = amdgcn_cvt_pk_bf16_f32(value,
                                                                    value);
                    const unsigned row = mi * 16 + r;
                    const unsigned col =
                        wid * 64 + ni * 16 + q * 4 + component;
                    shm.output[row * kTileCols + col] =
                        static_cast<unsigned short>(packed);
                }
            }
        }
    }

    __device__ static void WriteBack(const BufferResource &out, Shm &shm,
                                     unsigned tile_col, unsigned tid) {
        const unsigned m_lane = tid / 32;
        const unsigned n_lane = tid % 32;
        const auto *output = reinterpret_cast<const unsigned *>(shm.output);
        const unsigned base = m_lane * (kTileCols / 2) + n_lane;
#pragma unroll
        for (unsigned mr = 0; mr < 4; ++mr) {
            const unsigned output_row_offset =
                shm.output_row_offsets[m_lane + mr * 8];
#pragma unroll
            for (unsigned nr = 0; nr < 4; ++nr) {
                const unsigned value =
                    output[base + mr * 8 * (kTileCols / 2) + nr * 32];
                const unsigned col = tile_col + nr * 64 + n_lane * 2;
                const unsigned vo =
                    output_row_offset + col * sizeof(__hip_bfloat16);
                BufferAtomicWriteBf16x2(out, vo, value);
            }
        }
    }
};

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
    using AccumulatorEpilogue = W2AccumulatorEpilogue<TileSchedule>;

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
                                    unsigned d, unsigned wtid) {
        const unsigned vo =
            (tokens[kId] * Config::kDim + d + wtid * 2) *
            sizeof(__hip_bfloat16);
        BufferAtomicWriteBf16x2(out, vo, value.x);
        BufferAtomicWriteBf16x2(out, vo + 256, value.y);
    }

    __device__ static void WriteBack(const BufferResource &out,
                                     const uint2 o[kTokenBatch],
                                     const unsigned tokens[kTokenBatch],
                                     unsigned d, unsigned wtid) {
        WriteOne<0>(out, o[0], tokens, d, wtid);
        WriteOne<1>(out, o[1], tokens, d, wtid);
        WriteOne<2>(out, o[2], tokens, d, wtid);
        WriteOne<3>(out, o[3], tokens, d, wtid);
        WriteOne<4>(out, o[4], tokens, d, wtid);
        WriteOne<5>(out, o[5], tokens, d, wtid);
        WriteOne<6>(out, o[6], tokens, d, wtid);
        WriteOne<7>(out, o[7], tokens, d, wtid);
    }

    template <class RouteWeights>
    __device__ static void Run(const BufferResource &out, Shm &shm,
                               TileSchedule &tiles,
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

#pragma unroll
        for (unsigned d = 0; d < Config::kDim; d += 2 * kGroupDim) {
#pragma unroll
            for (unsigned curr = 0, next = 1; curr < 2;
                 curr++, next = 1 - curr) {
                const unsigned tile_d = d + curr * kGroupDim;
                if (tile_d >= Config::kDim) {
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
                typename AccumulatorEpilogue::BiasPrefetch bias{};
                if (tile_k == 0)
                    AccumulatorEpilogue::PrefetchBias(bias, tiles, tile_d,
                                                      tid);
                if (tile_k == 0)
                    AccumulatorEpilogue::Apply(t, bias, sorted_weights);
                else
                    AccumulatorEpilogue::template Apply<
                        typename AccumulatorEpilogue::NoopBias>(
                        t, bias, sorted_weights);
                uint2 o[kAccumFragments];
                for (unsigned i = 0; i < kAccumFragments; i++) {
                    o[i] = ToBf16Rn(t[i]);
                }

                WriteShm(shm, curr, o, wid, wtid);
                if (tile_d != 0) {
                    WriteBack(out, ret, tokens, tile_d - kGroupDim, wtid);
                }
                __syncthreads();
            }
        }

        __syncthreads();
        uint2 ret[kTokenBatch];
        ReadShm(shm, (Config::kDim / kGroupDim - 1) % kStage, ret, wid,
                wtid);
        WriteBack(out, ret, tokens, Config::kDim - kGroupDim, wtid);
    }
};

} // namespace causalflow::petit::rocm::moe
