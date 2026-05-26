#pragma once

#include "causalflow/petit/tal/tensor/layout.h"
#include "moe/rocm/fused_moe.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <unsigned kId>
__device__ static inline void ConditionalWrite(unsigned mask, void *base,
                                               unsigned vo, uint2 value) {
    asm volatile("s_setvskip %0, %1\n\t"
                 "global_atomic_pk_add_bf16 %3, %4, %2\n\t"
                 "global_atomic_pk_add_bf16 %3, %5, %2 offset:256\n\t"
                 "s_setvskip 0, 0\n\t"
                 :
                 : "s"(mask), "n"(kId), "s"(reinterpret_cast<uintptr_t>(base)),
                   "v"(vo), "v"(value.x), "v"(value.y)
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
    uint4 m_bits;
    for (int j = 0; j < 4; j++) {
        const float f = reinterpret_cast<const float *>(&m)[j];
        const uint u = reinterpret_cast<const uint *>(&m)[j];
        const uint rounded = u + 0x8000u;
        reinterpret_cast<uint *>(&m_bits)[j] = isnan(f) ? 0x7fff0000u : rounded;
    }
    uint2 o;
    o.x = amdgcn_perm_b32(m_bits.y, m_bits.x, 0x07060302);
    o.y = amdgcn_perm_b32(m_bits.w, m_bits.z, 0x07060302);
    return o;
}

template <class Trait, unsigned kGroupDim, unsigned kTokenBatch>
struct OnestageFusedMoEStage2Op {
    static constexpr unsigned kNumWarps = 4;
    static constexpr unsigned kStage = 2;
    static constexpr unsigned kSubGroupSize = 16;
    static constexpr unsigned kSubGroupPadding = 2;
    static constexpr unsigned kSubGroupRowWords =
        2 * kSubGroupSize + kSubGroupPadding;
    static constexpr unsigned kAccumFragments = Trait::kAccumFragments;
    static constexpr unsigned kActivationFragments =
        Trait::kActivationFragments;
    static constexpr unsigned kTokenPairs = kTokenBatch / 2;

    static_assert(Trait::kNumWarps == kNumWarps, "");
    static_assert(kTokenBatch % 2 == 0, "");
    static_assert(kAccumFragments % 2 == 0, "");

    using Shm =
        unsigned[kStage][kAccumFragments * kSubGroupSize * kSubGroupRowWords];

    __device__ static void WriteShm(Shm &shm, unsigned stage,
                                    const uint2 o[kAccumFragments],
                                    unsigned wid, unsigned wtid) {
        using namespace causalflow::tal;

        using WriteLayout = Layout<
            Shape<Shape<_2, C<kTokenPairs>>, C<kNumWarps>, Shape<_16, _2>>,
            Stride<Stride<C<kNumWarps * 16 * kSubGroupRowWords>,
                          C<16 * kSubGroupRowWords>>,
                   C<kTokenPairs * kSubGroupRowWords>,
                   Stride<_2, C<kSubGroupRowWords>>>>;
        WriteLayout layout;
        for (unsigned i = 0; i < kAccumFragments; i++) {
            const unsigned base = layout(make_coord(i, wid, wtid));
            shm[stage][base] = o[i].x;
            shm[stage][base + 1] = o[i].y;
        }
    }

    __device__ static void ReadShm(Shm &shm, unsigned stage,
                                   uint2 o[kTokenBatch], unsigned wid,
                                   unsigned wtid) {
        using namespace causalflow::tal;
        static constexpr unsigned kHalfStrideWords = 32 * kSubGroupRowWords;

        using ReadLayout = Layout<
            Shape<Shape<C<kTokenPairs>, _2>, C<kNumWarps>, Shape<_2, _16>>,
            Stride<Stride<_8, C<2 * kHalfStrideWords>>, _2,
                   Stride<_1, C<kSubGroupRowWords>>>>;
        ReadLayout layout;
        for (unsigned i = 0; i < kTokenBatch; i++) {
            const unsigned base = layout(make_coord(i, wid, wtid));
            o[i] = uint2{shm[stage][base], shm[stage][base + kHalfStrideWords]};
        }
    }

    __device__ static void WriteBack(uint4 *__restrict__ out,
                                     const uint2 o[kTokenBatch],
                                     const unsigned tokens[kTokenBatch],
                                     unsigned invalid_token_mask, unsigned dim,
                                     unsigned d, unsigned wtid) {
        unsigned vo[kTokenBatch];
        for (unsigned i = 0; i < kTokenBatch; i++) {
            vo[i] = (tokens[i] * dim + d + wtid * 2) * sizeof(__hip_bfloat16);
        }
        ConditionalWrite<0>(invalid_token_mask, out, vo[0], o[0]);
        ConditionalWrite<1>(invalid_token_mask, out, vo[1], o[1]);
        ConditionalWrite<2>(invalid_token_mask, out, vo[2], o[2]);
        ConditionalWrite<3>(invalid_token_mask, out, vo[3], o[3]);
        ConditionalWrite<4>(invalid_token_mask, out, vo[4], o[4]);
        ConditionalWrite<5>(invalid_token_mask, out, vo[5], o[5]);
        ConditionalWrite<6>(invalid_token_mask, out, vo[6], o[6]);
        ConditionalWrite<7>(invalid_token_mask, out, vo[7], o[7]);
    }

    __device__ static void Run(uint4 *__restrict__ out, Shm &shm, Trait &trait,
                               unsigned dim,
                               const typename Trait::InputRegs &input,
                               float2 sorted_weights,
                               const unsigned tokens[kTokenBatch],
                               unsigned invalid_token_mask, unsigned tid,
                               unsigned wid, unsigned wtid) {
        unsigned curr = 0, next = 1;
        trait.LoadStage(curr, tid, wid, wtid);
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
                trait.LoadStage(next, tid, wid, wtid);
                uint2 ret[kTokenBatch];
                ReadShm(shm, next, ret, wid, wtid);
                trait.Matmul(t, input, curr, wtid);
                MultRouteWeights<kAccumFragments>(t, sorted_weights);
                uint2 o[kAccumFragments];
                for (unsigned i = 0; i < kAccumFragments; i++) {
                    o[i] = ToBf16Rn(t[i]);
                }

                WriteShm(shm, curr, o, wid, wtid);
                if (tile_d != 0) {
                    WriteBack(out, ret, tokens, invalid_token_mask, dim,
                              tile_d - kGroupDim, wtid);
                }
                __syncthreads();
            }
        }

        __syncthreads();
        uint2 ret[kTokenBatch];
        ReadShm(shm, (dim / kGroupDim - 1) % kStage, ret, wid, wtid);
        WriteBack(out, ret, tokens, invalid_token_mask, dim, dim - kGroupDim,
                  wtid);
    }
};

} // namespace causalflow::petit::rocm::moe
