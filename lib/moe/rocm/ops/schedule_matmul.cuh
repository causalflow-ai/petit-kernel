#pragma once

#include "gemm/rocm/amd_intrinsics.cuh"
#include "gemm/rocm/quantization/fp4/quantization_utils.cuh"
#include "moe/rocm/fused_moe.cuh"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

namespace detail {

template <unsigned kTileN_, unsigned kInputBits, unsigned kWeightBits>
struct MatmulTile {
    static constexpr unsigned kTileM = 32;
    static constexpr unsigned kTileN = kTileN_;
    static constexpr unsigned kTileK = 256;
    static constexpr unsigned kKStages = 2;

    // Each policy describes one wave's tile. Weight fragments are split
    // between the two K stages; activations cover the complete K256 tile.
    static constexpr unsigned kWeightFragments =
        kTileN * kTileK * kWeightBits /
        (8 * sizeof(uint4) * kWarpSize * kKStages);
    static constexpr unsigned kActivationFragments =
        kTileM * kTileK * kInputBits / (8 * sizeof(uint4) * kWarpSize);
    static constexpr unsigned kAccumFragments =
        kTileM * kTileN * sizeof(float) / (sizeof(float4) * kWarpSize);
};

__device__ inline float4 Fma4(float4 a, float4 s, float4 c) {
    const auto *a2 = reinterpret_cast<const float2 *>(&a);
    const auto *c2 = reinterpret_cast<const float2 *>(&c);
    const auto *s2 = reinterpret_cast<const float2 *>(&s);
    float4 r;
    auto *r2 = reinterpret_cast<float2 *>(&r);
    r2[0] = amdgcn_pk_fma_f32(s2[0], c2[0], a2[0]);
    r2[1] = amdgcn_pk_fma_f32(s2[1], c2[1], a2[1]);
    return r;
}

__device__ inline float4 Fma4(float4 a, float s, float4 c) {
    return Fma4(a, float4{s, s, s, s}, c);
}

template <class T> __device__ T GetDppValue(T src, unsigned ctrl) {
    static constexpr unsigned kDppRowNewBcastBase = 0x150;
    static_assert(sizeof(T) == sizeof(int));

    const auto bits = reinterpret_cast<const int &>(src);
    int dst;
    if (ctrl == 0) {
        dst = __builtin_amdgcn_mov_dpp(bits, kDppRowNewBcastBase, 0xf, 0xf, 0);
    } else if (ctrl == 1) {
        dst = __builtin_amdgcn_mov_dpp(bits, kDppRowNewBcastBase + 1, 0xf,
                                       0xf, 0);
    } else if (ctrl == 2) {
        dst = __builtin_amdgcn_mov_dpp(bits, kDppRowNewBcastBase + 2, 0xf,
                                       0xf, 0);
    } else {
        dst = __builtin_amdgcn_mov_dpp(bits, kDppRowNewBcastBase + 3, 0xf,
                                       0xf, 0);
    }
    return reinterpret_cast<const T &>(dst);
}

__device__ inline float4 LoadFp8E8M0Scale(unsigned packed) {
    union {
        v4f f;
        unsigned u[4];
    };
    for (unsigned i = 0; i < 4; ++i) {
        u[i] = ((packed >> (8 * i)) & 0xffu) << 23;
    }
    // FP4 payloads are embedded into BF8 lanes before MFMA. CDNA3 BF8 uses
    // FNUZ (bias 16), while gfx950 uses OCP BF8 (bias 15).
#if defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__)
    static const v4f kExpBias = {16384.0f, 16384.0f, 16384.0f, 16384.0f};
#else
    static const v4f kExpBias = {32768.0f, 32768.0f, 32768.0f, 32768.0f};
#endif
    f *= kExpBias;
    return reinterpret_cast<const float4 &>(f);
}

__device__ inline float LoadE8M0ScaleByte(unsigned packed,
                                           unsigned byte_idx) {
    const unsigned bits = ((packed >> (byte_idx * 8)) & 0xffu) << 23;
    return reinterpret_cast<const float &>(bits);
}

template <unsigned kByteIdx>
__device__ unsigned CvtFp4ByteToBf16x2(unsigned q, float scale) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__) &&                  \
    __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_bf16_fp4)
    auto v = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(q, scale, kByteIdx);
    return reinterpret_cast<const unsigned &>(v);
#else
    return 0;
#endif
}

template <unsigned kPairIdx>
__device__ uint2 CvtFp4WordToBf16x4(unsigned q, float scale) {
    return uint2{CvtFp4ByteToBf16x2<2 * kPairIdx>(q, scale),
                 CvtFp4ByteToBf16x2<2 * kPairIdx + 1>(q, scale)};
}

__device__ inline float4 ScaledMxFp4Mfma(
    unsigned opsel_a, unsigned opsel_b, const uint4 &a, unsigned scale_a,
    const uint4 &b, unsigned scale_b, float4 acc) {
    auto dispatch = []<class Op>(unsigned opsel, Op op) -> float4 {
        if (opsel == 0)
            return op.template operator()<0>();
        if (opsel == 1)
            return op.template operator()<1>();
        if (opsel == 2)
            return op.template operator()<2>();
        return op.template operator()<3>();
    };
    return dispatch(opsel_a, [&]<unsigned kOpSelA>() -> float4 {
        return dispatch(opsel_b, [&]<unsigned kOpSelB>() -> float4 {
            return mma_scale_m16n16k128_fp4_fp4_f32<kOpSelA, kOpSelB>(
                a, scale_a, b, scale_b, acc);
        });
    });
}

} // namespace detail

// Shared FP8 MFMA path. The schedule supplies the DPP policy because stage1
// and stage2 broadcast scales from different logical stages.
struct BlockScaleFp8Matmul : detail::MatmulTile<64, 8, 8> {

    __device__ static void Matmul(float4 t[kAccumFragments],
                                 const uint4 w[kWeightFragments],
                                 const uint4 x[kActivationFragments],
                                 float4 x_scale,
                                 float w_scale, unsigned stage) {
        for (int i = 0; i < 4; i++) {
            const uint2 *w_2 = reinterpret_cast<const uint2 *>(w) + i * 4;
            for (int row = 0; row < 2; ++row) {
                const int tg = row * 4;
                const uint2 *x_2 =
                    reinterpret_cast<const uint2 *>(x + tg + stage * 2);
                float4 m_acc{0, 0, 0, 0};
                m_acc = mma_m16n16k128_fp8_fp8_f32(w_2, x_2, m_acc);
                const float x_s = stage == 0
                                      ? (row == 0 ? x_scale.x : x_scale.y)
                                      : (row == 0 ? x_scale.z : x_scale.w);
                const float w_sdpp =
                    detail::GetDppValue(w_scale, (stage << 1) + (i >> 1));
                t[i * 2 + row] =
                    detail::Fma4(t[i * 2 + row], w_sdpp * x_s, m_acc);
            }
        }
    }

};

// Petit MXFP4 uses the FP8 MFMA path after converting FP4 payloads to BF8.
struct PetitMxFp4Matmul : detail::MatmulTile<64, 8, 4> {

    __device__ static void
    Matmul(float4 t[kAccumFragments], const uint4 w[kWeightFragments],
           const uint4 x[kActivationFragments], float4 x_scale,
           const unsigned packed_scale[kWeightFragments / 2], unsigned stage) {
        const unsigned wtid = threadIdx.x % kWarpSize;
        static constexpr unsigned kFragmentsPerRow = kActivationFragments / 2;
#pragma unroll
        for (unsigned i = 0; i < kWeightFragments; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned src_lane =
                    ((i & 1) * 4 + wtid / 16) * 8 + stage * 4 + j;
                const unsigned s = __shfl(packed_scale[i / 2], src_lane);
                const float4 w_scale = detail::LoadFp8E8M0Scale(s);
                const unsigned qw =
                    reinterpret_cast<const unsigned *>(&w[i])[j];
                uint2 bf8;
                causalflow::petit::rocm::quantization::detail::Fp4ToBf8(
                    reinterpret_cast<unsigned *>(&bf8), qw);

                for (int row = 0; row < 2; row++) {
                    const uint2 *x_2 = reinterpret_cast<const uint2 *>(
                        x + row * kFragmentsPerRow + stage * 2);
                    const float x_s = reinterpret_cast<const float *>(
                        &x_scale)[stage * 2 + row];

                    const v4f x_s4 = {x_s, x_s, x_s, x_s};
                    const v4f fs =
                        reinterpret_cast<const v4f &>(w_scale) * x_s4;

                    float4 m_acc{0, 0, 0, 0};
                    m_acc = mma_m16n16k32_bf8_fp8_f32(bf8, x_2[j], m_acc);
                    t[i * 2 + row] =
                        detail::Fma4(t[i * 2 + row],
                                     reinterpret_cast<const float4 &>(fs),
                                     m_acc);
                }
            }
        }
    }

};

// Native MXFP4 converts FP4 payloads to BF16 and uses BF16 MFMA directly.
struct Bf16MxFp4Matmul : detail::MatmulTile<64, 16, 4> {

    __device__ static void
    Matmul(float4 t[kAccumFragments], const uint4 w[kWeightFragments],
           const uint4 x[kActivationFragments],
           const unsigned packed_scale[kWeightFragments / 2], unsigned stage) {
        static constexpr unsigned kFragmentsPerRow = kActivationFragments / 2;

        for (unsigned i = 0; i < kWeightFragments; ++i) {
            const unsigned *qw = reinterpret_cast<const unsigned *>(&w[i]);
            const float scale =
                detail::LoadE8M0ScaleByte(packed_scale[i / 2],
                                          2 * stage + (i & 1));
            for (unsigned j = 0; j < 4; ++j) {
                for (unsigned row = 0; row < 2; ++row) {
                    const uint2 *x_2 = reinterpret_cast<const uint2 *>(
                        x + row * kFragmentsPerRow + stage * 4);
                    float4 m_acc{0, 0, 0, 0};
                    m_acc = mma_m16n16k16_bf16(
                        detail::CvtFp4WordToBf16x4<0>(qw[j], scale),
                        x_2[j * 2], m_acc);
                    m_acc = mma_m16n16k16_bf16(
                        detail::CvtFp4WordToBf16x4<1>(qw[j], scale),
                        x_2[j * 2 + 1], m_acc);
                    static constexpr float4 kOne{1.0f, 1.0f, 1.0f, 1.0f};
                    t[i * 2 + row] =
                        detail::Fma4(t[i * 2 + row], kOne, m_acc);
                }
            }
        }
    }

};

template <unsigned kTileN_>
struct NativeMxFp4Matmul : detail::MatmulTile<kTileN_, 4, 4> {
    using Base = detail::MatmulTile<kTileN_, 4, 4>;
    static constexpr unsigned kActivationFragments =
        Base::kActivationFragments;
    static constexpr unsigned kAccumFragments = Base::kAccumFragments;
    static constexpr unsigned kWeightFragments = Base::kWeightFragments;

    // With weight as operand A, lane 16*q+r owns
    // C[16*m16+r, 16*n_fragment+4*q+component]. Accumulators therefore use
    // t[n_fragment * 2 + m16].
    __device__ static void
    Matmul(float4 t[kAccumFragments],
           const uint4 w[2][kWeightFragments],
           const uint4 x[kActivationFragments], unsigned scale_x,
           const unsigned scale_w[kWeightFragments / 2]) {
#pragma unroll
        for (unsigned k128 = 0; k128 < 2; ++k128) {
#pragma unroll
            for (unsigned n_fragment = 0; n_fragment < kWeightFragments;
                 ++n_fragment) {
#pragma unroll
                for (unsigned m16 = 0; m16 < 2; ++m16) {
                    const unsigned t_idx = n_fragment * 2 + m16;
                    t[t_idx] = detail::ScaledMxFp4Mfma(
                        2 * k128 + (n_fragment & 1), 2 * k128 + m16,
                        w[k128][n_fragment], scale_w[n_fragment / 2],
                        x[m16 * 2 + k128], scale_x, t[t_idx]);
                }
            }
        }
    }
};

} // namespace causalflow::petit::rocm::moe
