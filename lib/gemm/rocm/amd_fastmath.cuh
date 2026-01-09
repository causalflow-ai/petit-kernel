#pragma once

#include "amd_intrinsics.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::fastmath {

__device__ static inline __hip_bfloat162 hmul2(__hip_bfloat162 a,
                                               __hip_bfloat162 b) {
    unsigned a_u = reinterpret_cast<const unsigned &>(a);
    unsigned b_u = reinterpret_cast<const unsigned &>(b);
    // Use perm to extract bf16 values into high 16 bits of each word (as f32)
    uint2 a2{amdgcn_perm_b32(a_u, 0, 0x05040c0c),
             amdgcn_perm_b32(a_u, 0, 0x07060c0c)},
        b2{amdgcn_perm_b32(b_u, 0, 0x05040c0c),
           amdgcn_perm_b32(b_u, 0, 0x07060c0c)};
    float2 r2 = amdgcn_pk_mul_f32(reinterpret_cast<float2 &>(a2),
                                  reinterpret_cast<float2 &>(b2));
    const unsigned *r2_u = reinterpret_cast<const unsigned *>(&r2);
    unsigned c = amdgcn_perm_b32(r2_u[1], r2_u[0], 0x07060302);
    return reinterpret_cast<const __hip_bfloat162 &>(c);
}

__device__ static inline half2 hmul2(half2 a, half2 b) { return __hmul2(a, b); }

} // namespace causalflow::petit::rocm::fastmath