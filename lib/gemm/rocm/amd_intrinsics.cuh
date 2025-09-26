#pragma once

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm {

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef _Float16 v4h __attribute__((ext_vector_type(4)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef short v4s __attribute__((ext_vector_type(4)));

static constexpr unsigned kWarpSize = 64;

__device__ v4i llvm_amdgcn_raw_buffer_load_v4i32(
    v4i rsrc, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

__device__ void llvm_amdgcn_raw_buffer_store_v4i32(
    v4i data, v4i rsrc, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

__device__ v2i llvm_amdgcn_raw_buffer_load_v2i32(
    v4i rsrc, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.raw.buffer.load.v2i32");

__device__ static inline float2 amdgcn_pk_mul_f32(float2 a, float2 b) {
    float2 ret;
    asm("v_pk_mul_f32 %0, %1, %2" : "=v"(ret) : "v"(a), "v"(b));
    return ret;
}

__device__ inline unsigned amdgcn_perm_b32(unsigned hi, unsigned lo,
                                           unsigned s) {
    return __builtin_amdgcn_perm(hi, lo, s);
}

__device__ inline static unsigned amdgcn_pk_add_i16(unsigned a, unsigned b) {
    unsigned r;
    asm("v_pk_add_i16 %0, %1, %2;" : "=v"(r) : "v"(a), "v"(b));
    return r;
}

__device__ inline static unsigned amdgcn_pk_mad_i16(unsigned a, unsigned b,
                                                    unsigned c) {
    unsigned r;
    asm("v_pk_mad_i16 %0, %1, %2, %3;" : "=v"(r) : "v"(a), "v"(b), "r"(c));
    return r;
}

__device__ static inline float4 mma_m16n16k16_fp16(uint2 fa, uint2 fb,
                                                   float4 c) {
    v4f ret = __builtin_amdgcn_mfma_f32_16x16x16f16(
        *reinterpret_cast<v4h *>(&fa), *reinterpret_cast<v4h *>(&fb),
        *reinterpret_cast<v4f *>(&c), 0, 0, 0);
    return *reinterpret_cast<const float4 *>(&ret);
}

__device__ static inline float4 mma_m16n16k16_bf16(uint2 fa, uint2 fb,
                                                   float4 c) {
    v4f ret = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
        *reinterpret_cast<v4s *>(&fa), *reinterpret_cast<v4s *>(&fb),
        *reinterpret_cast<v4f *>(&c), 0, 0, 0);
    // v4f ret;
    // asm("v_mfma_f32_16x16x16bf16 %0, %1, %2, %3;"
    //     : "=v"(ret)
    //     : "v"(reinterpret_cast<v4s &>(fa)), "v"(reinterpret_cast<v4s &>(fb)),
    //       "v"(reinterpret_cast<v4f &>(c)));
    return *reinterpret_cast<const float4 *>(&ret);
}

__device__ static inline float4 mma_m16n16k32_fp8_fp8_f32(uint2 fa, uint2 fb,
                                                          float4 c) {
#if defined(__gfx942__)
    v4f ret = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
        *reinterpret_cast<const long *>(&fa),
        *reinterpret_cast<const long *>(&fb), *reinterpret_cast<v4f *>(&c), 0,
        0, 0);
    return *reinterpret_cast<const float4 *>(&ret);
#else
    return {0, 0, 0, 0};
#endif
}

// BufferResource specifies the memory region to be accessed when using the
// buffer_{load,sture}_* instructions. It resides in scalar registers which
// alleivates the register pressures.
//
// See the Buffer Resource Descriptor in the GCN ISA references for more
// details.
union BufferResource {
    static constexpr unsigned kDataFormatU32Config = 4 << 15;
    enum { kNone = 0, kGLCBit = 1 << 0, kSLCBit = 1 << 1 };

    v4i content;
    struct {
        uintptr_t ptr;
        unsigned range;
        unsigned config;
    } v;

    template <int kAux>
    __device__ inline uint4 Load(int voffset, int soffset) const {
        v4i v =
            llvm_amdgcn_raw_buffer_load_v4i32(content, voffset, soffset, kAux);
        return *reinterpret_cast<const uint4 *>(&v);
    }

    template <int kAux>
    __device__ inline void Store(int voffset, int soffset, uint4 data) const {
        v4i v = *reinterpret_cast<const v4i *>(&data);
        llvm_amdgcn_raw_buffer_store_v4i32(v, content, voffset, soffset, kAux);
    }

    template <int kAux>
    __device__ inline uint2 LoadU64(int voffset, int soffset) const {
        v2i v =
            llvm_amdgcn_raw_buffer_load_v2i32(content, voffset, soffset, kAux);
        return *reinterpret_cast<const uint2 *>(&v);
    }
};

// In CDNA, OOB access to SHM are discarded. We leverage this property to
// eliminate the branches.
template <class T>
__device__ static inline T *GetConditionShmPtr(T *ptr, bool cond) {
    static constexpr unsigned kMaxShmSize = 64 * 1024;
    return cond ? ptr : reinterpret_cast<T *>(kMaxShmSize);
}

} // namespace causalflow::petit::rocm
