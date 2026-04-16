#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) ||      \
     defined(__gfx950__)) &&                                                   \
    __has_builtin(__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8)
#define HAS_AMD_FP8_MFMA 1
#else
#define HAS_AMD_FP8_MFMA 0
#endif

#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) ||      \
     defined(__gfx950__)) &&                                                   \
    __has_builtin(__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8)
#define HAS_AMD_BF8_FP8_MFMA 1
#else
#define HAS_AMD_BF8_FP8_MFMA 0
#endif

#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) ||      \
     defined(__gfx950__)) &&                                                   \
    __has_builtin(__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8)
#define HAS_AMD_FP8_BF8_MFMA 1
#else
#define HAS_AMD_FP8_BF8_MFMA 0
#endif

#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    __has_builtin(__builtin_amdgcn_cvt_pk_fp8_f32)
#define HAS_AMD_FP8_PACK_CONVERSION 1
#else
#define HAS_AMD_FP8_PACK_CONVERSION 0
#endif

#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    __has_builtin(__builtin_amdgcn_cvt_pk_f32_bf8)
#define HAS_AMD_BF8_PACK_CONVERSION 1
#else
#define HAS_AMD_BF8_PACK_CONVERSION 0
#endif

#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    __has_builtin(__builtin_amdgcn_sched_barrier)
#define HAS_AMD_SCHED_BARRIER 1
#else
#define HAS_AMD_SCHED_BARRIER 0
#endif

#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    __has_builtin(__builtin_amdgcn_sched_group_barrier)
#define HAS_AMD_SCHED_GROUP_BARRIER 1
#else
#define HAS_AMD_SCHED_GROUP_BARRIER 0
#endif

namespace causalflow::petit::rocm {

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef _Float16 v4h __attribute__((ext_vector_type(4)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef short v4s __attribute__((ext_vector_type(4)));
typedef float v16f __attribute__((ext_vector_type(16)));

static constexpr unsigned kWarpSize = 64;

// Direct loads from global memory to LDS.
// This maps to MUBUF buffer_load_* with `lds` and is the most robust way to
// express "async global->LDS" across ROCm toolchains (avoids inline-asm syntax
// issues and SGPR tuple constraints).
__device__ void llvm_amdgcn_raw_buffer_load_lds(
    v4i rsrc, __attribute__((address_space(3))) unsigned *lds_ptr, int size,
    int voffset, int soffset, int offset,
    int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

__device__ v4i llvm_amdgcn_raw_buffer_load_v4i32(
    v4i rsrc, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

__device__ void llvm_amdgcn_raw_buffer_store_v4i32(
    v4i data, v4i rsrc, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

__device__ v2i llvm_amdgcn_raw_buffer_load_v2i32(
    v4i rsrc, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.raw.buffer.load.v2i32");

__device__ int llvm_amdgcn_raw_buffer_load_i32(
    v4i rsrc, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ static inline float2 amdgcn_pk_mul_f32(float2 a, float2 b) {
    v2f ret =
        reinterpret_cast<const v2f &>(a) * reinterpret_cast<const v2f &>(b);
    return reinterpret_cast<const float2 &>(ret);
}

__device__ static inline float2 amdgcn_pk_add_f32(float2 a, float2 b) {
    v2f ret =
        reinterpret_cast<const v2f &>(a) + reinterpret_cast<const v2f &>(b);
    return reinterpret_cast<const float2 &>(ret);
}

__device__ static inline float2 amdgcn_pk_fma_f32(float2 a, float2 b,
                                                  float2 c) {
    v2f ret = __builtin_elementwise_fma(reinterpret_cast<const v2f &>(a),
                                        reinterpret_cast<const v2f &>(b),
                                        reinterpret_cast<const v2f &>(c));
    return reinterpret_cast<const float2 &>(ret);
}

__device__ static inline float amdgcn_exp2f(float x) {
    return __builtin_amdgcn_exp2f(x);
}

__device__ inline unsigned amdgcn_perm_b32(unsigned hi, unsigned lo,
                                           unsigned s) {
    return __builtin_amdgcn_perm(hi, lo, s);
}

__device__ inline int amdgcn_ds_permute_b32(int index, int src) {
    return __builtin_amdgcn_ds_permute(index, src);
}

template <bool kWordHi>
__device__ inline unsigned amdgcn_cvt_pk_fp8_f32(float a, float b,
                                                 unsigned old) {
#if HAS_AMD_FP8_PACK_CONVERSION
    return __builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, kWordHi);
#else
    return old;
#endif
}

template <bool kWordHi>
__device__ inline v2f amdgcn_cvt_pk_f32_bf8(unsigned src) {
#if HAS_AMD_BF8_PACK_CONVERSION
    return __builtin_amdgcn_cvt_pk_f32_bf8(src, kWordHi);
#else
    (void)src;
    return {0.0f, 0.0f};
#endif
}

template <unsigned kMask> __device__ inline void amdgcn_sched_barrier() {
#if HAS_AMD_SCHED_BARRIER
    __builtin_amdgcn_sched_barrier(kMask);
#endif
}

template <unsigned kMask, unsigned kCount, unsigned kGroup>
__device__ inline void amdgcn_sched_group_barrier() {
#if HAS_AMD_SCHED_GROUP_BARRIER
    __builtin_amdgcn_sched_group_barrier(kMask, kCount, kGroup);
#endif
}

__device__ inline static half2 amdgcn_pk_float22half2(float a, float b) {
    auto v = __builtin_amdgcn_cvt_pkrtz(a, b);
    return reinterpret_cast<const half2 &>(v);
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
    return *reinterpret_cast<const float4 *>(&ret);
}

__device__ static inline float4 mma_m16n16k32_fp8_fp8_f32(uint2 fa, uint2 fb,
                                                          float4 c) {
#if HAS_AMD_FP8_MFMA
    v4f ret = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
        *reinterpret_cast<const long *>(&fa),
        *reinterpret_cast<const long *>(&fb), *reinterpret_cast<v4f *>(&c), 0,
        0, 0);
    return *reinterpret_cast<const float4 *>(&ret);
#else
    return {0, 0, 0, 0};
#endif
}

__device__ static inline float4 mma_m16n16k32_bf8_fp8_f32(uint2 fa, uint2 fb,
                                                          float4 c) {
#if HAS_AMD_BF8_FP8_MFMA
    v4f ret = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(
        *reinterpret_cast<const long *>(&fa),
        *reinterpret_cast<const long *>(&fb), *reinterpret_cast<v4f *>(&c), 0,
        0, 0);
    return *reinterpret_cast<const float4 *>(&ret);
#else
    return {0, 0, 0, 0};
#endif
}

__device__ static inline float4 mma_m16n16k32_fp8_bf8_f32(uint2 fa, uint2 fb,
                                                          float4 c) {
#if HAS_AMD_FP8_BF8_MFMA
    v4f ret = __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(
        *reinterpret_cast<const long *>(&fa),
        *reinterpret_cast<const long *>(&fb), *reinterpret_cast<v4f *>(&c), 0,
        0, 0);
    return *reinterpret_cast<const float4 *>(&ret);
#else
    return {0, 0, 0, 0};
#endif
}

__device__ static inline v16f mma_m32n32k8_fp16(uint2 fa, uint2 fb, v16f c) {
#if defined(__HIP_DEVICE_COMPILE__)
    v16f ret = __builtin_amdgcn_mfma_f32_32x32x8f16(
        reinterpret_cast<const v4h &>(fa), reinterpret_cast<const v4h &>(fb), c,
        0, 0, 0);
#else
    v16f ret = {
        0,
    };
#endif
    return ret;
}

__device__ static inline v16f mma_m32n32k8_bf16(uint2 fa, uint2 fb, v16f c) {
#if defined(__gfx942__)
    v16f ret = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
        reinterpret_cast<const v4s &>(fa), reinterpret_cast<const v4s &>(fb), c,
        0, 0, 0);
#else
    v16f ret = {
        0,
    };
#endif
    return ret;
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

    template <int kAux>
    __device__ inline unsigned LoadU32(int voffset, int soffset) const {
        int v =
            llvm_amdgcn_raw_buffer_load_i32(content, voffset, soffset, kAux);
        return static_cast<unsigned>(v);
    }

    template <int kAux, int kSize, int kOffset>
    __device__ inline void
    LoadLds(__attribute__((address_space(3))) unsigned *lds_ptr, int voffset,
            int soffset) const {
        llvm_amdgcn_raw_buffer_load_lds(content, lds_ptr, kSize, voffset,
                                        soffset, kOffset, kAux);
    }
};

// In CDNA, OOB access to SHM are discarded. We leverage this property to
// eliminate the branches.
template <class T>
__device__ static inline T *GetConditionShmPtr(T *ptr, bool cond) {
    static constexpr unsigned kMaxShmSize = 64 * 1024;
    return cond ? ptr : reinterpret_cast<T *>(kMaxShmSize);
}

template <int kVmCnt = -1, int kExpCnt = -1, int kLgkmCnt = -1>
__device__ static inline void amdgcn_s_waitcnt() {
    static_assert(kVmCnt < 64, "");
    static_assert(kExpCnt < 8, "");
    static_assert(kLgkmCnt < 16, "");
    static constexpr unsigned kVmCntU = (unsigned)kVmCnt & 63;
    static constexpr unsigned kValue =
        ((kVmCntU & 48) << 10) | ((((unsigned)kLgkmCnt) & 15) << 8) |
        (((unsigned)kExpCnt & 7) << 4) | (kVmCntU & 15);
    __builtin_amdgcn_s_waitcnt(kValue);
}

template <int kVmCnt = -1, int kExpCnt = -1, int kLgkmCnt = -1>
__device__ static inline void amdgcn_s_waitcnt_barrier() {
    amdgcn_s_waitcnt<kVmCnt, kExpCnt, kLgkmCnt>();
    __builtin_amdgcn_s_barrier();
}

template <unsigned kPrio> __device__ static inline void amdgcn_s_setprio() {
    static_assert(kPrio <= 3, "s_setprio supports priority values in [0, 3]");
    __builtin_amdgcn_s_setprio(kPrio);
}

} // namespace causalflow::petit::rocm
