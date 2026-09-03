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

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__) &&                  \
    __has_builtin(__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4)
#define HAS_AMD_SCALE_FP4_MFMA 1
#else
#define HAS_AMD_SCALE_FP4_MFMA 0
#endif

namespace causalflow::petit::rocm {

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef _Float16 v4h __attribute__((ext_vector_type(4)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef short v4s __attribute__((ext_vector_type(4)));
typedef float v16f __attribute__((ext_vector_type(16)));

static constexpr unsigned kWarpSize = 64;

// Return the sum of values from lane 0 through the calling lane. This is a
// scan, rather than a wave reduction: every lane receives a different prefix.
__device__ inline unsigned amdgcn_wave_inclusive_add(unsigned value,
                                                      unsigned lane) {
    unsigned remote =
        __builtin_amdgcn_mov_dpp(value, 0x111, 0xf, 0xf, true);
    if (lane >= 1)
        value += remote;
    remote = __builtin_amdgcn_mov_dpp(value, 0x112, 0xf, 0xf, true);
    if (lane >= 2)
        value += remote;
    remote = __builtin_amdgcn_mov_dpp(value, 0x114, 0xf, 0xf, true);
    if (lane >= 4)
        value += remote;
    remote = __builtin_amdgcn_mov_dpp(value, 0x118, 0xf, 0xf, true);
    if (lane >= 8)
        value += remote;

    const unsigned source16 = (lane & 0x30u) - 1u;
    remote = __builtin_amdgcn_ds_bpermute(source16 * sizeof(unsigned), value);
    if (lane >= 16)
        value += remote;
    const unsigned source32 = (lane & 0x30u) - 17u;
    remote = __builtin_amdgcn_ds_bpermute(source32 * sizeof(unsigned), value);
    if (lane >= 32)
        value += remote;
    return value;
}

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

__device__ void llvm_amdgcn_raw_buffer_store_v2i32(
    v2i data, v4i rsrc, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.raw.buffer.store.v2i32");

__device__ void llvm_amdgcn_raw_buffer_store_i32(
    int data, v4i rsrc, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.raw.buffer.store.i32");

__device__ v2i llvm_amdgcn_raw_buffer_load_v2i32(
    v4i rsrc, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.raw.buffer.load.v2i32");

__device__ int llvm_amdgcn_raw_buffer_load_i32(
    v4i rsrc, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ int llvm_amdgcn_raw_buffer_atomic_add_i32(
    int data, v4i rsrc, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.raw.buffer.atomic.add.i32");

__device__ int llvm_amdgcn_raw_buffer_atomic_or_i32(
    int data, v4i rsrc, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.raw.buffer.atomic.or.i32");

__device__ long llvm_amdgcn_raw_buffer_atomic_add_i64(
    long data, v4i rsrc, int voffset, int soffset,
    int aux) __asm("llvm.amdgcn.raw.buffer.atomic.add.i64");

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

__device__ static inline unsigned amdgcn_cvt_pk_bf16_f32(float a, float b) {
#if defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__)
    unsigned packed;
    asm("v_cvt_pk_bf16_f32 %0, %1, %2"
        : "=v"(packed)
        : "v"(a), "v"(b));
    return packed;
#else
    const unsigned a_bits = __builtin_bit_cast(unsigned, a);
    const unsigned b_bits = __builtin_bit_cast(unsigned, b);
    const unsigned a_rounded =
        (a_bits & 0x7fffffffu) > 0x7f800000u
            ? 0x7fff0000u
            : a_bits + 0x7fffu + ((a_bits >> 16) & 1u);
    const unsigned b_rounded =
        (b_bits & 0x7fffffffu) > 0x7f800000u
            ? 0x7fff0000u
            : b_bits + 0x7fffu + ((b_bits >> 16) & 1u);
    return (a_rounded >> 16) | (b_rounded & 0xffff0000u);
#endif
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

__device__ static inline float4
mma_m16n16k128_fp8_fp8_f32(const uint2 *fa, const uint2 *fb, float4 c) {
#if HAS_AMD_SCALE_FP4_MFMA
    v4f ret = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        *reinterpret_cast<const v8i *>(fa), *reinterpret_cast<const v8i *>(fb),
        *reinterpret_cast<v4f *>(&c), 0, 0, 0, 0, 0, 0);
    return reinterpret_cast<const float4 &>(ret);
#elif HAS_AMD_FP8_MFMA
    c = mma_m16n16k32_fp8_fp8_f32(fa[0], fb[0], c);
    c = mma_m16n16k32_fp8_fp8_f32(fa[1], fb[1], c);
    c = mma_m16n16k32_fp8_fp8_f32(fa[2], fb[2], c);
    c = mma_m16n16k32_fp8_fp8_f32(fa[3], fb[3], c);
    return c;
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

template <int kOpSelA, int kOpSelB>
__device__ static inline float4
mma_scale_m16n16k128_fp4_fp4_f32(uint4 fa, unsigned scale_a, uint4 fb,
                                 unsigned scale_b, float4 c) {
#if HAS_AMD_SCALE_FP4_MFMA
    const auto a = reinterpret_cast<const v4i &>(fa);
    const auto b = reinterpret_cast<const v4i &>(fb);
    v4f ret = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        v8i{a[0], a[1], a[2], a[3], 0, 0, 0, 0},
        v8i{b[0], b[1], b[2], b[3], 0, 0, 0, 0},
        reinterpret_cast<const v4f &>(c), 4, 4, kOpSelA, scale_a, kOpSelB,
        scale_b);
    return reinterpret_cast<const float4 &>(ret);
#else
    (void)fa;
    (void)scale_a;
    (void)fb;
    (void)scale_b;
    return c;
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
    enum {
        kNone = 0,
        // gfx94+: bit 0 = SC0, bit 1 = NT, bit 3 = SWZ, bit 4 = SC1.
        // Keep the legacy GLC/SLC names for existing call sites.
        kSC0Bit = 1 << 0,
        kNTBit = 1 << 1,
        kSWZBit = 1 << 3,
        kSC1Bit = 1 << 4,
        kGLCBit = kSC0Bit,
        kSLCBit = kNTBit,
    };

    static constexpr unsigned kAtomicScopeAgent = kNone;
    static constexpr unsigned kAtomicScopeSystem = kSC1Bit;

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
    __device__ inline void StoreU64(int voffset, int soffset,
                                    uint2 data) const {
        v2i v = *reinterpret_cast<const v2i *>(&data);
        llvm_amdgcn_raw_buffer_store_v2i32(v, content, voffset, soffset, kAux);
    }

    template <int kAux>
    __device__ inline unsigned LoadU32(int voffset, int soffset) const {
        int v =
            llvm_amdgcn_raw_buffer_load_i32(content, voffset, soffset, kAux);
        return static_cast<unsigned>(v);
    }

    template <int kAux>
    __device__ inline void StoreU32(int voffset, int soffset,
                                    unsigned data) const {
        llvm_amdgcn_raw_buffer_store_i32(static_cast<int>(data), content,
                                         voffset, soffset, kAux);
    }

    template <int kAux>
    __device__ inline int AtomicAddI32(int voffset, int soffset,
                                       int data) const {
        return llvm_amdgcn_raw_buffer_atomic_add_i32(data, content, voffset,
                                                     soffset, kAux);
    }

    template <int kAux>
    __device__ inline unsigned AtomicOrU32(int voffset, int soffset,
                                           unsigned data) const {
        return static_cast<unsigned>(llvm_amdgcn_raw_buffer_atomic_or_i32(
            static_cast<int>(data), content, voffset, soffset, kAux));
    }

    template <int kAux>
    __device__ inline unsigned long long
    AtomicAddU64(int voffset, int soffset, unsigned long long data) const {
        return static_cast<unsigned long long>(
            llvm_amdgcn_raw_buffer_atomic_add_i64(
                static_cast<long>(data), content, voffset, soffset, kAux));
    }

    template <int kAux, int kSize, int kOffset>
    __device__ inline void
    LoadLds(__attribute__((address_space(3))) unsigned *lds_ptr, int voffset,
            int soffset) const {
#if defined(__gfx950__)
        llvm_amdgcn_raw_buffer_load_lds(content, lds_ptr, kSize, voffset,
                                        soffset, kOffset, kAux);
#else
        (void)lds_ptr;
        (void)voffset;
        (void)soffset;
#endif
    }
};

// In CDNA, OOB access to SHM are discarded. We leverage this property to
// eliminate the branches.
template <class T>
__device__ static inline T *GetConditionShmPtr(T *ptr, bool cond) {
    static constexpr unsigned kMaxShmSize = 160 * 1024;
    return cond ? ptr : reinterpret_cast<T *>(kMaxShmSize);
}

template <class T>
__device__ static inline __attribute__((address_space(3))) T *
GetConditionShmPtr(__attribute__((address_space(3))) T *ptr, bool cond) {
    static constexpr unsigned kMaxShmSize = 160 * 1024;
    return cond ? ptr
                : reinterpret_cast<__attribute__((address_space(3))) T *>(
                      kMaxShmSize);
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
