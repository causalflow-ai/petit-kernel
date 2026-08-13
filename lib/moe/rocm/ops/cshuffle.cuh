#pragma once

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

struct RowMajorCShuffle {
    template <unsigned kGroupDim, unsigned kAccumFragments,
              unsigned kTokenPairs, unsigned kNumWarps,
              unsigned kSubGroupRowWords, class Shm>
    __device__ static void Write(Shm &shm, unsigned stage,
                                 const uint2 o[kAccumFragments], unsigned wid,
                                 unsigned wtid) {
        static_assert(kGroupDim == 256, "row-major C-shuffle requires N256");
        auto *const dst =
            reinterpret_cast<unsigned short *>(&shm[stage][0]);
        const unsigned n_lane = wtid % 16;
        const unsigned m4 = wtid / 16;
#pragma unroll
        for (unsigned i = 0; i < kAccumFragments; ++i) {
            const unsigned fragment = i / 2;
            const unsigned m16 = i % 2;
            const unsigned n = wid * 64 + fragment * 16 + n_lane;
#pragma unroll
            for (unsigned component = 0; component < 4; ++component) {
                const unsigned m = m16 * 16 + m4 * 4 + component;
                dst[m * kGroupDim + n] =
                    reinterpret_cast<const unsigned short *>(&o[i])[component];
            }
        }
    }

    template <unsigned kGroupDim, unsigned kTokenBatch, unsigned kNumWarps,
              unsigned kSubGroupRowWords, class Shm>
    __device__ static void Read(Shm &shm, unsigned stage,
                                uint2 o[kTokenBatch], unsigned wid,
                                unsigned wtid) {
        static_assert(kGroupDim == 256, "row-major C-shuffle requires N256");
        const auto *const src = &shm[stage][0];
#pragma unroll
        for (unsigned i = 0; i < kTokenBatch; ++i) {
            const unsigned m = wid * kTokenBatch + i;
            const unsigned n = wtid * 2;
            o[i].x = src[(m * kGroupDim + n) / 2];
            o[i].y = src[(m * kGroupDim + 128 + n) / 2];
        }
    }
};

struct InterleavedRowMajorCShuffle {
    template <unsigned kGroupDim, unsigned kAccumFragments,
              unsigned kTokenPairs, unsigned kNumWarps,
              unsigned kSubGroupRowWords, class Shm>
    __device__ static void Write(Shm &shm, unsigned stage,
                                 const uint2 o[kAccumFragments], unsigned wid,
                                 unsigned wtid) {
        static_assert(kGroupDim == 256 && kAccumFragments == 8);
        auto *const dst =
            reinterpret_cast<unsigned short *>(&shm[stage][0]);
        const unsigned row = wtid % 16;
#pragma unroll
        for (unsigned i = 0; i < kAccumFragments; ++i) {
            const unsigned physical_row = (i & 1u) * 16 + row;
            const unsigned col_base =
                (i / 2) * 64 + wid * 16 + (wtid / 16) * 4;
#pragma unroll
            for (unsigned component = 0; component < 4; ++component) {
                dst[physical_row * kGroupDim + col_base + component] =
                    reinterpret_cast<const unsigned short *>(&o[i])[component];
            }
        }
    }

    template <unsigned kGroupDim, unsigned kTokenBatch, unsigned kNumWarps,
              unsigned kSubGroupRowWords, class Shm>
    __device__ static void Read(Shm &shm, unsigned stage,
                                uint2 o[kTokenBatch], unsigned wid,
                                unsigned wtid) {
        RowMajorCShuffle::template Read<kGroupDim, kTokenBatch, kNumWarps,
                                        kSubGroupRowWords>(shm, stage, o, wid,
                                                           wtid);
    }
};

struct BlockedVectorRowMajorCShuffle {
    template <unsigned kGroupDim, unsigned kAccumFragments,
              unsigned kTokenPairs, unsigned kNumWarps,
              unsigned kSubGroupRowWords, class Shm>
    __device__ static void Write(Shm &shm, unsigned stage,
                                 const uint2 o[kAccumFragments], unsigned wid,
                                 unsigned wtid) {
        static_assert(kGroupDim == 256 && kAccumFragments == 8);
        auto *const dst =
            reinterpret_cast<unsigned short *>(&shm[stage][0]);
        const unsigned row = wtid % 16;
#pragma unroll
        for (unsigned i = 0; i < kAccumFragments; ++i) {
            const unsigned physical_row = (i & 1u) * 16 + row;
            const unsigned col_base =
                wid * 64 + (i / 2) * 16 + (wtid / 16) * 4;
#pragma unroll
            for (unsigned component = 0; component < 4; ++component) {
                dst[physical_row * kGroupDim + col_base + component] =
                    reinterpret_cast<const unsigned short *>(&o[i])[component];
            }
        }
    }

    template <unsigned kGroupDim, unsigned kTokenBatch, unsigned kNumWarps,
              unsigned kSubGroupRowWords, class Shm>
    __device__ static void Read(Shm &shm, unsigned stage,
                                uint2 o[kTokenBatch], unsigned wid,
                                unsigned wtid) {
        RowMajorCShuffle::template Read<kGroupDim, kTokenBatch, kNumWarps,
                                        kSubGroupRowWords>(shm, stage, o, wid,
                                                           wtid);
    }
};

} // namespace causalflow::petit::rocm::moe
