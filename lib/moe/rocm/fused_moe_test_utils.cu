#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/amd_intrinsics.cuh"
#include "gemm/rocm/quantization/fp4/quantization_utils.cuh"
#include "gemm/rocm/quantization/types.h"
#include "moe/rocm/ops/activation.cuh"
#include "moe/rocm/fused_moe_test_utils.h"

#include <type_traits>

namespace causalflow::petit::rocm::moe::test_utils {

__global__ void ElementwiseMultiplyKernel(const __hip_bfloat16 *a,
                                          const __hip_bfloat16 *b,
                                          __hip_bfloat16 *out, unsigned count) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) *
                                    __bfloat162float(b[idx]));
    }
}

template <class Input>
__global__ void ApplyOpenAIActivationKernel(const Input *gate, const Input *up,
                                            __hip_bfloat16 *out,
                                            unsigned count) {
    const unsigned idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < count) {
        float4 gate4;
        float4 up4;
        auto *g = reinterpret_cast<float *>(&gate4);
        auto *u = reinterpret_cast<float *>(&up4);
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned elem = idx + i;
            if constexpr (std::is_same_v<Input, __hip_bfloat16>) {
                g[i] = elem < count ? __bfloat162float(gate[elem]) : 0.0f;
                u[i] = elem < count ? __bfloat162float(up[elem]) : 0.0f;
            } else {
                g[i] = elem < count ? static_cast<float>(gate[elem]) : 0.0f;
                u[i] = elem < count ? static_cast<float>(up[elem]) : 0.0f;
            }
        }
        const float4 out4 = OpenAISwiGLUOp::Apply(gate4, up4);
        const auto *o = reinterpret_cast<const float *>(&out4);
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned elem = idx + i;
            if (elem < count) {
                out[elem] = __float2bfloat16(o[i]);
            }
        }
    }
}

__global__ void WeightedRouteScatterKernel(const __hip_bfloat16 *route_out,
                                           const unsigned *route_tokens,
                                           const float *route_weights,
                                           float *token_out, unsigned routes,
                                           unsigned cols) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(routes) * cols;
    if (idx >= total) {
        return;
    }

    const unsigned route = static_cast<unsigned>(idx / cols);
    const unsigned col = static_cast<unsigned>(idx % cols);
    const unsigned token = route_tokens[route];
    const float weighted =
        __bfloat162float(route_out[idx]) * route_weights[route];
    atomicAdd(token_out + static_cast<size_t>(token) * cols + col, weighted);
}

__global__ void RepackBf16BiasDppLayoutKernel(__hip_bfloat16 *output,
                                              const __hip_bfloat16 *input,
                                              unsigned rows, unsigned cols) {
    using namespace causalflow::tal;
    static constexpr unsigned kColsPerTile = 512;
    __shared__ __hip_bfloat16 tile[kColsPerTile];

    const unsigned tid = threadIdx.x;
    const unsigned tile_id = blockIdx.x;
    const unsigned row = blockIdx.y;
    const unsigned tiles = CeilingDiv<unsigned>(cols, kColsPerTile);
    (void)rows;

    for (unsigned idx = tid; idx < kColsPerTile; idx += blockDim.x) {
        const unsigned col = tile_id * kColsPerTile + idx;
        tile[idx] = col < cols
                        ? input[static_cast<size_t>(row) * cols + col]
                        : __float2bfloat16(0.0f);
    }
    __syncthreads();

    using SourceLayout = Layout<
        Shape<Shape<C<4>, C<2>>, Shape<C<4>, C<4>>, C<4>>,
        Stride<Stride<_1, C<256>>, Stride<C<64>, C<4>>, C<16>>>;
    const SourceLayout source_layout;
    const size_t output_base =
        (static_cast<size_t>(row) * tiles + tile_id) * kColsPerTile;
    for (unsigned idx = tid; idx < kColsPerTile; idx += blockDim.x) {
        output[output_base + idx] = tile[source_layout(idx)];
    }
}

template <unsigned kBlockSize>
__global__ static void
DequantizeShuffledBlockScaleFp8Kernel(const unsigned char *q,
                                      const float *scale, __hip_bfloat16 *dq,
                                      unsigned rows, unsigned cols) {
    using namespace causalflow::tal;
    static constexpr unsigned kQuantBlockK = 128;
    static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(unsigned char);
    static constexpr unsigned kTileRows = 128;
    static constexpr unsigned kTileCols = 128;
    static constexpr unsigned kBlockN = 16;
    static constexpr unsigned kBlockK = 32;
    static constexpr unsigned kPackK = 16;
    static constexpr unsigned kKk = kBlockK / kPackK;
    static constexpr unsigned kPackVecPerPackK = kPackK / 4;
    static constexpr unsigned kTileColBlocks = kTileCols / kBlockK;
    static constexpr unsigned kTileRowBlocks = kTileRows / kBlockN;
    static constexpr unsigned kRowBlockVecs = kTileColBlocks * kKk * kBlockN;

    __shared__ uint4 shm_u4[kTileRows * kTileCols / kVecSize];

    const unsigned tid = threadIdx.x, id_m = blockIdx.y, id_n = blockIdx.x,
                   id_e = blockIdx.z;
    [[assume(tid < kBlockSize)]];

    const uint4 *input_ptr = reinterpret_cast<const uint4 *>(
        q + id_e * rows * cols + id_m * cols / kBlockK * kTileRowBlocks +
        id_n * kTileColBlocks);
    const float tile_scale =
        scale[id_e * rows / kQuantBlockK * cols / kQuantBlockK +
              id_m * cols / kQuantBlockK + id_n];
    const v4f tile_scale4{tile_scale, tile_scale, tile_scale, tile_scale};

    for (unsigned idx = tid; idx < kTileRows * kTileCols / kVecSize;
         idx += kBlockSize) {
        const unsigned row = idx / kRowBlockVecs;
        const unsigned col = idx % kRowBlockVecs;
        shm_u4[idx] = input_ptr[(row * cols / kBlockK + id_n * kTileColBlocks) *
                                    kRowBlockVecs +
                                col];
    }
    __syncthreads();

    auto *const shm = reinterpret_cast<unsigned *>(shm_u4);
    using PackedShape = Shape<C<kPackVecPerPackK>, C<kBlockN>, C<kKk>,
                              C<kTileColBlocks>, C<kTileRowBlocks>>;

    struct DequantVec4StoreBf16 {
        using PackedType = uint2;

        __device__ static PackedType Convert(v4f out) {
            PackedType packed;
            auto *bf16x2 = reinterpret_cast<__hip_bfloat162 *>(&packed);
            bf16x2[0] = __float22bfloat162_rn(float2{out.x, out.y});
            bf16x2[1] = __float22bfloat162_rn(float2{out.z, out.w});
            return packed;
        }
        __device__ static PackedType *Ptr(__hip_bfloat16 *ptr) {
            return reinterpret_cast<PackedType *>(ptr);
        }
    };

    using Store = DequantVec4StoreBf16;
    const auto output_layout = make_layout(
        PackedShape{}, make_stride(_1{}, cols / 4, C<kPackVecPerPackK>{},
                                   C<kBlockK / 4>{}, kBlockN * (cols / 4)));
    __hip_bfloat16 *output_ptr =
        dq + id_e * rows * cols + id_m * kTileRows * cols + id_n * kTileCols;
    auto o = Store::Ptr(output_ptr);
    for (unsigned idx = tid; idx < kTileRows * kTileCols / 4;
         idx += kBlockSize) {
#if defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__)
        __hip_fp8x4_e4m3 packed;
#else
        __hip_fp8x4_e4m3_fnuz packed;
#endif
        packed.__x = shm[idx];
        const float4 out_fp4 = static_cast<float4>(packed);
        v4f out = reinterpret_cast<const v4f &>(out_fp4);
        out *= tile_scale4;
        o[output_layout(idx)] = Store::Convert(out);
    }
}

__global__ void RepackBf16MxFp4CDNA4WeightsKernel(unsigned *__restrict__ output,
                                                  const unsigned *__restrict__ input,
                                                  unsigned rows,
                                                  unsigned cols) {
    using namespace causalflow::tal;
    static constexpr unsigned kRowsPerTile = 256;
    static constexpr unsigned kColsPerTile = 128;
    static constexpr unsigned kWordsPerTile = kRowsPerTile * kColsPerTile / 8;
    __shared__ unsigned tile[kWordsPerTile];

    const unsigned tid = threadIdx.x;
    const unsigned n256 = blockIdx.y;
    const unsigned k128 = blockIdx.x;
    const unsigned k128_blocks = cols / kColsPerTile;
    const unsigned row_words = cols / 8;
    (void)rows;

    for (unsigned idx = tid; idx < kWordsPerTile; idx += blockDim.x) {
        const unsigned tile_row = idx / 16;
        const unsigned tile_word_col = idx % 16;
        const unsigned row = n256 * kRowsPerTile + tile_row;
        const unsigned word_col = k128 * 16 + tile_word_col;
        tile[idx] = input[static_cast<size_t>(row) * row_words + word_col];
    }
    __syncthreads();

    const auto input_tile_layout = make_layout(
        make_shape(make_shape(C<4>{}, C<16>{}, C<4>{}),
                   make_shape(C<4>{}, C<4>{})),
        make_stride(make_stride(_1{}, C<16>{}, C<4>{}),
                    make_stride(C<256>{}, C<1024>{})));
    const auto output_tile_layout = make_layout(
        make_shape(make_shape(C<4>{}, C<16>{}, C<4>{}),
                   make_shape(C<4>{}, C<4>{})),
        make_stride(make_stride(_1{}, C<4>{}, C<64>{}),
                    make_stride(256 * k128_blocks, 1024 * k128_blocks)));

    for (unsigned idx = tid; idx < kWordsPerTile; idx += blockDim.x) {
        const size_t output_base =
            static_cast<size_t>(n256) * 4096 * k128_blocks + k128 * 256;
        output[output_base + output_tile_layout(idx)] =
            tile[input_tile_layout(idx)];
    }
}

__global__ void RepackPetitMxFp4MoeWeightsKernel(unsigned *__restrict__ output,
                                                 const unsigned *__restrict__ input,
                                                 unsigned rows,
                                                 unsigned cols) {
    using namespace causalflow::tal;
    static constexpr unsigned kRowsPerTile = 16;
    static constexpr unsigned kColsPerTile = 128;
    static constexpr unsigned kWordsPerTile = kRowsPerTile * kColsPerTile / 8;
    __shared__ unsigned tile[kWordsPerTile];

    const unsigned tid = threadIdx.x;
    const unsigned row16_block = blockIdx.y;
    const unsigned k128 = blockIdx.x;
    const unsigned k128_blocks = cols / kColsPerTile;
    const unsigned row_words = cols / 8;
    (void)rows;

    const auto baseline_tile_layout =
        make_layout(make_shape(C<16>{}, C<16>{}),
                    make_stride(_1{}, row_words));
    const size_t baseline_base =
        static_cast<size_t>(row16_block) * kRowsPerTile * row_words +
        k128 * 16;
    for (unsigned idx = tid; idx < kWordsPerTile; idx += blockDim.x) {
        tile[idx] = input[baseline_base + baseline_tile_layout(idx)];
    }
    __syncthreads();

    const auto source_tile_layout =
        make_layout(make_shape(make_shape(C<4>{}, C<16>{}), C<4>{}),
                    make_stride(make_stride(C<4>{}, C<16>{}), _1{}));
    const auto destination_tile_layout =
        make_layout(make_shape(make_shape(C<4>{}, C<16>{}), C<4>{}),
                    make_stride(make_stride(_1{}, C<4>{}), C<64>{}));

    const size_t destination_base =
        static_cast<size_t>(row16_block) * 256 * k128_blocks + k128 * 256;
    for (unsigned idx = tid; idx < kWordsPerTile; idx += blockDim.x) {
        output[destination_base + destination_tile_layout(idx)] =
            quantization::fp4::PetitFormat(tile[source_tile_layout(idx)]);
    }
}

__global__ void RepackPetitMxFp4MoeScalesKernel(unsigned *__restrict__ output,
                                                const unsigned *__restrict__ input,
                                                unsigned rows,
                                                unsigned scale_cols) {
    using namespace causalflow::tal;
    static constexpr unsigned kRowsPerTile = 256;
    static constexpr unsigned kScalesPerTile = 4;
    static constexpr unsigned kBytesPerTile = kRowsPerTile * kScalesPerTile;
    __shared__ unsigned char tile[kBytesPerTile];

    const unsigned tid = threadIdx.x;
    const unsigned n256 = blockIdx.y;
    const unsigned kblock = blockIdx.x;
    const unsigned kblocks = scale_cols / kScalesPerTile;
    const auto *in_bytes = reinterpret_cast<const unsigned char *>(input);
    auto *out_bytes = reinterpret_cast<unsigned char *>(output);
    (void)rows;

    for (unsigned idx = tid; idx < kBytesPerTile; idx += blockDim.x) {
        const unsigned tile_row = idx / kScalesPerTile;
        const unsigned tile_col = idx % kScalesPerTile;
        const unsigned row = n256 * kRowsPerTile + tile_row;
        const unsigned scale_col = kblock * kScalesPerTile + tile_col;
        tile[idx] = in_bytes[static_cast<size_t>(row) * scale_cols + scale_col];
    }
    __syncthreads();

    const auto input_tile_layout = make_layout(
        make_shape(make_shape(C<4>{}, C<4>{}), C<4>{},
                   make_shape(C<4>{}, C<4>{})),
        make_stride(make_stride(C<4>{}, _1{}), C<256>{},
                    make_stride(C<16>{}, C<64>{})));
    const auto output_tile_layout = make_layout(
        make_shape(make_shape(C<4>{}, C<4>{}), C<4>{},
                   make_shape(C<4>{}, C<4>{})),
        make_stride(make_stride(_1{}, C<4>{}), C<16>{},
                    make_stride(C<64>{}, C<256>{})));

    for (unsigned idx = tid; idx < kBytesPerTile; idx += blockDim.x) {
        const size_t output_base =
            static_cast<size_t>(n256) * 1024 * kblocks + kblock * 1024;
        out_bytes[output_base + output_tile_layout(idx)] =
            tile[input_tile_layout(idx)];
    }
}

__global__ void RepackBf16MxFp4CDNA4ScalesKernel(unsigned *__restrict__ output,
                                                 const unsigned *__restrict__ input,
                                                 unsigned rows,
                                                 unsigned scale_cols) {
    using namespace causalflow::tal;
    static constexpr unsigned kRowsPerTile = 256;
    static constexpr unsigned kScalesPerTile = 4;
    static constexpr unsigned kBytesPerTile = kRowsPerTile * kScalesPerTile;
    __shared__ unsigned char tile[kBytesPerTile];

    const unsigned tid = threadIdx.x;
    const unsigned n256 = blockIdx.y;
    const unsigned k128 = blockIdx.x;
    const unsigned k128_blocks = scale_cols / kScalesPerTile;
    const auto *in_bytes = reinterpret_cast<const unsigned char *>(input);
    (void)rows;

    for (unsigned idx = tid; idx < kBytesPerTile; idx += blockDim.x) {
        const unsigned tile_row = idx / kScalesPerTile;
        const unsigned tile_col = idx % kScalesPerTile;
        const unsigned row = n256 * kRowsPerTile + tile_row;
        const unsigned scale_col = k128 * kScalesPerTile + tile_col;
        tile[idx] = in_bytes[static_cast<size_t>(row) * scale_cols + scale_col];
    }
    __syncthreads();

    const unsigned out_words_per_tile = kBytesPerTile / sizeof(unsigned);
    const auto input_tile_layout =
        make_layout(make_shape(C<4>{}, C<4>{}, C<16>{}, C<4>{}),
                    make_stride(C<256>{}, C<64>{}, C<4>{}, _1{}));
    const auto output_tile_layout =
        make_layout(make_shape(C<16>{}, C<4>{}, C<4>{}),
                    make_stride(_1{}, C<16>{}, C<64>{}));
    for (unsigned idx = tid; idx < out_words_per_tile; idx += blockDim.x) {
        const unsigned wid = idx / 64;
        const unsigned k32 = (idx % 64) / 16;
        const unsigned n16 = idx % 16;
        unsigned word = 0;
        for (unsigned n_iter = 0; n_iter < 4; ++n_iter) {
            word |= static_cast<unsigned>(
                        tile[input_tile_layout(
                            make_coord(n_iter, wid, n16, k32))])
                    << (8 * n_iter);
        }
        const size_t output_base =
            static_cast<size_t>(n256) * 256 * k128_blocks + k128 * 256;
        output[output_base + output_tile_layout(idx)] = word;
    }
}

hipError_t ApplyElementwiseMultiply(const __hip_bfloat16 *a,
                                    const __hip_bfloat16 *b,
                                    __hip_bfloat16 *out, unsigned count,
                                    hipStream_t stream) {
    static constexpr unsigned kThreads = 256;
    const dim3 block(kThreads);
    const dim3 grid(tal::CeilingDiv<unsigned>(count, block.x));
    ElementwiseMultiplyKernel<<<grid, block, 0, stream>>>(a, b, out, count);
    return hipGetLastError();
}

template <class Input>
hipError_t ApplyOpenAISwiGLU(const Input *gate, const Input *up,
                             __hip_bfloat16 *out, unsigned count,
                             hipStream_t stream) {
    static constexpr unsigned kThreads = 256;
    const dim3 block(kThreads);
    const dim3 grid(tal::CeilingDiv<unsigned>(tal::CeilingDiv(count, 4u),
                                              block.x));
    ApplyOpenAIActivationKernel<<<grid, block, 0, stream>>>(gate, up, out,
                                                            count);
    return hipGetLastError();
}

template hipError_t
ApplyOpenAISwiGLU<__hip_bfloat16>(const __hip_bfloat16 *, const __hip_bfloat16 *,
                                  __hip_bfloat16 *, unsigned, hipStream_t);
template hipError_t ApplyOpenAISwiGLU<float>(
    const float *, const float *, __hip_bfloat16 *, unsigned, hipStream_t);

hipError_t ScatterWeightedRoutes(const __hip_bfloat16 *route_out,
                                 const unsigned *route_tokens,
                                 const float *route_weights, float *token_out,
                                 unsigned routes, unsigned cols,
                                 hipStream_t stream) {
    static constexpr unsigned kThreads = 256;
    const dim3 block(kThreads);
    const dim3 grid(tal::CeilingDiv<unsigned>(routes * cols, block.x));
    WeightedRouteScatterKernel<<<grid, block, 0, stream>>>(
        route_out, route_tokens, route_weights, token_out, routes, cols);
    return hipGetLastError();
}

hipError_t RepackBf16BiasDppLayout(__hip_bfloat16 *output,
                                   const __hip_bfloat16 *input, unsigned rows,
                                   unsigned cols, hipStream_t stream) {
    static constexpr unsigned kThreads = 256;
    if (output == nullptr || input == nullptr || rows == 0 || cols == 0) {
        return hipErrorInvalidValue;
    }

    static constexpr unsigned kColsPerTile = 512;
    const dim3 block(kThreads);
    const dim3 grid(tal::CeilingDiv<unsigned>(cols, kColsPerTile), rows);
    RepackBf16BiasDppLayoutKernel<<<grid, block, 0, stream>>>(output, input,
                                                              rows, cols);
    return hipGetLastError();
}

hipError_t DequantizeShuffledBlockScaleFp8(const unsigned char *q,
                                           const float *scale,
                                           __hip_bfloat16 *dq, unsigned rows,
                                           unsigned cols, unsigned experts,
                                           hipStream_t stream) {
    static constexpr unsigned kThreads = 256;
    static constexpr unsigned kQuantBlockK = 128;
    if (rows % kQuantBlockK != 0 || cols % kQuantBlockK != 0) {
        return hipErrorInvalidValue;
    }

    const dim3 block(kThreads);
    const dim3 grid(cols / kQuantBlockK, rows / kQuantBlockK, experts);
    DequantizeShuffledBlockScaleFp8Kernel<kThreads>
        <<<grid, block, 0, stream>>>(q, scale, dq, rows, cols);
    return hipGetLastError();
}

hipError_t RepackNativeMxFp4Weights(unsigned *output, const unsigned *input,
                                       unsigned rows, unsigned cols,
                                       hipStream_t stream) {
    static constexpr unsigned kRowsPerTile = 256;
    static constexpr unsigned kColsPerTile = 128;
    static constexpr unsigned kThreads = 256;
    if (output == nullptr || input == nullptr || rows % kRowsPerTile != 0 ||
        cols % kColsPerTile != 0) {
        return hipErrorInvalidValue;
    }

    const dim3 block(kThreads);
    const dim3 grid(cols / kColsPerTile, rows / kRowsPerTile);
    RepackBf16MxFp4CDNA4WeightsKernel<<<grid, block, 0, stream>>>(output, input,
                                                                  rows, cols);
    return hipGetLastError();
}

hipError_t RepackNativeMxFp4Scales(unsigned *output, const unsigned *input,
                                      unsigned rows, unsigned scale_cols,
                                      hipStream_t stream) {
    static constexpr unsigned kRowsPerTile = 256;
    static constexpr unsigned kScalesPerTile = 4;
    static constexpr unsigned kThreads = 256;
    if (output == nullptr || input == nullptr || rows % kRowsPerTile != 0 ||
        scale_cols % kScalesPerTile != 0) {
        return hipErrorInvalidValue;
    }

    const dim3 block(kThreads);
    const dim3 grid(scale_cols / kScalesPerTile, rows / kRowsPerTile);
    RepackBf16MxFp4CDNA4ScalesKernel<<<grid, block, 0, stream>>>(
        output, input, rows, scale_cols);
    return hipGetLastError();
}

hipError_t RepackPetitMxFp4Weights(unsigned *output, const unsigned *input,
                                     unsigned rows, unsigned cols,
                                     hipStream_t stream) {
    static constexpr unsigned kRowsPerTile = 16;
    static constexpr unsigned kColsPerTile = 128;
    static constexpr unsigned kThreads = 256;
    if (output == nullptr || input == nullptr || rows % kRowsPerTile != 0 ||
        cols % kColsPerTile != 0) {
        return hipErrorInvalidValue;
    }

    const dim3 block(kThreads);
    const dim3 grid(cols / kColsPerTile, rows / kRowsPerTile);
    RepackPetitMxFp4MoeWeightsKernel<<<grid, block, 0, stream>>>(output, input,
                                                                 rows, cols);
    return hipGetLastError();
}

hipError_t RepackPetitMxFp4Scales(unsigned *output, const unsigned *input,
                                    unsigned rows, unsigned scale_cols,
                                    hipStream_t stream) {
    static constexpr unsigned kRowsPerTile = 256;
    static constexpr unsigned kScalesPerTile = 4;
    static constexpr unsigned kThreads = 256;
    if (output == nullptr || input == nullptr || rows % kRowsPerTile != 0 ||
        scale_cols % kScalesPerTile != 0) {
        return hipErrorInvalidValue;
    }

    const dim3 block(kThreads);
    const dim3 grid(scale_cols / kScalesPerTile, rows / kRowsPerTile);
    RepackPetitMxFp4MoeScalesKernel<<<grid, block, 0, stream>>>(
        output, input, rows, scale_cols);
    return hipGetLastError();
}

} // namespace causalflow::petit::rocm::moe::test_utils
