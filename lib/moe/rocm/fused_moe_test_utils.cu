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

template <class Output>
__global__ void AddRowBiasKernel(Output *data, const __hip_bfloat16 *bias,
                                 unsigned rows, unsigned cols) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(rows) * cols;
    if (idx >= count) {
        return;
    }
    const float value = [&] {
        if constexpr (std::is_same_v<Output, float>) {
            return data[idx];
        } else {
            return __bfloat162float(data[idx]);
        }
    }();
    const float sum = value + __bfloat162float(bias[idx % cols]);
    if constexpr (std::is_same_v<Output, float>) {
        data[idx] = sum;
    } else {
        data[idx] = __float2bfloat16(sum);
    }
}

__device__ float ScaleFloatForMxFp4(float max_abs, unsigned &scale_byte) {
    if (max_abs < 1.0e-12f) {
        scale_byte = 127u;
        return 1.0f;
    }
    const float required = max_abs * (1.0f / 6.0f);
    const unsigned required_bits = reinterpret_cast<const unsigned &>(required);
    scale_byte = (required_bits >> 23) & 0xffu;
    if (scale_byte < 0xffu && (required_bits & 0x7fffffu)) {
        ++scale_byte;
    }
    const unsigned bits = scale_byte << 23;
    return reinterpret_cast<const float &>(bits);
}

__device__ unsigned CvtScaleFp4x2(__hip_bfloat162 value, float scale) {
#if defined(__gfx950__) &&                                                     \
    __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16)
    return __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(0u, value, scale, 0) &
           0xffu;
#else
    (void)value;
    (void)scale;
    return 0;
#endif
}

__device__ __hip_bfloat162 CvtFp4ByteToBf16x2(unsigned packed, float scale) {
#if defined(__gfx950__) &&                                                     \
    __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_bf16_fp4)
    return __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(packed, scale, 0);
#else
    (void)packed;
    (void)scale;
    return {};
#endif
}

static constexpr unsigned kMxFp4ScaleGroupsPerDequantBlock = 16;
static constexpr unsigned kMxFp4PairsPerScaleGroup = 16;
static constexpr unsigned kMxFp4DequantThreads =
    kMxFp4ScaleGroupsPerDequantBlock * kMxFp4PairsPerScaleGroup;

__global__ void QuantizeDequantMxFp4Kernel(__hip_bfloat16 *data,
                                           unsigned rows, unsigned cols) {
    static constexpr unsigned kGroup = 32;
    __shared__ float max_abs_shm[kGroup];

    const unsigned row = blockIdx.y;
    const unsigned group = blockIdx.x;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= kGroup) {
        return;
    }

    const unsigned col = group * kGroup + lane;
    const bool valid = col < cols;
    const size_t base = static_cast<size_t>(row) * cols;
    const float value = valid ? __bfloat162float(data[base + col]) : 0.0f;
    max_abs_shm[lane] = fabsf(value);
    __syncthreads();

    for (unsigned offset = kGroup / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            max_abs_shm[lane] =
                fmaxf(max_abs_shm[lane], max_abs_shm[lane + offset]);
        }
        __syncthreads();
    }

    unsigned scale_byte;
    const float scale = ScaleFloatForMxFp4(max_abs_shm[0], scale_byte);
    if (lane < kGroup / 2) {
        const unsigned even_col = group * kGroup + 2 * lane;
        const __hip_bfloat16 a =
            even_col < cols ? data[base + even_col]
                            : __float2bfloat16(0.0f);
        const __hip_bfloat16 b =
            (even_col + 1 < cols) ? data[base + even_col + 1]
                                  : __float2bfloat16(0.0f);
        const unsigned packed = CvtScaleFp4x2(__hip_bfloat162(a, b), scale);
        const __hip_bfloat162 bf16x2 = CvtFp4ByteToBf16x2(packed, scale);
        if (even_col < cols) {
            data[base + even_col] = bf16x2.x;
        }
        if (even_col + 1 < cols) {
            data[base + even_col + 1] = bf16x2.y;
        }
    }
}

__global__ void DequantizeNativeMxFp4ActivationsKernel(
    const unsigned char *__restrict__ q,
    const unsigned char *__restrict__ scale,
    __hip_bfloat16 *__restrict__ dq, unsigned cols) {
    __shared__ float scale_values[kMxFp4ScaleGroupsPerDequantBlock];

    const unsigned row = blockIdx.y;
    const unsigned scale_group_begin =
        blockIdx.x * kMxFp4ScaleGroupsPerDequantBlock;
    const unsigned scale_cols = cols / 32;
    const unsigned valid_scale_groups =
        min(kMxFp4ScaleGroupsPerDequantBlock,
            scale_cols - scale_group_begin);
    const unsigned tid = threadIdx.x;

    if (tid < valid_scale_groups) {
        const unsigned scale_byte =
            scale[static_cast<size_t>(row) * scale_cols + scale_group_begin +
                  tid];
        const unsigned scale_bits = scale_byte << 23;
        scale_values[tid] = reinterpret_cast<const float &>(scale_bits);
    }
    __syncthreads();

    const unsigned local_scale_group = tid / kMxFp4PairsPerScaleGroup;
    if (local_scale_group >= valid_scale_groups) {
        return;
    }

    const unsigned pair_col =
        (scale_group_begin + local_scale_group) * kMxFp4PairsPerScaleGroup +
        (tid % kMxFp4PairsPerScaleGroup);
    const size_t pair_idx = static_cast<size_t>(row) * (cols / 2) + pair_col;
    const __hip_bfloat162 bf16x2 =
        CvtFp4ByteToBf16x2(q[pair_idx], scale_values[local_scale_group]);
    reinterpret_cast<unsigned *>(dq)[pair_idx] =
        reinterpret_cast<const unsigned &>(bf16x2);
}

__global__ void RepackBf16BiasKernel(__hip_bfloat16 *output,
                                     const __hip_bfloat16 *input,
                                     unsigned rows, unsigned cols) {
    static constexpr unsigned kColsPerTile = 256;
    const unsigned tid = threadIdx.x;
    const unsigned tile_id = blockIdx.x;
    const unsigned row = blockIdx.y;
    const unsigned tiles = tal::CeilingDiv<unsigned>(cols, kColsPerTile);
    const unsigned padded_cols = tiles * kColsPerTile;
    (void)rows;

    const unsigned col = tile_id * kColsPerTile + tid;
    using namespace causalflow::tal;
    // Packed coordinates are [wave][q][fragment][component]. Logical bias is
    // [wave][fragment][q][component].
    const auto source_layout =
        make_layout(make_shape(C<4>{}, C<4>{}, C<4>{}, C<4>{}),
                    make_stride(_1{}, C<16>{}, C<4>{}, C<64>{}));
    const unsigned src_col =
        tile_id * kColsPerTile + source_layout(tid);
    output[static_cast<size_t>(row) * padded_cols + col] =
        src_col < cols
            ? input[static_cast<size_t>(row) * cols + src_col]
            : __float2bfloat16(0.0f);
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

__global__ void RepackNativeMxFp4WeightsKernel(
    unsigned *__restrict__ output, const unsigned *__restrict__ input,
    unsigned rows, unsigned cols) {
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

__global__ void RepackPetitMxFp4WeightsKernel(
    unsigned *__restrict__ output, const unsigned *__restrict__ input,
    unsigned rows, unsigned cols) {
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
        make_stride(make_stride(C<64>{}, C<4>{}, _1{}),
                    make_stride(256 * k128_blocks, 1024 * k128_blocks)));
    for (unsigned idx = tid; idx < kWordsPerTile; idx += blockDim.x) {
        const size_t output_base =
            static_cast<size_t>(n256) * 4096 * k128_blocks + k128 * 256;
        output[output_base + output_tile_layout(idx)] =
            quantization::fp4::PetitFormat(tile[input_tile_layout(idx)]);
    }
}

__global__ void RepackNativeMxFp4ScalesKernel(
    unsigned *__restrict__ output, const unsigned *__restrict__ input,
    unsigned rows, unsigned scale_cols) {
    using namespace causalflow::tal;
    static constexpr unsigned kRowsPerTile = 32;
    static constexpr unsigned kScalesPerTile = 8;
    static constexpr unsigned kBytesPerTile =
        kRowsPerTile * kScalesPerTile;
    const unsigned idx = threadIdx.x;
    const unsigned n32 = blockIdx.y;
    const unsigned k256 = blockIdx.x;
    const unsigned k256_blocks = scale_cols / kScalesPerTile;
    const unsigned src_base =
        n32 * kRowsPerTile * scale_cols + k256 * kScalesPerTile;
    const unsigned dst_base =
        (n32 * k256_blocks + k256) * kBytesPerTile;
    const auto src = make_layout(
        make_shape(make_shape(C<2>{}, C<16>{}),
                   make_shape(C<2>{}, C<4>{})),
        make_stride(make_stride(16 * scale_cols, scale_cols),
                    make_stride(C<4>{}, _1{})));
    const auto dst = make_layout(
        make_shape(make_shape(C<2>{}, C<16>{}),
                   make_shape(C<2>{}, C<4>{})),
        make_stride(make_stride(_1{}, C<4>{}),
                    make_stride(C<2>{}, C<64>{})));
    auto *out = reinterpret_cast<unsigned char *>(output);
    const auto *in = reinterpret_cast<const unsigned char *>(input);
    (void)rows;
    out[dst_base + dst(idx)] = in[src_base + src(idx)];
}

__global__ void RepackPetitMxFp4ScalesKernel(
    unsigned *__restrict__ output, const unsigned *__restrict__ input,
    unsigned rows, unsigned scale_cols) {
    using namespace causalflow::tal;
    static constexpr unsigned kRowsPerTile = 32;
    static constexpr unsigned kScalesPerTile = 8;
    static constexpr unsigned kBytesPerTile =
        kRowsPerTile * kScalesPerTile;
    const unsigned idx = threadIdx.x;
    const unsigned n32 = blockIdx.y;
    const unsigned k256 = blockIdx.x;
    const unsigned k256_blocks = scale_cols / kScalesPerTile;
    const unsigned src_base =
        n32 * kRowsPerTile * scale_cols + k256 * kScalesPerTile;
    const unsigned dst_base =
        (n32 * k256_blocks + k256) * kBytesPerTile;
    const auto src = make_layout(make_shape(C<8>{}, C<4>{}, C<8>{}),
                                 make_stride(4 * scale_cols, scale_cols, _1{}));
    const auto dst = make_layout(make_shape(C<8>{}, C<4>{}, C<8>{}),
                                 make_stride(C<32>{}, _1{}, C<4>{}));
    auto *out = reinterpret_cast<unsigned char *>(output);
    const auto *in = reinterpret_cast<const unsigned char *>(input);
    (void)rows;
    out[dst_base + dst(idx)] = in[src_base + src(idx)];
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

hipError_t AddFloatRowBias(float *data, const __hip_bfloat16 *bias,
                           unsigned rows, unsigned cols, hipStream_t stream) {
    static constexpr unsigned kThreads = 256;
    const size_t count = static_cast<size_t>(rows) * cols;
    const dim3 grid(tal::CeilingDiv<size_t>(count, kThreads));
    AddRowBiasKernel<<<grid, kThreads, 0, stream>>>(data, bias, rows, cols);
    return hipGetLastError();
}

hipError_t AddBf16RowBias(__hip_bfloat16 *data, const __hip_bfloat16 *bias,
                          unsigned rows, unsigned cols, hipStream_t stream) {
    static constexpr unsigned kThreads = 256;
    const size_t count = static_cast<size_t>(rows) * cols;
    const dim3 grid(tal::CeilingDiv<size_t>(count, kThreads));
    AddRowBiasKernel<<<grid, kThreads, 0, stream>>>(data, bias, rows, cols);
    return hipGetLastError();
}

hipError_t QuantizeDequantMxFp4(__hip_bfloat16 *data, unsigned rows,
                                unsigned cols, hipStream_t stream) {
    static constexpr unsigned kGroup = 32;
    if (data == nullptr || rows == 0 || cols == 0) {
        return hipErrorInvalidValue;
    }

    const dim3 block(kGroup);
    const dim3 grid(tal::CeilingDiv<unsigned>(cols, kGroup), rows);
    QuantizeDequantMxFp4Kernel<<<grid, block, 0, stream>>>(data, rows, cols);
    return hipGetLastError();
}

hipError_t DequantizeNativeMxFp4Activations(
    const unsigned char *q, const unsigned char *scale, __hip_bfloat16 *dq,
    unsigned rows, unsigned cols, hipStream_t stream) {
    if (q == nullptr || scale == nullptr || dq == nullptr || rows == 0 ||
        cols == 0 || cols % 32 != 0) {
        return hipErrorInvalidValue;
    }

    const dim3 block(kMxFp4DequantThreads);
    const dim3 grid(tal::CeilingDiv<unsigned>(
                        cols / 32, kMxFp4ScaleGroupsPerDequantBlock),
                    rows);
    DequantizeNativeMxFp4ActivationsKernel<<<grid, block, 0, stream>>>(
        q, scale, dq, cols);
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

hipError_t RepackMxFp4Bias(__hip_bfloat16 *output,
                           const __hip_bfloat16 *input, unsigned rows,
                           unsigned cols, hipStream_t stream) {
    static constexpr unsigned kThreads = 256;
    if (output == nullptr || input == nullptr || rows == 0 || cols == 0) {
        return hipErrorInvalidValue;
    }

    static constexpr unsigned kColsPerTile = 256;
    const dim3 block(kThreads);
    const dim3 grid(tal::CeilingDiv<unsigned>(cols, kColsPerTile), rows);
    RepackBf16BiasKernel<<<grid, block, 0, stream>>>(output, input, rows, cols);
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
    RepackNativeMxFp4WeightsKernel<<<grid, block, 0, stream>>>(
        output, input, rows, cols);
    return hipGetLastError();
}

hipError_t RepackNativeMxFp4Scales(unsigned *output, const unsigned *input,
                                   unsigned rows, unsigned scale_cols,
                                   hipStream_t stream) {
    static constexpr unsigned kThreads = 256;
    if (output == nullptr || input == nullptr || rows % 32 != 0 ||
        scale_cols % 8 != 0) {
        return hipErrorInvalidValue;
    }

    const dim3 block(kThreads);
    const dim3 grid(scale_cols / 8, rows / 32);
    RepackNativeMxFp4ScalesKernel<<<grid, block, 0, stream>>>(
        output, input, rows, scale_cols);
    return hipGetLastError();
}

hipError_t RepackPetitMxFp4Weights(unsigned *output, const unsigned *input,
                                   unsigned rows, unsigned cols,
                                   hipStream_t stream) {
    static constexpr unsigned kRowsPerTile = 256;
    static constexpr unsigned kColsPerTile = 128;
    static constexpr unsigned kThreads = 256;
    if (output == nullptr || input == nullptr ||
        rows % kRowsPerTile != 0 ||
        cols % kColsPerTile != 0) {
        return hipErrorInvalidValue;
    }

    const dim3 block(kThreads);
    const dim3 grid(cols / kColsPerTile, rows / kRowsPerTile);
    RepackPetitMxFp4WeightsKernel<<<grid, block, 0, stream>>>(
        output, input, rows, cols);
    return hipGetLastError();
}

hipError_t RepackPetitMxFp4Scales(unsigned *output, const unsigned *input,
                                  unsigned rows, unsigned scale_cols,
                                  hipStream_t stream) {
    static constexpr unsigned kThreads = 256;
    if (output == nullptr || input == nullptr || rows % 32 != 0 ||
        scale_cols % 8 != 0) {
        return hipErrorInvalidValue;
    }

    const dim3 block(kThreads);
    const dim3 grid(scale_cols / 8, rows / 32);
    RepackPetitMxFp4ScalesKernel<<<grid, block, 0, stream>>>(
        output, input, rows, scale_cols);
    return hipGetLastError();
}

} // namespace causalflow::petit::rocm::moe::test_utils
