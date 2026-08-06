#include "moe/rocm/fused_moe.h"
#include "moe/rocm/fused_moe_test_utils.h"
#include "moe/rocm/quantization.cuh"
#include "causalflow/petit/tal/tensor/layout.h"
#include "tests/fp8_sampler.h"
#include "utils/hip_helper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <execution>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <gflags/gflags.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

DEFINE_string(backend, "fused_moe_rocm",
              "Backend to use for Fused MoE. Only 'fused_moe_rocm' is "
              "supported.");
DEFINE_string(kernel_variant, "fp8_blockscale_silu",
              "Unique fused MoE kernel variant. Supported values: "
              "fp8_blockscale_silu, fp8_petit_mxfp4_silu, "
              "fp8_petit_mxfp4_openai_bias, "
              "bf16_native_mxfp4_openai_bias, "
              "mxfp4_native_mxfp4_openai_bias.");
DEFINE_int32(tokens, 256, "Number of tokens.");
DEFINE_int32(dim, 4096, "Model dimension.");
DEFINE_int32(inter_dim, 1024, "Intermediate (FFN) dimension.");
DEFINE_int32(experts, 8, "Number of experts.");
DEFINE_int32(topk, 2, "Top-k routes per token.");
DEFINE_int32(warmup, 10, "Number of warmup iterations.");
DEFINE_int32(repeat, 100, "Number of benchmark iterations.");
DEFINE_int32(seed, 42, "Random seed.");
DEFINE_bool(persistent, false,
            "Enable persistent scheduling for fused MoE kernel launch.");
DEFINE_int32(num_persistent_tgs, 0,
             "Total persistent workgroups across all split-k tiles. "
             "0 disables persistent scheduling unless --persistent is set.");
DEFINE_int32(persistent_tgs_per_cu, 2,
             "When --persistent=true and --num_persistent_tgs=0, launch "
             "multiProcessorCount * persistent_tgs_per_cu workgroups.");

namespace moe = causalflow::petit::rocm::moe;
namespace moe_test = causalflow::petit::rocm::moe::test_utils;
using causalflow::CheckHIPStatus;

namespace {

constexpr unsigned kBlockN = 16;
constexpr unsigned kBlockK = 32;
constexpr unsigned kPackK = 16;
constexpr unsigned kKk = kBlockK / kPackK;
constexpr unsigned kSortedTokenPadding = 32;
constexpr unsigned kFp8ScaleGroup = 128;
constexpr unsigned kMxFp4ScaleGroup = 32;
constexpr unsigned kMxFp4ScaleMin = 1;
constexpr unsigned kMxFp4ScaleMax = 122;

enum class ActStorage {
    kFp8,
    kBf16,
    kNativeMxFp4,
};

enum class WeightStorage {
    kFp8BlockScale,
    kPetitMxFp4,
    kNativeMxFp4,
};

struct KernelVariant {
    const char *name;
    ActStorage act;
    WeightStorage weight;
    moe::FusedMoESolutionId solution_id;
    bool has_bias;
};

static constexpr moe::FusedMoESolutionId kFp8BlockScaleSiluSolutionId =
    moe::FusedMoESolutionId::Make(
        moe::FusedMoEDataType::kChannelScaleFp8,
        moe::FusedMoEDataType::kBlockScaleFp8, moe::FusedMoEDataType::kNone,
        moe::FusedMoEWeightOrdering::kPetitFp8,
        moe::FusedMoEMfmaShape::kMfmaFp816x16x32,
        moe::FusedMoEStages::kOneStage,
        moe::FusedMoEActivationFunction::kSiluDot,
        moe::FusedMoEStage1Buffering::kDoubleBuffer);

static constexpr moe::FusedMoESolutionId kFp8PetitMxFp4SiluSolutionId =
    moe::FusedMoESolutionId::Make(
        moe::FusedMoEDataType::kChannelScaleFp8, moe::FusedMoEDataType::kMxFp4,
        moe::FusedMoEDataType::kNone, moe::FusedMoEWeightOrdering::kPetitMxFp4,
        moe::FusedMoEMfmaShape::kMfmaFp816x16x32,
        moe::FusedMoEStages::kOneStage,
        moe::FusedMoEActivationFunction::kSiluDot,
        moe::FusedMoEStage1Buffering::kSingleBuffer);

static constexpr moe::FusedMoESolutionId kFp8PetitMxFp4OpenAIBiasSolutionId =
    moe::FusedMoESolutionId::Make(
        moe::FusedMoEDataType::kChannelScaleFp8, moe::FusedMoEDataType::kMxFp4,
        moe::FusedMoEDataType::kBf16, moe::FusedMoEWeightOrdering::kPetitMxFp4,
        moe::FusedMoEMfmaShape::kMfmaFp816x16x32,
        moe::FusedMoEStages::kOneStage,
        moe::FusedMoEActivationFunction::kOpenAISwiGLU,
        moe::FusedMoEStage1Buffering::kDoubleBuffer);

static constexpr moe::FusedMoESolutionId kBf16NativeMxFp4OpenAIBiasSolutionId =
    moe::FusedMoESolutionId::Make(
        moe::FusedMoEDataType::kBf16, moe::FusedMoEDataType::kMxFp4,
        moe::FusedMoEDataType::kBf16,
        moe::FusedMoEWeightOrdering::kNativeMxFp4,
        moe::FusedMoEMfmaShape::kMfmaBf16MxFp4,
        moe::FusedMoEStages::kOneStage,
        moe::FusedMoEActivationFunction::kOpenAISwiGLU,
        moe::FusedMoEStage1Buffering::kDoubleBuffer);

static constexpr moe::FusedMoESolutionId
    kMxFp4NativeMxFp4OpenAIBiasSolutionId = moe::FusedMoESolutionId::Make(
        moe::FusedMoEDataType::kMxFp4, moe::FusedMoEDataType::kMxFp4,
        moe::FusedMoEDataType::kBf16,
        moe::FusedMoEWeightOrdering::kNativeMxFp4,
        moe::FusedMoEMfmaShape::kMfmaScaleFp4MxFp4,
        moe::FusedMoEStages::kOneStage,
        moe::FusedMoEActivationFunction::kOpenAISwiGLU,
        moe::FusedMoEStage1Buffering::kDoubleBuffer);

static constexpr KernelVariant kKernelVariants[] = {
    {"fp8_blockscale_silu", ActStorage::kFp8,
     WeightStorage::kFp8BlockScale, kFp8BlockScaleSiluSolutionId, false},
    {"fp8_petit_mxfp4_silu", ActStorage::kFp8,
     WeightStorage::kPetitMxFp4, kFp8PetitMxFp4SiluSolutionId, false},
    {"fp8_petit_mxfp4_openai_bias", ActStorage::kFp8,
     WeightStorage::kPetitMxFp4, kFp8PetitMxFp4OpenAIBiasSolutionId, true},
    {"bf16_native_mxfp4_openai_bias", ActStorage::kBf16,
     WeightStorage::kNativeMxFp4, kBf16NativeMxFp4OpenAIBiasSolutionId, true},
    {"mxfp4_native_mxfp4_openai_bias", ActStorage::kNativeMxFp4,
     WeightStorage::kNativeMxFp4, kMxFp4NativeMxFp4OpenAIBiasSolutionId, true},
};

const KernelVariant *FindVariant(const std::string &name) {
    for (const auto &variant : kKernelVariants) {
        if (name == variant.name) {
            return &variant;
        }
    }
    return nullptr;
}

template <class T> constexpr T CeilingDiv(T x, T y) { return (x + y - 1) / y; }

static inline unsigned FloatAsBits(float x) {
    union {
        float f;
        unsigned u;
    } v{x};
    return v.u;
}

static inline unsigned short FloatAsBf16Bits(float x) {
    union {
        float f;
        unsigned u;
    } in{x};
    const unsigned lsb = (in.u >> 16) & 1u;
    in.u += 0x7fffu + lsb;
    return static_cast<unsigned short>(in.u >> 16);
}

template <class T> T *CopyToDevice(const std::vector<T> &src) {
    if (src.empty()) {
        return nullptr;
    }
    T *dst = nullptr;
    CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&dst),
                             src.size() * sizeof(T)));
    CheckHIPStatus(hipMemcpy(dst, src.data(), src.size() * sizeof(T),
                             hipMemcpyHostToDevice));
    return dst;
}

template <class T> void FreeDevice(T *&ptr) {
    if (ptr != nullptr) {
        CheckHIPStatus(hipFree(ptr));
        ptr = nullptr;
    }
}

static void FillRandomU32Parallel(std::vector<unsigned> *data, uint64_t seed,
                                  uint64_t stream, bool mask_fp4_zero) {
    unsigned *const base = data->data();
    std::for_each(std::execution::par, data->begin(), data->end(),
                  [&](unsigned &value) {
                      const size_t idx = static_cast<size_t>(&value - base);
                      value = static_cast<unsigned>(
                          moe_test::MixU64(seed ^ (stream << 32) ^ idx));
                      if (mask_fp4_zero) {
                          value =
                              moe_test::MaskNegativeZeroOnNativeFp4Format(
                                  value);
                      }
                  });
}

template <class T, class Sampler>
static void FillWeightsParallel(std::vector<T> *data, const Sampler &sampler,
                                uint64_t seed, uint64_t stream) {
    T *const base = data->data();
    std::for_each(
        std::execution::par, data->begin(), data->end(), [&](T &value) {
            const size_t idx = static_cast<size_t>(&value - base);
            moe_test::FixedU32Rng rng{static_cast<uint32_t>(
                moe_test::MixU64(seed ^ (stream << 32) ^ idx))};
            value = sampler(rng);
        });
}

template <class Generator4>
static void FillFp8InputsParallel4(std::vector<unsigned char> *data,
                                   const Generator4 &generator4) {
    if ((data->size() % 4) != 0) {
        throw std::runtime_error(
            "FillFp8InputsParallel4 expects size % 4 == 0");
    }

    auto *const packed_ptr = reinterpret_cast<unsigned *>(data->data());
    std::span<unsigned> packed(packed_ptr, data->size() / 4);
    unsigned *const base = packed.data();
    std::for_each(std::execution::par, packed.begin(), packed.end(),
                  [&](unsigned &slot) {
                      const size_t group = static_cast<size_t>(&slot - base);
                      slot = generator4(group * 4);
                  });
}

struct HostInputs {
    std::vector<unsigned char> act_fp8;
    std::vector<unsigned short> act_bf16;
    std::vector<unsigned> act_mxfp4;
    std::vector<unsigned char> scale_act_mxfp4;

    std::vector<unsigned char> w13_fp8;
    std::vector<unsigned char> w2_fp8;
    std::vector<unsigned> w13_mxfp4;
    std::vector<unsigned> w2_mxfp4;

    std::vector<unsigned> sorted_token_ids;
    std::vector<float> sorted_weights;
    std::vector<unsigned> sorted_expert_ids;
    std::vector<unsigned> num_valid_ids;

    std::vector<float> scale_act_fp8;
    std::vector<float> scale_fc1_fp8;
    std::vector<float> scale_fc2_fp8;
    std::vector<unsigned> fc2_scale_fp8_bits;
    std::vector<unsigned char> scale_fc1_mxfp4;
    std::vector<unsigned char> scale_fc2_mxfp4;

    std::vector<unsigned short> w13_bias;
    std::vector<unsigned short> w2_bias;

    unsigned num_valid_token_ids = 0;
    unsigned num_valid_m_blocks = 0;
};

struct DeviceBuffers {
    unsigned char *act_fp8 = nullptr;
    unsigned short *act_bf16 = nullptr;
    unsigned *act_mxfp4_native = nullptr;
    unsigned *scale_act_mxfp4_native = nullptr;

    unsigned char *w1_fp8 = nullptr;
    unsigned char *w2_fp8 = nullptr;
    unsigned *w1_mxfp4 = nullptr;
    unsigned *w2_mxfp4 = nullptr;

    unsigned short *out_bf16 = nullptr;

    unsigned *sorted_token_ids = nullptr;
    float *sorted_weights = nullptr;
    unsigned *sorted_expert_ids = nullptr;
    unsigned *num_valid_ids = nullptr;

    float *input_scale_t = nullptr;
    float *scale_fc1_fp8 = nullptr;
    unsigned *fc2_scale_fp8_bits = nullptr;
    unsigned *scale_fc1_mxfp4 = nullptr;
    unsigned *scale_fc2_mxfp4 = nullptr;

    __hip_bfloat16 *w13_bias = nullptr;
    __hip_bfloat16 *w2_bias = nullptr;

    void AllocateAndPrepare(const HostInputs &in, const KernelVariant &variant,
                            unsigned tokens, unsigned dim,
                            unsigned inter_dim, unsigned experts);
    void Free();
};

struct BenchmarkResult {
    double device_ms = 0.0;
    double tflops = 0.0;
    double bandwidth_gbps = 0.0;
};

__global__ void ShuffleWeightLayout16x32Kernel(const unsigned char *src,
                                               unsigned char *dst,
                                               unsigned experts, unsigned rows,
                                               unsigned cols) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t per_expert = static_cast<size_t>(rows) * cols;
    const size_t total = static_cast<size_t>(experts) * per_expert;
    if (idx >= total) {
        return;
    }

    const unsigned cb = cols / kBlockK;
    const size_t expert = idx / per_expert;
    const size_t local = idx - expert * per_expert;
    const unsigned row = static_cast<unsigned>(local / cols);
    const unsigned col = static_cast<unsigned>(local % cols);

    const unsigned rbi = row / kBlockN;
    const unsigned bni = row % kBlockN;
    const unsigned cbi = col / kBlockK;
    const unsigned inner = col % kBlockK;
    const unsigned kki = inner / kPackK;
    const unsigned kpi = inner % kPackK;

    const size_t dst_local =
        (((((static_cast<size_t>(rbi) * cb + cbi) * kKk + kki) * kBlockN +
           bni) *
          kPackK) +
         kpi);
    dst[expert * per_expert + dst_local] = src[idx];
}

__global__ void Transpose2DKernel(const float *src, float *dst, unsigned rows,
                                  unsigned cols) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * cols;
    if (idx >= total) {
        return;
    }
    const unsigned r = static_cast<unsigned>(idx / cols);
    const unsigned c = static_cast<unsigned>(idx % cols);
    dst[static_cast<size_t>(c) * rows + r] = src[idx];
}

__global__ void PackAiterSortedMxFp4ActivationScalesKernel(
    const unsigned char *__restrict__ src,
    const unsigned *__restrict__ sorted_token_ids,
    unsigned char *__restrict__ dst, unsigned tokens, unsigned dim) {
    constexpr unsigned kRoutesPerGroup = kSortedTokenPadding;
    constexpr unsigned kScaleColsPerK256 = 8;
    constexpr unsigned kRowsPerHalf = 16;
    constexpr unsigned kColsPerHalf = 4;
    static_assert(kRoutesPerGroup == 32, "");

    const unsigned route_group = blockIdx.x;
    const unsigned k256 = blockIdx.y;

    __shared__ unsigned route_tokens[kRoutesPerGroup];

    using namespace causalflow::tal;
    const auto route_layout =
        make_layout(make_shape(C<2>{}, C<kRowsPerHalf>{}),
                    make_stride(C<kRowsPerHalf>{}, _1{}));
    const auto scale_col_layout =
        make_layout(make_shape(C<2>{}, C<kColsPerHalf>{}),
                    make_stride(C<kColsPerHalf>{}, _1{}));
    const auto sorted_scale_layout = make_layout(
        make_shape(C<kColsPerHalf>{}, C<kRowsPerHalf>{},
                   make_shape(C<2>{}, C<2>{})),
        make_stride(C<kRowsPerHalf * 4>{}, C<4>{},
                    make_stride(C<2>{}, _1{})));

    const unsigned n4 = threadIdx.x;
    const unsigned m16 = threadIdx.y;
    const unsigned byte = threadIdx.z;
    const unsigned group2 = byte >> 1;
    const unsigned group0 = byte & 1u;
    const unsigned route = route_layout(make_coord(group0, m16));
    if (n4 == 0 && group2 == 0) {
        route_tokens[route] =
            sorted_token_ids[route_group * kRoutesPerGroup + route] &
            0x00ffffffu;
    }
    __syncthreads();

    const unsigned token = route_tokens[route];
    const unsigned local_scale_col =
        scale_col_layout(make_coord(group2, n4));
    unsigned char scale = 127;
    if (token < tokens) {
        scale = src[static_cast<size_t>(token) * (dim / kMxFp4ScaleGroup) +
                    k256 * kScaleColsPerK256 + local_scale_col];
    }
    const unsigned dst_offset = sorted_scale_layout(
        make_coord(n4, m16, make_coord(group2, group0)));
    dst[static_cast<size_t>(route_group) * dim +
        k256 * kRoutesPerGroup * kScaleColsPerK256 + dst_offset] = scale;
}

static void BuildSortedRoutingFromTopK(const std::vector<unsigned> &topk_ids,
                                       const std::vector<float> &topk_weights,
                                       unsigned tokens, unsigned experts,
                                       unsigned topk, HostInputs *out) {
    const unsigned routes = tokens * topk;
    const unsigned max_num_tokens_padded =
        routes + experts * kSortedTokenPadding - topk;
    const unsigned max_num_m_blocks =
        CeilingDiv<unsigned>(max_num_tokens_padded, kSortedTokenPadding);
    const unsigned init_val = (topk << 24) | tokens;

    out->sorted_token_ids.assign(max_num_tokens_padded, init_val);
    out->sorted_weights.assign(max_num_tokens_padded, 0.0f);
    out->sorted_expert_ids.assign(max_num_m_blocks,
                                  std::numeric_limits<unsigned>::max());
    out->num_valid_ids.assign(2, 0u);

    unsigned sorted_ids_begin = 0;
    unsigned sorted_expert_ids_begin = 0;
    for (unsigned expert = 0; expert < experts; ++expert) {
        unsigned tokens_num = 0;
        for (unsigned token = 0; token < tokens; ++token) {
            for (unsigned slot = 0; slot < topk; ++slot) {
                const size_t idx = static_cast<size_t>(token) * topk + slot;
                if (topk_ids[idx] != expert) {
                    continue;
                }
                out->sorted_token_ids[sorted_ids_begin + tokens_num] =
                    (slot << 24) | token;
                out->sorted_weights[sorted_ids_begin + tokens_num] =
                    topk_weights[idx];
                ++tokens_num;
            }
        }
        if (tokens_num == 0) {
            continue;
        }
        const unsigned sorted_expert_ids_num =
            CeilingDiv<unsigned>(tokens_num, kSortedTokenPadding);
        const unsigned tokens_num_pad =
            sorted_expert_ids_num * kSortedTokenPadding;
        for (unsigned i = 0; i < sorted_expert_ids_num; ++i) {
            out->sorted_expert_ids[sorted_expert_ids_begin + i] = expert;
        }
        sorted_ids_begin += tokens_num_pad;
        sorted_expert_ids_begin += sorted_expert_ids_num;
    }

    out->num_valid_ids[0] = sorted_ids_begin;
    out->num_valid_ids[1] = tokens;
    out->num_valid_token_ids = sorted_ids_begin;
    out->num_valid_m_blocks = sorted_expert_ids_begin;
}

static HostInputs MakeInputs(const KernelVariant &variant, unsigned tokens,
                             unsigned dim, unsigned inter_dim,
                             unsigned experts, unsigned topk, unsigned seed) {
    HostInputs in;
    const unsigned fp8_n_blocks = dim / kFp8ScaleGroup;
    const unsigned fp8_k_blocks = inter_dim / kFp8ScaleGroup;
    const unsigned w1_rows = 2 * inter_dim;

    std::mt19937 gen(seed);
    std::normal_distribution<float> input_dist(0.0f, 0.5f);
    std::normal_distribution<float> spike_dist(0.0f, 8.0f);
    std::normal_distribution<float> scale_dist(1.0e-4f, 2.0e-5f);
    std::normal_distribution<float> bias_dist(0.0f, 0.02f);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> rw_dist(0.0f, 1.0f);
    std::uniform_int_distribution<unsigned> expert_dist(0, experts - 1);
    std::uniform_int_distribution<unsigned> mx_scale_dist(kMxFp4ScaleMin,
                                                          kMxFp4ScaleMax);
    causalflow::petit::tests::fp8_sampler::FP8E4M3DiscreteSampler
        weight_sampler(40.0);

    if (variant.act == ActStorage::kFp8) {
        in.act_fp8.resize(static_cast<size_t>(tokens) * dim);
        in.scale_act_fp8.resize(static_cast<size_t>(tokens) * fp8_n_blocks);
        FillFp8InputsParallel4(&in.act_fp8, [&](size_t idx) {
            float4 values;
            for (int i = 0; i < 4; ++i) {
                std::mt19937 lane_gen(static_cast<uint32_t>(
                    moe_test::MixU64(static_cast<uint64_t>(seed) ^
                                     (0x17bULL << 32) ^ (idx + i))));
                float x = input_dist(lane_gen);
                if (prob_dist(lane_gen) < 0.002f) {
                    x += spike_dist(lane_gen);
                }
                reinterpret_cast<float *>(&values)[i] = x;
            }
            return __hip_fp8x4_e4m3_fnuz(values).__x;
        });
        for (float &s : in.scale_act_fp8) {
            s = std::max(1.0e-8f, scale_dist(gen));
        }
    } else if (variant.act == ActStorage::kBf16) {
        in.act_bf16.resize(static_cast<size_t>(tokens) * dim);
        for (size_t i = 0; i < in.act_bf16.size(); ++i) {
            float x = input_dist(gen);
            if (prob_dist(gen) < 0.002f) {
                x += spike_dist(gen);
            }
            in.act_bf16[i] = FloatAsBf16Bits(x);
        }
    } else {
        in.act_mxfp4.resize(static_cast<size_t>(tokens) * dim / 8);
        in.scale_act_mxfp4.resize(static_cast<size_t>(tokens) * dim /
                                  kMxFp4ScaleGroup);
        FillRandomU32Parallel(&in.act_mxfp4, seed, 0x511u, true);
        std::generate(in.scale_act_mxfp4.begin(), in.scale_act_mxfp4.end(),
                      [&]() {
                          return static_cast<unsigned char>(
                              mx_scale_dist(gen));
                      });
    }

    if (variant.weight == WeightStorage::kFp8BlockScale) {
        in.w13_fp8.resize(static_cast<size_t>(experts) * w1_rows * dim);
        in.w2_fp8.resize(static_cast<size_t>(experts) * dim * inter_dim);
        in.scale_fc1_fp8.resize(static_cast<size_t>(experts) *
                                (w1_rows / kFp8ScaleGroup) * fp8_n_blocks);
        in.scale_fc2_fp8.resize(static_cast<size_t>(experts) * fp8_n_blocks *
                                fp8_k_blocks);
        in.fc2_scale_fp8_bits.resize(static_cast<size_t>(experts) *
                                     fp8_k_blocks * fp8_n_blocks);
        FillWeightsParallel(&in.w13_fp8, weight_sampler, seed, 0x913u);
        FillWeightsParallel(&in.w2_fp8, weight_sampler, seed, 0x2d5u);
        for (float &s : in.scale_fc1_fp8) {
            s = std::max(1.0e-8f, scale_dist(gen));
        }
        for (float &s : in.scale_fc2_fp8) {
            s = std::max(1.0e-8f, scale_dist(gen));
        }
        for (unsigned e = 0; e < experts; ++e) {
            for (unsigned nb = 0; nb < fp8_n_blocks; ++nb) {
                for (unsigned kb = 0; kb < fp8_k_blocks; ++kb) {
                    const size_t idx =
                        (static_cast<size_t>(e) * fp8_n_blocks + nb) *
                            fp8_k_blocks +
                        kb;
                    in.fc2_scale_fp8_bits[idx] =
                        FloatAsBits(in.scale_fc2_fp8[idx]);
                }
            }
        }
    } else {
        in.w13_mxfp4.resize(static_cast<size_t>(experts) * w1_rows * dim / 8);
        in.w2_mxfp4.resize(static_cast<size_t>(experts) * dim * inter_dim / 8);
        in.scale_fc1_mxfp4.resize(static_cast<size_t>(experts) * w1_rows *
                                  dim / kMxFp4ScaleGroup);
        in.scale_fc2_mxfp4.resize(static_cast<size_t>(experts) * dim *
                                  inter_dim / kMxFp4ScaleGroup);
        FillRandomU32Parallel(&in.w13_mxfp4, seed, 0x913u, true);
        FillRandomU32Parallel(&in.w2_mxfp4, seed, 0x2d5u, true);
        std::generate(in.scale_fc1_mxfp4.begin(), in.scale_fc1_mxfp4.end(),
                      [&]() {
                          return static_cast<unsigned char>(
                              mx_scale_dist(gen));
                      });
        std::generate(in.scale_fc2_mxfp4.begin(), in.scale_fc2_mxfp4.end(),
                      [&]() {
                          return static_cast<unsigned char>(
                              mx_scale_dist(gen));
                      });
    }

    std::vector<unsigned> topk_ids(static_cast<size_t>(tokens) * topk);
    std::vector<float> topk_weights(static_cast<size_t>(tokens) * topk);
    for (unsigned token = 0; token < tokens; ++token) {
        float sum = 0.0f;
        for (unsigned slot = 0; slot < topk; ++slot) {
            const size_t idx = static_cast<size_t>(token) * topk + slot;
            topk_ids[idx] = expert_dist(gen);
            topk_weights[idx] = rw_dist(gen);
            sum += topk_weights[idx];
        }
        sum = std::max(sum, 1.0e-12f);
        for (unsigned slot = 0; slot < topk; ++slot) {
            const size_t idx = static_cast<size_t>(token) * topk + slot;
            topk_weights[idx] /= sum;
        }
    }
    BuildSortedRoutingFromTopK(topk_ids, topk_weights, tokens, experts, topk,
                               &in);

    if (variant.has_bias) {
        in.w13_bias.resize(static_cast<size_t>(experts) * 2 * inter_dim);
        in.w2_bias.resize(static_cast<size_t>(experts) * dim);
        for (auto &v : in.w13_bias) {
            v = FloatAsBf16Bits(bias_dist(gen));
        }
        for (auto &v : in.w2_bias) {
            v = FloatAsBf16Bits(bias_dist(gen));
        }
    }

    return in;
}

void DeviceBuffers::AllocateAndPrepare(const HostInputs &in,
                                       const KernelVariant &variant,
                                       unsigned tokens, unsigned dim,
                                       unsigned inter_dim, unsigned experts) {
    const size_t out_bytes =
        static_cast<size_t>(tokens) * dim * sizeof(unsigned short);
    const unsigned fp8_n_blocks = dim / kFp8ScaleGroup;
    const unsigned w1_rows = 2 * inter_dim;
    constexpr unsigned kThreads = 256;
    const dim3 block(kThreads);
    const auto launch_1d = [&](size_t n) {
        return dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads));
    };

    CheckHIPStatus(hipMalloc(&out_bf16, out_bytes));
    sorted_token_ids = CopyToDevice(in.sorted_token_ids);
    sorted_weights = CopyToDevice(in.sorted_weights);
    sorted_expert_ids = CopyToDevice(in.sorted_expert_ids);
    num_valid_ids = CopyToDevice(in.num_valid_ids);

    if (variant.act == ActStorage::kFp8) {
        act_fp8 = CopyToDevice(in.act_fp8);
        float *scale_act_tmp = CopyToDevice(in.scale_act_fp8);
        CheckHIPStatus(hipMalloc(&input_scale_t,
                                 in.scale_act_fp8.size() * sizeof(float)));
        Transpose2DKernel<<<launch_1d(in.scale_act_fp8.size()), block>>>(
            scale_act_tmp, input_scale_t, tokens, fp8_n_blocks);
        CheckHIPStatus(hipGetLastError());
        CheckHIPStatus(hipFree(scale_act_tmp));
    } else if (variant.act == ActStorage::kBf16) {
        act_bf16 = CopyToDevice(in.act_bf16);
    } else {
        const size_t native_scale_words =
            static_cast<size_t>(in.sorted_expert_ids.size()) * (dim / 256) *
            64;
        act_mxfp4_native = CopyToDevice(in.act_mxfp4);
        CheckHIPStatus(hipMalloc(&scale_act_mxfp4_native,
                                 native_scale_words * sizeof(unsigned)));
        unsigned char *scale_tmp = CopyToDevice(in.scale_act_mxfp4);
        PackAiterSortedMxFp4ActivationScalesKernel<<<
            dim3(static_cast<unsigned>(in.sorted_expert_ids.size()), dim / 256),
            dim3(4, 16, 4)>>>(
            scale_tmp, sorted_token_ids,
            reinterpret_cast<unsigned char *>(scale_act_mxfp4_native), tokens,
            dim);
        CheckHIPStatus(hipGetLastError());
        CheckHIPStatus(hipFree(scale_tmp));
    }

    if (variant.weight == WeightStorage::kFp8BlockScale) {
        unsigned char *w13_fp8_tmp = CopyToDevice(in.w13_fp8);
        unsigned char *w2_fp8_tmp = CopyToDevice(in.w2_fp8);
        CheckHIPStatus(hipMalloc(&w1_fp8, in.w13_fp8.size()));
        CheckHIPStatus(hipMalloc(&w2_fp8, in.w2_fp8.size()));
        scale_fc1_fp8 = CopyToDevice(in.scale_fc1_fp8);
        fc2_scale_fp8_bits = CopyToDevice(in.fc2_scale_fp8_bits);

        ShuffleWeightLayout16x32Kernel<<<launch_1d(in.w13_fp8.size()), block>>>(
            w13_fp8_tmp, w1_fp8, experts, w1_rows, dim);
        CheckHIPStatus(hipGetLastError());
        ShuffleWeightLayout16x32Kernel<<<launch_1d(in.w2_fp8.size()), block>>>(
            w2_fp8_tmp, w2_fp8, experts, dim, inter_dim);
        CheckHIPStatus(hipGetLastError());
        CheckHIPStatus(hipFree(w13_fp8_tmp));
        CheckHIPStatus(hipFree(w2_fp8_tmp));
    } else {
        unsigned *w13_tmp = CopyToDevice(in.w13_mxfp4);
        unsigned *w2_tmp = CopyToDevice(in.w2_mxfp4);
        unsigned char *scale_fc1_tmp = CopyToDevice(in.scale_fc1_mxfp4);
        unsigned char *scale_fc2_tmp = CopyToDevice(in.scale_fc2_mxfp4);
        CheckHIPStatus(hipMalloc(&w1_mxfp4,
                                 in.w13_mxfp4.size() * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&w2_mxfp4,
                                 in.w2_mxfp4.size() * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&scale_fc1_mxfp4,
                                 in.scale_fc1_mxfp4.size()));
        CheckHIPStatus(hipMalloc(&scale_fc2_mxfp4,
                                 in.scale_fc2_mxfp4.size()));

        const unsigned w13_rows = experts * w1_rows;
        const unsigned w2_rows = experts * dim;
        if (variant.weight == WeightStorage::kPetitMxFp4) {
            CheckHIPStatus(moe_test::RepackPetitMxFp4Weights(
                w1_mxfp4, w13_tmp, w13_rows, dim));
            CheckHIPStatus(moe_test::RepackPetitMxFp4Weights(
                w2_mxfp4, w2_tmp, w2_rows, inter_dim));
            CheckHIPStatus(moe_test::RepackPetitMxFp4Scales(
                scale_fc1_mxfp4, reinterpret_cast<const unsigned *>(scale_fc1_tmp),
                w13_rows, dim / kMxFp4ScaleGroup));
            CheckHIPStatus(moe_test::RepackPetitMxFp4Scales(
                scale_fc2_mxfp4, reinterpret_cast<const unsigned *>(scale_fc2_tmp),
                w2_rows, inter_dim / kMxFp4ScaleGroup));
        } else {
            CheckHIPStatus(moe_test::RepackNativeMxFp4Weights(
                w1_mxfp4, w13_tmp, w13_rows, dim));
            CheckHIPStatus(moe_test::RepackNativeMxFp4Weights(
                w2_mxfp4, w2_tmp, w2_rows, inter_dim));
            CheckHIPStatus(moe_test::RepackNativeMxFp4Scales(
                scale_fc1_mxfp4, reinterpret_cast<const unsigned *>(scale_fc1_tmp),
                w13_rows, dim / kMxFp4ScaleGroup));
            CheckHIPStatus(moe_test::RepackNativeMxFp4Scales(
                scale_fc2_mxfp4, reinterpret_cast<const unsigned *>(scale_fc2_tmp),
                w2_rows, inter_dim / kMxFp4ScaleGroup));
        }
        CheckHIPStatus(hipFree(w13_tmp));
        CheckHIPStatus(hipFree(w2_tmp));
        CheckHIPStatus(hipFree(scale_fc1_tmp));
        CheckHIPStatus(hipFree(scale_fc2_tmp));
    }

    if (variant.has_bias) {
        auto *w13_logical = CopyToDevice(in.w13_bias);
        auto *w2_logical = CopyToDevice(in.w2_bias);
        const unsigned w13_packed_cols = CeilingDiv<unsigned>(inter_dim, 512) *
                                         512;
        const unsigned w2_packed_cols = CeilingDiv<unsigned>(dim, 512) * 512;
        CheckHIPStatus(hipMalloc(&w13_bias,
                                 static_cast<size_t>(experts) * 2 *
                                     w13_packed_cols *
                                     sizeof(__hip_bfloat16)));
        CheckHIPStatus(hipMalloc(&w2_bias,
                                 static_cast<size_t>(experts) *
                                     w2_packed_cols *
                                     sizeof(__hip_bfloat16)));
        CheckHIPStatus(moe_test::RepackBf16BiasDppLayout(
            w13_bias, reinterpret_cast<const __hip_bfloat16 *>(w13_logical),
            experts * 2, inter_dim));
        CheckHIPStatus(moe_test::RepackBf16BiasDppLayout(
            w2_bias, reinterpret_cast<const __hip_bfloat16 *>(w2_logical),
            experts, dim));
        CheckHIPStatus(hipFree(w13_logical));
        CheckHIPStatus(hipFree(w2_logical));
    }

    CheckHIPStatus(hipDeviceSynchronize());
}

void DeviceBuffers::Free() {
    FreeDevice(act_fp8);
    FreeDevice(act_bf16);
    FreeDevice(act_mxfp4_native);
    FreeDevice(scale_act_mxfp4_native);
    FreeDevice(w1_fp8);
    FreeDevice(w2_fp8);
    FreeDevice(w1_mxfp4);
    FreeDevice(w2_mxfp4);
    FreeDevice(out_bf16);
    FreeDevice(sorted_token_ids);
    FreeDevice(sorted_weights);
    FreeDevice(sorted_expert_ids);
    FreeDevice(num_valid_ids);
    FreeDevice(input_scale_t);
    FreeDevice(scale_fc1_fp8);
    FreeDevice(fc2_scale_fp8_bits);
    FreeDevice(scale_fc1_mxfp4);
    FreeDevice(scale_fc2_mxfp4);
    FreeDevice(w13_bias);
    FreeDevice(w2_bias);
}

static double KernelBytes(const HostInputs &in, const KernelVariant &variant,
                          unsigned tokens, unsigned dim, unsigned inter_dim) {
    double bytes = static_cast<double>(in.sorted_token_ids.size() *
                                       sizeof(unsigned)) +
                   static_cast<double>(in.sorted_weights.size() *
                                       sizeof(float)) +
                   static_cast<double>(in.sorted_expert_ids.size() *
                                       sizeof(unsigned)) +
                   static_cast<double>(tokens) * dim *
                       sizeof(unsigned short);
    if (variant.act == ActStorage::kFp8) {
        bytes += static_cast<double>(in.act_fp8.size());
        bytes += static_cast<double>(in.scale_act_fp8.size() * sizeof(float));
    } else if (variant.act == ActStorage::kBf16) {
        bytes += static_cast<double>(in.act_bf16.size() *
                                     sizeof(unsigned short));
    } else {
        bytes += static_cast<double>(in.act_mxfp4.size() * sizeof(unsigned));
        bytes += static_cast<double>(in.sorted_expert_ids.size()) *
                 (dim / 256) * 64 * sizeof(unsigned);
    }

    if (variant.weight == WeightStorage::kFp8BlockScale) {
        bytes += static_cast<double>(in.w13_fp8.size());
        bytes += static_cast<double>(in.w2_fp8.size());
        bytes += static_cast<double>(in.scale_fc1_fp8.size() * sizeof(float));
        bytes += static_cast<double>(in.fc2_scale_fp8_bits.size() *
                                     sizeof(unsigned));
    } else {
        bytes += static_cast<double>(in.w13_mxfp4.size() * sizeof(unsigned));
        bytes += static_cast<double>(in.w2_mxfp4.size() * sizeof(unsigned));
        bytes += static_cast<double>(in.scale_fc1_mxfp4.size());
        bytes += static_cast<double>(in.scale_fc2_mxfp4.size());
    }

    if (variant.has_bias) {
        bytes += static_cast<double>(in.w13_bias.size() *
                                     sizeof(unsigned short));
        bytes += static_cast<double>(in.w2_bias.size() *
                                     sizeof(unsigned short));
    }
    (void)inter_dim;
    return bytes;
}

static bool RunBenchmark(const HostInputs &in, const DeviceBuffers &dev,
                         const KernelVariant &variant, unsigned tokens,
                         unsigned dim, unsigned inter_dim, unsigned experts,
                         unsigned topk, unsigned num_persistent_tgs,
                         BenchmarkResult *result) {
    const size_t out_bytes =
        static_cast<size_t>(tokens) * dim * sizeof(unsigned short);
    const unsigned *act = nullptr;
    const unsigned *scales_act = nullptr;
    if (variant.act == ActStorage::kFp8) {
        act = reinterpret_cast<const unsigned *>(dev.act_fp8);
        scales_act = reinterpret_cast<const unsigned *>(dev.input_scale_t);
    } else if (variant.act == ActStorage::kBf16) {
        act = reinterpret_cast<const unsigned *>(dev.act_bf16);
    } else {
        act = dev.act_mxfp4_native;
        scales_act = dev.scale_act_mxfp4_native;
    }

    const unsigned *w13 = nullptr;
    const unsigned *w2 = nullptr;
    const unsigned *scales_w13 = nullptr;
    const unsigned *scales_w2 = nullptr;
    if (variant.weight == WeightStorage::kFp8BlockScale) {
        w13 = reinterpret_cast<const unsigned *>(dev.w1_fp8);
        w2 = reinterpret_cast<const unsigned *>(dev.w2_fp8);
        scales_w13 = reinterpret_cast<const unsigned *>(dev.scale_fc1_fp8);
        scales_w2 = dev.fc2_scale_fp8_bits;
    } else {
        w13 = dev.w1_mxfp4;
        w2 = dev.w2_mxfp4;
        scales_w13 = dev.scale_fc1_mxfp4;
        scales_w2 = dev.scale_fc2_mxfp4;
    }

    moe::FusedMoE1StageParams params{
        reinterpret_cast<unsigned *>(dev.out_bf16),
        act,
        w13,
        w2,
        dev.sorted_token_ids,
        reinterpret_cast<const unsigned *>(dev.sorted_weights),
        dev.sorted_expert_ids,
        dev.num_valid_ids,
        topk,
        scales_act,
        scales_w13,
        scales_w2,
        static_cast<unsigned>(in.sorted_expert_ids.size()),
        tokens,
        dim,
        inter_dim,
        experts,
        nullptr,
        num_persistent_tgs,
        dev.w13_bias,
        dev.w2_bias,
    };

    CheckHIPStatus(hipMemset(dev.out_bf16, 0, out_bytes));
    const int err = moe::FusedMoEMatmul1Stage(params, variant.solution_id.Repr());
    if (err != 0) {
        return false;
    }

    for (int i = 0; i < FLAGS_warmup; ++i) {
        if (moe::FusedMoEMatmul1Stage(params, variant.solution_id.Repr()) !=
            0) {
            return false;
        }
    }
    CheckHIPStatus(hipDeviceSynchronize());
    CheckHIPStatus(hipMemset(dev.out_bf16, 0, out_bytes));

    hipEvent_t ev_start;
    hipEvent_t ev_stop;
    CheckHIPStatus(hipEventCreate(&ev_start));
    CheckHIPStatus(hipEventCreate(&ev_stop));

    CheckHIPStatus(hipEventRecord(ev_start, 0));
    for (int i = 0; i < FLAGS_repeat; ++i) {
        moe::FusedMoEMatmul1Stage(params, variant.solution_id.Repr());
    }
    CheckHIPStatus(hipEventRecord(ev_stop, 0));
    CheckHIPStatus(hipEventSynchronize(ev_stop));

    float device_total_ms = 0.0f;
    CheckHIPStatus(hipEventElapsedTime(&device_total_ms, ev_start, ev_stop));
    CheckHIPStatus(hipEventDestroy(ev_start));
    CheckHIPStatus(hipEventDestroy(ev_stop));

    const double device_ms = static_cast<double>(device_total_ms) /
                             static_cast<double>(FLAGS_repeat);
    const double device_s = device_ms * 1e-3;
    const double routes = static_cast<double>(tokens) * topk;
    const double flops = 6.0 * routes * static_cast<double>(dim) *
                         static_cast<double>(inter_dim);
    const double tflops = flops / device_s / 1e12;
    const double bandwidth_gbps =
        KernelBytes(in, variant, tokens, dim, inter_dim) / device_s / 1e9;

    result->device_ms = device_ms;
    result->tflops = tflops;
    result->bandwidth_gbps = bandwidth_gbps;
    return true;
}

} // namespace

static bool ValidateBackend(const char *, const std::string &value) {
    return value == "fused_moe_rocm";
}

static bool ValidateKernelVariant(const char *, const std::string &value) {
    return FindVariant(value) != nullptr;
}

static void RegisterValidators() {
    gflags::RegisterFlagValidator(&FLAGS_backend, &ValidateBackend);
    gflags::RegisterFlagValidator(&FLAGS_kernel_variant,
                                   &ValidateKernelVariant);
}

int main(int argc, char **argv) {
    RegisterValidators();
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const KernelVariant *variant = FindVariant(FLAGS_kernel_variant);
    if (variant == nullptr) {
        std::cerr << "unsupported kernel_variant: " << FLAGS_kernel_variant
                  << "\n";
        return 1;
    }

    if (FLAGS_tokens <= 0 || FLAGS_dim <= 0 || FLAGS_inter_dim <= 0 ||
        FLAGS_experts <= 0 || FLAGS_topk <= 0) {
        std::cerr << "tokens, dim, inter_dim, experts, and topk must all be "
                     "positive.\n";
        return 1;
    }
    if (FLAGS_warmup < 0 || FLAGS_repeat <= 0) {
        std::cerr
            << "warmup must be non-negative and repeat must be positive.\n";
        return 1;
    }
    if (FLAGS_num_persistent_tgs < 0 || FLAGS_persistent_tgs_per_cu <= 0) {
        std::cerr << "num_persistent_tgs must be >= 0 and "
                     "persistent_tgs_per_cu must be > 0.\n";
        return 1;
    }
    if (FLAGS_topk > FLAGS_experts) {
        std::cerr << "topk must be <= experts.\n";
        return 1;
    }
    if (FLAGS_topk > 255) {
        std::cerr << "topk must be <= 255 because route slot id is packed in 8 "
                     "bits.\n";
        return 1;
    }
    if (FLAGS_tokens >= (1 << 24)) {
        std::cerr << "tokens must be < 2^24 because token id is packed in 24 "
                     "bits.\n";
        return 1;
    }
    if ((FLAGS_dim % 256) != 0 || (FLAGS_inter_dim % 256) != 0) {
        std::cerr << "dim and inter_dim must both be divisible by 256.\n";
        return 1;
    }

    const unsigned tokens = static_cast<unsigned>(FLAGS_tokens);
    const unsigned dim = static_cast<unsigned>(FLAGS_dim);
    const unsigned inter_dim = static_cast<unsigned>(FLAGS_inter_dim);
    const unsigned experts = static_cast<unsigned>(FLAGS_experts);
    const unsigned topk = static_cast<unsigned>(FLAGS_topk);
    const unsigned seed = static_cast<unsigned>(FLAGS_seed);
    unsigned num_persistent_tgs =
        static_cast<unsigned>(FLAGS_num_persistent_tgs);
    if (num_persistent_tgs == 0 && FLAGS_persistent) {
        int device = 0;
        CheckHIPStatus(hipGetDevice(&device));
        hipDeviceProp_t props;
        CheckHIPStatus(hipGetDeviceProperties(&props, device));
        num_persistent_tgs = static_cast<unsigned>(props.multiProcessorCount *
                                                   FLAGS_persistent_tgs_per_cu);
    }

    HostInputs in =
        MakeInputs(*variant, tokens, dim, inter_dim, experts, topk, seed);
    DeviceBuffers dev;
    dev.AllocateAndPrepare(in, *variant, tokens, dim, inter_dim, experts);

    BenchmarkResult result;
    const bool ok = RunBenchmark(in, dev, *variant, tokens, dim, inter_dim,
                                 experts, topk, num_persistent_tgs, &result);
    dev.Free();

    if (!ok) {
        std::cerr << "Fused MoE benchmark failed for kernel_variant="
                  << variant->name << ".\n";
        return 1;
    }

    fmt::print(
        "backend={} kernel_variant={} solution_id={} kernel=32x256x256 "
        "tokens={} dim={} inter_dim={} experts={} "
        "topk={} valid_ids={} valid_m_blocks={} persistent_tgs={} launch={} "
        "device_ms={:.4f} tflops={:.3f} gbps={:.3f}\n",
        FLAGS_backend, variant->name, variant->solution_id.Repr(), tokens, dim,
        inter_dim, experts, topk, in.num_valid_token_ids,
        in.num_valid_m_blocks, num_persistent_tgs,
        num_persistent_tgs > 0 ? "persistent" : "standard", result.device_ms,
        result.tflops, result.bandwidth_gbps);
    return 0;
}
