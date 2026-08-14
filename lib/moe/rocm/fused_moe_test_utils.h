#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "gemm/rocm/quantization/types.h"
#include "tests/fp8_sampler.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace causalflow::petit::rocm::moe::test_utils {

inline uint64_t MixU64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct FixedU32Rng {
    using result_type = uint32_t;

    uint32_t value;

    static constexpr result_type min() { return 0; }

    static constexpr result_type max() {
        return std::numeric_limits<result_type>::max();
    }

    result_type operator()() { return value; }
};

inline float U32ToOpenUnitFloat(uint32_t value) {
    static constexpr float kInvRange = 1.0f / 4294967297.0f;
    return (static_cast<float>(value) + 1.0f) * kInvRange;
}

inline float HashToStandardNormal(uint64_t seed) {
    static constexpr float kTwoPi = 6.2831853071795864769f;
    const float u1 = U32ToOpenUnitFloat(static_cast<uint32_t>(MixU64(seed)));
    const float u2 = U32ToOpenUnitFloat(
        static_cast<uint32_t>(MixU64(seed ^ 0x9e3779b97f4a7c15ULL)));
    const float radius = std::sqrt(-2.0f * std::log(u1));
    const float theta = kTwoPi * u2;
    return radius * std::cos(theta);
}

inline uint64_t IndexedSeed(uint64_t seed, uint64_t stream, size_t idx) {
    return MixU64(seed ^ MixU64(stream) ^ MixU64(idx));
}

inline unsigned MaskNegativeZeroOnNativeFp4Format(unsigned v) {
    unsigned out = 0;
    for (unsigned i = 0; i < 8; ++i) {
        unsigned nibble = (v >> (i * 4)) & 0xfu;
        if (nibble == 0x8u) {
            nibble = 0u;
        }
        out |= nibble << (i * 4);
    }
    return out;
}

static constexpr float kReplayLikeSensitiveAtol = 2.05e-2f;
static constexpr unsigned kReplayLikeSensitiveTokens = 2;
static constexpr unsigned kReplayLikeSensitiveTopK = 8;
static constexpr unsigned
    kReplayLikeSensitiveTopKIds[kReplayLikeSensitiveTokens]
                               [kReplayLikeSensitiveTopK] = {
                                   {9, 18, 29, 30, 25, 4, 11, 3},
                                   {15, 12, 20, 23, 31, 7, 6, 21},
};

inline float ReplayLikeSensitiveTopKWeight(unsigned token, unsigned slot) {
    return 2.0f * (0.25f + 0.025f * static_cast<float>((token + 3 * slot) % 9));
}

template <unsigned kTokens, unsigned kTopK>
void ApplyReplayLikeSensitiveTopKPatterns(std::vector<unsigned> &topk_ids,
                                          std::vector<float> &topk_weights) {
    static_assert(kTokens == kReplayLikeSensitiveTokens);
    static_assert(kTopK == kReplayLikeSensitiveTopK);
    for (unsigned token = 0; token < kTokens; ++token) {
        for (unsigned slot = 0; slot < kTopK; ++slot) {
            const size_t idx = static_cast<size_t>(token) * kTopK + slot;
            topk_ids[idx] = kReplayLikeSensitiveTopKIds[token][slot];
            topk_weights[idx] = ReplayLikeSensitiveTopKWeight(token, slot);
        }
    }
}

template <class T, size_t kN, class Generator>
void FillParallelIndexed(std::span<T, kN> data, const Generator &generator) {
    T *const base = data.data();
    std::for_each(std::execution::par, data.begin(), data.end(), [&](T &value) {
        const size_t idx = static_cast<size_t>(&value - base);
        value = generator(idx);
    });
}

template <size_t kN, class Generator4>
void FillParallelIndexed4(std::span<unsigned char, kN> data,
                          const Generator4 &generator4) {
    static_assert(kN == std::dynamic_extent || (kN % 4) == 0,
                  "FillParallelIndexed4 expects size % 4 == 0");
    EXPECT_EQ(data.size() % 4, 0u);

    auto *const packed_ptr = reinterpret_cast<unsigned *>(data.data());
    std::span<unsigned> packed(packed_ptr, data.size() / 4);
    unsigned *const base = packed.data();
    std::for_each(std::execution::par, packed.begin(), packed.end(),
                  [&](unsigned &slot) {
                      const size_t group = static_cast<size_t>(&slot - base);
                      slot = generator4(group * 4);
                  });
}

template <class T, size_t kN, class Sampler>
void FillWeightsParallel(std::span<T, kN> data, const Sampler &sampler,
                         uint64_t seed, uint64_t stream) {
    T *const base = data.data();
    std::for_each(std::execution::par, data.begin(), data.end(), [&](T &value) {
        const size_t idx = static_cast<size_t>(&value - base);
        FixedU32Rng rng{
            static_cast<uint32_t>(MixU64(seed ^ (stream << 32) ^ idx))};
        value = sampler(rng);
    });
}

template <class Config>
struct MxFp4TestData {
    static constexpr unsigned kW13Rows = 2 * Config::kInterDim;
    static constexpr unsigned kW13Words =
        Config::kExperts * kW13Rows * Config::kDim / sizeof(unsigned) / 2;
    static constexpr unsigned kW2Words =
        Config::kExperts * Config::kDim * Config::kInterDim /
        sizeof(unsigned) / 2;
    static constexpr unsigned kScaleW13Bytes =
        Config::kExperts * kW13Rows * Config::kDim / Config::kMxScaleGroup;
    static constexpr unsigned kScaleW2Bytes =
        Config::kExperts * Config::kDim * Config::kInterDim /
        Config::kMxScaleGroup;

    std::vector<unsigned> w1;
    std::vector<unsigned> w2;
    std::vector<unsigned char> scale_fc1;
    std::vector<unsigned char> scale_fc2;

    static const MxFp4TestData &Get() {
        static const MxFp4TestData dataset = [] {
            static constexpr unsigned kSeed = 42;
            MxFp4TestData d;
            d.w1.resize(kW13Words);
            d.w2.resize(kW2Words);
            d.scale_fc1.resize(kScaleW13Bytes);
            d.scale_fc2.resize(kScaleW2Bytes);

            FillParallelIndexed(std::span(d.w1), [&](size_t idx) {
                return MaskNegativeZeroOnNativeFp4Format(static_cast<unsigned>(
                    MixU64(kSeed ^ (0x2d5ULL << 32) ^ idx)));
            });
            FillParallelIndexed(std::span(d.w2), [&](size_t idx) {
                return MaskNegativeZeroOnNativeFp4Format(static_cast<unsigned>(
                    MixU64(kSeed ^ (0x2d5ULL << 32) ^ idx)));
            });

            // Keep end-to-end MoE outputs in a range where fixed absolute
            // tolerances test math/packing errors instead of BF16 codepoint
            // differences in route/split-K accumulation.
            static constexpr unsigned kMxScaleMin = 1;
            static constexpr unsigned kMxScaleMax = 122;
            std::mt19937 gen(kSeed);
            std::uniform_int_distribution<unsigned> dist(kMxScaleMin,
                                                         kMxScaleMax);
            std::generate(d.scale_fc1.begin(), d.scale_fc1.end(), [&]() {
                return static_cast<unsigned char>(dist(gen));
            });
            std::generate(d.scale_fc2.begin(), d.scale_fc2.end(), [&]() {
                return static_cast<unsigned char>(dist(gen));
            });

            if constexpr (requires { Config::AdjustScalePatterns(d); }) {
                Config::AdjustScalePatterns(d);
            }
            return d;
        }();
        return dataset;
    }
};

template <class Config> struct ReferenceDeviceContext {
    static constexpr unsigned kSortedTokenPadding = Config::kSortedTokenPadding;
    static constexpr unsigned kActScaleGroup = 128;

    static constexpr unsigned kTokens = Config::kTokens;
    static constexpr unsigned kDim = Config::kDim;
    static constexpr unsigned kInterDim = Config::kInterDim;
    static constexpr unsigned kExperts = Config::kExperts;
    static constexpr unsigned kTopK = Config::kTopK;

    static_assert(kTopK <= kExperts, "");
    static_assert(kDim % 256 == 0 && kInterDim % 256 == 0, "");

    static constexpr unsigned kRoutes = kTokens * kTopK;
    static constexpr unsigned kActNBlocks = kDim / kActScaleGroup;

    static constexpr unsigned kMaxNumTokensPadded =
        kRoutes + kExperts * kSortedTokenPadding - kTopK;
    static constexpr unsigned kMaxNumMBlocks =
        tal::CeilingDiv<unsigned>(kMaxNumTokensPadded, kSortedTokenPadding);

    unsigned sorted_token_ids[kMaxNumTokensPadded];
    float sorted_weights[kMaxNumTokensPadded];
    unsigned sorted_expert_ids[kMaxNumMBlocks];
    unsigned num_valid_ids[2];

    __hip_bfloat16 dq_act[kTokens * kDim];
    alignas(16) __hip_bfloat16 dq_w2[kExperts * kInterDim * kDim];

    __hip_bfloat16 a[kRoutes * kDim];
    __hip_bfloat16 gate[kRoutes * kInterDim];
    __hip_bfloat16 up[kRoutes * kInterDim];
    __hip_bfloat16 act[kRoutes * kInterDim];
    __hip_bfloat16 route_out[kRoutes * kDim];
    unsigned route_tokens[kRoutes];
    float route_weights[kRoutes];
    float reference_acc[kTokens * kDim];

    alignas(16) unsigned char q_act[kTokens * kDim];
    float scale_act_t[kTokens * kActNBlocks];
    alignas(16) unsigned short out[kTokens * kDim];
};

class DeviceContextAccessorBase {
  public:
    virtual ~DeviceContextAccessorBase() = default;
    virtual unsigned *sorted_token_ids() const = 0;
    virtual float *sorted_weights() const = 0;
    virtual unsigned *sorted_expert_ids() const = 0;
    virtual unsigned *num_valid_ids() const = 0;

    virtual unsigned char *q_act() const = 0;
    virtual float *scale_act_t() const = 0;

    virtual __hip_bfloat16 *dq_act() const = 0;
    virtual unsigned short *out() const = 0;
    virtual const __hip_bfloat16 *dq_w13() const = 0;
    virtual const __hip_bfloat16 *dq_w2() const = 0;
    virtual __hip_bfloat16 *a() const = 0;
    virtual __hip_bfloat16 *gate() const = 0;
    virtual __hip_bfloat16 *up() const = 0;
    virtual __hip_bfloat16 *act() const = 0;
    virtual __hip_bfloat16 *route_out() const = 0;
    virtual unsigned *route_tokens() const = 0;
    virtual float *route_weights() const = 0;
    virtual float *reference_acc() const = 0;
    virtual const __hip_bfloat16 *logical_w13_bias() const = 0;
    virtual const __hip_bfloat16 *logical_w2_bias() const = 0;
};

template <class Context>
class DeviceContextAccessor final : public DeviceContextAccessorBase {
  public:
    explicit DeviceContextAccessor(Context *ctx) : ctx_(ctx) {}

    unsigned *sorted_token_ids() const override {
        return ctx_->sorted_token_ids;
    }

    float *sorted_weights() const override { return ctx_->sorted_weights; }

    unsigned *sorted_expert_ids() const override {
        return ctx_->sorted_expert_ids;
    }

    unsigned *num_valid_ids() const override { return ctx_->num_valid_ids; }
    unsigned char *q_act() const override { return ctx_->q_act; }
    float *scale_act_t() const override { return ctx_->scale_act_t; }
    __hip_bfloat16 *dq_act() const override { return ctx_->dq_act; }

    unsigned short *out() const override { return ctx_->out; }
    const __hip_bfloat16 *dq_w13() const override { return ctx_->dq_w13; }
    const __hip_bfloat16 *dq_w2() const override { return ctx_->dq_w2; }
    __hip_bfloat16 *a() const override { return ctx_->a; }
    __hip_bfloat16 *gate() const override { return ctx_->gate; }
    __hip_bfloat16 *up() const override { return ctx_->up; }
    __hip_bfloat16 *act() const override { return ctx_->act; }
    __hip_bfloat16 *route_out() const override { return ctx_->route_out; }
    unsigned *route_tokens() const override { return ctx_->route_tokens; }
    float *route_weights() const override { return ctx_->route_weights; }
    float *reference_acc() const override { return ctx_->reference_acc; }
    const __hip_bfloat16 *logical_w13_bias() const override {
        if constexpr (requires { ctx_->logical_w13_bias; }) {
            return ctx_->logical_w13_bias;
        }
        return nullptr;
    }
    const __hip_bfloat16 *logical_w2_bias() const override {
        if constexpr (requires { ctx_->logical_w2_bias; }) {
            return ctx_->logical_w2_bias;
        }
        return nullptr;
    }

  private:
    Context *ctx_;
};

hipError_t DequantizeShuffledBlockScaleFp8(const unsigned char *q,
                                           const float *scale,
                                           __hip_bfloat16 *dq, unsigned rows,
                                           unsigned cols, unsigned experts,
                                           hipStream_t stream = nullptr);

hipError_t ApplyElementwiseMultiply(const __hip_bfloat16 *a,
                                    const __hip_bfloat16 *b,
                                    __hip_bfloat16 *out, unsigned count,
                                    hipStream_t stream = nullptr);

hipError_t AddFloatRowBias(float *data, const __hip_bfloat16 *bias,
                           unsigned rows, unsigned cols,
                           hipStream_t stream = nullptr);

hipError_t AddBf16RowBias(__hip_bfloat16 *data, const __hip_bfloat16 *bias,
                          unsigned rows, unsigned cols,
                          hipStream_t stream = nullptr);

template <class Input>
hipError_t ApplyOpenAISwiGLU(const Input *gate, const Input *up,
                             __hip_bfloat16 *out, unsigned count,
                             hipStream_t stream = nullptr);

extern template hipError_t
ApplyOpenAISwiGLU<__hip_bfloat16>(const __hip_bfloat16 *, const __hip_bfloat16 *,
                                  __hip_bfloat16 *, unsigned, hipStream_t);
extern template hipError_t ApplyOpenAISwiGLU<float>(
    const float *, const float *, __hip_bfloat16 *, unsigned, hipStream_t);

hipError_t ScatterWeightedRoutes(const __hip_bfloat16 *route_out,
                                 const unsigned *route_tokens,
                                 const float *route_weights, float *token_out,
                                 unsigned routes, unsigned cols,
                                 hipStream_t stream = nullptr);

hipError_t QuantizeDequantMxFp4(__hip_bfloat16 *data, unsigned rows,
                                unsigned cols, hipStream_t stream = nullptr);

hipError_t DequantizeNativeMxFp4Activations(
    const unsigned char *q, const unsigned char *scale, __hip_bfloat16 *dq,
    unsigned rows, unsigned cols, hipStream_t stream = nullptr);

hipError_t RepackMxFp4Bias(__hip_bfloat16 *output,
                           const __hip_bfloat16 *input, unsigned rows,
                           unsigned cols, hipStream_t stream = nullptr);

hipError_t RepackNativeMxFp4Weights(unsigned *output, const unsigned *input,
                                    unsigned rows, unsigned cols,
                                    hipStream_t stream = nullptr);

hipError_t RepackNativeMxFp4Scales(unsigned *output, const unsigned *input,
                                   unsigned rows, unsigned scale_cols,
                                   hipStream_t stream = nullptr);

hipError_t RepackPetitMxFp4Weights(unsigned *output, const unsigned *input,
                                   unsigned rows, unsigned cols,
                                   hipStream_t stream = nullptr);

hipError_t RepackPetitMxFp4Scales(unsigned *output, const unsigned *input,
                                  unsigned rows, unsigned scale_cols,
                                  hipStream_t stream = nullptr);

class HipBlasLtRunner {
  public:
    HipBlasLtRunner();
    ~HipBlasLtRunner();

    void RunRowMajorGemm(const __hip_bfloat16 *d_a, const __hip_bfloat16 *d_b,
                         __hip_bfloat16 *d_c, unsigned m, unsigned n,
                         unsigned k) const;
    void RunRowMajorGemmToFloat(const __hip_bfloat16 *d_a,
                                const __hip_bfloat16 *d_b, float *d_c,
                                unsigned m, unsigned n, unsigned k) const;
    void RunRowMajorGemmAccumulate(const __hip_bfloat16 *d_a,
                                   const __hip_bfloat16 *d_b,
                                   __hip_bfloat16 *d_c, unsigned m, unsigned n,
                                   unsigned k) const;
    void RunRowMajorGemmSwish(const __hip_bfloat16 *d_a,
                              const __hip_bfloat16 *d_b, __hip_bfloat16 *d_c,
                              unsigned m, unsigned n, unsigned k) const;

  private:
    void RunRowMajorGemmWithDesc(hipblasLtMatmulDesc_t desc,
                                 const __hip_bfloat16 *d_a,
                                 const __hip_bfloat16 *d_b, __hip_bfloat16 *d_c,
                                 unsigned m, unsigned n, unsigned k,
                                 float beta) const;
    void RunRowMajorGemmToFloatWithDesc(hipblasLtMatmulDesc_t desc,
                                        const __hip_bfloat16 *d_a,
                                        const __hip_bfloat16 *d_b, float *d_c,
                                        unsigned m, unsigned n,
                                        unsigned k) const;

    static constexpr size_t kWorkspaceSize = 16 * 1024 * 1024;
    hipblasLtHandle_t handle_ = nullptr;
    hipblasLtMatmulDesc_t default_desc_ = nullptr;
    hipblasLtMatmulDesc_t swish_desc_ = nullptr;
    void *workspace_ = nullptr;
};

struct TestRunnerConfig {
    enum class ReferenceActivation {
        kSiluDot,
        kOpenAISwiGLU,
    };
    enum class ReferenceIntermediate {
        kBf16,
        kMxFp4,
    };

    unsigned tokens;
    unsigned dim;
    unsigned inter_dim;
    unsigned experts;
    unsigned topk;
    unsigned act_scale_group;
    unsigned sorted_token_padding;
    unsigned routes;
    unsigned max_num_tokens_padded;
    unsigned max_num_m_blocks;
    float scale_inv_std;
    float scale_inv_mean;
    float per_element_atol;
    float ocp_fp8_per_element_atol;
    float per_element_rtol;
    ReferenceActivation reference_activation;
    ReferenceIntermediate reference_intermediate;
};

template <class Config, class Context>
constexpr TestRunnerConfig MakeTestRunnerConfig() {
    return TestRunnerConfig{
        .tokens = Context::kTokens,
        .dim = Context::kDim,
        .inter_dim = Context::kInterDim,
        .experts = Context::kExperts,
        .topk = Context::kTopK,
        .act_scale_group = Context::kActScaleGroup,
        .sorted_token_padding = Context::kSortedTokenPadding,
        .routes = Context::kRoutes,
        .max_num_tokens_padded = Context::kMaxNumTokensPadded,
        .max_num_m_blocks = Context::kMaxNumMBlocks,
        .scale_inv_std = Config::kScaleInvStd,
        .scale_inv_mean = Config::kScaleInvMean,
        .per_element_atol = Config::kPerElementAtol,
        .ocp_fp8_per_element_atol = Config::kOcpFp8PerElementAtol,
        .per_element_rtol = Config::kPerElementRtol,
        .reference_activation = [] {
            if constexpr (requires { Config::kReferenceActivation; }) {
                return Config::kReferenceActivation;
            } else {
                return TestRunnerConfig::ReferenceActivation::kSiluDot;
            }
        }(),
        .reference_intermediate = [] {
            if constexpr (requires { Config::kReferenceIntermediate; }) {
                return Config::kReferenceIntermediate;
            } else {
                return TestRunnerConfig::ReferenceIntermediate::kBf16;
            }
        }(),
    };
}

class TestRunnerBase {
  public:
    static constexpr unsigned kSeed = 42;
    static constexpr float kWeightStd = 40.0f;
    static constexpr float kInputStd = 0.5f;
    static constexpr float kInputMean = 0.05f;
    static constexpr float kInputSpikeProb = 0.002f;
    static constexpr float kInputSpikeStd = 8.0f;

    explicit TestRunnerBase(TestRunnerConfig config);
    virtual ~TestRunnerBase();

    void Initialize();
    void RunKernel();
    void RunReferenceOnly();
    void RunTest();

  protected:
    float SampleScale(std::mt19937 &gen) const;
    const TestRunnerConfig &config() const { return config_; }

    virtual DeviceContextAccessorBase &HostAccessor() = 0;
    virtual DeviceContextAccessorBase &DeviceAccessor() = 0;
    virtual void CopyHostToDeviceContext() = 0;
    virtual void InitializeW13HostData() = 0;
    virtual void InitializeW2HostData() = 0;
    virtual void InitializeInputHostData(std::mt19937 &gen);
    virtual void AdjustScalePatterns() = 0;
    virtual void AdjustTopKPatterns(std::vector<unsigned> &,
                                    std::vector<float> &) {}
    virtual int RunKernelImpl() = 0;
    virtual void DequantizeWeights() = 0;
    virtual void PrepareDequantizedActivations();

    HipBlasLtRunner gemm_;

  private:
    void InitializeHostData();
    void ComputeReferences();

    TestRunnerConfig config_;
    causalflow::petit::tests::fp8_sampler::FP8E4M3Format fp8_format_;
    std::vector<unsigned short> reference_out_;
};

} // namespace causalflow::petit::rocm::moe::test_utils
