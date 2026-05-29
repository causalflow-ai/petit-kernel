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

  private:
    Context *ctx_;
};

hipError_t DequantizeShuffledBlockScaleFp8(const unsigned char *q,
                                           const float *scale,
                                           __hip_bfloat16 *dq, unsigned rows,
                                           unsigned cols, unsigned experts,
                                           hipStream_t stream = nullptr);

int DequantMxFp4WithGroup4(unsigned *output, const unsigned *input,
                           const unsigned *scales, float global_scale,
                           quantization::DataType out_type, unsigned m,
                           unsigned n, hipStream_t stream = nullptr);

class HipBlasLtRunner {
  public:
    HipBlasLtRunner();
    ~HipBlasLtRunner();

    void RunRowMajorGemm(const __hip_bfloat16 *d_a, const __hip_bfloat16 *d_b,
                         __hip_bfloat16 *d_c, unsigned m, unsigned n,
                         unsigned k) const;
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

    static constexpr size_t kWorkspaceSize = 16 * 1024 * 1024;
    hipblasLtHandle_t handle_ = nullptr;
    hipblasLtMatmulDesc_t default_desc_ = nullptr;
    hipblasLtMatmulDesc_t swish_desc_ = nullptr;
    void *workspace_ = nullptr;
};

struct TestRunnerConfig {
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
    virtual void AdjustScalePatterns() = 0;
    virtual void AdjustTopKPatterns(std::vector<unsigned> &,
                                    std::vector<float> &) {}
    virtual int RunKernelImpl() = 0;
    virtual void DequantizeWeights() = 0;

    HipBlasLtRunner gemm_;

  private:
    void InitializeHostData();
    void PrepareDequantizedActivations();
    void ComputeReferences();

    TestRunnerConfig config_;
    causalflow::petit::tests::fp8_sampler::FP8E4M3Format fp8_format_;
    std::vector<unsigned short> reference_out_;
};

} // namespace causalflow::petit::rocm::moe::test_utils
