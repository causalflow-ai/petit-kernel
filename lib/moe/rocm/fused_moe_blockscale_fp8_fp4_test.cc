#include "causalflow/petit/tal/algorithm.h"
#include "gemm/rocm/quantization/fp4/gemm_fp4.h"
#include "gemm/rocm/quantization/gemm.h"
#include "gemm/rocm/quantization/types.h"
#include "moe/rocm/fused_moe.h"
#include "moe/rocm/fused_moe_test_utils.h"
#include "moe/rocm/quantization.cuh"
#include "tests/fp8_sampler.h"
#include "utils/hip_helper.h"
#include "utils/test_utils.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <vector>

namespace causalflow::petit::rocm::moe {

namespace {

namespace quant = causalflow::petit::rocm::quantization;
namespace fp4 = causalflow::petit::rocm::quantization::fp4;
namespace moe_test = causalflow::petit::rocm::moe::test_utils;
using moe_test::FillParallelIndexed;
using moe_test::MixU64;

static constexpr FusedMoESolutionId kTestSolutionId = FusedMoESolutionId::Make(
    FusedMoEDataType::kChannelScaleFp8, FusedMoEDataType::kMxFp4,
    FusedMoEDataType::kNone, FusedMoEWeightOrdering::kPetitMxFp4,
    FusedMoEMfmaShape::kMfmaFp816x16x32, FusedMoEStages::kOneStage,
    FusedMoEActivationFunction::kSiluDot,
    FusedMoEStage1Buffering::kSingleBuffer);

static inline unsigned MaskNegativeZeroOnNativeFp4Format(unsigned v) {
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

static inline unsigned PetitFormatWord(unsigned v) {
    static constexpr unsigned kSignOffsets[8] = {7, 15, 23, 31, 24, 16, 8, 0};
    static constexpr unsigned kValueOffsets[8] = {1, 9, 17, 25, 28, 20, 12, 4};
    unsigned out = 0;
    for (unsigned lane = 0; lane < 8; ++lane) {
        const unsigned u = (v >> (lane * 4)) & 0xfu;
        unsigned val = u & 0x7u;
        const unsigned sign = val == 0 ? 0 : u >> 3;
        if (lane >= 4) {
            val = __builtin_bitreverse32(val) >> 29;
        }
        out |= sign << kSignOffsets[lane];
        out |= val << kValueOffsets[lane];
    }
    return out;
}

static inline unsigned SampleMomentFp4Nibble(uint64_t seed, float zero_prob,
                                             float magnitude_mean,
                                             float magnitude_std) {
    const float zero_draw = moe_test::U32ToOpenUnitFloat(
        static_cast<uint32_t>(MixU64(seed ^ 0x243f6a8885a308d3ULL)));
    if (zero_draw < zero_prob) {
        return 0;
    }
    const float mag_sample =
        magnitude_mean + magnitude_std * moe_test::HashToStandardNormal(seed);
    const unsigned magnitude = static_cast<unsigned>(
        std::clamp(static_cast<int>(std::lround(mag_sample)), 1, 7));
    const unsigned sign =
        static_cast<unsigned>(MixU64(seed ^ 0x13198a2e03707344ULL) & 1ULL);
    return magnitude | (sign << 3);
}

static inline unsigned MakeMomentFp4Word(size_t idx, uint64_t stream,
                                         float zero_prob, float magnitude_mean,
                                         float magnitude_std) {
    unsigned native_word = 0;
    for (unsigned lane = 0; lane < 8; ++lane) {
        const uint64_t seed = moe_test::IndexedSeed(42, stream, idx * 8 + lane);
        native_word |= SampleMomentFp4Nibble(seed, zero_prob, magnitude_mean,
                                             magnitude_std)
                       << (4 * lane);
    }
    return native_word;
}

static inline unsigned MakeMomentW13NativeWord(size_t idx) {
    return MakeMomentFp4Word(idx, 0xbb67ae8584caa73bULL, 0.119f, 3.10f, 1.76f);
}

static inline unsigned MakeMomentW2NativeWord(size_t idx) {
    return MakeMomentFp4Word(idx, 0x6a09e667f3bcc909ULL, 0.299f, 2.58f, 1.77f);
}

static inline unsigned char SampleMomentScale(uint64_t seed, float mean,
                                              float stddev) {
    const float sample = mean + stddev * moe_test::HashToStandardNormal(seed);
    return static_cast<unsigned char>(
        std::clamp(static_cast<int>(std::lround(sample)), 1, 237));
}

template <unsigned kTokens_, unsigned kDim_, unsigned kInterDim_,
          unsigned kExperts_, unsigned kTopK_>
struct TestConfig {
    static constexpr unsigned kTokens = kTokens_;
    static constexpr unsigned kDim = kDim_;
    static constexpr unsigned kInterDim = kInterDim_;
    static constexpr unsigned kExperts = kExperts_;
    static constexpr unsigned kTopK = kTopK_;

    static constexpr unsigned kScaleGroup = 128;
    static constexpr unsigned kMxScaleGroup = 32;
    static constexpr unsigned kSortedTokenPadding = 32;
    static constexpr float kScaleInvStd = 8.0e-4f;
    static constexpr float kScaleInvMean = 5.0e-3f;
    static constexpr float kPerElementAtol = 2.5e-2f;
    static constexpr float kOcpFp8PerElementAtol = kPerElementAtol;
    static constexpr float kPerElementRtol = 0.0f;
    static constexpr bool kUseNativeCheckpointLayout = false;

    template <class Context> static void AdjustScalePatterns(Context &) {}
    static void AdjustTopKPatterns(std::vector<unsigned> &,
                                   std::vector<float> &) {}
};

struct ReplayLikeSensitiveConfig : TestConfig<2, 7168, 2048, 32, 8> {
    static constexpr float kPerElementAtol = moe_test::kReplayLikeSensitiveAtol;
    static constexpr float kOcpFp8PerElementAtol = kPerElementAtol;
    static constexpr float kScaleInvStd = 2.0e-2f;
    static constexpr float kScaleInvMean = 1.0e-1f;
    template <class Context> static void AdjustScalePatterns(Context &ctx) {
        FillParallelIndexed(std::span(ctx.w1), [&](size_t idx) {
            return PetitFormatWord(MakeMomentW13NativeWord(idx));
        });
        FillParallelIndexed(std::span(ctx.w2), [&](size_t idx) {
            return PetitFormatWord(MakeMomentW2NativeWord(idx));
        });
        FillParallelIndexed(std::span(ctx.scale_fc1), [&](size_t idx) {
            return SampleMomentScale(
                moe_test::IndexedSeed(42, 0x3c6ef372fe94f82bULL, idx), 117.0f,
                1.45f);
        });
        FillParallelIndexed(std::span(ctx.scale_fc2), [&](size_t idx) {
            return SampleMomentScale(
                moe_test::IndexedSeed(42, 0xa54ff53a5f1d36f1ULL, idx), 122.0f,
                1.15f);
        });
    }

    static void AdjustTopKPatterns(std::vector<unsigned> &topk_ids,
                                   std::vector<float> &topk_weights) {
        moe_test::ApplyReplayLikeSensitiveTopKPatterns<kTokens, kTopK>(
            topk_ids, topk_weights);
    }
};

template <class Config>
struct DeviceContext : public moe_test::ReferenceDeviceContext<Config> {
    using Base = moe_test::ReferenceDeviceContext<Config>;

    using Base::kDim;
    using Base::kExperts;
    using Base::kInterDim;
    using Base::kMaxNumMBlocks;
    using Base::kRoutes;
    using Base::kSortedTokenPadding;
    using Base::kTokens;
    using Base::kTopK;

    static constexpr unsigned kMxScaleGroup = Config::kMxScaleGroup;
    static constexpr unsigned kW1Rows = 2 * kInterDim;
    static constexpr unsigned kW2WordsPerExpert =
        kDim * kInterDim / sizeof(unsigned) / 2;
    __hip_bfloat16 dq_w13[kExperts * 2 * kDim * kInterDim];
    alignas(16) unsigned w1[kExperts * kW1Rows * kDim / sizeof(unsigned) / 2];
    alignas(16) unsigned w2[kExperts * kW2WordsPerExpert];
    alignas(
        16) unsigned char scale_fc1[kExperts * kW1Rows * kDim / kMxScaleGroup];
    alignas(16) unsigned char scale_fc2[kExperts * kDim * kInterDim /
                                        kMxScaleGroup];
};

template <class Config> class TestRunner : public moe_test::TestRunnerBase {
  public:
    using Context = DeviceContext<Config>;
    using Base = moe_test::TestRunnerBase;

    TestRunner();
    ~TestRunner() override;

    void InitializeW13HostData() override;
    void InitializeW2HostData() override;
    void DequantizeWeights() override;

  protected:
    moe_test::DeviceContextAccessorBase &HostAccessor() override {
        return *h_ctx_accessor_;
    }
    moe_test::DeviceContextAccessorBase &DeviceAccessor() override {
        return *d_ctx_accessor_;
    }
    void CopyHostToDeviceContext() override;
    void AdjustScalePatterns() override {
        Config::AdjustScalePatterns(*h_ctx_);
    }
    void AdjustTopKPatterns(std::vector<unsigned> &topk_ids,
                            std::vector<float> &topk_weights) override {
        Config::AdjustTopKPatterns(topk_ids, topk_weights);
    }
    int RunKernelImpl() override;

  private:
    std::unique_ptr<Context> h_ctx_;
    Context *d_ctx_ = nullptr;
    std::unique_ptr<moe_test::DeviceContextAccessor<Context>> h_ctx_accessor_;
    std::unique_ptr<moe_test::DeviceContextAccessor<Context>> d_ctx_accessor_;
};

template <class Config>
TestRunner<Config>::TestRunner()
    : Base(moe_test::MakeTestRunnerConfig<Config, Context>()),
      h_ctx_(std::make_unique<Context>()) {
    CheckHIPStatus(
        hipMalloc(reinterpret_cast<void **>(&d_ctx_), sizeof(Context)));
    h_ctx_accessor_ =
        std::make_unique<moe_test::DeviceContextAccessor<Context>>(
            h_ctx_.get());
    d_ctx_accessor_ =
        std::make_unique<moe_test::DeviceContextAccessor<Context>>(d_ctx_);
}

template <class Config> TestRunner<Config>::~TestRunner() {
    CheckHIPStatus(hipFree(d_ctx_));
    d_ctx_ = nullptr;
}

template <class Config> void TestRunner<Config>::CopyHostToDeviceContext() {
    CheckHIPStatus(hipMemcpy(d_ctx_, h_ctx_.get(), sizeof(Context),
                             hipMemcpyHostToDevice));
}

template <class Config> int TestRunner<Config>::RunKernelImpl() {
    FusedMoE1StageParams params{
        reinterpret_cast<unsigned *>(d_ctx_->out),
        reinterpret_cast<const unsigned *>(d_ctx_->q_act),
        reinterpret_cast<const unsigned *>(d_ctx_->w1),
        reinterpret_cast<const unsigned *>(d_ctx_->w2),
        reinterpret_cast<const unsigned *>(d_ctx_->sorted_token_ids),
        reinterpret_cast<const unsigned *>(d_ctx_->sorted_weights),
        reinterpret_cast<const unsigned *>(d_ctx_->sorted_expert_ids),
        d_ctx_->num_valid_ids,
        Context::kTopK,
        reinterpret_cast<const unsigned *>(d_ctx_->scale_act_t),
        reinterpret_cast<const unsigned *>(d_ctx_->scale_fc1),
        reinterpret_cast<const unsigned *>(d_ctx_->scale_fc2),
        Context::kMaxNumMBlocks,
        Context::kTokens,
        Context::kDim,
        Context::kInterDim,
        Context::kExperts,
        nullptr,
        0,
    };
    return FusedMoEMatmul1Stage(params, kTestSolutionId.Repr());
}

template <class Config> void TestRunner<Config>::InitializeW13HostData() {
    FillParallelIndexed(std::span(h_ctx_->w1), [&](size_t idx) {
        return MaskNegativeZeroOnNativeFp4Format(static_cast<unsigned>(
            MixU64(Base::kSeed ^ (0x2d5ULL << 32) ^ idx)));
    });

    // Match lib/tests/quantization.cc:GemmMPTestData::GenerateScales for MXFP4.
    static constexpr unsigned kMxScaleMin = 1;
    static constexpr unsigned kMxScaleNoOverflowMax = 237;
    std::mt19937 gen(Base::kSeed);
    std::uniform_int_distribution<unsigned> dist(kMxScaleMin,
                                                 kMxScaleNoOverflowMax);
    std::generate(std::begin(h_ctx_->scale_fc1), std::end(h_ctx_->scale_fc1),
                  [&]() { return static_cast<unsigned char>(dist(gen)); });
}

template <class Config> void TestRunner<Config>::InitializeW2HostData() {
    FillParallelIndexed(std::span(h_ctx_->w2), [&](size_t idx) {
        return MaskNegativeZeroOnNativeFp4Format(static_cast<unsigned>(
            MixU64(Base::kSeed ^ (0x2d5ULL << 32) ^ idx)));
    });

    // Match lib/tests/quantization.cc:GemmMPTestData::GenerateScales for MXFP4.
    static constexpr unsigned kMxScaleMin = 1;
    static constexpr unsigned kMxScaleNoOverflowMax = 237;
    std::mt19937 gen(Base::kSeed);
    std::uniform_int_distribution<unsigned> dist(kMxScaleMin,
                                                 kMxScaleNoOverflowMax);
    std::generate(std::begin(h_ctx_->scale_fc2), std::end(h_ctx_->scale_fc2),
                  [&]() { return static_cast<unsigned char>(dist(gen)); });
}

template <class Config> void TestRunner<Config>::DequantizeWeights() {
    int err;
    for (unsigned expert = 0; expert < Context::kExperts; ++expert) {
        auto *dq_w13_expert = d_ctx_->dq_w13 + static_cast<size_t>(expert) * 2 *
                                                   Context::kDim *
                                                   Context::kInterDim;
        auto *w1_expert = d_ctx_->w1 + static_cast<size_t>(expert) *
                                           Context::kW1Rows * Context::kDim /
                                           sizeof(unsigned) / 2;
        auto *scale_fc1_expert =
            d_ctx_->scale_fc1 + static_cast<size_t>(expert) * Context::kW1Rows *
                                    Context::kDim / Context::kMxScaleGroup;
        err = moe_test::DequantMxFp4WithGroup4(
            reinterpret_cast<unsigned *>(dq_w13_expert), w1_expert,
            reinterpret_cast<const unsigned *>(scale_fc1_expert), 1.0f,
            quant::kDataTypeBf16, Context::kInterDim * 2, Context::kDim);
        ASSERT_EQ(err, 0) << "DequantMxFp4 failed";

        auto *dq_w2_expert = d_ctx_->dq_w2 + static_cast<size_t>(expert) *
                                                 Context::kInterDim *
                                                 Context::kDim;
        auto *w2_expert = d_ctx_->w2 + static_cast<size_t>(expert) *
                                           Context::kW2WordsPerExpert;
        auto *scale_fc2_expert =
            d_ctx_->scale_fc2 + static_cast<size_t>(expert) * Context::kDim *
                                    Context::kInterDim / Context::kMxScaleGroup;
        err = moe_test::DequantMxFp4WithGroup4(
            reinterpret_cast<unsigned *>(dq_w2_expert), w2_expert,
            reinterpret_cast<const unsigned *>(scale_fc2_expert), 1.0f,
            quant::kDataTypeBf16, Context::kDim, Context::kInterDim);
        ASSERT_EQ(err, 0) << "DequantMxFp4 failed";
    }
}

class FusedMoEBlockScaleFP8MxFp4Test : public ::testing::Test {
  public:
    template <class Config> void RunComparisonTest() {
        TestRunner<Config> runner;
        runner.Initialize();
        runner.RunTest();
    }
};

TEST_F(FusedMoEBlockScaleFP8MxFp4Test, SmallMatchesPythonStyleReference) {
    RunComparisonTest<TestConfig<4, 256, 512, 4, 2>>();
}

TEST_F(FusedMoEBlockScaleFP8MxFp4Test,
       MediumMatchesPythonStyleReference) {
    RunComparisonTest<TestConfig<8, 256, 512, 8, 2>>();
}

TEST_F(FusedMoEBlockScaleFP8MxFp4Test,
       Large512MatchesPythonStyleReference) {
    RunComparisonTest<TestConfig<512, 4096, 1024, 8, 2>>();
}

TEST_F(FusedMoEBlockScaleFP8MxFp4Test,
       Large1024MatchesPythonStyleReference) {
    RunComparisonTest<TestConfig<1024, 4096, 1024, 8, 2>>();
}

TEST_F(FusedMoEBlockScaleFP8MxFp4Test,
       Large1537MatchesPythonStyleReference) {
    RunComparisonTest<TestConfig<1537, 4096, 1024, 8, 2>>();
}

TEST_F(FusedMoEBlockScaleFP8MxFp4Test,
       DeepSeekLikeMatchesPythonStyleReference) {
    RunComparisonTest<TestConfig<8, 7168, 2048, 33, 9>>();
}

TEST_F(FusedMoEBlockScaleFP8MxFp4Test,
       ReplayLikeSensitiveMatchesPythonStyleReference) {
    RunComparisonTest<ReplayLikeSensitiveConfig>();
}

} // namespace
} // namespace causalflow::petit::rocm::moe
