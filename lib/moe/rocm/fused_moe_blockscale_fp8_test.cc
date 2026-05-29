#include "quantization.cuh"

#include "causalflow/petit/tal/algorithm.h"
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

namespace moe_test = causalflow::petit::rocm::moe::test_utils;
using moe_test::FillParallelIndexed;
using moe_test::FillWeightsParallel;

static constexpr FusedMoESolutionId kTestSolutionId = FusedMoESolutionId::Make(
    FusedMoEDataType::kChannelScaleFp8, FusedMoEDataType::kBlockScaleFp8,
    FusedMoEDataType::kNone, FusedMoEWeightOrdering::kPetitFp8,
    FusedMoEMfmaShape::kMfmaFp816x16x32, FusedMoEStages::kOneStage,
    FusedMoEActivationFunction::kSiluDot,
    FusedMoEStage1Buffering::kDoubleBuffer);

template <unsigned kTokens_, unsigned kDim_, unsigned kInterDim_,
          unsigned kExperts_, unsigned kTopK_>
struct TestConfig {
    static constexpr unsigned kTokens = kTokens_;
    static constexpr unsigned kDim = kDim_;
    static constexpr unsigned kInterDim = kInterDim_;
    static constexpr unsigned kExperts = kExperts_;
    static constexpr unsigned kTopK = kTopK_;

    static constexpr unsigned kScaleGroup = 128;
    static constexpr unsigned kSortedTokenPadding = 32;
    static constexpr float kScaleInvStd = 8.0e-4f;
    static constexpr float kScaleInvMean = 5.0e-3f;
    static constexpr float kPerElementAtol = 2.5e-2f;
    static constexpr float kOcpFp8PerElementAtol = 1.3e-1f;
    static constexpr float kPerElementRtol = 0.0f;

    template <class Context> static void AdjustScalePatterns(Context &) {}
    static void AdjustTopKPatterns(std::vector<unsigned> &,
                                   std::vector<float> &) {}
};

struct ReplayLikeSensitiveConfig : TestConfig<2, 7168, 2048, 32, 8> {
    static constexpr float kPerElementAtol = moe_test::kReplayLikeSensitiveAtol;
    static constexpr float kOcpFp8PerElementAtol = 1.3e-1f;
    static constexpr float kScaleInvStd = 2.0e-2f;
    static constexpr float kScaleInvMean = 1.0e-1f;

    template <class Context> static void AdjustScalePatterns(Context &ctx) {
        causalflow::petit::tests::fp8_sampler::FP8E4M3QuantizedNormalSampler
            w13_sampler(0.0, 1.2);
        causalflow::petit::tests::fp8_sampler::FP8E4M3QuantizedNormalSampler
            w2_sampler(0.0, 1.0);
        FillParallelIndexed(std::span(ctx.w1), [&](size_t idx) {
            moe_test::FixedU32Rng rng{static_cast<uint32_t>(moe_test::MixU64(
                moe_test::IndexedSeed(42, 0xbb67ae8584caa73bULL, idx)))};
            return w13_sampler(rng);
        });
        FillParallelIndexed(std::span(ctx.w2), [&](size_t idx) {
            moe_test::FixedU32Rng rng{static_cast<uint32_t>(moe_test::MixU64(
                moe_test::IndexedSeed(42, 0x6a09e667f3bcc909ULL, idx)))};
            return w2_sampler(rng);
        });
        FillParallelIndexed(std::span(ctx.scale_fc1), [&](size_t idx) {
            const float scale =
                0.020f +
                0.004f * moe_test::HashToStandardNormal(moe_test::IndexedSeed(
                             42, 0x3c6ef372fe94f82bULL, idx));
            return std::max(scale, 1.0e-8f);
        });
        FillParallelIndexed(std::span(ctx.scale_fc2), [&](size_t idx) {
            const float scale =
                0.025f +
                0.004f * moe_test::HashToStandardNormal(moe_test::IndexedSeed(
                             42, 0xa54ff53a5f1d36f1ULL, idx));
            return std::max(scale, 1.0e-8f);
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

    static constexpr unsigned kScaleGroup = Config::kScaleGroup;
    using Base::kDim;
    using Base::kExperts;
    using Base::kInterDim;
    using Base::kRoutes;
    using Base::kSortedTokenPadding;
    using Base::kTokens;
    using Base::kTopK;
    static constexpr unsigned kW1Rows = 2 * kInterDim;
    static constexpr unsigned kNBlocks = kDim / kScaleGroup;
    static constexpr unsigned kKBlocks = kInterDim / kScaleGroup;

    __hip_bfloat16 dq_w13[kExperts * 2 * kDim * kInterDim];
    float scale_fc1[kExperts * (kW1Rows / kScaleGroup) * kNBlocks];
    alignas(16) unsigned char w1[kExperts * kW1Rows * kDim];
    float scale_fc2[kExperts * kNBlocks * kKBlocks];

    alignas(16) unsigned char w2[kExperts * kDim * kInterDim];
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
    causalflow::petit::tests::fp8_sampler::FP8E4M3DiscreteSampler
        weight_sampler(Base::kWeightStd);
    // Generate weights directly in the kernel's shuffled layout.
    FillWeightsParallel(std::span(h_ctx_->w1), weight_sampler, Base::kSeed,
                        0x913u);
    std::mt19937 gen(Base::kSeed ^ 0x5f3u);
    auto local_gen_scale = [&]() { return Base::SampleScale(gen); };
    FillRandomValue(local_gen_scale, std::span(h_ctx_->scale_fc1));
}

template <class Config> void TestRunner<Config>::InitializeW2HostData() {
    causalflow::petit::tests::fp8_sampler::FP8E4M3DiscreteSampler
        weight_sampler(Base::kWeightStd);
    // Generate weights directly in the kernel's shuffled layout.
    FillWeightsParallel(std::span(h_ctx_->w2), weight_sampler, Base::kSeed,
                        0x2d5u);
    std::mt19937 gen(Base::kSeed ^ 0x6b5u);
    auto local_gen_scale = [&]() { return Base::SampleScale(gen); };
    FillRandomValue(local_gen_scale, std::span(h_ctx_->scale_fc2));
}

template <class Config> void TestRunner<Config>::DequantizeWeights() {
    CheckHIPStatus(moe_test::DequantizeShuffledBlockScaleFp8(
        d_ctx_->w1, d_ctx_->scale_fc1, d_ctx_->dq_w13, Context::kInterDim,
        Context::kDim, 2 * Context::kExperts));
    CheckHIPStatus(moe_test::DequantizeShuffledBlockScaleFp8(
        d_ctx_->w2, d_ctx_->scale_fc2, d_ctx_->dq_w2, Context::kDim,
        Context::kInterDim, Context::kExperts));
    CheckHIPStatus(hipDeviceSynchronize());
}

class FusedMoEBlockScaleFP8Test : public ::testing::Test {
  public:
    template <class Config> void RunReferenceTest() {
        TestRunner<Config> runner;
        runner.Initialize();
        runner.RunTest();
    }
};

TEST_F(FusedMoEBlockScaleFP8Test, SmallMatchesPythonStyleReference) {
    RunReferenceTest<TestConfig<4, 256, 512, 4, 2>>();
}

TEST_F(FusedMoEBlockScaleFP8Test, MediumMatchesPythonStyleReference) {
    RunReferenceTest<TestConfig<8, 256, 512, 8, 2>>();
}

TEST_F(FusedMoEBlockScaleFP8Test, Large512MatchesPythonStyleReference) {
    RunReferenceTest<TestConfig<512, 4096, 1024, 8, 2>>();
}

TEST_F(FusedMoEBlockScaleFP8Test, Large1024MatchesPythonStyleReference) {
    RunReferenceTest<TestConfig<1024, 4096, 1024, 8, 2>>();
}

TEST_F(FusedMoEBlockScaleFP8Test, Large1537MatchesPythonStyleReference) {
    RunReferenceTest<TestConfig<1537, 4096, 1024, 8, 2>>();
}

TEST_F(FusedMoEBlockScaleFP8Test, DeepSeekLikeMatchesPythonStyleReference) {
    RunReferenceTest<TestConfig<8, 7168, 2048, 33, 9>>();
}

TEST_F(FusedMoEBlockScaleFP8Test, ReplayLikeSensitiveMatchesReference) {
    RunReferenceTest<ReplayLikeSensitiveConfig>();
}

struct UltraHarshStage1ScaleConfig : TestConfig<32, 512, 512, 1, 1> {
    static constexpr float kScaleInvStd = 2.0e-4f;
    static constexpr float kScaleInvMean = 1.0e-3f;
    static constexpr float kPerElementAtol = 2.5e-2f;
    static constexpr float kPerElementRtol = 0.0f;

    template <class Context> static void AdjustScalePatterns(Context &ctx) {
        auto mod4_mult = [](unsigned idx) {
            switch (idx & 3u) {
            case 0:
                return 1.0f;
            case 1:
                return 2.0f;
            case 2:
                return 4.0f;
            default:
                return 8.0f;
            }
        };
        auto block4_mult = [](unsigned idx) {
            switch (idx & 3u) {
            case 0:
                return 1.0f;
            case 1:
                return 1.0f;
            case 2:
                return 2.0f;
            default:
                return 4.0f;
            }
        };

        for (unsigned token = 0; token < Context::kTokens; ++token) {
            const float token_m = mod4_mult(token);
            for (unsigned block = 0; block < Context::kActNBlocks; ++block) {
                const float block_m = block4_mult(block);
                ctx.scale_act_t[block * Context::kTokens + token] *=
                    token_m * block_m;
            }
        }

        const unsigned w1_row_blocks = Context::kW1Rows / Context::kScaleGroup;
        for (unsigned expert = 0; expert < Context::kExperts; ++expert) {
            for (unsigned rb = 0; rb < w1_row_blocks; ++rb) {
                const float row_m = mod4_mult(rb);
                for (unsigned block = 0; block < Context::kNBlocks; ++block) {
                    const float block_m = block4_mult(block);
                    ctx.scale_fc1[(expert * w1_row_blocks + rb) *
                                      Context::kNBlocks +
                                  block] *= row_m * block_m;
                }
            }
        }
    }
};

TEST_F(FusedMoEBlockScaleFP8Test, SingleExpertUltraHarshScaleMatchesReference) {
    RunReferenceTest<UltraHarshStage1ScaleConfig>();
}

} // namespace causalflow::petit::rocm::moe
