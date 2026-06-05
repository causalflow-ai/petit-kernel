#include "gemm/rocm/quantization/fp4/gemm_fp4.h"
#include "gemm/rocm/quantization/types.h"
#include "moe/rocm/fused_moe.h"
#include "moe/rocm/fused_moe_test_utils.h"
#include "utils/hip_helper.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>

namespace causalflow::petit::rocm::moe {

namespace {

namespace quant = causalflow::petit::rocm::quantization;
namespace fp4 = causalflow::petit::rocm::quantization::fp4;
namespace moe_test = causalflow::petit::rocm::moe::test_utils;

static constexpr FusedMoESolutionId kTestSolutionId = FusedMoESolutionId::Make(
    FusedMoEDataType::kChannelScaleFp8, FusedMoEDataType::kMxFp4,
    FusedMoEDataType::kNone, FusedMoEWeightOrdering::kPetitMxFp4,
    FusedMoEMfmaShape::kMfmaFp816x16x32, FusedMoEStages::kOneStage,
    FusedMoEActivationFunction::kSiluDot,
    FusedMoEStage1Buffering::kSingleBuffer);

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

template <class Config>
class Fp8InputMxFp4Runner : public moe_test::TestRunnerBase {
  public:
    using Context = DeviceContext<Config>;
    using Base = moe_test::TestRunnerBase;

    Fp8InputMxFp4Runner();
    ~Fp8InputMxFp4Runner() override;

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
    void AdjustScalePatterns() override {}
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
Fp8InputMxFp4Runner<Config>::Fp8InputMxFp4Runner()
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

template <class Config> Fp8InputMxFp4Runner<Config>::~Fp8InputMxFp4Runner() {
    CheckHIPStatus(hipFree(d_ctx_));
    d_ctx_ = nullptr;
}

template <class Config>
void Fp8InputMxFp4Runner<Config>::CopyHostToDeviceContext() {
    CheckHIPStatus(hipMemcpy(d_ctx_, h_ctx_.get(), sizeof(Context),
                             hipMemcpyHostToDevice));
}

template <class Config> int Fp8InputMxFp4Runner<Config>::RunKernelImpl() {
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

template <class Config>
void Fp8InputMxFp4Runner<Config>::InitializeW13HostData() {
    const auto &dataset = moe_test::MxFp4TestData<Config>::Get();
    std::copy(dataset.w1.begin(), dataset.w1.end(), std::begin(h_ctx_->w1));
    std::copy(dataset.scale_fc1.begin(), dataset.scale_fc1.end(),
              std::begin(h_ctx_->scale_fc1));
}

template <class Config>
void Fp8InputMxFp4Runner<Config>::InitializeW2HostData() {
    const auto &dataset = moe_test::MxFp4TestData<Config>::Get();
    std::copy(dataset.w2.begin(), dataset.w2.end(), std::begin(h_ctx_->w2));
    std::copy(dataset.scale_fc2.begin(), dataset.scale_fc2.end(),
              std::begin(h_ctx_->scale_fc2));
}

template <class Config> void Fp8InputMxFp4Runner<Config>::DequantizeWeights() {
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
        err = fp4::DequantMxFp4(
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
        err = fp4::DequantMxFp4(
            reinterpret_cast<unsigned *>(dq_w2_expert), w2_expert,
            reinterpret_cast<const unsigned *>(scale_fc2_expert), 1.0f,
            quant::kDataTypeBf16, Context::kDim, Context::kInterDim);
        ASSERT_EQ(err, 0) << "DequantMxFp4 failed";
    }

    auto repack_weights = [](unsigned *dst, size_t words, unsigned rows,
                             unsigned cols) {
        unsigned *baseline = nullptr;
        CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&baseline),
                                 words * sizeof(unsigned)));
        CheckHIPStatus(hipMemcpy(baseline, dst, words * sizeof(unsigned),
                                 hipMemcpyDeviceToDevice));
        CheckHIPStatus(moe_test::RepackMoeMxFp4Weights(dst, baseline, rows,
                                                       cols, nullptr));
        CheckHIPStatus(hipFree(baseline));
    };
    auto repack_scales = [](unsigned char *dst, size_t bytes, unsigned rows,
                            unsigned scale_cols) {
        unsigned *baseline = nullptr;
        CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&baseline), bytes));
        CheckHIPStatus(hipMemcpy(baseline, dst, bytes, hipMemcpyDeviceToDevice));
        CheckHIPStatus(moe_test::RepackMoeMxFp4Scales(
            reinterpret_cast<unsigned *>(dst), baseline, rows, scale_cols,
            nullptr));
        CheckHIPStatus(hipFree(baseline));
    };

    repack_weights(d_ctx_->w1, std::size(d_ctx_->w1), Context::kW1Rows,
                   Context::kDim);
    repack_weights(d_ctx_->w2, std::size(d_ctx_->w2), Context::kDim,
                   Context::kInterDim);
    repack_scales(d_ctx_->scale_fc1, std::size(d_ctx_->scale_fc1),
                  Context::kW1Rows, Context::kDim / Context::kMxScaleGroup);
    repack_scales(d_ctx_->scale_fc2, std::size(d_ctx_->scale_fc2),
                  Context::kDim, Context::kInterDim / Context::kMxScaleGroup);
}

class FusedMoEMxFp4Test : public ::testing::Test {
  public:
    template <class Config> void RunComparisonTest() {
        Fp8InputMxFp4Runner<Config> runner;
        runner.Initialize();
        runner.RunTest();
    }
};

TEST_F(FusedMoEMxFp4Test, SmallMatchesPythonStyleReference) {
    RunComparisonTest<TestConfig<4, 256, 512, 4, 2>>();
}

TEST_F(FusedMoEMxFp4Test, MediumMatchesPythonStyleReference) {
    RunComparisonTest<TestConfig<8, 256, 512, 8, 2>>();
}

TEST_F(FusedMoEMxFp4Test, Large512MatchesPythonStyleReference) {
    RunComparisonTest<TestConfig<512, 4096, 1024, 8, 2>>();
}

TEST_F(FusedMoEMxFp4Test, Large1024MatchesPythonStyleReference) {
    RunComparisonTest<TestConfig<1024, 4096, 1024, 8, 2>>();
}

TEST_F(FusedMoEMxFp4Test, Large1537MatchesPythonStyleReference) {
    RunComparisonTest<TestConfig<1537, 4096, 1024, 8, 2>>();
}

TEST_F(FusedMoEMxFp4Test, DeepSeekLikeMatchesPythonStyleReference) {
    RunComparisonTest<TestConfig<8, 7168, 2048, 33, 9>>();
}

TEST_F(FusedMoEMxFp4Test, ReplayLikeSensitiveMatchesPythonStyleReference) {
    RunComparisonTest<ReplayLikeSensitiveConfig>();
}

} // namespace
} // namespace causalflow::petit::rocm::moe
