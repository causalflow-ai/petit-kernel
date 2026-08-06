#include "gemm/rocm/quantization/fp4/gemm_fp4.h"
#include "gemm/rocm/quantization/types.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "moe/rocm/fused_moe.h"
#include "moe/rocm/fused_moe_test_utils.h"
#include "utils/hip_helper.h"

#include <gtest/gtest.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

namespace causalflow::petit::rocm::moe {

namespace {

namespace quant = causalflow::petit::rocm::quantization;
namespace fp4 = causalflow::petit::rocm::quantization::fp4;
namespace moe_test = causalflow::petit::rocm::moe::test_utils;
namespace fp8_sampler = causalflow::petit::tests::fp8_sampler;

static constexpr FusedMoESolutionId kFp8PetitMxFp4SolutionId =
    FusedMoESolutionId::Make(
        FusedMoEDataType::kChannelScaleFp8, FusedMoEDataType::kMxFp4,
        FusedMoEDataType::kNone, FusedMoEWeightOrdering::kPetitMxFp4,
        FusedMoEMfmaShape::kMfmaFp816x16x32, FusedMoEStages::kOneStage,
        FusedMoEActivationFunction::kSiluDot,
        FusedMoEStage1Buffering::kSingleBuffer);

static constexpr FusedMoESolutionId kFp8PetitMxFp4BiasSolutionId =
    FusedMoESolutionId::Make(
        FusedMoEDataType::kChannelScaleFp8, FusedMoEDataType::kMxFp4,
        FusedMoEDataType::kBf16, FusedMoEWeightOrdering::kPetitMxFp4,
        FusedMoEMfmaShape::kMfmaFp816x16x32, FusedMoEStages::kOneStage,
        FusedMoEActivationFunction::kOpenAISwiGLU,
        FusedMoEStage1Buffering::kDoubleBuffer);

static constexpr FusedMoESolutionId kBf16NativeMxFp4BiasSolutionId =
    FusedMoESolutionId::Make(
        FusedMoEDataType::kBf16, FusedMoEDataType::kMxFp4,
        FusedMoEDataType::kBf16, FusedMoEWeightOrdering::kNativeMxFp4,
        FusedMoEMfmaShape::kMfmaBf16MxFp4, FusedMoEStages::kOneStage,
        FusedMoEActivationFunction::kOpenAISwiGLU,
        FusedMoEStage1Buffering::kDoubleBuffer);

static constexpr FusedMoESolutionId kMxFp4NativeMxFp4BiasSolutionId =
    FusedMoESolutionId::Make(
        FusedMoEDataType::kMxFp4, FusedMoEDataType::kMxFp4,
        FusedMoEDataType::kBf16, FusedMoEWeightOrdering::kNativeMxFp4,
        FusedMoEMfmaShape::kMfmaScaleFp4MxFp4, FusedMoEStages::kOneStage,
        FusedMoEActivationFunction::kOpenAISwiGLU,
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

struct NativeMxFp4LayoutSensitiveConfig : TestConfig<32, 3072, 256, 1, 1> {
    static constexpr float kPerElementAtol = 2.5e-2f;
    static constexpr float kOcpFp8PerElementAtol = kPerElementAtol;

    template <class Context>
    static void AdjustNativeMxFp4Activations(Context &ctx) {
        constexpr unsigned kRowVecs = kDim / kMxScaleGroup;
        auto input_pattern = [](unsigned token, unsigned row_vec,
                                unsigned word) {
            unsigned out = 0;
            for (unsigned i = 0; i < 8; ++i) {
                unsigned nibble =
                    1 + ((token * 7 + row_vec * 5 + word * 3 + i) % 7);
                if (((token + row_vec + word + i) & 1u) != 0) {
                    nibble |= 0x8u;
                }
                out |= nibble << (i * 4);
            }
            return out;
        };
        auto *q_words = reinterpret_cast<unsigned *>(ctx.q_mx_act);
        for (unsigned token = 0; token < kTokens; ++token) {
            for (unsigned row_vec = 0; row_vec < kRowVecs; ++row_vec) {
                const size_t base =
                    (static_cast<size_t>(token) * kRowVecs + row_vec) * 4;
                for (unsigned word = 0; word < 4; ++word) {
                    q_words[base + word] = input_pattern(token, row_vec, word);
                }
                ctx.scale_mx_act[static_cast<size_t>(token) * kRowVecs +
                                 row_vec] = 127;
            }
        }
    }

    static void AdjustTopKPatterns(std::vector<unsigned> &topk_ids,
                                   std::vector<float> &topk_weights) {
        std::fill(topk_ids.begin(), topk_ids.end(), 0u);
        std::fill(topk_weights.begin(), topk_weights.end(), 1.0f);
    }
};

float DecodeFp4E2M1(unsigned nibble) {
    static constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                      2.0f, 3.0f, 4.0f, 6.0f};
    const float mag = kMag[nibble & 0x7u];
    return (nibble & 0x8u) ? -mag : mag;
}

float DecodeE8M0(unsigned char scale) {
    union {
        unsigned u;
        float f;
    } v{static_cast<unsigned>(scale) << 23};
    return v.f;
}

template <class Config> struct Bf16InputMxFp4Config : Config {
    static constexpr auto kReferenceActivation =
        moe_test::TestRunnerConfig::ReferenceActivation::kOpenAISwiGLU;
};

template <class Config> struct NativeInputMxFp4Config : Config {
    static constexpr auto kReferenceActivation =
        moe_test::TestRunnerConfig::ReferenceActivation::kOpenAISwiGLU;
    static constexpr auto kReferenceIntermediate =
        moe_test::TestRunnerConfig::ReferenceIntermediate::kMxFp4;
};

template <class Config> struct Fp8InputMxFp4BiasConfig : Config {
    static constexpr auto kReferenceActivation =
        moe_test::TestRunnerConfig::ReferenceActivation::kOpenAISwiGLU;
};

template <> struct Fp8InputMxFp4BiasConfig<ReplayLikeSensitiveConfig>
    : ReplayLikeSensitiveConfig {
    static constexpr auto kReferenceActivation =
        moe_test::TestRunnerConfig::ReferenceActivation::kOpenAISwiGLU;
    // The FP8 path quantizes the activated hidden state before W2; the
    // HipBLASLt reference intentionally keeps that intermediate in BF16.
    static constexpr float kPerElementAtol = 1.2e-1f;
    static constexpr float kOcpFp8PerElementAtol = kPerElementAtol;
};

struct GptOssHiddenConfig : TestConfig<17, 3072, 4096, 32, 4> {};

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
    alignas(16) unsigned char q_mx_act[kTokens * kDim / 2];
    alignas(16) unsigned char scale_mx_act[kTokens * kDim / kMxScaleGroup];
    alignas(16) unsigned char scale_mx_act_sorted[kMaxNumMBlocks * kDim];
};

fp8_sampler::FP8E4M3Format CurrentDeviceFp8Format() {
    int dev = 0;
    CheckHIPStatus(hipGetDevice(&dev));
    hipDeviceProp_t props;
    CheckHIPStatus(hipGetDeviceProperties(&props, dev));
    const char *arch = props.gcnArchName;
    if (std::strncmp(arch, "gfx950", 6) == 0 ||
        std::strncmp(arch, "gfx1200", 7) == 0 ||
        std::strncmp(arch, "gfx1201", 7) == 0) {
        return fp8_sampler::FP8E4M3Format::kOcp;
    }
    return fp8_sampler::FP8E4M3Format::kFnuz;
}

unsigned char EncodeFp8E4M3(float value, fp8_sampler::FP8E4M3Format format) {
    if (format == fp8_sampler::FP8E4M3Format::kOcp) {
        return __hip_fp8_e4m3(value).__x;
    }
    return __hip_fp8_e4m3_fnuz(value).__x;
}

template <class Context> void GenerateNativeMxFp4Activations(Context *ctx) {
    auto gen_q = [&](size_t idx) {
        return moe_test::MaskNegativeZeroOnNativeFp4Format(
            static_cast<unsigned>(
                moe_test::MixU64(moe_test::TestRunnerBase::kSeed ^
                                 (0x511ULL << 32) ^ idx)));
    };
    moe_test::FillParallelIndexed(
        std::span(reinterpret_cast<unsigned *>(ctx->q_mx_act),
                  std::size(ctx->q_mx_act) / sizeof(unsigned)),
        gen_q);

    std::mt19937 gen(moe_test::TestRunnerBase::kSeed);
    std::uniform_int_distribution<unsigned> scale_dist(115, 119);
    std::generate(std::begin(ctx->scale_mx_act),
                  std::end(ctx->scale_mx_act), [&]() {
                      return static_cast<unsigned char>(scale_dist(gen));
                  });
    if constexpr (requires { Context::AdjustNativeMxFp4Activations(*ctx); }) {
        Context::AdjustNativeMxFp4Activations(*ctx);
    }
}

template <class Context>
__global__ void PackAiterSortedMxFp4ActivationScalesKernel(Context *ctx) {
    constexpr unsigned kRoutesPerGroup = Context::kSortedTokenPadding;
    constexpr unsigned kScaleColsPerK256 = 8;
    constexpr unsigned kRowsPerHalf = 16;
    constexpr unsigned kColsPerHalf = 4;
    static_assert(kRoutesPerGroup == 32, "");
    static_assert(Context::kMxScaleGroup == 32, "");
    static_assert(Context::kDim % (Context::kMxScaleGroup *
                                   kScaleColsPerK256) == 0,
                  "");

    constexpr unsigned kTileBytes = kRoutesPerGroup * kScaleColsPerK256;
    static_assert(kTileBytes == 256, "");

    const unsigned route_group = blockIdx.x;
    const unsigned k256 = blockIdx.y;
    __shared__ unsigned tokens[kRoutesPerGroup];

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
        tokens[route] =
            ctx->sorted_token_ids[route_group * kRoutesPerGroup + route] &
            0x00ffffffu;
    }
    __syncthreads();

    const unsigned local_scale_col =
        scale_col_layout(make_coord(group2, n4));
    const unsigned token = tokens[route];
    unsigned char scale = 127;
    if (token < Context::kTokens) {
        scale =
            ctx->scale_mx_act[static_cast<size_t>(token) *
                                  (Context::kDim / Context::kMxScaleGroup) +
                              k256 * kScaleColsPerK256 + local_scale_col];
    }
    const unsigned dst = sorted_scale_layout(
        make_coord(n4, m16, make_coord(group2, group0)));
    ctx->scale_mx_act_sorted[static_cast<size_t>(route_group) * Context::kDim +
                             k256 * kTileBytes + dst] = scale;
}

template <class Context>
void PackAiterSortedMxFp4ActivationScales(Context *d_ctx) {
    constexpr unsigned kScaleColsPerK256 = 8;
    constexpr unsigned kScaleBlocksPerRouteGroup =
        Context::kDim / (Context::kMxScaleGroup * kScaleColsPerK256);
    constexpr unsigned kTileBytes =
        Context::kSortedTokenPadding * kScaleColsPerK256;
    static_assert(kTileBytes == 256, "");

    dim3 grid(Context::kMaxNumMBlocks, kScaleBlocksPerRouteGroup);
    PackAiterSortedMxFp4ActivationScalesKernel<Context>
        <<<grid, dim3(4, 16, 4)>>>(d_ctx);
    CheckHIPStatus(hipGetLastError());
}

template <class Context> void DeriveFp8ActivationsFromNativeMxFp4(Context *ctx) {
    const auto format = CurrentDeviceFp8Format();
    constexpr unsigned kFp8ScaleGroup = 128;
    const unsigned scale_cols = Context::kDim / Context::kMxScaleGroup;
    for (unsigned token = 0; token < Context::kTokens; ++token) {
        for (unsigned block = 0; block < Context::kDim / kFp8ScaleGroup;
             ++block) {
            unsigned base_scale = 255;
            for (unsigned sub = 0; sub < kFp8ScaleGroup / Context::kMxScaleGroup;
                 ++sub) {
                base_scale = std::min<unsigned>(
                    base_scale,
                    ctx->scale_mx_act[static_cast<size_t>(token) * scale_cols +
                                      block * 4 + sub]);
            }
            const float scale = DecodeE8M0(static_cast<unsigned char>(base_scale));
            ctx->scale_act_t[block * Context::kTokens + token] = scale;
            for (unsigned offset = 0; offset < kFp8ScaleGroup; ++offset) {
                const unsigned col = block * kFp8ScaleGroup + offset;
                const unsigned byte =
                    ctx->q_mx_act[static_cast<size_t>(token) *
                                      Context::kDim / 2 +
                                  col / 2];
                const unsigned nibble = (byte >> ((col & 1u) * 4)) & 0xfu;
                const unsigned mx_scale =
                    ctx->scale_mx_act[static_cast<size_t>(token) * scale_cols +
                                      col / Context::kMxScaleGroup];
                const float native_value =
                    DecodeFp4E2M1(nibble) *
                    DecodeE8M0(static_cast<unsigned char>(mx_scale));
                ctx->q_act[static_cast<size_t>(token) * Context::kDim + col] =
                    EncodeFp8E4M3(native_value / scale, format);
            }
        }
    }
}

template <class Config, class RunnerConfig = Config,
          unsigned long kSolutionRepr = kFp8PetitMxFp4SolutionId.Repr()>
class Fp8InputMxFp4Runner : public moe_test::TestRunnerBase {
  public:
    using Context = DeviceContext<RunnerConfig>;
    using Base = moe_test::TestRunnerBase;

    Fp8InputMxFp4Runner();
    ~Fp8InputMxFp4Runner() override;

    void InitializeW13HostData() override;
    void InitializeW2HostData() override;
    void InitializeInputHostData(std::mt19937 &) override;
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

template <class Config, class RunnerConfig, unsigned long kSolutionRepr>
Fp8InputMxFp4Runner<Config, RunnerConfig,
                    kSolutionRepr>::Fp8InputMxFp4Runner()
    : Base(moe_test::MakeTestRunnerConfig<RunnerConfig, Context>()),
      h_ctx_(std::make_unique<Context>()) {
    CheckHIPStatus(
        hipMalloc(reinterpret_cast<void **>(&d_ctx_), sizeof(Context)));
    h_ctx_accessor_ =
        std::make_unique<moe_test::DeviceContextAccessor<Context>>(
            h_ctx_.get());
    d_ctx_accessor_ =
        std::make_unique<moe_test::DeviceContextAccessor<Context>>(d_ctx_);
}

template <class Config, class RunnerConfig, unsigned long kSolutionRepr>
Fp8InputMxFp4Runner<Config, RunnerConfig,
                    kSolutionRepr>::~Fp8InputMxFp4Runner() {
    CheckHIPStatus(hipFree(d_ctx_));
    d_ctx_ = nullptr;
}

template <class Config, class RunnerConfig, unsigned long kSolutionRepr>
void Fp8InputMxFp4Runner<Config, RunnerConfig,
                         kSolutionRepr>::CopyHostToDeviceContext() {
    CheckHIPStatus(hipMemcpy(d_ctx_, h_ctx_.get(), sizeof(Context),
                             hipMemcpyHostToDevice));
}

template <class Config, class RunnerConfig, unsigned long kSolutionRepr>
int Fp8InputMxFp4Runner<Config, RunnerConfig, kSolutionRepr>::RunKernelImpl() {
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
    return FusedMoEMatmul1Stage(params, kSolutionRepr);
}

template <class Config, class RunnerConfig, unsigned long kSolutionRepr>
void Fp8InputMxFp4Runner<Config, RunnerConfig,
                         kSolutionRepr>::InitializeW13HostData() {
    const auto &dataset = moe_test::MxFp4TestData<Config>::Get();
    std::copy(dataset.w1.begin(), dataset.w1.end(), std::begin(h_ctx_->w1));
    std::copy(dataset.scale_fc1.begin(), dataset.scale_fc1.end(),
              std::begin(h_ctx_->scale_fc1));
}

template <class Config, class RunnerConfig, unsigned long kSolutionRepr>
void Fp8InputMxFp4Runner<Config, RunnerConfig,
                         kSolutionRepr>::InitializeW2HostData() {
    const auto &dataset = moe_test::MxFp4TestData<Config>::Get();
    std::copy(dataset.w2.begin(), dataset.w2.end(), std::begin(h_ctx_->w2));
    std::copy(dataset.scale_fc2.begin(), dataset.scale_fc2.end(),
              std::begin(h_ctx_->scale_fc2));
}

template <class Config, class RunnerConfig, unsigned long kSolutionRepr>
void Fp8InputMxFp4Runner<Config, RunnerConfig,
                         kSolutionRepr>::InitializeInputHostData(std::mt19937 &) {
    GenerateNativeMxFp4Activations(h_ctx_.get());
    DeriveFp8ActivationsFromNativeMxFp4(h_ctx_.get());
    Base::PrepareDequantizedActivations();
}

template <class Config, class RunnerConfig, unsigned long kSolutionRepr>
void Fp8InputMxFp4Runner<Config, RunnerConfig,
                         kSolutionRepr>::DequantizeWeights() {
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
        CheckHIPStatus(moe_test::RepackPetitMxFp4Weights(dst, baseline, rows,
                                                       cols, nullptr));
        CheckHIPStatus(hipFree(baseline));
    };
    auto repack_scales = [](unsigned char *dst, size_t bytes, unsigned rows,
                            unsigned scale_cols) {
        unsigned *baseline = nullptr;
        CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&baseline), bytes));
        CheckHIPStatus(hipMemcpy(baseline, dst, bytes, hipMemcpyDeviceToDevice));
        CheckHIPStatus(moe_test::RepackPetitMxFp4Scales(
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

template <class Config>
class Bf16InputMxFp4Runner : public moe_test::TestRunnerBase {
  public:
    using RunnerConfig = Bf16InputMxFp4Config<Config>;
    using Context = DeviceContext<RunnerConfig>;
    using Base = moe_test::TestRunnerBase;

    Bf16InputMxFp4Runner()
        : Base(moe_test::MakeTestRunnerConfig<RunnerConfig, Context>()),
          h_ctx_(std::make_unique<Context>()) {
        CheckHIPStatus(
            hipMalloc(reinterpret_cast<void **>(&d_ctx_), sizeof(Context)));
        h_ctx_accessor_ =
            std::make_unique<moe_test::DeviceContextAccessor<Context>>(
                h_ctx_.get());
        d_ctx_accessor_ =
            std::make_unique<moe_test::DeviceContextAccessor<Context>>(d_ctx_);
    }

    ~Bf16InputMxFp4Runner() override { CheckHIPStatus(hipFree(d_ctx_)); }

    void InitializeInputHostData(std::mt19937 &) override {
        GenerateNativeMxFp4Activations(h_ctx_.get());
    }

    void InitializeW13HostData() override {
        const auto &dataset = moe_test::MxFp4TestData<Config>::Get();
        std::copy(dataset.w1.begin(), dataset.w1.end(), std::begin(h_ctx_->w1));
        std::copy(dataset.scale_fc1.begin(), dataset.scale_fc1.end(),
                  std::begin(h_ctx_->scale_fc1));
    }

    void InitializeW2HostData() override {
        const auto &dataset = moe_test::MxFp4TestData<Config>::Get();
        std::copy(dataset.w2.begin(), dataset.w2.end(), std::begin(h_ctx_->w2));
        std::copy(dataset.scale_fc2.begin(), dataset.scale_fc2.end(),
                  std::begin(h_ctx_->scale_fc2));
    }

    void PrepareDequantizedActivations() override {}

    void DequantizeWeights() override {
        int err;
        for (unsigned expert = 0; expert < Context::kExperts; ++expert) {
            auto *dq_w13_expert =
                d_ctx_->dq_w13 + static_cast<size_t>(expert) * 2 *
                                     Context::kDim * Context::kInterDim;
            auto *w1_expert =
                d_ctx_->w1 + static_cast<size_t>(expert) * Context::kW1Rows *
                                  Context::kDim / sizeof(unsigned) / 2;
            auto *scale_fc1_expert =
                d_ctx_->scale_fc1 +
                static_cast<size_t>(expert) * Context::kW1Rows *
                    Context::kDim / Context::kMxScaleGroup;
            err = fp4::DequantMxFp4(
                reinterpret_cast<unsigned *>(dq_w13_expert), w1_expert,
                reinterpret_cast<const unsigned *>(scale_fc1_expert), 1.0f,
                quant::kDataTypeBf16, Context::kInterDim * 2, Context::kDim);
            ASSERT_EQ(err, 0) << "DequantMxFp4 failed";

            auto *dq_w2_expert =
                d_ctx_->dq_w2 + static_cast<size_t>(expert) *
                                    Context::kInterDim * Context::kDim;
            auto *w2_expert =
                d_ctx_->w2 + static_cast<size_t>(expert) *
                                  Context::kW2WordsPerExpert;
            auto *scale_fc2_expert =
                d_ctx_->scale_fc2 + static_cast<size_t>(expert) *
                                        Context::kDim * Context::kInterDim /
                                        Context::kMxScaleGroup;
            err = fp4::DequantMxFp4(
                reinterpret_cast<unsigned *>(dq_w2_expert), w2_expert,
                reinterpret_cast<const unsigned *>(scale_fc2_expert), 1.0f,
                quant::kDataTypeBf16, Context::kDim, Context::kInterDim);
            ASSERT_EQ(err, 0) << "DequantMxFp4 failed";
        }

        RepackWeights(d_ctx_->w1, std::size(d_ctx_->w1),
                      Context::kExperts * 2 * Context::kInterDim,
                      Context::kDim);
        RepackWeights(d_ctx_->w2, std::size(d_ctx_->w2),
                      Context::kExperts * Context::kDim, Context::kInterDim);
        RepackScales(d_ctx_->scale_fc1, std::size(d_ctx_->scale_fc1),
                     Context::kExperts * 2 * Context::kInterDim,
                     Context::kDim / Context::kMxScaleGroup);
        RepackScales(d_ctx_->scale_fc2, std::size(d_ctx_->scale_fc2),
                     Context::kExperts * Context::kDim,
                     Context::kInterDim / Context::kMxScaleGroup);
        CheckHIPStatus(moe_test::DequantizeNativeMxFp4Activations(
            d_ctx_->q_mx_act, d_ctx_->scale_mx_act, d_ctx_->dq_act,
            Context::kTokens, Context::kDim));
        CheckHIPStatus(hipMemcpy(h_ctx_->dq_act, d_ctx_->dq_act,
                                 sizeof(h_ctx_->dq_act),
                                 hipMemcpyDeviceToHost));
        CheckHIPStatus(hipDeviceSynchronize());
    }

  protected:
    moe_test::DeviceContextAccessorBase &HostAccessor() override {
        return *h_ctx_accessor_;
    }
    moe_test::DeviceContextAccessorBase &DeviceAccessor() override {
        return *d_ctx_accessor_;
    }
    void CopyHostToDeviceContext() override {
        CheckHIPStatus(hipMemcpy(d_ctx_, h_ctx_.get(), sizeof(Context),
                                 hipMemcpyHostToDevice));
    }
    void AdjustScalePatterns() override {}
    void AdjustTopKPatterns(std::vector<unsigned> &topk_ids,
                            std::vector<float> &topk_weights) override {
        Config::AdjustTopKPatterns(topk_ids, topk_weights);
    }
    int RunKernelImpl() override {
        FusedMoE1StageParams params{
            reinterpret_cast<unsigned *>(d_ctx_->out),
            reinterpret_cast<const unsigned *>(d_ctx_->dq_act),
            reinterpret_cast<const unsigned *>(d_ctx_->w1),
            reinterpret_cast<const unsigned *>(d_ctx_->w2),
            reinterpret_cast<const unsigned *>(d_ctx_->sorted_token_ids),
            reinterpret_cast<const unsigned *>(d_ctx_->sorted_weights),
            reinterpret_cast<const unsigned *>(d_ctx_->sorted_expert_ids),
            d_ctx_->num_valid_ids,
            Context::kTopK,
            nullptr,
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
        return FusedMoEMatmul1Stage(params, kBf16NativeMxFp4BiasSolutionId.Repr());
    }

  private:
    static void RepackWeights(unsigned *dst, size_t words, unsigned rows,
                              unsigned cols) {
        unsigned *baseline = nullptr;
        CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&baseline),
                                 words * sizeof(unsigned)));
        CheckHIPStatus(hipMemcpy(baseline, dst, words * sizeof(unsigned),
                                 hipMemcpyDeviceToDevice));
        CheckHIPStatus(
            moe_test::RepackNativeMxFp4Weights(dst, baseline, rows, cols));
        CheckHIPStatus(hipFree(baseline));
    }

    static void RepackScales(unsigned char *dst, size_t bytes, unsigned rows,
                             unsigned scale_cols) {
        unsigned *baseline = nullptr;
        CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&baseline), bytes));
        CheckHIPStatus(hipMemcpy(baseline, dst, bytes, hipMemcpyDeviceToDevice));
        CheckHIPStatus(moe_test::RepackNativeMxFp4Scales(
            reinterpret_cast<unsigned *>(dst), baseline, rows, scale_cols));
        CheckHIPStatus(hipFree(baseline));
    }

    std::unique_ptr<Context> h_ctx_;
    Context *d_ctx_ = nullptr;
    std::unique_ptr<moe_test::DeviceContextAccessor<Context>> h_ctx_accessor_;
    std::unique_ptr<moe_test::DeviceContextAccessor<Context>> d_ctx_accessor_;
};

template <class Config>
class NativeInputMxFp4Runner : public moe_test::TestRunnerBase {
  public:
    using RunnerConfig = NativeInputMxFp4Config<Config>;
    using Context = DeviceContext<RunnerConfig>;
    using Base = moe_test::TestRunnerBase;

    NativeInputMxFp4Runner()
        : Base(moe_test::MakeTestRunnerConfig<RunnerConfig, Context>()),
          h_ctx_(std::make_unique<Context>()) {
        CheckHIPStatus(
            hipMalloc(reinterpret_cast<void **>(&d_ctx_), sizeof(Context)));
        h_ctx_accessor_ =
            std::make_unique<moe_test::DeviceContextAccessor<Context>>(
                h_ctx_.get());
        d_ctx_accessor_ =
            std::make_unique<moe_test::DeviceContextAccessor<Context>>(d_ctx_);
    }

    ~NativeInputMxFp4Runner() override { CheckHIPStatus(hipFree(d_ctx_)); }

    void InitializeInputHostData(std::mt19937 &) override {
        GenerateNativeMxFp4Activations(h_ctx_.get());
    }

    void InitializeW13HostData() override {
        const auto &dataset = moe_test::MxFp4TestData<Config>::Get();
        std::copy(dataset.w1.begin(), dataset.w1.end(), std::begin(h_ctx_->w1));
        std::copy(dataset.scale_fc1.begin(), dataset.scale_fc1.end(),
                  std::begin(h_ctx_->scale_fc1));
    }

    void InitializeW2HostData() override {
        const auto &dataset = moe_test::MxFp4TestData<Config>::Get();
        std::copy(dataset.w2.begin(), dataset.w2.end(), std::begin(h_ctx_->w2));
        std::copy(dataset.scale_fc2.begin(), dataset.scale_fc2.end(),
                  std::begin(h_ctx_->scale_fc2));
    }

    void PrepareDequantizedActivations() override {}

    void DequantizeWeights() override {
        int err;
        for (unsigned expert = 0; expert < Context::kExperts; ++expert) {
            auto *dq_w13_expert =
                d_ctx_->dq_w13 + static_cast<size_t>(expert) * 2 *
                                     Context::kDim * Context::kInterDim;
            auto *w1_expert =
                d_ctx_->w1 + static_cast<size_t>(expert) * Context::kW1Rows *
                                  Context::kDim / sizeof(unsigned) / 2;
            auto *scale_fc1_expert =
                d_ctx_->scale_fc1 +
                static_cast<size_t>(expert) * Context::kW1Rows *
                    Context::kDim / Context::kMxScaleGroup;
            err = fp4::DequantMxFp4(
                reinterpret_cast<unsigned *>(dq_w13_expert), w1_expert,
                reinterpret_cast<const unsigned *>(scale_fc1_expert), 1.0f,
                quant::kDataTypeBf16, Context::kInterDim * 2, Context::kDim);
            ASSERT_EQ(err, 0) << "DequantMxFp4 failed";

            auto *dq_w2_expert =
                d_ctx_->dq_w2 + static_cast<size_t>(expert) *
                                    Context::kInterDim * Context::kDim;
            auto *w2_expert =
                d_ctx_->w2 + static_cast<size_t>(expert) *
                                  Context::kW2WordsPerExpert;
            auto *scale_fc2_expert =
                d_ctx_->scale_fc2 + static_cast<size_t>(expert) *
                                        Context::kDim * Context::kInterDim /
                                        Context::kMxScaleGroup;
            err = fp4::DequantMxFp4(
                reinterpret_cast<unsigned *>(dq_w2_expert), w2_expert,
                reinterpret_cast<const unsigned *>(scale_fc2_expert), 1.0f,
                quant::kDataTypeBf16, Context::kDim, Context::kInterDim);
            ASSERT_EQ(err, 0) << "DequantMxFp4 failed";
        }

        RepackWeights(d_ctx_->w1, std::size(d_ctx_->w1),
                      Context::kExperts * 2 * Context::kInterDim,
                      Context::kDim);
        CheckHIPStatus(hipDeviceSynchronize());
        RepackWeights(d_ctx_->w2, std::size(d_ctx_->w2),
                      Context::kExperts * Context::kDim, Context::kInterDim);
        CheckHIPStatus(hipDeviceSynchronize());
        RepackScales(d_ctx_->scale_fc1, std::size(d_ctx_->scale_fc1),
                     Context::kExperts * 2 * Context::kInterDim,
                     Context::kDim / Context::kMxScaleGroup);
        CheckHIPStatus(hipDeviceSynchronize());
        RepackScales(d_ctx_->scale_fc2, std::size(d_ctx_->scale_fc2),
                     Context::kExperts * Context::kDim,
                     Context::kInterDim / Context::kMxScaleGroup);
        CheckHIPStatus(hipDeviceSynchronize());
        CheckHIPStatus(moe_test::DequantizeNativeMxFp4Activations(
            d_ctx_->q_mx_act, d_ctx_->scale_mx_act, d_ctx_->dq_act,
            Context::kTokens, Context::kDim));
        CheckHIPStatus(hipMemcpy(h_ctx_->dq_act, d_ctx_->dq_act,
                                 sizeof(h_ctx_->dq_act),
                                 hipMemcpyDeviceToHost));
        CheckHIPStatus(hipDeviceSynchronize());
    }

  protected:
    moe_test::DeviceContextAccessorBase &HostAccessor() override {
        return *h_ctx_accessor_;
    }
    moe_test::DeviceContextAccessorBase &DeviceAccessor() override {
        return *d_ctx_accessor_;
    }
    void CopyHostToDeviceContext() override {
        CheckHIPStatus(hipMemcpy(d_ctx_, h_ctx_.get(), sizeof(Context),
                                 hipMemcpyHostToDevice));
        PackAiterSortedMxFp4ActivationScales(d_ctx_);
        CheckHIPStatus(hipDeviceSynchronize());
    }
    void AdjustScalePatterns() override {}
    void AdjustTopKPatterns(std::vector<unsigned> &topk_ids,
                            std::vector<float> &topk_weights) override {
        Config::AdjustTopKPatterns(topk_ids, topk_weights);
    }
    int RunKernelImpl() override {
        FusedMoE1StageParams params{
            reinterpret_cast<unsigned *>(d_ctx_->out),
            reinterpret_cast<const unsigned *>(d_ctx_->q_mx_act),
            reinterpret_cast<const unsigned *>(d_ctx_->w1),
            reinterpret_cast<const unsigned *>(d_ctx_->w2),
            reinterpret_cast<const unsigned *>(d_ctx_->sorted_token_ids),
            reinterpret_cast<const unsigned *>(d_ctx_->sorted_weights),
            reinterpret_cast<const unsigned *>(d_ctx_->sorted_expert_ids),
            d_ctx_->num_valid_ids,
            Context::kTopK,
            reinterpret_cast<const unsigned *>(d_ctx_->scale_mx_act_sorted),
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
        return FusedMoEMatmul1Stage(params,
                                    kMxFp4NativeMxFp4BiasSolutionId.Repr());
    }

  private:
    static void RepackWeights(unsigned *dst, size_t words, unsigned rows,
                              unsigned cols) {
        unsigned *baseline = nullptr;
        CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&baseline),
                                 words * sizeof(unsigned)));
        CheckHIPStatus(hipMemcpy(baseline, dst, words * sizeof(unsigned),
                                 hipMemcpyDeviceToDevice));
        CheckHIPStatus(
            moe_test::RepackNativeMxFp4Weights(dst, baseline, rows, cols));
        CheckHIPStatus(hipFree(baseline));
    }

    static void RepackScales(unsigned char *dst, size_t bytes, unsigned rows,
                             unsigned scale_cols) {
        unsigned *baseline = nullptr;
        CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&baseline), bytes));
        CheckHIPStatus(hipMemcpy(baseline, dst, bytes, hipMemcpyDeviceToDevice));
        CheckHIPStatus(moe_test::RepackNativeMxFp4Scales(
            reinterpret_cast<unsigned *>(dst), baseline, rows, scale_cols));
        CheckHIPStatus(hipFree(baseline));
    }

    std::unique_ptr<Context> h_ctx_;
    Context *d_ctx_ = nullptr;
    std::unique_ptr<moe_test::DeviceContextAccessor<Context>> h_ctx_accessor_;
    std::unique_ptr<moe_test::DeviceContextAccessor<Context>> d_ctx_accessor_;
};

class FusedMoEMxFp4Test : public ::testing::Test {
  public:
    template <class Config> void RunComparisonTest() {
        {
            SCOPED_TRACE("FP8 input MXFP4 MoE");
            Fp8InputMxFp4Runner<Config> runner;
            runner.Initialize();
            runner.RunTest();
        }
        {
            SCOPED_TRACE("FP8 input MXFP4 MoE with BF16 bias layout");
            Fp8InputMxFp4Runner<
                Config, Fp8InputMxFp4BiasConfig<Config>,
                kFp8PetitMxFp4BiasSolutionId.Repr()>
                runner;
            runner.Initialize();
            runner.RunTest();
        }
        {
            SCOPED_TRACE("BF16 input MXFP4 MoE");
            Bf16InputMxFp4Runner<Config> runner;
            runner.Initialize();
            runner.RunTest();
        }
        if (SupportsNativeScaleFp4()) {
            SCOPED_TRACE("Native MXFP4 input MXFP4 MoE");
            NativeInputMxFp4Runner<Config> runner;
            runner.Initialize();
            runner.RunTest();
        }
    }

  protected:
    static bool SupportsNativeScaleFp4() {
        int dev = 0;
        CheckHIPStatus(hipGetDevice(&dev));
        hipDeviceProp_t props;
        CheckHIPStatus(hipGetDeviceProperties(&props, dev));
        return std::strncmp(props.gcnArchName, "gfx950", 6) == 0;
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

TEST_F(FusedMoEMxFp4Test, GptOssHiddenMatchesPythonStyleReference) {
    RunComparisonTest<GptOssHiddenConfig>();
}

TEST_F(FusedMoEMxFp4Test, NativeMxFp4LayoutSensitiveMoEMatchesReference) {
    if (!SupportsNativeScaleFp4()) {
        GTEST_SKIP() << "native scaled FP4 MFMA requires gfx950";
    }

    NativeInputMxFp4Runner<NativeMxFp4LayoutSensitiveConfig> runner;
    runner.Initialize();
    runner.RunTest();
}

} // namespace
} // namespace causalflow::petit::rocm::moe
