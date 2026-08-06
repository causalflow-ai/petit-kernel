#include "moe/rocm/fused_moe_test_utils.h"
#include "tests/fp8_sampler.h"
#include "utils/hip_helper.h"
#include "utils/test_utils.h"

#include <hip/hip_fp8.h>

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace causalflow::petit::rocm::moe::test_utils {
namespace {

namespace fp8_sampler = causalflow::petit::tests::fp8_sampler;

void CheckHipblasStatus(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "HipBLASLt status: " << status << std::endl;
        throw std::runtime_error("HipBLASLt failure");
    }
}

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

float BitsAsFloat(unsigned u) {
    union {
        unsigned u;
        float f;
    } v{u};
    return v.f;
}

float Bf16BitsAsFloat(std::uint16_t u) {
    return BitsAsFloat(static_cast<unsigned>(u) << 16);
}

std::uint16_t FloatToBf16Bits(float x) {
    union {
        float f;
        unsigned u;
    } v{x};
    const unsigned lsb = (v.u >> 16) & 1u;
    v.u += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(v.u >> 16);
}

__hip_bfloat16 FloatAsBf16(float x) {
    union {
        __hip_bfloat16 bf16;
        std::uint16_t u;
    } v{};
    v.u = FloatToBf16Bits(x);
    return v.bf16;
}

template <class T> T *CopyToDevice(const std::vector<T> &host) {
    T *device = nullptr;
    CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&device),
                             host.size() * sizeof(T)));
    CheckHIPStatus(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                             hipMemcpyHostToDevice));
    return device;
}

struct ErrorStats {
    float abs_max;
    float rel_max;
};

ErrorStats ComputeErrorStats(const std::vector<float> &pred,
                             const std::vector<float> &ref, float atol,
                             float rtol) {
    EXPECT_EQ(pred.size(), ref.size());

    float abs_max = 0.0f;
    float rel_max = 0.0f;

    for (size_t i = 0; i < pred.size(); ++i) {
        const float ae = std::abs(pred[i] - ref[i]);
        const float denom = atol + rtol * std::abs(ref[i]);
        const float re = ae / std::max(denom, 1.0e-12f);
        abs_max = std::max(abs_max, ae);
        rel_max = std::max(rel_max, re);
    }

    return {
        .abs_max = abs_max,
        .rel_max = rel_max,
    };
}

void BuildSortedRoutingFromTopK(DeviceContextAccessorBase &ctx_accessor,
                                const std::vector<unsigned> &topk_ids,
                                const std::vector<float> &topk_weights,
                                const TestRunnerConfig &config) {
    const unsigned init_val = (config.topk << 24) | config.tokens;
    unsigned *sorted_token_ids = ctx_accessor.sorted_token_ids();
    float *sorted_weights = ctx_accessor.sorted_weights();
    unsigned *sorted_expert_ids = ctx_accessor.sorted_expert_ids();
    unsigned *num_valid_ids = ctx_accessor.num_valid_ids();

    EXPECT_EQ(static_cast<unsigned>(topk_ids.size()), config.routes);
    EXPECT_EQ(config.max_num_m_blocks,
              tal::CeilingDiv<unsigned>(config.max_num_tokens_padded,
                                        config.sorted_token_padding));

    std::fill_n(sorted_token_ids, config.max_num_tokens_padded, init_val);
    std::fill_n(sorted_weights, config.max_num_tokens_padded, 0.0f);
    std::fill_n(sorted_expert_ids, config.max_num_m_blocks,
                static_cast<unsigned>(-1));
    num_valid_ids[0] = 0;
    num_valid_ids[1] = 0;

    unsigned sorted_ids_begin = 0;
    unsigned sorted_expert_ids_begin = 0;
    for (unsigned expert = 0; expert < config.experts; ++expert) {
        unsigned tokens_num = 0;
        for (unsigned token = 0; token < config.tokens; ++token) {
            for (unsigned slot = 0; slot < config.topk; ++slot) {
                const size_t idx =
                    static_cast<size_t>(token) * config.topk + slot;
                if (topk_ids[idx] != expert) {
                    continue;
                }
                sorted_token_ids[sorted_ids_begin + tokens_num] =
                    (slot << 24) | token;
                sorted_weights[sorted_ids_begin + tokens_num] =
                    topk_weights[idx];
                ++tokens_num;
            }
        }
        const unsigned sorted_expert_ids_num =
            tal::CeilingDiv<unsigned>(tokens_num, config.sorted_token_padding);
        if (tokens_num == 0) {
            continue;
        }

        const unsigned tokens_num_pad =
            sorted_expert_ids_num * config.sorted_token_padding;
        for (unsigned i = 0; i < sorted_expert_ids_num; ++i) {
            sorted_expert_ids[sorted_expert_ids_begin + i] = expert;
        }
        sorted_ids_begin += tokens_num_pad;
        sorted_expert_ids_begin += sorted_expert_ids_num;
    }

    num_valid_ids[0] = sorted_ids_begin;
    num_valid_ids[1] = config.tokens;
}

void ComputeHipBlasLtReference(DeviceContextAccessorBase &host_ctx_accessor,
                               DeviceContextAccessorBase &device_ctx_accessor,
                               HipBlasLtRunner &gemm,
                               std::span<unsigned short> reference_out,
                               unsigned tokens, unsigned dim,
                               unsigned inter_dim, unsigned experts,
                               unsigned sorted_token_padding,
                               unsigned max_num_m_blocks,
                               TestRunnerConfig::ReferenceActivation
                                   reference_activation,
                               TestRunnerConfig::ReferenceIntermediate
                                   reference_intermediate) {
    std::fill(reference_out.begin(), reference_out.end(), 0u);
    const size_t token_out_bytes =
        static_cast<size_t>(tokens) * dim * sizeof(float);
    CheckHIPStatus(
        hipMemset(device_ctx_accessor.reference_acc(), 0, token_out_bytes));

    const unsigned *sorted_token_ids = host_ctx_accessor.sorted_token_ids();
    const float *sorted_weights = host_ctx_accessor.sorted_weights();
    const unsigned *sorted_expert_ids = host_ctx_accessor.sorted_expert_ids();
    const unsigned num_valid_ids = host_ctx_accessor.num_valid_ids()[0];
    const __hip_bfloat16 *host_dq_act = host_ctx_accessor.dq_act();
    unsigned *h_route_tokens = host_ctx_accessor.route_tokens();
    float *h_route_weights = host_ctx_accessor.route_weights();
    const unsigned num_valid_m_blocks =
        tal::CeilingDiv<unsigned>(num_valid_ids, sorted_token_padding);

    for (unsigned expert = 0; expert < experts; ++expert) {
        std::vector<unsigned> routes;
        routes.reserve(tokens);
        for (unsigned i = 0; i < num_valid_ids; ++i) {
            const unsigned route_group = i / sorted_token_padding;
            if (route_group >= num_valid_m_blocks ||
                route_group >= max_num_m_blocks ||
                sorted_expert_ids[route_group] != expert) {
                continue;
            }
            const unsigned token = sorted_token_ids[i] & 0x00ffffffu;
            if (token < tokens) {
                routes.push_back(i);
            }
        }
        if (routes.empty()) {
            continue;
        }

        const unsigned m_e = static_cast<unsigned>(routes.size());
        std::vector<__hip_bfloat16> h_a(static_cast<size_t>(m_e) * dim);
        for (unsigned i = 0; i < m_e; ++i) {
            const unsigned route = routes[i];
            const unsigned token = sorted_token_ids[route] & 0x00ffffffu;
            std::copy_n(host_dq_act + static_cast<size_t>(token) * dim, dim,
                        h_a.begin() + static_cast<size_t>(i) * dim);
            host_ctx_accessor.route_tokens()[i] = token;
            host_ctx_accessor.route_weights()[i] = sorted_weights[route];
        }

        CheckHIPStatus(hipMemcpy(device_ctx_accessor.a(), h_a.data(),
                                 h_a.size() * sizeof(__hip_bfloat16),
                                 hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(device_ctx_accessor.route_tokens(),
                                 h_route_tokens,
                                 static_cast<size_t>(m_e) * sizeof(unsigned),
                                 hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(
            device_ctx_accessor.route_weights(), h_route_weights,
            static_cast<size_t>(m_e) * sizeof(float), hipMemcpyHostToDevice));

        const __hip_bfloat16 *d_b_gate =
            device_ctx_accessor.dq_w13() +
            (static_cast<size_t>(expert) * 2 + 0) * dim * inter_dim;
        const __hip_bfloat16 *d_b_up =
            device_ctx_accessor.dq_w13() +
            (static_cast<size_t>(expert) * 2 + 1) * dim * inter_dim;
        const __hip_bfloat16 *d_b2 =
            device_ctx_accessor.dq_w2() +
            static_cast<size_t>(expert) * inter_dim * dim;

        const size_t inter_bytes =
            static_cast<size_t>(m_e) * inter_dim * sizeof(__hip_bfloat16);
        CheckHIPStatus(hipMemset(device_ctx_accessor.gate(), 0, inter_bytes));
        CheckHIPStatus(hipMemset(device_ctx_accessor.up(), 0, inter_bytes));
        CheckHIPStatus(hipMemset(device_ctx_accessor.act(), 0, inter_bytes));

        const bool use_silu_dot =
            reference_activation ==
            TestRunnerConfig::ReferenceActivation::kSiluDot;
        const unsigned elem_count = m_e * inter_dim;
        if (use_silu_dot) {
            auto run_stage1_gemm = [&](const __hip_bfloat16 *d_b,
                                       __hip_bfloat16 *d_c, bool swish) {
                if (swish) {
                    gemm.RunRowMajorGemmSwish(device_ctx_accessor.a(), d_b,
                                              d_c, m_e, inter_dim, dim);
                } else {
                    gemm.RunRowMajorGemm(device_ctx_accessor.a(), d_b, d_c,
                                         m_e, inter_dim, dim);
                }
            };
            run_stage1_gemm(d_b_gate, device_ctx_accessor.gate(), true);
            run_stage1_gemm(d_b_up, device_ctx_accessor.up(), false);
            CheckHIPStatus(ApplyElementwiseMultiply(
                device_ctx_accessor.gate(), device_ctx_accessor.up(),
                device_ctx_accessor.act(), elem_count));
        } else {
            float *d_gate = nullptr;
            float *d_up = nullptr;
            CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&d_gate),
                                     static_cast<size_t>(elem_count) *
                                         sizeof(float)));
            CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&d_up),
                                     static_cast<size_t>(elem_count) *
                                         sizeof(float)));
            gemm.RunRowMajorGemmToFloat(device_ctx_accessor.a(), d_b_gate,
                                        d_gate, m_e, inter_dim, dim);
            gemm.RunRowMajorGemmToFloat(device_ctx_accessor.a(), d_b_up, d_up,
                                        m_e, inter_dim, dim);
            CheckHIPStatus(ApplyOpenAISwiGLU(
                d_gate, d_up, device_ctx_accessor.act(), elem_count));
            CheckHIPStatus(hipFree(d_up));
            CheckHIPStatus(hipFree(d_gate));
        }
        if (reference_intermediate ==
            TestRunnerConfig::ReferenceIntermediate::kMxFp4) {
            CheckHIPStatus(QuantizeDequantMxFp4(device_ctx_accessor.act(), m_e,
                                                inter_dim));
        }

        const size_t route_bytes =
            static_cast<size_t>(m_e) * dim * sizeof(__hip_bfloat16);
        CheckHIPStatus(
            hipMemset(device_ctx_accessor.route_out(), 0, route_bytes));
        gemm.RunRowMajorGemm(device_ctx_accessor.act(), d_b2,
                             device_ctx_accessor.route_out(), m_e, dim,
                             inter_dim);

        CheckHIPStatus(ScatterWeightedRoutes(
            device_ctx_accessor.route_out(), device_ctx_accessor.route_tokens(),
            device_ctx_accessor.route_weights(),
            device_ctx_accessor.reference_acc(), m_e, dim));
    }

    CheckHIPStatus(hipMemcpy(host_ctx_accessor.reference_acc(),
                             device_ctx_accessor.reference_acc(),
                             static_cast<size_t>(tokens) * dim * sizeof(float),
                             hipMemcpyDeviceToHost));
    CheckHIPStatus(hipDeviceSynchronize());
    for (size_t i = 0; i < reference_out.size(); ++i) {
        reference_out[i] =
            FloatToBf16Bits(host_ctx_accessor.reference_acc()[i]);
    }
}

} // namespace

HipBlasLtRunner::HipBlasLtRunner() {
    static constexpr hipblasOperation_t kTransposed = HIPBLAS_OP_T;
    static constexpr hipblasLtEpilogue_t kDefaultEpilogue =
        HIPBLASLT_EPILOGUE_DEFAULT;
    static constexpr hipblasLtEpilogue_t kSwishEpilogue =
        HIPBLASLT_EPILOGUE_SWISH_EXT;

    CheckHipblasStatus(hipblasLtCreate(&handle_));

    CheckHipblasStatus(hipblasLtMatmulDescCreate(
        &default_desc_, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CheckHipblasStatus(hipblasLtMatmulDescSetAttribute(
        default_desc_, HIPBLASLT_MATMUL_DESC_TRANSA, &kTransposed,
        sizeof(kTransposed)));
    CheckHipblasStatus(hipblasLtMatmulDescSetAttribute(
        default_desc_, HIPBLASLT_MATMUL_DESC_EPILOGUE, &kDefaultEpilogue,
        sizeof(kDefaultEpilogue)));

    CheckHipblasStatus(hipblasLtMatmulDescCreate(
        &swish_desc_, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CheckHipblasStatus(hipblasLtMatmulDescSetAttribute(
        swish_desc_, HIPBLASLT_MATMUL_DESC_TRANSA, &kTransposed,
        sizeof(kTransposed)));
    CheckHipblasStatus(hipblasLtMatmulDescSetAttribute(
        swish_desc_, HIPBLASLT_MATMUL_DESC_EPILOGUE, &kSwishEpilogue,
        sizeof(kSwishEpilogue)));

    CheckHIPStatus(hipMalloc(&workspace_, kWorkspaceSize));
}

HipBlasLtRunner::~HipBlasLtRunner() {
    if (workspace_ != nullptr) {
        CheckHIPStatus(hipFree(workspace_));
    }
    if (default_desc_ != nullptr) {
        CheckHipblasStatus(hipblasLtMatmulDescDestroy(default_desc_));
    }
    if (swish_desc_ != nullptr) {
        CheckHipblasStatus(hipblasLtMatmulDescDestroy(swish_desc_));
    }
    if (handle_ != nullptr) {
        CheckHipblasStatus(hipblasLtDestroy(handle_));
    }
}

void HipBlasLtRunner::RunRowMajorGemm(const __hip_bfloat16 *d_a,
                                      const __hip_bfloat16 *d_b,
                                      __hip_bfloat16 *d_c, unsigned m,
                                      unsigned n, unsigned k) const {
    RunRowMajorGemmWithDesc(default_desc_, d_a, d_b, d_c, m, n, k, 0.0f);
}

void HipBlasLtRunner::RunRowMajorGemmToFloat(const __hip_bfloat16 *d_a,
                                             const __hip_bfloat16 *d_b,
                                             float *d_c, unsigned m,
                                             unsigned n, unsigned k) const {
    RunRowMajorGemmToFloatWithDesc(default_desc_, d_a, d_b, d_c, m, n, k);
}

void HipBlasLtRunner::RunRowMajorGemmAccumulate(const __hip_bfloat16 *d_a,
                                                const __hip_bfloat16 *d_b,
                                                __hip_bfloat16 *d_c, unsigned m,
                                                unsigned n, unsigned k) const {
    RunRowMajorGemmWithDesc(default_desc_, d_a, d_b, d_c, m, n, k, 1.0f);
}

void HipBlasLtRunner::RunRowMajorGemmSwish(const __hip_bfloat16 *d_a,
                                           const __hip_bfloat16 *d_b,
                                           __hip_bfloat16 *d_c, unsigned m,
                                           unsigned n, unsigned k) const {
    RunRowMajorGemmWithDesc(swish_desc_, d_a, d_b, d_c, m, n, k, 0.0f);
}

void HipBlasLtRunner::RunRowMajorGemmWithDesc(hipblasLtMatmulDesc_t desc,
                                              const __hip_bfloat16 *d_a,
                                              const __hip_bfloat16 *d_b,
                                              __hip_bfloat16 *d_c, unsigned m,
                                              unsigned n, unsigned k,
                                              float beta) const {
    static constexpr float kAlpha = 1.0f;

    hipblasLtMatrixLayout_t layout_a = nullptr, layout_b = nullptr,
                            layout_c = nullptr;
    CheckHipblasStatus(
        hipblasLtMatrixLayoutCreate(&layout_a, HIP_R_16BF, k, n, k));
    CheckHipblasStatus(
        hipblasLtMatrixLayoutCreate(&layout_b, HIP_R_16BF, k, m, k));
    CheckHipblasStatus(
        hipblasLtMatrixLayoutCreate(&layout_c, HIP_R_16BF, n, m, n));

    CheckHipblasStatus(hipblasLtMatmul(
        handle_, desc, &kAlpha, d_b, layout_a, d_a, layout_b, &beta, d_c,
        layout_c, d_c, layout_c, nullptr, workspace_, kWorkspaceSize, nullptr));

    CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_a));
    CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_b));
    CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_c));
}

void HipBlasLtRunner::RunRowMajorGemmToFloatWithDesc(
    hipblasLtMatmulDesc_t desc, const __hip_bfloat16 *d_a,
    const __hip_bfloat16 *d_b, float *d_c, unsigned m, unsigned n,
    unsigned k) const {
    static constexpr float kAlpha = 1.0f;
    static constexpr float kBeta = 0.0f;

    hipblasLtMatrixLayout_t layout_a = nullptr, layout_b = nullptr,
                            layout_c = nullptr;
    CheckHipblasStatus(
        hipblasLtMatrixLayoutCreate(&layout_a, HIP_R_16BF, k, n, k));
    CheckHipblasStatus(
        hipblasLtMatrixLayoutCreate(&layout_b, HIP_R_16BF, k, m, k));
    CheckHipblasStatus(
        hipblasLtMatrixLayoutCreate(&layout_c, HIP_R_32F, n, m, n));

    CheckHipblasStatus(hipblasLtMatmul(
        handle_, desc, &kAlpha, d_b, layout_a, d_a, layout_b, &kBeta, d_c,
        layout_c, d_c, layout_c, nullptr, workspace_, kWorkspaceSize, nullptr));

    CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_a));
    CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_b));
    CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_c));
}

TestRunnerBase::TestRunnerBase(TestRunnerConfig config)
    : config_(config), fp8_format_(CurrentDeviceFp8Format()) {
    reference_out_.resize(static_cast<size_t>(config_.tokens) * config_.dim,
                          0u);
}

TestRunnerBase::~TestRunnerBase() = default;

float TestRunnerBase::SampleScale(std::mt19937 &gen) const {
    std::normal_distribution<float> scale_dist(config_.scale_inv_mean,
                                               config_.scale_inv_std);
    return std::max(1e-8f, scale_dist(gen));
}

void TestRunnerBase::Initialize() {
    InitializeHostData();
    CopyHostToDeviceContext();
    DequantizeWeights();
}

void TestRunnerBase::InitializeHostData() {
    auto &ctx_accessor = HostAccessor();
    std::mt19937 gen(kSeed);
    std::normal_distribution<float> rw_logit_dist(0.0f, 4.0f);
    std::uniform_int_distribution<unsigned> expert_dist(0, config_.experts - 1);

    InitializeInputHostData(gen);
    InitializeW13HostData();
    InitializeW2HostData();
    AdjustScalePatterns();

    std::vector<unsigned> topk_ids(config_.routes);
    std::vector<float> topk_weights(config_.routes);
    for (unsigned token = 0; token < config_.tokens; ++token) {
        float max_logit = -INFINITY;
        for (unsigned slot = 0; slot < config_.topk; ++slot) {
            const size_t idx = static_cast<size_t>(token) * config_.topk + slot;
            topk_ids[idx] = expert_dist(gen);
            topk_weights[idx] = rw_logit_dist(gen);
            max_logit = std::max(max_logit, topk_weights[idx]);
        }
        float sum = 0.0f;
        for (unsigned slot = 0; slot < config_.topk; ++slot) {
            const size_t idx = static_cast<size_t>(token) * config_.topk + slot;
            topk_weights[idx] = expf(topk_weights[idx] - max_logit);
            sum += topk_weights[idx];
        }
        sum = std::max(sum, 1.0e-12f);
        for (unsigned slot = 0; slot < config_.topk; ++slot) {
            const size_t idx = static_cast<size_t>(token) * config_.topk + slot;
            topk_weights[idx] /= sum;
        }
    }
    AdjustTopKPatterns(topk_ids, topk_weights);

    BuildSortedRoutingFromTopK(ctx_accessor, topk_ids, topk_weights, config_);
    std::fill_n(ctx_accessor.out(),
                static_cast<size_t>(config_.tokens) * config_.dim, 0);
}

void TestRunnerBase::InitializeInputHostData(std::mt19937 &gen) {
    auto &ctx_accessor = HostAccessor();
    fp8_sampler::FP8E4M3QuantizedNormalSampler input_sampler(
        kInputMean, kInputStd, fp8_format_);
    auto gen_input4 = [&](size_t idx) {
        float4 values;
        for (int i = 0; i < 4; ++i) {
            const size_t element_idx = idx + i;
            FixedU32Rng base_rng{
                static_cast<uint32_t>(IndexedSeed(kSeed, 0x11du, element_idx))};
            unsigned char value = input_sampler(base_rng);
            float x =
                fp8_sampler::FP8E4M3DiscreteSampler::Decode(value, fp8_format_);
            const float spike_prob = U32ToOpenUnitFloat(
                static_cast<uint32_t>(IndexedSeed(kSeed, 0x3c7u, element_idx)));
            if (spike_prob < kInputSpikeProb) {
                x += kInputSpikeStd * HashToStandardNormal(IndexedSeed(
                                          kSeed, 0x4d9u, element_idx));
            }
            reinterpret_cast<float *>(&values)[i] = x;
        }
        if (fp8_format_ == fp8_sampler::FP8E4M3Format::kOcp) {
            return __hip_fp8x4_e4m3(values).__x;
        }
        return __hip_fp8x4_e4m3_fnuz(values).__x;
    };
    auto gen_scale = [&]() { return SampleScale(gen); };

    FillParallelIndexed4(
        std::span(ctx_accessor.q_act(),
                  static_cast<size_t>(config_.tokens) * config_.dim),
        gen_input4);
    FillRandomValue(gen_scale,
                    std::span(ctx_accessor.scale_act_t(),
                              static_cast<size_t>(config_.tokens) *
                                  (config_.dim / config_.act_scale_group)));
}

void TestRunnerBase::PrepareDequantizedActivations() {
    auto &ctx_accessor = HostAccessor();
    __hip_bfloat16 *dq_act = ctx_accessor.dq_act();
    unsigned char *q_act = ctx_accessor.q_act();
    float *scale_act_t = ctx_accessor.scale_act_t();
    for (unsigned token = 0; token < config_.tokens; ++token) {
        for (unsigned col = 0; col < config_.dim; ++col) {
            const float s =
                scale_act_t[(col / config_.act_scale_group) * config_.tokens +
                            token];
            if (fp8_format_ == fp8_sampler::FP8E4M3Format::kOcp) {
                __hip_fp8_e4m3 value;
                value.__x = q_act[token * config_.dim + col];
                dq_act[token * config_.dim + col] =
                    FloatAsBf16(static_cast<float>(value) * s);
            } else {
                __hip_fp8_e4m3_fnuz value;
                value.__x = q_act[token * config_.dim + col];
                dq_act[token * config_.dim + col] =
                    FloatAsBf16(static_cast<float>(value) * s);
            }
        }
    }
}

void TestRunnerBase::ComputeReferences() {
    ComputeHipBlasLtReference(
        HostAccessor(), DeviceAccessor(), gemm_, std::span(reference_out_),
        config_.tokens, config_.dim, config_.inter_dim, config_.experts,
        config_.sorted_token_padding, config_.max_num_m_blocks,
        config_.reference_activation, config_.reference_intermediate);
}

void TestRunnerBase::RunReferenceOnly() {
    PrepareDequantizedActivations();
    ComputeReferences();
}

void TestRunnerBase::RunKernel() {
    CheckHIPStatus(hipMemset(DeviceAccessor().out(), 0,
                             static_cast<size_t>(config_.tokens) * config_.dim *
                                 sizeof(unsigned short)));
    const int err = RunKernelImpl();
    ASSERT_EQ(err, 0) << "Kernel launch failed with err=" << err;
    CheckHIPStatus(hipDeviceSynchronize());
}

void TestRunnerBase::RunTest() {
    RunReferenceOnly();
    RunKernel();

    std::vector<unsigned short> actual_bits(
        static_cast<size_t>(config_.tokens) * config_.dim);
    CheckHIPStatus(hipMemcpy(actual_bits.data(), DeviceAccessor().out(),
                             actual_bits.size() * sizeof(unsigned short),
                             hipMemcpyDeviceToHost));
    CheckHIPStatus(hipDeviceSynchronize());

    std::vector<float> expected(actual_bits.size());
    std::vector<float> actual(actual_bits.size());
    for (size_t i = 0; i < actual_bits.size(); ++i) {
        expected[i] = Bf16BitsAsFloat(reference_out_[i]);
        actual[i] = Bf16BitsAsFloat(actual_bits[i]);
    }

    const float per_element_atol =
        fp8_format_ == fp8_sampler::FP8E4M3Format::kOcp
            ? config_.ocp_fp8_per_element_atol
            : config_.per_element_atol;
    const ErrorStats stats = ComputeErrorStats(
        actual, expected, per_element_atol, config_.per_element_rtol);
    if (stats.rel_max >= 1.0f) {
        struct Mismatch {
            size_t idx;
            float ae;
            float re;
        };
        std::vector<Mismatch> mm;
        mm.reserve(actual.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            const float ae = std::abs(actual[i] - expected[i]);
            const float denom = per_element_atol + config_.per_element_rtol *
                                                       std::abs(expected[i]);
            const float re = ae / std::max(denom, 1.0e-12f);
            mm.push_back({i, ae, re});
        }
        std::partial_sort(
            mm.begin(), mm.begin() + std::min<size_t>(8, mm.size()), mm.end(),
            [](const Mismatch &a, const Mismatch &b) { return a.re > b.re; });
        std::cerr << "Top mismatches (by normalized err):\n";
        for (size_t j = 0; j < std::min<size_t>(8, mm.size()); ++j) {
            const size_t idx = mm[j].idx;
            const unsigned token = idx / config_.dim;
            const unsigned col = idx % config_.dim;
            std::cerr << "  idx=" << idx << " token=" << token << " col=" << col
                      << " expected=" << expected[idx]
                      << " actual=" << actual[idx] << " abs=" << mm[j].ae
                      << " rel_norm=" << mm[j].re << "\n";
        }
    }
    EXPECT_LT(stats.rel_max, 1.0f)
        << "normalized relative max too large: " << stats.rel_max
        << " (atol=" << per_element_atol
        << ", rtol=" << config_.per_element_rtol
        << ", abs_max=" << stats.abs_max << ")";
}

} // namespace causalflow::petit::rocm::moe::test_utils
