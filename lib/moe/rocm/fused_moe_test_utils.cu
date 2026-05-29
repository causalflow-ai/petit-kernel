#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/amd_intrinsics.cuh"
#include "gemm/rocm/quantization/fp4/quantization_utils.cuh"
#include "gemm/rocm/quantization/types.h"
#include "moe/rocm/fused_moe_test_utils.h"
#include "tests/fp8_sampler.h"
#include "utils/hip_helper.h"
#include "utils/test_utils.h"

namespace causalflow::petit::rocm::moe::test_utils {
namespace {

void CheckHipblasStatus(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "HipBLASLt status: " << status << std::endl;
        throw std::runtime_error("HipBLASLt failure");
    }
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

__host__ __device__ std::uint16_t FloatToBf16Bits(float x) {
    union {
        float f;
        unsigned u;
    } v{x};
    const unsigned lsb = (v.u >> 16) & 1u;
    v.u += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(v.u >> 16);
}

__host__ __device__ __hip_bfloat16 FloatAsBf16(float x) {
    union {
        __hip_bfloat16 bf16;
        std::uint16_t u;
    } v{};
    v.u = FloatToBf16Bits(x);
    return v.bf16;
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

} // namespace

__global__ void ElementwiseMultiplyKernel(const __hip_bfloat16 *a,
                                          const __hip_bfloat16 *b,
                                          __hip_bfloat16 *out, unsigned count) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] =
            FloatAsBf16(__bfloat162float(a[idx]) * __bfloat162float(b[idx]));
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
        __hip_fp8x4_e4m3_fnuz packed;
        packed.__x = shm[idx];
        const float4 out_fp4 = static_cast<float4>(packed);
        v4f out = reinterpret_cast<const v4f &>(out_fp4);
        out *= tile_scale4;
        o[output_layout(idx)] = Store::Convert(out);
    }
}

// The layout reshuffle a 64(n)x128(k) block to (4(n), 16(k), 16(n), 8(k)). The
// each 8 elements are permuted according to the Petit layout for fast
// dequantization. The scale is packed as 4-byte unsigned along the K dimension
// of the 64x128 block.
struct DequantTraitMxFp4Group4 {
    static constexpr unsigned kNumWarps = 4;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;
    static constexpr unsigned kRowGroupSize = 32;
    static constexpr unsigned kGroupK = 128;
    static constexpr unsigned kGroupN = 64;
    static constexpr unsigned kIsNativeQWFormat = false;
    static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(unsigned);

    using UDQ = quantization::UnifiedDequantizerForMxFp4Bf16<true>;
    using Scale = __hip_bfloat16;

    __device__ static auto GetFetchQWLayout(unsigned size_k, unsigned size_n) {
        using namespace causalflow::tal;
        using namespace quantization::fp4;
        using Shape = Shape<_1, _1, C<kWarpSize>, C<kNumWarps>>;
        auto stride = make_shape(
            C<kWarpSize>{}, size_k * kGroupN / kPackFactor / kVecSize, _1{},
            kGroupN / kNumWarps / kPackFactor / kVecSize);
        return make_layout(Shape{}, stride);
    }

    __device__ static auto GetFetchScaleLayout(unsigned size_k,
                                               unsigned size_n) {
        using namespace causalflow::tal;
        return make_layout(
            Shape<_1, _1, C<kWarpSize>>{},
            make_stride(size_k / kRowGroupSize * 4, C<kWarpSize>{}, _1{}));
    }

    __device__ static uint2 GetScale(unsigned *shm_scales, unsigned tid) {
        uint2 v;
        auto *u = reinterpret_cast<__hip_bfloat162 *>(&v);
        unsigned wid = tid / kWarpSize, wtid = tid % kWarpSize;
        unsigned s = shm_scales[wid * 16 + wtid % 16];
        u[0] = UDQ::DequantScales(s & 0xffff);
        u[1] = UDQ::DequantScales(s >> 16);
        return v;
    }

    __device__ static auto GetOutputLayout(unsigned size_k, unsigned size_n) {
        using namespace causalflow::tal;
        using namespace quantization::fp4;
        static constexpr unsigned kOutU128 = (sizeof(uint4) / sizeof(uint)) *
                                             kPackFactor * sizeof(half) /
                                             sizeof(uint4);
        static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(half);
        auto stride_out =
            make_stride(C<kGroupK / kVecSize>{}, size_k * kGroupN / kVecSize,
                        make_stride(make_stride(size_k / kVecSize, _1{}),
                                    size_k * 16 / kVecSize),
                        C<32 * sizeof(half) / sizeof(uint4)>{});
        using Shape =
            Shape<_1, _1, Shape<Shape<_16, _4>, C<kNumWarps>>, C<kOutU128>>;
        auto layout_out = make_layout(Shape{}, stride_out);
        return layout_out;
    }
};

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

void ComputeHipBlasLtReference(DeviceContextAccessorBase &host_ctx_accessor,
                               DeviceContextAccessorBase &device_ctx_accessor,
                               HipBlasLtRunner &gemm,
                               std::span<unsigned short> reference_out,
                               unsigned tokens, unsigned dim,
                               unsigned inter_dim, unsigned experts,
                               unsigned sorted_token_padding,
                               unsigned max_num_m_blocks) {
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
        gemm.RunRowMajorGemmSwish(device_ctx_accessor.a(), d_b_gate,
                                  device_ctx_accessor.gate(), m_e, inter_dim,
                                  dim);
        gemm.RunRowMajorGemm(device_ctx_accessor.a(), d_b_up,
                             device_ctx_accessor.up(), m_e, inter_dim, dim);

        const unsigned elem_count = m_e * inter_dim;
        {
            static constexpr unsigned kThreads = 256;
            const dim3 block(kThreads);
            const dim3 grid(tal::CeilingDiv<unsigned>(elem_count, block.x));
            ElementwiseMultiplyKernel<<<grid, block>>>(
                device_ctx_accessor.gate(), device_ctx_accessor.up(),
                device_ctx_accessor.act(), elem_count);
            CheckHIPStatus(hipGetLastError());
        }

        const size_t route_bytes =
            static_cast<size_t>(m_e) * dim * sizeof(__hip_bfloat16);
        CheckHIPStatus(
            hipMemset(device_ctx_accessor.route_out(), 0, route_bytes));
        gemm.RunRowMajorGemm(device_ctx_accessor.act(), d_b2,
                             device_ctx_accessor.route_out(), m_e, dim,
                             inter_dim);

        {
            static constexpr unsigned kThreads = 256;
            const dim3 block(kThreads);
            const dim3 grid(tal::CeilingDiv<unsigned>(m_e * dim, block.x));
            WeightedRouteScatterKernel<<<grid, block>>>(
                device_ctx_accessor.route_out(),
                device_ctx_accessor.route_tokens(),
                device_ctx_accessor.route_weights(),
                device_ctx_accessor.reference_acc(), m_e, dim);
            CheckHIPStatus(hipGetLastError());
        }
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

TestRunnerBase::TestRunnerBase(TestRunnerConfig config) : config_(config) {
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

    causalflow::petit::tests::fp8_sampler::FP8E4M3QuantizedNormalSampler
        input_sampler(kInputMean, kInputStd);
    auto gen_input4 = [&](size_t idx) {
        float4 values;
        for (int i = 0; i < 4; ++i) {
            const size_t element_idx = idx + i;
            FixedU32Rng base_rng{
                static_cast<uint32_t>(IndexedSeed(kSeed, 0x11du, element_idx))};
            unsigned char value = input_sampler(base_rng);
            float x = causalflow::petit::tests::fp8_sampler::
                FP8E4M3DiscreteSampler::Decode(value);
            const float spike_prob = U32ToOpenUnitFloat(
                static_cast<uint32_t>(IndexedSeed(kSeed, 0x3c7u, element_idx)));
            if (spike_prob < kInputSpikeProb) {
                x += kInputSpikeStd * HashToStandardNormal(IndexedSeed(
                                          kSeed, 0x4d9u, element_idx));
            }
            reinterpret_cast<float *>(&values)[i] = x;
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
            __hip_fp8_e4m3_fnuz value;
            value.__x = q_act[token * config_.dim + col];
            dq_act[token * config_.dim + col] =
                FloatAsBf16(static_cast<float>(value) * s);
        }
    }
}

void TestRunnerBase::ComputeReferences() {
    ComputeHipBlasLtReference(
        HostAccessor(), DeviceAccessor(), gemm_, std::span(reference_out_),
        config_.tokens, config_.dim, config_.inter_dim, config_.experts,
        config_.sorted_token_padding, config_.max_num_m_blocks);
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

    const ErrorStats stats = ComputeErrorStats(
        actual, expected, config_.per_element_atol, config_.per_element_rtol);
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
            const float denom =
                config_.per_element_atol +
                config_.per_element_rtol * std::abs(expected[i]);
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
        << " (atol=" << config_.per_element_atol
        << ", rtol=" << config_.per_element_rtol
        << ", abs_max=" << stats.abs_max << ")";
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

int DequantMxFp4WithGroup4(unsigned *output, const unsigned *input,
                           const unsigned *scales, float global_scale,
                           quantization::DataType out_type, unsigned m,
                           unsigned n, hipStream_t stream) {
    namespace fp4 = causalflow::petit::rocm::quantization::fp4;
    using Trait = DequantTraitMxFp4Group4;

    if (out_type != quantization::kDataTypeBf16) {
        return -1;
    }
    // The dequantized output is materialized as a row-major [m, n] matrix.
    if (m % Trait::kGroupK != 0 || n % Trait::kGroupN != 0) {
        return -1;
    }

    dim3 grid(m / Trait::kGroupK, n / Trait::kGroupN);
    dim3 block(Trait::kThreads);
    global_scale *= Trait::UDQ::GlobalScaleFactor();
    fp4::DequantizeFp4Kernel<Trait><<<grid, block, 0, stream>>>(
        reinterpret_cast<uint4 *>(output),
        reinterpret_cast<const uint4 *>(input),
        reinterpret_cast<const unsigned char *>(scales), global_scale, m, n);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

} // namespace causalflow::petit::rocm::moe::test_utils
