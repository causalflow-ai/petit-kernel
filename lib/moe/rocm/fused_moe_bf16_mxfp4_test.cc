#include "gemm/rocm/quantization/fp4/gemm_fp4.h"
#include "moe/rocm/fused_moe.h"
#include "moe/rocm/fused_moe_test_utils.h"
#include "utils/hip_helper.h"

#include <gtest/gtest.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <span>
#include <vector>

namespace causalflow::petit::rocm::moe {

namespace {

namespace moe_test = causalflow::petit::rocm::moe::test_utils;

static constexpr FusedMoESolutionId kBf16NativeMxFp4BiasSolutionId =
    FusedMoESolutionId::Make(
        FusedMoEDataType::kBf16, FusedMoEDataType::kMxFp4,
        FusedMoEDataType::kBf16, FusedMoEWeightOrdering::kNativeMxFp4,
        FusedMoEMfmaShape::kMfmaBf16MxFp4, FusedMoEStages::kOneStage,
        FusedMoEActivationFunction::kOpenAISwiGLU,
        FusedMoEStage1Buffering::kDoubleBuffer);

__host__ __device__ std::uint16_t FloatToBf16Bits(float x) {
    union {
        float f;
        unsigned u;
    } v{x};
    const unsigned lsb = (v.u >> 16) & 1u;
    v.u += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(v.u >> 16);
}

__host__ __device__ float Bf16BitsToFloat(std::uint16_t u) {
    union {
        unsigned u;
        float f;
    } v{static_cast<unsigned>(u) << 16};
    return v.f;
}

float Bf16ToFloat(__hip_bfloat16 v) {
    std::uint16_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return Bf16BitsToFloat(bits);
}

__host__ __device__ __hip_bfloat16 FloatAsBf16(float x) {
    union {
        __hip_bfloat16 bf16;
        std::uint16_t u;
    } v{};
    v.u = FloatToBf16Bits(x);
    return v.bf16;
}

void SetBaselineMxFp4(std::vector<unsigned char> &q, unsigned rows,
                      unsigned cols, unsigned row, unsigned col,
                      unsigned nibble) {
    if (row >= rows || col >= cols || nibble >= 16) {
        throw std::runtime_error("invalid baseline MXFP4 coordinate");
    }
    const size_t idx = static_cast<size_t>(row) * (cols / 2) + col / 2;
    if ((col & 1u) == 0) {
        q[idx] = static_cast<unsigned char>((q[idx] & 0xF0u) | nibble);
    } else {
        q[idx] = static_cast<unsigned char>((q[idx] & 0x0Fu) | (nibble << 4));
    }
}

int RunBf16MxFp4(uint4 *__restrict__ out, const uint4 *act,
                   const uint4 *w13, const uint4 *w2,
                   const uint4 *sorted_token_ids,
                   const uint4 *sorted_weights,
                   const uint4 *sorted_expert_ids,
                   const unsigned *num_valid_ids, unsigned topk,
                   const uint4 *scales_w13, const unsigned *scales_w2,
                   unsigned max_num_m_blocks, unsigned m, unsigned n,
                   unsigned k, unsigned num_experts,
                   const void *w13_bias = nullptr,
                   const void *w2_bias = nullptr) {
    FusedMoE1StageParams params{
        reinterpret_cast<unsigned *>(out),
        reinterpret_cast<const unsigned *>(act),
        reinterpret_cast<const unsigned *>(w13),
        reinterpret_cast<const unsigned *>(w2),
        reinterpret_cast<const unsigned *>(sorted_token_ids),
        reinterpret_cast<const unsigned *>(sorted_weights),
        reinterpret_cast<const unsigned *>(sorted_expert_ids),
        num_valid_ids,
        topk,
        nullptr,
        reinterpret_cast<const unsigned *>(scales_w13),
        scales_w2,
        max_num_m_blocks,
        m,
        n,
        k,
        num_experts,
        nullptr,
        0,
        w13_bias,
        w2_bias,
    };
    return FusedMoEMatmul1Stage(params, kBf16NativeMxFp4BiasSolutionId.Repr());
}

template <class T> T *CopyToDevice(std::span<const T> host) {
    T *device = nullptr;
    CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&device),
                             host.size() * sizeof(T)));
    CheckHIPStatus(hipMemcpy(device, host.data(), host.size_bytes(),
                             hipMemcpyHostToDevice));
    return device;
}

template <class T> T *CopyToDevice(const std::vector<T> &host) {
    return CopyToDevice(std::span<const T>(host));
}

template <class... T> void FreeDevice(T *...ptrs) {
    (CheckHIPStatus(hipFree(ptrs)), ...);
}

void ApplyOpenAISwiGLU(const std::vector<__hip_bfloat16> &gate,
                       const std::vector<__hip_bfloat16> &up,
                       std::vector<__hip_bfloat16> &out) {
    ASSERT_EQ(gate.size(), up.size());
    ASSERT_EQ(gate.size(), out.size());
    auto *d_gate = CopyToDevice(gate);
    auto *d_up = CopyToDevice(up);
    auto *d_out = CopyToDevice(out);
    CheckHIPStatus(moe_test::ApplyOpenAISwiGLU(
        d_gate, d_up, d_out, static_cast<unsigned>(out.size())));
    CheckHIPStatus(hipDeviceSynchronize());
    CheckHIPStatus(hipMemcpy(out.data(), d_out, out.size() * sizeof(out[0]),
                             hipMemcpyDeviceToHost));
    FreeDevice(d_out, d_up, d_gate);
}

std::span<const unsigned> Words(const std::vector<unsigned char> &bytes) {
    if (bytes.size() % sizeof(unsigned) != 0) {
        throw std::runtime_error("byte size is not word-aligned");
    }
    return {reinterpret_cast<const unsigned *>(bytes.data()),
            bytes.size() / sizeof(unsigned)};
}

struct NativeMxFp4DeviceLayout {
    unsigned *w13 = nullptr;
    unsigned *w2 = nullptr;
    unsigned *w13_scale = nullptr;
    unsigned *w2_scale = nullptr;
    unsigned *w13_baseline = nullptr;
    unsigned *w2_baseline = nullptr;
    unsigned *w13_scale_baseline = nullptr;
    unsigned *w2_scale_baseline = nullptr;

    NativeMxFp4DeviceLayout(std::span<const unsigned> w13_baseline_host,
                            std::span<const unsigned> w2_baseline_host,
                            std::span<const unsigned> w13_scale_baseline_host,
                            std::span<const unsigned> w2_scale_baseline_host,
                            unsigned dim, unsigned inter_dim,
                            unsigned experts = 1) {
        w13_baseline = CopyToDevice(w13_baseline_host);
        w2_baseline = CopyToDevice(w2_baseline_host);
        w13_scale_baseline = CopyToDevice(w13_scale_baseline_host);
        w2_scale_baseline = CopyToDevice(w2_scale_baseline_host);

        CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&w13),
                                 w13_baseline_host.size() * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&w2),
                                 w2_baseline_host.size() * sizeof(unsigned)));
        CheckHIPStatus(
            hipMalloc(reinterpret_cast<void **>(&w13_scale),
                      w13_scale_baseline_host.size() * sizeof(unsigned)));
        CheckHIPStatus(
            hipMalloc(reinterpret_cast<void **>(&w2_scale),
                      w2_scale_baseline_host.size() * sizeof(unsigned)));

        CheckHIPStatus(moe_test::RepackNativeMxFp4Weights(
            w13, w13_baseline, experts * 2 * inter_dim, dim));
        CheckHIPStatus(moe_test::RepackNativeMxFp4Weights(
            w2, w2_baseline, experts * dim, inter_dim));
        CheckHIPStatus(moe_test::RepackNativeMxFp4Scales(
            w13_scale, w13_scale_baseline, experts * 2 * inter_dim, dim / 32));
        CheckHIPStatus(moe_test::RepackNativeMxFp4Scales(
            w2_scale, w2_scale_baseline, experts * dim, inter_dim / 32));
    }

    NativeMxFp4DeviceLayout(const NativeMxFp4DeviceLayout &) = delete;
    NativeMxFp4DeviceLayout &
    operator=(const NativeMxFp4DeviceLayout &) = delete;

    ~NativeMxFp4DeviceLayout() {
        Free(w2_scale_baseline);
        Free(w13_scale_baseline);
        Free(w2_baseline);
        Free(w13_baseline);
        Free(w2_scale);
        Free(w13_scale);
        Free(w2);
        Free(w13);
    }

  private:
    static void Free(void *ptr) {
        if (ptr != nullptr) {
            CheckHIPStatus(hipFree(ptr));
        }
    }
};

std::vector<unsigned short> RunSingleExpertBf16MxFp4Kernel(
    const std::vector<__hip_bfloat16> &act,
    const std::vector<unsigned char> &w13_baseline_bytes,
    const std::vector<unsigned char> &w2_baseline_bytes, unsigned tokens,
    unsigned dim, unsigned inter_dim) {
    static constexpr unsigned kTopK = 1;
    static constexpr unsigned kSortedTokenPadding = 32;
    static constexpr unsigned kMaxNumTokensPadded = 32;
    static constexpr unsigned kMaxNumMBlocks = 1;
    static constexpr unsigned kScaleGroup = 32;
    static constexpr unsigned char kScale = 127;

    std::vector<unsigned> sorted_token_ids(kMaxNumTokensPadded,
                                           (kTopK << 24) | tokens);
    std::vector<float> sorted_weights(kMaxNumTokensPadded, 0.0f);
    for (unsigned token = 0; token < tokens; ++token) {
        sorted_token_ids[token] = token;
        sorted_weights[token] = 1.0f;
    }
    std::vector<unsigned> sorted_expert_ids(kMaxNumMBlocks, 0);
    std::vector<unsigned> num_valid_ids = {kSortedTokenPadding, tokens};

    std::vector<unsigned char> w13_scale(
        static_cast<size_t>(2 * inter_dim) * (dim / kScaleGroup), kScale);
    std::vector<unsigned char> w2_scale(
        static_cast<size_t>(dim) * (inter_dim / kScaleGroup), kScale);

    const auto w13_baseline = Words(w13_baseline_bytes);
    const auto w2_baseline = Words(w2_baseline_bytes);
    const auto w13_scale_baseline = Words(w13_scale);
    const auto w2_scale_baseline = Words(w2_scale);
    NativeMxFp4DeviceLayout weights(w13_baseline, w2_baseline,
                                    w13_scale_baseline, w2_scale_baseline, dim,
                                    inter_dim);

    std::vector<unsigned short> out(static_cast<size_t>(tokens) * dim, 0);
    auto *d_out = CopyToDevice(out);
    auto *d_act = CopyToDevice(act);
    auto *d_sorted_token_ids = CopyToDevice(sorted_token_ids);
    auto *d_sorted_weights = CopyToDevice(sorted_weights);
    auto *d_sorted_expert_ids = CopyToDevice(sorted_expert_ids);
    auto *d_num_valid_ids = CopyToDevice(num_valid_ids);

    const int err = RunBf16MxFp4(
        reinterpret_cast<uint4 *>(d_out), reinterpret_cast<const uint4 *>(d_act),
        reinterpret_cast<const uint4 *>(weights.w13),
        reinterpret_cast<const uint4 *>(weights.w2),
        reinterpret_cast<const uint4 *>(d_sorted_token_ids),
        reinterpret_cast<const uint4 *>(d_sorted_weights),
        reinterpret_cast<const uint4 *>(d_sorted_expert_ids), d_num_valid_ids,
        kTopK, reinterpret_cast<const uint4 *>(weights.w13_scale),
        weights.w2_scale,
        kMaxNumMBlocks, tokens, dim, inter_dim, 1);
    if (err != 0) {
        throw std::runtime_error("FusedMoEMatmul1Stage GPT-OSS failed");
    }
    CheckHIPStatus(hipDeviceSynchronize());
    CheckHIPStatus(hipMemcpy(out.data(), d_out, out.size() * sizeof(out[0]),
                             hipMemcpyDeviceToHost));

    FreeDevice(d_out, d_act, d_sorted_token_ids, d_sorted_weights,
               d_sorted_expert_ids, d_num_valid_ids);
    return out;
}

void ExpectCloseBf16Output(const std::vector<unsigned short> &out,
                           const std::vector<float> &expected, unsigned cols,
                           float atol) {
    ASSERT_EQ(out.size(), expected.size());
    float max_abs_err = 0.0f;
    size_t max_idx = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const float actual = Bf16BitsToFloat(out[i]);
        const float abs_err = std::abs(actual - expected[i]);
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            max_idx = i;
        }
    }
    EXPECT_LT(max_abs_err, atol)
        << "max_idx=" << max_idx << " token=" << (max_idx / cols)
        << " col=" << (max_idx % cols) << " expected=" << expected[max_idx]
        << " actual=" << Bf16BitsToFloat(out[max_idx]);
}

void RunStage1IdentityW2Test(bool verify_up_path) {
    static constexpr unsigned kTokens = 16;
    static constexpr unsigned kDim = 256;
    static constexpr unsigned kInterDim = 256;
    static constexpr unsigned char kOne = 2;

    std::vector<__hip_bfloat16> act(kTokens * kDim);
    for (unsigned token = 0; token < kTokens; ++token) {
        for (unsigned col = 0; col < kDim; ++col) {
            const int pattern =
                static_cast<int>((token * 11 + col * 5) % 33) - 16;
            float x = static_cast<float>(pattern) / 16.0f;
            if (verify_up_path && col == 0) {
                x = 1.0f;
            }
            act[static_cast<size_t>(token) * kDim + col] = FloatAsBf16(x);
        }
    }

    std::vector<unsigned char> w13_baseline(
        static_cast<size_t>(2 * kInterDim) * kDim / 2, 0);
    std::vector<unsigned char> w2_baseline(
        static_cast<size_t>(kDim) * kInterDim / 2, 0);
    for (unsigned i = 0; i < kInterDim; ++i) {
        if (verify_up_path) {
            SetBaselineMxFp4(w13_baseline, 2 * kInterDim, kDim, i, 0, kOne);
            SetBaselineMxFp4(w13_baseline, 2 * kInterDim, kDim, kInterDim + i, i,
                            kOne);
        } else {
            SetBaselineMxFp4(w13_baseline, 2 * kInterDim, kDim, i, i, kOne);
        }
        SetBaselineMxFp4(w2_baseline, kDim, kInterDim, i, i, kOne);
    }

    const auto out = RunSingleExpertBf16MxFp4Kernel(
        act, w13_baseline, w2_baseline, kTokens, kDim, kInterDim);

    std::vector<__hip_bfloat16> gate_in(kTokens * kDim);
    std::vector<__hip_bfloat16> up_in(kTokens * kDim);
    std::vector<__hip_bfloat16> h(kTokens * kDim);
    for (unsigned token = 0; token < kTokens; ++token) {
        for (unsigned col = 0; col < kDim; ++col) {
            const size_t idx = static_cast<size_t>(token) * kDim + col;
            gate_in[idx] =
                verify_up_path ? act[static_cast<size_t>(token) * kDim]
                               : act[idx];
            up_in[idx] = verify_up_path ? act[idx] : FloatAsBf16(0.0f);
        }
    }
    ApplyOpenAISwiGLU(gate_in, up_in, h);
    std::vector<float> expected(h.size());
    for (size_t i = 0; i < h.size(); ++i) {
        expected[i] = Bf16ToFloat(h[i]);
    }
    ExpectCloseBf16Output(out, expected, kDim, 1.0e-3f);
}

void RunStage2StructuredW2Test() {
    static constexpr unsigned kTokens = 16;
    static constexpr unsigned kDim = 256;
    static constexpr unsigned kInterDim = 256;
    static constexpr unsigned char kHalf = 1;
    static constexpr unsigned char kOne = 2;

    std::vector<__hip_bfloat16> act(kTokens * kDim);
    for (unsigned token = 0; token < kTokens; ++token) {
        for (unsigned col = 0; col < kDim; ++col) {
            const int pattern =
                static_cast<int>((token * 7 + col * 3) % 29) - 14;
            float x = static_cast<float>(pattern) / 20.0f;
            if (col == 0) {
                x = 1.0f;
            }
            act[static_cast<size_t>(token) * kDim + col] = FloatAsBf16(x);
        }
    }

    std::vector<unsigned char> w13_baseline(
        static_cast<size_t>(2 * kInterDim) * kDim / 2, 0);
    std::vector<unsigned char> w2_baseline(
        static_cast<size_t>(kDim) * kInterDim / 2, 0);
    for (unsigned i = 0; i < kInterDim; ++i) {
        SetBaselineMxFp4(w13_baseline, 2 * kInterDim, kDim, i, 0, kOne);
        SetBaselineMxFp4(w13_baseline, 2 * kInterDim, kDim, kInterDim + i, i,
                        kOne);
    }
    for (unsigned row = 0; row < kDim; ++row) {
        SetBaselineMxFp4(w2_baseline, kDim, kInterDim, row, row, kOne);
        SetBaselineMxFp4(w2_baseline, kDim, kInterDim, row,
                        (row + 17) % kInterDim, kHalf);
    }

    const auto out = RunSingleExpertBf16MxFp4Kernel(
        act, w13_baseline, w2_baseline, kTokens, kDim, kInterDim);

    std::vector<__hip_bfloat16> gate_in(kTokens * kInterDim);
    std::vector<__hip_bfloat16> up_in(kTokens * kInterDim);
    std::vector<__hip_bfloat16> h(kTokens * kInterDim);
    for (unsigned token = 0; token < kTokens; ++token) {
        for (unsigned col = 0; col < kInterDim; ++col) {
            const size_t idx = static_cast<size_t>(token) * kInterDim + col;
            gate_in[idx] = act[static_cast<size_t>(token) * kDim];
            up_in[idx] = act[static_cast<size_t>(token) * kDim + col];
        }
    }
    ApplyOpenAISwiGLU(gate_in, up_in, h);

    std::vector<float> expected(kTokens * kDim);
    for (unsigned token = 0; token < kTokens; ++token) {
        for (unsigned col = 0; col < kDim; ++col) {
            const auto h_idx = static_cast<size_t>(token) * kInterDim;
            const float y =
                Bf16ToFloat(h[h_idx + col]) +
                0.5f * Bf16ToFloat(h[h_idx + ((col + 17) % kInterDim)]);
            expected[static_cast<size_t>(token) * kDim + col] =
                Bf16BitsToFloat(FloatToBf16Bits(y));
        }
    }
    ExpectCloseBf16Output(out, expected, kDim, 1.0e-3f);
}

void RunSingleExpertBiasCoverageTest() {
    static constexpr unsigned kTokens = 16;
    static constexpr unsigned kDim = 512;
    static constexpr unsigned kInterDim = 512;
    static constexpr unsigned kTopK = 1;
    static constexpr unsigned kSortedTokenPadding = 32;
    static constexpr unsigned kMaxNumTokensPadded = 32;
    static constexpr unsigned kMaxNumMBlocks = 1;
    static constexpr unsigned kScaleGroup = 32;
    static constexpr unsigned char kOne = 2;
    static constexpr unsigned char kScale = 127;

    std::vector<__hip_bfloat16> act(kTokens * kDim);
    for (unsigned token = 0; token < kTokens; ++token) {
        for (unsigned col = 0; col < kDim; ++col) {
            const int pattern =
                static_cast<int>((token * 13 + col * 7) % 17) - 8;
            act[static_cast<size_t>(token) * kDim + col] =
                FloatAsBf16(static_cast<float>(pattern) / 128.0f);
        }
    }

    std::vector<unsigned char> w13_baseline(
        static_cast<size_t>(2 * kInterDim) * kDim / 2, 0);
    std::vector<unsigned char> w2_baseline(
        static_cast<size_t>(kDim) * kInterDim / 2, 0);
    for (unsigned i = 0; i < kDim; ++i) {
        SetBaselineMxFp4(w2_baseline, kDim, kInterDim, i, i, kOne);
    }
    SetBaselineMxFp4(w2_baseline, kDim, kInterDim, 0, 510, kOne);
    SetBaselineMxFp4(w2_baseline, kDim, kInterDim, 1, 511, kOne);
    SetBaselineMxFp4(w2_baseline, kDim, kInterDim, 254, 256, kOne);
    SetBaselineMxFp4(w2_baseline, kDim, kInterDim, 255, 257, kOne);

    std::vector<unsigned> sorted_token_ids(kMaxNumTokensPadded,
                                           (kTopK << 24) | kTokens);
    std::vector<float> sorted_weights(kMaxNumTokensPadded, 0.0f);
    for (unsigned token = 0; token < kTokens; ++token) {
        sorted_token_ids[token] = token;
        sorted_weights[token] = 1.0f;
    }
    std::vector<unsigned> sorted_expert_ids(kMaxNumMBlocks, 0);
    std::vector<unsigned> num_valid_ids = {kSortedTokenPadding, kTokens};

    std::vector<unsigned char> w13_scale(
        static_cast<size_t>(2 * kInterDim) * (kDim / kScaleGroup), kScale);
    std::vector<unsigned char> w2_scale(
        static_cast<size_t>(kDim) * (kInterDim / kScaleGroup), kScale);

    std::vector<__hip_bfloat16> w13_bias(2 * kInterDim);
    std::vector<__hip_bfloat16> w2_bias(kDim);
    for (unsigned col = 0; col < kInterDim; ++col) {
        const float small = static_cast<float>(static_cast<int>(col % 11) - 5);
        w13_bias[col] = FloatAsBf16(0.015625f * small);
        w13_bias[kInterDim + col] = FloatAsBf16(0.03125f * small);
    }
    for (unsigned col = 0; col < kDim; ++col) {
        const float small = static_cast<float>(static_cast<int>(col % 11) - 5);
        w2_bias[col] = FloatAsBf16(0.0078125f * small);
    }
    w13_bias[0] = FloatAsBf16(3.0f);
    w13_bias[1] = FloatAsBf16(-2.0f);
    w13_bias[254] = FloatAsBf16(4.0f);
    w13_bias[255] = FloatAsBf16(-3.0f);
    w13_bias[256] = FloatAsBf16(3.5f);
    w13_bias[257] = FloatAsBf16(-2.5f);
    w13_bias[510] = FloatAsBf16(4.5f);
    w13_bias[511] = FloatAsBf16(-3.5f);
    w13_bias[kInterDim + 0] = FloatAsBf16(2.5f);
    w13_bias[kInterDim + 1] = FloatAsBf16(-1.5f);
    w13_bias[kInterDim + 254] = FloatAsBf16(1.75f);
    w13_bias[kInterDim + 255] = FloatAsBf16(-2.25f);
    w13_bias[kInterDim + 256] = FloatAsBf16(2.75f);
    w13_bias[kInterDim + 257] = FloatAsBf16(-1.75f);
    w13_bias[kInterDim + 510] = FloatAsBf16(3.25f);
    w13_bias[kInterDim + 511] = FloatAsBf16(-2.75f);
    w2_bias[0] = FloatAsBf16(5.0f);
    w2_bias[1] = FloatAsBf16(-4.0f);
    w2_bias[254] = FloatAsBf16(6.0f);
    w2_bias[255] = FloatAsBf16(-5.0f);

    const auto w13_baseline_words = Words(w13_baseline);
    const auto w2_baseline_words = Words(w2_baseline);
    const auto w13_scale_baseline = Words(w13_scale);
    const auto w2_scale_baseline = Words(w2_scale);
    NativeMxFp4DeviceLayout weights(w13_baseline_words, w2_baseline_words,
                                    w13_scale_baseline, w2_scale_baseline,
                                    kDim, kInterDim);

    std::vector<unsigned short> out(kTokens * kDim, 0);
    auto *d_out = CopyToDevice(out);
    auto *d_act = CopyToDevice(act);
    auto *d_sorted_token_ids = CopyToDevice(sorted_token_ids);
    auto *d_sorted_weights = CopyToDevice(sorted_weights);
    auto *d_sorted_expert_ids = CopyToDevice(sorted_expert_ids);
    auto *d_num_valid_ids = CopyToDevice(num_valid_ids);
    auto *d_w13_bias_logical = CopyToDevice(w13_bias);
    auto *d_w2_bias_logical = CopyToDevice(w2_bias);
    __hip_bfloat16 *d_w13_bias = nullptr;
    __hip_bfloat16 *d_w2_bias = nullptr;
    const unsigned w13_bias_padded_cols = ((kInterDim + 255) / 256) * 256;
    const unsigned w2_bias_padded_cols = ((kDim + 255) / 256) * 256;
    CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&d_w13_bias),
                             2 * w13_bias_padded_cols *
                                 sizeof(__hip_bfloat16)));
    CheckHIPStatus(hipMalloc(reinterpret_cast<void **>(&d_w2_bias),
                             w2_bias_padded_cols * sizeof(__hip_bfloat16)));
    CheckHIPStatus(moe_test::RepackMxFp4Bias(
        d_w13_bias, d_w13_bias_logical, 2, kInterDim));
    CheckHIPStatus(moe_test::RepackMxFp4Bias(
        d_w2_bias, d_w2_bias_logical, 1, kDim));

    const int err = RunBf16MxFp4(
        reinterpret_cast<uint4 *>(d_out), reinterpret_cast<const uint4 *>(d_act),
        reinterpret_cast<const uint4 *>(weights.w13),
        reinterpret_cast<const uint4 *>(weights.w2),
        reinterpret_cast<const uint4 *>(d_sorted_token_ids),
        reinterpret_cast<const uint4 *>(d_sorted_weights),
        reinterpret_cast<const uint4 *>(d_sorted_expert_ids), d_num_valid_ids,
        kTopK, reinterpret_cast<const uint4 *>(weights.w13_scale),
        weights.w2_scale,
        kMaxNumMBlocks, kTokens, kDim, kInterDim, 1, d_w13_bias, d_w2_bias);
    ASSERT_EQ(err, 0);
    CheckHIPStatus(hipDeviceSynchronize());
    CheckHIPStatus(hipMemcpy(out.data(), d_out, out.size() * sizeof(out[0]),
                             hipMemcpyDeviceToHost));

    std::vector<__hip_bfloat16> gate_in(kInterDim);
    std::vector<__hip_bfloat16> up_in(kInterDim);
    std::vector<__hip_bfloat16> h(kInterDim);
    for (unsigned col = 0; col < kInterDim; ++col) {
        gate_in[col] = w13_bias[col];
        up_in[col] = w13_bias[kInterDim + col];
    }
    ApplyOpenAISwiGLU(gate_in, up_in, h);

    std::vector<float> expected(kTokens * kDim);
    for (unsigned token = 0; token < kTokens; ++token) {
        for (unsigned col = 0; col < kDim; ++col) {
            float y = Bf16ToFloat(h[col]);
            if (col == 0) {
                y += Bf16ToFloat(h[510]);
            } else if (col == 1) {
                y += Bf16ToFloat(h[511]);
            } else if (col == 254) {
                y += Bf16ToFloat(h[256]);
            } else if (col == 255) {
                y += Bf16ToFloat(h[257]);
            }
            expected[static_cast<size_t>(token) * kDim + col] =
                Bf16BitsToFloat(FloatToBf16Bits(y + Bf16ToFloat(w2_bias[col])));
        }
    }
    ExpectCloseBf16Output(out, expected, kDim, 1.0e-7f);

    FreeDevice(d_out, d_act, d_sorted_token_ids, d_sorted_weights,
               d_sorted_expert_ids, d_num_valid_ids, d_w13_bias, d_w2_bias,
               d_w13_bias_logical, d_w2_bias_logical);
}

TEST(FusedMoEBf16MxFp4, Stage1XW1AndHMatchIdentityW2) {
    RunStage1IdentityW2Test(false);
}

TEST(FusedMoEBf16MxFp4, Stage1XW3AndHMatchIdentityW2) {
    RunStage1IdentityW2Test(true);
}

TEST(FusedMoEBf16MxFp4, Stage2StructuredW2MatchesKnownH) {
    RunStage2StructuredW2Test();
}

TEST(FusedMoEBf16MxFp4, BiasAppliesToW13AndW2EarlyAndLateDims) {
    RunSingleExpertBiasCoverageTest();
}

} // namespace
} // namespace causalflow::petit::rocm::moe
