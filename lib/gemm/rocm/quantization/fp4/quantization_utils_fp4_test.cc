#include "gemm/rocm/quantization/gemm.h"
#include "gemm_fp4.h"
#include "tests/floating_points.h"
#include "utils/hip_helper.h"
#include "utils/test_utils.h"

#include <climits>
#include <fstream>
#include <gtest/gtest.h>
#include <random>

namespace causalflow::petit::rocm::quantization::fp4 {

int DequantNvFp4(unsigned *output, const unsigned *input,
                 const unsigned *scales, float global_scale, DataType out_type,
                 unsigned k, unsigned n);

int DequantPetitFp4(unsigned *output, const unsigned *input,
                    const unsigned *scales, float global_scale,
                    DataType out_type, unsigned k, unsigned n);

static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(char);
static constexpr unsigned kPackFactor = 32 / 4;
static constexpr unsigned kQuantVecSize = sizeof(uint4) / sizeof(unsigned);
static constexpr unsigned kRowGroupSize = 16;

template <class Element, unsigned kM, unsigned kN> struct DeviceContext {
    using ScaleType = tests::cpu_numeric::fp8_e4m3_t;
    static constexpr unsigned kOutVecSize = sizeof(uint4) / sizeof(Element);
    uint4 d_weights_quant[kM * kN / kPackFactor / kQuantVecSize];
    uint4 d_scales[kM * kN / kRowGroupSize / kVecSize];
    uint4 d_reference[kM * kN / kOutVecSize];
    uint4 d_petit_weights[kM * kN / kPackFactor / kQuantVecSize];
    uint4 d_petit_scales[kM * kN / kRowGroupSize / kVecSize];
    uint4 d_output[kM * kN / kOutVecSize];

    static DeviceContext *PrepareDevice();
    void CompareOutputsFromDevice() const;
};

template <class Element, unsigned kM, unsigned kN>
DeviceContext<Element, kM, kN> *
DeviceContext<Element, kM, kN>::PrepareDevice() {
    DeviceContext<Element, kM, kN> *d_ctx;
    CheckHIPStatus(hipMalloc(&d_ctx, sizeof(DeviceContext<Element, kM, kN>)));

    std::vector<unsigned> h_qweights(kM * kN / kPackFactor);
    std::vector<ScaleType> h_scales(kM * kN / kRowGroupSize);

    std::mt19937 gen(42);
    std::uniform_int_distribution<unsigned> dist_q(0, UINT_MAX);
    auto gen_q = [&]() { return dist_q(gen); };

    // Only generate positive scales based on how preprocessing of the scales is
    // done
    std::uniform_int_distribution<unsigned> dist_scale(
        std::numeric_limits<ScaleType>::denorm_min().to_bits(),
        std::numeric_limits<ScaleType>::max().to_bits());
    auto gen_scale_fp8 = [&]() {
        return ScaleType::from_bits(dist_scale(gen));
    };

    FillRandomValue(gen_q, &h_qweights);
    FillRandomValue(gen_scale_fp8, &h_scales);

    CheckHIPStatus(hipMemcpy(d_ctx->d_weights_quant, h_qweights.data(),
                             h_qweights.size() * sizeof(unsigned),
                             hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(d_ctx->d_scales, h_scales.data(),
                             h_scales.size() * sizeof(ScaleType),
                             hipMemcpyHostToDevice));
    return d_ctx;
}

template <class Element, unsigned kM, unsigned kN>
void DeviceContext<Element, kM, kN>::CompareOutputsFromDevice() const {
    std::vector<Element> h_reference(kM * kN), h_petit_output(kM * kN);
    CheckHIPStatus(hipMemcpy(h_reference.data(), d_reference,
                             h_reference.size() * sizeof(Element),
                             hipMemcpyDeviceToHost));
    CheckHIPStatus(hipMemcpy(h_petit_output.data(), d_output,
                             h_petit_output.size() * sizeof(Element),
                             hipMemcpyDeviceToHost));
    CheckHIPStatus(hipDeviceSynchronize());

    for (unsigned i = 0; i < kM * kN; ++i) {
        EXPECT_EQ(h_reference[i], h_petit_output[i])
            << "Output and reference differ at index " << i;
    }
}

class NvFp4ToPetitFp4Test : public ::testing::Test {
  public:
    template <class Element, unsigned kM, unsigned kN>
    void TestConvert(float global_scale, DataType out_type) const {
        auto d_ctx = DeviceContext<Element, kM, kN>::PrepareDevice();

        DequantNvFp4(reinterpret_cast<unsigned *>(d_ctx->d_reference),
                     reinterpret_cast<const unsigned *>(d_ctx->d_weights_quant),
                     reinterpret_cast<const unsigned *>(d_ctx->d_scales),
                     global_scale, out_type, kM, kN);

        RepackNvFp4ToPetitFp4Weights(
            reinterpret_cast<unsigned *>(d_ctx->d_petit_weights),
            reinterpret_cast<const unsigned *>(d_ctx->d_weights_quant), kM, kN,
            nullptr);

        RepackNvFp4ToPetitFp4Scales(
            reinterpret_cast<unsigned *>(d_ctx->d_petit_scales),
            reinterpret_cast<const unsigned *>(d_ctx->d_scales), kM, kN,
            nullptr);

        DequantPetitFp4(
            reinterpret_cast<unsigned *>(d_ctx->d_output),
            reinterpret_cast<const unsigned *>(d_ctx->d_petit_weights),
            reinterpret_cast<const unsigned *>(d_ctx->d_petit_scales),
            global_scale, out_type, kM, kN);

        d_ctx->CompareOutputsFromDevice();
        CheckHIPStatus(hipFree(d_ctx));
    }
};

// This struct is used to ensure that the dequantization algorithm is
// mathematically accurate. To cover all possible input combinations,
// we use exhaustive testing for quantized weights and scales.
template <class Element> struct ExhaustiveDeviceContext {
    using ScaleType = tests::cpu_numeric::fp8_e4m3_t;
    static constexpr unsigned kQweightBitMin = 0;
    static constexpr unsigned kQweightBitMax = 15;
    static constexpr unsigned kScaleBitMin =
        std::numeric_limits<ScaleType>::denorm_min().to_bits();
    static constexpr unsigned kScaleBitMax =
        std::numeric_limits<ScaleType>::max().to_bits();

    static unsigned GenerateQweight(unsigned i) {
        return std::clamp(i, kQweightBitMin, kQweightBitMax);
    }

    static ScaleType GenerateScale(unsigned i) {
        i = std::clamp(i, kScaleBitMin, kScaleBitMax);
        return ScaleType::from_bits(static_cast<uint8_t>(i));
    }

    static float GenerateOutput(unsigned i_qweight, unsigned i_scale) {

        unsigned qweight = GenerateQweight(i_qweight);

        ScaleType scale = GenerateScale(i_scale);

        static const float fp4_values[] = {
            0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
        };
        return scale.to_fp32() * fp4_values[qweight];
    }

    // Ensure input data alignment to meet the dequantizer's requirements.
    static constexpr unsigned kM = 256;
    static constexpr unsigned kN = 128;
    static constexpr unsigned kOutVecSize = sizeof(uint4) / sizeof(Element);
    uint4 d_weights_quant[kM * kN / kPackFactor / kQuantVecSize];
    uint4 d_scales[kM * kN / kRowGroupSize / kVecSize];
    uint4 d_petit_weights[kM * kN / kPackFactor / kQuantVecSize];
    uint4 d_petit_scales[kM * kN / kRowGroupSize / kVecSize];
    uint4 d_output[kM * kN / kOutVecSize];

    static ExhaustiveDeviceContext *PrepareDevice();
    void ValidateOutputsFromDevice() const;
};

template <class Element>
ExhaustiveDeviceContext<Element> *
ExhaustiveDeviceContext<Element>::PrepareDevice() {
    ExhaustiveDeviceContext<Element> *d_ctx;
    CheckHIPStatus(hipMalloc(&d_ctx, sizeof(ExhaustiveDeviceContext<Element>)));

    std::vector<unsigned> h_qweights(kM * kN / kPackFactor);
    std::vector<ScaleType> h_scales(kM * kN / kRowGroupSize);

    for (unsigned i = 0; i < kM * kN / kPackFactor; ++i) {
        unsigned row = i % (kM / kPackFactor);
        unsigned group_i = row * kPackFactor / kRowGroupSize;

        unsigned qweight = GenerateQweight(group_i);
        unsigned vec = qweight * 0x11111111;
        h_qweights[i] = vec;
    }

    for (unsigned i = 0; i < kM * kN / kRowGroupSize; ++i) {
        unsigned col = i / (kM / kRowGroupSize);

        ScaleType scale = GenerateScale(col);
        h_scales[i] = scale;
    }

    CheckHIPStatus(hipMemcpy(d_ctx->d_weights_quant, h_qweights.data(),
                             h_qweights.size() * sizeof(unsigned),
                             hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(d_ctx->d_scales, h_scales.data(),
                             h_scales.size() * sizeof(ScaleType),
                             hipMemcpyHostToDevice));
    return d_ctx;
}

template <class Element>
void ExhaustiveDeviceContext<Element>::ValidateOutputsFromDevice() const {
    std::vector<Element> h_petit_output(kM * kN);
    CheckHIPStatus(hipMemcpy(h_petit_output.data(), d_output,
                             h_petit_output.size() * sizeof(Element),
                             hipMemcpyDeviceToHost));
    CheckHIPStatus(hipDeviceSynchronize());

    // Only validate one output from each group since the qweights in the group
    // are the same
    for (unsigned i = 0; i < kM * kN / kRowGroupSize; ++i) {
        unsigned row = i % (kM / kRowGroupSize);
        unsigned col = i / (kM / kRowGroupSize);
        float reference_fp32 = GenerateOutput(row, col);
        Element reference = Element::from_fp32(reference_fp32);
        EXPECT_EQ(reference, h_petit_output[i * kRowGroupSize])
            << "Output and reference differ at (" << row << ", " << col << ")";
    }
}

class ExhaustiveFp4DequantTest : public ::testing::Test {
  public:
    template <class Element> void TestAllCombinations(DataType out_type) const {
        using Context = ExhaustiveDeviceContext<Element>;
        auto d_ctx = Context::PrepareDevice();
        constexpr unsigned kM = Context::kM;
        constexpr unsigned kN = Context::kN;

        RepackNvFp4ToPetitFp4Weights(
            reinterpret_cast<unsigned *>(d_ctx->d_petit_weights),
            reinterpret_cast<const unsigned *>(d_ctx->d_weights_quant), kM, kN,
            nullptr);

        RepackNvFp4ToPetitFp4Scales(
            reinterpret_cast<unsigned *>(d_ctx->d_petit_scales),
            reinterpret_cast<const unsigned *>(d_ctx->d_scales), kM, kN,
            nullptr);

        DequantPetitFp4(
            reinterpret_cast<unsigned *>(d_ctx->d_output),
            reinterpret_cast<const unsigned *>(d_ctx->d_petit_weights),
            reinterpret_cast<const unsigned *>(d_ctx->d_petit_scales), 1.0f,
            out_type, kM, kN);

        d_ctx->ValidateOutputsFromDevice();
        CheckHIPStatus(hipFree(d_ctx));
    }
};

using tests::cpu_numeric::bf16_t;
using tests::cpu_numeric::fp16_t;

TEST_F(NvFp4ToPetitFp4Test, TestLayout128x16Bf16) {
    TestConvert<bf16_t, 512, 512>(1.0, kDataTypeBf16);
}

TEST_F(NvFp4ToPetitFp4Test, TestLayout128x16Fp16) {
    TestConvert<fp16_t, 512, 512>(1.0, kDataTypeFp16);
}

// Validate all possible combinations of qweights and scales to cover all corner
// cases
TEST_F(ExhaustiveFp4DequantTest, AllFp4ScaleCombinationsBf16) {
    TestAllCombinations<bf16_t>(kDataTypeBf16);
}

TEST_F(ExhaustiveFp4DequantTest, AllFp4ScaleCombinationsFp16) {
    TestAllCombinations<fp16_t>(kDataTypeFp16);
}

} // namespace causalflow::petit::rocm::quantization::fp4
