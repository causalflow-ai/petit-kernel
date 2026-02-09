#include "gemm/rocm/quantization/gemm.h"
#include "gemm_fp4.h"
#include "tests/quantization.h"
#include "utils/hip_helper.h"

#include <climits>
#include <cmath>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>

namespace causalflow::petit::rocm::quantization::fp4 {

int DequantMxFp4(unsigned *output, const unsigned *input,
                 const unsigned *scales, float global_scale, DataType out_type,
                 unsigned k, unsigned n);

int DequantPetitMxFp4(unsigned *output, const unsigned *input,
                      const unsigned *scales, float global_scale,
                      DataType out_type, unsigned k, unsigned n);

static inline void CheckHipblasStatus(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "HipBLAS Error: " << status << std::endl;
        throw std::runtime_error("HipBLAS Error");
    }
}

MATCHER_P(IsNearBf16, ref, "") {
    unsigned a_f32 = (unsigned)arg << 16, b_f32 = (unsigned)ref << 16;
    float a_f = reinterpret_cast<const float &>(a_f32);
    float b_f = reinterpret_cast<const float &>(b_f32);

    if (std::abs(a_f - b_f) < std::max<float>(1e-2, fabs(b_f) * 0.01f)) {
        return true;
    }

    if (result_listener->IsInterested()) {
        *result_listener << "Expected bfloat16 value near " << std::hex << "0x"
                         << ref << " (" << b_f << "), but got " << std::hex
                         << "0x" << arg << " (" << a_f << ")";
    }

    return false;
}

MATCHER_P(IsNearFp16, ref, "") {
    float a_f = __half2float(arg);
    float b_f = __half2float(ref);

    if (std::abs(a_f - b_f) < std::max<float>(1e-2, fabs(b_f) * 0.01f)) {
        return true;
    }

    unsigned short a_u16 = reinterpret_cast<const unsigned short &>(arg);
    unsigned short b_u16 = reinterpret_cast<const unsigned short &>(ref);

    if (result_listener->IsInterested()) {
        *result_listener << "Expected float value near " << std::hex << ref
                         << " (0x" << b_u16 << "), but got " << std::hex << arg
                         << " (0x" << a_u16 << ")";
    }

    return false;
}

using GemmMPTestData = tests::quantization::GemmMPTestData;

class GemmFp4Fp16Test : public ::testing::Test {
  public:
    static constexpr size_t kWorkspaceSize = 32 * 1024 * 1024;
    void SetUp() override;
    void TearDown() override;

    void ComputeReference(GemmMPTestData *ctx) const;
    void CopyAndCompareOutput(GemmMPTestData *ctx) const;
    void TestGemmBySolutionId(unsigned m, unsigned n, unsigned k,
                              float global_scale, unsigned long solution_id,
                              DataType data_type, DataType b_type,
                              bool require_high_precision,
                              unsigned group_size = 16);
    void TestGemm(unsigned m, unsigned n, unsigned k, float global_scale,
                  SolutionId sol_id,
                  DataType b_type = DataType::kDataTypeFp4e2m1,
                  unsigned group_size = 16);

    hipblasLtHandle_t handle_;
    hipblasLtMatmulDesc_t matmul_desc_;
    void *d_workspace_;
    float *d_global_scale_;
    std::unique_ptr<hal::Device> dev_;
    DataType dequant_type_;
};

void GemmFp4Fp16Test::SetUp() {
    static constexpr hipblasOperation_t kTransposed = HIPBLAS_OP_T;
    CheckHIPStatus(hipMalloc(&d_workspace_, kWorkspaceSize));
    CheckHIPStatus(hipMalloc(&d_global_scale_, sizeof(float)));

    CheckHipblasStatus(hipblasLtCreate(&handle_));
    CheckHipblasStatus(hipblasLtMatmulDescCreate(
        &matmul_desc_, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CheckHipblasStatus(hipblasLtMatmulDescSetAttribute(
        matmul_desc_, HIPBLASLT_MATMUL_DESC_TRANSA, &kTransposed,
        sizeof(hipblasOperation_t)));

    auto plat = hal::GetPlatform("rocm");
    ASSERT_EQ(absl::OkStatus(), plat->GetDevice(0, &dev_));
}

void GemmFp4Fp16Test::TearDown() {
    CheckHIPStatus(hipFree(d_workspace_));
    CheckHipblasStatus(hipblasLtMatmulDescDestroy(matmul_desc_));

    CheckHipblasStatus(hipblasLtDestroy(handle_));
}

void GemmFp4Fp16Test::ComputeReference(GemmMPTestData *ctx) const {
    static constexpr float kAlpha = 1.0f;
    static constexpr float kBeta = 0.0f;

    hipDataType type_a = HIP_R_16F, type_c = HIP_R_16F;
    switch (dequant_type_) {
    case DataType::kDataTypeFp16:
        type_a = HIP_R_16F;
        type_c = HIP_R_16F;
        break;
    case DataType::kDataTypeBf16:
        type_a = HIP_R_16BF;
        type_c = HIP_R_16BF;
        break;
    case DataType::kDataTypeFp8e4m3:
        type_a = HIP_R_8F_E4M3_FNUZ;
        type_c = HIP_R_16F;
        break;
    default:
        ASSERT_TRUE(false) << "Invalid dequant type";
        break;
    }

    hipblasLtMatrixLayout_t layout_a, layout_b, layout_c;
    CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_a, type_a, ctx->k(),
                                                   ctx->m(), ctx->k()));
    CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_b, type_a, ctx->k(),
                                                   ctx->n(), ctx->k()));
    CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_c, type_c, ctx->n(),
                                                   ctx->m(), ctx->n()));

    // Compute C in row-major order
    CheckHIPStatus(hipMemset(ctx->reference(), 0, ctx->OutputSize()));

    // rocBLAS expects matrices in column-major format, so we transpose the
    // operation C = A * B becomes C^T = B^T * A^T
    CheckHipblasStatus(hipblasLtMatmul(
        handle_, matmul_desc_, &kAlpha, ctx->weights(), layout_b, ctx->input(),
        layout_a, &kBeta, ctx->reference(), layout_c, ctx->reference(),
        layout_c, nullptr, d_workspace_, kWorkspaceSize, nullptr));

    CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_a));
    CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_b));
    CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_c));
}

void GemmFp4Fp16Test::CopyAndCompareOutput(GemmMPTestData *ctx) const {
    std::vector<unsigned short> h_output(ctx->m() * ctx->n()),
        h_reference(ctx->m() * ctx->n());
    CheckHIPStatus(hipMemcpy(h_output.data(), ctx->output(),
                             h_output.size() * sizeof(h_output[0]),
                             hipMemcpyDeviceToHost));
    CheckHIPStatus(hipMemcpy(h_reference.data(), ctx->reference(),
                             h_reference.size() * sizeof(h_reference[0]),
                             hipMemcpyDeviceToHost));
    CheckHIPStatus(hipDeviceSynchronize());

    for (unsigned i = 0; i < ctx->m() * ctx->n(); ++i) {
        if (dequant_type_ == DataType::kDataTypeFp16) {
            const half *output_ptr =
                reinterpret_cast<const half *>(&h_output[i]);
            const half *ref_ptr =
                reinterpret_cast<const half *>(&h_reference[i]);
            EXPECT_THAT(output_ptr[0], IsNearFp16(ref_ptr[0]))
                << "Output and reference differ at index " << i;
        } else if (dequant_type_ == DataType::kDataTypeBf16) {
            EXPECT_THAT(h_output[i], IsNearBf16(h_reference[i]))
                << "Output and reference differ at index " << i;
        }
    }
}

void GemmFp4Fp16Test::TestGemmBySolutionId(unsigned m, unsigned n, unsigned k,
                                           float global_scale,
                                           unsigned long solution_id,
                                           DataType data_type, DataType b_type,
                                           bool require_high_precision,
                                           unsigned group_size) {
    const bool is_mxfp4 = b_type == DataType::kDataTypeMxFp4e2m1;
    if (is_mxfp4) {
        ASSERT_EQ(data_type, DataType::kDataTypeBf16)
            << "MXFP4 only supports BF16 input/output type";
        dequant_type_ = DataType::kDataTypeBf16;
    } else {
        dequant_type_ = data_type;
    }

    GemmMPTestData ctx(dev_.get(), dequant_type_, b_type, m, n, k, group_size);
    ASSERT_EQ(absl::OkStatus(), ctx.PrepareData(false));
    CheckHIPStatus(hipMemcpy(d_global_scale_, &global_scale, sizeof(float),
                             hipMemcpyHostToDevice));

    unsigned *weights_ptr = reinterpret_cast<unsigned *>(ctx.weights_quant());
    unsigned *scales_ptr = reinterpret_cast<unsigned *>(ctx.scales());
    unsigned *d_petit_weights = nullptr;
    unsigned *d_petit_scales = nullptr;

    int err = 0;
    if (is_mxfp4) {
        CheckHIPStatus(hipMalloc(&d_petit_weights, ctx.WeightsQuantSize()));
        CheckHIPStatus(hipMalloc(&d_petit_scales, ctx.ScalesSize()));
        fp4::RepackNvFp4ToPetitFp4Weights(
            d_petit_weights,
            reinterpret_cast<const unsigned *>(ctx.weights_quant()), k, n,
            nullptr);
        fp4::RepackMxFp4ToPetitFp4Scales(
            d_petit_scales, reinterpret_cast<const unsigned *>(ctx.scales()), k,
            n, nullptr);
        weights_ptr = d_petit_weights;
        scales_ptr = d_petit_scales;
        err = DequantPetitMxFp4(reinterpret_cast<unsigned *>(ctx.weights()),
                                d_petit_weights, d_petit_scales, global_scale,
                                dequant_type_, k, n);
        ASSERT_EQ(err, 0) << "DequantPetitMxFp4 failed";
    } else {
        err = DequantPetitFp4(
            reinterpret_cast<unsigned *>(ctx.weights()),
            reinterpret_cast<const unsigned *>(ctx.weights_quant()),
            reinterpret_cast<const unsigned *>(ctx.scales()), global_scale,
            dequant_type_, k, n);
        ASSERT_EQ(err, 0) << "DequantPetitFp4 failed";
    }

    ComputeReference(&ctx);

    PetitSolutionHints hints;
    hints.a_type = data_type;
    hints.b_type = b_type;
    hints.c_type = data_type;
    hints.require_high_precision = require_high_precision;

    if (is_mxfp4) {
        err = GemmMxFp4Fp16Grid(reinterpret_cast<unsigned *>(ctx.output()),
                                reinterpret_cast<const unsigned *>(ctx.input()),
                                weights_ptr, scales_ptr, d_global_scale_, m, n,
                                k, hints, solution_id, nullptr);
    } else {
        err = GemmFp4Fp16Grid(reinterpret_cast<unsigned *>(ctx.output()),
                              reinterpret_cast<const unsigned *>(ctx.input()),
                              weights_ptr, scales_ptr, d_global_scale_, m, n, k,
                              hints, solution_id, nullptr);
    }
    ASSERT_EQ(err, 0);

    if (d_petit_weights) {
        CheckHIPStatus(hipFree(d_petit_weights));
    }
    if (d_petit_scales) {
        CheckHIPStatus(hipFree(d_petit_scales));
    }

    CopyAndCompareOutput(&ctx);
}

void GemmFp4Fp16Test::TestGemm(unsigned m, unsigned n, unsigned k,
                               float global_scale, SolutionId sol_id,
                               DataType b_type, unsigned group_size) {
    if (b_type == DataType::kDataTypeMxFp4e2m1) {
        ASSERT_EQ(sol_id.mfma_type, MatmulMfmaType::kMatmulMfmaTypeBf16)
            << "MXFP4 only supports BF16 accumulation";
    }

    auto data_type = sol_id.mfma_type == MatmulMfmaType::kMatmulMfmaTypeBf16
                         ? DataType::kDataTypeBf16
                         : DataType::kDataTypeFp16;
    bool require_high_precision =
        sol_id.features & MatmulFeatures::kMatmulFeatures_HighPrecision;
    TestGemmBySolutionId(m, n, k, global_scale, sol_id.Repr(), data_type,
                         b_type, require_high_precision, group_size);
}

static inline constexpr SolutionId
Fp4MNK(int features, MatmulElementB element_b, MatmulMfmaType mfma_type,
       unsigned tile_m, unsigned tile_n, unsigned tile_k, unsigned partition_m,
       unsigned partition_n, unsigned partition_k) {
    return SolutionId::MultiStage((MatmulFeatures)features, element_b,
                                  mfma_type, tile_m, tile_n, tile_k,
                                  MatmulWarpPartition::kMatmulWarpPartition_NK,
                                  partition_m, partition_n, partition_k);
}

static inline SolutionId Fp4Bf16(MatmulElementB element_b, unsigned tile_m,
                                 unsigned tile_n, unsigned tile_k,
                                 unsigned partition_m, unsigned partition_n,
                                 unsigned partition_k) {

    return Fp4MNK(MatmulFeatures::kMatmulFeatures_Grid, element_b,
                  MatmulMfmaType::kMatmulMfmaTypeBf16, tile_m, tile_n, tile_k,
                  partition_m, partition_n, partition_k);
}

static inline constexpr SolutionId Fp4Hp(MatmulMfmaType mfma_type,
                                         unsigned tile_m, unsigned tile_n,
                                         unsigned tile_k, unsigned partition_m,
                                         unsigned partition_n,
                                         unsigned partition_k) {
    return Fp4MNK(MatmulFeatures::kMatmulFeatures_Grid |
                      MatmulFeatures::kMatmulFeatures_HighPrecision,
                  MatmulElementB::kMatmulTypeBNvFp4, mfma_type, tile_m, tile_n,
                  tile_k, partition_m, partition_n, partition_k);
}

#define TEST_BF16(m, n, k, partition_m, partition_n, partition_k)                   \
    TEST_F(                                                                         \
        GemmFp4Fp16Test,                                                            \
        TestGemm_##m##x##n##x##k##_##partition_m##x##partition_n##x##partition_k) { \
        TestGemm(m, std::lcm(n, 32), std::lcm(k, 256), 1.0f,                        \
                 Fp4Bf16(MatmulElementB::kMatmulTypeBNvFp4, m / 16, n / 16,         \
                         k / 16, partition_m, partition_n, partition_k));           \
    }

// Use high precision for fp16 since MI210 flushes denormals to zero causing
// loss of precision
TEST_F(GemmFp4Fp16Test, TestGemm16x32x256Fp16HighPrecision) {
    TestGemm(16, 64, 256, 1.0f,
             Fp4Hp(MatmulMfmaType::kMatmulMfmaTypeFp16, 1, 2, 16, 1, 1, 4));
}

TEST_F(GemmFp4Fp16Test, TestGemm16x32x256Bf16HighPrecision) {
    TestGemm(16, 64, 256, 1.0f,
             Fp4Hp(MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 2, 16, 1, 1, 4));
}

TEST_BF16(64, 32, 128, 4, 1, 1)
TEST_BF16(16, 32, 256, 1, 1, 4)
TEST_BF16(32, 32, 256, 2, 1, 2)
TEST_BF16(32, 32, 256, 1, 1, 4)
TEST_BF16(64, 32, 256, 2, 1, 2)
TEST_BF16(16, 32, 512, 1, 1, 4)
TEST_BF16(32, 32, 512, 2, 1, 2)
TEST_BF16(16, 64, 128, 1, 2, 2)
TEST_BF16(32, 64, 128, 2, 2, 1)
TEST_BF16(64, 64, 128, 2, 2, 1)
TEST_BF16(96, 64, 128, 2, 2, 1)
TEST_BF16(128, 64, 128, 2, 2, 1)
TEST_BF16(160, 64, 128, 2, 2, 1)
TEST_BF16(16, 64, 256, 1, 2, 2)
TEST_BF16(32, 64, 256, 2, 2, 1)
TEST_BF16(64, 64, 256, 2, 2, 1)
TEST_BF16(16, 64, 512, 1, 2, 2)
TEST_BF16(32, 64, 512, 2, 2, 1)
TEST_BF16(64, 96, 128, 2, 1, 2)
TEST_BF16(96, 96, 128, 2, 1, 2)
TEST_BF16(16, 128, 64, 1, 4, 1)
TEST_BF16(128, 128, 64, 2, 2, 1)
TEST_BF16(192, 128, 64, 2, 2, 1)
TEST_BF16(224, 128, 64, 2, 2, 1)
TEST_BF16(256, 128, 64, 2, 2, 1)
TEST_BF16(32, 128, 128, 2, 2, 1)
TEST_BF16(64, 128, 128, 2, 2, 1)
TEST_BF16(80, 128, 128, 1, 2, 2)
TEST_BF16(160, 128, 64, 2, 2, 1)
TEST_BF16(128, 192, 64, 2, 2, 1)
TEST_BF16(160, 192, 64, 2, 2, 1)
TEST_BF16(192, 192, 64, 2, 2, 1)
TEST_BF16(224, 192, 64, 2, 2, 1)
TEST_BF16(256, 192, 64, 2, 2, 1)
TEST_BF16(128, 256, 64, 2, 2, 1)
TEST_BF16(160, 256, 64, 2, 2, 1)
TEST_BF16(192, 256, 64, 2, 2, 1)
TEST_BF16(224, 256, 64, 2, 2, 1)
TEST_BF16(256, 256, 64, 2, 2, 1)

TEST_F(GemmFp4Fp16Test, TestGemm_32x32x256_2x1x2_Pipeline_DoubleBuffer) {
    for (auto k : {512, 768, 1024}) {
        TestGemm(32, 32, k, 1.0f,
                 Fp4MNK(MatmulFeatures::kMatmulFeatures_Grid,
                        MatmulElementB::kMatmulTypeBNvFp4,
                        MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 2, 16, 2, 1,
                        2));
    }
}

TEST_F(GemmFp4Fp16Test, TestGemm_256x256x64_2x2x1_Pipeline_SingleBuffer) {
    for (auto k : {512, 768, 1024}) {
        TestGemm(256, 256, k, 1.0f,
                 Fp4MNK(MatmulFeatures::kMatmulFeatures_Grid,
                        MatmulElementB::kMatmulTypeBNvFp4,
                        MatmulMfmaType::kMatmulMfmaTypeBf16, 16, 16, 4, 2, 2,
                        1));
    }
}

TEST_F(GemmFp4Fp16Test, TestGemmMx_32x64x256_2x2x1) {
    TestGemm(32, 64, 256, 1.0f,
             Fp4Bf16(MatmulElementB::kMatmulTypeBMxFp4, 2, 4, 16, 2, 2, 1),
             DataType::kDataTypeMxFp4e2m1, 32);
}

TEST_F(GemmFp4Fp16Test, TestGemmMx_64x64x256_2x2x1) {
    TestGemm(64, 64, 256, 1.0f,
             Fp4Bf16(MatmulElementB::kMatmulTypeBMxFp4, 4, 4, 16, 2, 2, 1),
             DataType::kDataTypeMxFp4e2m1, 32);
}

TEST_F(GemmFp4Fp16Test, TestGemmMx_64x128x256_NormalPrecision) {
    TestGemm(64, 128, 256, 1.0f,
             Fp4Bf16(MatmulElementB::kMatmulTypeBMxFp4, 2, 4, 16, 2, 2, 1),
             DataType::kDataTypeMxFp4e2m1, 32);
}

TEST_F(GemmFp4Fp16Test, TestGemmMx_64x128x256_HighPrecision) {
    TestGemm(64, 128, 256, 1.0f,
             Fp4Hp(MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 16, 2, 2, 1),
             DataType::kDataTypeMxFp4e2m1, 32);
}

} // namespace causalflow::petit::rocm::quantization::fp4
