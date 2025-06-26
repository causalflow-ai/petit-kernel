#include "gemm/rocm/quantization/gemm.h"
#include "gemm_fp4.h"
#include "tests/quantization.h"
#include "utils/hip_helper.h"

#include <climits>
#include <gtest/gtest.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>

namespace causalflow::petit::rocm::quantization::fp4 {

static inline void CheckHipblasStatus(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "HipBLAS Error: " << status << std::endl;
        throw std::runtime_error("HipBLAS Error");
    }
}

using GemmMPTestData = tests::quantization::GemmMPTestData;

class GemmFp4Fp16Test : public ::testing::Test {
  public:
    static constexpr size_t kWorkspaceSize = 32 * 1024 * 1024;
    void SetUp() override;
    void TearDown() override;

    void ComputeReference(GemmMPTestData *ctx) const;
    void CopyAndCompareOutput(GemmMPTestData *ctx) const;
    void TestGemm(unsigned m, unsigned n, unsigned k, float global_scale,
                  SolutionId sol_id);

    hipblasLtHandle_t handle_;
    hipblasLtMatmulDesc_t matmul_desc_;
    void *d_workspace_;
    std::unique_ptr<hal::Device> dev_;
    DataType dequant_type_;
};

void GemmFp4Fp16Test::SetUp() {
    static constexpr hipblasOperation_t kTransposed = HIPBLAS_OP_T;
    CheckHIPStatus(hipMalloc(&d_workspace_, kWorkspaceSize));

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
            float of = __half2float(output_ptr[0]);
            float rf = __half2float(ref_ptr[0]);
            EXPECT_NEAR(of, rf, std::min<float>(1e-2, fabs(rf) * 0.01f))
                << "Output and reference differ at index " << i;
        } else if (dequant_type_ == DataType::kDataTypeBf16) {
            unsigned o = (unsigned)h_output[i] << 16,
                     r = (unsigned)h_reference[i] << 16;
            float of = reinterpret_cast<const float &>(o);
            float rf = reinterpret_cast<const float &>(r);
            EXPECT_NEAR(of, rf, std::min<float>(1e-2, fabs(rf) * 0.01f))
                << "Output and reference differ at index " << i;
        }
    }
}

void GemmFp4Fp16Test::TestGemm(unsigned m, unsigned n, unsigned k,
                               float global_scale, SolutionId sol_id) {
    if (sol_id.mfma_type == MatmulMfmaType::kMatmulMfmaTypeBf16) {
        dequant_type_ = DataType::kDataTypeBf16;
    } else {
        dequant_type_ = DataType::kDataTypeFp16;
    }
    GemmMPTestData ctx(dev_.get(), dequant_type_, DataType::kDataTypeFp4e2m1, m,
                       n, k, 16);
    ASSERT_EQ(absl::OkStatus(), ctx.PrepareData(false));

    DequantPetitFp4(reinterpret_cast<unsigned *>(ctx.weights()),
                    reinterpret_cast<const unsigned *>(ctx.weights_quant()),
                    reinterpret_cast<const unsigned *>(ctx.scales()),
                    global_scale, dequant_type_, k, n);
    ComputeReference(&ctx);

    auto data_type = sol_id.mfma_type == MatmulMfmaType::kMatmulMfmaTypeBf16
                         ? DataType::kDataTypeBf16
                         : DataType::kDataTypeFp16;
    PetitSolutionHints hints;
    hints.a_type = data_type;
    hints.b_type = DataType::kDataTypeFp4e2m1;
    hints.c_type = data_type;
    hints.require_high_precision =
        sol_id.features & MatmulFeatures::kMatmulFeatures_HighPrecision;

    int err =
        GemmFp4Fp16Grid(reinterpret_cast<unsigned *>(ctx.output()),
                        reinterpret_cast<const unsigned *>(ctx.input()),
                        reinterpret_cast<const unsigned *>(ctx.weights_quant()),
                        reinterpret_cast<const unsigned *>(ctx.scales()),
                        global_scale, m, n, k, hints, sol_id.Repr(), nullptr);
    ASSERT_EQ(err, 0);
    CopyAndCompareOutput(&ctx);
}

static inline constexpr SolutionId
Fp4MNK(int features, MatmulPipeline pipeline, MatmulMfmaType mfma_type,
       unsigned tile_m, unsigned tile_n, unsigned tile_k, unsigned partition_m,
       unsigned partition_n, unsigned partition_k) {
    return SolutionId::MultiStage(pipeline, (MatmulFeatures)features,
                                  MatmulElementB::kMatmulTypeBFp4, mfma_type,
                                  tile_m, tile_n, tile_k,
                                  MatmulWarpPartition::kMatmulWarpPartition_NK,
                                  partition_m, partition_n, partition_k);
}

static inline constexpr SolutionId Fp4(MatmulPipeline pipeline,
                                       MatmulMfmaType mfma_type,
                                       unsigned tile_m, unsigned tile_n,
                                       unsigned tile_k) {
    return Fp4MNK(MatmulFeatures::kMatmulFeatures_Grid, pipeline, mfma_type,
                  tile_m, tile_n, tile_k, 1, tile_n, 4 / tile_n);
}

static inline constexpr SolutionId Fp4Hp(MatmulPipeline pipeline,
                                         MatmulMfmaType mfma_type,
                                         unsigned tile_m, unsigned tile_n,
                                         unsigned tile_k) {
    return Fp4MNK(MatmulFeatures::kMatmulFeatures_Grid |
                      MatmulFeatures::kMatmulFeatures_HighPrecision,
                  pipeline, mfma_type, tile_m, tile_n, tile_k, 1, tile_n,
                  4 / tile_n);
}

// Use high precision for fp16 since MI210 flushes denormals to zero causing
// loss of precision
TEST_F(GemmFp4Fp16Test, TestGemm16x32x256Fp16HighPrecision) {
    TestGemm(16, 64, 256, 1.0f,
             Fp4Hp(MatmulPipeline::kMatmulPipeline_2,
                   MatmulMfmaType::kMatmulMfmaTypeFp16, 1, 2, 16));
}

TEST_F(GemmFp4Fp16Test, TestGemm16x64x128Bf16HighPrecision) {
    TestGemm(16, 64, 128, 1.0f,
             Fp4Hp(MatmulPipeline::kMatmulPipeline_2,
                   MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 4, 8));
}

TEST_F(GemmFp4Fp16Test, TestGemm16x64x128Bf16) {
    TestGemm(16, 64, 128, 1.0f,
             Fp4(MatmulPipeline::kMatmulPipeline_2,
                 MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 4, 8));
}

TEST_F(GemmFp4Fp16Test, TestGemm32x64x128Bf16) {
    TestGemm(64, 128, 256, 1.0f,
             Fp4(MatmulPipeline::kMatmulPipeline_2,
                 MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 8));
}

TEST_F(GemmFp4Fp16Test, TestGemm64x64x128Bf16) {
    TestGemm(64, 64, 128, 1.0f,
             Fp4(MatmulPipeline::kMatmulPipeline_2,
                 MatmulMfmaType::kMatmulMfmaTypeBf16, 4, 4, 8));
}

TEST_F(GemmFp4Fp16Test, TestGemm16x32x256Bf16) {
    TestGemm(16, 64, 256, 1.0f,
             Fp4(MatmulPipeline::kMatmulPipeline_2,
                 MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 2, 16));
}

TEST_F(GemmFp4Fp16Test, TestGemm32x32x256Bf16) {
    TestGemm(64, 128, 512, 1.0f,
             Fp4(MatmulPipeline::kMatmulPipeline_2,
                 MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 2, 16));
}

TEST_F(GemmFp4Fp16Test, TestGemm64x32x256Bf16) {
    TestGemm(64, 64, 256, 1.0f,
             Fp4(MatmulPipeline::kMatmulPipeline_1,
                 MatmulMfmaType::kMatmulMfmaTypeBf16, 4, 2, 16));
}

TEST_F(GemmFp4Fp16Test, TestGemm16x64x512) {
    TestGemm(16, 64, 512, 1.0f,
             Fp4(MatmulPipeline::kMatmulPipeline_2,
                 MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 1, 32));
}

TEST_F(GemmFp4Fp16Test, TestGemm16x64x512_SingleStage) {
    TestGemm(32, 64, 512, 1.0f,
             Fp4(MatmulPipeline::kMatmulPipeline_1,
                 MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 1, 32));
}

TEST_F(GemmFp4Fp16Test, TestGemm16x64x512_16x32x512) {
    TestGemm(16, 64, 512, 1.0f,
             Fp4MNK(MatmulFeatures::kMatmulFeatures_Grid,
                    MatmulPipeline::kMatmulPipeline_1,
                    MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 4, 32, 1, 2, 1));
}

TEST_F(GemmFp4Fp16Test, TestGemm32x64x512_32x32x512) {
    TestGemm(32, 64, 512, 1.0f,
             Fp4MNK(MatmulFeatures::kMatmulFeatures_Grid,
                    MatmulPipeline::kMatmulPipeline_1,
                    MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 32, 1, 2, 1));
}

TEST_F(GemmFp4Fp16Test, TestGemm32x64x512_16x32x512) {
    TestGemm(32, 64, 512, 1.0f,
             Fp4MNK(MatmulFeatures::kMatmulFeatures_Grid,
                    MatmulPipeline::kMatmulPipeline_1,
                    MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 32, 2, 2, 1));
}

TEST_F(GemmFp4Fp16Test, TestGemm16x64x512_16x32x256) {
    TestGemm(16, 64, 512, 1.0f,
             Fp4MNK(MatmulFeatures::kMatmulFeatures_Grid,
                    MatmulPipeline::kMatmulPipeline_1,
                    MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 4, 32, 1, 2, 2));
}

TEST_F(GemmFp4Fp16Test, TestGemm32x64x512_32x32x256) {
    TestGemm(32, 64, 512, 1.0f,
             Fp4MNK(MatmulFeatures::kMatmulFeatures_Grid,
                    MatmulPipeline::kMatmulPipeline_1,
                    MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 32, 1, 2, 2));
}

TEST_F(GemmFp4Fp16Test, TestGemm32x64x512_16x32x256) {
    TestGemm(32, 64, 512, 1.0f,
             Fp4MNK(MatmulFeatures::kMatmulFeatures_Grid,
                    MatmulPipeline::kMatmulPipeline_1,
                    MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 32, 2, 2, 2));
}

} // namespace causalflow::petit::rocm::quantization::fp4