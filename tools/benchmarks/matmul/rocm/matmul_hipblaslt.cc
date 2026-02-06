#include "matmul.h"
#include "utils/hip_helper.h"
#include "utils/monad_runner.h"

#include <absl/status/status.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>

namespace causalflow::petit::benchmark::matmul {

namespace rocm {

static inline void CheckHipblasStatus(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "HipBLAS Error: " << status << std::endl;
        throw std::runtime_error("HipBLAS Error");
    }
}

class HipBLASLtStridedBatchMatmul : public Matmul {
  public:
    static constexpr size_t kWorkspaceSize = 32 * 1024 * 1024;
    explicit HipBLASLtStridedBatchMatmul(int m, int n, int k,
                                         const Matmul::DataType a_type,
                                         const Matmul::DataType c_type);

    absl::Status PrepareForBatchExecution(void *d, const void *a, const void *b,
                                          const void *c, long stride_a,
                                          long stride_b, long stride_c,
                                          int batch) override;
    absl::Status EnumerateAlgorithms() override;

    virtual size_t GetAlgorithmCount() const override {
        return algorithms_.size();
    }

    virtual std::string GetAlgorithmRepr(size_t index) const override {
        return std::string(reinterpret_cast<const char *>(&algorithms_[index]),
                           sizeof(hipblasLtMatmulAlgo_t));
    }

    virtual absl::Status SetAlgorithm(AlgorithmDescriptor algo) override;
    absl::Status Execute(size_t repeat) override;
    virtual ~HipBLASLtStridedBatchMatmul();

  private:
    absl::Status Close();

    hipblasLtHandle_t handle_;
    void *d_workspace_;
    void *d_d_;
    const void *d_a_, *d_b_, *d_c_;

    hipblasLtMatrixLayout_t layout_a_, layout_b_, layout_c_;
    hipblasLtMatmulDesc_t matmul_desc_;
    hipblasLtMatmulAlgo_t algo_;

    std::vector<hipblasLtMatmulAlgo_t> algorithms_;
};

class HipBLASLtMatmulFactory : public MatmulFactory {
  public:
    virtual const char *GetPlatformName() const override { return "rocm"; }
    virtual absl::Status
    CreateMatmul(hal::Device *dev, const Matmul::DataType a_type,
                 const Matmul::DataType c_type, int m, int n, int k,
                 std::unique_ptr<Matmul> *result) override {
        *result = std::make_unique<HipBLASLtStridedBatchMatmul>(m, n, k, a_type,
                                                                c_type);
        return absl::OkStatus();
    }

    virtual ~HipBLASLtMatmulFactory() = default;
};

static hipDataType GetHipblasDataType(const Matmul::DataType type) {
    switch (type) {
    case Matmul::DataType::kFp32:
        return HIP_R_32F;
    case Matmul::DataType::kFp16:
        return HIP_R_16F;
    case Matmul::DataType::kBf16:
        return HIP_R_16BF;
    case Matmul::DataType::kFp8e5m2:
        return HIP_R_8F_E5M2_FNUZ;
    case Matmul::DataType::kFp8e4m3:
        return HIP_R_8F_E4M3_FNUZ;
    }
    throw std::runtime_error("Unknown data type");
}

HipBLASLtStridedBatchMatmul::HipBLASLtStridedBatchMatmul(
    int m, int n, int k, const Matmul::DataType a_type,
    const Matmul::DataType c_type) {
    static constexpr hipblasOperation_t kTransposed = HIPBLAS_OP_T;

    CheckHipblasStatus(hipblasLtCreate(&handle_));
    CheckHipblasStatus(hipblasLtMatmulDescCreate(
        &matmul_desc_, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    // Let A be in row-major and B in col-major to keep consistency with
    // densse-linears.
    CheckHipblasStatus(hipblasLtMatmulDescSetAttribute(
        matmul_desc_, HIPBLASLT_MATMUL_DESC_TRANSA, &kTransposed,
        sizeof(hipblasOperation_t)));

    CheckHipblasStatus(hipblasLtMatrixLayoutCreate(
        &layout_a_, GetHipblasDataType(a_type), k, m, k));
    CheckHipblasStatus(hipblasLtMatrixLayoutCreate(
        &layout_b_, GetHipblasDataType(a_type), k, n, k));
    CheckHipblasStatus(hipblasLtMatrixLayoutCreate(
        &layout_c_, GetHipblasDataType(c_type), n, m, n));

    CheckHIPStatus(hipMalloc(&d_workspace_, kWorkspaceSize));
    hipblasLtMatmulPreference_t preference = nullptr;
    CheckHipblasStatus(hipblasLtMatmulPreferenceCreate(&preference));

    hipblasLtMatmulHeuristicResult_t res;

    int r = 0;
    CheckHipblasStatus(hipblasLtMatmulAlgoGetHeuristic(
        handle_, matmul_desc_, layout_a_, layout_b_, layout_c_, layout_c_,
        preference, 1, &res, &r));
    algo_ = res.algo;
    CheckHipblasStatus(hipblasLtMatmulPreferenceDestroy(preference));
}

HipBLASLtStridedBatchMatmul::~HipBLASLtStridedBatchMatmul() {
    auto stat = Close();
    (void)stat;
}

absl::Status HipBLASLtStridedBatchMatmul::Close() {
    CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_a_));
    CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_b_));
    CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_c_));
    CheckHipblasStatus(hipblasLtMatmulDescDestroy(matmul_desc_));
    CheckHipblasStatus(hipblasLtDestroy(handle_));
    CheckHIPStatus(hipFree(d_workspace_));
    return absl::OkStatus();
}

absl::Status HipBLASLtStridedBatchMatmul::PrepareForBatchExecution(
    void *d, const void *a, const void *b, const void *c, long stride_a,
    long stride_b, long stride_c, int batch) {
    d_d_ = d;
    d_a_ = a;
    d_b_ = b;
    d_c_ = c;

    struct {
        hipblasLtMatrixLayout_t layout;
        long stride;
    } mats[] = {
        {layout_a_, stride_a},
        {layout_b_, stride_b},
        {layout_c_, stride_c},
    };

    for (const auto m : mats) {
        CheckHipblasStatus(hipblasLtMatrixLayoutSetAttribute(
            m.layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch,
            sizeof(batch)));
        CheckHipblasStatus(hipblasLtMatrixLayoutSetAttribute(
            m.layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &m.stride,
            sizeof(m.stride)));
    }

    return absl::OkStatus();
}

absl::Status
HipBLASLtStridedBatchMatmul::SetAlgorithm(AlgorithmDescriptor algo_desc) {
    if (algo_desc.tag == AlgorithmDescriptor::kDefault) {
        hipblasLtMatmulPreference_t preference = nullptr;
        CheckHipblasStatus(hipblasLtMatmulPreferenceCreate(&preference));

        hipblasLtMatmulHeuristicResult_t res;

        CheckHipblasStatus(hipblasLtMatmulPreferenceSetAttribute(
            preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &kWorkspaceSize, sizeof(kWorkspaceSize)));

        int r = 0;
        CheckHipblasStatus(hipblasLtMatmulAlgoGetHeuristic(
            handle_, matmul_desc_, layout_a_, layout_b_, layout_c_, layout_c_,
            preference, 1, &res, &r));
        algo_ = res.algo;
        CheckHipblasStatus(hipblasLtMatmulPreferenceDestroy(preference));
    } else if (algo_desc.tag == AlgorithmDescriptor::kIndex) {
        auto idx = stoi(algo_desc.repr);
        algo_ = algorithms_.at(idx);
    } else if (algo_desc.tag == AlgorithmDescriptor::kOpaqueRepresentation) {
        if (algo_desc.repr.size() != sizeof(hipblasLtMatmulAlgo_t)) {
            return absl::InvalidArgumentError(
                "Invalid hipBLASLt algorithm representation size");
        }
        algo_ = *reinterpret_cast<const hipblasLtMatmulAlgo_t *>(
            algo_desc.repr.c_str());

        if (algo_.max_workspace_bytes == 0 ||
            algo_.max_workspace_bytes > kWorkspaceSize) {
            return absl::InvalidArgumentError("Incompatible algorithm");
        }
    }
    return absl::OkStatus();
}

absl::Status HipBLASLtStridedBatchMatmul::EnumerateAlgorithms() {
    enum { kMaxAlgorithms = 10240 };
    std::vector<hipblasLtMatmulHeuristicResult_t> algo_ids(kMaxAlgorithms);
    int num_algo_ids = 0;
    hipblasLtMatmulPreference_t preference = nullptr;
    CheckHipblasStatus(hipblasLtMatmulPreferenceCreate(&preference));
    CheckHipblasStatus(hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceSize,
        sizeof(kWorkspaceSize)));
    CheckHipblasStatus(hipblasLtMatmulAlgoGetHeuristic(
        handle_, matmul_desc_, layout_a_, layout_b_, layout_c_, layout_c_,
        preference, kMaxAlgorithms, algo_ids.data(), &num_algo_ids));

    algorithms_.clear();
    algo_ids.resize(num_algo_ids);
    std::vector<hipblasLtMatmulAlgo_t> algos;
    for (const auto &algo : algo_ids) {
        if (algo.state == HIPBLAS_STATUS_SUCCESS) {
            if (algo.workspaceSize > kWorkspaceSize) {
                std::cerr << "Workspace size is too large: "
                          << algo.workspaceSize << std::endl;
                throw std::runtime_error("Workspace size is too large");
            }
            algorithms_.emplace_back(algo.algo);
        }
    }

    CheckHipblasStatus(hipblasLtMatmulPreferenceDestroy(preference));
    return absl::OkStatus();
}

absl::Status HipBLASLtStridedBatchMatmul::Execute(size_t repeat) {
    static constexpr float kAlpha = 1.0f;
    static constexpr float kBeta = 0.0f;

    for (size_t i = 0; i < repeat; ++i) {
        CheckHipblasStatus(hipblasLtMatmul(
            handle_, matmul_desc_, &kAlpha, d_b_, layout_b_, d_a_, layout_a_,
            &kBeta, d_c_, layout_c_, d_d_, layout_c_, &algo_, d_workspace_,
            kWorkspaceSize, nullptr));
    }

    return absl::OkStatus();
}

} // namespace rocm

std::unique_ptr<MatmulFactory> CreateMatmulFactoryHipBLASLtBackend() {
    return std::unique_ptr<MatmulFactory>(new rocm::HipBLASLtMatmulFactory);
}

} // namespace causalflow::petit::benchmark::matmul
