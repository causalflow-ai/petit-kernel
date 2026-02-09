#include "gemm/rocm/quantization/gemm.h"
#include "matmul.h"
#include "tests/quantization.h"
#include "utils/hip_helper.h"

#include <absl/status/status.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit {

namespace benchmark::matmul {

namespace rocm {

using causalflow::petit::rocm::quantization::MatmulElementB;
using causalflow::petit::rocm::quantization::MatmulFeatures;
using causalflow::petit::rocm::quantization::MatmulMfmaType;
using causalflow::petit::rocm::quantization::MatmulWarpPartition;
using causalflow::petit::rocm::quantization::PetitSolutionHints;
using causalflow::petit::rocm::quantization::SolutionId;

namespace fp4 = causalflow::petit::rocm::quantization::fp4;
using GemmDataType = causalflow::petit::rocm::quantization::DataType;

static inline GemmDataType GetFp4GemmBTypeFromGroupSize(int groupsize) {
    if (groupsize == Matmul::kMxFp4GroupSize) {
        return GemmDataType::kDataTypeMxFp4e2m1;
    }
    return GemmDataType::kDataTypeFp4e2m1;
}

static inline auto GetFp4GemmKernelFromGroupSize(int groupsize)
    -> decltype(&fp4::GemmFp4Fp16Grid) {
    if (groupsize == Matmul::kMxFp4GroupSize) {
        return fp4::GemmMxFp4Fp16Grid;
    }
    return fp4::GemmFp4Fp16Grid;
}

class PetitMatmulFp4Base : public Matmul {
  public:
    explicit PetitMatmulFp4Base(int m, int n, int k, DataType a_type,
                                int groupsize = 16);

    virtual void FreeQuantizedMemory();
    absl::Status PrepareForBatchExecution(void *d, const void *a, const void *b,
                                          const void *c, long stride_a,
                                          long stride_b, long stride_c,
                                          int batch) override;

    virtual size_t GetAlgorithmCount() const override {
        return available_sols_.size();
    }

    virtual std::string GetAlgorithmRepr(size_t index) const override {
        auto sol = available_sols_[index].Repr();
        return std::string(reinterpret_cast<const char *>(&sol), sizeof(sol));
    }

    virtual absl::Status SetAlgorithm(AlgorithmDescriptor algo_desc) override {
        if (algo_desc.tag == AlgorithmDescriptor::kDefault) {
            algo_ = available_sols_[0];
        } else if (algo_desc.tag == AlgorithmDescriptor::kIndex) {
            auto idx = stoi(algo_desc.repr);
            algo_ = available_sols_[idx];
        } else if (algo_desc.tag ==
                   AlgorithmDescriptor::kOpaqueRepresentation) {
            algo_ =
                *reinterpret_cast<const SolutionId *>(algo_desc.repr.c_str());
        }
        return absl::OkStatus();
    }

    virtual ~PetitMatmulFp4Base() { FreeQuantizedMemory(); }

    static inline GemmDataType GetGemmDataType(DataType a_type) {
        GemmDataType real_a_type;
        if (a_type == DataType::kFp16) {
            real_a_type = GemmDataType::kDataTypeFp16;
        } else if (a_type == DataType::kBf16) {
            real_a_type = GemmDataType::kDataTypeBf16;
        } else {
            throw std::runtime_error("Invalid input type");
        }
        return real_a_type;
    }

    int m_, n_, k_;
    int groupsize_;
    long stride_a_, stride_b_, stride_c_;
    long stride_b_quant_, stride_scales_;
    void *d_d_;
    const void *d_a_, *d_b_;
    unsigned *d_b_quant_, *d_scales_;
    void *d_workspace_;
    float *d_global_scale_;
    int batch_;
    SolutionId algo_;
    std::vector<SolutionId> available_sols_;
};

class PetitMatmulFp4Fp16 : public PetitMatmulFp4Base {
  public:
    explicit PetitMatmulFp4Fp16(int m, int n, int k, int groupsize,
                                DataType a_type)
        : PetitMatmulFp4Base(m, n, k, a_type, groupsize) {
        auto gemm_a_type = GetGemmDataType(a_type);
        hints_.a_type = gemm_a_type;
        hints_.b_type = GetFp4GemmBTypeFromGroupSize(groupsize_);
        hints_.c_type = gemm_a_type;
        hints_.require_high_precision = false;
    }
    using ElementA = unsigned short;

    absl::Status Execute(size_t repeat) override {
        auto gemm = GetFp4GemmKernelFromGroupSize(groupsize_);
        for (size_t i = 0; i < repeat; ++i) {
            for (int b = 0; b < batch_; ++b) {
                gemm(
                    reinterpret_cast<unsigned *>(d_d_) +
                        b * stride_c_ / (sizeof(unsigned) / sizeof(half)),
                    reinterpret_cast<const unsigned *>(d_a_) +
                        b * stride_a_ / (sizeof(unsigned) / sizeof(ElementA)),
                    reinterpret_cast<const unsigned *>(d_b_quant_) +
                        b * stride_b_quant_ / sizeof(unsigned),
                    d_scales_ + b * stride_scales_ / sizeof(unsigned),
                    d_global_scale_, m_, n_, k_, hints_, algo_.Repr(), nullptr);
            }
        }
        return absl::OkStatus();
    }

  private:
    PetitSolutionHints hints_;
};

PetitMatmulFp4Base::PetitMatmulFp4Base(int m, int n, int k, DataType a_type,
                                       int groupsize)
    : m_(m), n_(n), k_(k), groupsize_(groupsize), d_b_quant_(nullptr),
      d_scales_(nullptr), d_global_scale_(nullptr) {
    static constexpr unsigned kMaxSols = 1024;
    available_sols_.resize(kMaxSols);
    unsigned available_sols_count = available_sols_.size();
    GemmDataType gemm_a_type = GetGemmDataType(a_type);
    PetitSolutionHints hints;
    hints.a_type = gemm_a_type;
    hints.b_type = GetFp4GemmBTypeFromGroupSize(groupsize_);
    hints.c_type = gemm_a_type;
    hints.require_high_precision = false;
    int err = fp4::GemmGetSolutions(hints, m, n, k, available_sols_.data(),
                                    &available_sols_count);
    if (err != 0) {
        throw std::runtime_error("Failed to get solutions");
    }
    available_sols_.resize(available_sols_count);
}

absl::Status PetitMatmulFp4Base::PrepareForBatchExecution(
    void *d, const void *a, const void *b, const void *c, long stride_a,
    long stride_b, long stride_c, int batch) {
    static constexpr float kGlobalScale = 1.0f;

    // Check alignment requirements
    if (n_ % 16 != 0 || k_ % groupsize_ != 0) {
        return absl::InvalidArgumentError(
            "m, n must be multiples of 16, k must be multiple of groupsize");
    }

    d_d_ = d;
    d_a_ = a;
    d_b_ = b;
    batch_ = batch;
    stride_a_ = stride_a;
    stride_b_ = stride_b;
    stride_b_quant_ = stride_b / 4;
    stride_scales_ = n_ * k_ / groupsize_;
    stride_c_ = stride_c;

    FreeQuantizedMemory();
    CheckHIPStatus(hipMalloc(&d_global_scale_, sizeof(float)));
    CheckHIPStatus(hipMalloc(&d_b_quant_, n_ * k_ * batch_ / 2));
    CheckHIPStatus(hipMalloc(&d_scales_, n_ * k_ * batch_ / groupsize_ *
                                             sizeof(unsigned char)));

    std::vector<unsigned char> h_scales(n_ * k_ * batch_ / groupsize_);
    std::vector<unsigned> h_qweights(k_ * n_ * batch_ / (32 / 4));

    causalflow::petit::tests::quantization::GenerateQuantizedWeightsFp4(
        m_, n_, k_, groupsize_, &h_qweights, h_scales);

    CheckHIPStatus(hipMemcpy(d_scales_, h_scales.data(),
                             h_scales.size() * sizeof(unsigned char),
                             hipMemcpyHostToDevice));

    if (groupsize_ == Matmul::kMxFp4GroupSize) {
        unsigned *d_scales_repacked = nullptr;
        CheckHIPStatus(hipMalloc(
            &d_scales_repacked,
            n_ * k_ * batch_ / groupsize_ * sizeof(unsigned char)));
        for (int b = 0; b < batch_; ++b) {
            fp4::RepackMxFp4ToPetitFp4Scales(
                d_scales_repacked + b * stride_scales_ / sizeof(unsigned),
                d_scales_ + b * stride_scales_ / sizeof(unsigned), k_, n_,
                nullptr);
        }
        CheckHIPStatus(hipFree(d_scales_));
        d_scales_ = d_scales_repacked;
    }
    CheckHIPStatus(hipMemcpy(d_b_quant_, h_qweights.data(),
                             h_qweights.size() * sizeof(unsigned),
                             hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(d_global_scale_, &kGlobalScale, sizeof(float),
                             hipMemcpyHostToDevice));

    return absl::OkStatus();
}

void PetitMatmulFp4Base::FreeQuantizedMemory() {
    if (d_b_quant_) {
        CheckHIPStatus(hipFree(d_b_quant_));
    }
    if (d_scales_) {
        CheckHIPStatus(hipFree(d_scales_));
    }
    if (d_global_scale_) {
        CheckHIPStatus(hipFree(d_global_scale_));
    }
    d_b_quant_ = nullptr;
    d_scales_ = nullptr;
    d_global_scale_ = nullptr;
}

class PetitMatmulFactory : public MatmulFactory {
  public:
    virtual const char *GetPlatformName() const override { return "rocm"; }
    virtual absl::Status
    CreateMatmul(hal::Device *, const Matmul::DataType a_type,
                 const Matmul::DataType c_type, const Matmul::BType b_type,
                 int m, int n, int k,
                 std::unique_ptr<Matmul> *result) override {
        if ((a_type == Matmul::DataType::kFp16 &&
             c_type == Matmul::DataType::kFp16) ||
            (a_type == Matmul::DataType::kBf16 &&
             c_type == Matmul::DataType::kBf16)) {
            auto groupsize = Matmul::GetFp4GroupSize(b_type);
            if (!groupsize.has_value()) {
                return absl::InvalidArgumentError(
                    "Invalid btype for petit backend. Expected nvfp4 or "
                    "mxfp4");
            }
            *result =
                std::make_unique<PetitMatmulFp4Fp16>(m, n, k, *groupsize, a_type);
        } else {
            return absl::InvalidArgumentError(
                "Invalid combination of input and output types");
        }
        return absl::OkStatus();
    }
};

} // namespace rocm

std::unique_ptr<MatmulFactory> CreateMatmulFactoryPetitBackend() {
    return std::unique_ptr<MatmulFactory>(new rocm::PetitMatmulFactory);
}

} // namespace benchmark::matmul

} // namespace causalflow::petit
