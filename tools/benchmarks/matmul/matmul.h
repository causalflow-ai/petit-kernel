#include "hal/device.h"

#include <absl/status/status.h>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace causalflow::petit::benchmark::matmul {

class Matmul {
  public:
    enum DataType {
        kFp8e5m2,
        kFp8e4m3,
        kFp16,
        kBf16,
        kFp32,
    };

    enum class BType {
        kFp16,
        kBf16,
        kNvFp4,
        kMxFp4,
    };

    static constexpr int kNvFp4GroupSize = 16;
    static constexpr int kMxFp4GroupSize = 32;

    static inline std::optional<BType> ParseBType(std::string_view b_type) {
        if (b_type == "fp16") {
            return BType::kFp16;
        }
        if (b_type == "bf16") {
            return BType::kBf16;
        }
        if (b_type == "nvfp4") {
            return BType::kNvFp4;
        }
        if (b_type == "mxfp4") {
            return BType::kMxFp4;
        }
        return std::nullopt;
    }

    static inline std::optional<DataType> GetDenseBDataType(BType b_type) {
        switch (b_type) {
        case BType::kFp16:
            return DataType::kFp16;
        case BType::kBf16:
            return DataType::kBf16;
        case BType::kNvFp4:
        case BType::kMxFp4:
            return std::nullopt;
        }
        return std::nullopt;
    }

    static inline std::optional<int> GetFp4GroupSize(BType b_type) {
        switch (b_type) {
        case BType::kNvFp4:
            return kNvFp4GroupSize;
        case BType::kMxFp4:
            return kMxFp4GroupSize;
        case BType::kFp16:
        case BType::kBf16:
            return std::nullopt;
        }
        return std::nullopt;
    }

    static inline bool IsHipBLASLtCompatibleBType(BType b_type) {
        return GetDenseBDataType(b_type).has_value();
    }

    static inline bool IsPetitCompatibleBType(BType b_type) {
        return GetFp4GroupSize(b_type).has_value();
    }

    // Describe an algorithm used in the GEMM operation. Note that the indicies
    // are potentially non-deterministic.
    struct AlgorithmDescriptor {
        enum { kDefault, kIndex, kOpaqueRepresentation } tag;
        std::string repr;
    };
    // Stride batch GEMM
    virtual absl::Status PrepareForBatchExecution(void *d, const void *a,
                                                  const void *b, const void *c,
                                                  long stride_a, long stride_b,
                                                  long stride_c,
                                                  int batch_count) = 0;

    // Enumerate algorithms for tuning
    virtual absl::Status EnumerateAlgorithms() { return absl::OkStatus(); }
    virtual size_t GetAlgorithmCount() const { return 0; }
    virtual std::string GetAlgorithmRepr(size_t index) const { return ""; }

    virtual absl::Status SetAlgorithm(AlgorithmDescriptor algo) = 0;
    virtual absl::Status Execute(size_t repeat) = 0;
    virtual ~Matmul() = default;
};

class MatmulFactory {
  public:
    static std::unique_ptr<MatmulFactory> Create(const std::string &backend);
    virtual const char *GetPlatformName() const = 0;
    virtual absl::Status CreateMatmul(hal::Device *dev, Matmul::DataType a_type,
                                      Matmul::DataType c_type,
                                      Matmul::BType b_type, int m, int n,
                                      int k,
                                      std::unique_ptr<Matmul> *result) = 0;
    virtual ~MatmulFactory() = default;
};

} // namespace causalflow::petit::benchmark::matmul
