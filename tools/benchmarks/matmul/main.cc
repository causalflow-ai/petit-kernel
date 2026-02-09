#include "gemm/cpu/half_float.h"
#include "hal/device.h"
#include "matmul.h"
#include "utils/monad_runner.h"
#include "utils/test_utils.h"

#include <absl/status/status.h>
#include <absl/strings/escaping.h>
#include <chrono>
#include <cstdio>
#include <fmt/core.h>
#include <gflags/gflags.h>
#include <memory>
#include <random>
#include <string_view>

DEFINE_string(backend, "hipblaslt",
              "Backend to use for matmul. Available: "
              "rocblas, hipblaslt, petit");
DEFINE_int32(m, 128, "Number of rows in A and C");
DEFINE_int32(n, 4096, "Number of columns in B and C");
DEFINE_int32(k, 4096, "Number of columns in A and rows in B");
DEFINE_int32(batch, 1, "Number of concurrent matmul to be performed");
DEFINE_int32(warmup, 10, "Number of warmup iterations");
DEFINE_int32(repeat, 100, "Number of iterations to repeat");
DEFINE_string(algo, "",
              "Choose specific algorithm to run the GEMM. You can specify "
              "'tune' if you want to run the tuning process");
DEFINE_string(atype, "fp16",
              "The type of the inputs (fp8e5m2, fp8e4m3, fp16, bf16, fp32)");
DEFINE_string(ctype, "fp16",
              "The type of the results (fp8e5m2, fp8e4m3, fp16, bf16, fp32)");
DEFINE_string(btype, "fp16",
              "The type of matrix B. For petit: nvfp4/mxfp4. "
              "For hipblaslt: fp16/bf16");

using namespace causalflow::petit::benchmark;
namespace hal = causalflow::petit::hal;
using AlgorithmDescriptor = matmul::Matmul::AlgorithmDescriptor;
using DataType = matmul::Matmul::DataType;
using BType = matmul::Matmul::BType;
using causalflow::petit::FillRandomValue;

static constexpr std::string_view kSupportedDataTypes =
    "fp8e5m2, fp8e4m3, fp16, bf16, fp32";
static constexpr std::string_view kSupportedBTypes =
    "fp16, bf16, nvfp4, mxfp4";

class DataTypeTrait {
  public:
    virtual ~DataTypeTrait() = default;
    virtual size_t GetTypeBits() const = 0;
    virtual DataType GetDataType() const = 0;
    virtual const char *GetName() const = 0;
    virtual void FillRandom(std::mt19937 &e, void *data, size_t size) const = 0;
};

class Fp16Trait : public DataTypeTrait {
  public:
    size_t GetTypeBits() const override { return 16; }
    const char *GetName() const override { return "fp16"; }
    DataType GetDataType() const override { return DataType::kFp16; }
    void FillRandom(std::mt19937 &e, void *data, size_t size) const override {
        std::uniform_real_distribution<> dist(-1, 1);
        auto gen_half = [&]() { return half_float::half(dist(e)); };
        FillRandomValue(
            gen_half,
            std::span(reinterpret_cast<half_float::half *>(data), size));
    }
};

class Bf16Trait : public DataTypeTrait {
  public:
    size_t GetTypeBits() const override { return 16; }
    const char *GetName() const override { return "bf16"; }
    DataType GetDataType() const override { return DataType::kBf16; }
    void FillRandom(std::mt19937 &e, void *data, size_t size) const override {
        std::uniform_int_distribution<> dis(0, 65535);
        auto gen_float = [&]() { return dis(e) & ~0x8000; };
        FillRandomValue(
            gen_float,
            std::span(reinterpret_cast<unsigned short *>(data), size));
    }
};

class Fp32Trait : public DataTypeTrait {
  public:
    size_t GetTypeBits() const override { return 32; }
    const char *GetName() const override { return "fp32"; }
    DataType GetDataType() const override { return DataType::kFp32; }
    void FillRandom(std::mt19937 &e, void *data, size_t size) const override {
        std::uniform_real_distribution<> dis(-1, 1);
        auto gen_float = [&]() { return dis(e); };
        FillRandomValue(gen_float,
                        std::span(reinterpret_cast<float *>(data), size));
    }
};

class Fp8e5m2Trait : public DataTypeTrait {
  public:
    size_t GetTypeBits() const override { return 8; }
    const char *GetName() const override { return "fp8e5m2"; }
    DataType GetDataType() const override { return DataType::kFp8e5m2; }
    void FillRandom(std::mt19937 &e, void *data, size_t size) const override {
        // Byte format for -1 to 1
        std::uniform_int_distribution<> dis(0, 255);
        auto gen_float = [&]() { return (unsigned char)(dis(e) & ~0x40); };
        FillRandomValue(
            gen_float,
            std::span(reinterpret_cast<unsigned char *>(data), size));
    }
};

class Fp8e4m3Trait : public DataTypeTrait {
  public:
    size_t GetTypeBits() const override { return 8; }
    const char *GetName() const override { return "fp8e4m3"; }
    DataType GetDataType() const override { return DataType::kFp8e4m3; }
    void FillRandom(std::mt19937 &e, void *data, size_t size) const override {
        // Byte format for -1 to 1
        std::uniform_int_distribution<> dis(0, 255);
        auto gen_float = [&]() { return (unsigned char)(dis(e) & ~0x40); };
        FillRandomValue(
            gen_float,
            std::span(reinterpret_cast<unsigned char *>(data), size));
    }
};

static const DataTypeTrait *kDataTypeTraits[] = {
    new Fp16Trait(),    new Bf16Trait(),    new Fp32Trait(),
    new Fp8e5m2Trait(), new Fp8e4m3Trait(),
};

static inline const DataTypeTrait *ParseDataType(const std::string &data_type) {
    for (size_t i = 0; i < sizeof(kDataTypeTraits) / sizeof(kDataTypeTraits[0]);
         ++i) {
        if (std::string_view(kDataTypeTraits[i]->GetName()) == data_type) {
            return kDataTypeTraits[i];
        }
    }
    return nullptr;
}

static inline const DataTypeTrait *GetDataTypeTrait(DataType data_type) {
    for (size_t i = 0; i < sizeof(kDataTypeTraits) / sizeof(kDataTypeTraits[0]);
         ++i) {
        if (kDataTypeTraits[i]->GetDataType() == data_type) {
            return kDataTypeTraits[i];
        }
    }
    return nullptr;
}

static inline bool IsBTypeCompatibleWithBackend(const std::string &backend,
                                                BType b_type) {
    if (backend == "petit") {
        return matmul::Matmul::IsPetitCompatibleBType(b_type);
    }
    if (backend == "hipblaslt") {
        return matmul::Matmul::IsHipBLASLtCompatibleBType(b_type);
    }
    return true;
}

static inline std::string_view GetSupportedBTypesForBackend(
    const std::string &backend) {
    if (backend == "petit") {
        return "nvfp4, mxfp4";
    }
    if (backend == "hipblaslt") {
        return "fp16, bf16";
    }
    return kSupportedBTypes;
}

static bool ValidateDataTypeFlag(const char *flagname,
                                 const std::string &value) {
    if (ParseDataType(value)) {
        return true;
    }
    std::cerr << "Invalid value for --" << flagname << ": " << value
              << ". Supported: " << kSupportedDataTypes << std::endl;
    return false;
}

static bool ValidateBTypeFlag(const char *flagname, const std::string &value) {
    auto b_type = matmul::Matmul::ParseBType(value);
    if (!b_type.has_value()) {
        std::cerr << "Invalid value for --" << flagname << ": " << value
                  << ". Supported: " << kSupportedBTypes << std::endl;
        return false;
    }
    return true;
}

[[maybe_unused]] static const bool kATypeFlagValidator =
    gflags::RegisterFlagValidator(&FLAGS_atype, &ValidateDataTypeFlag);
[[maybe_unused]] static const bool kCTypeFlagValidator =
    gflags::RegisterFlagValidator(&FLAGS_ctype, &ValidateDataTypeFlag);
[[maybe_unused]] static const bool kBTypeFlagValidator =
    gflags::RegisterFlagValidator(&FLAGS_btype, &ValidateBTypeFlag);

static absl::Status InitializeData(hal::Device *dev, void **d_a, void **d_b,
                                   void **d_c, const DataTypeTrait *a_type,
                                   const DataTypeTrait *b_type,
                                   const DataTypeTrait *c_type) {

    const size_t a_size =
        FLAGS_m * FLAGS_k * a_type->GetTypeBits() * FLAGS_batch / 8;
    const size_t b_size =
        FLAGS_k * FLAGS_n * b_type->GetTypeBits() * FLAGS_batch / 8;
    const size_t c_size =
        FLAGS_m * FLAGS_n * c_type->GetTypeBits() * FLAGS_batch / 8;

    std::mt19937 e(42);
    std::vector<unsigned char> a(a_size), b(b_size);
    a_type->FillRandom(e, a.data(), FLAGS_m * FLAGS_k);
    b_type->FillRandom(e, b.data(), FLAGS_k * FLAGS_n);

    causalflow::petit::MonadRunner<absl::Status> runner(absl::OkStatus());
    runner.Run([&]() { return dev->Malloc(d_a, a_size); })
        .Run([&]() { return dev->Malloc(d_b, b_size); })
        .Run([&]() { return dev->Malloc(d_c, c_size); })
        .Run([&]() { return dev->CopyToDevice(*d_a, a.data(), a_size); })
        .Run([&]() { return dev->CopyToDevice(*d_b, b.data(), b_size); });

    return runner.code();
}

static absl::Status RunMatmul(std::chrono::duration<double> *elapsed,
                              AlgorithmDescriptor algo, hal::Device *dev,
                              matmul::Matmul *matmul, void *d_c, void *d_a,
                              void *d_b) {
    causalflow::petit::MonadRunner<absl::Status> runner(absl::OkStatus());
    runner
        .Run([&]() {
            return matmul->PrepareForBatchExecution(
                d_c, d_a, d_b, d_c, FLAGS_m * FLAGS_k, FLAGS_k * FLAGS_n,
                FLAGS_m * FLAGS_n, FLAGS_batch);
        })
        .Run([&]() { return matmul->SetAlgorithm(algo); })
        .Run([&]() { return matmul->Execute(FLAGS_warmup); })
        .Run([&]() { return dev->Synchronize(); })
        .Run([&]() {
            auto start = std::chrono::high_resolution_clock::now();
            auto stat = matmul->Execute(FLAGS_repeat);
            stat = dev->Synchronize();
            auto end = std::chrono::high_resolution_clock::now();
            *elapsed = end - start;
            return stat;
        });
    return runner.code();
}

static void PrintResult(const std::string &algo,
                        std::chrono::duration<double> elapsed) {
    auto ops =
        static_cast<double>(FLAGS_m) * FLAGS_n * FLAGS_k * FLAGS_batch * 2;

    fmt::println("Matmul {}x{}x{} {}:{}. Backend: {}, batch: {}, algorithm: "
                 "{}, {} times total "
                 "{:.6f} ms. {:.4f} TFLOPS",
                 FLAGS_m, FLAGS_n, FLAGS_k, FLAGS_atype, FLAGS_ctype,
                 FLAGS_backend, FLAGS_batch, algo, FLAGS_repeat,
                 elapsed.count() * 1e3,
                 ops * FLAGS_repeat / elapsed.count() / 1e12);
}

static void TuneMatmul(hal::Device *dev, const DataTypeTrait *a_type,
                       const DataTypeTrait *c_type, BType b_type,
                       matmul::MatmulFactory *factory, void *d_c, void *d_a,
                       void *d_b) {
    std::unique_ptr<matmul::Matmul> matmul;
    causalflow::petit::MonadRunner<absl::Status> runner(absl::OkStatus());
    runner
        .Run([&]() {
            return factory->CreateMatmul(dev, a_type->GetDataType(),
                                         c_type->GetDataType(), b_type,
                                         FLAGS_m, FLAGS_n, FLAGS_k, &matmul);
        })
        .Run([&]() {
            return matmul->PrepareForBatchExecution(
                d_c, d_a, d_b, d_c, FLAGS_m * FLAGS_k, FLAGS_k * FLAGS_n,
                FLAGS_m * FLAGS_n, FLAGS_batch);
        })
        .Run([&]() { return matmul->EnumerateAlgorithms(); });

    fmt::println("Finished enumerating {} algorithms",
                 matmul->GetAlgorithmCount());

    if (!runner.code().ok()) {
        std::cerr << "Failed to tune the GEMM: " << runner.code().ToString()
                  << std::endl;
        return;
    }

    std::vector<std::pair<std::string, std::chrono::duration<double>>> results;

    // XXX: warm ups now run multiple times
    for (size_t i = 0; i < matmul->GetAlgorithmCount(); ++i) {
        AlgorithmDescriptor algo_desc;
        algo_desc.tag = AlgorithmDescriptor::kIndex;
        algo_desc.repr = std::to_string(i);

        std::chrono::duration<double> elapsed;
        auto stat =
            RunMatmul(&elapsed, algo_desc, dev, matmul.get(), d_c, d_a, d_b);
        if (!stat.ok()) {
            std::cerr << "Failed to run the matmul for repr: "
                      << absl::BytesToHexString(matmul->GetAlgorithmRepr(i))
                      << std::endl;
            continue;
        }
        results.emplace_back(matmul->GetAlgorithmRepr(i), elapsed);
    }

    std::sort(
        results.begin(), results.end(),
        [&](const auto &a, const auto &b) { return a.second < b.second; });

    for (size_t i = 0; i < results.size() && i < 5; ++i) {
        auto &result = results[i];
        PrintResult(absl::BytesToHexString(result.first), result.second);
    }
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    auto a_type = ParseDataType(FLAGS_atype);
    auto c_type = ParseDataType(FLAGS_ctype);
    auto b_type = matmul::Matmul::ParseBType(FLAGS_btype);
    if (!a_type || !c_type || !b_type.has_value()) {
        std::cerr << "Invalid flags after validation" << std::endl;
        return 1;
    }
    if (!IsBTypeCompatibleWithBackend(FLAGS_backend, *b_type)) {
        std::cerr << "Invalid b type for backend '" << FLAGS_backend
                  << "': " << FLAGS_btype << ". Supported: "
                  << GetSupportedBTypesForBackend(FLAGS_backend) << std::endl;
        return 1;
    }

    const DataTypeTrait *b_data_type = a_type;
    auto dense_b_data_type = matmul::Matmul::GetDenseBDataType(*b_type);
    if (dense_b_data_type.has_value()) {
        b_data_type = GetDataTypeTrait(*dense_b_data_type);
    }
    if (!b_data_type) {
        std::cerr << "Internal error: failed to resolve B data type trait"
                  << std::endl;
        return 1;
    }

    auto factory = matmul::MatmulFactory::Create(FLAGS_backend.c_str());
    if (!factory) {
        std::cerr << "Unknown backend: " << FLAGS_backend << std::endl;
        return 1;
    }

    auto plat = hal::GetPlatform(factory->GetPlatformName());
    if (!plat) {
        std::cerr << "Unknown platform\n";
        return 1;
    }

    std::unique_ptr<hal::Device> dev;
    auto stat = plat->GetDevice(0, &dev);
    if (!stat.ok()) {
        std::cerr << "Failed to create device: " << stat.ToString()
                  << std::endl;
        return 1;
    }

    void *d_a, *d_b, *d_c;
    causalflow::petit::MonadRunner<absl::Status> runner(absl::OkStatus());

    runner.Run([&]() {
        return InitializeData(dev.get(), &d_a, &d_b, &d_c, a_type,
                              b_data_type, c_type);
    });

    if (!runner.code().ok()) {
        std::cerr << "Failed to initialize data: " << runner.code().ToString()
                  << std::endl;
        return 1;
    }

    if (FLAGS_algo == "tune") {
        TuneMatmul(dev.get(), a_type, c_type, *b_type, factory.get(), d_c, d_a,
                   d_b);
        return 0;
    }

    AlgorithmDescriptor algo;
    if (FLAGS_algo == "") {
        algo.tag = AlgorithmDescriptor::kDefault;
    } else {
        algo.tag = AlgorithmDescriptor::kOpaqueRepresentation;
        algo.repr = absl::HexStringToBytes(FLAGS_algo);
    }

    std::chrono::duration<double> elapsed;
    std::unique_ptr<matmul::Matmul> matmul;
    runner
        .Run([&]() {
            return factory->CreateMatmul(dev.get(), a_type->GetDataType(),
                                         c_type->GetDataType(), *b_type,
                                         FLAGS_m, FLAGS_n, FLAGS_k, &matmul);
        })
        .Run([&]() {
            return RunMatmul(&elapsed, algo, dev.get(), matmul.get(), d_c, d_a,
                             d_b);
        });
    if (!runner.code().ok()) {
        std::cerr << "Failed to run matmul: " << runner.code().ToString()
                  << std::endl;
        return 1;
    }

    PrintResult(FLAGS_algo, elapsed);

    return 0;
}
