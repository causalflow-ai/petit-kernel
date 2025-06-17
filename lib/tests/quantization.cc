#include "quantization.h"
#include "utils/monad_runner.h"
#include "utils/test_utils.h"
#include <random>

namespace causalflow::petit::tests::quantization {

void GenerateQuantizedWeightsFp4(unsigned m, unsigned n, unsigned k,
                                 unsigned group_size,
                                 std::vector<unsigned> *qweights,
                                 std::span<unsigned char> scales) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<unsigned> dist(0, 255 / 2);
    std::uniform_int_distribution<unsigned> dist_q(0, UINT_MAX);
    // Limit the range of the input and scale so that the final results will
    // stay in fp16. The generated values do not have NaN.
    auto gen_fp8_e4m3_fnuz = [&]() {
        auto v = dist(gen);
        unsigned sgn = 0;
        unsigned mantissa = v & 0x7;
        unsigned exp = ((v & 0x18) >> 3) + 8;

        return (sgn << 7) | (exp << 3) | mantissa;
    };
    auto gen_q = [&]() { return dist_q(gen); };

    FillRandomValue(gen_fp8_e4m3_fnuz, scales);
    FillRandomValue(gen_q, qweights);
}

GemmMPTestData::GemmMPTestData(hal::Device *dev, DataType type_a,
                               DataType type_b, unsigned m, unsigned n,
                               unsigned k, unsigned group_size)
    : dev_(dev), type_a_(type_a), type_b_(type_b), m_(m), n_(n), k_(k),
      group_size_(group_size), d_data_(nullptr) {}

GemmMPTestData::~GemmMPTestData() {
    if (d_data_) {
        auto r = dev_->Free(d_data_);
        (void)r;
    }
}

absl::Status GemmMPTestData::GenerateInputs(std::mt19937 *gen) {
    h_input_.resize(m_ * k_ * ElementSize());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    if (type_a_ == DataType::kDataTypeFp16) {
        auto gen_half = [&]() { return half_float::half(dist(*gen)); };
        FillRandomValue(
            gen_half,
            std::span(reinterpret_cast<half_float::half *>(h_input_.data()),
                      m_ * k_));
    } else if (type_a_ == DataType::kDataTypeBf16) {
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        auto gen_bf16 = [&]() {
            float v = dist(*gen);
            unsigned short r = reinterpret_cast<const unsigned &>(v) >> 16;
            return r;
        };
        FillRandomValue(gen_bf16, std::span(reinterpret_cast<unsigned short *>(
                                                h_input_.data()),
                                            m_ * k_));
    } else if (type_a_ == DataType::kDataTypeFp8e4m3) {
        std::uniform_int_distribution<unsigned> dist(0, 255 / 4);
        // Limit the range of the input and scale so that the final results will
        // stay in fp16. The generated values do not have NaN.
        // FIXME: Should it be fp8e4m3 or fp8e4m3fnuz?
        auto gen_fp8_e4m3_fnuz = [&]() {
            auto v = dist(*gen);
            unsigned sgn = (v & 0x20) != 0;
            unsigned mantissa = v & 0x7;
            unsigned exp = ((v & 0x18) >> 3) + 8;

            return (sgn << 7) | (exp << 3) | mantissa;
        };
        FillRandomValue(gen_fp8_e4m3_fnuz, &h_input_);
    } else {
        return absl::InvalidArgumentError("Invalid data type");
    }
    return absl::OkStatus();
}

absl::Status GemmMPTestData::GenerateScales(std::mt19937 *gen) {
    if (type_b_ == DataType::kDataTypeFp4e2m1) {
        h_scales_.resize(k_ / group_size_ * n_ * sizeof(unsigned char));
        std::uniform_int_distribution<unsigned> dist(1, 196);
        // FIXME: Do we need to make sure the scale is sufficiently small?
        auto gen_scale_fp8_e5m3 = [&]() { return dist(*gen); };
        FillRandomValue(
            gen_scale_fp8_e5m3,
            std::span(reinterpret_cast<unsigned char *>(h_scales_.data()),
                      k_ / group_size_ * n_));
    } else if (type_b_ == DataType::kDataTypeInt4) {
        h_scales_.resize(k_ / group_size_ * n_ * sizeof(unsigned short));
        std::uniform_real_distribution<float> dist(0.001f, 0.01f);
        auto gen_scale_half = [&]() { return half_float::half(dist(*gen)); };
        auto gen_scale_bf16 = [&]() {
            float v = dist(*gen);
            unsigned short r = reinterpret_cast<const unsigned &>(v) >> 16;
            return r;
        };
        if (type_a_ == DataType::kDataTypeFp16) {
            FillRandomValue(gen_scale_half,
                            std::span(reinterpret_cast<half_float::half *>(
                                          h_scales_.data()),
                                      k_ / group_size_ * n_));
        } else if (type_a_ == DataType::kDataTypeBf16) {
            FillRandomValue(
                gen_scale_bf16,
                std::span(reinterpret_cast<unsigned short *>(h_scales_.data()),
                          k_ / group_size_ * n_));
        } else {
            return absl::InvalidArgumentError("Invalid data type");
        }
    } else {
        return absl::InvalidArgumentError("Invalid data type");
    }
    return absl::OkStatus();
}

absl::Status GemmMPTestData::GenerateQWeights(std::mt19937 *gen) {
    std::uniform_int_distribution<unsigned> dist_q(0, UINT_MAX);
    auto gen_q = [&]() { return dist_q(*gen); };
    FillRandomValue(gen_q, &h_qweights_);
    return absl::OkStatus();
}

absl::Status GemmMPTestData::PrepareData(bool use_zeros) {
    h_qweights_.resize(k_ * n_ / (32 / 4));
    h_zeros_.resize(k_ / group_size_ * n_ / 8);

    std::mt19937 gen(42);
    MonadRunner runner(absl::OkStatus());
    runner.Run([&]() { return GenerateInputs(&gen); })
        .Run([&]() { return GenerateScales(&gen); })
        .Run([&]() { return GenerateQWeights(&gen); })
        .Run([&]() {
            if (use_zeros) {
                std::uniform_int_distribution<unsigned> dist_q(0, UINT_MAX);
                auto gen_q = [&]() { return dist_q(gen); };
                FillRandomValue(gen_q, &h_zeros_);
                return absl::OkStatus();
            } else {
                h_zeros_.assign(h_zeros_.size(), 0x88888888);
                return absl::OkStatus();
            }
        })
        .Run([&]() {
            return dev_->Malloc(reinterpret_cast<void **>(&d_data_),
                                TotalSize());
        })
        .Run([&]() { return dev_->Memset(workspace(), 0, WorkspaceSize()); })
        .Run([&]() {
            return dev_->CopyToDevice(input(), h_input_.data(), InputSize());
        })
        .Run([&]() {
            return dev_->CopyToDevice(weights_quant(), h_qweights_.data(),
                                      WeightsQuantSize());
        })
        .Run([&]() {
            return dev_->CopyToDevice(scales(), h_scales_.data(), ScalesSize());
        })
        .Run([&]() {
            return dev_->CopyToDevice(zeros(), h_zeros_.data(), ZerosSize());
        });
    return runner.code();
}

} // namespace causalflow::petit::tests::quantization
