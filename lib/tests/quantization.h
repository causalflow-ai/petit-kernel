#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "gemm/cpu/half_float.h"
#include "gemm/rocm/quantization/types.h"
#include "hal/device.h"

#include <random>
#include <span>
#include <vector>

namespace causalflow::petit::tests::quantization {

void GenerateQuantizedWeightsFp4(unsigned m, unsigned n, unsigned k,
                                 unsigned group_size,
                                 std::vector<unsigned> *qweights,
                                 std::span<unsigned char> scales);

class GemmMPTestData {
  public:
    using DataType = rocm::quantization::DataType;
    static constexpr unsigned kTile = 16;
    static constexpr unsigned kTileN = 64;
    // Output is fp16
    static constexpr unsigned kElementCSize = sizeof(unsigned short);

    explicit GemmMPTestData(hal::Device *dev, DataType type_a, DataType type_b,
                            unsigned m, unsigned n, unsigned k,
                            unsigned group_size);
    ~GemmMPTestData();
    absl::Status PrepareData(bool use_zeros);
    unsigned m() const { return m_; }
    unsigned n() const { return n_; }
    unsigned k() const { return k_; }
    unsigned GroupSize() const { return group_size_; }
    size_t WorkspaceSize() const {
        // Should be (kM / kGroupM) * (kN / kGroupN) * sizeof(uint). It is an
        // overestimate.
        return tal::CeilingDiv(m_, kTile) * tal::CeilingDiv(n_, kTile) *
               sizeof(int);
    }
    size_t InputSize() const { return m_ * k_ * ElementSize(); }
    size_t WeightsSize() const { return k_ * n_ * ElementSize(); }
    size_t WeightsQuantSize() const { return k_ * n_ / 2; }
    size_t ScalesSize() const {
        return tal::CeilingDiv(k_ * n_, group_size_) * ElementSize();
    }
    size_t ZerosSize() const {
        return tal::CeilingDiv(k_ * n_, group_size_ * 2);
    }
    size_t OutputSize() const { return m_ * n_ * kElementCSize; }
    size_t TotalSize() const {
        return WorkspaceSize() + InputSize() + WeightsSize() +
               WeightsQuantSize() + ScalesSize() + ZerosSize() +
               2 * OutputSize();
    }

    unsigned ElementSize() const {
        switch (type_a_) {
        case DataType::kDataTypeFp16:
        case DataType::kDataTypeBf16:
            return 2;
        case DataType::kDataTypeFp8e4m3:
        case DataType::kDataTypeFp8e5m2Fnuz:
            return 1;
        default:
            throw std::runtime_error("Invalid quant data type");
        }
    }

    char *workspace() const { return d_data_; }
    char *input() const { return d_data_ + WorkspaceSize(); }
    char *weights() const { return input() + InputSize(); }
    char *weights_quant() const { return weights() + WeightsSize(); }
    char *scales() const { return weights_quant() + WeightsQuantSize(); }
    char *zeros() const { return scales() + ScalesSize(); }
    char *reference() const { return zeros() + ZerosSize(); }
    char *output() const { return reference() + OutputSize(); }

    // private:
    absl::Status GenerateInputs(std::mt19937 *gen);
    absl::Status GenerateScales(std::mt19937 *gen);
    absl::Status GenerateQWeights(std::mt19937 *gen);

    hal::Device *dev_;
    const DataType type_a_;
    const DataType type_b_;
    unsigned m_, n_, k_, group_size_;
    std::vector<unsigned char> h_input_;
    std::vector<unsigned> h_qweights_;
    std::vector<unsigned char> h_scales_;
    std::vector<unsigned> h_zeros_;
    char *d_data_;
};

} // namespace causalflow::petit::tests::quantization
