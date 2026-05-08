#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace causalflow::petit::tests::fp8_sampler {

// FP8E4M3DiscreteSampler
// ----------------------
// This class generates raw FP8 E4M3 bytes directly, without first generating
// float values and then casting.
//
// High-level distribution idea:
// 1. FP8 E4M3 values naturally group into exponent bins:
//      - subnormals
//      - normal exponent bins e = -6 .. 7
//      - the top finite exponent bin
// 2. Within each exponent bin, we sample uniformly across mantissas.
// 3. We choose the sign uniformly, so the final distribution always has mean 0.
// 4. To control the scale (sigma), we do not scale sampled FP8 values
//    afterwards.
//    Instead, we change how often each exponent bin is chosen.
// 5. We use a simple arithmetic-sequence family for bin probabilities:
//
//      p_b = 1 / B + d * (b - (B - 1) / 2)
//
//    where B is the number of bins and d is solved directly from the desired
//    sigma using the per-bin second moments.
// 6. This uses all bins, preserves diversity, and avoids expensive float->FP8
//    conversion on CPU.
//
// Notes:
// - Output is the raw FP8 E4M3 byte encoding.
// - The class models a symmetric discrete distribution over FP8 values.
// - The realized sigma is exact for this exponent-bin mixture model.
// - The requested sigma must fall inside the feasible range of the arithmetic
//   family. If it does not, construction throws std::invalid_argument.

class FP8E4M3DiscreteSampler {
  public:
    using result_type = uint8_t;

    // Creates a sampler whose emitted FP8 values have approximately the
    // requested sigma under the exponent-bin mixture model described above.
    explicit FP8E4M3DiscreteSampler(double sigma);
    FP8E4M3DiscreteSampler() : FP8E4M3DiscreteSampler(1.0) {}

    FP8E4M3DiscreteSampler(const FP8E4M3DiscreteSampler &) = default;
    FP8E4M3DiscreteSampler &operator=(const FP8E4M3DiscreteSampler &) = default;
    FP8E4M3DiscreteSampler(FP8E4M3DiscreteSampler &&) = default;
    FP8E4M3DiscreteSampler &operator=(FP8E4M3DiscreteSampler &&) = default;
    ~FP8E4M3DiscreteSampler() = default;

    template <typename UniformRandomNumberGenerator>
    result_type operator()(UniformRandomNumberGenerator &rng) const {
        const auto sample_u32 = static_cast<uint32_t>(rng());
        return SampleFromU32(sample_u32);
    }

    result_type min() const { return std::numeric_limits<result_type>::min(); }
    result_type max() const { return std::numeric_limits<result_type>::max(); }

    void reset() const {}

    bool operator==(const FP8E4M3DiscreteSampler &other) const {
        return sigma_ == other.sigma_;
    }

    bool operator!=(const FP8E4M3DiscreteSampler &other) const {
        return !(*this == other);
    }

    double sigma() const;
    double realized_sigma() const;

    // Decodes a raw FP8 E4M3 byte to float.
    static float Decode(uint8_t value);

  private:
    uint8_t SampleFromU32(uint32_t random_u32) const;

    double sigma_;
    double slope_;
    std::array<uint32_t, 15> bin_cutoffs_{};
};

class FP8E4M3QuantizedNormalSampler {
  public:
    using result_type = uint8_t;

    FP8E4M3QuantizedNormalSampler(double mean, double sigma);

    template <typename UniformRandomNumberGenerator>
    result_type operator()(UniformRandomNumberGenerator &rng) const {
        const auto sample_u32 = static_cast<uint32_t>(rng());
        return SampleFromU32(sample_u32);
    }

    result_type min() const { return std::numeric_limits<result_type>::min(); }
    result_type max() const { return std::numeric_limits<result_type>::max(); }

  private:
    uint8_t SampleFromU32(uint32_t random_u32) const;

    std::vector<uint8_t> codes_;
    std::vector<double> cdf_;
};

} // namespace causalflow::petit::tests::fp8_sampler
