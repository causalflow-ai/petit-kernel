#include "fp8_sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace causalflow::petit::tests::fp8_sampler {
namespace {

constexpr int kNumBins = 16;
constexpr int kMaxElements = 8;
constexpr uint32_t kBinResolution = 1u << 20;

struct LutData {
    uint8_t magnitude_codes_[kNumBins][kMaxElements]{};
};

const LutData &GetLut();
void InitializeLut(LutData &lut);

constexpr double BinSecondMoment(int bin) {
    return bin == 0 ? (20.0 / 262144.0)
           : bin == 15
               ? 128000.0
               : (bin <= 7 ? (137.0 / 64.0) / (1ULL << (2 * (7 - bin)))
                           : (137.0 / 64.0) * (1ULL << (2 * (bin - 7))));
}

constexpr double ClosedFormUniformSecondMoment() {
    return (BinSecondMoment(0) + BinSecondMoment(1) + BinSecondMoment(2) +
            BinSecondMoment(3) + BinSecondMoment(4) + BinSecondMoment(5) +
            BinSecondMoment(6) + BinSecondMoment(7) + BinSecondMoment(8) +
            BinSecondMoment(9) + BinSecondMoment(10) + BinSecondMoment(11) +
            BinSecondMoment(12) + BinSecondMoment(13) + BinSecondMoment(14) +
            BinSecondMoment(15)) /
           static_cast<double>(kNumBins);
}

double ArithmeticGain() {
    const double kMid = (static_cast<double>(kNumBins) - 1.0) / 2.0;
    double gain = 0.0;
    for (int bin = 0; bin < kNumBins; ++bin) {
        gain += (static_cast<double>(bin) - kMid) * BinSecondMoment(bin);
    }
    return gain;
}

double UniformPBinRange() {
    return 2.0 / (static_cast<double>(kNumBins) *
                  (static_cast<double>(kNumBins) - 1.0));
}

double MinSigmaSquared() {
    return ClosedFormUniformSecondMoment() -
           UniformPBinRange() * ArithmeticGain();
}

double MaxSigmaSquared() {
    return ClosedFormUniformSecondMoment() +
           UniformPBinRange() * ArithmeticGain();
}

double SolveSlope(double sigma) {
    return (sigma * sigma - ClosedFormUniformSecondMoment()) / ArithmeticGain();
}

double BinProb(int bin, double slope) {
    return 1.0 / static_cast<double>(kNumBins) +
           slope * (static_cast<double>(bin) -
                    (static_cast<double>(kNumBins) - 1.0) / 2.0);
}

std::array<uint32_t, kNumBins - 1> BuildBinCutoffs(double slope) {
    std::array<uint32_t, kNumBins - 1> cutoffs{};
    constexpr long double kNumBinsLongDouble =
        static_cast<long double>(kNumBins);
    constexpr long double kBinResolutionLongDouble =
        static_cast<long double>(kBinResolution);

    uint32_t prev = 0;
    for (int bin = 0; bin < kNumBins - 1; ++bin) {
        const long double n = static_cast<long double>(bin + 1);
        const long double cumulative =
            n / kNumBinsLongDouble + static_cast<long double>(slope) * n *
                                         (n - kNumBinsLongDouble) * 0.5L;
        uint32_t cutoff;
        if (cumulative <= 0.0L) {
            cutoff = 0;
        } else if (cumulative >= 1.0L) {
            cutoff = kBinResolution;
        } else {
            cutoff =
                static_cast<uint32_t>(cumulative * kBinResolutionLongDouble);
        }
        if (cutoff < prev) {
            cutoff = prev;
        }
        cutoffs[bin] = cutoff;
        prev = cutoff;
    }
    return cutoffs;
}

int BinFromU20(uint32_t bin_u,
               const std::array<uint32_t, kNumBins - 1> &bin_cutoffs) {
    for (int bin = 0; bin < kNumBins - 1; ++bin) {
        if (bin_u < bin_cutoffs[bin]) {
            return bin;
        }
    }
    return kNumBins - 1;
}

double SigmaLowerBound() { return std::sqrt(MinSigmaSquared()); }

double SigmaUpperBound() { return std::sqrt(MaxSigmaSquared()); }

void ValidateSigma(double sigma) {
    if (!(sigma > 0.0) || !std::isfinite(sigma)) {
        throw std::invalid_argument("sigma must be finite and positive");
    }

    if (sigma < SigmaLowerBound() || sigma > SigmaUpperBound()) {
        throw std::invalid_argument(
            "requested sigma is outside the feasible range of the "
            "all-bin arithmetic distribution family");
    }

    const double slope = SolveSlope(sigma);
    const double min_prob =
        slope < 0.0 ? BinProb(15, slope) : BinProb(0, slope);
    if (min_prob <= 0.0) {
        throw std::invalid_argument(
            "requested sigma is outside the feasible range of the "
            "all-bin arithmetic distribution family");
    }
}

uint8_t NumMagnitudeCodes(int bin) { return (bin == 0 || bin == 15) ? 7u : 8u; }

uint8_t SampleMagnitudeCode(int bin, uint32_t random_12bits) {
    const auto &codes = GetLut().magnitude_codes_[bin];
    const uint32_t mantissa_bits = random_12bits >> 1;
    const uint32_t size = NumMagnitudeCodes(bin);

    const uint32_t index = static_cast<uint32_t>(
        (static_cast<uint64_t>(mantissa_bits) * size) >> 11);
    return codes[index];
}

uint8_t
SampleFromU32Impl(uint32_t random_u32,
                  const std::array<uint32_t, kNumBins - 1> &bin_cutoffs) {
    const uint32_t bin_u = random_u32 & (kBinResolution - 1u);
    const uint32_t attr_u = random_u32 >> 20;

    const int bin = BinFromU20(bin_u, bin_cutoffs);
    const uint8_t magnitude_code = SampleMagnitudeCode(bin, attr_u);
    const uint8_t sign = static_cast<uint8_t>(attr_u & 1u);

    return static_cast<uint8_t>(magnitude_code | (sign << 7));
}

const LutData &GetLut() {
    static const LutData lut = []() {
        LutData data;
        InitializeLut(data);
        return data;
    }();
    return lut;
}

void InitializeLut(LutData &lut) {
    for (int mantissa = 1; mantissa <= 7; ++mantissa) {
        lut.magnitude_codes_[0][mantissa - 1] = static_cast<uint8_t>(mantissa);
    }

    for (int bin = 1; bin <= 14; ++bin) {
        const int exponent = bin;
        for (int mantissa = 0; mantissa <= 7; ++mantissa) {
            lut.magnitude_codes_[bin][mantissa] =
                static_cast<uint8_t>((exponent << 3) | mantissa);
        }
    }

    for (int mantissa = 0; mantissa <= 6; ++mantissa) {
        lut.magnitude_codes_[15][mantissa] =
            static_cast<uint8_t>((0xF << 3) | mantissa);
    }
}

} // namespace

FP8E4M3DiscreteSampler::FP8E4M3DiscreteSampler(double sigma)
    : sigma_(sigma), slope_(SolveSlope(sigma)) {
    ValidateSigma(sigma);
    bin_cutoffs_ = BuildBinCutoffs(slope_);
}

double FP8E4M3DiscreteSampler::sigma() const { return sigma_; }

uint8_t FP8E4M3DiscreteSampler::SampleFromU32(uint32_t random_u32) const {
    return SampleFromU32Impl(random_u32, bin_cutoffs_);
}

double FP8E4M3DiscreteSampler::realized_sigma() const {
    return std::sqrt(ClosedFormUniformSecondMoment() +
                     slope_ * ArithmeticGain());
}

float FP8E4M3DiscreteSampler::Decode(uint8_t value) {
    const int sign = (value >> 7) ? -1 : 1;
    const int exponent = (value >> 3) & 0xF;
    const int mantissa = value & 0x7;

    if (exponent == 0) {
        if (mantissa == 0) {
            return 0.0f;
        }
        return sign * std::ldexp(static_cast<float>(mantissa), -9);
    }

    if (exponent == 0xF) {
        if (mantissa == 0x7) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, 8);
    }

    const int e = exponent - 7;
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, e);
}

FP8E4M3QuantizedNormalSampler::FP8E4M3QuantizedNormalSampler(double mean,
                                                             double sigma) {
    if (!std::isfinite(mean)) {
        throw std::invalid_argument("mean must be finite");
    }
    if (!(sigma > 0.0) || !std::isfinite(sigma)) {
        throw std::invalid_argument("sigma must be finite and positive");
    }

    struct CodePoint {
        double value;
        uint8_t code;
    };

    std::vector<CodePoint> points;
    points.reserve(254);
    for (unsigned code = 0; code < 256; ++code) {
        if ((code & 0x7f) == 0x7f || code == 0x80) {
            continue;
        }
        points.push_back(
            {.value = static_cast<double>(
                 FP8E4M3DiscreteSampler::Decode(static_cast<uint8_t>(code))),
             .code = static_cast<uint8_t>(code)});
    }

    std::sort(points.begin(), points.end(),
              [](const CodePoint &a, const CodePoint &b) {
                  return a.value < b.value;
              });

    auto gaussian_cdf = [&](double x) {
        return 0.5 * std::erfc(-(x - mean) / (sigma * std::sqrt(2.0)));
    };

    cdf_.reserve(points.size());
    codes_.reserve(points.size());
    double cumulative = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        const double lo = i == 0
                              ? -std::numeric_limits<double>::infinity()
                              : 0.5 * (points[i - 1].value + points[i].value);
        const double hi = i + 1 == points.size()
                              ? std::numeric_limits<double>::infinity()
                              : 0.5 * (points[i].value + points[i + 1].value);
        const double prob = gaussian_cdf(hi) - gaussian_cdf(lo);
        cumulative += std::max(prob, 0.0);
        codes_.push_back(points[i].code);
        cdf_.push_back(cumulative);
    }

    if (!cdf_.empty()) {
        for (double &value : cdf_) {
            value /= cumulative;
        }
        cdf_.back() = 1.0;
    }
}

uint8_t
FP8E4M3QuantizedNormalSampler::SampleFromU32(uint32_t random_u32) const {
    const double u = (static_cast<double>(random_u32) + 0.5) / 4294967296.0;
    const auto it = std::upper_bound(cdf_.begin(), cdf_.end(), u);
    const size_t idx = static_cast<size_t>(it - cdf_.begin());
    return codes_[std::min(idx, codes_.size() - 1)];
}

} // namespace causalflow::petit::tests::fp8_sampler
