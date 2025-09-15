#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "gemm/cpu/half_float.h"

namespace causalflow::petit::tests::cpu_numeric {

namespace detail {

constexpr int kFp32MantBits = 23;
constexpr int kFp32ExpBias = 127;

// The conversion function is from:
// https://github.com/ROCm/clr/blob/0f2d6024245abde73eaff463cdc1f10f193395b1/hipamd/include/hip/amd_detail/amd_hip_fp8.h#L395
// This has been modified and simplified to ensure compilation in
// non-HIP-language environments
template <bool kIsBf8, bool kIsFnuz = false, bool kClip = false>
constexpr float CvtFp32Fp8(uint8_t x) {
    constexpr int kExpBits = kIsBf8 ? 5 : 4;
    constexpr int kMantBits = kIsBf8 ? 2 : 3;
    constexpr int kExpBias = (1 << (kExpBits - 1)) - 1 + (kIsFnuz ? 1 : 0);

    uint32_t sign = x >> 7, mantissa = x & ((1 << kMantBits) - 1);
    int exponent = (x & 0x7f) >> kMantBits;

    if (kIsFnuz) {
        if (x == 0x80) {
            return NAN;
        }
        if (x == 0) {
            return 0.0f;
        }
    } else {
        if (x == 0x80) {
            return -0.0f;
        }
        if (x == 0) {
            return 0.0f;
        }
        if (kIsBf8) {
            if ((x & 0x7c) == 0x7c) {
                if ((x & 0x3) == 0) {
                    if (kClip) {
                        return sign ? -57344.0f : 57344.0f;
                    } else {
                        return sign ? -INFINITY : INFINITY;
                    }
                }
                return NAN;
            }
        } else if ((x & 0x7f) == 0x7f) {
            return NAN;
        }
    }

    // subnormal input
    if (exponent == 0) {
        // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
        int sh = 1 + __builtin_clz(mantissa) - (32 - kMantBits);
        mantissa <<= sh;
        exponent += 1 - sh;
        mantissa &= ((1u << kMantBits) - 1);
    }

    const int kExpOffset = kFp32ExpBias - kExpBias;
    exponent += kExpOffset;
    mantissa <<= kFp32MantBits - kMantBits;

    uint32_t retval = (sign << 31) | (exponent << kFp32MantBits) | mantissa;
    return std::bit_cast<float>(retval);
}

// The conversion function is from:
// https://github.com/ROCm/clr/blob/0f2d6024245abde73eaff463cdc1f10f193395b1/hipamd/include/hip/amd_detail/amd_hip_bf16.h#L206
constexpr uint16_t CvtBf16Fp32(float x_) {
    uint32_t x = std::bit_cast<uint32_t>(x_);
    if (~(x & 0x7f800000)) {
        // When the exponent bits are not all 1s, then the value is zero,
        // normal, or subnormal. We round the bfloat16 mantissa up by adding
        // 0x7fff, plus 1 if the least significant bit of the bfloat16 mantissa
        // is 1 (odd). This causes the bfloat16's mantissa to be incremented by
        // 1 if the 16 least significant bits of the float mantissa are greater
        // than 0x8000, or if they are equal to 0x8000 and the least significant
        // bit of the bfloat16 mantissa is 1 (odd). This causes it to be rounded
        // to even when the lower 16 bits are exactly 0x8000. If the bfloat16
        // mantissa already has the value 0x7f, then incrementing it causes it
        // to become 0x00 and the exponent is incremented by one, which is the
        // next higher FP value to the unrounded bfloat16 value. When the
        // bfloat16 value is subnormal with an exponent of 0x00 and a mantissa
        // of 0x7f, it may be rounded up to a normal value with an exponent of
        // 0x01 and a mantissa of 0x00. When the bfloat16 value has an exponent
        // of 0xfe and a mantissa of 0x7f, incrementing it causes it to become
        // an exponent of 0xff and a mantissa of 0x00, which is Inf, the next
        // higher value to the unrounded value.
        x += 0x7fff + ((x >> 16) & 1); // Round to nearest, round to even
    } else if (x & 0xffff) {
        // When all of the exponent bits are 1, the value is Inf or NaN.
        // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
        // mantissa bit. Quiet NaN is indicated by the most significant mantissa
        // bit being 1. Signaling NaN is indicated by the most significant
        // mantissa bit being 0 but some other bit(s) being 1. If any of the
        // lower 16 bits of the mantissa are 1, we set the least significant bit
        // of the bfloat16 mantissa, in order to preserve signaling NaN in case
        // the bloat16's mantissa bits are all 0.
        x |= 0x10000; // Preserve signaling NaN
    }
    return static_cast<uint16_t>(x >> 16);
}

} // namespace detail

struct fp8_e4m3_t {
    using storage_t = uint8_t;

    static constexpr fp8_e4m3_t from_bits(storage_t bits) {
        fp8_e4m3_t result;
        result.storage_ = bits;
        return result;
    }

    constexpr storage_t to_bits() const { return storage_; }

    float to_fp32() const { return detail::CvtFp32Fp8<false>(storage_); }

  private:
    storage_t storage_;
};

struct fp16_t {
    using storage_t = uint16_t;

    static constexpr fp16_t from_bits(storage_t bits) {
        fp16_t result;
        result.storage_ = bits;
        return result;
    }

    constexpr storage_t to_bits() const { return storage_; }

    static fp16_t from_fp32(float val) {
        uint16_t bits = std::bit_cast<uint16_t>(half_float::half(val));
        return from_bits(bits);
    }

    constexpr bool is_nan() const { return (storage_ & 0x7fff) > 0x7c00; }

    constexpr bool is_inf() const { return (storage_ & 0x7fff) == 0x7c00; }

    constexpr bool is_zero() const { return (storage_ & 0x7fff) == 0; }

  private:
    storage_t storage_;
};

struct bf16_t {
    using storage_t = uint16_t;

    static constexpr bf16_t from_bits(storage_t bits) {
        bf16_t result;
        result.storage_ = bits;
        return result;
    }

    constexpr storage_t to_bits() const { return storage_; }

    static bf16_t from_fp32(float val) {
        return from_bits(detail::CvtBf16Fp32(val));
    }

    constexpr bool is_nan() const { return (storage_ & 0x7fff) > 0x7f80; }

    constexpr bool is_inf() const { return (storage_ & 0x7fff) == 0x7f80; }

    constexpr bool is_zero() const { return (storage_ & 0x7fff) == 0; }

  private:
    storage_t storage_;
};

inline constexpr bool operator==(const fp16_t &lhs, const fp16_t &rhs) {
    if (lhs.is_nan() || rhs.is_nan()) {
        return false;
    }
    if (lhs.is_zero() && rhs.is_zero()) {
        return true;
    }
    return lhs.to_bits() == rhs.to_bits();
}

inline constexpr bool operator==(const bf16_t &lhs, const bf16_t &rhs) {
    if (lhs.is_nan() || rhs.is_nan()) {
        return false;
    }
    if (lhs.is_zero() && rhs.is_zero()) {
        return true;
    }
    return lhs.to_bits() == rhs.to_bits();
}

} // namespace causalflow::petit::tests::cpu_numeric

namespace std {
template <>
class numeric_limits<causalflow::petit::tests::cpu_numeric::fp8_e4m3_t> {
    using T = causalflow::petit::tests::cpu_numeric::fp8_e4m3_t;

  public:
    static constexpr bool is_specialized = true;
    static constexpr T denorm_min() noexcept { return T::from_bits(1); }
    static constexpr T max() noexcept { return T::from_bits(0x7e); }
};
} // namespace std
