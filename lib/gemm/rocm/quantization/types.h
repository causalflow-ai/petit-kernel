#pragma once

namespace causalflow::petit::rocm::quantization {
enum DataType {
    kDataTypeInt4,
    kDataTypeFp8e4m3,
    kDataTypeFp8e8m0,
    kDataTypeFp4e2m1,
    kDataTypeFp16,
    kDataTypeBf16,
    kDataTypeFp8e5m2Fnuz,
};

} // namespace causalflow::petit::rocm::quantization