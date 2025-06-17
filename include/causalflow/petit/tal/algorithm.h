#pragma once

#include "causalflow/petit/tal/detail/helper_macro.h"

namespace causalflow::tal {

template <class T>
static inline TAL_HOST_DEVICE constexpr T CeilingDiv(T x, T y) {
    return (x + y - 1) / y;
}
} // namespace causalflow::tal