#pragma once

namespace causalflow::petit::rocm::moe {

// Placeholder for future MFMA-oriented schedule tuning.
template <class Config> struct WarpSchedule {
    static constexpr unsigned kWarpTileM = Config::kGroupM;
    static constexpr unsigned kWarpTileN = Config::kGroupN / Config::kNumWarps;
    static constexpr unsigned kMmaTileN = 16;
    static constexpr unsigned kAccumulatorRows = 2;
    static constexpr unsigned kColumnIters = kWarpTileN / kMmaTileN;
    static constexpr unsigned kAccumulatorFragments =
        kAccumulatorRows * kColumnIters;

    static_assert(Config::kGroupN % Config::kNumWarps == 0, "");
    static_assert(kWarpTileN % kMmaTileN == 0, "");
    static_assert(kAccumulatorFragments % 2 == 0, "");
};

} // namespace causalflow::petit::rocm::moe
