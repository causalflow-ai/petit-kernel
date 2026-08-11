#pragma once

#include "moe/rocm/memory_ops.cuh"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <class Config> struct MxFp4WeightConfig {
    using W13 = MxFp4WeightLayout<Config::kNumWarps, Config::kGroupDim>;
    using W2 = MxFp4WeightLayout<Config::kNumWarps, Config::kGroupN>;

    static constexpr unsigned kScaleGroupK = 128;
    static constexpr unsigned kScaleGroupN = 64;
    static constexpr unsigned kWeightVecSize = sizeof(uint4) * 2;
    static constexpr unsigned kScaleGroupsPerOutputTile =
        Config::kGroupN / kScaleGroupN;
    static constexpr unsigned kScaleBytesPerWord =
        sizeof(unsigned) / sizeof(unsigned char);
    static constexpr unsigned kScaleWordBytes = sizeof(unsigned);
};

template <class Config> struct MxFp4W13 {
    using Traits = MxFp4WeightConfig<Config>;
    using W13 = typename Traits::W13;

    static constexpr unsigned kScaleGroupK = Traits::kScaleGroupK;
    static constexpr unsigned kScaleGroupN = Traits::kScaleGroupN;
    static constexpr unsigned kWeightVecSize = Traits::kWeightVecSize;
    static constexpr unsigned kScaleGroupsPerOutputTile =
        Traits::kScaleGroupsPerOutputTile;
    static constexpr unsigned kScaleBytesPerWord =
        Traits::kScaleBytesPerWord;
    static constexpr unsigned kScaleWordBytes = Traits::kScaleWordBytes;

    __device__ void Initialize(const uint4 *w13_base,
                               const unsigned *scales_w13, unsigned expert_id,
                               unsigned tile_k, unsigned dim,
                               unsigned inter_dim) {
        static constexpr unsigned kRowGroupSize = W13::kRowGroupSize;

        const uint4 *w1_ptr =
            w13_base + expert_id * (2 * inter_dim * dim) / kWeightVecSize +
            tile_k * W13::kGroupN * dim / kWeightVecSize;
        const unsigned w13_scale_words_per_expert =
            (2 * dim * inter_dim) / kRowGroupSize / kScaleBytesPerWord;
        const unsigned scale_words_per_k_tile =
            (dim / kScaleGroupK) * kScaleGroupsPerOutputTile * kWarpSize;
        const unsigned *scale_w1_ptr =
            scales_w13 + expert_id * w13_scale_words_per_expert +
            tile_k * scale_words_per_k_tile;
        const unsigned w13_value_range = W13::kGroupN * dim / 2;
        const unsigned w13_scale_range =
            scale_words_per_k_tile * kScaleWordBytes;
        w1_.Initialize(w1_ptr, w13_value_range, scale_w1_ptr, w13_scale_range,
                       dim);
        w3_.Initialize(w1_ptr + inter_dim * dim / kWeightVecSize,
                       w13_value_range,
                       scale_w1_ptr + w13_scale_words_per_expert / 2,
                       w13_scale_range, dim);
    }

    W13 w1_, w3_;
};

template <class Config> struct MxFp4W2 {
    using Traits = MxFp4WeightConfig<Config>;
    using W2 = typename Traits::W2;

    static constexpr unsigned kWeightVecSize = Traits::kWeightVecSize;
    static constexpr unsigned kScaleBytesPerWord =
        Traits::kScaleBytesPerWord;
    static constexpr unsigned kScaleWordBytes = Traits::kScaleWordBytes;

    __device__ void Initialize(const uint4 *w2, const unsigned *scales_w2,
                               unsigned expert_id, unsigned tile_k) {
        static constexpr unsigned kRowGroupSize = W2::kRowGroupSize;
        static constexpr unsigned kDim = Config::kDim;
        static constexpr unsigned kInterDim = Config::kInterDim;
        const unsigned value_bytes_per_expert = kDim * kInterDim / 2;
        const unsigned scale_words_per_expert =
            kDim * kInterDim / kRowGroupSize / kScaleBytesPerWord;
        const unsigned value_tile_offset =
            tile_k * 2 * kWarpSize * sizeof(uint4);
        const unsigned scale_tile_offset = tile_k * kWarpSize;
        const auto *value_ptr = reinterpret_cast<const unsigned char *>(w2) +
                                expert_id * value_bytes_per_expert +
                                value_tile_offset;
        const unsigned *scale_ptr =
            scales_w2 + expert_id * scale_words_per_expert + scale_tile_offset;
        w2_.Initialize(value_ptr, value_bytes_per_expert - value_tile_offset,
                       scale_ptr,
                       (scale_words_per_expert - scale_tile_offset) *
                           kScaleWordBytes,
                       kInterDim);
    }

    W2 w2_;
};

template <class Config> struct MxFp4Weights {
    using W13Weights = MxFp4W13<Config>;
    using W2Weights = MxFp4W2<Config>;
    using W13 = typename W13Weights::W13;
    using W2 = typename W2Weights::W2;

    W13Weights w13_;
    W2Weights w2_;
};

} // namespace causalflow::petit::rocm::moe
