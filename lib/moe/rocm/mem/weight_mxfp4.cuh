#pragma once

#include "moe/rocm/memory_ops.cuh"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <class Config> struct MxFp4WeightConfig {
    using W13 = MxFp4WeightLayout<Config::kNumWarps, Config::kGroupDim>;
    using W2 = MxFp4WeightLayout<Config::kNumWarps, Config::kGroupN>;

    static constexpr unsigned kScaleGroupK = 128;
    static constexpr unsigned kScaleGroupN = 64;
    static constexpr unsigned kLayoutN = 16;
    static constexpr unsigned kWeightVecSize = sizeof(uint4) * 2;
    static constexpr unsigned kScaleGroupsPerOutputTile =
        Config::kGroupN / kScaleGroupN;
    static constexpr unsigned kStage2GroupKBlocks =
        Config::kStage2GroupInterDim / kScaleGroupK;
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
                               const unsigned *scales_w13,
                               unsigned expert_id, unsigned tile_k,
                               unsigned dim, unsigned inter_dim) {
        static_assert(
            Config::kGroupN == 256,
            "The scale layout requires 256 elements per output tile");
        static constexpr unsigned kRowGroupSize = W13::kRowGroupSize;

        const uint4 *w1_ptr =
            w13_base + expert_id * (2 * inter_dim * dim) / kWeightVecSize +
            tile_k * Config::kGroupDim * dim / kWeightVecSize;
        const unsigned w13_scale_words_per_expert =
            (2 * dim * inter_dim) / kRowGroupSize / kScaleBytesPerWord;
        const unsigned scale_words_per_k_tile =
            (dim / kScaleGroupK) * kScaleGroupsPerOutputTile * kWarpSize;
        const unsigned *scale_w1_ptr =
            scales_w13 + expert_id * w13_scale_words_per_expert +
            tile_k * scale_words_per_k_tile;
        const unsigned w13_value_range = Config::kGroupDim * dim / 2;
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

    static constexpr unsigned kScaleGroupN = Traits::kScaleGroupN;
    static constexpr unsigned kLayoutN = Traits::kLayoutN;
    static constexpr unsigned kWeightVecSize = Traits::kWeightVecSize;
    static constexpr unsigned kScaleGroupsPerOutputTile =
        Traits::kScaleGroupsPerOutputTile;
    static constexpr unsigned kStage2GroupKBlocks =
        Traits::kStage2GroupKBlocks;
    static constexpr unsigned kScaleBytesPerWord =
        Traits::kScaleBytesPerWord;
    static constexpr unsigned kScaleWordBytes = Traits::kScaleWordBytes;

    __device__ void Initialize(const uint4 *w2, const unsigned *scales_w2,
                               unsigned expert_id, unsigned tile_n,
                               unsigned tile_k, unsigned dim,
                               unsigned inter_dim) {
        static constexpr unsigned kRowGroupSize = W2::kRowGroupSize;

        const unsigned w2_value_n_tile_offset =
            tile_n * Config::kGroupN * inter_dim / kWeightVecSize;
        const unsigned w2_value_k_tile_offset =
            tile_k * Config::kStage2GroupInterDim * kLayoutN / kWeightVecSize;
        const uint4 *w2_ptr =
            w2 + expert_id * (dim * inter_dim) / kWeightVecSize +
            w2_value_n_tile_offset + w2_value_k_tile_offset;
        const unsigned w2_scale_words_per_expert =
            (dim * inter_dim) / kRowGroupSize / kScaleBytesPerWord;
        const unsigned w2_scale_n_tile_offset =
            tile_n * Config::kGroupN * inter_dim / kRowGroupSize /
            kScaleBytesPerWord;
        const unsigned w2_scale_k_tile_offset =
            tile_k * kStage2GroupKBlocks * kScaleGroupsPerOutputTile *
            kWarpSize;
        const unsigned *scale_w2_ptr =
            scales_w2 + expert_id * w2_scale_words_per_expert +
            w2_scale_n_tile_offset + w2_scale_k_tile_offset;
        const unsigned w2_value_range =
            (inter_dim * dim) / 2 -
            w2_value_n_tile_offset * sizeof(uint4) -
            w2_value_k_tile_offset * sizeof(uint4);
        const unsigned w2_scale_range =
            w2_scale_words_per_expert * kScaleWordBytes -
            w2_scale_n_tile_offset * kScaleWordBytes -
            w2_scale_k_tile_offset * kScaleWordBytes;
        w2_.Initialize(w2_ptr, w2_value_range, scale_w2_ptr, w2_scale_range,
                       inter_dim);
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
