#pragma once

#include "moe/rocm/memory_ops.cuh"

#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <class Config> struct BlockScaleFp8WeightConfig {
    using WeightScalar = __hip_fp8_e4m3;
    using W13 = W13Layout<WeightScalar, Config::kNumWarps, Config::kGroupN>;
    using W2 = W2Layout<WeightScalar, Config::kNumWarps, Config::kGroupN>;

    static constexpr unsigned kScaleBlockSize = 128;
    static constexpr unsigned kVecSize = sizeof(uint4);
    static constexpr unsigned kGroupDimScaleBlocks =
        Config::kGroupDim / kScaleBlockSize;
    static constexpr unsigned kScaleWordBytes = sizeof(unsigned);
    static constexpr unsigned kW2ShuffleBlockN = 16;
};

template <class Config> struct BlockScaleFp8W13 {
    using Traits = BlockScaleFp8WeightConfig<Config>;
    using W13 = typename Traits::W13;

    static constexpr unsigned kVecSize = Traits::kVecSize;
    static constexpr unsigned kScaleBlockSize = Traits::kScaleBlockSize;
    static constexpr unsigned kGroupDimScaleBlocks =
        Traits::kGroupDimScaleBlocks;

    __device__ void Initialize(const uint4 *w13_base,
                               const unsigned *scales_w13,
                               unsigned expert_id, unsigned tile_k,
                               unsigned n_blocks, unsigned k_blocks,
                               unsigned dim, unsigned inter_dim) {
        const uint4 *w1_ptr =
            w13_base + expert_id * (2 * inter_dim * dim) / kVecSize +
            tile_k * Config::kGroupDim * dim / kVecSize;
        const unsigned *scale_w1_ptr =
            scales_w13 + expert_id * (2 * k_blocks * n_blocks) +
            tile_k * kGroupDimScaleBlocks * (dim / kScaleBlockSize);
        const unsigned w13_value_range = Config::kGroupDim * dim;
        const unsigned w13_combined_value_range =
            inter_dim * dim + w13_value_range;
        const unsigned w13_scale_range =
            kGroupDimScaleBlocks * n_blocks * sizeof(float);
        const unsigned w13_combined_scale_range =
            W13::ScaleProjectionOffsetBytes(dim, inter_dim) +
            w13_scale_range;
        w1_.Initialize(w1_ptr, w13_combined_value_range, scale_w1_ptr,
                       w13_combined_scale_range, dim);
    }

    W13 w1_;
};

template <class Config> struct BlockScaleFp8W2 {
    using Traits = BlockScaleFp8WeightConfig<Config>;
    using W2 = typename Traits::W2;

    static constexpr unsigned kVecSize = Traits::kVecSize;
    static constexpr unsigned kScaleBlockSize = Traits::kScaleBlockSize;
    static constexpr unsigned kGroupDimScaleBlocks =
        Traits::kGroupDimScaleBlocks;
    static constexpr unsigned kScaleWordBytes = Traits::kScaleWordBytes;
    static constexpr unsigned kW2ShuffleBlockN = Traits::kW2ShuffleBlockN;

    __device__ void Initialize(const uint4 *w2, const unsigned *scales_w2,
                               unsigned expert_id, unsigned tile_k,
                               unsigned n_blocks, unsigned k_blocks,
                               unsigned dim, unsigned inter_dim) {
        // w2 is pre-shuffled as [rbi][cbi][kki][bni][kpi] with
        // blockN=16/blockK=32. Advancing logical K by 256 (= 8 * 32)
        // means advancing cbi by 8 blocks, i.e. 8 * (2 * 16 * 16) =
        // 4096 fp8 elements.
        const uint4 *w2_ptr =
            w2 + expert_id * (dim * inter_dim) / kVecSize +
            tile_k * (Config::kGroupDim * kW2ShuffleBlockN) / kVecSize;

        const unsigned *scale_w2_ptr =
            scales_w2 + expert_id * (n_blocks * k_blocks) +
            tile_k * kGroupDimScaleBlocks;
        const unsigned w2_value_range =
            dim * inter_dim - tile_k * Config::kGroupDim * kW2ShuffleBlockN;
        const unsigned w2_scale_range =
            n_blocks * k_blocks * kScaleWordBytes -
            tile_k * kGroupDimScaleBlocks * kScaleWordBytes;
        w2_.Initialize(w2_ptr, w2_value_range, scale_w2_ptr, w2_scale_range,
                       inter_dim);
    }

    W2 w2_;
};

template <class Config> struct BlockScaleFp8Weights {
    using W13Weights = BlockScaleFp8W13<Config>;
    using W2Weights = BlockScaleFp8W2<Config>;
    using W13 = typename W13Weights::W13;
    using W2 = typename W2Weights::W2;

    W13Weights w13_;
    W2Weights w2_;
};

} // namespace causalflow::petit::rocm::moe
