#pragma once

#include "gemm/rocm/quantization/gemm.h"

namespace causalflow::petit::rocm::quantization::fp4 {

namespace detail {

static constexpr SolutionId SolIdNK(int features, MatmulPipeline pipeline,
                                    MatmulMfmaType mfma_type, unsigned tile_m,
                                    unsigned tile_n, unsigned tile_k,
                                    unsigned partition_m, unsigned partition_n,
                                    unsigned partition_k) {
    return SolutionId::MultiStage(pipeline, (MatmulFeatures)features,
                                  MatmulElementB::kMatmulTypeBFp4, mfma_type,
                                  tile_m, tile_n, tile_k,
                                  MatmulWarpPartition::kMatmulWarpPartition_NK,
                                  partition_m, partition_n, partition_k);
}

static constexpr SolutionId SolId(int features, MatmulPipeline pipeline,
                                  MatmulMfmaType mfma_type, unsigned tile_m,
                                  unsigned tile_n, unsigned tile_k) {
    return SolIdNK(features, pipeline, mfma_type, tile_m, tile_n, tile_k, 1,
                   tile_n, 4 / tile_n);
}

static constexpr SolutionId kAvailableSolutions[] = {
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 1, 4, 8),
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 2, 4, 8),
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 4, 4, 8),
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 1, 2, 16),
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 2, 2, 16),
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_1,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 4, 2, 16),
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 1, 1, 32),
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_1,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 2, 1, 32),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeFp16, 1, 4, 32, 1, 2, 1),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeFp16, 2, 4, 32, 1, 2, 1),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeFp16, 2, 4, 32, 2, 2, 1),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeFp16, 1, 4, 32, 1, 2, 2),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeFp16, 2, 4, 32, 1, 2, 2),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeFp16, 2, 4, 32, 2, 2, 2),

    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 1, 4, 8),
    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 2, 4, 8),
    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 4, 4, 8),
    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 1, 2, 16),
    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 2, 2, 16),
    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_1,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 4, 2, 16),
    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 1, 1, 32),
    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_1,
          MatmulMfmaType::kMatmulMfmaTypeFp16, 2, 1, 32),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid |
                MatmulFeatures::kMatmulFeatures_HighPrecision,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeFp16, 1, 4, 32, 1, 2, 1),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid |
                MatmulFeatures::kMatmulFeatures_HighPrecision,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeFp16, 2, 4, 32, 1, 2, 1),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid |
                MatmulFeatures::kMatmulFeatures_HighPrecision,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeFp16, 2, 4, 32, 2, 2, 1),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid |
                MatmulFeatures::kMatmulFeatures_HighPrecision,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeFp16, 1, 4, 32, 1, 2, 2),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid |
                MatmulFeatures::kMatmulFeatures_HighPrecision,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeFp16, 2, 4, 32, 1, 2, 2),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid |
                MatmulFeatures::kMatmulFeatures_HighPrecision,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeFp16, 2, 4, 32, 2, 2, 2),

    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 4, 8),
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 8),
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 4, 4, 8),
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 2, 16),
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 2, 16),
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_1,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 4, 2, 16),
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 1, 32),
    SolId(MatmulFeatures::kMatmulFeatures_Grid,
          MatmulPipeline::kMatmulPipeline_1,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 1, 32),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 4, 32, 1, 2, 1),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 32, 1, 2, 1),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 32, 2, 2, 1),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 4, 32, 1, 2, 2),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 32, 1, 2, 2),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 32, 2, 2, 2),

    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 4, 8),
    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 8),
    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 4, 4, 8),
    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 2, 16),
    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 2, 16),
    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_1,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 4, 2, 16),
    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_2,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 1, 32),
    SolId(MatmulFeatures::kMatmulFeatures_Grid |
              MatmulFeatures::kMatmulFeatures_HighPrecision,
          MatmulPipeline::kMatmulPipeline_1,
          MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 1, 32),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid |
                MatmulFeatures::kMatmulFeatures_HighPrecision,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 4, 32, 1, 2, 1),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid |
                MatmulFeatures::kMatmulFeatures_HighPrecision,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 32, 1, 2, 1),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid |
                MatmulFeatures::kMatmulFeatures_HighPrecision,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 32, 2, 2, 1),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid |
                MatmulFeatures::kMatmulFeatures_HighPrecision,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeBf16, 1, 4, 32, 1, 2, 2),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid |
                MatmulFeatures::kMatmulFeatures_HighPrecision,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 32, 1, 2, 2),
    SolIdNK(MatmulFeatures::kMatmulFeatures_Grid |
                MatmulFeatures::kMatmulFeatures_HighPrecision,
            MatmulPipeline::kMatmulPipeline_1,
            MatmulMfmaType::kMatmulMfmaTypeBf16, 2, 4, 32, 2, 2, 2),

};

} // namespace detail
} // namespace causalflow::petit::rocm::quantization::fp4