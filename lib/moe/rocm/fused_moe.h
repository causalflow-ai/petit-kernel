#pragma once

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

enum {
    kFusedMoEErrorInvalidSolution = 1,
    kFusedMoEErrorInvalidArgument = 2,
    kFusedMoEErrorUnsupported = 3,
};

enum class FusedMoEDataType : unsigned {
    kNone,
    kMxFp4,
    kNvFp4,
    kChannelScaleFp8,
    kBlockScaleFp8,
    kBf16,
};

enum class FusedMoEWeightOrdering : unsigned {
    kNativeMxFp4,
    kPetitMxFp4,
    kPetitFp8,
};

enum class FusedMoEStages : unsigned {
    kOneStage,
    kTwoStage,
};

enum class FusedMoEMfmaShape : unsigned {
    kMfmaFp816x16x32,
};

enum class FusedMoEActivationFunction : unsigned {
    kSiluDot,
    kOpenAISwiGLU,
};

enum class FusedMoEStage1Buffering : unsigned {
    kSingleBuffer,
    kDoubleBuffer,
};

struct FusedMoESolutionId {
    FusedMoEDataType act_dtype : 4;
    FusedMoEDataType weight_dtype : 4;
    FusedMoEDataType bias_dtype : 4;
    FusedMoEWeightOrdering weight_ordering : 2;
    FusedMoEMfmaShape mfma : 2;
    FusedMoEStages stages : 4;
    FusedMoEActivationFunction activation : 3;
    FusedMoEStage1Buffering stage1_buffering : 1;
    unsigned long padding : 40;

    constexpr unsigned long Repr() const {
        return (static_cast<unsigned long>(act_dtype) << 0) |
               (static_cast<unsigned long>(weight_dtype) << 4) |
               (static_cast<unsigned long>(bias_dtype) << 8) |
               (static_cast<unsigned long>(weight_ordering) << 12) |
               (static_cast<unsigned long>(mfma) << 14) |
               (static_cast<unsigned long>(stages) << 16) |
               (static_cast<unsigned long>(activation) << 20) |
               (static_cast<unsigned long>(stage1_buffering) << 23);
    }

    static constexpr FusedMoESolutionId FromRepr(unsigned long repr) {
        return FusedMoESolutionId{
            static_cast<FusedMoEDataType>((repr >> 0) & 0xf),
            static_cast<FusedMoEDataType>((repr >> 4) & 0xf),
            static_cast<FusedMoEDataType>((repr >> 8) & 0xf),
            static_cast<FusedMoEWeightOrdering>((repr >> 12) & 0x3),
            static_cast<FusedMoEMfmaShape>((repr >> 14) & 0x3),
            static_cast<FusedMoEStages>((repr >> 16) & 0xf),
            static_cast<FusedMoEActivationFunction>((repr >> 20) & 0x7),
            static_cast<FusedMoEStage1Buffering>((repr >> 23) & 0x1),
            0,
        };
    }

    static constexpr FusedMoESolutionId
    Make(FusedMoEDataType act_dtype, FusedMoEDataType weight_dtype,
         FusedMoEDataType bias_dtype, FusedMoEWeightOrdering weight_ordering,
         FusedMoEMfmaShape mfma, FusedMoEStages stages,
         FusedMoEActivationFunction activation,
         FusedMoEStage1Buffering stage1_buffering) {
        return FusedMoESolutionId{
            act_dtype, weight_dtype, bias_dtype, weight_ordering,
            mfma,      stages,       activation,  stage1_buffering,
            0,
        };
    }
};
static_assert(sizeof(FusedMoESolutionId) == 8, "");

struct FusedMoE1StageParams {
    unsigned *out;
    const unsigned *act;
    const unsigned *w13;
    const unsigned *w2;
    const unsigned *sorted_token_ids;
    const unsigned *sorted_weights;
    const unsigned *sorted_expert_ids;
    const unsigned *num_valid_ids;
    unsigned topk;
    const unsigned *scales_act;
    const unsigned *scales_w13;
    const unsigned *scales_w2;
    unsigned max_num_m_blocks;
    unsigned m;
    unsigned n;
    unsigned k;
    unsigned num_experts;
    hipStream_t stream;
    unsigned num_persistent_tgs;
    const void *w13_bias = nullptr;
    const void *w2_bias = nullptr;
};

int FusedMoEMatmul1Stage(FusedMoE1StageParams params,
                         unsigned long solution_id);

template <unsigned long kRepr> struct FusedMoESolutionAdapter {
    static int Invoke(FusedMoE1StageParams params);
};

} // namespace causalflow::petit::rocm::moe
