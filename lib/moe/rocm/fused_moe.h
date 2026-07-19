#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

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
    kMfmaBf16MxFp4,
    kMfmaScaleFp4MxFp4,
};

enum class FusedMoEActivationFunction : unsigned {
    kSiluDot,
    kOpenAISwiGLU,
};

enum class FusedMoEStage1Buffering : unsigned {
    kSingleBuffer,
    kDoubleBuffer,
};

enum class FusedMoEWeightLoadPolicy : unsigned {
    kCached,
    kNonTemporal,
};

enum class MegaMoETileShape : unsigned { kN256, kN128 };

enum class MegaMoEProducerGeometry : unsigned {
    kCta56,
    kCta64,
    kCta128,
    kCta192,
};

// W13 names include both gate and up projections.  The default local two-stage
// tile computes M32 x (N128 gate + N128 up); the large tile computes
// M64 x (N256 gate + N256 up) with the same four waves.
enum class FusedMoEStage1TileShape : unsigned { kM32N256, kM64N512 };

struct FusedMoESolutionId {
    FusedMoEDataType act_dtype : 4;
    FusedMoEDataType weight_dtype : 4;
    FusedMoEDataType bias_dtype : 4;
    FusedMoEWeightOrdering weight_ordering : 2;
    FusedMoEMfmaShape mfma : 2;
    FusedMoEStages stages : 4;
    FusedMoEActivationFunction activation : 3;
    FusedMoEStage1Buffering stage1_buffering : 1;
    unsigned long dim_div64 : 8;
    unsigned long inter_dim_div64 : 8;
    FusedMoEWeightLoadPolicy weight_load_policy : 1;
    unsigned long padding : 23;

    static constexpr unsigned kShapeAlignment = 64;
    static constexpr unsigned kMaxShapeDiv64 = 0xff;

    constexpr unsigned Dim() const { return dim_div64 * kShapeAlignment; }
    constexpr unsigned InterDim() const {
        return inter_dim_div64 * kShapeAlignment;
    }

    static constexpr bool IsShapeEncodable(unsigned dim, unsigned inter_dim) {
        return dim != 0 && inter_dim != 0 && dim % kShapeAlignment == 0 &&
               inter_dim % kShapeAlignment == 0 &&
               dim / kShapeAlignment <= kMaxShapeDiv64 &&
               inter_dim / kShapeAlignment <= kMaxShapeDiv64;
    }

    constexpr FusedMoESolutionId WithShape(unsigned dim,
                                           unsigned inter_dim) const {
        return FusedMoESolutionId{
            act_dtype,
            weight_dtype,
            bias_dtype,
            weight_ordering,
            mfma,
            stages,
            activation,
            stage1_buffering,
            dim / kShapeAlignment,
            inter_dim / kShapeAlignment,
            weight_load_policy,
            padding,
        };
    }

    constexpr FusedMoESolutionId
    WithWeightLoadPolicy(FusedMoEWeightLoadPolicy policy) const {
        return FusedMoESolutionId{
            act_dtype,
            weight_dtype,
            bias_dtype,
            weight_ordering,
            mfma,
            stages,
            activation,
            stage1_buffering,
            dim_div64,
            inter_dim_div64,
            policy,
            padding,
        };
    }

    // MegaMoE reuses the common fields in bits [0, 23] and has a compact,
    // independent layout in bits [24, 49]. Bits [50, 63] are reserved.
    constexpr unsigned NumRanksLog2() const { return (Repr() >> 24) & 0x3; }
    constexpr unsigned NumExperts() const {
        return (((Repr() >> 26) & 0xf) + 1) * 32;
    }
    constexpr unsigned TopK() const { return (Repr() >> 30) & 0xf; }
    constexpr unsigned HiddenSizeDiv64() const {
        return (Repr() >> 34) & 0xff;
    }
    constexpr MegaMoETileShape W2TileShape() const {
        return static_cast<MegaMoETileShape>((Repr() >> 42) & 0x1);
    }
    constexpr unsigned MegaInterDimDiv64() const {
        return (((Repr() >> 43) & 0x1f) + 1) * 8;
    }
    constexpr MegaMoEProducerGeometry ProducerGeometry() const {
        return static_cast<MegaMoEProducerGeometry>((Repr() >> 48) & 0x3);
    }
    constexpr unsigned ProducerBlocks() const {
        switch (ProducerGeometry()) {
        case MegaMoEProducerGeometry::kCta56:
            return 56;
        case MegaMoEProducerGeometry::kCta64:
            return 64;
        case MegaMoEProducerGeometry::kCta128:
            return 128;
        case MegaMoEProducerGeometry::kCta192:
            return 192;
        }
        return 0;
    }

    constexpr FusedMoEStage1TileShape Stage1TileShape() const {
        return static_cast<FusedMoEStage1TileShape>((Repr() >> 41) & 0x1);
    }

    constexpr unsigned Stage1TileM() const {
        return 32u << static_cast<unsigned>(Stage1TileShape());
    }

    constexpr unsigned Stage1TileN() const {
        return 256u << static_cast<unsigned>(Stage1TileShape());
    }

    constexpr FusedMoESolutionId
    WithStage1TileShape(FusedMoEStage1TileShape shape) const {
        constexpr unsigned long kMask = 1ul << 41;
        return FromRepr((Repr() & ~kMask) |
                        (static_cast<unsigned long>(shape) << 41));
    }

    constexpr FusedMoESolutionId WithMegaMoEConfig(
        unsigned num_ranks, unsigned experts, unsigned num_topk,
        unsigned hidden_size, unsigned inter_dim,
        MegaMoEProducerGeometry producer_geometry,
        MegaMoETileShape w2_tile_shape = MegaMoETileShape::kN256) const {
        unsigned rank_log2 = 0;
        for (; num_ranks > 1; num_ranks >>= 1)
            ++rank_log2;
        return FromRepr(
            (Repr() & 0x00fffffful) |
            (static_cast<unsigned long>(rank_log2) << 24) |
            (static_cast<unsigned long>(experts / 32 - 1) << 26) |
            (static_cast<unsigned long>(num_topk) << 30) |
            (static_cast<unsigned long>(hidden_size / 64) << 34) |
            (static_cast<unsigned long>(w2_tile_shape) << 42) |
            (static_cast<unsigned long>(inter_dim / 512 - 1) << 43) |
            (static_cast<unsigned long>(producer_geometry) << 48));
    }

    constexpr unsigned long Repr() const {
        return (static_cast<unsigned long>(act_dtype) << 0) |
               (static_cast<unsigned long>(weight_dtype) << 4) |
               (static_cast<unsigned long>(bias_dtype) << 8) |
               (static_cast<unsigned long>(weight_ordering) << 12) |
               (static_cast<unsigned long>(mfma) << 14) |
               (static_cast<unsigned long>(stages) << 16) |
               (static_cast<unsigned long>(activation) << 20) |
               (static_cast<unsigned long>(stage1_buffering) << 23) |
               (static_cast<unsigned long>(dim_div64) << 24) |
               (static_cast<unsigned long>(inter_dim_div64) << 32) |
               (static_cast<unsigned long>(weight_load_policy) << 40) |
               (static_cast<unsigned long>(padding) << 41);
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
            (repr >> 24) & 0xff,
            (repr >> 32) & 0xff,
            static_cast<FusedMoEWeightLoadPolicy>((repr >> 40) & 0x1),
            (repr >> 41) & 0x7fffff,
        };
    }

    static constexpr FusedMoESolutionId
    MakeBase(FusedMoEDataType act_dtype, FusedMoEDataType weight_dtype,
             FusedMoEDataType bias_dtype,
             FusedMoEWeightOrdering weight_ordering, FusedMoEMfmaShape mfma,
             FusedMoEStages stages, FusedMoEActivationFunction activation,
             FusedMoEStage1Buffering stage1_buffering) {
        return FusedMoESolutionId{
            act_dtype,  weight_dtype,
            bias_dtype, weight_ordering,
            mfma,       stages,
            activation, stage1_buffering,
            0,          0,
            FusedMoEWeightLoadPolicy::kCached,
            0,
        };
    }

    static constexpr FusedMoESolutionId MakeMegaBase(
        FusedMoEDataType act_dtype, FusedMoEDataType weight_dtype,
        FusedMoEDataType bias_dtype, FusedMoEWeightOrdering weight_ordering,
        FusedMoEMfmaShape mfma, FusedMoEStages stages,
        FusedMoEActivationFunction activation,
        FusedMoEStage1Buffering stage1_buffering) {
        return MakeBase(act_dtype, weight_dtype, bias_dtype, weight_ordering,
                        mfma, stages, activation, stage1_buffering);
    }

    static constexpr FusedMoESolutionId
    Make(FusedMoEDataType act_dtype, FusedMoEDataType weight_dtype,
         FusedMoEDataType bias_dtype, FusedMoEWeightOrdering weight_ordering,
         FusedMoEMfmaShape mfma, FusedMoEStages stages,
         FusedMoEActivationFunction activation,
         FusedMoEStage1Buffering stage1_buffering, unsigned dim,
         unsigned inter_dim) {
        return MakeBase(act_dtype, weight_dtype, bias_dtype, weight_ordering,
                        mfma, stages, activation, stage1_buffering)
            .WithShape(dim, inter_dim);
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

struct FusedMoE2StageCommonParams {
    void *intermediate;
    std::size_t intermediate_bytes;
    const unsigned *sorted_token_ids;
    const unsigned *sorted_expert_ids;
    const unsigned *num_valid_ids;
    unsigned topk;
    unsigned max_num_m_blocks;
    unsigned m;
    unsigned n;
    unsigned k;
    unsigned num_experts;
    hipStream_t stream;
    unsigned num_persistent_tgs;
};

struct FusedMoE2Stage1Params {
    FusedMoE2StageCommonParams common;
    const unsigned *act;
    const unsigned *w13;
    const unsigned *scales_act;
    const unsigned *scales_w13;
    const void *w13_bias = nullptr;
};

struct FusedMoE2Stage2Params {
    FusedMoE2StageCommonParams common;
    unsigned *out;
    const unsigned *w2;
    const unsigned *sorted_weights;
    const unsigned *scales_w2;
    const void *w2_bias = nullptr;
};

std::size_t FusedMoE2StageWorkspaceSize(unsigned max_num_m_blocks,
                                        unsigned inter_dim,
                                        unsigned long solution_id);

int FusedMoEMatmul2Stage1(FusedMoE2Stage1Params params,
                          unsigned long solution_id);

int FusedMoEMatmul2Stage2(FusedMoE2Stage2Params params,
                          unsigned long solution_id);

template <unsigned long kRepr> struct FusedMoESolutionAdapter {
    static int Invoke(FusedMoE1StageParams params);
};

} // namespace causalflow::petit::rocm::moe
