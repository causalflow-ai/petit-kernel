#pragma once

#include "causalflow/petit/tal/algorithm.h"
#include "moe/rocm/memory_ops.cuh"
#include "moe/rocm/ops/stage1_accumulator_lds.cuh"
#include "moe/rocm/quantization.cuh"

namespace causalflow::petit::rocm::moe {

// Storage for stage-1 activations consumed by an MXFP4 stage-2 GEMM. Values
// are E2M1 nibbles. E8M0 scales are stored separately in row-major order and
// assembled into the stage-2 MFMA lane layout when loaded.
struct MxFp4ActivationLayout {
    TAL_HOST_DEVICE static constexpr unsigned ValueBytes(unsigned rows,
                                                         unsigned inter_dim) {
        return rows * inter_dim / 2;
    }

    TAL_HOST_DEVICE static constexpr unsigned PaddedScaleRows(unsigned rows) {
        return tal::CeilingDiv<unsigned>(rows, 256) * 256;
    }

    TAL_HOST_DEVICE static constexpr unsigned ScaleCols(unsigned inter_dim) {
        return tal::CeilingDiv<unsigned>(inter_dim / 32, 8) * 8;
    }

    TAL_HOST_DEVICE static constexpr unsigned ScaleBytes(unsigned rows,
                                                         unsigned inter_dim) {
        return PaddedScaleRows(rows) * ScaleCols(inter_dim);
    }

    TAL_HOST_DEVICE static constexpr unsigned
    ScaleOffset(unsigned row, unsigned col, unsigned scale_cols) {
        return row * scale_cols + col;
    }
};

// Shared producer for local and distributed two-stage MoE. It quantizes one
// FP32 stage-1 accumulator tile and stores packed values and scales.
template <class Config> struct MxFp4ActivationQuantizer {
    static constexpr unsigned kRowsPerTile = Config::kGroupM;
    static constexpr unsigned kTileCols = Config::kStage1GroupN;
    static constexpr unsigned kInputFragments =
        (kRowsPerTile * kTileCols) /
        (Config::kNumWarps * kWarpSize) / 4;
    using QuantizeShm = float[kRowsPerTile * kTileCols];

    struct Quantized {
        unsigned short value;
        unsigned scale;
    };

    TAL_DEVICE static void
    StoreAccumulator(QuantizeShm &shm,
                     const float4 h[kInputFragments], unsigned wid,
                     unsigned wtid) {
        StoreStage1AccumulatorLds2D<
            kRowsPerTile, kTileCols, Config::kStage1WarpsM,
            Config::kStage1WarpsN>(shm, h, wid, wtid);
    }

    TAL_DEVICE static Quantized Quantize(const QuantizeShm &shm,
                                         unsigned row,
                                         unsigned col_lane) {
        const auto *value = reinterpret_cast<const float4 *>(
            &shm[row * kTileCols + col_lane * 4]);
        float max_abs = AiterMxFp4Quantization::MaximumAbs(*value);
        max_abs = ReduceMaximum<0x041f>(max_abs);
        max_abs = ReduceMaximum<0x081f>(max_abs);
        max_abs = ReduceMaximum<0x101f>(max_abs);

        Quantized result;
        result.scale = QuantizeMxFp4<AiterMxFp4Quantization, 1>(
            reinterpret_cast<unsigned char *>(&result.value), value,
            max_abs);
        return result;
    }

    template <int kStoreScope = BufferResource::kNone>
    TAL_DEVICE static void
    Store(const BufferResource &workspace, unsigned value_base,
          unsigned scale_base,
          unsigned value_row, unsigned scale_row, unsigned tile_n,
          unsigned col_lane, unsigned inter_dim, unsigned scale_cols,
          const Quantized &quantized) {
        const unsigned col_local = col_lane * 4;
        const unsigned value_offset =
            value_row * (inter_dim / 2) + tile_n * (kTileCols / 2) +
            col_local / 2;
        const unsigned partner = __builtin_amdgcn_mov_dpp(
            static_cast<unsigned>(quantized.value), 0xb1, 0xf, 0xf, false);
        if ((col_lane & 1u) == 0) {
            const unsigned packed = static_cast<unsigned>(quantized.value) |
                                    (partner << 16);
            workspace.template StoreU32<kStoreScope |
                                        BufferResource::kNTBit>(
                value_offset, value_base, packed);
        }

        const unsigned scale0 = __shfl(quantized.scale, 0, 32);
        const unsigned scale1 = __shfl(quantized.scale, 8, 32);
        const unsigned scale2 = __shfl(quantized.scale, 16, 32);
        const unsigned scale3 = __shfl(quantized.scale, 24, 32);
        if ((col_lane & 31u) == 0) {
            const unsigned packed = scale0 | (scale1 << 8) | (scale2 << 16) |
                                    (scale3 << 24);
            const unsigned scale_col =
                tile_n * (kTileCols / 32) + (col_lane / 32) * 4;
            workspace.template StoreU32<kStoreScope>(
                MxFp4ActivationLayout::ScaleOffset(scale_row, scale_col,
                                                    scale_cols),
                scale_base, packed);
        }
    }

  private:
    template <unsigned kPattern>
    TAL_DEVICE static float ReduceMaximum(float value) {
        const unsigned bits = reinterpret_cast<const unsigned &>(value);
        const unsigned peer_bits = __builtin_amdgcn_ds_swizzle(bits, kPattern);
        const float peer = reinterpret_cast<const float &>(peer_bits);
        return __builtin_elementwise_maximum(value, peer);
    }
};

// Shared consumer for local and distributed two-stage MoE. It loads one
// packed activation tile and presents the register layout required by W2.
template <class Config, class InputRegs> struct MxFp4Stage2Input {
    static constexpr unsigned kRowsPerTile = Config::kStage2GroupM;
    static constexpr unsigned kGroupDim = Config::kGroupDim;
    static constexpr unsigned kVectorsPerRow =
        (kGroupDim / 2) / sizeof(uint4);
    static constexpr unsigned kScaleCols =
        MxFp4ActivationLayout::ScaleCols(Config::kInterDim);
    static constexpr unsigned kLdsStages = 2;
    static constexpr unsigned kLdsVectorsPerRow = 16;
    static constexpr unsigned kInvalidValueOffset = (unsigned)-16;
    static_assert(sizeof(uint4) == 16);

    using InputShm =
        uint4[kLdsStages][kRowsPerTile][kLdsVectorsPerRow];

    struct Prefetch {
        uint4 value;
        unsigned scale;
    };

    template <int kAux = BufferResource::kNone>
    TAL_DEVICE static Prefetch
    LoadTile(const BufferResource &workspace, unsigned value_voffset,
             unsigned value_soffset, unsigned scale_voffset,
             unsigned scale_soffset,
             unsigned tile_k, bool valid, unsigned wtid) {
        const unsigned value_offset =
            valid ? value_voffset + tile_k * kGroupDim / 2
                  : kInvalidValueOffset;
        return {
            workspace.template Load<kAux>(
                value_offset, value_soffset),
            LoadScaleWord<kAux>(workspace, scale_voffset, scale_soffset,
                                tile_k, wtid),
        };
    }

    TAL_DEVICE static void StoreLds(InputShm &shm, const uint4 &value,
                                    unsigned stage, unsigned tid) {
        const unsigned row = tid / kVectorsPerRow;
        const unsigned vector = tid % kVectorsPerRow;
        shm[stage][row][vector ^ (row & 15u)] = value;
    }

    TAL_DEVICE static InputRegs ReadLds(InputShm &shm, unsigned stage,
                                        unsigned scale, unsigned wtid) {
        InputRegs input;
#pragma unroll
        for (unsigned row_group = 0; row_group < 2; ++row_group) {
#pragma unroll
            for (unsigned k128 = 0; k128 < 2; ++k128) {
                const unsigned row = wtid % 16 + row_group * 16;
                const unsigned vector = wtid / 16 + k128 * 4;
                input.x[row_group * 2 + k128] =
                    shm[stage][row][vector ^ (row & 15u)];
            }
        }
        input.scale[0] = scale;
        return input;
    }

  private:
    template <int kAux>
    TAL_DEVICE static unsigned
    LoadScaleWord(const BufferResource &workspace, unsigned scale_voffset,
                  unsigned scale_soffset, unsigned tile_k, unsigned wtid) {
        // A K256 tile contains 32 rows by eight scales, or exactly 64 u32
        // words. Have every lane load one distinct row-major word, then
        // exchange those words within the wave to assemble the MFMA scale
        // layout. This retains aligned u32 producer stores without issuing
        // four coherent global loads in every consumer lane.
        const unsigned load_row = wtid & 31u;
        const unsigned load_col = tile_k * 8 + (wtid >> 5) * 4;
        const unsigned loaded = workspace.template LoadU32<kAux>(
            scale_voffset + load_row * kScaleCols + load_col,
            scale_soffset);

        const unsigned row16 = wtid & 15u;
        const unsigned scale4 = wtid >> 4;
        const unsigned shift = scale4 * 8;
        return ((__shfl(loaded, row16) >> shift) & 0xffu) |
               (((__shfl(loaded, row16 + 16) >> shift) & 0xffu) << 8) |
               (((__shfl(loaded, row16 + 32) >> shift) & 0xffu) << 16) |
               (((__shfl(loaded, row16 + 48) >> shift) & 0xffu) << 24);
    }
};

} // namespace causalflow::petit::rocm::moe
