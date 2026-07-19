#pragma once

#include "moe/rocm/mem/bias.cuh"
#include "moe/rocm/ops/cshuffle.cuh"
#include "moe/rocm/ops/schedule_matmul.cuh"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::moe {

template <class Weight> struct MxFp4Tile {
    uint4 value[2][Weight::kLoadGlobal];
    unsigned scale[Weight::kLoadGlobal / 2];

    static_assert(Weight::kLoadGlobal == 2 || Weight::kLoadGlobal == 4 ||
                      Weight::kLoadGlobal == 8,
                  "MXFP4 tiles require two, four, or eight N16 fragments");
};

template <class Config_, class Weight_> struct BlockScaleFp8TileOps {
    using Config = Config_;
    using Weight = Weight_;
    using Input = typename Config::Input;
    using MatmulOp = BlockScaleFp8Matmul;
    using CShuffle = InterleavedRowMajorCShuffle;

    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kKStages = 2;
    static constexpr unsigned kActivationFragments = 8;
    static constexpr unsigned kAccumFragments = 8;
    static constexpr unsigned kOutputPacksPerToken = 2;
    static constexpr bool kStage2BiasUsesTileK = true;

    struct Shm {
        unsigned act[Input::kShmInputElements];
        float scale[Input::kShmScaleElements];
    };

    struct InputRegs {
        uint4 x[kActivationFragments];
        float4 scale;
    };

    struct Tile {
        uint4 value[2][Weight::kTileLoads];
        float scale;
    };

    static_assert(Weight::kTileLoads == 8,
                  "FP8 N256 tile requires eight weight fragments");

    __device__ static void
    PrefetchInput(Input &input, Shm *shm, unsigned wid, unsigned wtid,
                  const unsigned tokens[Input::kTokenBatch], unsigned m) {
        input.FetchAsync(shm->act, wid, wtid, tokens);
        input.FetchScaleAsync(shm->scale, wid, wtid, tokens, m);
    }

    __device__ static void ReadInput(Input &input, InputRegs &regs,
                                     const Shm *shm, unsigned wtid) {
        input.FetchToRegs(regs.x, shm->act, wtid);
        regs.scale = input.FetchScaleToReg(shm->scale, wtid);
    }

    __device__ static void Load(Weight &weight, Tile &tile, unsigned tid,
                                unsigned wid, unsigned wtid) {
        weight.LoadTile(tile.value[0], 0, wid, wtid);
        weight.LoadTile(tile.value[1], 1, wid, wtid);
        tile.scale = weight.LoadScale(tid);
    }

    __device__ static void
    LoadProjection(Weight &weight, Tile &tile, unsigned tid, unsigned wid,
                   unsigned wtid, unsigned value_offset,
                   unsigned scale_offset) {
        weight.LoadTile(tile.value[0], 0, wid, wtid, value_offset);
        weight.LoadTile(tile.value[1], 1, wid, wtid, value_offset);
        tile.scale = weight.LoadScale(tid, scale_offset);
    }

    __device__ static void Matmul(float4 t[kAccumFragments], const Tile &tile,
                                  const InputRegs &input, unsigned) {
        MatmulOp::Matmul(t, tile.value[0], input.x, input.scale, tile.scale,
                         0);
        MatmulOp::Matmul(t, tile.value[1], input.x, input.scale, tile.scale,
                         1);
    }

};

template <class Config_, class Weight_> struct PetitMxFp4TileOps {
    using Config = Config_;
    using Weight = Weight_;
    using Input = typename Config::Input;
    using MatmulOp = PetitMxFp4Matmul;
    using CShuffle = BlockedVectorRowMajorCShuffle;

    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kKStages = Config::kGroupDim / 128;
    static constexpr unsigned kActivationFragments = Config::kGroupDim / 32;
    static constexpr unsigned kAccumFragments = 2 * Weight::kLoadGlobal;
    static constexpr unsigned kOutputPacksPerToken = Config::kGroupN / 128;
    static constexpr bool kStage2BiasUsesTileK = true;

    struct Shm {
        unsigned act[Input::kShmInputElements];
        float scale[Input::kThreads];
    };

    struct InputRegs {
        uint4 x[kActivationFragments];
        float4 scale;
    };

    using Tile = MxFp4Tile<Weight>;

    static_assert(kKStages == 2, "Petit MXFP4 schedules require K256 tiles");
    static_assert(kActivationFragments == MatmulOp::kActivationFragments,
                  "Petit MXFP4 input fragments must match the MFMA policy");
    static_assert(Weight::kLoadGlobal == MatmulOp::kWeightFragments,
                  "Petit MXFP4 weight fragments must match the MFMA policy");
    static_assert(kAccumFragments == Config::kGroupDim / 32,
                  "MXFP4 accumulator layout must cover the N256 tile");

    __device__ static void
    PrefetchInput(Input &input, Shm *shm, unsigned wid, unsigned wtid,
                  const unsigned tokens[Input::kTokenBatch], unsigned m) {
        input.FetchAsync(shm->act, wid, wtid, tokens);
        input.FetchScaleAsync(shm->scale, wid, wtid, tokens, m);
    }

    __device__ static void ReadInput(Input &input, InputRegs &regs,
                                     const Shm *shm, unsigned wtid) {
        input.FetchToRegsFP4(regs.x, shm->act, wtid);
        regs.scale = input.FetchScaleToReg(shm->scale, wtid);
    }

    __device__ static void Load(Weight &weight, Tile &tile, unsigned,
                                unsigned wid, unsigned wtid) {
        weight.LoadTile(tile.value[0], 0, wid, wtid);
        weight.LoadTile(tile.value[1], 1, wid, wtid);
        tile.scale[0] = weight.LoadScale(wid, wtid, 0);
        tile.scale[1] = weight.LoadScale(wid, wtid, 1);
    }

    __device__ static void
    LoadProjection(Weight &weight, Tile &tile, unsigned, unsigned wid,
                   unsigned wtid, unsigned value_offset,
                   unsigned scale_offset) {
        weight.LoadTile(tile.value[0], 0, wid, wtid, value_offset);
        weight.LoadTile(tile.value[1], 1, wid, wtid, value_offset);
        tile.scale[0] = weight.LoadScale(wid, wtid, 0, scale_offset);
        tile.scale[1] = weight.LoadScale(wid, wtid, 1, scale_offset);
    }

    __device__ static void Matmul(float4 t[kAccumFragments], const Tile &tile,
                                  const InputRegs &input, unsigned) {
#pragma unroll
        for (unsigned k = 0; k < kKStages; ++k) {
            MatmulOp::Matmul(t, tile.value[k], input.x, input.scale,
                             tile.scale, k);
        }
    }

};

template <class Config_, class Weight_> struct Bf16MxFp4TileOps {
    using Config = Config_;
    using Weight = Weight_;
    using Input = typename Config::Input;
    using MatmulOp = Bf16MxFp4Matmul;
    using CShuffle = BlockedVectorRowMajorCShuffle;

    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kKStages = Config::kGroupDim / 128;
    static constexpr unsigned kActivationFragments = Config::kGroupDim / 16;
    static constexpr unsigned kAccumFragments = 2 * Weight::kLoadGlobal;
    static constexpr unsigned kOutputPacksPerToken = Config::kGroupN / 128;
    static constexpr bool kStage2BiasUsesTileK = false;

    using Shm = typename Input::Shm;
    struct InputRegs {
        uint4 x[kActivationFragments];
    };

    using Tile = MxFp4Tile<Weight>;

    static_assert(kKStages == 2, "BF16 x MXFP4 schedules require K256 tiles");
    static_assert(kActivationFragments == Input::kActivationFragments,
                  "BF16 input fragments must match the MFMA policy");
    static_assert(kActivationFragments == MatmulOp::kActivationFragments,
                  "BF16 input fragments must match the MFMA tile");
    static_assert(Weight::kLoadGlobal == MatmulOp::kWeightFragments,
                  "MXFP4 weight fragments must match the MFMA tile");

    __device__ static void
    PrefetchInput(Input &input, Shm *shm, unsigned wid, unsigned wtid,
                  const unsigned tokens[Input::kTokenBatch], unsigned) {
        uint4 regs[Input::kLoadGlobal];
        input.FetchGlobal(regs, wtid, tokens);
        input.StoreShared(regs, wid, wtid, shm);
        input.AdvanceStep();
    }

    __device__ static void ReadInput(Input &input, InputRegs &regs,
                                     const Shm *shm, unsigned wtid) {
        input.FetchToRegs(regs.x, *shm, wtid);
    }

    __device__ static void Load(Weight &weight, Tile &tile, unsigned,
                                unsigned wid, unsigned wtid) {
        weight.LoadTile(tile.value[0], 0, wid, wtid);
        weight.LoadTile(tile.value[1], 1, wid, wtid);
        tile.scale[0] = weight.LoadScale(wid, wtid, 0);
        tile.scale[1] = weight.LoadScale(wid, wtid, 1);
    }

    __device__ static void
    LoadProjection(Weight &weight, Tile &tile, unsigned, unsigned wid,
                   unsigned wtid, unsigned value_offset,
                   unsigned scale_offset) {
        weight.LoadTile(tile.value[0], 0, wid, wtid, value_offset);
        weight.LoadTile(tile.value[1], 1, wid, wtid, value_offset);
        tile.scale[0] = weight.LoadScale(wid, wtid, 0, scale_offset);
        tile.scale[1] = weight.LoadScale(wid, wtid, 1, scale_offset);
    }

    __device__ static void Matmul(float4 t[kAccumFragments], const Tile &tile,
                                  const InputRegs &input, unsigned) {
#pragma unroll
        for (unsigned k = 0; k < kKStages; ++k) {
            MatmulOp::Matmul(t, tile.value[k], input.x, tile.scale, k);
        }
    }

};

template <class Config_, class Weight_> struct NativeMxFp4TileOps {
    using Config = Config_;
    using Weight = Weight_;
    using Input = typename Config::Input;
    static constexpr unsigned kWaveTileN = Weight::kWaveTileN;
    using MatmulOp = NativeMxFp4Matmul<kWaveTileN>;
    using CShuffle = BlockedVectorRowMajorCShuffle;

    static constexpr unsigned kNumWarps = Config::kNumWarps;
    static constexpr unsigned kKStages = Config::kGroupDim / 128;
    static constexpr unsigned kActivationFragments = 2 * kKStages;
    static constexpr unsigned kScaleFragments = kKStages / 2;
    static constexpr unsigned kAccumFragments = 2 * Weight::kLoadGlobal;
    static constexpr unsigned kOutputPacksPerToken = Config::kGroupN / 128;
    static constexpr bool kStage2BiasUsesTileK = false;
    static constexpr int kWeightLoadAux = [] {
        if constexpr (requires { Config::kWeightLoadAux; })
            return Config::kWeightLoadAux;
        else
            return TargetWeightLoadPolicy::kAux;
    }();

    using Shm = typename Input::Shm;
    struct InputRegs {
        uint4 x[kActivationFragments];
        unsigned scale[kScaleFragments];
    };

    using Tile = MxFp4Tile<Weight>;

    static_assert(kKStages == 2, "native MXFP4 schedules require K256 tiles");
    static_assert(kActivationFragments == Input::kActivationFragments,
                  "native MXFP4 input fragments must match the MFMA policy");
    static_assert(kActivationFragments == MatmulOp::kActivationFragments,
                  "native MXFP4 input fragments must match the MFMA tile");
    static_assert(Weight::kLoadGlobal == MatmulOp::kWeightFragments,
                  "native MXFP4 weight fragments must match the MFMA tile");

    __device__ static void
    PrefetchInput(Input &input, Shm *shm, unsigned wid, unsigned wtid,
                  const unsigned tokens[Input::kTokenBatch], unsigned m) {
        input.FetchAsync(shm->act, wid, wtid, tokens);
        input.FetchScaleAsync(shm->scale, wid, wtid, tokens, m);
    }

    __device__ static void ReadInput(Input &input, InputRegs &regs,
                                     const Shm *shm, unsigned wtid) {
        input.FetchToRegs(regs.x, shm->act, wtid);
        regs.scale[0] = input.FetchScaleToReg(shm->scale, wtid);
        input.AdvanceScaleStep();
    }

    __device__ static void Load(Weight &weight, Tile &tile, unsigned,
                                unsigned wid, unsigned wtid) {
        weight.template LoadTile<kWeightLoadAux>(tile.value[0], 0, wid, wtid);
        weight.template LoadTile<kWeightLoadAux>(tile.value[1], 1, wid, wtid);
#pragma unroll
        for (unsigned n32_pair = 0;
             n32_pair < Weight::kLoadGlobal / 2; ++n32_pair)
            tile.scale[n32_pair] =
                weight.LoadScale(wid, wtid, n32_pair);
    }

    __device__ static void
    LoadProjection(Weight &weight, Tile &tile, unsigned, unsigned wid,
                   unsigned wtid, unsigned value_offset,
                   unsigned scale_offset) {
        weight.template LoadTile<kWeightLoadAux>(tile.value[0], 0, wid, wtid,
                                                 value_offset);
        weight.template LoadTile<kWeightLoadAux>(tile.value[1], 1, wid, wtid,
                                                 value_offset);
#pragma unroll
        for (unsigned n32_pair = 0;
             n32_pair < Weight::kLoadGlobal / 2; ++n32_pair)
            tile.scale[n32_pair] =
                weight.LoadScale(wid, wtid, n32_pair, scale_offset);
    }

    __device__ static void Matmul(float4 t[kAccumFragments], const Tile &tile,
                                  const InputRegs &input, unsigned) {
        MatmulOp::Matmul(t, tile.value, input.x, input.scale[0], tile.scale);
    }

};

} // namespace causalflow::petit::rocm::moe
