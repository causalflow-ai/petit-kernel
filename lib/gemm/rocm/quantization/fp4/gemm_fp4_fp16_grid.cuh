#include "gemm/gpu/quantization/reduce.cuh"
#include "gemm/rocm/quantization/gemm.h"
#include "gemm/rocm/quantization/kernel_config.cuh"
#include "gemm_fp4.h"
#include "utils/assert.h"
#include "warp_schedule_fp16.cuh"

#include <cmath>
#include <cstdio>
#include <functional>
#include <hip/hip_runtime.h>
#include <span>
#include <unordered_map>

namespace causalflow::petit::rocm::quantization::fp4 {

static constexpr unsigned kTile = 16;
unsigned long ChooseDefaultFp4Fp16Solution(unsigned m, unsigned n, unsigned k,
                                           const PetitSolutionHints &hints);
struct QWeightOnDiskLayout128x16 {
    // kLayoutM is dimension K
    static constexpr unsigned kLayoutM = 128;
    static constexpr unsigned kLayoutN = 16;

    // How many consecutive elements on the (m,n) orders
    static constexpr unsigned kLayoutElementsM = 4;
    static constexpr unsigned kLayoutElementsN = 1;
    static_assert(kLayoutElementsM * kLayoutElementsN == 4, "");
};

struct QWeightOnDiskLayout64x32 {
    // kLayoutM is dimension K
    static constexpr unsigned kLayoutM = 64;
    static constexpr unsigned kLayoutN = 32;

    // How many consecutive elements on the (m,n) orders
    static constexpr unsigned kLayoutElementsM = 2;
    static constexpr unsigned kLayoutElementsN = 2;
    static_assert(kLayoutElementsM * kLayoutElementsN == 4, "");
};

template <class TS, class WP> struct WarpMatmulLayoutTrait {
    using ElementA = typename TS::ElementA;
    // using DiskLayout = QWeightOnDiskLayout128x16;
    using DiskLayout = QWeightOnDiskLayout64x32;
    static constexpr unsigned kLayoutM = DiskLayout::kLayoutM;
    static constexpr unsigned kLayoutN = DiskLayout::kLayoutN;
    static constexpr unsigned kGroupK = TS::kGroupK;

    static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(ElementA);
    static constexpr unsigned kMmaM = 16;
    static constexpr unsigned kMmaTileK = kVecSize * kWarpSize / kMmaM;

    // The number of tiles in the m dimension that a warp is responsible to.
    static constexpr unsigned kWarpTileM = TS::kNumTileM / WP::kPartitionM;

    // Each warp works on a kLayoutM x kLayoutN tile.
    static constexpr unsigned kWarpAtomK = kGroupK / kLayoutM / WP::kPartitionK;
    static constexpr unsigned kWarpAtomN =
        TS::kGroupN / kLayoutN / WP::kPartitionN;
    static_assert(kGroupK % (kLayoutM * WP::kPartitionK) == 0 && kWarpAtomK > 0,
                  "");
    static_assert(TS::kGroupN % (kLayoutN * WP::kPartitionN) == 0 &&
                      kWarpAtomN > 0,
                  "");

    // Determine by the data layout and how the 4-uint quantized weights are
    // arranged.
    static constexpr unsigned kAccTilePerThread = kLayoutN / kTile;
    static constexpr unsigned kThreadAccumTileNRegs =
        kWarpAtomN * kAccTilePerThread;

    static_assert(kThreadAccumTileNRegs > 0, "");
    static_assert(kGroupK % kLayoutM == 0,
                  "kGroupK must be a multiple of kLayoutM");
    static_assert(TS::kNumTileN % WP::kPartitionN == 0 &&
                      TS::kNumTileK % WP::kPartitionK == 0,
                  "The number of tiles must be divisible by the partition "
                  "dimensions");

    using ThreadAccum = float4[kWarpTileM][kThreadAccumTileNRegs];
    using WarpAccumLayout = tal::Layout<
        tal::Shape<tal::C<kWarpAtomN>,
                   tal::Shape<tal::C<DiskLayout::kLayoutElementsM>,
                              tal::C<4 / DiskLayout::kLayoutElementsM>>>,
        tal::Stride<tal::C<kAccTilePerThread>, tal::Stride<tal::_0, tal::_1>>>;
    using WarpMatmulRegALayout = tal::Layout<
        tal::Shape<tal::Shape<tal::C<DiskLayout::kLayoutElementsM>,
                              tal::C<4 / DiskLayout::kLayoutElementsM>>>,
        tal::Stride<tal::Stride<tal::_1, tal::_0>>>;

    static constexpr unsigned kLayoutElementsM = DiskLayout::kLayoutElementsM;
    static constexpr unsigned kReadBatchA = DiskLayout::kLayoutElementsM *
                                            kPackFactor * sizeof(ElementA) /
                                            sizeof(uint4);

    __device__ static inline unsigned XorShmLayout(unsigned tile_idx_m,
                                                   unsigned tile_idx_k,
                                                   unsigned batch_id,
                                                   unsigned row, unsigned col) {
        static constexpr unsigned kBatchStride = DiskLayout::kLayoutElementsM;

        unsigned base = tile_idx_m * kTile * kGroupK / kVecSize +
                        tile_idx_k * kTile * kLayoutM / kVecSize +
                        batch_id * kWarpSize;
        unsigned xor_stride = col * kBatchStride + batch_id;
        return base + col * kMmaM + row ^ xor_stride;
    }

    __device__ static inline unsigned WriteShmCoordA(unsigned tid,
                                                     unsigned steps) {
        static_assert(WP::kThreads % (kGroupK / kVecSize) == 0,
                      "Unaligned GroupK");
        static constexpr unsigned kRowBlock = kGroupK / kVecSize;
        static constexpr unsigned kLoopInc = WP::kThreads / kRowBlock;

        unsigned row = tid / kRowBlock + kLoopInc * steps,
                 col = tid % kRowBlock;
        unsigned tile_idx_m = row / kTile,
                 tile_idx_k = col / (kLayoutM / kVecSize);
        unsigned row_in_tile = row % kTile,
                 col_in_tile = col % (kLayoutM / kVecSize);
        unsigned batch_id = col_in_tile % kLayoutElementsM;
        return XorShmLayout(tile_idx_m, tile_idx_k, batch_id, row_in_tile,
                            col_in_tile / kLayoutElementsM);
    }

    __device__ static inline unsigned ReadShmCoordA(unsigned tile_idx_m,
                                                    unsigned tile_idx_k,
                                                    unsigned batch_id,
                                                    unsigned wtid) {
        unsigned row_in_tile = wtid % kMmaM, col_in_tile = wtid / kMmaM;
        return XorShmLayout(tile_idx_m, tile_idx_k, batch_id, row_in_tile,
                            col_in_tile);
    }
};

template <class TileShape_, class WarpPartition_, unsigned kPipelineStages,
          bool kHighPrecision_>
struct GEMMFp4Fp16Config {
    using TS = TileShape_;
    using WP = WarpPartition_;
    using ElementA = typename TS::ElementA;
    using WarpMatmulLayout = WarpMatmulLayoutTrait<TS, WP>;

    static constexpr unsigned kLayoutM = WarpMatmulLayout::kLayoutM;
    static constexpr unsigned kLayoutN = WarpMatmulLayout::kLayoutN;

    static constexpr bool kHighPrecision = kHighPrecision_;
    static constexpr DataType kElementBTypeId = TS::kElementBTypeId;

    static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(ElementA);
    static constexpr unsigned kScaleVecSize = sizeof(uint4) / sizeof(unsigned char);

    static constexpr unsigned kTile = TS::kTile;
    static constexpr unsigned kNumTileM = TS::kNumTileM;
    static constexpr unsigned kNumTileN = TS::kNumTileN;
    static constexpr unsigned kNumTileK = TS::kNumTileK;
    static constexpr unsigned kGroupSize = TS::kGroupSize;
    static constexpr unsigned kGroupM = TS::kGroupM;
    static constexpr unsigned kGroupN = TS::kGroupN;
    static constexpr unsigned kGroupK = TS::kGroupK;

    static constexpr unsigned kNumWarps = WP::kNumWarps;
    static constexpr unsigned kThreads = WP::kThreads;

    static constexpr unsigned kWarpTileM = WarpMatmulLayout::kWarpTileM;

    static constexpr unsigned kStages = kPipelineStages;

    static constexpr bool kUseZeroPoints = false;
    static constexpr bool kZpInShm = false;

    // Determine by the data layout and how the 4-uint quantized weights are
    // arranged.
    static constexpr unsigned kThreadAccumTileNRegs =
        WarpMatmulLayout::kThreadAccumTileNRegs;
    using ThreadAccum = typename WarpMatmulLayout::ThreadAccum;
    using WarpAccumLayout = typename WarpMatmulLayout::WarpAccumLayout;
    using WarpMatmulRegALayout =
        typename WarpMatmulLayout::WarpMatmulRegALayout;

    __device__ static inline unsigned WriteShmCoordA(unsigned tid,
                                                     unsigned steps) {
        return WarpMatmulLayout::WriteShmCoordA(tid, steps);
    }

    __device__ static inline unsigned ReadShmCoordA(unsigned tile_idx_m,
                                                    unsigned tile_idx_k,
                                                    unsigned batch_id,
                                                    unsigned wtid) {
        return WarpMatmulLayout::ReadShmCoordA(tile_idx_m, tile_idx_k, batch_id,
                                               wtid);
    }
};

template <class Config> using WPMatmul = WarpPartitionMatmul<Config>;

template <class Config, unsigned kStages> struct PipelineContext {
    using LayoutA = MatrixALayout<Config>;
    using LayoutB = MatrixBLayout<Config>;
    using LayoutScale = ScaleLayout<Config>;

    static constexpr unsigned kVecSize = Config::kVecSize;
    static constexpr unsigned kGroupN = Config::kGroupN;
    static constexpr unsigned kGroupK = Config::kGroupK;
    static constexpr unsigned kLayoutM = Config::kLayoutM;
    static constexpr unsigned kScaleZpBatch =
        Config::kGroupK / Config::kGroupSize;

    unsigned n_;
    BufferResource r_a, r_b, r_scale;
    uint4 reg_a[kStages][LayoutA::kLoadGlobalA],
        reg_b[kStages][LayoutB::kLoadGlobalB];
    uint4 reg_packed_scales[kStages][LayoutScale::kLoadScale];

    __device__ void Initialize(const uint4 *a_ptr, const uint4 *b_ptr,
                               const uint4 *scale_ptr, unsigned tile_m,
                               unsigned n, unsigned k) {

        n_ = n;
        // The buffer load instruction automatically masks OOB accesses.
        r_a = LayoutA::InitializeBufferResource(tile_m, k, a_ptr);
        r_b = LayoutB::InitializeBufferResource(n, b_ptr);
        r_scale = LayoutScale::InitializeBufferResource(n, scale_ptr);
    }

    __device__ void AdvanceGlobalPtr() {
        r_a.v.ptr += kGroupK / kVecSize * sizeof(uint4);
        r_b.v.ptr += n_ * kGroupK / kPackFactor * sizeof(unsigned);
        r_scale.v.ptr +=
            n_ * kScaleZpBatch * sizeof(uint4) / Config::kScaleVecSize;
    }
};

template <class Config, class PipelineContext>
__device__ static void LoadGlobal(PipelineContext &ctx, unsigned n, unsigned k,
                                  unsigned stage, unsigned wid, unsigned tid) {
    using LayoutA = MatrixALayout<Config>;
    using LayoutB = MatrixBLayout<Config>;
    using LayoutScale = ScaleLayout<Config>;

    LayoutA::FetchGlobal(ctx.r_a, n, k, tid, ctx.reg_a[stage]);
    LayoutB::FetchGlobal(ctx.r_b, n, k, tid, ctx.reg_b[stage]);
    LayoutScale::FetchScale(ctx.r_scale, n, tid, ctx.reg_packed_scales[stage]);
}

template <class Config, class PipelineContext>
__device__ static void
StoreShm(const PipelineContext &ctx,
         typename ShmBuf<Config>::Data *__restrict__ shm_data,
         unsigned stage_reg, unsigned tid) {
    using LayoutA = MatrixALayout<Config>;
    using LayoutB = MatrixBLayout<Config>;
    using LayoutScale = ScaleLayout<Config>;

    LayoutA::StoreShared(ctx.reg_a[stage_reg], tid, &shm_data->a);
    LayoutB::StoreShared(ctx.reg_b[stage_reg], tid, &shm_data->b);
    LayoutScale::StoreScaleShm(ctx.reg_packed_scales[stage_reg], tid,
                               &shm_data->scales[0]);
}

template <class Config> struct MultiStagePipeline {
    using Matmul = WPMatmul<Config>;
    using WP = Config::WP;
    using ThreadAccum = Config::ThreadAccum;
    using Shm = ShmBuf<Config>;

    static constexpr bool kSingleBuffer = Shm::kSingleBuffer;
    static constexpr unsigned kStages = Config::kStages;
    static_assert(kStages == 2, "Only supports 2 stages");

    static constexpr unsigned kGroupK = Config::kGroupK;
    static constexpr unsigned kGroupSize = Config::kGroupSize;

    __device__ static void HotLoopScheduler() {
        static constexpr int kSchedGroupId = 0;
        static constexpr unsigned kLoadGlobalA = Shm::LayoutA::kLoadGlobalA;
        static constexpr unsigned kLoadGlobalB = Shm::LayoutB::kLoadGlobalB;
        static constexpr unsigned kLoadScale = Shm::LayoutScale::kLoadScale;

        static constexpr unsigned kInstGlobals =
            kLoadGlobalA + kLoadGlobalB + kLoadScale;
        static constexpr unsigned kInstMFMA =
            (Config::kGroupM / kTile / WP::kPartitionM) *
            (Config::kGroupN / kTile / WP::kPartitionN) *
            (Config::kGroupK / kTile / WP::kPartitionK);

        // The number of instructions required to start the MFMA pipeline
        static constexpr unsigned kInstLoadForFirstPipeline =
            kLoadGlobalA + 1 + 1;
        static constexpr unsigned kInstMFMAPerLoad =
            kInstMFMA / kInstGlobals > 12
                ? 4
                : (kInstMFMA / kInstGlobals > 6 ? 2 : 1);

        // Load A and first B and scales to start the pipeline
        __builtin_amdgcn_sched_group_barrier(0x100, kInstLoadForFirstPipeline,
                                             kSchedGroupId);

        // Ideal schedule order: ds_read + ds_write + buffer_load
        // The heuristics of the instruction scheduler in LLVM fails to schedule
        // the ds_read, so we do not specify the ds_read in the barrier.

        // ds_write
        for (int i = 0; i < kInstGlobals; i++) {
            __builtin_amdgcn_sched_group_barrier(0x200, 1, kSchedGroupId);
            __builtin_amdgcn_sched_group_barrier(0x8, kInstMFMAPerLoad,
                                                 kSchedGroupId);
        }

        // buffer_load
        for (int i = 0; i < kInstGlobals; i++) {
            __builtin_amdgcn_sched_group_barrier(0x20, 1, kSchedGroupId);
            __builtin_amdgcn_sched_group_barrier(0x8, kInstMFMAPerLoad * 2,
                                                 kSchedGroupId);
        }

        static_assert(kInstMFMAPerLoad * kInstGlobals <= kInstMFMA, "");
        __builtin_amdgcn_sched_barrier(0);
    }

    template <class PipelineContext>
    __device__ static void Run(ThreadAccum *__restrict__ acc,
                               typename Shm::Layout *__restrict__ shm_buf,
                               PipelineContext &ctx, unsigned n, unsigned k,
                               unsigned wid, unsigned wtid, unsigned tid) {
        const unsigned k_total = k / kGroupK;

        unsigned curr_stage = 0;
        unsigned next_stage = curr_stage ^ 1;
        LoadGlobal<Config, PipelineContext>(ctx, n, k, curr_stage, wid, tid);
        __builtin_amdgcn_sched_barrier(0x7dfu);

        __builtin_amdgcn_s_waitcnt(0 | (7 << 4) | (15 << 8));
        StoreShm<Config, PipelineContext>(ctx, &shm_buf->data[0], 0, tid);

        if (k_total > 1) {
            ctx.AdvanceGlobalPtr();
            __syncthreads();
            LoadGlobal<Config, PipelineContext>(ctx, n, k, next_stage, wid,
                                                tid);
        }

        typename Matmul::DataA a;
        typename Matmul::DataB b;
        // For single buffer pipeline, two matmuls shared the same shared memory
        // stage.
        Matmul matmul_0(shm_buf, 0, wid, wtid);
        Matmul matmul_1(shm_buf, kSingleBuffer ? 0 : 1, wid, wtid);

        unsigned k_idx = 0;
        for (; k_idx + 3 < k_total; k_idx += 2) {
            __builtin_amdgcn_sched_barrier(0);
            ctx.AdvanceGlobalPtr();
            __syncthreads();

            matmul_0.Prefetch(&a, &b);
            LoadGlobal<Config, PipelineContext>(ctx, n, k, 0, wid, tid);
            // For single buffer pipeline, we separate the two stages with
            // `syncthreads` to avoid read-write hazard.
            // For double buffer pipeline, we overlap the `StoreShm` with the
            // `PipelineCompute` of the last matmul for latency hiding.
            if constexpr (kSingleBuffer) {
                matmul_0.PipelineCompute(&a, &b, acc);
                __syncthreads();
                StoreShm<Config, PipelineContext>(ctx, &shm_buf->data[0], 1,
                                                  tid);
            } else {
                StoreShm<Config, PipelineContext>(ctx, &shm_buf->data[1], 1,
                                                  tid);
                matmul_0.PipelineCompute(&a, &b, acc);
            }

            HotLoopScheduler();

            ctx.AdvanceGlobalPtr();
            __syncthreads();

            matmul_1.Prefetch(&a, &b);
            LoadGlobal<Config, PipelineContext>(ctx, n, k, 1, wid, tid);
            if constexpr (kSingleBuffer) {
                matmul_1.PipelineCompute(&a, &b, acc);
                __syncthreads();
                StoreShm<Config, PipelineContext>(ctx, &shm_buf->data[0], 0,
                                                  tid);
            } else {
                StoreShm<Config, PipelineContext>(ctx, &shm_buf->data[0], 0,
                                                  tid);
                matmul_1.PipelineCompute(&a, &b, acc);
            }

            HotLoopScheduler();
        }

#pragma unroll
        for (int i = 0; i < 3; i++) {
            if (k_idx + i >= k_total) {
                break;
            }
            ctx.AdvanceGlobalPtr();
            __syncthreads();
            Matmul matmul(shm_buf, kSingleBuffer ? 0 : curr_stage, wid, wtid);
            matmul.Prefetch(&a, &b);

            if (k_idx + i + 2 < k_total) {
                LoadGlobal<Config, PipelineContext>(ctx, n, k, curr_stage, wid,
                                                    tid);
            }

            if constexpr (kSingleBuffer) {
                matmul.PipelineCompute(&a, &b, acc);

                if (k_idx + i + 1 < k_total) {
                    __syncthreads();
                    StoreShm<Config, PipelineContext>(ctx, &shm_buf->data[0],
                                                      next_stage, tid);
                }
            } else {
                if (k_idx + i + 1 < k_total) {
                    StoreShm<Config, PipelineContext>(
                        ctx, &shm_buf->data[next_stage], next_stage, tid);
                }

                matmul.PipelineCompute(&a, &b, acc);
            }

            curr_stage = next_stage;
            next_stage = curr_stage ^ 1;
        }
    }
};

// 4-bit quantization Matrix multiplication.
// C = A * B
// A (m * k) is fp16, B / zeros are 4-bit quantized, scales are fp16
// B (k * n) is stored in the Petit layout, where each tile has 8 subtiles
// of 16x16 subtiles. See DequantizePetit for more details of the layout.
//
// C is stored in the row-major order.
template <class Config>
__launch_bounds__(Config::kThreads) __global__
    void GemmFp4Fp16KernelGrid(uint4 *__restrict__ C,
                               const uint4 *__restrict__ A,
                               const uint4 *__restrict__ B,
                               const uint4 *__restrict__ scales,
                               const float *__restrict__ global_scale_ptr,
                               const unsigned m, const unsigned n,
                               const unsigned k) {
    static constexpr unsigned kGroupM = Config::kGroupM;
    static constexpr unsigned kGroupN = Config::kGroupN;
    static constexpr unsigned kGroupNInts = kGroupN / kPackFactor;
    static constexpr unsigned kLayoutM = Config::kLayoutM;
    static constexpr unsigned kVecSize = Config::kVecSize;
    static constexpr unsigned kScaleVecSize = Config::kScaleVecSize;
    static constexpr unsigned kGroupSize = Config::kGroupSize;
    static constexpr unsigned kResultVecSize =
        sizeof(uint4) / sizeof(unsigned short);

    const unsigned tid = threadIdx.x, gid_m = blockIdx.x, gid_n = blockIdx.y;
    const unsigned wid = tid / kWarpSize, wtid = tid % kWarpSize;

    __shared__ typename ShmBuf<Config>::Layout shm_buf;
    using ThreadAccum = Config::ThreadAccum;
    using Pipeline = MultiStagePipeline<Config>;
    using ArchMma =
        MmaSelector<typename Config::ElementA, Config::kElementBTypeId,
                                Config::kHighPrecision>;
    float global_scale =
        *global_scale_ptr * ArchMma::UDQ::GlobalScaleFactor();

    [[assume(tid < Config::kThreads)]];

    ThreadAccum acc;
    for (int i = 0; i < sizeof(acc) / sizeof(float4); i++) {
        reinterpret_cast<float4 *>(&acc)[i] = {0.0f, 0.0f, 0.0f, 0.0f};
    };

    const uint4 *a_ptr = A + gid_m * k * (kGroupM / kVecSize);
    const uint4 *b_ptr = B + gid_n * (kLayoutM * kGroupNInts / kQuantVecSize);
    const uint4 *scale_ptr =
        scales + gid_n * (kLayoutM / kGroupSize) * kGroupN / kScaleVecSize;

    unsigned tile_m = std::min(kGroupM, m - gid_m * kGroupM);
    PipelineContext<Config, Pipeline::kStages> ctx;
    ctx.Initialize(a_ptr, b_ptr, scale_ptr, tile_m, n, k);

    Pipeline::Run(&acc, &shm_buf, ctx, n, k, wid, wtid, tid);

    gpu::quantization::BlockReduce<typename ShmBuf<Config>::Layout, Config>(
        &shm_buf, acc, wid, wtid);

    uint4 *c_ptr = C + gid_m * n * (kGroupM / kResultVecSize) +
                   gid_n * (kGroupN / kResultVecSize);

    WriteResult<Config, true>(global_scale, &shm_buf, acc, c_ptr, tid, tile_m,
                              n);
}

template <MatmulMfmaType kMfmaType> struct MfmaElementTypes;

template <> struct MfmaElementTypes<MatmulMfmaType::kMatmulMfmaTypeFp16> {
    using ElementA = half;
};

template <> struct MfmaElementTypes<MatmulMfmaType::kMatmulMfmaTypeBf16> {
    using ElementA = __hip_bfloat16;
};

template <> struct MfmaElementTypes<MatmulMfmaType::kMatmulMfmaTypeFp8> {
    using ElementA = uchar;
};

template <SolutionId id> struct ConfigSelector {
    static constexpr DataType kElementBTypeId =
        id.element_b == kMatmulTypeBMxFp4 ? kDataTypeMxFp4e2m1
                                          : kDataTypeFp4e2m1;
    static constexpr unsigned kGroupSize =
        kElementBTypeId == kDataTypeMxFp4e2m1 ? 32 : 16;
    static constexpr unsigned kNumTilesM = id.tile_m;
    static constexpr unsigned kNumTilesN = id.tile_n;
    static constexpr unsigned kNumTilesK = id.tile_k * 4;
    static constexpr unsigned kPipelineStages = 2;
    static constexpr unsigned kPartitionM = id.warp_partition_m;
    static constexpr unsigned kPartitionN = id.warp_partition_n;
    static constexpr unsigned kPartitionK = id.warp_partition_k;
    static constexpr MatmulMfmaType kMfmaType = id.mfma_type;
    static constexpr bool kHighPrecision =
        id.features & kMatmulFeatures_HighPrecision;

    using ElementA = MfmaElementTypes<kMfmaType>::ElementA;
    using TS = TileShape<ElementA, kElementBTypeId, kGroupSize, kNumTilesM,
                         kNumTilesN, kNumTilesK>;
    using WP = WarpPartition<kPartitionM, kPartitionN, kPartitionK>;
    using Config = GEMMFp4Fp16Config<TS, WP, kPipelineStages, kHighPrecision>;
    using ArchMma =
        MmaSelector<typename Config::ElementA, kElementBTypeId,
                    kHighPrecision>;

    static constexpr unsigned kNumWarps = WP::kNumWarps;

    static int Invoke(unsigned *c, const unsigned *a, const unsigned *b,
                      const unsigned *scales, const float *global_scale,
                      const unsigned m, const unsigned n, const unsigned k,
                      hipStream_t stream) {

        uint4 *c4 = reinterpret_cast<uint4 *>(c);
        const uint4 *a4 = reinterpret_cast<const uint4 *>(a);
        if (n % Config::kGroupN != 0 || k % Config::kGroupK != 0) {
            return kErrorProblemShape;
        }

        dim3 blockDim(kNumWarps * kWarpSize);
        dim3 gridDim(tal::CeilingDiv<unsigned>(m, kTile * kNumTilesM),
                     tal::CeilingDiv<unsigned>(n, kTile * kNumTilesN));
        GemmFp4Fp16KernelGrid<Config><<<gridDim, blockDim, 0, stream>>>(
            c4, a4, reinterpret_cast<const uint4 *>(b),
            reinterpret_cast<const uint4 *>(scales), global_scale, m, n, k);
        return 0;
    }
};

} // namespace causalflow::petit::rocm::quantization::fp4
