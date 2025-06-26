#include "algo_variants.h"
#include "gemm/gpu/quantization/reduce.cuh"
#include "gemm/rocm/quantization/kernel_config.cuh"
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

template <class TileShape_, class WarpPartition_, unsigned kPipelineStages,
          bool kHighPrecision_>
struct GEMMFp4Fp16Config {
    // kLayoutM is dimension K
    static constexpr unsigned kLayoutM = 128;
    static constexpr unsigned kLayoutN = 16;

    using TS = TileShape_;
    using WP = WarpPartition_;
    using ElementA = typename TS::ElementA;
    static constexpr bool kHighPrecision = kHighPrecision_;

    static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(ElementA);
    static constexpr unsigned kScaleVecSize =
        sizeof(uint4) / sizeof(unsigned char);

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

    // The number of tiles in the m dimension that a warp is responsible to.
    static constexpr unsigned kWarpTileM = kNumTileM / WP::kPartitionM;

    static constexpr unsigned kMmaM = 16;
    static constexpr unsigned kMmaN = 16;
    static constexpr unsigned kMmaTileK = kVecSize * kWarpSize / kMmaM;
    static constexpr unsigned kStages = kPipelineStages;

    static constexpr bool kUseZeroPoints = false;
    static constexpr bool kZpInShm = false;

    static constexpr unsigned kThreadAccumTileNRegs =
        kNumTileN / WP::kPartitionN / (kMmaN / kTile);
    using ThreadAccum = float4[kWarpTileM][kThreadAccumTileNRegs];

    static_assert(kThreadAccumTileNRegs > 0, "");
    static_assert(kGroupK % kLayoutM == 0,
                  "kGroupK must be a multiple of kLayoutM");
    static_assert(kNumTileN % WP::kPartitionN == 0 &&
                      kNumTileK % WP::kPartitionK == 0,
                  "The number of tiles must be divisible by the partition "
                  "dimensions");

    __device__ static inline unsigned XorShmLayout(unsigned row, unsigned col) {
        static constexpr unsigned kBatchStride = kMmaTileK / kVecSize;
        static constexpr unsigned kBatches = kWarpSize / kMmaM;
        return row * kGroupK / kVecSize + col ^
               (row % kBatchStride +
                row / kBatchStride * kBatchStride * kBatches);
    }

    __device__ static inline unsigned WriteShmCoordA(unsigned idx) {
        unsigned row = idx / (kGroupK / kVecSize),
                 col = idx % (kGroupK / kVecSize);
        return XorShmLayout(row, col);
    }

    __device__ static inline unsigned ReadShmCoordA(unsigned tile_idx_m,
                                                    unsigned tile_idx_k,
                                                    unsigned batch_id,
                                                    unsigned wtid) {
        unsigned row_in_tile = wtid % kMmaM, col_in_tile = wtid / kMmaM;
        unsigned row = tile_idx_m * kTile + row_in_tile;
        unsigned col = (kLayoutM / kVecSize) * tile_idx_k +
                       col_in_tile * (kMmaTileK / kVecSize) + batch_id;
        return XorShmLayout(row, col);
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
         typename ShmBuf<Config>::Layout *__restrict__ shm_buf, unsigned stage,
         unsigned tid) {
    using LayoutA = MatrixALayout<Config>;
    using LayoutB = MatrixBLayout<Config>;
    using LayoutScale = ScaleLayout<Config>;

    LayoutA::StoreShared(ctx.reg_a[stage], tid, &shm_buf->data[stage].a);
    LayoutB::StoreShared(ctx.reg_b[stage], tid, &shm_buf->data[stage].b);
    LayoutScale::StoreScaleShm(ctx.reg_packed_scales[stage], tid,
                               &shm_buf->data[stage].scales[0]);
}

template <class Config> struct SingleStagePipeline {
    using LayoutA = MatrixALayout<Config>;
    using LayoutB = MatrixBLayout<Config>;
    using Matmul = WPMatmul<Config>;
    using ThreadAccum = Config::ThreadAccum;

    static constexpr unsigned kStages = 1;
    static constexpr unsigned kGroupK = Config::kGroupK;
    static constexpr unsigned kGroupSize = Config::kGroupSize;

    template <class PipelineContext>
    __device__ static void
    Run(ThreadAccum *__restrict__ acc,
        typename ShmBuf<Config>::Layout *__restrict__ shm_buf,
        PipelineContext &ctx, unsigned n, unsigned k, unsigned wid,
        unsigned wtid, unsigned tid) {
        const unsigned k_total = k / kGroupK;

        unsigned curr_stage = 0;
        for (unsigned k_idx = 0; k_idx < k_total; k_idx++) {
            LoadGlobal<Config, PipelineContext>(ctx, n, k, curr_stage, wid,
                                                tid);
            StoreShm<Config, PipelineContext>(ctx, shm_buf, curr_stage, tid);
            ctx.AdvanceGlobalPtr();
            __syncthreads();

            Matmul::Compute(shm_buf, acc, curr_stage, wid, wtid);
            __syncthreads();
        }
    }
};

template <class Config> struct MultiStagePipeline {
    using Matmul = WPMatmul<Config>;
    using ThreadAccum = Config::ThreadAccum;

    static constexpr unsigned kStages = 2;
    static constexpr unsigned kGroupK = Config::kGroupK;
    static constexpr unsigned kGroupSize = Config::kGroupSize;

    template <class PipelineContext>
    __device__ static void
    Run(ThreadAccum *__restrict__ acc,
        typename ShmBuf<Config>::Layout *__restrict__ shm_buf,
        PipelineContext &ctx, unsigned n, unsigned k, unsigned wid,
        unsigned wtid, unsigned tid) {
        const unsigned k_total = k / kGroupK;

        unsigned curr_stage = 0;
        unsigned next_stage = curr_stage ^ 1;
        LoadGlobal<Config, PipelineContext>(ctx, n, k, curr_stage, wid, tid);

        if (k_total > 1) {
            ctx.AdvanceGlobalPtr();
            __syncthreads();
            LoadGlobal<Config, PipelineContext>(ctx, n, k, next_stage, wid,
                                                tid);
        }
        StoreShm<Config, PipelineContext>(ctx, shm_buf, curr_stage, tid);

        unsigned k_idx = 0;
        for (; k_idx + 3 < k_total; k_idx += 2) {
#pragma unroll
            for (unsigned curr_stage = 0; curr_stage < 2; curr_stage++) {
                unsigned next_stage = curr_stage ^ 1;
                ctx.AdvanceGlobalPtr();
                __syncthreads();

                LoadGlobal<Config, PipelineContext>(ctx, n, k, curr_stage, wid,
                                                    tid);

                StoreShm<Config, PipelineContext>(ctx, shm_buf, next_stage,
                                                  tid);
                Matmul::Compute(shm_buf, acc, curr_stage, wid, wtid);
            }
        }

#pragma unroll
        for (int i = 0; i < 3; i++) {
            if (k_idx + i >= k_total) {
                break;
            }
            ctx.AdvanceGlobalPtr();
            __syncthreads();

            if (k_idx + i + 2 < k_total) {
                LoadGlobal<Config, PipelineContext>(ctx, n, k, curr_stage, wid,
                                                    tid);
            }

            if (k_idx + i + 1 < k_total) {
                StoreShm<Config, PipelineContext>(ctx, shm_buf, next_stage,
                                                  tid);
            }

            Matmul::Compute(shm_buf, acc, curr_stage, wid, wtid);
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
                               float global_scale, const unsigned m,
                               const unsigned n, const unsigned k) {
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
    using Pipeline =
        std::conditional_t<Config::kStages == 1, SingleStagePipeline<Config>,
                           MultiStagePipeline<Config>>;

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
    static constexpr unsigned kGroupSize = 16;
    static constexpr unsigned kNumTilesM = id.tile_m;
    static constexpr unsigned kNumTilesN = id.tile_n;
    static constexpr unsigned kNumTilesK = id.tile_k * 4;
    static constexpr unsigned kPipelineStages = id.pipeline + 1;
    static constexpr unsigned kPartitionM = id.warp_partition_m;
    static constexpr unsigned kPartitionN = id.warp_partition_n;
    static constexpr unsigned kPartitionK = id.warp_partition_k;
    static constexpr MatmulMfmaType kMfmaType = id.mfma_type;
    static constexpr bool kHighPrecision =
        id.features & kMatmulFeatures_HighPrecision;

    using ElementA = MfmaElementTypes<kMfmaType>::ElementA;
    using TS =
        TileShape<ElementA, kGroupSize, kNumTilesM, kNumTilesN, kNumTilesK>;
    using WP = WarpPartition<kPartitionM, kPartitionN, kPartitionK>;
    using Config = GEMMFp4Fp16Config<TS, WP, kPipelineStages, kHighPrecision>;
    using ArchMma = MmaSelector<typename Config::ElementA, kHighPrecision>;
    using DQ = ArchMma::DQ;
    using DS = ArchMma::DS;

    static constexpr unsigned kNumWarps = WP::kNumWarps;

    static int Invoke(unsigned *c, const unsigned *a, const unsigned *b,
                      const unsigned *scales, float global_scale,
                      const unsigned m, const unsigned n, const unsigned k,
                      hipStream_t stream) {
        global_scale *= DS::GlobalScaleFactor();

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

class Dispatcher {
  public:
    using Solutions = decltype(fp4::detail::kAvailableSolutions);
    static constexpr size_t kN = sizeof(Solutions) / sizeof(SolutionId);
    using Call = std::function<int(
        unsigned *, const unsigned *, const unsigned *, const unsigned *, float,
        const unsigned, const unsigned, const unsigned, hipStream_t)>;
    static Dispatcher &GetInstance() {
        static Dispatcher instance;
        return instance;
    }

    int Dispatch(unsigned *c, const unsigned *a, const unsigned *b,
                 const unsigned *scales, float global_scale, const unsigned m,
                 const unsigned n, const unsigned k, unsigned long solution_id,
                 hipStream_t stream) {
        const auto it = solution_id_to_call_.find(solution_id);
        if (it == solution_id_to_call_.end()) {
            return kErrorKernelShape;
        }
        return it->second(c, a, b, scales, global_scale, m, n, k, stream);
    }

  protected:
    std::unordered_map<unsigned long, Call> solution_id_to_call_;
    template <SolutionId... sols>
    static std::unordered_map<unsigned long, Call> ToDispatchEntries() {
        return {
            {sols.Repr(),
             [](unsigned *c, const unsigned *a, const unsigned *b,
                const unsigned *scales, float global_scale, const unsigned m,
                const unsigned n, const unsigned k, hipStream_t stream) {
                 using Trait = ConfigSelector<sols>;
                 return Trait::Invoke(c, a, b, scales, global_scale, m, n, k,
                                      stream);
             }}...};
    }

    template <auto Array, std::size_t... Is>
    auto MakeDispatchMapWithIndices(std::index_sequence<Is...>) {
        return ToDispatchEntries<Array[Is]...>();
    }

    Dispatcher()
        : solution_id_to_call_(
              MakeDispatchMapWithIndices<fp4::detail::kAvailableSolutions>(
                  std::make_index_sequence<kN>{})) {}
};

int GemmFp4Fp16Grid(unsigned *c, const unsigned *a, const unsigned *b,
                    const unsigned *scales, float global_scale,
                    const unsigned m, const unsigned n, const unsigned k,
                    const PetitSolutionHints &hints, unsigned long solution_id,
                    hipStream_t stream) {
    if (m == 0 || n == 0 || k == 0) {
        return 0;
    }

    if (solution_id == -1) {
        solution_id = ChooseDefaultFp4Fp16Solution(m, n, k, hints);
    }

    if (solution_id == -1) {
        return kErrorProblemShape;
    }

    return Dispatcher::GetInstance().Dispatch(c, a, b, scales, global_scale, m,
                                              n, k, solution_id, stream);
}

} // namespace causalflow::petit::rocm::quantization::fp4