#include "gemm/rocm/quantization/gemm.h"

#include <functional>
#include <vector>

#include <fmt/core.h>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <gflags/gflags.h>

DEFINE_string(source_list_cmake, "source_list.cmake",
              "CMake fragments of the source files");
DEFINE_string(source_list, "solutions.inl", "Lsits of all solutions");
DEFINE_string(output_dir, "", "Output directory of individual kernel files");

namespace causalflow::petit::rocm::quantization::fp4 {

struct TileShapeAndWarpPartition {
    unsigned tile_m : 8;
    unsigned tile_n : 8;
    unsigned tile_k : 8;
    unsigned warp_partition_m : 4;
    unsigned warp_partition_n : 4;
    unsigned warp_partition_k : 4;
};

using TSWP = TileShapeAndWarpPartition;

static constexpr unsigned kTile = 16;
static constexpr unsigned kLayoutM = 64;
static constexpr unsigned kLayoutN = 32;

// TileShapeAndWarpPartition, order by (n, k, m)
static constexpr TSWP kTSWP[] = {
    {4, 2, 8, 4, 1, 1},   {1, 2, 16, 1, 1, 4},  {2, 2, 16, 2, 1, 2},
    {2, 2, 16, 1, 1, 4},  {4, 2, 16, 2, 1, 2},  {1, 2, 32, 1, 1, 4},
    {2, 2, 32, 2, 1, 2},

    {1, 4, 8, 1, 2, 2},   {2, 4, 8, 2, 2, 1},   {4, 4, 8, 2, 2, 1},
    {6, 4, 8, 2, 2, 1},   {8, 4, 8, 2, 2, 1},   {10, 4, 8, 2, 2, 1},
    {1, 4, 16, 1, 2, 2},  {2, 4, 16, 2, 2, 1},  {4, 4, 16, 2, 2, 1},
    {1, 4, 32, 1, 2, 2},  {2, 4, 32, 2, 2, 1},

    {4, 6, 8, 2, 1, 2},   {6, 6, 8, 2, 1, 2},   {1, 8, 4, 1, 4, 1},
    {8, 8, 4, 2, 2, 1},   {12, 8, 4, 2, 2, 1},  {14, 8, 4, 2, 2, 1},
    {16, 8, 4, 2, 2, 1},  {2, 8, 8, 2, 2, 1},   {4, 8, 8, 2, 2, 1},
    {5, 8, 8, 1, 2, 2},   {10, 8, 4, 2, 2, 1},

    {8, 12, 4, 2, 2, 1},  {10, 12, 4, 2, 2, 1}, {12, 12, 4, 2, 2, 1},
    {14, 12, 4, 2, 2, 1}, {16, 12, 4, 2, 2, 1}, {8, 16, 4, 2, 2, 1},
    {10, 16, 4, 2, 2, 1}, {12, 16, 4, 2, 2, 1}, {14, 16, 4, 2, 2, 1},
    {16, 16, 4, 2, 2, 1},
};

static std::vector<SolutionId> FromTSWPList() {
    std::vector<SolutionId> solutions;
    for (const auto &tswp : kTSWP) {
        if (tswp.tile_k * kTile % kLayoutM != 0 ||
            tswp.tile_n * kTile % kLayoutN != 0) {
            throw std::runtime_error(fmt::format(
                "Tile shape {}x{} is not aligned with layout {}x{}",
                tswp.tile_m * kTile, tswp.tile_n * kTile, kLayoutM, kLayoutN));
        }
        solutions.push_back(SolutionId::MultiStage(
            MatmulFeatures::kMatmulFeatures_Grid,
            MatmulElementB::kMatmulTypeBFp4,
            MatmulMfmaType::kMatmulMfmaTypeFp16, tswp.tile_m, tswp.tile_n,
            tswp.tile_k, MatmulWarpPartition::kMatmulWarpPartition_NK,
            tswp.warp_partition_m, tswp.warp_partition_n,
            tswp.warp_partition_k));
    }
    return solutions;
}

class SolutionListBuilder {
  public:
    using Call = std::function<SolutionId(SolutionId)>;
    SolutionListBuilder &Fork(std::initializer_list<Call> calls) {
        std::vector<SolutionId> n;
        for (const auto &call : calls) {
            for (const auto &s : solutions_) {
                n.push_back(call(s));
            }
        }
        solutions_ = std::move(n);
        return *this;
    }

    const std::vector<SolutionId> &Get() const { return solutions_; }

    std::vector<SolutionId> solutions_;
};
} // namespace causalflow::petit::rocm::quantization::fp4

using namespace causalflow::petit::rocm::quantization;

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    fp4::SolutionListBuilder builder;
    builder.solutions_ = fp4::FromTSWPList();
    builder
        .Fork({[](SolutionId s) {
                   s.mfma_type = MatmulMfmaType::kMatmulMfmaTypeFp16;
                   return s;
               },
               [](SolutionId s) {
                   s.mfma_type = MatmulMfmaType::kMatmulMfmaTypeBf16;
                   return s;
               }})
        .Fork({[](SolutionId s) {
                   s.features = MatmulFeatures::kMatmulFeatures_Grid;
                   return s;
               },
               [](SolutionId s) {
                   s.features = static_cast<MatmulFeatures>(
                       MatmulFeatures::kMatmulFeatures_Grid |
                       MatmulFeatures::kMatmulFeatures_HighPrecision);
                   return s;
               }});

    fmt::ostream src_cmake = fmt::output_file(FLAGS_source_list_cmake);
    fmt::ostream src_inl = fmt::output_file(FLAGS_source_list);
    src_cmake.print("SET(GEMM_FP4_FP16_SRCS \n");
    for (const auto &s : builder.Get()) {
        src_cmake.print("    gemm_fp4_fp16_{:08x}.cu\n", s.Repr());
        src_inl.print("PETIT_KERNEL_IMPL({:08x})\n", s.Repr());
        fmt::ostream kernel = fmt::output_file(fmt::format(
            "{}/gemm_fp4_fp16_{:08x}.cu", FLAGS_output_dir, s.Repr()));
        kernel.print("PETIT_KERNEL_IMPL({:08x})\n", s.Repr());
    }
    src_cmake.print(")\n");
    return 0;
}
