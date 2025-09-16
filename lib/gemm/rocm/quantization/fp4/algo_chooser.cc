#include "gemm/rocm/quantization/gemm.h"
#include "gemm/rocm/quantization/types.h"
#include "gemm_fp4.h"

#include <optional>

namespace causalflow::petit::rocm::quantization::fp4 {

using causalflow::petit::rocm::quantization::DataType;
using causalflow::petit::rocm::quantization::MatmulElementB;
using causalflow::petit::rocm::quantization::MatmulMfmaType;
using causalflow::petit::rocm::quantization::SolutionId;

int GemmGetSolutions(const PetitSolutionHints &hints, unsigned m, unsigned n,
                     unsigned k, SolutionId *sols, unsigned *n_sols) {
    static constexpr unsigned kTile = 16;
    static constexpr unsigned kSolTileK = kTile * 4;

    unsigned sol_count = 0;
    if (hints.b_type != DataType::kDataTypeFp4e2m1) {
        return -1;
    }

    for (const auto &entries : SolutionMap::GetDispatchEntries()) {
        const auto &sol = reinterpret_cast<const SolutionId &>(entries.first);
        if (sol.element_b != MatmulElementB::kMatmulTypeBFp4) {
            continue;
        }

        bool is_high_precision =
            (sol.features & MatmulFeatures::kMatmulFeatures_HighPrecision) != 0;
        if (hints.require_high_precision ^ is_high_precision) {
            continue;
        }

        if ((sol.mfma_type == MatmulMfmaType::kMatmulMfmaTypeFp16 &&
             hints.a_type == DataType::kDataTypeFp16) ||
            (sol.mfma_type == MatmulMfmaType::kMatmulMfmaTypeBf16 &&
             hints.a_type == DataType::kDataTypeBf16)) {
            unsigned group_n = sol.tile_n * kTile,
                     group_k = sol.tile_k * kSolTileK;
            if (n % group_n == 0 && k % group_k == 0) {
                if (sols && sol_count < *n_sols) {
                    sols[sol_count] = sol;
                }
                sol_count++;
            }
        }
    }
    *n_sols = sol_count;
    return 0;
}

unsigned long ChooseDefaultFp4Fp16Solution(unsigned m, unsigned n, unsigned k,
                                           const PetitSolutionHints &hints) {
    static constexpr unsigned kTile = 16;

    auto is_available = [&](const SolutionId &sol) {
        unsigned group_n = sol.tile_n * kTile, group_k = sol.tile_k * 64;
        bool not_high_precision =
            (sol.features & MatmulFeatures::kMatmulFeatures_HighPrecision) == 0;
        return (hints.require_high_precision ^ not_high_precision) &&
               ((sol.mfma_type == MatmulMfmaType::kMatmulMfmaTypeFp16 &&
                 hints.a_type == DataType::kDataTypeFp16) ||
                (sol.mfma_type == MatmulMfmaType::kMatmulMfmaTypeBf16 &&
                 hints.a_type == DataType::kDataTypeBf16)) &&
               n % group_n == 0 && k % group_k == 0;
    };

    auto is_better = [&](const SolutionId &a, const SolutionId &b) {
        const unsigned tile_m_a = a.tile_m * kTile;
        const unsigned tile_m_b = b.tile_m * kTile;
        bool is_small_m = m <= 64;

        if (is_small_m) {
            if (a.tile_m != b.tile_m) {
                unsigned delta_a = std::abs(static_cast<int>(m % tile_m_a) -
                                            static_cast<int>(tile_m_a / 2));
                unsigned delta_b = std::abs(static_cast<int>(m % tile_m_b) -
                                            static_cast<int>(tile_m_b / 2));
                return delta_a < delta_b;
            } else if (a.warp_partition_k != b.warp_partition_k) {
                return a.warp_partition_k > b.warp_partition_k;
            } else if (a.tile_n != b.tile_n) {
                return a.tile_n < b.tile_n;
            }
        }

        if (a.tile_m + a.tile_n != b.tile_m + b.tile_n) {
            return a.tile_m + a.tile_n > b.tile_m + b.tile_n;
        } else if (a.tile_m != b.tile_m) {
            return a.tile_m > b.tile_m;
        } else if (a.tile_k != b.tile_k) {
            return a.tile_k > b.tile_k;
        }

        return a.Repr() > b.Repr();
    };

    std::optional<SolutionId> best_sol;
    for (const auto &entries : SolutionMap::GetDispatchEntries()) {
        const auto &sol = reinterpret_cast<const SolutionId &>(entries.first);
        if (!is_available(sol)) {
            continue;
        } else if (!best_sol) {
            best_sol = sol;
        } else if (is_better(sol, best_sol.value())) {
            best_sol = sol;
        }
    }
    return best_sol.value().Repr();
}

} // namespace causalflow::petit::rocm::quantization::fp4
