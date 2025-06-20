#include "algo_variants.h"
#include "gemm/rocm/quantization/gemm.h"
#include "gemm/rocm/quantization/types.h"

#include <algorithm>
#include <array>

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

    for (const auto &sol : detail::kAvailableSolutions) {
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
    static constexpr unsigned kAvailableSols =
        sizeof(detail::kAvailableSolutions) / sizeof(SolutionId);
    std::array<SolutionId, kAvailableSols> sols;
    auto is_available = [&](const SolutionId &sol) {
        unsigned group_n = sol.tile_n * kTile, group_k = sol.tile_k * 64;
        bool is_high_precision =
            (sol.features & MatmulFeatures::kMatmulFeatures_HighPrecision) != 0;
        return (hints.require_high_precision ^ is_high_precision) &&
               ((sol.mfma_type == MatmulMfmaType::kMatmulMfmaTypeFp16 &&
                 hints.a_type == DataType::kDataTypeFp16) ||
                (sol.mfma_type == MatmulMfmaType::kMatmulMfmaTypeBf16 &&
                 hints.a_type == DataType::kDataTypeBf16)) &&
               n % group_n == 0 && k % group_k == 0;
    };
    auto end = std::copy_if(detail::kAvailableSolutions,
                            detail::kAvailableSolutions + kAvailableSols,
                            sols.begin(), is_available);
    if (end == sols.begin()) {
        return -1;
    }

    // The function does not define a weak ordering, it may just crash with sort
    // / qsort
    std::stable_sort(sols.begin(), end,
                     [&](const SolutionId &a, const SolutionId &b) {
                         if (a.tile_k != b.tile_k) {
                             return a.tile_k > b.tile_k;
                         } else if (a.pipeline != b.pipeline) {
                             return a.pipeline > b.pipeline;
                         } else {
                             unsigned m_a = m / (a.tile_m * kTile),
                                      m_b = n / (b.tile_m * kTile);
                             return m_b > m_a;
                         }
                         return a.Repr() < b.Repr();
                     });
    return sols[0].Repr();
}

} // namespace causalflow::petit::rocm::quantization::fp4
