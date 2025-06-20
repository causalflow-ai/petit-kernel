#pragma once

#include "gemm/device_types.h"

namespace causalflow::petit::gpu::quantization {

template <class ShmBufLayout, class Config>
__device__ static void BlockReduce(ShmBufLayout *__restrict__ shm_buf,
                                   typename Config::ThreadAccum &acc,
                                   unsigned wid, unsigned wtid) {
    static constexpr unsigned kNumTileM = Config::kWarpTilesM;
    static constexpr unsigned kWarpsM = Config::WP::kPartitionM;
    static constexpr unsigned kWarpsN = Config::WP::kPartitionN;
    static constexpr unsigned kAccRows = Config::WP::kPartitionK;

    if constexpr (kAccRows <= 1) {
        return;
    }

    auto shm_red = &shm_buf->red;
    const unsigned acc_row = Config::WP::AccRowK(wid),
                   acc_col = Config::WP::WarpColN(wid),
                   warp_row = Config::WP::WarpRowM(wid);

    for (int m = 0; m < kNumTileM; m++) {
        for (int red_off = kAccRows / 2; red_off; red_off /= 2) {
            if (red_off <= acc_row && acc_row < 2 * red_off) {
                for (int j = 0; j < Config::kThreadAccumTileNRegs; j++) {
                    float4 *acc_wr =
                        &shm_red
                             ->acc[j][(acc_row - red_off) * kWarpsN * kWarpsM +
                                      warp_row * kWarpsN + acc_col][wtid];
                    if (red_off < kAccRows / 2) {
                        float4 rd =
                            shm_red->acc[j][acc_row * kWarpsN * kWarpsM +
                                            warp_row * kWarpsN + acc_col][wtid];
                        float *f = reinterpret_cast<float *>(&acc[m][j]);
                        for (int k = 0; k < 4; k++) {
                            f[k] += reinterpret_cast<const float *>(&rd)[k] +
                                    reinterpret_cast<const float *>(acc_wr)[k];
                        }
                    }
                    *acc_wr = acc[m][j];
                }
            }
            __syncthreads();
        }

        if (acc_row == 0) {
            for (int j = 0; j < Config::kThreadAccumTileNRegs; j++) {
                float4 rd = shm_red->acc[j][warp_row * kWarpsN + acc_col][wtid];
                float *f = reinterpret_cast<float *>(&acc[m][j]);
                for (int k = 0; k < 4; k++) {
                    f[k] += reinterpret_cast<const float *>(&rd)[k];
                }
            }
        }
        __syncthreads();
    }
}

} // namespace causalflow::petit::gpu::quantization