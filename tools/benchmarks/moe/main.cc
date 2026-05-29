#include "moe/rocm/fused_moe.h"
#include "moe/rocm/quantization.cuh"
#include "tests/fp8_sampler.h"
#include "utils/hip_helper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <execution>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#include <fmt/core.h>
#include <gflags/gflags.h>
#include <hip/hip_runtime.h>

DEFINE_string(backend, "fused_moe_rocm",
              "Backend to use for Fused MoE. Only 'fused_moe_rocm' is "
              "supported.");
DEFINE_int32(tokens, 256, "Number of tokens.");
DEFINE_int32(dim, 4096, "Model dimension.");
DEFINE_int32(inter_dim, 1024, "Intermediate (FFN) dimension.");
DEFINE_int32(experts, 8, "Number of experts.");
DEFINE_int32(topk, 2, "Top-k routes per token.");
DEFINE_int32(warmup, 10, "Number of warmup iterations.");
DEFINE_int32(repeat, 100, "Number of benchmark iterations.");
DEFINE_int32(seed, 42, "Random seed.");
DEFINE_bool(persistent, false,
            "Enable persistent scheduling for fused MoE kernel launch.");
DEFINE_int32(num_persistent_tgs, 0,
             "Total persistent workgroups across all split-k tiles. "
             "0 disables persistent scheduling unless --persistent is set.");
DEFINE_int32(persistent_tgs_per_cu, 2,
             "When --persistent=true and --num_persistent_tgs=0, launch "
             "multiProcessorCount * persistent_tgs_per_cu workgroups.");

static bool ValidateBackend(const char *, const std::string &value) {
    return value == "fused_moe_rocm";
}

static void RegisterValidators() {
    gflags::RegisterFlagValidator(&FLAGS_backend, &ValidateBackend);
}

namespace moe = causalflow::petit::rocm::moe;
using causalflow::CheckHIPStatus;

namespace {

constexpr unsigned kBlockN = 16;
constexpr unsigned kBlockK = 32;
constexpr unsigned kPackK = 16;
constexpr unsigned kKk = kBlockK / kPackK;
constexpr unsigned kSortedTokenPadding = 32;
constexpr unsigned kScaleGroup = 128;

template <class T> constexpr T CeilingDiv(T x, T y) { return (x + y - 1) / y; }

static inline unsigned FloatAsBits(float x) {
    union {
        float f;
        unsigned u;
    } v{x};
    return v.u;
}

static inline uint64_t MixU64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct FixedU32Rng {
    using result_type = uint32_t;

    uint32_t value;

    static constexpr result_type min() { return 0; }

    static constexpr result_type max() {
        return std::numeric_limits<result_type>::max();
    }

    result_type operator()() { return value; }
};

template <class T, class Sampler>
static void FillWeightsParallel(std::vector<T> *data, const Sampler &sampler,
                                uint64_t seed, uint64_t stream) {
    const size_t count = data->size();
    if (count == 0) {
        return;
    }
    T *const base = data->data();
    std::for_each(
        std::execution::par, data->begin(), data->end(), [&](T &value) {
            const size_t idx = static_cast<size_t>(&value - base);
            FixedU32Rng rng{
                static_cast<uint32_t>(MixU64(seed ^ (stream << 32) ^ idx))};
            value = sampler(rng);
        });
}

template <class Generator4>
static void FillFp8InputsParallel4(std::vector<unsigned char> *data,
                                   const Generator4 &generator4) {
    const size_t count = data->size();
    if (count == 0) {
        return;
    }
    if ((count % 4) != 0) {
        throw std::runtime_error(
            "FillFp8InputsParallel4 expects size % 4 == 0");
    }

    auto *const packed_ptr = reinterpret_cast<unsigned *>(data->data());
    std::span<unsigned> packed(packed_ptr, count / 4);
    unsigned *const base = packed.data();
    std::for_each(std::execution::par, packed.begin(), packed.end(),
                  [&](unsigned &slot) {
                      const size_t group = static_cast<size_t>(&slot - base);
                      slot = generator4(group * 4);
                  });
}

struct HostInputs {
    std::vector<unsigned char> act_fp8;
    std::vector<unsigned char> w13_fp8;
    std::vector<unsigned char> w2_fp8;

    std::vector<unsigned> sorted_token_ids;
    std::vector<float> sorted_weights;
    std::vector<unsigned> sorted_expert_ids;
    std::vector<unsigned> num_valid_ids;

    std::vector<float> scale_act;
    std::vector<float> scale_fc1;
    std::vector<float> scale_fc2;
    std::vector<unsigned> fc2_scale_aiter_bits;

    unsigned num_valid_token_ids = 0;
    unsigned num_valid_m_blocks = 0;
};

struct DeviceBuffers {
    unsigned char *act_fp8 = nullptr;
    unsigned char *w1_q_shuffled = nullptr;
    unsigned char *w2_q_shuffled = nullptr;
    unsigned short *out_bf16 = nullptr;

    unsigned *sorted_token_ids = nullptr;
    float *sorted_weights = nullptr;
    unsigned *sorted_expert_ids = nullptr;
    unsigned *num_valid_ids = nullptr;

    float *input_scale_t = nullptr;
    float *scale_fc1 = nullptr;
    unsigned *fc2_scale_aiter_bits = nullptr;

    void AllocateAndCopy(const HostInputs &in, unsigned tokens, unsigned dim,
                         unsigned inter_dim, unsigned experts);
    void Free();
};

struct BenchmarkResult {
    double device_ms = 0.0;
    double tflops = 0.0;
    double bandwidth_gbps = 0.0;
};

__global__ void ShuffleWeightLayout16x32Kernel(const unsigned char *src,
                                               unsigned char *dst,
                                               unsigned experts, unsigned rows,
                                               unsigned cols) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t per_expert = static_cast<size_t>(rows) * cols;
    const size_t total = static_cast<size_t>(experts) * per_expert;
    if (idx >= total) {
        return;
    }

    const unsigned cb = cols / kBlockK;
    const size_t expert = idx / per_expert;
    const size_t local = idx - expert * per_expert;
    const unsigned row = static_cast<unsigned>(local / cols);
    const unsigned col = static_cast<unsigned>(local % cols);

    const unsigned rbi = row / kBlockN;
    const unsigned bni = row % kBlockN;
    const unsigned cbi = col / kBlockK;
    const unsigned inner = col % kBlockK;
    const unsigned kki = inner / kPackK;
    const unsigned kpi = inner % kPackK;

    const size_t dst_local =
        (((((static_cast<size_t>(rbi) * cb + cbi) * kKk + kki) * kBlockN +
           bni) *
          kPackK) +
         kpi);
    dst[expert * per_expert + dst_local] = src[idx];
}

__global__ void Transpose2DKernel(const float *src, float *dst, unsigned rows,
                                  unsigned cols) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * cols;
    if (idx >= total) {
        return;
    }
    const unsigned r = static_cast<unsigned>(idx / cols);
    const unsigned c = static_cast<unsigned>(idx % cols);
    dst[static_cast<size_t>(c) * rows + r] = src[idx];
}

static void BuildSortedRoutingFromTopK(const std::vector<unsigned> &topk_ids,
                                       const std::vector<float> &topk_weights,
                                       unsigned tokens, unsigned experts,
                                       unsigned topk, HostInputs *out) {
    const unsigned routes = tokens * topk;
    const unsigned max_num_tokens_padded =
        routes + experts * kSortedTokenPadding - topk;
    const unsigned max_num_m_blocks =
        CeilingDiv<unsigned>(max_num_tokens_padded, kSortedTokenPadding);
    const unsigned init_val = (topk << 24) | tokens;

    out->sorted_token_ids.assign(max_num_tokens_padded, init_val);
    out->sorted_weights.assign(max_num_tokens_padded, 0.0f);
    out->sorted_expert_ids.assign(max_num_m_blocks,
                                  std::numeric_limits<unsigned>::max());
    out->num_valid_ids.assign(2, 0u);

    unsigned sorted_ids_begin = 0;
    unsigned sorted_expert_ids_begin = 0;
    for (unsigned expert = 0; expert < experts; ++expert) {
        unsigned tokens_num = 0;
        for (unsigned token = 0; token < tokens; ++token) {
            for (unsigned slot = 0; slot < topk; ++slot) {
                const size_t idx = static_cast<size_t>(token) * topk + slot;
                if (topk_ids[idx] != expert) {
                    continue;
                }
                out->sorted_token_ids[sorted_ids_begin + tokens_num] =
                    (slot << 24) | token;
                out->sorted_weights[sorted_ids_begin + tokens_num] =
                    topk_weights[idx];
                ++tokens_num;
            }
        }
        if (tokens_num == 0) {
            continue;
        }
        const unsigned sorted_expert_ids_num =
            CeilingDiv<unsigned>(tokens_num, kSortedTokenPadding);
        const unsigned tokens_num_pad =
            sorted_expert_ids_num * kSortedTokenPadding;
        for (unsigned i = 0; i < sorted_expert_ids_num; ++i) {
            out->sorted_expert_ids[sorted_expert_ids_begin + i] = expert;
        }
        sorted_ids_begin += tokens_num_pad;
        sorted_expert_ids_begin += sorted_expert_ids_num;
    }

    out->num_valid_ids[0] = sorted_ids_begin;
    // Kernel ABI matches the reference test path: [padded_valid_route_ids, m].
    out->num_valid_ids[1] = tokens;
    out->num_valid_token_ids = sorted_ids_begin;
    out->num_valid_m_blocks = sorted_expert_ids_begin;
}

static HostInputs MakeInputs(unsigned tokens, unsigned dim, unsigned inter_dim,
                             unsigned experts, unsigned topk, unsigned seed) {
    HostInputs in;
    const unsigned n_blocks = dim / kScaleGroup;
    const unsigned k_blocks = inter_dim / kScaleGroup;
    const unsigned w1_rows = 2 * inter_dim;

    const size_t act_count = static_cast<size_t>(tokens) * dim;
    const size_t w13_count = static_cast<size_t>(experts) * w1_rows * dim;
    const size_t w2_count = static_cast<size_t>(experts) * dim * inter_dim;

    in.act_fp8.resize(act_count);
    in.w13_fp8.resize(w13_count);
    in.w2_fp8.resize(w2_count);
    in.scale_act.resize(static_cast<size_t>(tokens) * n_blocks);
    in.scale_fc1.resize(static_cast<size_t>(experts) * (w1_rows / kScaleGroup) *
                        n_blocks);
    in.scale_fc2.resize(static_cast<size_t>(experts) * n_blocks * k_blocks);
    in.fc2_scale_aiter_bits.resize(static_cast<size_t>(experts) * k_blocks *
                                   n_blocks);

    std::mt19937 gen(seed);
    std::normal_distribution<float> input_dist(0.0f, 0.5f);
    std::normal_distribution<float> spike_dist(0.0f, 8.0f);
    std::normal_distribution<float> scale_dist(1.0e-4f, 2.0e-5f);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> rw_dist(0.0f, 1.0f);
    std::uniform_int_distribution<unsigned> expert_dist(0, experts - 1);
    causalflow::petit::tests::fp8_sampler::FP8E4M3DiscreteSampler
        weight_sampler(40.0);

    FillFp8InputsParallel4(&in.act_fp8, [&](size_t idx) {
        float4 values;
        for (int i = 0; i < 4; ++i) {
            std::mt19937 lane_gen(static_cast<uint32_t>(MixU64(
                static_cast<uint64_t>(seed) ^ (0x17bull << 32) ^ (idx + i))));
            float x = input_dist(lane_gen);
            if (prob_dist(lane_gen) < 0.002f) {
                x += spike_dist(lane_gen);
            }
            reinterpret_cast<float *>(&values)[i] = x;
        }
        return __hip_fp8x4_e4m3_fnuz(values).__x;
    });
    FillWeightsParallel(&in.w13_fp8, weight_sampler, seed, 0x913u);
    FillWeightsParallel(&in.w2_fp8, weight_sampler, seed, 0x2d5u);
    for (float &s : in.scale_act) {
        s = std::max(1.0e-8f, scale_dist(gen));
    }
    for (float &s : in.scale_fc1) {
        s = std::max(1.0e-8f, scale_dist(gen));
    }
    for (float &s : in.scale_fc2) {
        s = std::max(1.0e-8f, scale_dist(gen));
    }

    std::vector<unsigned> topk_ids(static_cast<size_t>(tokens) * topk);
    std::vector<float> topk_weights(static_cast<size_t>(tokens) * topk);
    for (unsigned token = 0; token < tokens; ++token) {
        float sum = 0.0f;
        for (unsigned slot = 0; slot < topk; ++slot) {
            const size_t idx = static_cast<size_t>(token) * topk + slot;
            topk_ids[idx] = expert_dist(gen);
            topk_weights[idx] = rw_dist(gen);
            sum += topk_weights[idx];
        }
        sum = std::max(sum, 1.0e-12f);
        for (unsigned slot = 0; slot < topk; ++slot) {
            const size_t idx = static_cast<size_t>(token) * topk + slot;
            topk_weights[idx] /= sum;
        }
    }
    BuildSortedRoutingFromTopK(topk_ids, topk_weights, tokens, experts, topk,
                               &in);

    // Keep FC2 scale as [experts, n_blocks, k_blocks] and bitcast to
    // unsigned for kernel ABI.
    for (unsigned e = 0; e < experts; ++e) {
        for (unsigned nb = 0; nb < n_blocks; ++nb) {
            for (unsigned kb = 0; kb < k_blocks; ++kb) {
                const size_t src =
                    (static_cast<size_t>(e) * n_blocks + nb) * k_blocks + kb;
                const size_t dst =
                    (static_cast<size_t>(e) * n_blocks + nb) * k_blocks + kb;
                in.fc2_scale_aiter_bits[dst] = FloatAsBits(in.scale_fc2[src]);
            }
        }
    }

    return in;
}

void DeviceBuffers::AllocateAndCopy(const HostInputs &in, unsigned tokens,
                                    unsigned dim, unsigned inter_dim,
                                    unsigned experts) {
    const size_t out_bytes =
        static_cast<size_t>(tokens) * dim * sizeof(unsigned short);
    const unsigned n_blocks = dim / kScaleGroup;
    const unsigned w1_rows = 2 * inter_dim;
    constexpr unsigned kThreads = 256;
    const dim3 block(kThreads);
    const auto launch_1d = [&](size_t n) {
        return dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads));
    };

    CheckHIPStatus(hipMalloc(&act_fp8, in.act_fp8.size()));
    CheckHIPStatus(hipMalloc(&w1_q_shuffled, in.w13_fp8.size()));
    CheckHIPStatus(hipMalloc(&w2_q_shuffled, in.w2_fp8.size()));
    CheckHIPStatus(hipMalloc(&out_bf16, out_bytes));

    CheckHIPStatus(hipMalloc(&sorted_token_ids,
                             in.sorted_token_ids.size() * sizeof(unsigned)));
    CheckHIPStatus(
        hipMalloc(&sorted_weights, in.sorted_weights.size() * sizeof(float)));
    CheckHIPStatus(hipMalloc(&sorted_expert_ids,
                             in.sorted_expert_ids.size() * sizeof(unsigned)));
    CheckHIPStatus(
        hipMalloc(&num_valid_ids, in.num_valid_ids.size() * sizeof(unsigned)));

    CheckHIPStatus(
        hipMalloc(&input_scale_t, in.scale_act.size() * sizeof(float)));
    CheckHIPStatus(hipMalloc(&scale_fc1, in.scale_fc1.size() * sizeof(float)));
    CheckHIPStatus(
        hipMalloc(&fc2_scale_aiter_bits,
                  in.fc2_scale_aiter_bits.size() * sizeof(unsigned)));

    unsigned char *w13_fp8_tmp = nullptr;
    unsigned char *w2_fp8_tmp = nullptr;
    float *scale_act_tmp = nullptr;
    CheckHIPStatus(hipMalloc(&w13_fp8_tmp, in.w13_fp8.size()));
    CheckHIPStatus(hipMalloc(&w2_fp8_tmp, in.w2_fp8.size()));
    CheckHIPStatus(
        hipMalloc(&scale_act_tmp, in.scale_act.size() * sizeof(float)));

    CheckHIPStatus(hipMemcpy(act_fp8, in.act_fp8.data(), in.act_fp8.size(),
                             hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(w13_fp8_tmp, in.w13_fp8.data(), in.w13_fp8.size(),
                             hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(w2_fp8_tmp, in.w2_fp8.data(), in.w2_fp8.size(),
                             hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(sorted_token_ids, in.sorted_token_ids.data(),
                             in.sorted_token_ids.size() * sizeof(unsigned),
                             hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(sorted_weights, in.sorted_weights.data(),
                             in.sorted_weights.size() * sizeof(float),
                             hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(sorted_expert_ids, in.sorted_expert_ids.data(),
                             in.sorted_expert_ids.size() * sizeof(unsigned),
                             hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(num_valid_ids, in.num_valid_ids.data(),
                             in.num_valid_ids.size() * sizeof(unsigned),
                             hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(scale_act_tmp, in.scale_act.data(),
                             in.scale_act.size() * sizeof(float),
                             hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(scale_fc1, in.scale_fc1.data(),
                             in.scale_fc1.size() * sizeof(float),
                             hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(fc2_scale_aiter_bits,
                             in.fc2_scale_aiter_bits.data(),
                             in.fc2_scale_aiter_bits.size() * sizeof(unsigned),
                             hipMemcpyHostToDevice));

    ShuffleWeightLayout16x32Kernel<<<launch_1d(in.w13_fp8.size()), block>>>(
        w13_fp8_tmp, w1_q_shuffled, experts, w1_rows, dim);
    CheckHIPStatus(hipGetLastError());
    ShuffleWeightLayout16x32Kernel<<<launch_1d(in.w2_fp8.size()), block>>>(
        w2_fp8_tmp, w2_q_shuffled, experts, dim, inter_dim);
    CheckHIPStatus(hipGetLastError());
    Transpose2DKernel<<<launch_1d(in.scale_act.size()), block>>>(
        scale_act_tmp, input_scale_t, tokens, n_blocks);
    CheckHIPStatus(hipGetLastError());
    CheckHIPStatus(hipDeviceSynchronize());

    CheckHIPStatus(hipFree(w13_fp8_tmp));
    CheckHIPStatus(hipFree(w2_fp8_tmp));
    CheckHIPStatus(hipFree(scale_act_tmp));
}

void DeviceBuffers::Free() {
    if (act_fp8 != nullptr) {
        CheckHIPStatus(hipFree(act_fp8));
        act_fp8 = nullptr;
    }
    if (w1_q_shuffled != nullptr) {
        CheckHIPStatus(hipFree(w1_q_shuffled));
        w1_q_shuffled = nullptr;
    }
    if (w2_q_shuffled != nullptr) {
        CheckHIPStatus(hipFree(w2_q_shuffled));
        w2_q_shuffled = nullptr;
    }
    if (out_bf16 != nullptr) {
        CheckHIPStatus(hipFree(out_bf16));
        out_bf16 = nullptr;
    }
    if (sorted_token_ids != nullptr) {
        CheckHIPStatus(hipFree(sorted_token_ids));
        sorted_token_ids = nullptr;
    }
    if (sorted_weights != nullptr) {
        CheckHIPStatus(hipFree(sorted_weights));
        sorted_weights = nullptr;
    }
    if (sorted_expert_ids != nullptr) {
        CheckHIPStatus(hipFree(sorted_expert_ids));
        sorted_expert_ids = nullptr;
    }
    if (num_valid_ids != nullptr) {
        CheckHIPStatus(hipFree(num_valid_ids));
        num_valid_ids = nullptr;
    }
    if (input_scale_t != nullptr) {
        CheckHIPStatus(hipFree(input_scale_t));
        input_scale_t = nullptr;
    }
    if (scale_fc1 != nullptr) {
        CheckHIPStatus(hipFree(scale_fc1));
        scale_fc1 = nullptr;
    }
    if (fc2_scale_aiter_bits != nullptr) {
        CheckHIPStatus(hipFree(fc2_scale_aiter_bits));
        fc2_scale_aiter_bits = nullptr;
    }
}

static bool RunBenchmark(const HostInputs &in, const DeviceBuffers &dev,
                         unsigned tokens, unsigned dim, unsigned inter_dim,
                         unsigned topk, unsigned num_persistent_tgs,
                         BenchmarkResult *result) {
    const size_t out_bytes =
        static_cast<size_t>(tokens) * dim * sizeof(unsigned short);
    CheckHIPStatus(hipMemset(dev.out_bf16, 0, out_bytes));
    const int err = moe::FusedMoEBlockScaleFP8(
        reinterpret_cast<uint4 *>(dev.out_bf16),
        reinterpret_cast<const uint4 *>(dev.act_fp8),
        reinterpret_cast<const uint4 *>(dev.w1_q_shuffled),
        reinterpret_cast<const uint4 *>(dev.w2_q_shuffled),
        reinterpret_cast<const uint4 *>(dev.sorted_token_ids),
        reinterpret_cast<const uint4 *>(dev.sorted_weights),
        reinterpret_cast<const uint4 *>(dev.sorted_expert_ids),
        dev.num_valid_ids, topk,
        reinterpret_cast<const uint4 *>(dev.input_scale_t),
        reinterpret_cast<const uint4 *>(dev.scale_fc1),
        dev.fc2_scale_aiter_bits,
        static_cast<unsigned>(in.sorted_expert_ids.size()), tokens, dim,
        inter_dim, nullptr, num_persistent_tgs);
    if (err != 0) {
        return false;
    }

    for (int i = 0; i < FLAGS_warmup; ++i) {
        const int warmup_err = moe::FusedMoEBlockScaleFP8(
            reinterpret_cast<uint4 *>(dev.out_bf16),
            reinterpret_cast<const uint4 *>(dev.act_fp8),
            reinterpret_cast<const uint4 *>(dev.w1_q_shuffled),
            reinterpret_cast<const uint4 *>(dev.w2_q_shuffled),
            reinterpret_cast<const uint4 *>(dev.sorted_token_ids),
            reinterpret_cast<const uint4 *>(dev.sorted_weights),
            reinterpret_cast<const uint4 *>(dev.sorted_expert_ids),
            dev.num_valid_ids, topk,
            reinterpret_cast<const uint4 *>(dev.input_scale_t),
            reinterpret_cast<const uint4 *>(dev.scale_fc1),
            dev.fc2_scale_aiter_bits,
            static_cast<unsigned>(in.sorted_expert_ids.size()), tokens, dim,
            inter_dim, nullptr, num_persistent_tgs);
        if (warmup_err != 0) {
            return false;
        }
    }
    CheckHIPStatus(hipDeviceSynchronize());
    CheckHIPStatus(hipMemset(dev.out_bf16, 0, out_bytes));

    hipEvent_t ev_start;
    hipEvent_t ev_stop;
    CheckHIPStatus(hipEventCreate(&ev_start));
    CheckHIPStatus(hipEventCreate(&ev_stop));

    CheckHIPStatus(hipEventRecord(ev_start, 0));
    for (int i = 0; i < FLAGS_repeat; ++i) {
        moe::FusedMoEBlockScaleFP8(
            reinterpret_cast<uint4 *>(dev.out_bf16),
            reinterpret_cast<const uint4 *>(dev.act_fp8),
            reinterpret_cast<const uint4 *>(dev.w1_q_shuffled),
            reinterpret_cast<const uint4 *>(dev.w2_q_shuffled),
            reinterpret_cast<const uint4 *>(dev.sorted_token_ids),
            reinterpret_cast<const uint4 *>(dev.sorted_weights),
            reinterpret_cast<const uint4 *>(dev.sorted_expert_ids),
            dev.num_valid_ids, topk,
            reinterpret_cast<const uint4 *>(dev.input_scale_t),
            reinterpret_cast<const uint4 *>(dev.scale_fc1),
            dev.fc2_scale_aiter_bits,
            static_cast<unsigned>(in.sorted_expert_ids.size()), tokens, dim,
            inter_dim, nullptr, num_persistent_tgs);
    }
    CheckHIPStatus(hipEventRecord(ev_stop, 0));
    CheckHIPStatus(hipEventSynchronize(ev_stop));

    float device_total_ms = 0.0f;
    CheckHIPStatus(hipEventElapsedTime(&device_total_ms, ev_start, ev_stop));
    CheckHIPStatus(hipEventDestroy(ev_start));
    CheckHIPStatus(hipEventDestroy(ev_stop));

    const double device_ms = static_cast<double>(device_total_ms) /
                             static_cast<double>(FLAGS_repeat);
    const double device_s = device_ms * 1e-3;

    // Fused MoE per route: gate GEMM + up GEMM + down GEMM.
    const double routes = static_cast<double>(tokens) * topk;
    const double flops = 6.0 * routes * static_cast<double>(dim) *
                         static_cast<double>(inter_dim);
    const double tflops = flops / device_s / 1e12;

    const double bytes =
        static_cast<double>(in.act_fp8.size()) +
        static_cast<double>(in.w13_fp8.size()) +
        static_cast<double>(in.w2_fp8.size()) +
        static_cast<double>(in.sorted_token_ids.size() * sizeof(unsigned)) +
        static_cast<double>(in.sorted_weights.size() * sizeof(float)) +
        static_cast<double>(in.sorted_expert_ids.size() * sizeof(unsigned)) +
        static_cast<double>(in.scale_act.size() * sizeof(float)) +
        static_cast<double>(in.scale_fc1.size() * sizeof(float)) +
        static_cast<double>(in.fc2_scale_aiter_bits.size() * sizeof(unsigned)) +
        static_cast<double>(tokens) * dim * sizeof(unsigned short);
    const double bandwidth_gbps = bytes / device_s / 1e9;

    result->device_ms = device_ms;
    result->tflops = tflops;
    result->bandwidth_gbps = bandwidth_gbps;
    return true;
}

} // namespace

int main(int argc, char **argv) {
    RegisterValidators();
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_tokens <= 0 || FLAGS_dim <= 0 || FLAGS_inter_dim <= 0 ||
        FLAGS_experts <= 0 || FLAGS_topk <= 0) {
        std::cerr << "tokens, dim, inter_dim, experts, and topk must all be "
                     "positive.\n";
        return 1;
    }
    if (FLAGS_warmup < 0 || FLAGS_repeat <= 0) {
        std::cerr
            << "warmup must be non-negative and repeat must be positive.\n";
        return 1;
    }
    if (FLAGS_num_persistent_tgs < 0 || FLAGS_persistent_tgs_per_cu <= 0) {
        std::cerr << "num_persistent_tgs must be >= 0 and "
                     "persistent_tgs_per_cu must be > 0.\n";
        return 1;
    }
    if (FLAGS_topk > FLAGS_experts) {
        std::cerr << "topk must be <= experts.\n";
        return 1;
    }
    if (FLAGS_topk > 255) {
        std::cerr << "topk must be <= 255 because route slot id is packed in 8 "
                     "bits.\n";
        return 1;
    }
    if (FLAGS_tokens >= (1 << 24)) {
        std::cerr << "tokens must be < 2^24 because token id is packed in 24 "
                     "bits.\n";
        return 1;
    }
    if ((FLAGS_dim % 256) != 0 || (FLAGS_inter_dim % 256) != 0) {
        std::cerr << "dim and inter_dim must both be divisible by 256.\n";
        return 1;
    }

    const unsigned tokens = static_cast<unsigned>(FLAGS_tokens);
    const unsigned dim = static_cast<unsigned>(FLAGS_dim);
    const unsigned inter_dim = static_cast<unsigned>(FLAGS_inter_dim);
    const unsigned experts = static_cast<unsigned>(FLAGS_experts);
    const unsigned topk = static_cast<unsigned>(FLAGS_topk);
    const unsigned seed = static_cast<unsigned>(FLAGS_seed);
    unsigned num_persistent_tgs =
        static_cast<unsigned>(FLAGS_num_persistent_tgs);
    if (num_persistent_tgs == 0 && FLAGS_persistent) {
        int device = 0;
        CheckHIPStatus(hipGetDevice(&device));
        hipDeviceProp_t props;
        CheckHIPStatus(hipGetDeviceProperties(&props, device));
        num_persistent_tgs = static_cast<unsigned>(props.multiProcessorCount *
                                                   FLAGS_persistent_tgs_per_cu);
    }

    HostInputs in = MakeInputs(tokens, dim, inter_dim, experts, topk, seed);
    DeviceBuffers dev;
    dev.AllocateAndCopy(in, tokens, dim, inter_dim, experts);

    BenchmarkResult result;
    const bool ok = RunBenchmark(in, dev, tokens, dim, inter_dim, topk,
                                 num_persistent_tgs, &result);
    dev.Free();

    if (!ok) {
        std::cerr << "FusedMoEBlockScaleFP8 benchmark failed.\n";
        return 1;
    }

    fmt::print(
        "backend={} kernel=32x256x256 tokens={} dim={} inter_dim={} experts={} "
        "topk={} valid_ids={} valid_m_blocks={} persistent_tgs={} launch={} "
        "device_ms={:.4f} tflops={:.3f} gbps={:.3f}\n",
        FLAGS_backend, tokens, dim, inter_dim, experts, topk,
        in.num_valid_token_ids, in.num_valid_m_blocks, num_persistent_tgs,
        num_persistent_tgs > 0 ? "persistent" : "standard", result.device_ms,
        result.tflops, result.bandwidth_gbps);
    return 0;
}
