#pragma once

#include "moe/rocm/fused_moe.h"

namespace causalflow::petit::rocm::moe {

struct MegaMoEParams {
    unsigned *out;
    unsigned output_row_stride;
    const unsigned *w13;
    const unsigned *w2;
    const unsigned *scales_w13;
    const unsigned *scales_w2;
    const unsigned *input_tokens;
    const unsigned *input_topk_ids;
    const float *input_topk_weights;
    unsigned num_tokens;
    unsigned hidden_size;
    unsigned inter_dim;
    const void *w13_bias;
    const void *w2_bias;
    void *workspace;
    unsigned rank;
    hipStream_t stream;
};

struct MegaMoEWorkspaceInfo {
    unsigned barrier_record_bytes;
    unsigned rank_sym_buffer_base;
    unsigned rank_slot_bytes;
    unsigned local_offset;
    unsigned local_bytes;
    unsigned input_tokens_offset;
    unsigned input_topk_expert_id_offset;
    unsigned input_topk_expert_weight_offset;
    unsigned input_token_bytes;
    unsigned max_tokens_per_rank;
    unsigned num_ranks;
    unsigned num_experts;
    unsigned topk;
    unsigned hidden_size;
    unsigned compute_hidden_size;
    FusedMoEDataType act_dtype;
};

int MegaMoECompute(MegaMoEParams params, unsigned long solution_id);
int MegaMoEQuantizeMxFp4(const void *input, unsigned char *output,
                         unsigned rows, unsigned cols,
                         unsigned input_row_stride, hipStream_t stream);
int GetMegaMoEWorkspaceInfo(unsigned rank, unsigned long solution_id,
                            MegaMoEWorkspaceInfo *info);

template <unsigned long kRepr> struct MegaMoESolutionAdapter {
    static int Invoke(MegaMoEParams params);
    static int GetWorkspaceInfo(unsigned rank, MegaMoEWorkspaceInfo *info);
};

} // namespace causalflow::petit::rocm::moe
