#include "pybind.h"
#include "gemm/rocm/quantization/gemm.h"
#include <torch/python.h>

using namespace causalflow::petit::pybind;
using namespace causalflow::petit::rocm::quantization;

PYBIND11_MODULE(ops, m) {
    pybind11::class_<VmmSymmetricHeap>(m, "VmmSymmetricHeap")
        .def(pybind11::init<int>(), pybind11::arg("world_size"))
        .def("local_tensor", &VmmSymmetricHeap::LocalTensor);
    m.def("repack_nvfp4", &RepackNvFp4, "Repack NVFP4 to Petit FP4");
    m.def("process_nvfp4_scales", &ProcessNvFp4Scales, "Process NVFP4 scales");
    m.def("process_mxfp4_scales", &ProcessMxFp4Scales, "Process MXFP4 scales");
    m.def("mul_nvfp4_a16", &MulNvFp4A16, "Multiply NVFP4 FP16");
    m.def("mul_mxfp4_a16", &MulMxFp4A16, "Multiply MXFP4 FP16");
    m.def("fmoe_matmul_1stage", &FusedMoeMatmul1Stage, pybind11::arg("out"),
          pybind11::arg("input_q"), pybind11::arg("w1_q"),
          pybind11::arg("w2_q"), pybind11::arg("sorted_token_ids"),
          pybind11::arg("sorted_weights"), pybind11::arg("sorted_expert_ids"),
          pybind11::arg("num_valid_ids"), pybind11::arg("topk"),
          pybind11::arg("input_scale"), pybind11::arg("w1_scale"),
          pybind11::arg("w2_scale"), pybind11::arg("solution_id"),
          pybind11::arg("num_persistent_tgs") = 0,
          pybind11::arg("w13_bias") = pybind11::none(),
          pybind11::arg("w2_bias") = pybind11::none(),
          "Unified one-stage fused MoE matmul dispatcher with caller-provided "
          "output");
    m.def("fmoe_matmul_2stage_workspace_size", &FusedMoe2StageWorkspaceSize,
          pybind11::arg("max_num_m_blocks"), pybind11::arg("inter_dim"),
          pybind11::arg("solution_id"));
    m.def("fmoe_matmul_2stage_stage1", &FusedMoeMatmul2Stage1,
          pybind11::arg("intermediate"), pybind11::arg("input_q"),
          pybind11::arg("w1_q"), pybind11::arg("sorted_token_ids"),
          pybind11::arg("sorted_expert_ids"), pybind11::arg("num_valid_ids"),
          pybind11::arg("topk"), pybind11::arg("input_scale"),
          pybind11::arg("w1_scale"), pybind11::arg("inter_dim"),
          pybind11::arg("num_experts"), pybind11::arg("solution_id"),
          pybind11::arg("num_persistent_tgs") = 0,
          pybind11::arg("w13_bias") = pybind11::none());
    m.def("fmoe_matmul_2stage_stage2", &FusedMoeMatmul2Stage2,
          pybind11::arg("out"), pybind11::arg("intermediate"),
          pybind11::arg("w2_q"), pybind11::arg("sorted_token_ids"),
          pybind11::arg("sorted_weights"), pybind11::arg("sorted_expert_ids"),
          pybind11::arg("num_valid_ids"), pybind11::arg("topk"),
          pybind11::arg("w2_scale"), pybind11::arg("inter_dim"),
          pybind11::arg("num_experts"), pybind11::arg("solution_id"),
          pybind11::arg("num_persistent_tgs") = 0,
          pybind11::arg("w2_bias") = pybind11::none());
    m.def("mega_moe_workspace_input_views", &MegaMoeWorkspaceInputViews,
          pybind11::arg("workspace"), pybind11::arg("max_tokens"),
          pybind11::arg("solution_id"));
    m.def("mega_moe_quantize_mxfp4", &MegaMoeQuantizeMxFp4,
          pybind11::arg("input"), pybind11::arg("output") = pybind11::none(),
          pybind11::arg("output_scales") = pybind11::none());
    m.def("mega_moe", &MegaMoe, pybind11::arg("workspace"),
          pybind11::arg("w13"), pybind11::arg("w2"),
          pybind11::arg("fc1_scale"), pybind11::arg("fc2_scale"),
          pybind11::arg("num_tokens"), pybind11::arg("solution_id"),
          pybind11::arg("w13_bias") = pybind11::none(),
          pybind11::arg("w2_bias") = pybind11::none(),
          pybind11::arg("out") = pybind11::none(),
          pybind11::arg("input_tokens") = pybind11::none(),
          pybind11::arg("input_topk_ids") = pybind11::none(),
          pybind11::arg("input_topk_weights") = pybind11::none());
    m.def("get_nvfp4_solutions", &GetNvFp4Solutions,
          "Get possible fp4 solutions");
    m.def("get_fp4_solutions", &GetNvFp4Solutions,
          "Get possible fp4 solutions");

    pybind11::class_<PetitSolutionHints>(m, "PetitSolutionHints")
        .def(pybind11::init<>())
        .def_readwrite("a_type", &PetitSolutionHints::a_type)
        .def_readwrite("b_type", &PetitSolutionHints::b_type)
        .def_readwrite("c_type", &PetitSolutionHints::c_type)
        .def_readwrite("require_high_precision",
                       &PetitSolutionHints::require_high_precision);
}
