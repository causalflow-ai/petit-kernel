#include "pybind.h"
#include "gemm/rocm/quantization/gemm.h"
#include <torch/python.h>

using namespace causalflow::petit::pybind;
using namespace causalflow::petit::rocm::quantization;

PYBIND11_MODULE(ops, m) {
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
          "Unified one-stage fused MoE matmul dispatcher with caller-provided output");
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
