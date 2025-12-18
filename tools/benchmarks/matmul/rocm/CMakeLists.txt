add_library(benchmark_matmul_rocm
  matmul_petit.cc
  matmul_hipblaslt.cc
) 

target_link_libraries(benchmark_matmul_rocm
  PRIVATE
    causalflow-petit-rocm-quantization
    causalflow-petit-tests
    absl::status roc::rocblas roc::hipblaslt 
)