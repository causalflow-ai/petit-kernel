add_executable(bench_fused_moe
  main.cc
  ${PROJECT_SOURCE_DIR}/lib/moe/rocm/fused_moe_test_utils.cu
)
set_source_files_properties(main.cc ${PROJECT_SOURCE_DIR}/lib/moe/rocm/fused_moe_test_utils.cu PROPERTIES
  LANGUAGE HIP
)

target_link_libraries(bench_fused_moe
  PRIVATE
    causalflow-petit-rocm-moe
    causalflow-petit-rocm-quantization
    causalflow-petit-tests
    fmt::fmt
    gflags
    GTest::gtest
    hip::host
    roc::hipblaslt
    TBB::tbb
)

if(COMMAND petit_set_moe_hip_architectures)
  petit_set_moe_hip_architectures(bench_fused_moe)
endif()
