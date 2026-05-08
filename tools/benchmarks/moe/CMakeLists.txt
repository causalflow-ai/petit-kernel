add_executable(bench_fused_moe main.cc)
set_source_files_properties(main.cc PROPERTIES
  LANGUAGE HIP
)

target_link_libraries(bench_fused_moe
  PRIVATE
    causalflow-petit-rocm-moe
    causalflow-petit-tests
    fmt::fmt
    gflags
    hip::host
    TBB::tbb
)

if(COMMAND petit_set_moe_hip_architectures)
  petit_set_moe_hip_architectures(bench_fused_moe)
endif()
