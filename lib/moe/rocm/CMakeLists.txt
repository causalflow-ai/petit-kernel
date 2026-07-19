function(petit_set_moe_hip_architectures target)
  set(moe_hip_architectures)
  foreach(arch IN LISTS CMAKE_HIP_ARCHITECTURES)
    if(arch MATCHES "^(gfx940|gfx941|gfx942|gfx950)")
      list(APPEND moe_hip_architectures ${arch})
    endif()
  endforeach()

  if(moe_hip_architectures)
    set_target_properties(${target} PROPERTIES
      HIP_ARCHITECTURES "${moe_hip_architectures}"
    )
  endif()
endfunction()

set(MOE_SRCS
  fused_moe.cc
  mega_moe.cc
)

set_source_files_properties(${MOE_SRCS} PROPERTIES
  LANGUAGE HIP
)
if(CMAKE_HIP_COMPILER_VERSION VERSION_GREATER_EQUAL 23)
  set_property(SOURCE fused_moe.cc mega_moe.cc APPEND PROPERTY
    COMPILE_OPTIONS "-Wno-error=pass-failed")
endif()

add_library(causalflow-petit-rocm-moe
  ${MOE_SRCS}
)
petit_set_moe_hip_architectures(causalflow-petit-rocm-moe)

target_link_libraries(causalflow-petit-rocm-moe hip::host)

add_executable(fused_moe_blockscale_fp8_test
  fused_moe_blockscale_fp8_test.cc
  fused_moe_test_utils.cc
  fused_moe_test_utils.cu
)
set_source_files_properties(fused_moe_blockscale_fp8_test.cc fused_moe_test_utils.cc fused_moe_test_utils.cu PROPERTIES
  LANGUAGE HIP
)
target_link_libraries(fused_moe_blockscale_fp8_test
  causalflow-petit-rocm-moe
  causalflow-petit-tests
  roc::hipblaslt
  GTest::gtest_main
  TBB::tbb
)
petit_set_moe_hip_architectures(fused_moe_blockscale_fp8_test)
add_test(fused_moe_blockscale_fp8_test fused_moe_blockscale_fp8_test)

add_executable(fused_moe_mxfp4_test
  fused_moe_mxfp4_test.cc
  fused_moe_bf16_mxfp4_test.cc
  fused_moe_test_utils.cc
  fused_moe_test_utils.cu
)
set_source_files_properties(fused_moe_mxfp4_test.cc fused_moe_bf16_mxfp4_test.cc fused_moe_test_utils.cc fused_moe_test_utils.cu PROPERTIES
  LANGUAGE HIP
)
target_link_libraries(fused_moe_mxfp4_test
  causalflow-petit-rocm-moe
  causalflow-petit-rocm-quantization
  causalflow-petit-tests
  roc::hipblaslt
  GTest::gtest_main
  TBB::tbb
)
petit_set_moe_hip_architectures(fused_moe_mxfp4_test)
add_test(fused_moe_mxfp4_test fused_moe_mxfp4_test)
