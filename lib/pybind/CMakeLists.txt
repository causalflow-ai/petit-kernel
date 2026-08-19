add_library(petit_kernels SHARED
    fp4.cc
    mega_moe.cc
    moe.cc
    pybind.cc
    vmm_symmetric_heap.cc
)

# Test-only kernels used by distributed communication microbenchmarks. Keep
# these out of the public ops module so benchmark scaffolding cannot become a
# serving-path dependency.
add_library(petit_kernel_test_ops MODULE EXCLUDE_FROM_ALL
    token_shuffle_test.cu
    vmm_symmetric_heap.cc
)
set_source_files_properties(token_shuffle_test.cu PROPERTIES
    LANGUAGE HIP
    COMPILE_OPTIONS "-std=c++20"
)
target_compile_options(petit_kernel_test_ops PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-pass-failed>
)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

foreach(target IN ITEMS petit_kernels petit_kernel_test_ops)
    target_link_libraries(${target} PRIVATE
        causalflow-petit-rocm-quantization
        causalflow-petit-rocm-moe
        ${TORCH_PYTHON_LIBRARY}
        ${TORCH_LIBRARIES}
    )

    target_include_directories(${target} PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${Python_INCLUDE_DIRS}
    )
endforeach()

set_target_properties(petit_kernels PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${LIBRARY_OUTPUT_NAME}"
)
set_target_properties(petit_kernel_test_ops PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_test_ops"
)

if(MSVC)
    set_target_properties(petit_kernels PROPERTIES SUFFIX ".pyd")
    set_target_properties(petit_kernel_test_ops PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(petit_kernels PROPERTIES SUFFIX ".so")
    set_target_properties(petit_kernel_test_ops PROPERTIES SUFFIX ".so")
endif()
