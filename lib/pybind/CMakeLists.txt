add_library(petit_kernels SHARED
    fp4.cc
    mega_moe.cc
    moe.cc
    pybind.cc
    vmm_symmetric_heap.cc
)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(petit_kernels PRIVATE
    causalflow-petit-rocm-quantization
    causalflow-petit-rocm-moe
    ${TORCH_PYTHON_LIBRARY}
    ${TORCH_LIBRARIES}
)

target_include_directories(petit_kernels PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

set_target_properties(petit_kernels PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${LIBRARY_OUTPUT_NAME}"
)

if(MSVC)
    set_target_properties(petit_kernels PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(petit_kernels PROPERTIES SUFFIX ".so")
endif()
