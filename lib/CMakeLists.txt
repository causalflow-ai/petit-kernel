add_subdirectory(gemm)

add_subdirectory(hal)
add_subdirectory(moe)
add_subdirectory(tests)

if (WITH_PYTHON)
  add_subdirectory(pybind)
endif ()
