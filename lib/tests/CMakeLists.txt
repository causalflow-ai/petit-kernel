add_library(causalflow-petit-tests quantization.cc fp8_sampler.cc)
target_link_libraries(causalflow-petit-tests hal absl::status)