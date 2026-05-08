cmake_minimum_required(VERSION 3.20)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HUNTER_PACKAGES gflags GTest abseil fmt spdlog RapidJSON)
set(HUNTER_FILEPATH_CONFIG ${CMAKE_SOURCE_DIR}/cmake/hunter_config.cmake)

include(FetchContent)
set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_deps")

FetchContent_Declare(SetupHunter GIT_REPOSITORY https://github.com/cpp-pm/gate)
FetchContent_MakeAvailable(SetupHunter)

project(petit_kernel LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

if (WITH_ROCM)
  message(STATUS "Building with ROCm support")
  find_package(HIP REQUIRED)
  find_package(rocblas REQUIRED)
  find_package(hipblaslt REQUIRED)
  enable_language(HIP)
  set(CMAKE_HIP_FLAGS "-std=c++20 -Wall")
endif()

if (WITH_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(Torch REQUIRED)
endif()

include(build_utils)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options(-Wall -fno-strict-aliasing)
if (WITH_PYTHON)
    add_compile_options(-fPIC)
else()
    # https://github.com/pytorch/pytorch/issues/99715 prevents us to turn on -Werror by default
    add_compile_options(-Werror)
endif()

find_package(fmt CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb)

enable_testing()

configure_file(include/causalflow/petit/config.h.in ${CMAKE_BINARY_DIR}/include/causalflow/petit/config.h)
include_directories(include lib ${CMAKE_BINARY_DIR}/include)

add_subdirectory(lib)
add_subdirectory(tools)
