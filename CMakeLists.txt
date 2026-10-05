cmake_minimum_required(VERSION 3.24)

# Must be set before project() so CMake's CUDA compiler check uses it.
# Turing: RTX 2060 (local) and T4 (Colab) are both sm_75.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(cuda_mc_pricer LANGUAGES CXX CUDA)

# Benchmarks in Debug are meaningless, so default to Release for single-config
# generators (Ninja, Makefiles). Multi-config generators (Visual Studio) pick
# the config at build time with --config.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)  # cudart; cuRAND device API and Thrust are header-only
find_package(OpenMP REQUIRED)

# Everything except main() lives in one library so the benchmark and the tests
# exercise exactly the same code.
add_library(mc_core STATIC
  src/black_scholes.cpp
  src/cpu_pricer.cpp
  src/gpu_naive.cu
  src/gpu_optimized.cu
)
target_include_directories(mc_core PUBLIC include)
target_link_libraries(mc_core PUBLIC CUDA::cudart OpenMP::OpenMP_CXX)

target_compile_options(mc_core PUBLIC
  # Warnings for host C++ only; nvcc's host pass would forward these to cl/gcc
  # and drown real problems in CUDA-header noise.
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4 /permissive->
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall -Wextra>
  # Maps SASS back to source lines in Nsight Compute; no runtime cost.
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

add_executable(mc_pricer src/main.cpp)
target_link_libraries(mc_pricer PRIVATE mc_core)

enable_testing()
add_executable(test_pricer tests/test_pricer.cpp)
target_link_libraries(test_pricer PRIVATE mc_core)
add_test(NAME test_pricer COMMAND test_pricer)
