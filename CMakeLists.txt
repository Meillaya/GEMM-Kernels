cmake_minimum_required(VERSION 3.10)
project(GEMM_Kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set a default CUDA architecture. This is required by modern CUDA versions.
# We'll use architecture 75 (Turing) as a default. This allows compiling
# the CUDA code even without a physical NVIDIA GPU present.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

include_directories(include)

add_executable(gemm_test
  src/main.cpp
  src/gemm_cpu.cpp
  src/gemm_cuda.cu
)

# Find the CUDA toolkit and link against it.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(gemm_test PRIVATE CUDA::cudart) 