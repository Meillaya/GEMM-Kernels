# Project Documentation

This document details the setup, initial code, tools, and issues encountered during the development of the `GEMM_Kernels` project.

## Project Goal

The main goal of this project is to implement optimized High-Performance Matrix Multiplication (GEMM) kernels in C++/CUDA for both CPUs and GPUs.

This project aims to help develop skills in:
- **Low-level optimization:** Techniques like loop unrolling, SIMD, and cache blocking.
- **Parallel programming:** Using frameworks such as OpenMP for CPU and CUDA for GPU. SYCL is a potential future addition.
- **Performance benchmarking:** Measuring performance in terms of FLOPs and memory bandwidth.

### Advanced Extensions
- Compare different parallelization strategies (e.g., tiling, warp-level optimizations in CUDA).
- Integrate the optimized kernels into a deep learning framework like PyTorch or TensorFlow as a custom operator.


## Project Setup

The project is a C++/CUDA implementation for matrix multiplication (GEMM). The initial goal is to compare the performance of CPU and GPU implementations as a baseline.

### Directory Structure

- `src/`: Contains the source code.
  - `main.cpp`: The main entry point of the application. It handles matrix initialization, calling the CPU and GPU implementations, and verifying the results.
  - `gemm_cpu.cpp`: Contains the CPU implementation of the matrix multiplication.
  - `gemm_cuda.cu`: Contains the CUDA implementation of the matrix multiplication, including the kernel and host-side wrapper function.
- `include/`: Contains the header files.
  - `gemm.h`: Header file containing the function prototypes for the CPU and GPU GEMM functions.
- `build/`: Directory where the compiled binaries are placed. This is a standard practice for CMake projects.
- `docs/`: Contains project documentation.
- `CMakeLists.txt`: The build script for CMake. It defines the project, finds dependencies (like CUDA), and specifies the executable to be built.
- `README.md`: This file, providing a summary and instructions.

### Tools Used

- **C++**: The core language for the CPU implementation and application logic.
- **CUDA**: For the GPU-accelerated matrix multiplication.
- **CMake**: The build system used to manage the compilation process, especially for a project involving both C++ and CUDA.
- **NVIDIA CUDA Toolkit**: Required for compiling the CUDA code. The `CMakeLists.txt` is configured to find and use it.

## Initial Code

The initial codebase consists of three main components:

1.  A CPU-based matrix multiplication function.
2.  A GPU-based matrix multiplication function using a naive CUDA kernel.
3.  A `main` function to drive the tests, compare results, and print basic information.

The CUDA kernel in `src/gemm_cuda.cu` is a straightforward implementation where each thread computes one element of the output matrix `C`.

## Build Process

The project is built using CMake. The `CMakeLists.txt` file is configured to:
1.  Identify the project as a C++/CUDA project.
2.  Set the C++ and CUDA standards to C++17.
3.  Set a default CUDA architecture (sm_75) to allow compilation even without a GPU.
4.  Include the `include` directory.
5.  Define an executable named `gemm_test` from the source files.
6.  Find and link the CUDAToolkit.

## Issues & Notes

- The initial setup required ensuring `CMake` could find the `CUDAToolkit`.
- The `CMakeLists.txt` was configured to set a default `CMAKE_CUDA_ARCHITECTURES` to prevent build failures on systems without a configured NVIDIA GPU or on newer CUDA toolkit versions that require it to be set. 