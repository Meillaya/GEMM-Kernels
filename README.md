# High-Performance Matrix Multiplication (GEMM) Kernels

## Goal
The goal of this project is to implement optimized GEMM (General Matrix Multiply) kernels in C++/CUDA for CPUs and GPUs. This project serves as a hands-on way to learn and apply various performance optimization techniques.

### Core Learning Objectives
- **Low-level optimization:** Implementing techniques like loop unrolling, SIMD, and cache blocking.
- **Parallel programming:** Gaining experience with OpenMP for CPUs and CUDA for GPUs.
- **Performance benchmarking:** Systematically measuring performance improvements (FLOPs, memory bandwidth).

### Advanced Goals
- Compare different parallelization strategies (e.g., tiling, warp-level optimizations in CUDA).
- Integrate the final optimized kernel into PyTorch or TensorFlow as a custom operator.

## Project Status
Currently, the project contains naive implementations for both CPU and GPU to serve as a baseline for future optimizations.

## Summary

The primary goal of this project is to explore the performance differences between a standard, single-threaded CPU implementation and a parallel GPU implementation of matrix multiplication.

The project is structured as follows:
- `src/main.cpp`: Main program to run and verify the different implementations.
- `src/gemm_cpu.cpp`: A naive CPU implementation.
- `src/gemm_cuda.cu`: A naive CUDA implementation.
- `include/gemm.h`: Header files for the GEMM functions.
- `CMakeLists.txt`: Build script for `CMake`.

For more detailed documentation on the project setup, tools, and code, please see `docs/project_documentation.md`.

## How to Build and Run

### Prerequisites

- A C++17 compliant compiler (e.g., GCC, Clang, MSVC).
- The NVIDIA CUDA Toolkit (version 11.0 or newer recommended).
- `CMake` (version 3.10 or newer).

### Build Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd GEMM-Kernels
    ```

2.  **Run CMake to configure the project (using Ninja):**
    I recommend using the [Ninja](https://ninja-build.org/) build system for faster builds. Create a `build` directory and configure the project with CMake, specifying Ninja as the generator.

    ```bash
    cmake -S . -B build -G Ninja
    ```
    *Note: If the CUDA toolkit is installed in a non-standard location, you may need to specify its root directory by setting the `CUDAToolkit_ROOT` environment variable.*

    **Alternative without Ninja:**
    If you don't have Ninja, you can use the default makefile generator:
    ```bash
    mkdir build
    cd build
    cmake ..
    ```

3.  **Compile the project:**
    If you configured with Ninja, run:
    ```bash
    cmake --build build
    ```
    If you used the default generator, run from inside the `build` directory:
    ```bash
    cmake --build .
    ```

### Running the Application

After a successful build, an executable named `gemm_test` (or `gemm_test.exe` on Windows) will be located in the `build` directory.

To run it, simply execute the following command from the project root:
```bash
./build/gemm_test
```

The program will:
1. Initialize two matrices, A and B, with random values.
2. Compute their product on the CPU.
3. Compute their product on the GPU.
4. Compare the results from the CPU and GPU to ensure correctness.
5. Print a success or failure message.
