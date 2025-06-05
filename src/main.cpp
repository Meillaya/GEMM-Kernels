#include <iostream>
#include <vector>
#include "gemm.h"

int main() {
    // Example dimensions
    int M = 2; // Rows of A, Rows of C
    int K = 3; // Cols of A, Rows of B
    int N = 2; // Cols of B, Cols of C

    // Initialize matrices
    std::vector<float> A = {1.0f, 2.0f, 3.0f,
                            4.0f, 5.0f, 6.0f}; // M x K

    std::vector<float> B = {7.0f, 8.0f,
                            9.0f, 10.0f,
                            11.0f, 12.0f}; // K x N

    std::vector<float> C_cpu(M * N); // M x N result matrix

    std::cout << "Performing CPU matrix multiplication..." << std::endl;
    matrix_multiply_cpu(A, B, C_cpu, M, K, N);

    // Print CPU result
    std::cout << "CPU Result matrix C (" << M << "x" << N << "):" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C_cpu[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Perform CUDA matrix multiplication
    std::vector<float> C_gpu(M * N);
    std::cout << "\nPerforming CUDA matrix multiplication..." << std::endl;
    matrix_multiply_cuda(A, B, C_gpu, M, K, N);

    // Print CUDA result
    std::cout << "CUDA Result matrix C (" << M << "x" << N << "):" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C_gpu[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}