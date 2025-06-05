#include "gemm.h"

// Function to perform basic matrix multiplication (C = A * B)
void matrix_multiply_cpu(const std::vector<float>& A,
                         const std::vector<float>& B,
                         std::vector<float>& C,
                         int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0.0f;
            for (int l = 0; l < K; ++l) {
                C[i * N + j] += A[i * K + l] * B[l * N + j];
            }
        }
    }
} 