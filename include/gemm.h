#ifndef GEMM_H
#define GEMM_H

#include <vector>

// CPU-based matrix multiplication
void matrix_multiply_cpu(const std::vector<float>& A,
                         const std::vector<float>& B,
                         std::vector<float>& C,
                         int M, int K, int N);

// CUDA-based matrix multiplication
void matrix_multiply_cuda(const std::vector<float>& A,
                          const std::vector<float>& B,
                          std::vector<float>& C,
                          int M, int K, int N);

#endif // GEMM_H 