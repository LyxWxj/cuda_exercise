#include "utils.cuh"

#include "cublas_v2.h"

#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error("cuBLAS call failed");                         \
    }                                                                          \
  } while (0)

void cublas_sgemm(float* A, float* B, float* C, size_t M, size_t N, size_t K) {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  static float alpha = 1.0;
  static float beta = 0.0;

  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
    B, CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C,
    CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT));
  CHECK_CUBLAS(cublasDestroy(handle));
}

void cublas_sgemm_tf32(float* A, float* B, float* C, size_t M, size_t N,
  size_t K) {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

  static float alpha = 1.0;
  static float beta = 0.0;

  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
    B, CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C,
    CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUBLAS(cublasDestroy(handle));
}
