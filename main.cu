#include "cutlass/gemm/device/gemm.h"
#include <cuda_runtime.h>
#include <iostream>

// 定义GEMM操作的参数类型
using Gemm = cutlass::gemm::device::Gemm<
  float, cutlass::layout::ColumnMajor,  // A矩阵：float，列优先
  float, cutlass::layout::ColumnMajor,  // B矩阵：float，列优先
  float, cutlass::layout::ColumnMajor   // C矩阵：float，列优先
>;

// 在你的内核函数中
void run_gemm() {
  // 定义问题尺寸
  int M = 512, N = 512, K = 512;

  // 在GPU设备内存中分配指针
  float* d_A, * d_B, * d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));

  float alpha = 1.0f, beta = 0.0f;

  // 构建参数
  Gemm::Arguments args(
    { M, N, K },                      // 问题尺寸
    { d_A, K },                       // A矩阵的布局和主维度 (leading dimension)
    { d_B, N },                       // B矩阵的布局和主维度
    { d_C, N },                       // C矩阵的布局和主维度
    { d_C, N },                       // D矩阵（结果）的布局和主维度
    { alpha, beta }                   // 标量乘数
  );

  // 实例化并运行GEMM内核
  Gemm gemm_op;
  gemm_op(args);

  // 释放内存
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
int main() {
  run_gemm();
  return 0;
}