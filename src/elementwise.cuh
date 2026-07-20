#pragma once
#ifndef ELEMENTWISE_CUH
#define ELEMENTWISE_CUH

#include "utils.cuh"

// ============================================================
// Template kernels for binary ops (a, b -> c)
// ============================================================

template<typename Op>
__global__ void binary_f32_kernel(float* a, float* b, float* c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = Op::apply(a[idx], b[idx]);
}

template<typename Op>
__global__ void binary_f32x4_kernel(float* a, float* b, float* c, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 ra = FLOAT4(a[idx]), rb = FLOAT4(b[idx]), rc;
    Apply4<Op>(ra, rb, rc);
    FLOAT4(c[idx]) = rc;
  }
}

template<typename Op>
__global__ void binary_f16_kernel(half* a, half* b, half* c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = Op::apply(a[idx], b[idx]);
}

template<typename Op>
__global__ void binary_f16x2_kernel(half* a, half* b, half* c, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 ra = HALF2(a[idx]), rb = HALF2(b[idx]), rc;
    Apply2<Op>(ra, rb, rc);
    HALF2(c[idx]) = rc;
  }
}

template<typename Op>
__global__ void binary_f16x8_kernel(half* a, half* b, half* c, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half pa[8], pb[8], pc[8];
  LDST128BITS(pa[0]) = LDST128BITS(a[idx]);
  LDST128BITS(pb[0]) = LDST128BITS(b[idx]);
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    half2 va = HALF2(pa[i]), vb = HALF2(pb[i]), vc;
    Apply2<Op>(va, vb, vc);
    HALF2(pc[i]) = vc;
  }
  if ((idx + 7) < N) {
    LDST128BITS(c[idx]) = LDST128BITS(pc[0]);
  }
  else {
    for (int i = 0; idx + i < N; ++i)
      c[idx + i] = Op::apply(a[idx + i], b[idx + i]);
  }
}

// ============================================================
// Macro: generate 5 kernel class structs for a binary Op.
// ============================================================

#define _BINARY_KERNEL_CLASS(name, Kernel, ElemType, GridExpr)                \
struct name {                                                                 \
  static void run(ElemType* a, ElemType* b, ElemType* c, int N,              \
                  dim3 grid, dim3 block, cudaStream_t stream = 0) {           \
    Kernel<<<grid, block, 0, stream>>>(a, b, c, N);                          \
  }                                                                           \
  static void run(ElemType* a, ElemType* b, ElemType* c, int N) {            \
    dim3 block(256);                                                          \
    dim3 grid(GridExpr);                                                      \
    Kernel<<<grid, block>>>(a, b, c, N);                                     \
  }                                                                           \
};

#define INSTANTIATE_BINARY_KERNELS(name, Op)                                  \
  _BINARY_KERNEL_CLASS(name##_f32_kernel,                                     \
    (binary_f32_kernel<Op>), float, (N + block.x - 1) / block.x)             \
  _BINARY_KERNEL_CLASS(name##_f32x4_kernel,                                   \
    (binary_f32x4_kernel<Op>), float, (N + 4 * block.x - 1) / (4 * block.x)) \
  _BINARY_KERNEL_CLASS(name##_f16_kernel,                                     \
    (binary_f16_kernel<Op>), half, (N + block.x - 1) / block.x)              \
  _BINARY_KERNEL_CLASS(name##_f16x2_kernel,                                   \
    (binary_f16x2_kernel<Op>), half, (N + 2 * block.x - 1) / (2 * block.x))  \
  _BINARY_KERNEL_CLASS(name##_f16x8_kernel,                                   \
    (binary_f16x8_kernel<Op>), half, (N + 8 * block.x - 1) / (8 * block.x))

// ============================================================
// Instantiate binary op kernels
// ============================================================

INSTANTIATE_BINARY_KERNELS(elementwise_add, AddOp)
INSTANTIATE_BINARY_KERNELS(elementwise_mul, MulOp)
INSTANTIATE_BINARY_KERNELS(elementwise_sub, SubOp)

#endif // ELEMENTWISE_CUH
