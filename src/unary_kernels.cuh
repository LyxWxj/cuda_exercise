#pragma once
#ifndef UNARY_KERNELS_CUH
#define UNARY_KERNELS_CUH

#include "utils.cuh"

// ============================================================
// Template kernels for unary ops (x -> y)
// ============================================================

template<typename Op>
__global__ void unary_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = Op::apply(x[idx]);
}

template<typename Op>
__global__ void unary_f32x4_kernel(float* x, float* y, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 rx = FLOAT4(x[idx]); float4 ry;
    Apply4<Op>(rx, ry);
    FLOAT4(y[idx]) = ry;
  }
}

template<typename Op>
__global__ void unary_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = Op::apply(x[idx]);
}

template<typename Op>
__global__ void unary_f16x2_kernel(half* x, half* y, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 rx = HALF2(x[idx]); half2 ry;
    Apply2<Op>(rx, ry);
    HALF2(y[idx]) = ry;
  }
}

template<typename Op>
__global__ void unary_f16x8_kernel(half* x, half* y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half2 rx0 = HALF2(x[idx+0]), rx1 = HALF2(x[idx+2]);
  half2 rx2 = HALF2(x[idx+4]), rx3 = HALF2(x[idx+6]);
  half2 ry0, ry1, ry2, ry3;
  Apply2<Op>(rx0, ry0); Apply2<Op>(rx1, ry1);
  Apply2<Op>(rx2, ry2); Apply2<Op>(rx3, ry3);
  if ((idx+0)<N) HALF2(y[idx+0]) = ry0;
  if ((idx+2)<N) HALF2(y[idx+2]) = ry1;
  if ((idx+4)<N) HALF2(y[idx+4]) = ry2;
  if ((idx+6)<N) HALF2(y[idx+6]) = ry3;
}

template<typename Op>
__global__ void unary_f16x8_pack_kernel(half* x, half* y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half px[8], py[8];
  LDST128BITS(px[0]) = LDST128BITS(x[idx]);
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    half2 vx = HALF2(px[i]); half2 vy;
    Apply2<Op>(vx, vy);
    HALF2(py[i]) = vy;
  }
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(py[0]);
  }
  else {
    for (int i = 0; idx + i < N; ++i) y[idx + i] = Op::apply(x[idx + i]);
  }
}

// ============================================================
// Macro: generate 6 kernel class structs for a unary Op.
// launch: f32/f16    → <<<(N/256, 1, 1), (256, 1, 1)>>>
//         f32x4      → <<<(N/1024, 1, 1), (256, 1, 1)>>>
//         f16x2      → <<<(N/512, 1, 1), (256, 1, 1)>>>
//         f16x8      → <<<(N/2048, 1, 1), (256, 1, 1)>>>
//         f16x8_pack → <<<(N/2048, 1, 1), (256, 1, 1)>>>
// ============================================================

#define _UNARY_KERNEL_CLASS(name, Kernel, ElemType, GridExpr)                 \
struct name {                                                                 \
  static void run(ElemType* x, ElemType* y, int N,                           \
                  dim3 grid, dim3 block, cudaStream_t stream = 0) {           \
    Kernel<<<grid, block, 0, stream>>>(x, y, N);                             \
  }                                                                           \
  static void run(ElemType* x, ElemType* y, int N) {                         \
    dim3 block(256);                                                          \
    dim3 grid(GridExpr);                                                      \
    Kernel<<<grid, block>>>(x, y, N);                                        \
  }                                                                           \
};

#define INSTANTIATE_UNARY_KERNELS(name, Op)                                   \
  _UNARY_KERNEL_CLASS(name##_f32_kernel,                                      \
    (unary_f32_kernel<Op>), float, (N + block.x - 1) / block.x)              \
  _UNARY_KERNEL_CLASS(name##_f32x4_kernel,                                    \
    (unary_f32x4_kernel<Op>), float, (N + 4 * block.x - 1) / (4 * block.x))  \
  _UNARY_KERNEL_CLASS(name##_f16_kernel,                                      \
    (unary_f16_kernel<Op>), half, (N + block.x - 1) / block.x)               \
  _UNARY_KERNEL_CLASS(name##_f16x2_kernel,                                    \
    (unary_f16x2_kernel<Op>), half, (N + 2 * block.x - 1) / (2 * block.x))   \
  _UNARY_KERNEL_CLASS(name##_f16x8_kernel,                                    \
    (unary_f16x8_kernel<Op>), half, (N + 8 * block.x - 1) / (8 * block.x))   \
  _UNARY_KERNEL_CLASS(name##_f16x8_pack_kernel,                               \
    (unary_f16x8_pack_kernel<Op>), half, (N + 8 * block.x - 1) / (8 * block.x))

// ============================================================
// Instantiate activation kernels
// ============================================================

INSTANTIATE_UNARY_KERNELS(relu, ReluOp)
INSTANTIATE_UNARY_KERNELS(elu, EluOp)
INSTANTIATE_UNARY_KERNELS(gelu, GeluOp)
INSTANTIATE_UNARY_KERNELS(swish, SwishOp)
INSTANTIATE_UNARY_KERNELS(hardswish, HardswishOp)
INSTANTIATE_UNARY_KERNELS(hardshrink, HardshrinkOp)

// ============================================================
// Standalone sigmoid kernels (clip + sigmoid, not templated)
// ============================================================

__global__ void sigmoid_f32_kernel(float* x, float* y, int N);
__global__ void sigmoid_f32x4_kernel(float* x, float* y, int N);
__global__ void sigmoid_f16_kernel(half* x, half* y, int N);
__global__ void sigmoid_f16x2_kernel(half* x, half* y, int N);
__global__ void sigmoid_f16x8_kernel(half* x, half* y, int N);
__global__ void sigmoid_f16x8_pack_kernel(half* x, half* y, int N);

_UNARY_KERNEL_CLASS(sigmoid_f32_kernel_class, sigmoid_f32_kernel,
  float, (N + block.x - 1) / block.x)
_UNARY_KERNEL_CLASS(sigmoid_f32x4_kernel_class, sigmoid_f32x4_kernel,
  float, (N + 4 * block.x - 1) / (4 * block.x))
_UNARY_KERNEL_CLASS(sigmoid_f16_kernel_class, sigmoid_f16_kernel,
  half, (N + block.x - 1) / block.x)
_UNARY_KERNEL_CLASS(sigmoid_f16x2_kernel_class, sigmoid_f16x2_kernel,
  half, (N + 2 * block.x - 1) / (2 * block.x))
_UNARY_KERNEL_CLASS(sigmoid_f16x8_kernel_class, sigmoid_f16x8_kernel,
  half, (N + 8 * block.x - 1) / (8 * block.x))
_UNARY_KERNEL_CLASS(sigmoid_f16x8_pack_kernel_class, sigmoid_f16x8_pack_kernel,
  half, (N + 8 * block.x - 1) / (8 * block.x))

#endif // UNARY_KERNELS_CUH
