#include "unary_kernels.cuh"
#include <type_traits>

// ============================================================
// Sigmoid kernel implementations: clip + sigmoid = 1/(1+exp(-x))
// ============================================================

// launch: <<<(N/256, 1, 1), (256, 1, 1)>>>
__global__ void sigmoid_f32_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = x[idx];
    clip(v);
    y[idx] = 1.0f / (1.0f + expf(-v));
  }
}

// launch: <<<(N/1024, 1, 1), (256, 1, 1)>>>
__global__ void sigmoid_f32x4_kernel(float* x, float* y, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  float4 reg_x = FLOAT4(x[idx]);
  CLIP4(reg_x);
  float4 reg_y;
  SIGMOID4(reg_x, reg_y);
  if (idx < N) FLOAT4(y[idx]) = reg_y;
}

// launch: <<<(N/256, 1, 1), (256, 1, 1)>>>
__global__ void sigmoid_f16_kernel(half* x, half* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const half f = __float2half(1.0f);
  if (idx < N) {
    half v = x[idx];
    clip(v);
    y[idx] = f / (f + hexp(-v));
  }
}

// launch: <<<(N/512, 1, 1), (256, 1, 1)>>>
__global__ void sigmoid_f16x2_kernel(half* x, half* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const half f = __float2half(1.0f);
  half2 reg_x = HALF2(x[idx]);
  half2 reg_y;
  CLIP2(reg_x);
  SIGMOID2(reg_x, reg_y);
  if ((idx + 0) < N) HALF2(y[idx]) = reg_y;
}

// launch: <<<(N/2048, 1, 1), (256, 1, 1)>>>
__global__ void sigmoid_f16x8_kernel(half* x, half* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  const half f = __float2half(1.0f);
  half2 reg_x_0 = HALF2(x[idx + 0]); CLIP2(reg_x_0);
  half2 reg_x_1 = HALF2(x[idx + 2]); CLIP2(reg_x_1);
  half2 reg_x_2 = HALF2(x[idx + 4]); CLIP2(reg_x_2);
  half2 reg_x_3 = HALF2(x[idx + 6]); CLIP2(reg_x_3);
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  SIGMOID2(reg_x_0, reg_y_0);
  SIGMOID2(reg_x_1, reg_y_1);
  SIGMOID2(reg_x_2, reg_y_2);
  SIGMOID2(reg_x_3, reg_y_3);
  if ((idx + 0) < N) HALF2(y[idx + 0]) = reg_y_0;
  if ((idx + 2) < N) HALF2(y[idx + 2]) = reg_y_1;
  if ((idx + 4) < N) HALF2(y[idx + 4]) = reg_y_2;
  if ((idx + 6) < N) HALF2(y[idx + 6]) = reg_y_3;
}

// launch: <<<(N/2048, 1, 1), (256, 1, 1)>>>
__global__ void sigmoid_f16x8_pack_kernel(half* x, half* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  const half f = __float2half(1.0f);
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    half v = __hmin(__hmax(pack_x[i], MIN_EXP_F16), MAX_EXP_F16);
    pack_y[i] = f / (f + hexp(-v));
  }
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}
