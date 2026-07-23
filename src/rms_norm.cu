#include "utils.cuh"

// ============================================================
// RMS Norm: y = x / sqrt(mean(x^2) + eps) * g
// x: (N, K), N = batch*seq_len, K = hidden_size
// Grid: (N), Block: (K or K/vec_width)
// ============================================================

// F32 scalar
template<const int NUM_THREADS = 256>
// launch: <<<(N, 1, 1), (K, 1, 1)>>>
__global__ void rms_norm_f32_kernel(float* x, float* y, float g, int N, int K) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  const float epsilon = 1e-5f;
  __shared__ float s_variance;

  float value = (idx < N * K) ? x[idx] : 0.0f;
  float variance = value * value;
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();

  if (idx < N * K) y[idx] = value * s_variance * g;
}

// F32x4
template<const int NUM_THREADS = 256 / 4>
// launch: <<<(N, 1, 1), (K/4, 1, 1)>>>
__global__ void rms_norm_f32x4_kernel(float* x, float* y, float g, int N, int K) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * blockDim.x + tid) * 4;
  const float epsilon = 1e-5f;
  __shared__ float s_variance;

  float4 rx = FLOAT4(x[idx]);
  float variance = rx.x * rx.x + rx.y * rx.y + rx.z * rx.z + rx.w * rx.w;
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();

  float4 ry;
  ry.x = rx.x * s_variance * g; ry.y = rx.y * s_variance * g;
  ry.z = rx.z * s_variance * g; ry.w = rx.w * s_variance * g;
  if (idx < N * K) FLOAT4(y[idx]) = ry;
}

// F16 scalar (f32 accumulation)
template<const int NUM_THREADS = 256>
// launch: <<<(N, 1, 1), (K, 1, 1)>>>
__global__ void rms_norm_f16_f32_kernel(half* x, half* y, float g, int N, int K) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  const float epsilon = 1e-5f;
  __shared__ float s_variance;

  float value = (idx < N * K) ? __half2float(x[idx]) : 0.0f;
  float variance = value * value;
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();

  if (idx < N * K) y[idx] = __float2half(value * s_variance * g);
}

// F16x8 pack (128-bit load, f32 accumulation)
template<const int NUM_THREADS = 256>
// launch: <<<(N, 1, 1), (K/8, 1, 1)>>>
__global__ void rms_norm_f16x8_pack_f32_kernel(half* x, half* y, float g, int N, int K) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * blockDim.x + tid) * 8;
  const float epsilon = 1e-5f;
  __shared__ float s_variance;
  half px[8], py[8];
  LDST128BITS(px[0]) = LDST128BITS(x[idx]);

  float variance = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) { float v = __half2float(px[i]); variance += v * v; }
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    float2 v2 = __half22float2(HALF2(px[i]));
    HALF2(py[i]) = __float22half2_rn(make_float2(v2.x * s_variance * g, v2.y * s_variance * g));
  }
  if ((idx + 7) < N * K) LDST128BITS(y[idx]) = LDST128BITS(py[0]);
}
