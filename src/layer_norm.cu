#include "utils.cuh"

// ============================================================
// NormTraits: abstracts load/accumulate/store for norm kernels
// ============================================================
template<typename T> struct NormTraits;

template<> struct NormTraits<float> {
  static constexpr int kVecWidth = 1;
  __device__ static float load(float* p, int, int) { return *p; }
  __device__ static void store(float* p, int, float v, float, float, float inv_std, float g, float b) {
    *p = v * inv_std * g + b;
  }
};

template<> struct NormTraits<float4> {
  static constexpr int kVecWidth = 4;
  float4 reg;
  __device__ float load(float* p, int idx, int N) {
    reg = (idx + 3 < N) ? FLOAT4(p[idx]) : make_float4(0,0,0,0);
    return reg.x + reg.y + reg.z + reg.w;
  }
  __device__ void store(float* p, int idx, float mean, float, float inv_std, float g, float b) {
    float4 r;
    r.x = (reg.x - mean) * inv_std * g + b;
    r.y = (reg.y - mean) * inv_std * g + b;
    r.z = (reg.z - mean) * inv_std * g + b;
    r.w = (reg.w - mean) * inv_std * g + b;
    if (idx + 3 < (int)(idx + 4)) FLOAT4(p[idx]) = r; // caller handles bounds
  }
};

template<> struct NormTraits<half> {
  static constexpr int kVecWidth = 1;
  __device__ static float load(half* p, int, int) { return __half2float(*p); }
  __device__ static void store(half* p, int, float v, float mean, float inv_std, float g, float b) {
    *p = __float2half((v - mean) * inv_std * g + b);
  }
};

// ============================================================
// Generic norm kernel — handles both LayerNorm and RMSNorm
//
// NormOp must provide:
//   NormOp(float* s_stat, int tid, float value, float K)
//     — computes stat (mean for LN, nothing for RMS), writes to smem
//   float NormOp::normalize(float value, ...)
//     — applies normalization
// ============================================================

// LayerNorm: mean + variance
template<const int NUM_THREADS>
// launch: <<<(N, 1, 1), (K, 1, 1)>>>
__global__ void layer_norm_f32_kernel(float* x, float* y, float g, float b, int N, int K) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  const float epsilon = 1e-5f;
  __shared__ float s_mean, s_variance;

  float value = (idx < N * K) ? x[idx] : 0.0f;
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  if (tid == 0) s_mean = sum / (float)K;
  __syncthreads();

  float variance = (value - s_mean) * (value - s_mean);
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();

  if (idx < N * K) y[idx] = ((value - s_mean) * s_variance) * g + b;
}

template<const int NUM_THREADS>
// launch: <<<(N, 1, 1), (K/4, 1, 1)>>>
__global__ void layer_norm_f32x4_kernel(float* x, float* y, float g, float b, int N, int K) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * blockDim.x + tid) * 4;
  const float epsilon = 1e-5f;
  __shared__ float s_mean, s_variance;

  float4 rx = FLOAT4(x[idx]);
  float value = rx.x + rx.y + rx.z + rx.w;
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  if (tid == 0) s_mean = sum / (float)K;
  __syncthreads();

  float4 rh; rh.x = rx.x - s_mean; rh.y = rx.y - s_mean;
  rh.z = rx.z - s_mean; rh.w = rx.w - s_mean;
  float variance = rh.x*rh.x + rh.y*rh.y + rh.z*rh.z + rh.w*rh.w;
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();

  float4 ry;
  ry.x = rh.x * s_variance * g + b; ry.y = rh.y * s_variance * g + b;
  ry.z = rh.z * s_variance * g + b; ry.w = rh.w * s_variance * g + b;
  if (idx < N * K) FLOAT4(y[idx]) = ry;
}

template<const int NUM_THREADS>
// launch: <<<(N, 1, 1), (K, 1, 1)>>>
__global__ void layer_norm_f16_f32_kernel(half* x, half* y, float g, float b, int N, int K) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  const float epsilon = 1e-5f;
  __shared__ float s_mean, s_variance;

  float value = (idx < N * K) ? __half2float(x[idx]) : 0.0f;
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  if (tid == 0) s_mean = sum / (float)K;
  __syncthreads();

  float variance = (value - s_mean) * (value - s_mean);
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();

  if (idx < N * K) y[idx] = __float2half(((value - s_mean) * s_variance) * g + b);
}

template<const int NUM_THREADS>
// launch: <<<(N, 1, 1), (K/8, 1, 1)>>>
__global__ void layer_norm_f16x8_pack_f32_kernel(half* x, half* y, float g, float b, int N, int K) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * blockDim.x + tid) * 8;
  const float epsilon = 1e-5f;
  __shared__ float s_mean, s_variance;
  half px[8], py[8];
  LDST128BITS(px[0]) = LDST128BITS(x[idx]);

  float value = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) value += __half2float(px[i]);
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  if (tid == 0) s_mean = sum / (float)K;
  __syncthreads();

  float variance = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) { float d = __half2float(px[i]) - s_mean; variance += d * d; }
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 8; ++i) py[i] = __float2half(((__half2float(px[i]) - s_mean) * s_variance) * g + b);
  if ((idx + 7) < N * K) LDST128BITS(y[idx]) = LDST128BITS(py[0]);
}
