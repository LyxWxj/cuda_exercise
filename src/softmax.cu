#include "utils.cuh"

struct __align__(8) MD {
  float m{}, d{};
};

// Warp-level reduce for (max, denominator) pair — online softmax
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
  constexpr unsigned int mask = 0xffffffff;
  MD temp;
#pragma unroll
  for (int stride = kWarpSize >> 1; stride; stride >>= 1) {
    temp.m = __shfl_xor_sync(mask, value.m, stride);
    temp.d = __shfl_xor_sync(mask, value.d, stride);
    bool if_greater = (value.m > temp.m);
    MD greater = if_greater ? value : temp;
    MD smaller = if_greater ? temp : value;
    value.d = greater.d + smaller.d * __expf(smaller.m - greater.m);
    value.m = greater.m;
  }
  return value;
}

// ============================================================
// Softmax per token: y[i] = exp(x[i]) / sum(exp(x[j]))
// x: (seq_len, head_dim), y: (seq_len, head_dim)
// launch: <<<(seq_len, 1, 1), (head_dim, 1, 1)>>>
// ============================================================

// F32 scalar
template<const int NUM_THREADS = 256>
// launch: <<<(N/K, 1, 1), (K, 1, 1)>>>
__global__ void softmax_f32_per_token_kernel(float* x, float* y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  float exp_val = (idx < N) ? expf(x[idx]) : 0.f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  if (idx < N) y[idx] = exp_val / exp_sum;
}

// F32x4 vectorized
template<const int NUM_THREADS = 256 / 4>
// launch: <<<(N/K, 1, 1), (K/4, 1, 1)>>>
__global__ void softmax_f32x4_per_token_kernel(float* x, float* y, int N) {
  const int tid = threadIdx.x;
  const int idx = 4 * (blockIdx.x * blockDim.x + tid);
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_exp;
  reg_exp.x = (idx + 0 < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx + 2 < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx + 3 < N) ? expf(reg_x.w) : 0.0f;
  float exp_val = reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  if (idx + 3 < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / exp_sum;
    reg_y.y = reg_exp.y / exp_sum;
    reg_y.z = reg_exp.z / exp_sum;
    reg_y.w = reg_exp.w / exp_sum;
    FLOAT4(y[idx]) = reg_y;
  }
}
