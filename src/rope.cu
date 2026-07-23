#include "utils.cuh"

#define BLOCK_SIZE 256
#define theta 10000.0f


// launch: <<<(N/256, 1, 1), (256, 1, 1)>>>  N = seq_len * head_dim/2
__global__ void rope_f32_kernel(float* x, float* out, int seq_len, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float x1 = x[idx * 2];
  float x2 = x[idx * 2 + 1];
  int token_pos = idx / N;
  int token_idx = idx % N;
  float exp_v = 1.0f / powf(theta, 2 * token_idx / (N * 2.0f));
  float sin_v = sinf(token_pos * exp_v);
  float cos_v = cosf(token_pos * exp_v);
  float out1 = x1 * cos_v - x2 * sin_v;
  float out2 = x1 * sin_v + x2 * cos_v;
  out[idx * 2] = out1;
  out[idx * 2 + 1] = out2;
}

// another index method of rope.
// launch: <<<(seq_len, 1, 1), (N=head_dim/2, 1, 1)>>>
__global__ void rope_f32_v2_kernel(float* x, float* out, const int seq_len, const int N) {
  int token_pos = blockIdx.x; // 0~(seq_len-1)
  int tid = threadIdx.x;
  int idx = 2 * tid;
  int offset = token_pos * N * 2;
  float x1 = x[offset + idx];
  float x2 = x[offset + idx + 1];
  float exp_v = 1.f / powf(theta, static_cast<float>(tid) / N) * token_pos;
  float sin_v = sinf(exp_v), cos_v = cosf(exp_v);
  out[offset + idx] = x1 * cos_v - x2 * sin_v;
  out[offset + idx + 1] = x1 * sin_v + x2 * cos_v;
}

// launch: <<<(N/256, 1, 1), (256, 1, 1)>>>  N = seq_len * head_dim/8
__global__ void rope_f32x4_pack_kernel(float* x, float* out, int seq_len,
  int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float4 x_v = FLOAT4(x[idx * 4]);
  int token_pos = idx / N;
  int token_idx = idx % N;
  float exp_f_v = 1.0f / powf(theta, 2 * token_idx * 2 / (N * 4.0f));
  float exp_s_v = 1.0f / powf(theta, 2 * (token_idx * 2 + 1) / (N * 4.0f));
  float sin_f_v = sinf(token_pos * exp_f_v);
  float cos_f_v = cosf(token_pos * exp_f_v);
  float sin_s_v = sinf(token_pos * exp_s_v);
  float cos_s_v = cosf(token_pos * exp_s_v);
  float4 out_v;
  out_v.x = x_v.x * cos_f_v - x_v.y * sin_f_v;
  out_v.y = x_v.x * sin_f_v + x_v.y * cos_f_v;
  out_v.z = x_v.z * cos_s_v - x_v.w * sin_s_v;
  out_v.w = x_v.z * sin_s_v + x_v.w * cos_s_v;
  FLOAT4(out[idx * 4]) = out_v;
}

// another index method of rope pack4.
// launch: <<<(seq_len, 1, 1), (N=head_dim/4, 1, 1)>>>
__global__ void rope_f32x4_pack_v2_kernel(float* x, float* out, const int seq_len, const int N) {
  const int token_pos = blockIdx.x;
  const int tid = threadIdx.x;
  const int idx = tid * 4;
  const int offset = token_pos * N * 4 + idx;
  float4 x1234 = FLOAT4(x[offset]);
  float exp_v = 1.f / powf(theta, static_cast<float>(2.0 * tid) / N) * token_pos;
  float cos_v = cosf(exp_v), sin_v = sinf(exp_v);
  out[offset] = x1234.x * cos_v - x1234.y * sin_v;
  out[offset + 1] = x1234.x * sin_v + x1234.y * cos_v;
  exp_v = 1.f / powf(theta, static_cast<float> (2.0 * tid + 1) / N) * token_pos;
  cos_v = cosf(exp_v), sin_v = sinf(exp_v);
  out[offset + 2] = x1234.z * cos_v - x1234.w * sin_v;
  out[offset + 3] = x1234.z * sin_v + x1234.w * cos_v;
}