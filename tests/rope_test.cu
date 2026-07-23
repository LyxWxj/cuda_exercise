#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "../src/rope.cu"

// Reference RoPE implementation
static void ref_rope(const float* x, float* out, int seq_len, int head_dim) {
  const int N = head_dim / 2;  // number of pairs per token
  for (int pos = 0; pos < seq_len; ++pos) {
    for (int i = 0; i < N; ++i) {
      float x1 = x[pos * head_dim + i * 2];
      float x2 = x[pos * head_dim + i * 2 + 1];
      float freq = 1.0f / powf(10000.0f, 2.0f * i / head_dim);
      float angle = pos * freq;
      float sin_v = sinf(angle), cos_v = cosf(angle);
      out[pos * head_dim + i * 2]     = x1 * cos_v - x2 * sin_v;
      out[pos * head_dim + i * 2 + 1] = x1 * sin_v + x2 * cos_v;
    }
  }
}

// ---------- rope_f32 (flat index) ----------
TEST(RoPE_F32, Basic) {
  const int seq_len = 4, head_dim = 64;
  const int N = head_dim / 2;  // pairs per token
  const int total = seq_len * N;
  std::vector<float> h_x(total * 2), h_ref(total * 2), h_gpu(total * 2);
  for (int i = 0; i < total * 2; ++i)
    h_x[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
  ref_rope(h_x.data(), h_ref.data(), seq_len, head_dim);

  float *d_x, *d_out;
  cudaMalloc(&d_x, total * 2 * sizeof(float));
  cudaMalloc(&d_out, total * 2 * sizeof(float));
  cudaMemcpy(d_x, h_x.data(), total * 2 * sizeof(float), cudaMemcpyHostToDevice);

  int block = 256, grid = (total + block - 1) / block;
  rope_f32_kernel<<<grid, block>>>(d_x, d_out, seq_len, N);
  cudaMemcpy(h_gpu.data(), d_out, total * 2 * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < total * 2; ++i)
    EXPECT_NEAR(h_gpu[i], h_ref[i], 1e-4f) << "at index " << i;
  cudaFree(d_x); cudaFree(d_out);
}

// ---------- rope_f32_v2 (block per token) ----------
TEST(RoPE_F32_V2, Basic) {
  const int seq_len = 4, head_dim = 64;
  const int N = head_dim / 2;
  const int total = seq_len * head_dim;
  std::vector<float> h_x(total), h_ref(total), h_gpu(total);
  for (int i = 0; i < total; ++i)
    h_x[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
  ref_rope(h_x.data(), h_ref.data(), seq_len, head_dim);

  float *d_x, *d_out;
  cudaMalloc(&d_x, total * sizeof(float));
  cudaMalloc(&d_out, total * sizeof(float));
  cudaMemcpy(d_x, h_x.data(), total * sizeof(float), cudaMemcpyHostToDevice);

  rope_f32_v2_kernel<<<seq_len, N>>>(d_x, d_out, seq_len, N);
  cudaMemcpy(h_gpu.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < total; ++i)
    EXPECT_NEAR(h_gpu[i], h_ref[i], 1e-4f) << "at index " << i;
  cudaFree(d_x); cudaFree(d_out);
}

// ---------- rope_f32x4_pack (float4, flat index) ----------
TEST(RoPE_F32x4, Basic) {
  const int seq_len = 4, head_dim = 64;
  const int N = head_dim / 4;  // float4 groups per token (each thread handles 1 float4 = 4 elements)
  const int total_threads = seq_len * N;
  const int total = seq_len * head_dim;
  std::vector<float> h_x(total), h_ref(total), h_gpu(total);
  for (int i = 0; i < total; ++i)
    h_x[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
  ref_rope(h_x.data(), h_ref.data(), seq_len, head_dim);

  float *d_x, *d_out;
  cudaMalloc(&d_x, total * sizeof(float));
  cudaMalloc(&d_out, total * sizeof(float));
  cudaMemcpy(d_x, h_x.data(), total * sizeof(float), cudaMemcpyHostToDevice);

  int block = 256, grid = (total_threads + block - 1) / block;
  rope_f32x4_pack_kernel<<<grid, block>>>(d_x, d_out, seq_len, N);
  cudaMemcpy(h_gpu.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < total; ++i)
    EXPECT_NEAR(h_gpu[i], h_ref[i], 1e-4f) << "at index " << i;
  cudaFree(d_x); cudaFree(d_out);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
