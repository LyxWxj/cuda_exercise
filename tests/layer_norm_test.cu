#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "../src/layer_norm.cu"

// Reference implementation
static void ref_layer_norm_f32(const float* x, float* y, float g, float b,
                               int N, int K) {
  const float epsilon = 1e-5f;
  for (int row = 0; row < N; ++row) {
    float sum = 0.0f;
    for (int col = 0; col < K; ++col) sum += x[row * K + col];
    float mean = sum / K;
    float var = 0.0f;
    for (int col = 0; col < K; ++col) {
      float d = x[row * K + col] - mean;
      var += d * d;
    }
    float inv_std = 1.0f / sqrtf(var / K + epsilon);
    for (int col = 0; col < K; ++col)
      y[row * K + col] = (x[row * K + col] - mean) * inv_std * g + b;
  }
}

// ---------- LayerNorm f32 ----------
TEST(LayerNormF32, K256) {
  const int N = 16, K = 256;
  std::vector<float> h_x(N * K), h_ref(N * K), h_gpu(N * K);
  for (int i = 0; i < N * K; ++i)
    h_x[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
  ref_layer_norm_f32(h_x.data(), h_ref.data(), 1.0f, 0.0f, N, K);

  float *d_x, *d_y;
  cudaMalloc(&d_x, N * K * sizeof(float));
  cudaMalloc(&d_y, N * K * sizeof(float));
  cudaMemcpy(d_x, h_x.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

  layer_norm_f32_kernel<256><<<N, 256>>>(d_x, d_y, 1.0f, 0.0f, N, K);
  cudaMemcpy(h_gpu.data(), d_y, N * K * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N * K; ++i)
    EXPECT_NEAR(h_gpu[i], h_ref[i], 1e-4f) << "at index " << i;
  cudaFree(d_x); cudaFree(d_y);
}

// ---------- LayerNorm f32x4 ----------
TEST(LayerNormF32x4, K256) {
  const int N = 16, K = 256;
  std::vector<float> h_x(N * K), h_ref(N * K), h_gpu(N * K);
  for (int i = 0; i < N * K; ++i)
    h_x[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
  ref_layer_norm_f32(h_x.data(), h_ref.data(), 1.0f, 0.0f, N, K);

  float *d_x, *d_y;
  cudaMalloc(&d_x, N * K * sizeof(float));
  cudaMalloc(&d_y, N * K * sizeof(float));
  cudaMemcpy(d_x, h_x.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

  layer_norm_f32x4_kernel<256 / 4><<<N, 256 / 4>>>(d_x, d_y, 1.0f, 0.0f, N, K);
  cudaMemcpy(h_gpu.data(), d_y, N * K * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N * K; ++i)
    EXPECT_NEAR(h_gpu[i], h_ref[i], 1e-4f) << "at index " << i;
  cudaFree(d_x); cudaFree(d_y);
}

// ---------- LayerNorm f16 (f32 accumulation) ----------
TEST(LayerNormF16F32, K256) {
  const int N = 16, K = 256;
  std::vector<half> h_x(N * K), h_gpu(N * K);
  std::vector<float> h_ref(N * K), h_xf(N * K);
  for (int i = 0; i < N * K; ++i) {
    h_xf[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    h_x[i] = __float2half(h_xf[i]);
  }
  ref_layer_norm_f32(h_xf.data(), h_ref.data(), 1.0f, 0.0f, N, K);

  half *d_x, *d_y;
  cudaMalloc(&d_x, N * K * sizeof(half));
  cudaMalloc(&d_y, N * K * sizeof(half));
  cudaMemcpy(d_x, h_x.data(), N * K * sizeof(half), cudaMemcpyHostToDevice);

  layer_norm_f16_f32_kernel<256><<<N, 256>>>(d_x, d_y, 1.0f, 0.0f, N, K);
  cudaMemcpy(h_gpu.data(), d_y, N * K * sizeof(half), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N * K; ++i)
    EXPECT_NEAR(__half2float(h_gpu[i]), h_ref[i], 5e-2f) << "at index " << i;
  cudaFree(d_x); cudaFree(d_y);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
