#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "../src/softmax.cu"

// Reference softmax per token
static void ref_softmax_f32(const float* x, float* y, int N, int K) {
  for (int row = 0; row < N; ++row) {
    float max_val = -1e30f;
    for (int col = 0; col < K; ++col)
      max_val = fmaxf(max_val, x[row * K + col]);
    float sum = 0.0f;
    for (int col = 0; col < K; ++col) {
      y[row * K + col] = expf(x[row * K + col] - max_val);
      sum += y[row * K + col];
    }
    for (int col = 0; col < K; ++col)
      y[row * K + col] /= sum;
  }
}

// ---------- softmax f32 ----------
TEST(SoftmaxF32, K256) {
  const int N = 16, K = 256;
  std::vector<float> h_x(N * K), h_ref(N * K), h_gpu(N * K);
  for (int i = 0; i < N * K; ++i)
    h_x[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
  ref_softmax_f32(h_x.data(), h_ref.data(), N, K);

  float *d_x, *d_y;
  cudaMalloc(&d_x, N * K * sizeof(float));
  cudaMalloc(&d_y, N * K * sizeof(float));
  cudaMemcpy(d_x, h_x.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

  softmax_f32_per_token_kernel<256><<<N, 256>>>(d_x, d_y, N * K);
  cudaMemcpy(h_gpu.data(), d_y, N * K * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N * K; ++i)
    EXPECT_NEAR(h_gpu[i], h_ref[i], 1e-4f) << "at index " << i;
  cudaFree(d_x); cudaFree(d_y);
}

// ---------- softmax f32x4 ----------
TEST(SoftmaxF32x4, K256) {
  const int N = 16, K = 256;
  std::vector<float> h_x(N * K), h_ref(N * K), h_gpu(N * K);
  for (int i = 0; i < N * K; ++i)
    h_x[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
  ref_softmax_f32(h_x.data(), h_ref.data(), N, K);

  float *d_x, *d_y;
  cudaMalloc(&d_x, N * K * sizeof(float));
  cudaMalloc(&d_y, N * K * sizeof(float));
  cudaMemcpy(d_x, h_x.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

  softmax_f32x4_per_token_kernel<256 / 4><<<N, 256 / 4>>>(d_x, d_y, N * K);
  cudaMemcpy(h_gpu.data(), d_y, N * K * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N * K; ++i)
    EXPECT_NEAR(h_gpu[i], h_ref[i], 1e-4f) << "at index " << i;
  cudaFree(d_x); cudaFree(d_y);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
