#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "../src/utils.cuh"
#include "../src/unary_kernels.cuh"

static float ref_elu(float x) {
    return x > 0.0f ? x : 1.0f * (expf(x) - 1.0f);
}

TEST(EluF32, Correctness) {
    const int N = 4096;
    std::vector<float> h_x(N), h_ref(N), h_gpu(N);
    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f;
        h_ref[i] = ref_elu(h_x[i]);
    }
    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256, grid = (N + block - 1) / block;
    elu_f32_kernel::run(d_x, d_y, N);
    cudaMemcpy(h_gpu.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_gpu[i], h_ref[i], 1e-5f) << "at index " << i;
    }
    cudaFree(d_x); cudaFree(d_y);
}

TEST(EluF32x4, Correctness) {
    const int N = 4096;
    std::vector<float> h_x(N), h_ref(N), h_gpu(N);
    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f;
        h_ref[i] = ref_elu(h_x[i]);
    }
    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256, grid = (N + 4 * block - 1) / (4 * block);
    elu_f32x4_kernel::run(d_x, d_y, N);
    cudaMemcpy(h_gpu.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_gpu[i], h_ref[i], 1e-5f) << "at index " << i;
    }
    cudaFree(d_x); cudaFree(d_y);
}

TEST(EluF16, Correctness) {
    const int N = 4096;
    std::vector<half> h_x(N), h_gpu(N);
    std::vector<float> h_ref(N);
    for (int i = 0; i < N; i++) {
        float v = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
        h_x[i] = __float2half(v);
        h_ref[i] = ref_elu(v);
    }
    half *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(half));
    cudaMalloc(&d_y, N * sizeof(half));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    int block = 256, grid = (N + block - 1) / block;
    elu_f16_kernel::run(d_x, d_y, N);
    cudaMemcpy(h_gpu.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(__half2float(h_gpu[i]), h_ref[i], 5e-2f) << "at index " << i;
    }
    cudaFree(d_x); cudaFree(d_y);
}

TEST(EluF16x2, Correctness) {
    const int N = 4096;
    std::vector<half> h_x(N), h_gpu(N);
    std::vector<float> h_ref(N);
    for (int i = 0; i < N; i++) {
        float v = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
        h_x[i] = __float2half(v);
        h_ref[i] = ref_elu(v);
    }
    half *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(half));
    cudaMalloc(&d_y, N * sizeof(half));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    int block = 256, grid = (N + 2 * block - 1) / (2 * block);
    elu_f16x2_kernel::run(d_x, d_y, N);
    cudaMemcpy(h_gpu.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(__half2float(h_gpu[i]), h_ref[i], 5e-2f) << "at index " << i;
    }
    cudaFree(d_x); cudaFree(d_y);
}

TEST(EluF16x8, Correctness) {
    const int N = 4096;
    std::vector<half> h_x(N), h_gpu(N);
    std::vector<float> h_ref(N);
    for (int i = 0; i < N; i++) {
        float v = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
        h_x[i] = __float2half(v);
        h_ref[i] = ref_elu(v);
    }
    half *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(half));
    cudaMalloc(&d_y, N * sizeof(half));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    int block = 256, grid = (N + 8 * block - 1) / (8 * block);
    elu_f16x8_kernel::run(d_x, d_y, N);
    cudaMemcpy(h_gpu.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(__half2float(h_gpu[i]), h_ref[i], 5e-2f) << "at index " << i;
    }
    cudaFree(d_x); cudaFree(d_y);
}

TEST(EluF16x8Pack, Correctness) {
    const int N = 4096;
    std::vector<half> h_x(N), h_gpu(N);
    std::vector<float> h_ref(N);
    for (int i = 0; i < N; i++) {
        float v = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
        h_x[i] = __float2half(v);
        h_ref[i] = ref_elu(v);
    }
    half *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(half));
    cudaMalloc(&d_y, N * sizeof(half));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    int block = 256, grid = (N + 8 * block - 1) / (8 * block);
    elu_f16x8_pack_kernel::run(d_x, d_y, N);
    cudaMemcpy(h_gpu.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(__half2float(h_gpu[i]), h_ref[i], 5e-2f) << "at index " << i;
    }
    cudaFree(d_x); cudaFree(d_y);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
