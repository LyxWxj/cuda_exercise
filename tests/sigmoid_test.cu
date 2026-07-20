#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "../src/utils.cuh"
#include "../src/sigmoid.cu"

static float ref_sigmoid_f32(float x) {
    float v = fminf(fmaxf(x, MIN_EXP_F32), MAX_EXP_F32);
    return 1.0f / (1.0f + expf(-v));
}

static void ref_sigmoid_f32_array(const float* x, float* y, int N) {
    for (int i = 0; i < N; i++) y[i] = ref_sigmoid_f32(x[i]);
}

static float ref_sigmoid_f16_as_float(float x) {
    half v = __hmin(__hmax(__float2half(x), MIN_EXP_F16), MAX_EXP_F16);
    float vf = __half2float(v);
    return 1.0f / (1.0f + expf(-vf));
}

static void ref_sigmoid_f16_array(const float* x, float* y, int N) {
    for (int i = 0; i < N; i++) y[i] = ref_sigmoid_f16_as_float(x[i]);
}

// ---------- sigmoid f32 ----------
TEST(SigmoidF32, Correctness) {
    const int N = 4096;
    std::vector<float> h_x(N), h_ref(N), h_gpu(N);
    for (int i = 0; i < N; i++) {
        h_x[i] = (static_cast<float>(rand()) / RAND_MAX * 200.0f - 100.0f);
    }
    ref_sigmoid_f32_array(h_x.data(), h_ref.data(), N);

    float *d_x, *d_y;
    ASSERT_EQ(cudaMalloc(&d_x, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    {
        int block = 256;
        int grid = (N + block - 1) / block;
        sigmoid_f32_kernel_class::run(d_x, d_y, N);
    }
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_gpu[i], h_ref[i], 1e-5f) << "at index " << i << " x=" << h_x[i];
    }

    cudaFree(d_x); cudaFree(d_y);
}

// ---------- sigmoid f32x4 ----------
TEST(SigmoidF32x4, Correctness) {
    const int N = 4096;
    std::vector<float> h_x(N), h_ref(N), h_gpu(N);
    for (int i = 0; i < N; i++) {
        h_x[i] = (static_cast<float>(rand()) / RAND_MAX * 200.0f - 100.0f);
    }
    ref_sigmoid_f32_array(h_x.data(), h_ref.data(), N);

    float *d_x, *d_y;
    ASSERT_EQ(cudaMalloc(&d_x, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    {
        int block = 256;
        int grid = (N + 4 * block - 1) / (4 * block);
        sigmoid_f32x4_kernel_class::run(d_x, d_y, N);
    }
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_gpu[i], h_ref[i], 1e-5f) << "at index " << i << " x=" << h_x[i];
    }

    cudaFree(d_x); cudaFree(d_y);
}

TEST(SigmoidF32x4, MisalignedN) {
    const int N = 7;
    std::vector<float> h_x(N), h_ref(N), h_gpu(N);
    for (int i = 0; i < N; i++) {
        h_x[i] = (static_cast<float>(rand()) / RAND_MAX * 200.0f - 100.0f);
    }
    ref_sigmoid_f32_array(h_x.data(), h_ref.data(), N);

    float *d_x, *d_y;
    ASSERT_EQ(cudaMalloc(&d_x, (N + 4) * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, (N + 4) * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    {
        int block = 256;
        int grid = (N + 4 * block - 1) / (4 * block);
        sigmoid_f32x4_kernel_class::run(d_x, d_y, N);
    }
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(h_gpu[i], h_ref[i], 1e-5f) << "at index " << i << " x=" << h_x[i];
    }

    cudaFree(d_x); cudaFree(d_y);
}

// ---------- sigmoid f16 ----------
TEST(SigmoidF16, Correctness) {
    const int N = 4096;
    std::vector<half> h_x(N);
    std::vector<float> h_xf(N), h_ref(N);
    std::vector<half> h_gpu(N);
    for (int i = 0; i < N; i++) {
        h_xf[i] = (static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f);
        h_x[i] = __float2half(h_xf[i]);
    }
    ref_sigmoid_f16_array(h_xf.data(), h_ref.data(), N);

    half *d_x, *d_y;
    ASSERT_EQ(cudaMalloc(&d_x, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

    {
        int block = 256;
        int grid = (N + block - 1) / block;
        sigmoid_f16_kernel_class::run(d_x, d_y, N);
    }
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        float gpu_val = __half2float(h_gpu[i]);
        EXPECT_NEAR(gpu_val, h_ref[i], 1e-2f) << "at index " << i << " x=" << h_xf[i];
    }

    cudaFree(d_x); cudaFree(d_y);
}

// ---------- sigmoid f16x2 ----------
TEST(SigmoidF16x2, Correctness) {
    const int N = 4096;
    std::vector<half> h_x(N);
    std::vector<float> h_xf(N), h_ref(N);
    std::vector<half> h_gpu(N);
    for (int i = 0; i < N; i++) {
        h_xf[i] = (static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f);
        h_x[i] = __float2half(h_xf[i]);
    }
    ref_sigmoid_f16_array(h_xf.data(), h_ref.data(), N);

    half *d_x, *d_y;
    ASSERT_EQ(cudaMalloc(&d_x, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

    {
        int block = 256;
        int grid = (N + 2 * block - 1) / (2 * block);
        sigmoid_f16x2_kernel_class::run(d_x, d_y, N);
    }
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        float gpu_val = __half2float(h_gpu[i]);
        EXPECT_NEAR(gpu_val, h_ref[i], 1e-2f) << "at index " << i << " x=" << h_xf[i];
    }

    cudaFree(d_x); cudaFree(d_y);
}

TEST(SigmoidF16x2, MisalignedN) {
    const int N = 3;
    std::vector<half> h_x(N);
    std::vector<float> h_xf(N), h_ref(N);
    std::vector<half> h_gpu(N);
    for (int i = 0; i < N; i++) {
        h_xf[i] = (static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f);
        h_x[i] = __float2half(h_xf[i]);
    }
    ref_sigmoid_f16_array(h_xf.data(), h_ref.data(), N);

    half *d_x, *d_y;
    ASSERT_EQ(cudaMalloc(&d_x, (N + 2) * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, (N + 2) * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

    {
        int block = 256;
        int grid = (N + 2 * block - 1) / (2 * block);
        sigmoid_f16x2_kernel_class::run(d_x, d_y, N);
    }
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        float gpu_val = __half2float(h_gpu[i]);
        EXPECT_NEAR(gpu_val, h_ref[i], 1e-2f) << "at index " << i << " x=" << h_xf[i];
    }

    cudaFree(d_x); cudaFree(d_y);
}

// ---------- sigmoid f16x8 ----------
TEST(SigmoidF16x8, Correctness) {
    const int N = 4096;
    std::vector<half> h_x(N);
    std::vector<float> h_xf(N), h_ref(N);
    std::vector<half> h_gpu(N);
    for (int i = 0; i < N; i++) {
        h_xf[i] = (static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f);
        h_x[i] = __float2half(h_xf[i]);
    }
    ref_sigmoid_f16_array(h_xf.data(), h_ref.data(), N);

    half *d_x, *d_y;
    ASSERT_EQ(cudaMalloc(&d_x, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

    {
        int block = 256;
        sigmoid_f16x8_kernel_class::run(d_x, d_y, N);
    }
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        float gpu_val = __half2float(h_gpu[i]);
        EXPECT_NEAR(gpu_val, h_ref[i], 1e-2f) << "at index " << i << " x=" << h_xf[i];
    }

    cudaFree(d_x); cudaFree(d_y);
}

// ---------- sigmoid f16x8 pack ----------
TEST(SigmoidF16x8Pack, Correctness) {
    const int N = 4096;
    std::vector<half> h_x(N);
    std::vector<float> h_xf(N), h_ref(N);
    std::vector<half> h_gpu(N);
    for (int i = 0; i < N; i++) {
        h_xf[i] = (static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f);
        h_x[i] = __float2half(h_xf[i]);
    }
    ref_sigmoid_f16_array(h_xf.data(), h_ref.data(), N);

    half *d_x, *d_y;
    ASSERT_EQ(cudaMalloc(&d_x, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

    {
        int block = 256;
        sigmoid_f16x8_pack_kernel_class::run(d_x, d_y, N);
    }
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        float gpu_val = __half2float(h_gpu[i]);
        EXPECT_NEAR(gpu_val, h_ref[i], 1e-2f) << "at index " << i << " x=" << h_xf[i];
    }

    cudaFree(d_x); cudaFree(d_y);
}

TEST(SigmoidF16x8Pack, MultipleOf8) {
    const int N = 8;
    std::vector<half> h_x(N);
    std::vector<float> h_xf(N), h_ref(N);
    std::vector<half> h_gpu(N);
    for (int i = 0; i < N; i++) {
        h_xf[i] = (static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f);
        h_x[i] = __float2half(h_xf[i]);
    }
    ref_sigmoid_f16_array(h_xf.data(), h_ref.data(), N);

    half *d_x, *d_y;
    ASSERT_EQ(cudaMalloc(&d_x, (N + 8) * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, (N + 8) * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, h_x.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

    {
        int block = 256;
        sigmoid_f16x8_pack_kernel_class::run(d_x, d_y, N);
    }
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_y, N * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        float gpu_val = __half2float(h_gpu[i]);
        EXPECT_NEAR(gpu_val, h_ref[i], 1e-2f) << "at index " << i << " x=" << h_xf[i];
    }

    cudaFree(d_x); cudaFree(d_y);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
