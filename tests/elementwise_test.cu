#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "../src/elementwise.cuh"

static void ref_add_f32(const float* a, const float* b, float* c, int N) {
    for (int i = 0; i < N; i++) c[i] = a[i] + b[i];
}

static void ref_add_f16(const half* a, const half* b, half* c, int N) {
    for (int i = 0; i < N; i++) c[i] = __hadd(a[i], b[i]);
}

template <typename T>
static void init_random(T* data, int N, float scale = 1.0f);

template <>
void init_random<float>(float* data, int N, float scale) {
    for (int i = 0; i < N; i++) data[i] = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * scale;
}

template <>
void init_random<half>(half* data, int N, float scale) {
    for (int i = 0; i < N; i++) data[i] = __float2half((static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * scale);
}

static void run_add_f32_kernel(const float* d_a, const float* d_b, float* d_c, int N) {
    elementwise_add_f32_kernel::run(const_cast<float*>(d_a), const_cast<float*>(d_b), d_c, N);
}

static void run_add_f32x4_kernel(const float* d_a, const float* d_b, float* d_c, int N) {
    elementwise_add_f32x4_kernel::run(const_cast<float*>(d_a), const_cast<float*>(d_b), d_c, N);
}

static void run_add_f16_kernel(const half* d_a, const half* d_b, half* d_c, int N) {
    elementwise_add_f16_kernel::run(const_cast<half*>(d_a), const_cast<half*>(d_b), d_c, N);
}

static void run_add_f16x2_kernel(const half* d_a, const half* d_b, half* d_c, int N) {
    elementwise_add_f16x2_kernel::run(const_cast<half*>(d_a), const_cast<half*>(d_b), d_c, N);
}

static void run_add_f16x8_pack_kernel(const half* d_a, const half* d_b, half* d_c, int N) {
    elementwise_add_f16x8_kernel::run(const_cast<half*>(d_a), const_cast<half*>(d_b), d_c, N);
}

// ---------- f32 ----------
TEST(ElementwiseAddF32, Correctness) {
    const int N = 4096;
    std::vector<float> h_a(N), h_b(N), h_ref(N), h_gpu(N);
    init_random(h_a.data(), N);
    init_random(h_b.data(), N);
    ref_add_f32(h_a.data(), h_b.data(), h_ref.data(), N);

    float *d_a, *d_b, *d_c;
    ASSERT_EQ(cudaMalloc(&d_a, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    run_add_f32_kernel(d_a, d_b, d_c, N);
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_gpu[i], h_ref[i]) << "at index " << i;
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

TEST(ElementwiseAddF32, SmallN) {
    const int N = 1;
    std::vector<float> h_a(N, 3.0f), h_b(N, 5.0f), h_ref(N, 8.0f), h_gpu(N, 0.0f);

    float *d_a, *d_b, *d_c;
    ASSERT_EQ(cudaMalloc(&d_a, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    run_add_f32_kernel(d_a, d_b, d_c, N);
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_FLOAT_EQ(h_gpu[0], 8.0f);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

// ---------- f32x4 ----------
TEST(ElementwiseAddF32x4, Correctness) {
    const int N = 4096;
    std::vector<float> h_a(N), h_b(N), h_ref(N), h_gpu(N);
    init_random(h_a.data(), N);
    init_random(h_b.data(), N);
    ref_add_f32(h_a.data(), h_b.data(), h_ref.data(), N);

    float *d_a, *d_b, *d_c;
    ASSERT_EQ(cudaMalloc(&d_a, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    run_add_f32x4_kernel(d_a, d_b, d_c, N);
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_gpu[i], h_ref[i]) << "at index " << i;
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

TEST(ElementwiseAddF32x4, MisalignedN) {
    const int N = 7;
    std::vector<float> h_a(N), h_b(N), h_ref(N), h_gpu(N);
    init_random(h_a.data(), N);
    init_random(h_b.data(), N);
    ref_add_f32(h_a.data(), h_b.data(), h_ref.data(), N);

    float *d_a, *d_b, *d_c;
    ASSERT_EQ(cudaMalloc(&d_a, (N + 4) * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, (N + 4) * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, (N + 4) * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    run_add_f32x4_kernel(d_a, d_b, d_c, N);
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_gpu[i], h_ref[i]) << "at index " << i;
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

// ---------- f16 ----------
TEST(ElementwiseAddF16, Correctness) {
    const int N = 4096;
    std::vector<half> h_a(N), h_b(N), h_ref(N), h_gpu(N);
    init_random(h_a.data(), N, 2.0f);
    init_random(h_b.data(), N, 2.0f);
    ref_add_f16(h_a.data(), h_b.data(), h_ref.data(), N);

    half *d_a, *d_b, *d_c;
    ASSERT_EQ(cudaMalloc(&d_a, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

    run_add_f16_kernel(d_a, d_b, d_c, N);
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_c, N * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(__half2float(h_gpu[i]), __half2float(h_ref[i])) << "at index " << i;
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

// ---------- f16x2 ----------
TEST(ElementwiseAddF16x2, Correctness) {
    const int N = 4096;
    std::vector<half> h_a(N), h_b(N), h_ref(N), h_gpu(N);
    init_random(h_a.data(), N, 2.0f);
    init_random(h_b.data(), N, 2.0f);
    ref_add_f16(h_a.data(), h_b.data(), h_ref.data(), N);

    half *d_a, *d_b, *d_c;
    ASSERT_EQ(cudaMalloc(&d_a, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

    run_add_f16x2_kernel(d_a, d_b, d_c, N);
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_c, N * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(__half2float(h_gpu[i]), __half2float(h_ref[i])) << "at index " << i;
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

TEST(ElementwiseAddF16x2, MisalignedN) {
    const int N = 3;
    std::vector<half> h_a(N), h_b(N), h_ref(N), h_gpu(N);
    init_random(h_a.data(), N, 2.0f);
    init_random(h_b.data(), N, 2.0f);
    ref_add_f16(h_a.data(), h_b.data(), h_ref.data(), N);

    half *d_a, *d_b, *d_c;
    ASSERT_EQ(cudaMalloc(&d_a, (N + 2) * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, (N + 2) * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, (N + 2) * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

    run_add_f16x2_kernel(d_a, d_b, d_c, N);
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_c, N * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(__half2float(h_gpu[i]), __half2float(h_ref[i])) << "at index " << i;
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

// ---------- f16x8 pack ----------
TEST(ElementwiseAddF16x8Pack, Correctness) {
    const int N = 4096;
    std::vector<half> h_a(N), h_b(N), h_ref(N), h_gpu(N);
    init_random(h_a.data(), N, 2.0f);
    init_random(h_b.data(), N, 2.0f);
    ref_add_f16(h_a.data(), h_b.data(), h_ref.data(), N);

    half *d_a, *d_b, *d_c;
    ASSERT_EQ(cudaMalloc(&d_a, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, N * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

    run_add_f16x8_pack_kernel(d_a, d_b, d_c, N);
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_c, N * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(__half2float(h_gpu[i]), __half2float(h_ref[i])) << "at index " << i;
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

TEST(ElementwiseAddF16x8Pack, SmallN) {
    const int N = 5;
    std::vector<half> h_a(N), h_b(N), h_ref(N), h_gpu(N);
    init_random(h_a.data(), N, 2.0f);
    init_random(h_b.data(), N, 2.0f);
    ref_add_f16(h_a.data(), h_b.data(), h_ref.data(), N);

    half *d_a, *d_b, *d_c;
    ASSERT_EQ(cudaMalloc(&d_a, (N + 8) * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, (N + 8) * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, (N + 8) * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), N * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);

    run_add_f16x8_pack_kernel(d_a, d_b, d_c, N);
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_c, N * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(__half2float(h_gpu[i]), __half2float(h_ref[i])) << "at index " << i;
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
