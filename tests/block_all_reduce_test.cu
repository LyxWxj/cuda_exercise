#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "../src/utils.cuh"
#include "../src/block_all_reduce.cu"

// ============================================================
// Reference implementations
// ============================================================
static float ref_sum_f32(const float* a, int N) {
    double s = 0;
    for (int i = 0; i < N; i++) s += a[i];
    return static_cast<float>(s);
}

static float ref_sum_f16(const half* a, int N) {
    double s = 0;
    for (int i = 0; i < N; i++) s += __half2float(a[i]);
    return static_cast<float>(s);
}

static float ref_dot_f32(const float* a, const float* b, int N) {
    double s = 0;
    for (int i = 0; i < N; i++) s += (double)a[i] * (double)b[i];
    return static_cast<float>(s);
}

static float ref_dot_f16(const half* a, const half* b, int N) {
    double s = 0;
    for (int i = 0; i < N; i++) s += (double)__half2float(a[i]) * (double)__half2float(b[i]);
    return static_cast<float>(s);
}

static float randf() {
    return static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
}

// ============================================================
// All-Reduce Sum Tests
// ============================================================

TEST(AllReduceF32, Scalar) {
    const int N = 4096;
    std::vector<float> h_a(N);
    for (int i = 0; i < N; i++) h_a[i] = randf();
    float ref = ref_sum_f32(h_a.data(), N);

    float *d_a, *d_y;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_y, sizeof(float));
    cudaMemset(d_y, 0, sizeof(float));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    block_all_reduce_sum_f32_f32_kernel::run(d_a, d_y, N);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float gpu;
    cudaMemcpy(&gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(gpu, ref, fabsf(ref) * 1e-4f + 1e-3f);
    cudaFree(d_a); cudaFree(d_y);
}

TEST(AllReduceF32, Vec4) {
    const int N = 4096;
    std::vector<float> h_a(N);
    for (int i = 0; i < N; i++) h_a[i] = randf();
    float ref = ref_sum_f32(h_a.data(), N);

    float *d_a, *d_y;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_y, sizeof(float));
    cudaMemset(d_y, 0, sizeof(float));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    block_all_reduce_sum_f32x4_f32_kernel::run(d_a, d_y, N);

    float gpu;
    cudaMemcpy(&gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(gpu, ref, fabsf(ref) * 1e-4f + 1e-3f);
    cudaFree(d_a); cudaFree(d_y);
}

TEST(AllReduceF16, Scalar) {
    const int N = 4096;
    std::vector<half> h_a(N);
    for (int i = 0; i < N; i++) h_a[i] = __float2half(randf());
    float ref = ref_sum_f16(h_a.data(), N);

    half *d_a; float *d_y;
    cudaMalloc(&d_a, N * sizeof(half));
    cudaMalloc(&d_y, sizeof(float));
    cudaMemset(d_y, 0, sizeof(float));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    block_all_reduce_sum_f16_f32_kernel::run(d_a, d_y, N);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float gpu;
    cudaMemcpy(&gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(gpu, ref, fabsf(ref) * 1e-2f + 1.0f);
    cudaFree(d_a); cudaFree(d_y);
}

TEST(AllReduceF16, Vec2) {
    const int N = 4096;
    std::vector<half> h_a(N);
    for (int i = 0; i < N; i++) h_a[i] = __float2half(randf());
    float ref = ref_sum_f16(h_a.data(), N);

    half *d_a; float *d_y;
    cudaMalloc(&d_a, N * sizeof(half));
    cudaMalloc(&d_y, sizeof(float));
    cudaMemset(d_y, 0, sizeof(float));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    block_all_reduce_sum_f16x2_f32_kernel::run(d_a, d_y, N);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float gpu;
    cudaMemcpy(&gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(gpu, ref, fabsf(ref) * 1e-2f + 1.0f);
    cudaFree(d_a); cudaFree(d_y);
}

TEST(AllReduceF16, Vec8Pack) {
    const int N = 4096;
    std::vector<half> h_a(N);
    for (int i = 0; i < N; i++) h_a[i] = __float2half(randf());
    float ref = ref_sum_f16(h_a.data(), N);

    half *d_a; float *d_y;
    cudaMalloc(&d_a, N * sizeof(half));
    cudaMalloc(&d_y, sizeof(float));
    cudaMemset(d_y, 0, sizeof(float));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    block_all_reduce_sum_f16x8_pack_f32_kernel::run(d_a, d_y, N);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float gpu;
    cudaMemcpy(&gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(gpu, ref, fabsf(ref) * 1e-2f + 1.0f);
    cudaFree(d_a); cudaFree(d_y);
}

// ============================================================
// Dot Product Tests
// ============================================================

TEST(DotProdF32, Scalar) {
    const int N = 4096;
    std::vector<float> h_a(N), h_b(N);
    for (int i = 0; i < N; i++) { h_a[i] = randf(); h_b[i] = randf(); }
    float ref = ref_dot_f32(h_a.data(), h_b.data(), N);

    float *d_a, *d_b, *d_y;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_y, sizeof(float));
    cudaMemset(d_y, 0, sizeof(float));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dot_prod_f32_f32_kernel::run(d_a, d_b, d_y, N);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float gpu;
    cudaMemcpy(&gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(gpu, ref, fabsf(ref) * 1e-4f + 1e-3f);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
}

TEST(DotProdF32, Vec4) {
    const int N = 4096;
    std::vector<float> h_a(N), h_b(N);
    for (int i = 0; i < N; i++) { h_a[i] = randf(); h_b[i] = randf(); }
    float ref = ref_dot_f32(h_a.data(), h_b.data(), N);

    float *d_a, *d_b, *d_y;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_y, sizeof(float));
    cudaMemset(d_y, 0, sizeof(float));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dot_prod_f32x4_f32_kernel::run(d_a, d_b, d_y, N);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float gpu;
    cudaMemcpy(&gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(gpu, ref, fabsf(ref) * 1e-4f + 1e-3f);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
}

TEST(DotProdF16, Scalar) {
    const int N = 4096;
    std::vector<half> h_a(N), h_b(N);
    for (int i = 0; i < N; i++) { h_a[i] = __float2half(randf()); h_b[i] = __float2half(randf()); }
    float ref = ref_dot_f16(h_a.data(), h_b.data(), N);

    half *d_a, *d_b; float *d_y;
    cudaMalloc(&d_a, N * sizeof(half));
    cudaMalloc(&d_b, N * sizeof(half));
    cudaMalloc(&d_y, sizeof(float));
    cudaMemset(d_y, 0, sizeof(float));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    dot_prod_f16_f32_kernel::run(d_a, d_b, d_y, N);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float gpu;
    cudaMemcpy(&gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(gpu, ref, fabsf(ref) * 1e-2f + 1.0f);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
}

TEST(DotProdF16, Vec2) {
    const int N = 4096;
    std::vector<half> h_a(N), h_b(N);
    for (int i = 0; i < N; i++) { h_a[i] = __float2half(randf()); h_b[i] = __float2half(randf()); }
    float ref = ref_dot_f16(h_a.data(), h_b.data(), N);

    half *d_a, *d_b; float *d_y;
    cudaMalloc(&d_a, N * sizeof(half));
    cudaMalloc(&d_b, N * sizeof(half));
    cudaMalloc(&d_y, sizeof(float));
    cudaMemset(d_y, 0, sizeof(float));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    dot_prod_f16x2_f32_kernel::run(d_a, d_b, d_y, N);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float gpu;
    cudaMemcpy(&gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(gpu, ref, fabsf(ref) * 1e-2f + 1.0f);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
}

TEST(DotProdF16, Vec8Pack) {
    const int N = 4096;
    std::vector<half> h_a(N), h_b(N);
    for (int i = 0; i < N; i++) { h_a[i] = __float2half(randf()); h_b[i] = __float2half(randf()); }
    float ref = ref_dot_f16(h_a.data(), h_b.data(), N);

    half *d_a, *d_b; float *d_y;
    cudaMalloc(&d_a, N * sizeof(half));
    cudaMalloc(&d_b, N * sizeof(half));
    cudaMalloc(&d_y, sizeof(float));
    cudaMemset(d_y, 0, sizeof(float));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    dot_prod_f16x8_pack_f32_kernel::run(d_a, d_b, d_y, N);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float gpu;
    cudaMemcpy(&gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NEAR(gpu, ref, fabsf(ref) * 1e-2f + 1.0f);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
