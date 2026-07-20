#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include <cstring>

#include "../src/utils.cuh"
#include "../src/hisogram.cu"

static void ref_histogram_i32(const int* a, int* y, int N, int num_bins) {
    for (int i = 0; i < num_bins; i++) y[i] = 0;
    for (int i = 0; i < N; i++) y[a[i]]++;
}

// ---------- histogram i32 ----------
TEST(HistogramI32, Correctness) {
    const int N = 4096;
    const int NUM_BINS = 64;
    std::vector<int> h_a(N), h_ref(NUM_BINS, 0), h_gpu(NUM_BINS, 0);

    for (int i = 0; i < N; i++) h_a[i] = rand() % NUM_BINS;

    ref_histogram_i32(h_a.data(), h_ref.data(), N, NUM_BINS);

    int *d_a, *d_y;
    ASSERT_EQ(cudaMalloc(&d_a, N * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, NUM_BINS * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), N * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_y, 0, NUM_BINS * sizeof(int)), cudaSuccess);

    {
        int block = 256;
        int grid = (N + block - 1) / block;
        histogram_i32_kernel<<<grid, block>>>(d_a, d_y, N);
    }
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_y, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < NUM_BINS; i++) {
        EXPECT_EQ(h_gpu[i], h_ref[i]) << "at bin " << i;
    }

    cudaFree(d_a); cudaFree(d_y);
}

TEST(HistogramI32, EmptyInput) {
    const int N = 0;
    const int NUM_BINS = 64;
    std::vector<int> h_ref(NUM_BINS, 0), h_gpu(NUM_BINS, 0);

    int *d_a, *d_y;
    ASSERT_EQ(cudaMalloc(&d_a, std::max(N, 1) * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, NUM_BINS * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_y, 0, NUM_BINS * sizeof(int)), cudaSuccess);

    {
        int block = 256;
        int grid = 0;
        histogram_i32_kernel<<<grid, block>>>(d_a, d_y, N);
    }
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_y, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < NUM_BINS; i++) {
        EXPECT_EQ(h_gpu[i], 0) << "at bin " << i;
    }

    cudaFree(d_a); cudaFree(d_y);
}

// ---------- histogram i32x4 ----------
TEST(HistogramI32x4, Correctness) {
    const int N = 4096;
    const int NUM_BINS = 64;
    std::vector<int> h_a(N), h_ref(NUM_BINS, 0), h_gpu(NUM_BINS, 0);

    for (int i = 0; i < N; i++) h_a[i] = rand() % NUM_BINS;

    ref_histogram_i32(h_a.data(), h_ref.data(), N, NUM_BINS);

    int *d_a, *d_y;
    ASSERT_EQ(cudaMalloc(&d_a, N * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, NUM_BINS * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), N * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_y, 0, NUM_BINS * sizeof(int)), cudaSuccess);

    {
        int block = 256;
        int grid = (N + 4 * block - 1) / (4 * block);
        histogram_i32x4_kernel<<<grid, block>>>(d_a, d_y, N);
    }
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_y, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < NUM_BINS; i++) {
        EXPECT_EQ(h_gpu[i], h_ref[i]) << "at bin " << i;
    }

    cudaFree(d_a); cudaFree(d_y);
}

TEST(HistogramI32x4, MultipleOf4) {
    const int N = 8;
    const int NUM_BINS = 16;
    std::vector<int> h_a(N), h_ref(NUM_BINS, 0), h_gpu(NUM_BINS, 0);

    for (int i = 0; i < N; i++) h_a[i] = rand() % NUM_BINS;

    ref_histogram_i32(h_a.data(), h_ref.data(), N, NUM_BINS);

    int *d_a, *d_y;
    ASSERT_EQ(cudaMalloc(&d_a, N * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, NUM_BINS * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), N * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_y, 0, NUM_BINS * sizeof(int)), cudaSuccess);

    {
        int block = 256;
        int grid = (N + 4 * block - 1) / (4 * block);
        histogram_i32x4_kernel<<<grid, block>>>(d_a, d_y, N);
    }
    ASSERT_EQ(cudaMemcpy(h_gpu.data(), d_y, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < NUM_BINS; i++) {
        EXPECT_EQ(h_gpu[i], h_ref[i]) << "at bin " << i;
    }

    cudaFree(d_a); cudaFree(d_y);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
