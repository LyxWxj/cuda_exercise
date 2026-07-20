#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "../src/utils.cuh"
#include "../src/hisogram.cu"

static void kernel_hist_i32(benchmark::State& state) {
    const int N = state.range(0);
    const int NUM_BINS = 1024;
    const int block = 256;
    const int grid = (N + block - 1) / block;

    int *d_a, *d_y;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_y, NUM_BINS * sizeof(int));
    cudaMemset(d_a, 0, N * sizeof(int));
    cudaMemset(d_y, 0, NUM_BINS * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        histogram_i32_kernel<<<grid, block>>>(d_a, d_y, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_y);

    state.SetBytesProcessed(int64_t(N) * sizeof(int) * state.iterations());
}
BENCHMARK(kernel_hist_i32)
    ->RangeMultiplier(4)
    ->Range(1024, 4 << 20)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

static void kernel_hist_i32x4(benchmark::State& state) {
    const int N = state.range(0);
    const int NUM_BINS = 1024;
    const int block = 256;
    const int grid = (N + 4 * block - 1) / (4 * block);

    int *d_a, *d_y;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_y, NUM_BINS * sizeof(int));
    cudaMemset(d_a, 0, N * sizeof(int));
    cudaMemset(d_y, 0, NUM_BINS * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        histogram_i32x4_kernel<<<grid, block>>>(d_a, d_y, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_y);

    state.SetBytesProcessed(int64_t(N) * sizeof(int) * state.iterations());
}
BENCHMARK(kernel_hist_i32x4)
    ->RangeMultiplier(4)
    ->Range(1024, 4 << 20)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
