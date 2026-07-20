#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../src/elementwise.cuh"

static void kernel_add_f32(benchmark::State& state) {
    const int N = state.range(0);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMemset(d_a, 0, N * sizeof(float));
    cudaMemset(d_b, 0, N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        elementwise_add_f32_kernel::run(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    state.SetBytesProcessed(int64_t(N) * 3 * sizeof(float) * state.iterations());
}
BENCHMARK(kernel_add_f32)
    ->RangeMultiplier(4)
    ->Range(1024, 4 << 20)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

static void kernel_add_f32x4(benchmark::State& state) {
    const int N = state.range(0);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMemset(d_a, 0, N * sizeof(float));
    cudaMemset(d_b, 0, N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        elementwise_add_f32x4_kernel::run(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    state.SetBytesProcessed(int64_t(N) * 3 * sizeof(float) * state.iterations());
}
BENCHMARK(kernel_add_f32x4)
    ->RangeMultiplier(4)
    ->Range(1024, 4 << 20)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

static void kernel_add_f16(benchmark::State& state) {
    const int N = state.range(0);

    half *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(half));
    cudaMalloc(&d_b, N * sizeof(half));
    cudaMalloc(&d_c, N * sizeof(half));
    cudaMemset(d_a, 0, N * sizeof(half));
    cudaMemset(d_b, 0, N * sizeof(half));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        elementwise_add_f16_kernel::run(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    state.SetBytesProcessed(int64_t(N) * 3 * sizeof(half) * state.iterations());
}
BENCHMARK(kernel_add_f16)
    ->RangeMultiplier(4)
    ->Range(1024, 4 << 20)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

static void kernel_add_f16x2(benchmark::State& state) {
    const int N = state.range(0);

    half *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(half));
    cudaMalloc(&d_b, N * sizeof(half));
    cudaMalloc(&d_c, N * sizeof(half));
    cudaMemset(d_a, 0, N * sizeof(half));
    cudaMemset(d_b, 0, N * sizeof(half));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        elementwise_add_f16x2_kernel::run(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    state.SetBytesProcessed(int64_t(N) * 3 * sizeof(half) * state.iterations());
}
BENCHMARK(kernel_add_f16x2)
    ->RangeMultiplier(4)
    ->Range(1024, 4 << 20)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

static void kernel_add_f16x8_pack(benchmark::State& state) {
    const int N = state.range(0);

    half *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(half));
    cudaMalloc(&d_b, N * sizeof(half));
    cudaMalloc(&d_c, N * sizeof(half));
    cudaMemset(d_a, 0, N * sizeof(half));
    cudaMemset(d_b, 0, N * sizeof(half));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        elementwise_add_f16x8_kernel::run(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    state.SetBytesProcessed(int64_t(N) * 3 * sizeof(half) * state.iterations());
}
BENCHMARK(kernel_add_f16x8_pack)
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
