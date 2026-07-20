#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../src/utils.cuh"
#include "../src/sigmoid.cu"

class SigmoidBench : public benchmark::Fixture {
public:
    float *d_x_f32{}, *d_y_f32{};
    half *d_x_f16{}, *d_y_f16{};
    int N{};

    void SetUp(const ::benchmark::State& state) override {
        N = state.range(0);
        cudaMalloc(&d_x_f32, N * sizeof(float));
        cudaMalloc(&d_y_f32, N * sizeof(float));
        cudaMalloc(&d_x_f16, N * sizeof(half));
        cudaMalloc(&d_y_f16, N * sizeof(half));
        cudaMemset(d_x_f32, 0, N * sizeof(float));
        cudaMemset(d_x_f16, 0, N * sizeof(half));
    }

    void TearDown(const ::benchmark::State&) override {
        cudaFree(d_x_f32); cudaFree(d_y_f32);
        cudaFree(d_x_f16); cudaFree(d_y_f16);
    }
};

BENCHMARK_DEFINE_F(SigmoidBench, F32)(benchmark::State& state) {
    const int block = 256;
    const int grid = (N + block - 1) / block;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        sigmoid_f32_kernel_class::run(d_x_f32, d_y_f32, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }

    cudaEventDestroy(start); cudaEventDestroy(stop);
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(SigmoidBench, F32)
    ->RangeMultiplier(4)
    ->Range(1024, 4 << 20)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(SigmoidBench, F32x4)(benchmark::State& state) {
    const int block = 256;
    const int grid = (N + 4 * block - 1) / (4 * block);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        sigmoid_f32x4_kernel_class::run(d_x_f32, d_y_f32, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }

    cudaEventDestroy(start); cudaEventDestroy(stop);
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(SigmoidBench, F32x4)
    ->RangeMultiplier(4)
    ->Range(1024, 4 << 20)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(SigmoidBench, F16)(benchmark::State& state) {
    const int block = 256;
    const int grid = (N + block - 1) / block;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        sigmoid_f16_kernel_class::run(d_x_f16, d_y_f16, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }

    cudaEventDestroy(start); cudaEventDestroy(stop);
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(SigmoidBench, F16)
    ->RangeMultiplier(4)
    ->Range(1024, 4 << 20)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(SigmoidBench, F16x2)(benchmark::State& state) {
    const int block = 256;
    const int grid = (N + 2 * block - 1) / (2 * block);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        sigmoid_f16x2_kernel_class::run(d_x_f16, d_y_f16, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }

    cudaEventDestroy(start); cudaEventDestroy(stop);
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(SigmoidBench, F16x2)
    ->RangeMultiplier(4)
    ->Range(1024, 4 << 20)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(SigmoidBench, F16x8)(benchmark::State& state) {
    const int block = 256;
    const int grid = (N + 8 * block - 1) / (8 * block);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        sigmoid_f16x8_kernel_class::run(d_x_f16, d_y_f16, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }

    cudaEventDestroy(start); cudaEventDestroy(stop);
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(SigmoidBench, F16x8)
    ->RangeMultiplier(4)
    ->Range(1024, 4 << 20)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(SigmoidBench, F16x8Pack)(benchmark::State& state) {
    const int block = 256;
    const int grid = (N + 8 * block - 1) / (8 * block);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        sigmoid_f16x8_pack_kernel_class::run(d_x_f16, d_y_f16, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }

    cudaEventDestroy(start); cudaEventDestroy(stop);
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(SigmoidBench, F16x8Pack)
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
