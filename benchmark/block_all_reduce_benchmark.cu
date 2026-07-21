#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../src/utils.cuh"
#include "../src/block_all_reduce.cu"

// ============================================================
// Helper: timed kernel run
// ============================================================
template<typename Fn>
static void BM_Run(benchmark::State& state, Fn&& fn) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (auto _ : state) {
        cudaEventRecord(start);
        fn();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        state.SetIterationTime(ms / 1000.0);
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

// ============================================================
// Fixture: all-reduce (1 input array → 1 scalar output)
// ============================================================
class AllReduceBench : public benchmark::Fixture {
public:
    float *d_a_f32{}, *d_y_f32{};
    half  *d_a_f16{};
    float *d_y_f16_acc{};
    int N{};

    void SetUp(const ::benchmark::State& state) override {
        N = state.range(0);
        cudaMalloc(&d_a_f32, N * sizeof(float));
        cudaMalloc(&d_y_f32, sizeof(float));
        cudaMalloc(&d_a_f16, N * sizeof(half));
        cudaMalloc(&d_y_f16_acc, sizeof(float));
        cudaMemset(d_a_f32, 0, N * sizeof(float));
        cudaMemset(d_a_f16, 0, N * sizeof(half));
        cudaMemset(d_y_f32, 0, sizeof(float));
        cudaMemset(d_y_f16_acc, 0, sizeof(float));
    }

    void TearDown(const ::benchmark::State&) override {
        cudaFree(d_a_f32); cudaFree(d_y_f32);
        cudaFree(d_a_f16); cudaFree(d_y_f16_acc);
    }
};

// ============================================================
// Fixture: dot product (2 input arrays → 1 scalar output)
// ============================================================
class DotProdBench : public benchmark::Fixture {
public:
    float *d_a_f32{}, *d_b_f32{}, *d_y_f32{};
    half  *d_a_f16{}, *d_b_f16{};
    float *d_y_f16_acc{};
    int N{};

    void SetUp(const ::benchmark::State& state) override {
        N = state.range(0);
        cudaMalloc(&d_a_f32, N * sizeof(float));
        cudaMalloc(&d_b_f32, N * sizeof(float));
        cudaMalloc(&d_y_f32, sizeof(float));
        cudaMalloc(&d_a_f16, N * sizeof(half));
        cudaMalloc(&d_b_f16, N * sizeof(half));
        cudaMalloc(&d_y_f16_acc, sizeof(float));
        cudaMemset(d_a_f32, 0, N * sizeof(float));
        cudaMemset(d_b_f32, 0, N * sizeof(float));
        cudaMemset(d_a_f16, 0, N * sizeof(half));
        cudaMemset(d_b_f16, 0, N * sizeof(half));
        cudaMemset(d_y_f32, 0, sizeof(float));
        cudaMemset(d_y_f16_acc, 0, sizeof(float));
    }

    void TearDown(const ::benchmark::State&) override {
        cudaFree(d_a_f32); cudaFree(d_b_f32); cudaFree(d_y_f32);
        cudaFree(d_a_f16); cudaFree(d_b_f16); cudaFree(d_y_f16_acc);
    }
};

// ============================================================
// All-Reduce benchmarks
// ============================================================

BENCHMARK_DEFINE_F(AllReduceBench, sum_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ block_all_reduce_sum_f32_f32_kernel::run(d_a_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(AllReduceBench, sum_f32)
    ->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(AllReduceBench, sum_f32x4)(benchmark::State& state) {
    BM_Run(state, [&]{ block_all_reduce_sum_f32x4_f32_kernel::run(d_a_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(AllReduceBench, sum_f32x4)
    ->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(AllReduceBench, sum_f16_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ block_all_reduce_sum_f16_f32_kernel::run(d_a_f16, d_y_f16_acc, N); });
    state.SetBytesProcessed(int64_t(N) * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(AllReduceBench, sum_f16_f32)
    ->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(AllReduceBench, sum_f16x2_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ block_all_reduce_sum_f16x2_f32_kernel::run(d_a_f16, d_y_f16_acc, N); });
    state.SetBytesProcessed(int64_t(N) * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(AllReduceBench, sum_f16x2_f32)
    ->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(AllReduceBench, sum_f16x8_pack_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ block_all_reduce_sum_f16x8_pack_f32_kernel::run(d_a_f16, d_y_f16_acc, N); });
    state.SetBytesProcessed(int64_t(N) * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(AllReduceBench, sum_f16x8_pack_f32)
    ->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

// ============================================================
// Dot Product benchmarks
// ============================================================

BENCHMARK_DEFINE_F(DotProdBench, dot_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ dot_prod_f32_f32_kernel::run(d_a_f32, d_b_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(DotProdBench, dot_f32)
    ->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(DotProdBench, dot_f32x4)(benchmark::State& state) {
    BM_Run(state, [&]{ dot_prod_f32x4_f32_kernel::run(d_a_f32, d_b_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(DotProdBench, dot_f32x4)
    ->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(DotProdBench, dot_f16_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ dot_prod_f16_f32_kernel::run(d_a_f16, d_b_f16, d_y_f16_acc, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(DotProdBench, dot_f16_f32)
    ->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(DotProdBench, dot_f16x2_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ dot_prod_f16x2_f32_kernel::run(d_a_f16, d_b_f16, d_y_f16_acc, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(DotProdBench, dot_f16x2_f32)
    ->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(DotProdBench, dot_f16x8_pack_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ dot_prod_f16x8_pack_f32_kernel::run(d_a_f16, d_b_f16, d_y_f16_acc, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(DotProdBench, dot_f16x8_pack_f32)
    ->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
