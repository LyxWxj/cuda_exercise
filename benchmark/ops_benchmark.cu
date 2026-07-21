#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../src/elementwise.cuh"
#include "../src/unary_kernels.cuh"
#include "../src/sigmoid.cu"

// ============================================================
// Fixture: unary ops (x -> y)
// ============================================================
class UnaryBench : public benchmark::Fixture {
public:
    float *d_x_f32{}, *d_y_f32{};
    half  *d_x_f16{}, *d_y_f16{};
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

// ============================================================
// Fixture: binary ops (a, b -> c)
// ============================================================
class BinaryBench : public benchmark::Fixture {
public:
    float *d_a_f32{}, *d_b_f32{}, *d_c_f32{};
    half  *d_a_f16{}, *d_b_f16{}, *d_c_f16{};
    int N{};

    void SetUp(const ::benchmark::State& state) override {
        N = state.range(0);
        cudaMalloc(&d_a_f32, N * sizeof(float));
        cudaMalloc(&d_b_f32, N * sizeof(float));
        cudaMalloc(&d_c_f32, N * sizeof(float));
        cudaMalloc(&d_a_f16, N * sizeof(half));
        cudaMalloc(&d_b_f16, N * sizeof(half));
        cudaMalloc(&d_c_f16, N * sizeof(half));
        cudaMemset(d_a_f32, 0, N * sizeof(float));
        cudaMemset(d_b_f32, 0, N * sizeof(float));
        cudaMemset(d_a_f16, 0, N * sizeof(half));
        cudaMemset(d_b_f16, 0, N * sizeof(half));
    }

    void TearDown(const ::benchmark::State&) override {
        cudaFree(d_a_f32); cudaFree(d_b_f32); cudaFree(d_c_f32);
        cudaFree(d_a_f16); cudaFree(d_b_f16); cudaFree(d_c_f16);
    }
};

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
// Elementwise Add
// ============================================================
BENCHMARK_DEFINE_F(BinaryBench, add_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ elementwise_add_f32_kernel::run(d_a_f32, d_b_f32, d_c_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 3 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(BinaryBench, add_f32)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(BinaryBench, add_f32x4)(benchmark::State& state) {
    BM_Run(state, [&]{ elementwise_add_f32x4_kernel::run(d_a_f32, d_b_f32, d_c_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 3 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(BinaryBench, add_f32x4)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(BinaryBench, add_f16)(benchmark::State& state) {
    BM_Run(state, [&]{ elementwise_add_f16_kernel::run(d_a_f16, d_b_f16, d_c_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 3 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(BinaryBench, add_f16)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(BinaryBench, add_f16x2)(benchmark::State& state) {
    BM_Run(state, [&]{ elementwise_add_f16x2_kernel::run(d_a_f16, d_b_f16, d_c_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 3 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(BinaryBench, add_f16x2)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(BinaryBench, add_f16x8)(benchmark::State& state) {
    BM_Run(state, [&]{ elementwise_add_f16x8_kernel::run(d_a_f16, d_b_f16, d_c_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 3 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(BinaryBench, add_f16x8)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

// ============================================================
// Elementwise Mul
// ============================================================
BENCHMARK_DEFINE_F(BinaryBench, mul_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ elementwise_mul_f32_kernel::run(d_a_f32, d_b_f32, d_c_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 3 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(BinaryBench, mul_f32)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(BinaryBench, mul_f32x4)(benchmark::State& state) {
    BM_Run(state, [&]{ elementwise_mul_f32x4_kernel::run(d_a_f32, d_b_f32, d_c_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 3 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(BinaryBench, mul_f32x4)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

// ============================================================
// Elementwise Sub
// ============================================================
BENCHMARK_DEFINE_F(BinaryBench, sub_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ elementwise_sub_f32_kernel::run(d_a_f32, d_b_f32, d_c_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 3 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(BinaryBench, sub_f32)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(BinaryBench, sub_f32x4)(benchmark::State& state) {
    BM_Run(state, [&]{ elementwise_sub_f32x4_kernel::run(d_a_f32, d_b_f32, d_c_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 3 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(BinaryBench, sub_f32x4)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

// ============================================================
// ReLU
// ============================================================
BENCHMARK_DEFINE_F(UnaryBench, relu_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ relu_f32_kernel::run(d_x_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, relu_f32)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, relu_f32x4)(benchmark::State& state) {
    BM_Run(state, [&]{ relu_f32x4_kernel::run(d_x_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, relu_f32x4)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, relu_f16)(benchmark::State& state) {
    BM_Run(state, [&]{ relu_f16_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, relu_f16)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, relu_f16x2)(benchmark::State& state) {
    BM_Run(state, [&]{ relu_f16x2_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, relu_f16x2)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, relu_f16x8)(benchmark::State& state) {
    BM_Run(state, [&]{ relu_f16x8_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, relu_f16x8)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, relu_f16x8_pack)(benchmark::State& state) {
    BM_Run(state, [&]{ relu_f16x8_pack_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, relu_f16x8_pack)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

// ============================================================
// ELU
// ============================================================
BENCHMARK_DEFINE_F(UnaryBench, elu_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ elu_f32_kernel::run(d_x_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, elu_f32)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, elu_f32x4)(benchmark::State& state) {
    BM_Run(state, [&]{ elu_f32x4_kernel::run(d_x_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, elu_f32x4)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, elu_f16)(benchmark::State& state) {
    BM_Run(state, [&]{ elu_f16_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, elu_f16)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, elu_f16x2)(benchmark::State& state) {
    BM_Run(state, [&]{ elu_f16x2_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, elu_f16x2)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, elu_f16x8)(benchmark::State& state) {
    BM_Run(state, [&]{ elu_f16x8_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, elu_f16x8)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, elu_f16x8_pack)(benchmark::State& state) {
    BM_Run(state, [&]{ elu_f16x8_pack_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, elu_f16x8_pack)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

// ============================================================
// GELU
// ============================================================
BENCHMARK_DEFINE_F(UnaryBench, gelu_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ gelu_f32_kernel::run(d_x_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, gelu_f32)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, gelu_f32x4)(benchmark::State& state) {
    BM_Run(state, [&]{ gelu_f32x4_kernel::run(d_x_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, gelu_f32x4)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, gelu_f16)(benchmark::State& state) {
    BM_Run(state, [&]{ gelu_f16_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, gelu_f16)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, gelu_f16x2)(benchmark::State& state) {
    BM_Run(state, [&]{ gelu_f16x2_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, gelu_f16x2)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, gelu_f16x8)(benchmark::State& state) {
    BM_Run(state, [&]{ gelu_f16x8_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, gelu_f16x8)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, gelu_f16x8_pack)(benchmark::State& state) {
    BM_Run(state, [&]{ gelu_f16x8_pack_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, gelu_f16x8_pack)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

// ============================================================
// Swish
// ============================================================
BENCHMARK_DEFINE_F(UnaryBench, swish_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ swish_f32_kernel::run(d_x_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, swish_f32)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, swish_f32x4)(benchmark::State& state) {
    BM_Run(state, [&]{ swish_f32x4_kernel::run(d_x_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, swish_f32x4)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, swish_f16)(benchmark::State& state) {
    BM_Run(state, [&]{ swish_f16_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, swish_f16)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, swish_f16x2)(benchmark::State& state) {
    BM_Run(state, [&]{ swish_f16x2_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, swish_f16x2)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, swish_f16x8)(benchmark::State& state) {
    BM_Run(state, [&]{ swish_f16x8_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, swish_f16x8)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, swish_f16x8_pack)(benchmark::State& state) {
    BM_Run(state, [&]{ swish_f16x8_pack_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, swish_f16x8_pack)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

// ============================================================
// HardSwish
// ============================================================
BENCHMARK_DEFINE_F(UnaryBench, hardswish_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ hardswish_f32_kernel::run(d_x_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, hardswish_f32)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, hardswish_f32x4)(benchmark::State& state) {
    BM_Run(state, [&]{ hardswish_f32x4_kernel::run(d_x_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, hardswish_f32x4)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, hardswish_f16)(benchmark::State& state) {
    BM_Run(state, [&]{ hardswish_f16_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, hardswish_f16)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, hardswish_f16x2)(benchmark::State& state) {
    BM_Run(state, [&]{ hardswish_f16x2_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, hardswish_f16x2)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, hardswish_f16x8)(benchmark::State& state) {
    BM_Run(state, [&]{ hardswish_f16x8_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, hardswish_f16x8)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, hardswish_f16x8_pack)(benchmark::State& state) {
    BM_Run(state, [&]{ hardswish_f16x8_pack_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, hardswish_f16x8_pack)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

// ============================================================
// HardShrink
// ============================================================
BENCHMARK_DEFINE_F(UnaryBench, hardshrink_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ hardshrink_f32_kernel::run(d_x_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, hardshrink_f32)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, hardshrink_f32x4)(benchmark::State& state) {
    BM_Run(state, [&]{ hardshrink_f32x4_kernel::run(d_x_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, hardshrink_f32x4)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, hardshrink_f16)(benchmark::State& state) {
    BM_Run(state, [&]{ hardshrink_f16_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, hardshrink_f16)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, hardshrink_f16x2)(benchmark::State& state) {
    BM_Run(state, [&]{ hardshrink_f16x2_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, hardshrink_f16x2)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, hardshrink_f16x8)(benchmark::State& state) {
    BM_Run(state, [&]{ hardshrink_f16x8_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, hardshrink_f16x8)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, hardshrink_f16x8_pack)(benchmark::State& state) {
    BM_Run(state, [&]{ hardshrink_f16x8_pack_kernel::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, hardshrink_f16x8_pack)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

// ============================================================
// Sigmoid
// ============================================================
BENCHMARK_DEFINE_F(UnaryBench, sigmoid_f32)(benchmark::State& state) {
    BM_Run(state, [&]{ sigmoid_f32_kernel_class::run(d_x_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, sigmoid_f32)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, sigmoid_f32x4)(benchmark::State& state) {
    BM_Run(state, [&]{ sigmoid_f32x4_kernel_class::run(d_x_f32, d_y_f32, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(float) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, sigmoid_f32x4)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, sigmoid_f16)(benchmark::State& state) {
    BM_Run(state, [&]{ sigmoid_f16_kernel_class::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, sigmoid_f16)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, sigmoid_f16x2)(benchmark::State& state) {
    BM_Run(state, [&]{ sigmoid_f16x2_kernel_class::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, sigmoid_f16x2)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, sigmoid_f16x8)(benchmark::State& state) {
    BM_Run(state, [&]{ sigmoid_f16x8_kernel_class::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, sigmoid_f16x8)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(UnaryBench, sigmoid_f16x8_pack)(benchmark::State& state) {
    BM_Run(state, [&]{ sigmoid_f16x8_pack_kernel_class::run(d_x_f16, d_y_f16, N); });
    state.SetBytesProcessed(int64_t(N) * 2 * sizeof(half) * state.iterations());
}
BENCHMARK_REGISTER_F(UnaryBench, sigmoid_f16x8_pack)->RangeMultiplier(4)->Range(1024, 4<<20)->UseManualTime()->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
