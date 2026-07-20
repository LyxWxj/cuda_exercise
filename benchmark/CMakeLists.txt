# benchmark/CMakeLists.txt

# --- elementwise benchmarks ---
add_executable(bench_elementwise elementwise_benchmark.cu)
target_link_libraries(bench_elementwise PRIVATE benchmark::benchmark CUDA::cudart)
set_target_properties(bench_elementwise PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- histogram benchmarks ---
add_executable(bench_histogram histogram_benchmark.cu)
target_link_libraries(bench_histogram PRIVATE benchmark::benchmark CUDA::cudart)
set_target_properties(bench_histogram PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- sigmoid benchmarks ---
add_executable(bench_sigmoid sigmoid_benchmark.cu)
target_link_libraries(bench_sigmoid PRIVATE benchmark::benchmark CUDA::cudart)
set_target_properties(bench_sigmoid PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# convenience target to run all benchmarks
add_custom_target(run_benchmarks
    COMMAND bench_elementwise
    COMMAND bench_histogram
    COMMAND bench_sigmoid
    DEPENDS bench_elementwise bench_histogram bench_sigmoid
)
