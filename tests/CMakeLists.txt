# tests/CMakeLists.txt

# --- elementwise tests ---
add_executable(test_elementwise elementwise_test.cu)
target_link_libraries(test_elementwise PRIVATE GTest::gtest GTest::gtest_main CUDA::cudart)
set_target_properties(test_elementwise PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME Elementwise COMMAND test_elementwise)

# --- histogram tests ---
add_executable(test_histogram histogram_test.cu)
target_link_libraries(test_histogram PRIVATE GTest::gtest GTest::gtest_main CUDA::cudart)
set_target_properties(test_histogram PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME Histogram COMMAND test_histogram)

# --- sigmoid tests ---
add_executable(test_sigmoid sigmoid_test.cu)
target_link_libraries(test_sigmoid PRIVATE GTest::gtest GTest::gtest_main CUDA::cudart)
set_target_properties(test_sigmoid PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME Sigmoid COMMAND test_sigmoid)

# --- relu tests ---
add_executable(test_relu relu_test.cu)
target_link_libraries(test_relu PRIVATE GTest::gtest GTest::gtest_main CUDA::cudart)
set_target_properties(test_relu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME Relu COMMAND test_relu)

# --- elu tests ---
add_executable(test_elu elu_test.cu)
target_link_libraries(test_elu PRIVATE GTest::gtest GTest::gtest_main CUDA::cudart)
set_target_properties(test_elu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME Elu COMMAND test_elu)

# --- gelu tests ---
add_executable(test_gelu gelu_test.cu)
target_link_libraries(test_gelu PRIVATE GTest::gtest GTest::gtest_main CUDA::cudart)
set_target_properties(test_gelu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME Gelu COMMAND test_gelu)

# --- swish tests ---
add_executable(test_swish swish_test.cu)
target_link_libraries(test_swish PRIVATE GTest::gtest GTest::gtest_main CUDA::cudart)
set_target_properties(test_swish PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME Swish COMMAND test_swish)

# --- hardswish tests ---
add_executable(test_hardswish hardswish_test.cu)
target_link_libraries(test_hardswish PRIVATE GTest::gtest GTest::gtest_main CUDA::cudart)
set_target_properties(test_hardswish PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME Hardswish COMMAND test_hardswish)

# --- hardshrink tests ---
add_executable(test_hardshrink hardshrink_test.cu)
target_link_libraries(test_hardshrink PRIVATE GTest::gtest GTest::gtest_main CUDA::cudart)
set_target_properties(test_hardshrink PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME Hardshrink COMMAND test_hardshrink)

# --- block_all_reduce + dot_product tests ---
add_executable(test_reduce block_all_reduce_test.cu)
target_link_libraries(test_reduce PRIVATE GTest::gtest GTest::gtest_main CUDA::cudart)
set_target_properties(test_reduce PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME Reduce COMMAND test_reduce)
