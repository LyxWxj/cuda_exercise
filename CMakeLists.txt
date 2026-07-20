cmake_minimum_required(VERSION 3.14)

project(cuda_exercise LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 设置 CUDA 标准
if(CUDA_ENABLE)
    enable_language(CUDA)
endif()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)



find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)


if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found. Please ensure CUDA is installed and CUDA_TOOLKIT_ROOT_DIR is set.")
endif()

message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Toolkit location: ${CUDAToolkit_LIBRARY_DIR}")


add_executable(cuda_exercise main.cu)
# 收集源文件
file(GLOB_RECURSE SRC_FILES 
    src/*.cxx
    src/*.cpp
    src/*.cu
)
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
target_include_directories(cuda_exercise PRIVATE
  "/home/lyxwxj/thirdParty/cutlass/include"
  "/home/lyxwxj/thirdParty/cutlass/tools/util/include"
)
set_target_properties(cuda_exercise PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)


target_link_libraries(cuda_exercise PRIVATE
  cuda
  cudart
)

# ============================================================
#  测试 & 基准测试 (使用系统已安装的库)
# ============================================================
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

# 启用测试
enable_testing()

# 添加子目录
add_subdirectory(tests)
add_subdirectory(benchmark)