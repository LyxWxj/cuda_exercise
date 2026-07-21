#include "utils.cuh"


// 
__global__ void mat_transpose_f32_col2row_kernel(float* x, float* y, const int row, const int col) {
  // x[row][col]->y[col][row]
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_row = global_idx / col, global_col = global_idx % col;
  if (global_idx < row * col)
    y[global_col * row + global_row] = x[global_idx];
}

__global__ void mat_transpose_f32_row2col_kernel(float* x, float* y,
  const int row, const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_col = global_idx / row, global_row = global_idx % row;
  if (global_idx < row * col) {
    y[global_idx] = x[global_row * col + global_col];
  }
}

__global__ void mat_transpose_f32x4_col2row_kernel(float* x, float* y,
  const int row,
  const int col) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_col = (global_idx * 4) % col;
  int global_row = (global_idx * 4) / col;

  if (global_row < row && global_col + 3 < col) {
    float4 x_val = reinterpret_cast<float4*>(x)[global_idx];

    y[global_col * row + global_row] = x_val.x;
    y[(global_col + 1) * row + global_row] = x_val.y;
    y[(global_col + 2) * row + global_row] = x_val.z;
    y[(global_col + 3) * row + global_row] = x_val.w;
  }
}

__global__ void mat_transpose_f32x4_row2col_kernel(float* x, float* y,
  const int row,
  const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_col = (global_idx * 4) / row;
  const int global_row = (global_idx * 4) % row;

  if (global_row < row && global_col < col) {
    float4 x_val;
    x_val.x = x[global_row * col + global_col];
    x_val.y = x[(global_row + 1) * col + global_col];
    x_val.z = x[(global_row + 2) * col + global_col];
    x_val.w = x[(global_row + 3) * col + global_col];
    reinterpret_cast<float4*>(y)[global_idx] = FLOAT4(x_val);
  }
}

// work for row == col
__global__ void mat_transpose_f32_diagonal2d_kernel(float* x, float* y, int row,
  int col) {
  const int block_y = blockIdx.x;
  const int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
  const int global_col = threadIdx.x + blockDim.x * block_x;
  const int global_row = threadIdx.y + blockDim.y * block_y;
  if (global_col < col && global_row < row) {
    y[global_row * col + global_col] = x[global_col * row + global_row];
  }
}

__global__ void mat_transpose_f32_col2row2d_kernel(float* x, float* y,
  const int row,
  const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x < col && global_y < row) {
    y[global_x * row + global_y] = x[global_y * col + global_x];
  }
}

__global__ void mat_transpose_f32_row2col2d_kernel(float* x, float* y,
  const int row,
  const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_y < col && global_x < row) {
    y[global_y * row + global_x] = x[global_x * col + global_y];
  }
}

__global__ void mat_transpose_f32x4_col2row2d_kernel(float* x, float* y,
  const int row,
  const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x * 4 + 3 < col && global_y < row) {
    float4 x_val = reinterpret_cast<float4*>(x)[global_y * col / 4 + global_x];
    y[(global_x * 4) * row + global_y] = x_val.x;
    y[(global_x * 4 + 1) * row + global_y] = x_val.y;
    y[(global_x * 4 + 2) * row + global_y] = x_val.z;
    y[(global_x * 4 + 3) * row + global_y] = x_val.w;
  }
}
__global__ void mat_transpose_f32x4_row2col2d_kernel(float* x, float* y,
  const int row,
  const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_y * 4 + 3 < row && global_x < col) {
    float4 x_val;
    x_val.x = x[(global_y * 4) * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];
    reinterpret_cast<float4*>(y)[global_x * row / 4 + global_y] =
      FLOAT4(x_val);
  }
}

__global__ void mat_transpose_f32x4_shared_col2row2d_kernel(float* x, float* y,
  const int row,
  const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4];
  if (global_x * 4 + 3 < col + 3 && global_y < row) {
    // load value from x to shared memory
    float4 x_val = reinterpret_cast<float4*>(x)[global_y * col / 4 + global_x];
    FLOAT4(tile[local_y][local_x * 4]) = FLOAT4(x_val);
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    // add STRIDE to satisfied different block size.
    constexpr int STRIDE = WARP_SIZE_S / 4;
    smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
    smem_val.y =
      tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
    smem_val.z =
      tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
    smem_val.w =
      tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];
    // map index n*n to (n/4)*(n*4)
    const int bid_y = blockIdx.y * blockDim.y;
    const int out_y = global_x * 4 + local_y / STRIDE;
    const int out_x = (local_y % STRIDE) * 4 + bid_y;
    reinterpret_cast<float4*>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);
  }
}

__global__ void mat_transpose_f32x4_shared_row2col2d_kernel(float* x, float* y,
  const int row,
  const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S];
  if (global_y * 4 < row && global_x < col) {
    // load value from x to shared memory
    float4 x_val;
    x_val.x = x[(global_y * 4) * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];
    tile[local_y * 4][local_x] = x_val.x;
    tile[local_y * 4 + 1][local_x] = x_val.y;
    tile[local_y * 4 + 2][local_x] = x_val.z;
    tile[local_y * 4 + 3][local_x] = x_val.w;
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    // add STRIDE to satisfied different block size.
    // map index n*n to (n/4)*(n*4)
    constexpr int STRIDE = WARP_SIZE_S / 4;
    smem_val.x = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4];
    smem_val.y =
      tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 1];
    smem_val.z =
      tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 2];
    smem_val.w =
      tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 3];
    const int bid_x = blockIdx.x * blockDim.x;
    const int bid_y = blockIdx.y * blockDim.y;

    const int out_y = bid_x + (local_y % STRIDE) * 4;
    const int out_x = bid_y * 4 + local_x * 4 + (local_y / STRIDE);
    y[out_y * row + out_x] = smem_val.x;
    y[(out_y + 1) * row + out_x] = smem_val.y;
    y[(out_y + 2) * row + out_x] = smem_val.z;
    y[(out_y + 3) * row + out_x] = smem_val.w;
  }
}

__global__ void mat_transpose_f32x4_shared_bcf_col2row2d_kernel(float* x,
  float* y,
  const int row,
  const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4 + PAD];
  if (global_x * 4 + 3 < col + 3 && global_y < row) {
    // load value from x to shared memory
    float4 x_val = reinterpret_cast<float4*>(x)[global_y * col / 4 + global_x];
    tile[local_y][local_x * 4] = x_val.x;
    tile[local_y][local_x * 4 + 1] = x_val.y;
    tile[local_y][local_x * 4 + 2] = x_val.z;
    tile[local_y][local_x * 4 + 3] = x_val.w;
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    // add STRIDE to satisfied different block size.
    constexpr int STRIDE = WARP_SIZE_S / 4;
    smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];
    smem_val.y =
      tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
    smem_val.z =
      tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
    smem_val.w =
      tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];
    // map index n*n to (n/4)*(n*4)
    const int bid_y = blockIdx.y * blockDim.y;
    const int out_y = global_x * 4 + local_y / STRIDE;
    const int out_x = (local_y % STRIDE) * 4 + bid_y;
    reinterpret_cast<float4*>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);
  }
}

__global__ void mat_transpose_f32x4_shared_bcf_row2col2d_kernel(float* x,
  float* y,
  const int row,
  const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S + PAD];
  if (global_y * 4 < row && global_x < col) {
    // load value from x to shared memory
    float4 x_val;
    x_val.x = x[(global_y * 4) * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];
    tile[local_y * 4][local_x] = x_val.x;
    tile[local_y * 4 + 1][local_x] = x_val.y;
    tile[local_y * 4 + 2][local_x] = x_val.z;
    tile[local_y * 4 + 3][local_x] = x_val.w;
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    // add STRIDE to satisfied different block size.
    // map index n*n to (n/4)*(n*4)
    constexpr int STRIDE = WARP_SIZE_S / 4;
    smem_val.x = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4];
    smem_val.y =
      tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 1];
    smem_val.z =
      tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 2];
    smem_val.w =
      tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + 3];
    const int bid_x = blockIdx.x * blockDim.x;
    const int bid_y = blockIdx.y * blockDim.y;

    const int out_y = bid_x + (local_y % STRIDE) * 4;
    const int out_x = bid_y * 4 + local_x * 4 + (local_y / STRIDE);
    y[out_y * row + out_x] = smem_val.x;
    y[(out_y + 1) * row + out_x] = smem_val.y;
    y[(out_y + 2) * row + out_x] = smem_val.z;
    y[(out_y + 3) * row + out_x] = smem_val.w;
  }
}

__global__ void mat_transpose_f32x4_shared_bcf_merge_write_row2col2d_kernel(
  float* x, float* y, const int row, const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S + PAD];
  if (global_y * 4 < row && global_x < col) {
    // load value from x to shared memory
    float4 x_val;
    x_val.x = x[(global_y * 4) * col + global_x];
    x_val.y = x[(global_y * 4 + 1) * col + global_x];
    x_val.z = x[(global_y * 4 + 2) * col + global_x];
    x_val.w = x[(global_y * 4 + 3) * col + global_x];
    tile[local_y * 4][local_x] = x_val.x;
    tile[local_y * 4 + 1][local_x] = x_val.y;
    tile[local_y * 4 + 2][local_x] = x_val.z;
    tile[local_y * 4 + 3][local_x] = x_val.w;
    __syncthreads();
    float4 smem_val;
    // load value from shared memory to y.
    smem_val.x = tile[local_x * 4][local_y];
    smem_val.y = tile[local_x * 4 + 1][local_y];
    smem_val.z = tile[local_x * 4 + 2][local_y];
    smem_val.w = tile[local_x * 4 + 3][local_y];

    const int gid_x = blockIdx.x * blockDim.x;
    const int gid_y = blockIdx.y * blockDim.y * 4;
    const int out_y = gid_y + local_x * 4;
    const int out_x = gid_x + local_y;
    reinterpret_cast<float4*>(y)[(out_x * row + out_y) / 4] = FLOAT4(smem_val);
  }
}