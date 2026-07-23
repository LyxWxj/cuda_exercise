#include "utils.cuh"


template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int stride = kWarpSize >> 1; stride; stride >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, stride);
  }
  return val;
}

// SGEMV: Warp SGEMV K32
// Assertion: K % 32 == 0
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
// launch: <<<(M/4, 1, 1), (32, 4, 1)>>>
__global__ void
sgemv_k32_f32_kernel(float* a, float* x, float* y,
  int M, int K) {
  int tx = threadIdx.x, // 0~31 
    ty = threadIdx.y; // 0~4
  int bx = blockIdx.x; // 0~M/4
  int lane = tx % WARP_SIZE; // 0~WARP_SIZE-1
  int m = bx * blockDim.y + ty; // (0~M/4)*4 + (0~3)
  if (m >= M) return;
  float sum = 0.f;
#pragma unroll
  for (int w = 0; w < K; w += WARP_SIZE) {
    // if NUM_WARPS >=2 accumulate the current line to WARP 0
    int k = w + lane;
    sum += a[m * K + k] * x[k];
  }
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  if (0 == lane) y[m] = sum;
}

// SGEMV: Warp SGEMV K128 + Vec4
// 假设K为128(32x4)的倍数 float4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
// launch: <<<(M/4, 1, 1), (32, 4, 1)>>>
__global__ void
sgemv_k128_f32x4_kernel(float* a, float* x, float* y,
  int M, int K) {
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x;
  int lane = tx % WARP_SIZE;
  int m = blockDim.y * bx + ty;
  if (m >= M) return;
  float sum = 0.f;
  const int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
  for (int w = 0; w < NUM_WARPS; ++w) {
    int k = (w * WARP_SIZE + lane) * 4;
    float4 regx = FLOAT4(x[k]);
    float4 rega = FLOAT4(a[m * K + k]);
    sum += rega.x * regx.x + rega.y * regx.y + rega.z * regx.z + rega.w * regx.w;
  }
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  if (lane == 0) y[m] = sum;
}

// SGEMV: Warp SGEMV K16
// 假设K为16 < 32,每个warp负责2行，每行有16个元素
// NUM_THREADS=128, NUM_WARPS=NUM_THREADS/WARP_SIZE;
// NUM_ROWS=NUM_WARPS * ROW_PER_WARP, grid(M/NUM_ROWS), block(32,NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
template <const int ROW_PER_WARP = 2>
// launch: <<<(M/NUM_ROWS, 1, 1), (32, NUM_WARPS, 1)>>>
__global__ void sgemv_k16_f32_kernel(float* A, float* x, float* y, int M,
  int K) {
  constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP;
  int tx = threadIdx.x;      // 0~31
  int ty = threadIdx.y;      // 0~NUM_WARPS
  int bx = blockIdx.x;       // 0~M/NUM_ROWS (NUM_ROWS=NUM_WARPS * ROW_PER_WARP)
  int lane = tx % WARP_SIZE; // 0~31
  int k = lane % K_WARP_SIZE; // 0~15
  // gloabl row of a: MxK and y:Mx1, blockDim.y=NUM_WARPS
  int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;
  if (m < M) {
    float sum = A[m * K + k] * x[k];
    sum = warp_reduce_sum_f32<K_WARP_SIZE>(sum);
    // 注意是k == 0，而不是lane == 0
    if (k == 0)
      y[m] = sum;
  }
}