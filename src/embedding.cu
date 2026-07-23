#include "utils.cuh"


// lauch: <<<(n,1,1),(emb_size,1,1)>>>
// launch: <<<(N/256, 1, 1), (256, 1, 1)>>>
__global__ void embedding_f32_kernel(
  const int* idx, float* weight, float* output,
  const int n, const int emb_size) {
  // idx: (n,) weight: (vocab,emb_size) output: (n, emb_size)
  const int tidx = threadIdx.x, bidx = blockIdx.x;
  int offset = idx[bidx] * emb_size;
  output[bidx * emb_size + tidx] = weight[offset + tidx];
}

// lauch: <<<(n, 1, 1),(emb_size/4, 1, 1)>>>
// launch: <<<(N/1024, 1, 1), (256, 1, 1)>>>
__global__ void embedding_f32x4_kernel(
  const int* idx, float* weight, float* output,
  const int n, const int emb_size) {
  const int tidx = threadIdx.x, bidx = blockIdx.x;
  const int offset = idx[bidx] * emb_size;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    output[bidx * emb_size + tidx + i] = weight[offset + tidx + i];
  }
}

// launch: <<<(N/1024, 1, 1), (256, 1, 1)>>>
__global__ void embedding_f32x4_pack_kernel(
  const int* idx, float* weight, float* output,
  const int n, const int emb_size) {
  const int tidx = threadIdx.x, bidx = blockIdx.x;
  const int offset = idx[bidx] * emb_size;
  LDST128BITS(output[bidx * emb_size + 4 * tidx]) = LDST128BITS(weight[offset + 4 * tidx]);
}

// lauch: <<<(n,1,1),(emb_size,1,1)>>>
// launch: <<<(N/256, 1, 1), (256, 1, 1)>>>
__global__ void embedding_f16_kernel(
  const int* idx, half* weight, half* output, int n, int emb_size) {
  const int tidx = threadIdx.x, bidx = blockIdx.x;
  const int offset = idx[bidx] * emb_size + tidx;
  output[bidx * emb_size + tidx] = weight[offset + tidx];
}

// lauch: <<<(n,1,1),(emb_size/8, 1, 1)>>>
// launch: <<<(N/2048, 1, 1), (256, 1, 1)>>>
__global__ void embedding_f16x8_kernel(
  const int* idx, half* weight, half* output, int n, int emb_size) {
  const int tidx = threadIdx.x, bidx = blockIdx.x;
  const int offset = idx[bidx] * emb_size + tidx;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    output[bidx * emb_size + tidx + i] = weight[offset + tidx + i];
  }
}

// launch: <<<(N/2048, 1, 1), (256, 1, 1)>>>
__global__ void embedding_f16x8_pack_kernel(
  const int* idx, half* weight, half* output, int n, int emb_size) {
  const int tidx = threadIdx.x, bidx = blockIdx.x;
  const int offset = idx[bidx] * emb_size + tidx;
  LDST128BITS(output[bidx * emb_size + 8 * tidx]) =
    LDST128BITS(weight[offset + 8 * tidx]);
}