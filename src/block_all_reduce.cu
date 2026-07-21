#include "utils.cuh"

// ============================================================
// Generic warp reduce: val = Op(val, shuffle(val, mask))
// Op defaults to AddOp (sum). Type-specific behavior (e.g. half
// uses __hadd) is handled by Op::apply, not by algorithm specialization.
// ============================================================
template<typename T, const int kWarpSize = WARP_SIZE, typename Op = AddOp>
__device__ __forceinline__ T warp_reduce(T val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = Op::apply(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// ============================================================
// atomicAdd wrapper: handles all types (float direct, others convert)
// ============================================================
template<typename T>
__device__ __forceinline__ void atomic_add(T* addr, T val);

template<>
__device__ __forceinline__ void atomic_add<float>(float* addr, float val) {
  atomicAdd(addr, val);
}

template<>
__device__ __forceinline__ void atomic_add<half>(half* addr, half val) {
  atomicAdd(addr, __half2float(val));
}

template<>
__device__ __forceinline__ void atomic_add<__nv_bfloat16>(__nv_bfloat16* addr, __nv_bfloat16 val) {
  atomicAdd(addr, __bfloat162float(val));
}

template<>
__device__ __forceinline__ void atomic_add<int32_t>(int32_t* addr, int32_t val) {
  atomicAdd(addr, val);
}

// ============================================================
// ReduceTraits: forward declaration (full specializations below)
// ============================================================
template<typename InputT, typename AccT> struct ReduceTraits;

// Sentinel types for packed variants (to disambiguate traits)
struct half8 {};       struct bf16_8 {};
struct fp8_e4m3_16 {}; struct fp8_e5m2_16 {}; struct int8_16 {};

// ============================================================
// Generic block all-reduce sum kernel
// ============================================================
template<typename InputT, typename AccT, int VecWidth, int NUM_THREADS>
__global__ void block_all_reduce_sum_kernel(
    typename ReduceTraits<InputT, AccT>::ptr_type* a, AccT* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * VecWidth;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ AccT reduce_smem[NUM_WARPS];

  // 1. Load VecWidth elements and reduce to single AccT
  AccT sum = (idx + VecWidth - 1 < N) ? ReduceTraits<InputT, AccT>::load(&a[idx])
                                      : ReduceTraits<InputT, AccT>::zero();

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;

  // 2. Warp-level reduce
  sum = warp_reduce<AccT, WARP_SIZE>(sum);

  // 3. Store warp partial sums to smem
  if (lane == 0)
    reduce_smem[warp] = sum;
  __syncthreads();

  // 4. Inter-warp reduce (warp 0)
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : ReduceTraits<InputT, AccT>::zero();
  if (warp == 0)
    sum = warp_reduce<AccT, NUM_WARPS>(sum);

  // 5. Atomic add to output
  if (tid == 0)
    atomic_add(y, sum);
}

// ============================================================
// ReduceTraits: type-specific load + conversion
// ============================================================

// --- float ---
template<> struct ReduceTraits<float, float> {
  using ptr_type = float;
  __device__ static float zero() { return 0.f; }
  __device__ static float load(float* p) { return *p; }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, float>::value>>
  __device__ static float load_op(float* a, float* b) {
    return Op::apply(*a, *b);
  }
};

template<> struct ReduceTraits<float4, float> {
  using ptr_type = float;
  __device__ static float zero() { return 0.f; }
  __device__ static float load(float* p) {
    float4 v = FLOAT4(p[0]);
    return v.x + v.y + v.z + v.w;
  }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, float>::value>>
  __device__ static float load_op(float* a, float* b) {
    float4 va = FLOAT4(a[0]), vb = FLOAT4(b[0]);
    return Op::apply(va.x, vb.x) + Op::apply(va.y, vb.y)
         + Op::apply(va.z, vb.z) + Op::apply(va.w, vb.w);
  }
};

// --- half ---
template<> struct ReduceTraits<half, float> {
  using ptr_type = half;
  __device__ static float zero() { return 0.f; }
  __device__ static float load(half* p) { return __half2float(*p); }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, half>::value>>
  __device__ static float load_op(half* a, half* b) {
    return __half2float(Op::apply(*a, *b));
  }
};

template<> struct ReduceTraits<half, half> {
  using ptr_type = half;
  __device__ static half zero() { return __float2half(0.f); }
  __device__ static half load(half* p) { return *p; }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, half>::value>>
  __device__ static half load_op(half* a, half* b) { return Op::apply(*a, *b); }
};

template<> struct ReduceTraits<half2, float> {
  using ptr_type = half;
  __device__ static float zero() { return 0.f; }
  __device__ static float load(half* p) {
    half2 v = HALF2(p[0]);
    return __half2float(v.x) + __half2float(v.y);
  }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, half>::value>>
  __device__ static float load_op(half* a, half* b) {
    half2 va = HALF2(a[0]), vb = HALF2(b[0]);
    return __half2float(Op::apply(va.x, vb.x)) + __half2float(Op::apply(va.y, vb.y));
  }
};

template<> struct ReduceTraits<half2, half> {
  using ptr_type = half;
  __device__ static half zero() { return __float2half(0.f); }
  __device__ static half load(half* p) {
    half2 v = HALF2(p[0]);
    return __hadd(v.x, v.y);
  }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, half>::value>>
  __device__ static half load_op(half* a, half* b) {
    half2 va = HALF2(a[0]), vb = HALF2(b[0]);
    return __hadd(Op::apply(va.x, vb.x), Op::apply(va.y, vb.y));
  }
};

// half x8 pack
template<> struct ReduceTraits<half8, float> {
  using ptr_type = half;
  __device__ static float zero() { return 0.f; }
  __device__ static float load(half* p) {
    half pack[8];
    LDST128BITS(pack[0]) = LDST128BITS(p[0]);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += __half2float(pack[i]);
    return s;
  }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, half>::value>>
  __device__ static float load_op(half* a, half* b) {
    half pa[8], pb[8];
    LDST128BITS(pa[0]) = LDST128BITS(a[0]);
    LDST128BITS(pb[0]) = LDST128BITS(b[0]);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += __half2float(Op::apply(pa[i], pb[i]));
    return s;
  }
};

template<> struct ReduceTraits<half8, half> {
  using ptr_type = half;
  __device__ static half zero() { return __float2half(0.f); }
  __device__ static half load(half* p) {
    half pack[8];
    LDST128BITS(pack[0]) = LDST128BITS(p[0]);
    half s = __float2half(0.f);
#pragma unroll
    for (int i = 0; i < 8; ++i) s = __hadd(s, pack[i]);
    return s;
  }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, half>::value>>
  __device__ static half load_op(half* a, half* b) {
    half pa[8], pb[8];
    LDST128BITS(pa[0]) = LDST128BITS(a[0]);
    LDST128BITS(pb[0]) = LDST128BITS(b[0]);
    half s = __float2half(0.f);
#pragma unroll
    for (int i = 0; i < 8; ++i) s = __hadd(s, Op::apply(pa[i], pb[i]));
    return s;
  }
};

// --- bfloat16 ---
template<> struct ReduceTraits<__nv_bfloat16, float> {
  using ptr_type = __nv_bfloat16;
  __device__ static float zero() { return 0.f; }
  __device__ static float load(__nv_bfloat16* p) { return __bfloat162float(*p); }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, __nv_bfloat16>::value>>
  __device__ static float load_op(__nv_bfloat16* a, __nv_bfloat16* b) {
    return __bfloat162float(Op::apply(*a, *b));
  }
};

template<> struct ReduceTraits<__nv_bfloat16, __nv_bfloat16> {
  using ptr_type = __nv_bfloat16;
  __device__ static __nv_bfloat16 zero() { return __float2bfloat16(0.f); }
  __device__ static __nv_bfloat16 load(__nv_bfloat16* p) { return *p; }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, __nv_bfloat16>::value>>
  __device__ static __nv_bfloat16 load_op(__nv_bfloat16* a, __nv_bfloat16* b) {
    return Op::apply(*a, *b);
  }
};

template<> struct ReduceTraits<__nv_bfloat162, float> {
  using ptr_type = __nv_bfloat16;
  __device__ static float zero() { return 0.f; }
  __device__ static float load(__nv_bfloat16* p) {
    __nv_bfloat162 v = BFLOAT2(p[0]);
    return __bfloat162float(v.x) + __bfloat162float(v.y);
  }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, __nv_bfloat16>::value>>
  __device__ static float load_op(__nv_bfloat16* a, __nv_bfloat16* b) {
    __nv_bfloat162 va = BFLOAT2(a[0]), vb = BFLOAT2(b[0]);
    return __bfloat162float(Op::apply(va.x, vb.x)) + __bfloat162float(Op::apply(va.y, vb.y));
  }
};

template<> struct ReduceTraits<__nv_bfloat162, __nv_bfloat16> {
  using ptr_type = __nv_bfloat16;
  __device__ static __nv_bfloat16 zero() { return __float2bfloat16(0.f); }
  __device__ static __nv_bfloat16 load(__nv_bfloat16* p) {
    __nv_bfloat162 v = BFLOAT2(p[0]);
    return __hadd(v.x, v.y);
  }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, __nv_bfloat16>::value>>
  __device__ static __nv_bfloat16 load_op(__nv_bfloat16* a, __nv_bfloat16* b) {
    __nv_bfloat162 va = BFLOAT2(a[0]), vb = BFLOAT2(b[0]);
    return __hadd(Op::apply(va.x, vb.x), Op::apply(va.y, vb.y));
  }
};

// bf16 x8 pack
template<> struct ReduceTraits<bf16_8, float> {
  using ptr_type = __nv_bfloat16;
  __device__ static float zero() { return 0.f; }
  __device__ static float load(__nv_bfloat16* p) {
    __nv_bfloat16 pack[8];
    LDST128BITS(pack[0]) = LDST128BITS(p[0]);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += __bfloat162float(pack[i]);
    return s;
  }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, __nv_bfloat16>::value>>
  __device__ static float load_op(__nv_bfloat16* a, __nv_bfloat16* b) {
    __nv_bfloat16 pa[8], pb[8];
    LDST128BITS(pa[0]) = LDST128BITS(a[0]);
    LDST128BITS(pb[0]) = LDST128BITS(b[0]);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += __bfloat162float(Op::apply(pa[i], pb[i]));
    return s;
  }
};

template<> struct ReduceTraits<bf16_8, __nv_bfloat16> {
  using ptr_type = __nv_bfloat16;
  __device__ static __nv_bfloat16 zero() { return __float2bfloat16(0.f); }
  __device__ static __nv_bfloat16 load(__nv_bfloat16* p) {
    __nv_bfloat16 pack[8];
    LDST128BITS(pack[0]) = LDST128BITS(p[0]);
    __nv_bfloat16 s = __float2bfloat16(0.f);
#pragma unroll
    for (int i = 0; i < 8; ++i) s = __hadd(s, pack[i]);
    return s;
  }
};

// --- fp8 ---
template<> struct ReduceTraits<__nv_fp8_storage_t, half> {
  using ptr_type = __nv_fp8_storage_t;
  __device__ static half zero() { return __float2half(0.f); }
  __device__ static half load(__nv_fp8_storage_t* p) {
    return __nv_cvt_fp8_to_halfraw(*p, __NV_E4M3);
  }
};

// fp8 x16 pack (e4m3)
template<> struct ReduceTraits<fp8_e4m3_16, half> {
  using ptr_type = __nv_fp8_storage_t;
  __device__ static half zero() { return __float2half(0.f); }
  __device__ static half load(__nv_fp8_storage_t* p) {
    __nv_fp8_storage_t pack[16];
    LDST128BITS(pack[0]) = LDST128BITS(p[0]);
    half s = __float2half(0.f);
#pragma unroll
    for (int i = 0; i < 16; ++i)
      s = __hadd(s, __nv_cvt_fp8_to_halfraw(pack[i], __NV_E4M3));
    return s;
  }
};

// fp8 x16 pack (e5m2)
template<> struct ReduceTraits<fp8_e5m2_16, half> {
  using ptr_type = __nv_fp8_storage_t;
  __device__ static half zero() { return __float2half(0.f); }
  __device__ static half load(__nv_fp8_storage_t* p) {
    __nv_fp8_storage_t pack[16];
    LDST128BITS(pack[0]) = LDST128BITS(p[0]);
    half s = __float2half(0.f);
#pragma unroll
    for (int i = 0; i < 16; ++i)
      s = __hadd(s, __nv_cvt_fp8_to_halfraw(pack[i], __NV_E5M2));
    return s;
  }
};

// --- int8 ---
template<> struct ReduceTraits<int8_t, int32_t> {
  using ptr_type = int8_t;
  __device__ static int32_t zero() { return 0; }
  __device__ static int32_t load(int8_t* p) { return static_cast<int32_t>(*p); }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, int8_t>::value>>
  __device__ static int32_t load_op(int8_t* a, int8_t* b) {
    return static_cast<int32_t>(Op::apply(*a, *b));
  }
};

// int8 x16 pack
template<> struct ReduceTraits<int8_16, int32_t> {
  using ptr_type = int8_t;
  __device__ static int32_t zero() { return 0; }
  __device__ static int32_t load(int8_t* p) {
    int8_t pack[16];
    LDST128BITS(pack[0]) = LDST128BITS(p[0]);
    int32_t s = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) s += static_cast<int32_t>(pack[i]);
    return s;
  }
  template<typename Op, typename = std::enable_if_t<detail::Op_is_binary<Op, int8_t>::value>>
  __device__ static int32_t load_op(int8_t* a, int8_t* b) {
    int8_t pa[16], pb[16];
    LDST128BITS(pa[0]) = LDST128BITS(a[0]);
    LDST128BITS(pb[0]) = LDST128BITS(b[0]);
    int32_t s = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i)
      s += static_cast<int32_t>(Op::apply(pa[i], pb[i]));
    return s;
  }
};

// ============================================================
// Macro: wrapper struct with static run() for named kernel access
// ============================================================
#define INSTANTIATE_REDUCE(name, InputT, AccT, VecW, Threads)                 \
  static void name##_impl(typename ReduceTraits<InputT, AccT>::ptr_type* a,   \
                          AccT* y, int N, dim3 g, dim3 b, cudaStream_t s) {   \
    block_all_reduce_sum_kernel<InputT, AccT, VecW, Threads><<<g,b,0,s>>>(a,y,N);\
  }                                                                           \
  struct name {                                                               \
    using _ptr_t = typename ReduceTraits<InputT, AccT>::ptr_type;             \
    static void run(_ptr_t* a, AccT* y, int N,                                \
                    dim3 grid, dim3 blk, cudaStream_t stream = 0) {            \
      name##_impl(a, y, N, grid, blk, stream);                                \
    }                                                                         \
    static void run(_ptr_t* a, AccT* y, int N) {                              \
      dim3 blk(Threads);                                                      \
      dim3 g((N + Threads * VecW - 1) / (Threads * VecW));                    \
      name##_impl(a, y, N, g, blk, 0);                                        \
    }                                                                         \
  }

// --- f32 ---
INSTANTIATE_REDUCE(block_all_reduce_sum_f32_f32_kernel,       float,           float,    1, 256);
INSTANTIATE_REDUCE(block_all_reduce_sum_f32x4_f32_kernel,     float4,          float,    4, 64);

// --- f16 ---
INSTANTIATE_REDUCE(block_all_reduce_sum_f16_f16_kernel,       half,            half,     1, 256);
INSTANTIATE_REDUCE(block_all_reduce_sum_f16_f32_kernel,       half,            float,    1, 256);
INSTANTIATE_REDUCE(block_all_reduce_sum_f16x2_f16_kernel,     half2,           half,     2, 128);
INSTANTIATE_REDUCE(block_all_reduce_sum_f16x2_f32_kernel,     half2,           float,    2, 128);
INSTANTIATE_REDUCE(block_all_reduce_sum_f16x8_pack_f16_kernel, half8,          half,     8, 32);
INSTANTIATE_REDUCE(block_all_reduce_sum_f16x8_pack_f32_kernel, half8,          float,    8, 32);

// --- bf16 ---
INSTANTIATE_REDUCE(block_all_reduce_sum_bf16_bf16_kernel,     __nv_bfloat16,  __nv_bfloat16, 1, 256);
INSTANTIATE_REDUCE(block_all_reduce_sum_bf16_f32_kernel,      __nv_bfloat16,  float,    1, 256);
INSTANTIATE_REDUCE(block_all_reduce_sum_bf16x2_bf16_kernel,   __nv_bfloat162, __nv_bfloat16, 2, 128);
INSTANTIATE_REDUCE(block_all_reduce_sum_bf16x2_f32_kernel,    __nv_bfloat162, float,    2, 128);
INSTANTIATE_REDUCE(block_all_reduce_sum_bf16x8_pack_bf16_kernel, bf16_8,      __nv_bfloat16, 8, 32);
INSTANTIATE_REDUCE(block_all_reduce_sum_bf16x8_pack_f32_kernel,  bf16_8,      float,    8, 32);

// --- fp8 ---
INSTANTIATE_REDUCE(block_all_reduce_sum_fp8_e4m3_f16_kernel,        __nv_fp8_storage_t, half, 1, 256);
INSTANTIATE_REDUCE(block_all_reduce_sum_fp8_e5m2_f16_kernel,        __nv_fp8_storage_t, half, 1, 256);
INSTANTIATE_REDUCE(block_all_reduce_sum_fp8_e4m3x16_pack_f16_kernel, fp8_e4m3_16,       half, 16, 16);
INSTANTIATE_REDUCE(block_all_reduce_sum_fp8_e5m2x16_pack_f16_kernel, fp8_e5m2_16,       half, 16, 16);

// --- int8 ---
INSTANTIATE_REDUCE(block_all_reduce_sum_i8_i32_kernel,        int8_t,          int32_t,  1, 256);
INSTANTIATE_REDUCE(block_all_reduce_sum_i8x16_pack_i32_kernel, int8_16,        int32_t,  16, 16);

// ============================================================
// Generic block binary-reduce sum kernel
// y = sum(ElemOp(a[i], b[i])) via atomicAdd
//
// ElemOp: binary op applied per-element (e.g. MulOp for dot product)
// ReduceTraits<InputT, AccT>::load_op<ElemOp>(a, b):
//   load VecWidth elements from a and b, apply ElemOp pairwise,
//   reduce to single AccT
// ============================================================
template<typename InputT, typename AccT, typename ElemOp, int VecWidth, int NUM_THREADS>
__global__ void block_binary_reduce_sum_kernel(
    typename ReduceTraits<InputT, AccT>::ptr_type* a,
    typename ReduceTraits<InputT, AccT>::ptr_type* b, AccT* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * VecWidth;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ AccT reduce_smem[NUM_WARPS];

  AccT sum = (idx + VecWidth - 1 < N)
                 ? ReduceTraits<InputT, AccT>::template load_op<ElemOp>(&a[idx], &b[idx])
                 : ReduceTraits<InputT, AccT>::zero();

  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;

  sum = warp_reduce<AccT, WARP_SIZE>(sum);

  if (lane == 0)
    reduce_smem[warp] = sum;
  __syncthreads();

  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : ReduceTraits<InputT, AccT>::zero();
  if (warp == 0)
    sum = warp_reduce<AccT, NUM_WARPS>(sum);

  if (tid == 0)
    atomic_add(y, sum);
}

// ============================================================
// Macro: binary reduce kernel wrapper struct
// ============================================================
#define INSTANTIATE_BINARY_REDUCE(name, InputT, AccT, ElemOp, VecW, Threads)  \
  void name##_impl(typename ReduceTraits<InputT, AccT>::ptr_type* a,          \
                   typename ReduceTraits<InputT, AccT>::ptr_type* b,          \
                   AccT* y, int N, dim3 g, dim3 bd, cudaStream_t s) {         \
    block_binary_reduce_sum_kernel<InputT, AccT, ElemOp, VecW, Threads>       \
        <<<g,bd,0,s>>>(a, b, y, N);                                           \
  }                                                                           \
  struct name {                                                               \
    using _ptr_t = typename ReduceTraits<InputT, AccT>::ptr_type;             \
    static void run(_ptr_t* a, _ptr_t* b, AccT* y, int N,                     \
                    dim3 grid, dim3 blk, cudaStream_t stream = 0) {            \
      name##_impl(a, b, y, N, grid, blk, stream);                             \
    }                                                                         \
    static void run(_ptr_t* a, _ptr_t* b, AccT* y, int N) {                   \
      dim3 blk(Threads);                                                      \
      dim3 g((N + Threads * VecW - 1) / (Threads * VecW));                    \
      name##_impl(a, b, y, N, g, blk, 0);                                     \
    }                                                                         \
  }

// --- dot product: sum(a[i] * b[i]) ---
INSTANTIATE_BINARY_REDUCE(dot_prod_f32_f32_kernel,        float,          float, MulOp, 1, 256);
INSTANTIATE_BINARY_REDUCE(dot_prod_f32x4_f32_kernel,      float4,         float, MulOp, 4, 64);
INSTANTIATE_BINARY_REDUCE(dot_prod_f16_f32_kernel,        half,           float, MulOp, 1, 256);
INSTANTIATE_BINARY_REDUCE(dot_prod_f16x2_f32_kernel,      half2,          float, MulOp, 2, 128);
INSTANTIATE_BINARY_REDUCE(dot_prod_f16x8_pack_f32_kernel, half8,          float, MulOp, 8, 32);
