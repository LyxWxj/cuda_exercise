#pragma once
#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cuda_fp8.h>
#include <cuda_fp8.hpp>
#include <thrust/complex.h>
#include <type_traits>
#include <mma.h>

#define CUDA_CHECK(call) \
do {\
  cudaError_t err = call; \
  if(err != cudaSuccess) { \
    std::cerr <<"CUDA error at " << __FILE__<<":" << __LINE__\
    << " - " << cudaGetErrorString(err) << std::endl; \
  }\
}while(0)

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
struct AddOp {
  template<typename T>
  __host__ __device__ static T apply(T a, T b) { return a + b; }
};
template<>
__host__ __device__ inline half AddOp::apply<half>(half a, half b) {
  return __hadd(a, b);
}

struct MulOp {
  template<typename T>
  __host__ __device__ static T apply(T a, T b) { return a * b; }
};

struct SubOp {
  template<typename T>
  __host__ __device__ static T apply(T a, T b) { return a - b; }
};

struct MinOp {
  template<typename T>
  __host__ __device__ static T apply(T a, T b) { return a < b ? a : b; }
};
template<>
__host__ __device__ inline half MinOp::apply<half>(half a, half b) {
  return __hmin(a, b);
}

struct MaxOp {
  template<typename T>
  __host__ __device__ static T apply(T a, T b) { return a > b ? a : b; }
};
template<>
__host__ __device__ inline half MaxOp::apply<half>(half a, half b) {
  return __hmax(a, b);
}

// In-place ops: mutate input, return void. Used with 1-arg TNApply / Apply*(A).
struct IncreaseOp {
  template<typename T>
  __host__ __device__ static void apply(T& e) { e = e + 1; }
};

struct ExpOp {
  template<typename T>
  __host__ __device__ static void apply(T& x) { x = expf(x); }
};

struct ClipOp {
  template<typename T>
  __host__ __device__ static void apply(T& x) {
    x = fminf(fmaxf(x, MIN_EXP_F32), MAX_EXP_F32);
  }
};
template<>
__host__ __device__ void ClipOp::apply(half& x) {
  x = __hmin(__hmax(x, MIN_EXP_F16), MAX_EXP_F16);
}

// Convenience: standalone clip() for scalar use
template<typename T>
__host__ __device__ void clip(T& x) { ClipOp::apply(x); }

// Pure unary op: takes value, returns result. Used with 2-arg TNApply / Apply*(A, B).
struct SigmoidOp {
  template<typename T>
  __host__ __device__ static T apply(T x) {
    T neg = -x;
    ExpOp::apply(neg);
    return static_cast<T>(1) / (static_cast<T>(1) + neg);
  }
};
template<>
__host__ __device__ half SigmoidOp::apply(half x) {
  half neg = __hneg(x);
  ExpOp::apply(neg);
  return __float2half(1.0f) / (__float2half(1.0f) + neg);
}

struct ReluOp {
  template<typename T>
  __host__ __device__ static T apply(T x) {
    return x > static_cast<T>(0) ? x : static_cast<T>(0);
  }
};
template<>
__host__ __device__ half ReluOp::apply(half x) {
  return __hgt(x, __float2half(0.0f)) ? x : __float2half(0.0f);
}

struct EluOp {
  static constexpr float alpha = 1.0f;
  template<typename T>
  __host__ __device__ static T apply(T x) {
    return x > static_cast<T>(0) ? x
      : static_cast<T>(alpha) * (expf(static_cast<float>(x)) - static_cast<T>(1));
  }
};
template<>
__host__ __device__ half EluOp::apply(half x) {
  if (__hgt(x, __float2half(0.0f))) return x;
  float xf = __half2float(x);
  return __float2half(alpha * (expf(xf) - 1.0f));
}

struct GeluOp {
  template<typename T>
  __host__ __device__ static T apply(T x) {
    float xf = static_cast<float>(x);
    float v = sqrtf(2.0f / M_PI) * (xf + 0.044715f * xf * xf * xf);
    return static_cast<T>(0.5f * xf * (1.0f + tanhf(v)));
  }
};
template<>
__host__ __device__ half GeluOp::apply(half x) {
  float xf = __half2float(x);
  float v = sqrtf(2.0f / M_PI) * (xf + 0.044715f * xf * xf * xf);
  return __float2half(0.5f * xf * (1.0f + tanhf(v)));
}

struct SwishOp {
  template<typename T>
  __host__ __device__ static T apply(T x) {
    return x * SigmoidOp::apply(x);
  }
};
template<>
__host__ __device__ half SwishOp::apply(half x) {
  return __hmul(x, SigmoidOp::apply(x));
}

struct HardswishOp {
  template<typename T>
  __host__ __device__ static T apply(T x) {
    T six = static_cast<T>(6);
    T three = static_cast<T>(3);
    T zero = static_cast<T>(0);
    T r = x + three;
    r = r > six ? six : (r < zero ? zero : r);
    return x * r / six;
  }
};
template<>
__host__ __device__ half HardswishOp::apply(half x) {
  half six = __float2half(6.0f);
  half three = __float2half(3.0f);
  half zero = __float2half(0.0f);
  half r = __hadd(x, three);
  r = __hgt(r, six) ? six : (__hlt(r, zero) ? zero : r);
  return __hmul(x, __hdiv(r, six));
}

struct HardshrinkOp {
  static constexpr float lambda = 0.5f;
  template<typename T>
  __host__ __device__ static T apply(T x) {
    T lam = static_cast<T>(lambda);
    T neg_lam = static_cast<T>(-lambda);
    return (x > lam || x < neg_lam) ? x : static_cast<T>(0);
  }
};
template<>
__host__ __device__ half HardshrinkOp::apply(half x) {
  half lam = __float2half(lambda);
  half neg_lam = __float2half(-lambda);
  return (__hgt(x, lam) || __hlt(x, neg_lam)) ? x : __float2half(0.0f);
}

// ============================================================
// VecTraits — extract element type and vector width
// ============================================================

template<typename VecType> struct VecTraits;

template<> struct VecTraits<float> {
  using elem_type = float;
  static constexpr int width = 1;
};
template<> struct VecTraits<float4> {
  using elem_type = float;
  static constexpr int width = 4;
};
template<> struct VecTraits<half> {
  using elem_type = half;
  static constexpr int width = 1;
};
template<> struct VecTraits<half2> {
  using elem_type = half;
  static constexpr int width = 2;
};

// ============================================================
// SFINAE detection traits — compile-time checks for Op and TN
// ============================================================

namespace detail {

  // --- TN member detection (x, y, z, w) ---

  template<typename TN, typename = void>
  struct has_member_x : std::false_type {};
  template<typename TN>
  struct has_member_x<TN, decltype(static_cast<void>(&TN::x))> : std::true_type {};

  template<typename TN, typename = void>
  struct has_member_y : std::false_type {};
  template<typename TN>
  struct has_member_y<TN, decltype(static_cast<void>(&TN::y))> : std::true_type {};

  template<typename TN, typename = void>
  struct has_member_z : std::false_type {};
  template<typename TN>
  struct has_member_z<TN, decltype(static_cast<void>(&TN::z))> : std::true_type {};

  template<typename TN, typename = void>
  struct has_member_w : std::false_type {};
  template<typename TN>
  struct has_member_w<TN, decltype(static_cast<void>(&TN::w))> : std::true_type {};

  template<typename TN>
  inline constexpr bool has_xy_v = has_member_x<TN>::value && has_member_y<TN>::value;

  template<typename TN>
  inline constexpr bool has_xyzw_v = has_xy_v<TN>
    && has_member_z<TN>::value && has_member_w<TN>::value;

  // --- Extract TN's element type (via VecTraits, or TN itself as fallback) ---

  template<typename TN, typename = void>
  struct ElemTypeHelper { using type = TN; };

  template<typename TN>
  struct ElemTypeHelper<TN, std::void_t<typename VecTraits<TN>::elem_type>> {
    using type = typename VecTraits<TN>::elem_type;
  };

  template<typename TN>
  using elem_type_t = typename ElemTypeHelper<TN>::type;

  // --- Op::apply signature detection ---

  // Binary: Op::apply(T, T) -> T
  template<typename Op, typename T, typename = void>
  struct Op_is_binary : std::false_type {};
  template<typename Op, typename T>
  struct Op_is_binary<Op, T,
    std::void_t<decltype(Op::template apply<T>(
      std::declval<T>(), std::declval<T>()))>>
    : std::true_type {};

  // Unary: Op::apply(T&) or Op::apply(T) — accepts a mutable lvalue
  template<typename Op, typename T, typename = void>
  struct Op_is_unary : std::false_type {};
  template<typename Op, typename T>
  struct Op_is_unary<Op, T,
    std::void_t<decltype(Op::template apply<T>(std::declval<T&>()))>>
    : std::true_type {};

} // namespace detail

// ============================================================
// TNApply — internal dispatch (no SFINAE, called by Apply*)
// ============================================================

template<typename Op, typename TN, typename... Fields>
__host__ __device__ void TNApply(TN& A, TN& B, TN& C, Fields... fields) {
  if constexpr (sizeof...(fields) > 0) {
    ((C.*fields = Op::apply(A.*fields, B.*fields)), ...);
  }
}

template<typename Op, typename TN, typename... Fields>
__host__ __device__ void TNApply(TN& A, TN& B, Fields... fields) {
  if constexpr (sizeof...(fields) > 0) {
    ((B.*fields = Op::apply(A.*fields)), ...);
  }
}

template<typename Op, typename TN, typename... Fields>
__host__ __device__ void TNApply(TN& A, Fields... fields) {
  if constexpr (sizeof...(fields) > 0) {
    ((Op::apply(A.*fields)), ...);
  }
}

// ============================================================
// Apply4 — requires TN::{x,y,z,w} and Op::apply with matching arity
// ============================================================

template<typename Op, typename TN>
__host__ __device__ std::enable_if_t<
  detail::has_xyzw_v<TN>&& detail::Op_is_binary<Op, detail::elem_type_t<TN>>::value>
  Apply4(TN& A, TN& B, TN& C) {
  TNApply<Op>(A, B, C, &TN::x, &TN::y, &TN::z, &TN::w);
}

template<typename Op, typename TN>
__host__ __device__ std::enable_if_t<
  detail::has_xyzw_v<TN>&& detail::Op_is_unary<Op, detail::elem_type_t<TN>>::value>
  Apply4(TN& A, TN& B) {
  TNApply<Op>(A, B, &TN::x, &TN::y, &TN::z, &TN::w);
}

template<typename Op, typename TN>
__host__ __device__ std::enable_if_t<
  detail::has_xyzw_v<TN>&& detail::Op_is_unary<Op, detail::elem_type_t<TN>>::value>
  Apply4(TN& A) {
  TNApply<Op>(A, &TN::x, &TN::y, &TN::z, &TN::w);
}

// ============================================================
// Apply2 — requires TN::{x,y} and Op::apply with matching arity
// ============================================================

template<typename Op, typename TN>
__host__ __device__ std::enable_if_t<
  detail::has_xy_v<TN>&& detail::Op_is_binary<Op, detail::elem_type_t<TN>>::value>
  Apply2(TN& A, TN& B, TN& C) {
  TNApply<Op>(A, B, C, &TN::x, &TN::y);
}

template<typename Op, typename TN>
__host__ __device__ std::enable_if_t<
  detail::has_xy_v<TN>&& detail::Op_is_unary<Op, detail::elem_type_t<TN>>::value>
  Apply2(TN& A, TN& B) {
  TNApply<Op>(A, B, &TN::x, &TN::y);
}

template<typename Op, typename TN>
__host__ __device__ std::enable_if_t<
  detail::has_xy_v<TN>&& detail::Op_is_unary<Op, detail::elem_type_t<TN>>::value>
  Apply2(TN& A) {
  TNApply<Op>(A, &TN::x, &TN::y);
}

template<typename TN>
__device__ void SIGMOID2(TN& A, TN& B) {
  Apply2<SigmoidOp>(A, B);
}

template<typename TN>
__device__ void SIGMOID4(TN& A, TN& B) {
  Apply4<SigmoidOp>(A, B);
}


template<typename TN>
__host__ __device__ void CLIP4(TN& A) {
  Apply4<ClipOp>(A);
}

template<typename TN>
__host__ __device__ void CLIP2(TN& A) {
  Apply2<ClipOp>(A);
}

inline int x_by_y_ceil(int x, int y) {
  return (x + y - 1) / y;
}

inline int x_by_y_floor(int x, int y) {
  return x / y;
}

// ============================================================
// Warp/Block reduce utilities (used by softmax, layer_norm, etc.)
// ============================================================

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template<const int NUM_THREADS = 256>
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  __shared__ float smem[NUM_WARPS];

  // Step 1: warp-level reduce
  val = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0) smem[warp] = val;
  __syncthreads();

  // Step 2: warp 0 reduces all warp partial sums
  val = (lane < NUM_WARPS) ? smem[lane] : 0.f;
  if (warp == 0) val = warp_reduce_sum_f32<NUM_WARPS>(val);
  if (lane == 0) smem[0] = val;
  __syncthreads();

  // Step 3: broadcast final result to all threads
  return smem[0];
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template<const int NUM_THREADS = 256>
__device__ __forceinline__ float block_reduce_max_f32(float val) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  __shared__ float smem[NUM_WARPS];

  val = warp_reduce_max_f32<WARP_SIZE>(val);
  if (lane == 0) smem[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? smem[lane] : -FLT_MAX;
  if (warp == 0) val = warp_reduce_max_f32<NUM_WARPS>(val);
  if (lane == 0) smem[0] = val;
  __syncthreads();
  return smem[0];
}

// ============================================================
// Kernel class macros — generate a class with static run()
// that wraps a __global__ kernel launch.
//
// BINARY_KERNEL_CLASS(name, Kernel, ElemType, GridExpr)
//   name      — class name (e.g. elementwise_add_f32_kernel)
//   Kernel    — __global__ kernel function
//   ElemType  — pointer element type (float, half, float4, half2)
//   GridExpr  — grid size expression in terms of N and block
//
// UNARY_KERNEL_CLASS(name, Kernel, ElemType, GridExpr)
//   Same pattern for unary kernels (x, y, N).
// ============================================================

// Macro to define a kernel class with static run() methods.
// Kernel must be a __global__ function name (no template args).
// For template kernels, use the INSTANTIATE_* macros below.

#define BINARY_KERNEL_CLASS(name, Kernel, ElemType, GridExpr)                 \
struct name {                                                                 \
  static void run(ElemType* a, ElemType* b, ElemType* c, int N,              \
                  dim3 grid, dim3 block, cudaStream_t stream = 0) {           \
    Kernel<<<grid, block, 0, stream>>>(a, b, c, N);                          \
  }                                                                           \
  static void run(ElemType* a, ElemType* b, ElemType* c, int N) {            \
    dim3 block(256);                                                          \
    dim3 grid(GridExpr);                                                      \
    Kernel<<<grid, block>>>(a, b, c, N);                                     \
  }                                                                           \
}

#define UNARY_KERNEL_CLASS(name, Kernel, ElemType, GridExpr)                  \
struct name {                                                                 \
  static void run(ElemType* x, ElemType* y, int N,                           \
                  dim3 grid, dim3 block, cudaStream_t stream = 0) {           \
    Kernel<<<grid, block, 0, stream>>>(x, y, N);                             \
  }                                                                           \
  static void run(ElemType* x, ElemType* y, int N) {                         \
    dim3 block(256);                                                          \
    dim3 grid(GridExpr);                                                      \
    Kernel<<<grid, block>>>(x, y, N);                                        \
  }                                                                           \
}

#define UNARY_OP_KERNEL(name, ElemType, GridExpr, KernelBody)                 \
__global__ void _##name##_impl(ElemType* x, ElemType* y, int N) KernelBody   \
struct name {                                                                 \
  static void run(ElemType* x, ElemType* y, int N,                           \
                  dim3 grid, dim3 block, cudaStream_t stream = 0) {           \
    _##name##_impl<<<grid, block, 0, stream>>>(x, y, N);                     \
  }                                                                           \
  static void run(ElemType* x, ElemType* y, int N) {                         \
    dim3 block(256);                                                          \
    dim3 grid(GridExpr);                                                      \
    _##name##_impl<<<grid, block>>>(x, y, N);                                \
  }                                                                           \
}

#endif // UTILS_CUH