#include "src/models/nemotron_h_moe/kernels/rocm/nb_ops.h"

#include <hip/hip_fp16.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include "src/models/nemotron_h_moe/cpu_tables.hpp"

namespace cpu = gufo::models::nemotron_h_moe::cpu;

namespace {

constexpr int kBlock = 256;
constexpr int kRouterBlock = 512;
constexpr int kRouterMaxExperts = 512;

// ggml block_q8_0 layout, kept local to avoid pulling the vendored headers.
struct BlockQ8_0 {
  __half scale;
  signed char qs[32];
};

inline int BlockCount(int64_t n) {
  return static_cast<int>((n + kBlock - 1) / kBlock);
}

__global__ void RmsNormKernel(const float* __restrict__ x,
                              const float* __restrict__ w,
                              float* __restrict__ out, int hidden, float eps) {
  const int row = blockIdx.x;
  const float* xr = x + static_cast<int64_t>(row) * hidden;
  float* orow = out + static_cast<int64_t>(row) * hidden;
  __shared__ float warp_sums[8];
  float acc = 0.0F;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = xr[i];
    acc += v * v;
  }
  // Warp reduce (wave-size agnostic), then across warps via shared memory.
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(~0ULL, acc, off);
  }
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0F;
    for (int i = 0; i < static_cast<int>(blockDim.x / warpSize); ++i) {
      total += warp_sums[i];
    }
    warp_sums[0] = rsqrtf(total / static_cast<float>(hidden) + eps);
  }
  __syncthreads();
  const float scale = warp_sums[0];
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    orow[i] = (xr[i] * scale) * w[i];
  }
}

__global__ void RmsNormGroupedKernel(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     float* __restrict__ out, int hidden,
                                     int groups, float eps) {
  const int row = blockIdx.x;
  const float* xr = x + static_cast<int64_t>(row) * hidden;
  float* orow = out + static_cast<int64_t>(row) * hidden;
  const float* wr = w + static_cast<int64_t>(row % groups) * hidden;
  __shared__ float warp_sums[8];
  float acc = 0.0F;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = xr[i];
    acc += v * v;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(~0ULL, acc, off);
  }
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0F;
    for (int i = 0; i < static_cast<int>(blockDim.x / warpSize); ++i) {
      total += warp_sums[i];
    }
    warp_sums[0] = rsqrtf(total / static_cast<float>(hidden) + eps);
  }
  __syncthreads();
  const float scale = warp_sums[0];
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    orow[i] = (xr[i] * scale) * wr[i];
  }
}

__global__ void GemmF32Kernel(const float* __restrict__ x,
                              const float* __restrict__ w,
                              float* __restrict__ out, int n_out, int k) {
  const int t = blockIdx.y;
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n_out) {
    return;
  }
  const float* xr = x + static_cast<int64_t>(t) * k;
  const float* wr = w + static_cast<int64_t>(r) * k;
  float acc = 0.0F;
  for (int i = 0; i < k; ++i) {
    acc += xr[i] * wr[i];
  }
  out[static_cast<int64_t>(t) * n_out + r] = acc;
}

__global__ void ScaleKernel(float* __restrict__ x, float s, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] *= s;
  }
}

__global__ void AddKernel(float* __restrict__ res,
                          const float* __restrict__ add, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    res[i] += add[i];
  }
}

__global__ void ReluSqrKernel(const float* __restrict__ src,
                              float* __restrict__ dst, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float v = src[i];
    dst[i] = v > 0.0F ? v * v : 0.0F;
  }
}

__global__ void DequantQ8_0RowKernel(const uint8_t* __restrict__ w,
                                     float* __restrict__ out, int cols) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cols) {
    return;
  }
  const int b = i >> 5;
  const int j = i & 31;
  const BlockQ8_0* block = reinterpret_cast<const BlockQ8_0*>(w);
  out[i] = static_cast<float>(block[b].qs[j]) * __half2float(block[b].scale);
}

// ggml NVFP4: 64 values per block, one UE4M3 scale per 16 sub-block (d[4]),
// 32 packed E2M1 nibbles; row stride = K/64 blocks.
constexpr int kNvfp4K = 64;
constexpr int kNvfp4Sub = 16;

__device__ __constant__ float kE2M1[16] = {
    0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F,  6.0F,
    -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};

__device__ __forceinline__ float Ue4m3ToF32(uint8_t v) {
  const int e = (v >> 3) & 0xF;
  const int m = v & 7;
  if (e == 0) {
    return static_cast<float>(m) * (1.0F / 512.0F);
  }
  const uint32_t bits =
      static_cast<uint32_t>(e + 120) << 23 | static_cast<uint32_t>(m) << 20;
  float f;
  __builtin_memcpy(&f, &bits, sizeof(f));
  return f;
}

// One block per output row; thread s owns sub-block s (16 values).
__global__ void GemvNvfp4F32Kernel(const float* __restrict__ x,
                                   const uint8_t* __restrict__ w,
                                   float* __restrict__ out, int n_out, int k) {
  const int row = blockIdx.x;
  const int s = static_cast<int>(threadIdx.x);
  const int n_sub = k / kNvfp4Sub;
  const float* xr = x;
  const uint8_t* wr = w + static_cast<int64_t>(row) * (k / kNvfp4K) * 36;
  float acc = 0.0F;
  for (int b = s; b < n_sub; b += kBlock) {
    const int blk = b / (kNvfp4K / kNvfp4Sub);
    const int sub = b % (kNvfp4K / kNvfp4Sub);
    const uint8_t* wp = wr + static_cast<int64_t>(blk) * 36;
    const float d = Ue4m3ToF32(wp[sub]);
    const uint8_t* qs = wp + 4 + sub * (kNvfp4Sub / 2);
    const float* xv = xr + b * kNvfp4Sub;
    // Each qs byte [j] carries the pair (j, j + 8) of the 16-element
    // sub-block, matching block_nvfp4 / dequantize_row_nvfp4.
    for (int i = 0; i < kNvfp4Sub / 2; ++i) {
      const uint8_t byte = qs[i];
      acc += kE2M1[byte & 0xF] * d * xv[i];
      acc += kE2M1[byte >> 4] * d * xv[i + kNvfp4Sub / 2];
    }
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(~0ULL, acc, off);
  }
  __shared__ float warp[8];
  const int lane = s % warpSize;
  if (lane == 0) {
    warp[s / warpSize] = acc;
  }
  __syncthreads();
  if (s == 0) {
    float sum = 0.0F;
    for (int i = 0; i < kBlock / warpSize; ++i) {
      sum += warp[i];
    }
    out[row] = sum;
  }
}

// ---------------------------------------------------------------------------
// F32-activation GEMV over Q8_0 / IQ2_XXS / NVFP4 / F32 weights. Decodes are
// transcribed from cpu_ops.cpp (upstream dequantize_row_* layouts).
constexpr int kQ8_0Elems = 32;  // 34 B per 32-value block
constexpr int kQ8_0Block = 34;
constexpr int kIq2Elems = 256;  // 66 B per 256-value block
constexpr int kIq2Sub = 32;     // 8 B of codes per 32-value sub-block

__device__ __constant__ uint64_t cIq2Grid[256];
__device__ __constant__ uint8_t cIq2Signs[128];

__device__ __forceinline__ float BlockReduceSum(float acc) {
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(~0ULL, acc, off);
  }
  __shared__ float warp[8];
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  if (lane == 0) {
    warp[threadIdx.x / warpSize] = acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    for (int i = 0; i < kBlock / warpSize; ++i) {
      sum += warp[i];
    }
    return sum;
  }
  return 0.0F;
}

template<int TYPE>
__device__ __forceinline__ int GemvChunk();
template<>
__device__ __forceinline__ int GemvChunk<0>() {
  return 16;
}
template<>
__device__ __forceinline__ int GemvChunk<1>() {
  return kQ8_0Elems;
}
template<>
__device__ __forceinline__ int GemvChunk<2>() {
  return kIq2Sub;
}
template<>
__device__ __forceinline__ int GemvChunk<3>() {
  return kNvfp4Sub;
}

// Thread `t` owns chunk c of one row; each chunk covers GemvChunk<TYPE>
// dot products accumulated per-thread into a block reduce.
template<int TYPE>
__device__ __forceinline__ float GemvChunkDot(const uint8_t* __restrict__ w,
                                              const float* __restrict__ x,
                                              int chunk, int k) {
  float acc = 0.0F;
  if (TYPE == kNblGemvF32) {
    const float* wr = reinterpret_cast<const float*>(w);
    const float4* xr = reinterpret_cast<const float4*>(x + chunk * 16);
    const float4* wv = reinterpret_cast<const float4*>(wr + chunk * 16);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 a4 = wv[i];
      const float4 x4 = xr[i];
      acc += a4.x * x4.x + a4.y * x4.y + a4.z * x4.z + a4.w * x4.w;
    }
  } else if (TYPE == kNblGemvQ8_0) {
    const uint8_t* blk = w + static_cast<int64_t>(chunk) * kQ8_0Block;
    const uint16_t dh =
        static_cast<uint16_t>(blk[0]) | static_cast<uint16_t>(blk[1] << 8);
    __half h;
    __builtin_memcpy(&h, &dh, sizeof(h));
    const float d = __half2float(h);
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
    const float* xv = x + chunk * kQ8_0Elems;
#pragma unroll
    for (int j = 0; j < kQ8_0Elems; ++j) {
      acc += static_cast<float>(qs[j]) * d * xv[j];
    }
  } else if (TYPE == kNblGemvIq2Xxs) {
    const uint8_t* blk = w + static_cast<int64_t>(chunk / 8) * 66;
    uint16_t dh = 0;
    uint32_t aux32[2] = {0, 0};
    __builtin_memcpy(&dh, blk, sizeof(dh));
    __builtin_memcpy(aux32, blk + 2 + (chunk % 8) * 8, sizeof(aux32));
    const float d = __half2float(*reinterpret_cast<__half*>(&dh));
    const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
    const float db = d * (0.5F + static_cast<float>(aux32[1] >> 28)) * 0.25F;
    const float* xv = x + chunk * kIq2Sub;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint64_t grid = cIq2Grid[aux8[l]];
      const uint8_t signs = cIq2Signs[(aux32[1] >> (7 * l)) & 127];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int8_t v = static_cast<int8_t>((grid >> (8 * j)) & 0xFF);
        acc += db * static_cast<float>(v) *
               ((signs & (1u << j)) != 0 ? -xv[l * 8 + j] : xv[l * 8 + j]);
      }
    }
  } else {
    const int blk = chunk / (kNvfp4K / kNvfp4Sub);
    const int sub = chunk % (kNvfp4K / kNvfp4Sub);
    const uint8_t* wp = w + static_cast<int64_t>(blk) * 36;
    const float d = Ue4m3ToF32(wp[sub]);
    const uint8_t* qs = wp + 4 + sub * (kNvfp4Sub / 2);
    const float* xv = x + chunk * kNvfp4Sub;
#pragma unroll
    for (int i = 0; i < kNvfp4Sub / 2; ++i) {
      const uint8_t byte = qs[i];
      acc += kE2M1[byte & 0xF] * d * xv[i];
      acc += kE2M1[byte >> 4] * d * xv[i + kNvfp4Sub / 2];
    }
  }
  return acc;
}

template<int TYPE>
__global__ void GemvF32Kernel(const float* __restrict__ x,
                              const uint8_t* __restrict__ w,
                              float* __restrict__ out, int n_out, int k,
                              int expert_stride,
                              const int32_t* __restrict__ ids, int slots,
                              int64_t x_stride) {
  const int row = static_cast<int>(blockIdx.x % static_cast<unsigned>(n_out));
  int64_t w_row = row;
  int64_t out_row = blockIdx.x;
  if (slots > 1) {
    const int slot = static_cast<int>(blockIdx.x / n_out);
    w_row = static_cast<int64_t>(__ldg(ids + slot)) * n_out + row;
    out_row = static_cast<int64_t>(slot) * n_out + row;
    x += static_cast<int64_t>(slot) * x_stride;
  }
  const uint8_t* wr = w + w_row * static_cast<int64_t>(expert_stride);
  const int n_chunk = k / GemvChunk<TYPE>();
  float acc = 0.0F;
  for (int c = static_cast<int>(threadIdx.x); c < n_chunk;
       c += static_cast<int>(blockDim.x)) {
    acc += GemvChunkDot<TYPE>(wr, x, c, k);
  }
  acc = BlockReduceSum(acc);
  if (threadIdx.x == 0) {
    out[out_row] = acc;
  }
}

__global__ void RouterKernel(const float* __restrict__ logits,
                             const float* __restrict__ bias, int n_experts,
                             int top_k, int norm, float scale,
                             int32_t* __restrict__ ids,
                             float* __restrict__ weights) {
  const int token = blockIdx.x;
  const float* lg = logits + static_cast<int64_t>(token) * n_experts;
  __shared__ float scores[kRouterMaxExperts];
  __shared__ int sel[16];
  __shared__ float num[16];
  for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
    scores[e] = 1.0F / (1.0F + expf(-lg[e])) + bias[e];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
    for (int k = 0; k < top_k; ++k) {
      // Serial top-k selection on one thread mirrors the oracle loop
      // exactly, including the tie-to-lowest-index rule.
      int best = -1;
      for (int e = 0; e < n_experts; ++e) {
        const float v = scores[e];
        if (v == -FLT_MAX) {
          continue;
        }
        if (best < 0 || v > scores[best]) {
          best = e;
        }
      }
      sel[k] = best;
      if (best < 0) {
        break;
      }
      total = k + 1;
      scores[best] = -FLT_MAX;
      num[k] = 1.0F / (1.0F + expf(-lg[best]));
    }
    float sum = 0.0F;
    for (int k = 0; k < total; ++k) {
      sum += num[k];
    }
    if (norm) {
      sum = sum < 6.103515625e-5F ? 6.103515625e-5F : sum;
    }
    const float inv = sum > 0.0F ? 1.0F / sum : 0.0F;
    for (int k = 0; k < top_k; ++k) {
      const bool chosen = k < total;
      ids[static_cast<int64_t>(token) * top_k + k] =
          chosen ? sel[k] : static_cast<int32_t>(-1);
      weights[static_cast<int64_t>(token) * top_k + k] =
          chosen ? num[k] * inv * scale : 0.0F;
    }
  }
}

__global__ void MoeCombineKernel(float* __restrict__ dst,
                                 const float* __restrict__ src,
                                 const float* __restrict__ w, int m,
                                 int top_k) {
  const int token = blockIdx.y;
  const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= m) {
    return;
  }
  float acc = 0.0F;
  for (int u = 0; u < top_k; ++u) {
    acc += w[static_cast<int64_t>(token) * top_k + u] *
           src[(static_cast<int64_t>(token) * top_k + u) * m + j];
  }
  dst[static_cast<int64_t>(token) * m + j] = acc;
}

int CheckLaunch(hipError_t err, const char* tag) {
  if (err != hipSuccess) {
    return -1;
  }
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

}  // namespace

extern "C" int nbl_rms_norm(const float* x, const float* w, float* out,
                            int rows, int hidden, float eps,
                            hipStream_t stream) {
  if (!x || !w || !out || rows <= 0 || hidden <= 0) {
    return -1;
  }
  RmsNormKernel<<<rows, kBlock, 0, stream>>>(x, w, out, hidden, eps);
  return CheckLaunch(hipGetLastError(), "nbl_rms_norm");
}

extern "C" int nbl_rms_norm_grouped(const float* x, const float* w, float* out,
                                    int rows, int hidden, int groups, float eps,
                                    hipStream_t stream) {
  if (!x || !w || !out || rows <= 0 || hidden <= 0 || groups <= 0) {
    return -1;
  }
  RmsNormGroupedKernel<<<rows, kBlock, 0, stream>>>(x, w, out, hidden, groups,
                                                    eps);
  return CheckLaunch(hipGetLastError(), "nbl_rms_norm_grouped");
}

extern "C" int nbl_gemm_f32(const float* x, const float* w, float* out,
                            int n_tokens, int n_out, int k,
                            hipStream_t stream) {
  if (!x || !w || !out || n_tokens <= 0 || n_out <= 0 || k <= 0) {
    return -1;
  }
  const dim3 grid((n_out + kBlock - 1) / kBlock, n_tokens);
  GemmF32Kernel<<<grid, kBlock, 0, stream>>>(x, w, out, n_out, k);
  return CheckLaunch(hipGetLastError(), "nbl_gemm_f32");
}

extern "C" int nbl_gemv_nvfp4_f32(const float* x, const void* w, float* out,
                                  int n_out, int k, hipStream_t stream) {
  if (!x || !w || !out || n_out <= 0 || k <= 0 || k % kNvfp4K != 0) {
    return -1;
  }
  GemvNvfp4F32Kernel<<<n_out, kBlock, 0, stream>>>(
      x, static_cast<const uint8_t*>(w), out, n_out, k);
  return CheckLaunch(hipGetLastError(), "nbl_gemv_nvfp4_f32");
}

int GemvRowBytes(int type, int k) {
  switch (type) {
    case kNblGemvF32:
      return k * 4;
    case kNblGemvQ8_0:
      return k / kQ8_0Elems * kQ8_0Block;
    case kNblGemvIq2Xxs:
      return k / kIq2Elems * 66;
    default:
      return k / kNvfp4K * 36;
  }
}

bool GemvTypeValid(int type, int k) {
  switch (type) {
    case kNblGemvF32:
      return k % 16 == 0;
    case kNblGemvQ8_0:
      return k % kQ8_0Elems == 0;
    case kNblGemvIq2Xxs:
      return k % kIq2Elems == 0;
    case kNblGemvNvfp4:
      return k % kNvfp4K == 0;
    default:
      return false;
  }
}

void EnsureIq2Tables() {
  static bool loaded = false;
  if (!loaded) {
    (void)hipMemcpyToSymbol(cIq2Grid, cpu::kIq2XxsGrid,
                            sizeof(cpu::kIq2XxsGrid));
    (void)hipMemcpyToSymbol(cIq2Signs, cpu::kSignsIq2Xs,
                            sizeof(cpu::kSignsIq2Xs));
    loaded = true;
  }
}

template<bool kMoe>
int LaunchGemv(int type, const void* w, const float* x, const int32_t* ids,
               float* out, int n_out, int k, int slots, int64_t x_stride,
               hipStream_t stream) {
  const int64_t rows = kMoe ? static_cast<int64_t>(slots) * n_out : n_out;
  const int stride = GemvRowBytes(type, k);
  const uint8_t* wb = static_cast<const uint8_t*>(w);
  switch (type) {
    case kNblGemvF32:
      GemvF32Kernel<kNblGemvF32><<<rows, kBlock, 0, stream>>>(
          x, wb, out, n_out, k, stride, ids, kMoe ? slots : 0, x_stride);
      break;
    case kNblGemvQ8_0:
      GemvF32Kernel<kNblGemvQ8_0><<<rows, kBlock, 0, stream>>>(
          x, wb, out, n_out, k, stride, ids, kMoe ? slots : 0, x_stride);
      break;
    case kNblGemvIq2Xxs:
      EnsureIq2Tables();
      GemvF32Kernel<kNblGemvIq2Xxs><<<rows, kBlock, 0, stream>>>(
          x, wb, out, n_out, k, stride, ids, kMoe ? slots : 0, x_stride);
      break;
    default:
      GemvF32Kernel<kNblGemvNvfp4><<<rows, kBlock, 0, stream>>>(
          x, wb, out, n_out, k, stride, ids, kMoe ? slots : 0, x_stride);
      break;
  }
  return CheckLaunch(hipGetLastError(),
                     kMoe ? "nbl_gemv_moe_f32" : "nbl_gemv_f32");
}

extern "C" int nbl_gemv_f32(int type, const void* w, const float* x, float* out,
                            int n_out, int k, hipStream_t stream) {
  if (!x || !w || !out || n_out <= 0 || k <= 0 || !GemvTypeValid(type, k)) {
    return -1;
  }
  return LaunchGemv<false>(type, w, x, nullptr, out, n_out, k, 0, 0, stream);
}

extern "C" int nbl_gemv_moe_f32(int type, const void* w, const float* x,
                                const int32_t* ids, float* out, int n_out,
                                int k, int slots, int64_t x_stride,
                                hipStream_t stream) {
  if (!x || !w || !ids || !out || n_out <= 0 || k <= 0 || slots <= 0 ||
      x_stride < 0 || !GemvTypeValid(type, k)) {
    return -1;
  }
  return LaunchGemv<true>(type, w, x, ids, out, n_out, k, slots, x_stride,
                          stream);
}

extern "C" int nbl_scale(float* x, float s, int64_t n, hipStream_t stream) {
  if (!x || n <= 0) {
    return -1;
  }
  ScaleKernel<<<BlockCount(n), kBlock, 0, stream>>>(x, s, n);
  return CheckLaunch(hipGetLastError(), "nbl_scale");
}

extern "C" int nbl_add(float* res, const float* add, int64_t n,
                       hipStream_t stream) {
  if (!res || !add || n <= 0) {
    return -1;
  }
  AddKernel<<<BlockCount(n), kBlock, 0, stream>>>(res, add, n);
  return CheckLaunch(hipGetLastError(), "nbl_add");
}

extern "C" int nbl_relu_sqr(const float* src, float* dst, int64_t n,
                            hipStream_t stream) {
  if (!src || !dst || n <= 0) {
    return -1;
  }
  ReluSqrKernel<<<BlockCount(n), kBlock, 0, stream>>>(src, dst, n);
  return CheckLaunch(hipGetLastError(), "nbl_relu_sqr");
}

extern "C" int nbl_dequant_row_q8_0(const void* w, int64_t row, int cols,
                                    float* out, hipStream_t stream) {
  if (!w || !out || row < 0 || cols <= 0 || cols % 32 != 0) {
    return -1;
  }
  const uint8_t* base = static_cast<const uint8_t*>(w) +
                        row * (static_cast<int64_t>(cols) / 32 *
                               static_cast<int64_t>(sizeof(BlockQ8_0)));
  DequantQ8_0RowKernel<<<BlockCount(cols), kBlock, 0, stream>>>(base, out,
                                                                cols);
  return CheckLaunch(hipGetLastError(), "nbl_dequant_row_q8_0");
}

extern "C" int nbl_router(const float* logits, const float* bias, int n_experts,
                          int top_k, int norm, float scale, int n_tokens,
                          int32_t* ids, float* weights, hipStream_t stream) {
  if (!logits || !bias || !ids || !weights || n_experts <= 0 ||
      n_experts > kRouterMaxExperts || top_k <= 0 || top_k > 16 ||
      n_tokens <= 0) {
    return -1;
  }
  RouterKernel<<<n_tokens, kRouterBlock, 0, stream>>>(
      logits, bias, n_experts, top_k, norm, scale, ids, weights);
  return CheckLaunch(hipGetLastError(), "nbl_router");
}

extern "C" int nbl_moe_combine(float* dst, const float* src, const float* w,
                               int m, int top_k, int n_tokens,
                               hipStream_t stream) {
  if (!dst || !src || !w || m <= 0 || top_k <= 0 || n_tokens <= 0) {
    return -1;
  }
  const dim3 grid(BlockCount(m), n_tokens);
  MoeCombineKernel<<<grid, kBlock, 0, stream>>>(dst, src, w, m, top_k);
  return CheckLaunch(hipGetLastError(), "nbl_moe_combine");
}
