#include "src/models/nemotron_h_moe/kernels/rocm/nb_ops.h"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>

#include "src/models/nemotron_h_moe/cpu_tables.hpp"

namespace cpu = gufo::models::nemotron_h_moe::cpu;

namespace {

constexpr int kBlock = 256;
constexpr int kRouterBlock = 512;
constexpr int kRouterMaxExperts = 512;
constexpr int kRouterWarpCount =
    kRouterBlock / 32;  // gfx1151 warpSize is 32 (runtime value is not const)

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

// Fallback for k not divisible by the tile k-step (never hit by Nemotron
// Bolt, whose router k is the 4096-wide hidden size).
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

namespace {

constexpr int kRingRows = 8;      // expert rows owned by one block
constexpr int kRingStageK = 128;  // k-width of one cooperative float tile
constexpr int kRingDepth = 2;     // tiles in flight (double buffered)

}  // namespace

// Decode router GEMM for one few-token pass: a block owns kRingRows expert
// rows; all 256 threads float4-stream kRingStageK-wide tiles into LDS with a
// kRingDepth-deep prefetch, while the first kRingRows threads fold their row
// from LDS in ascending k. The FMA order matches GemmF32Kernel exactly, so
// results are bitwise identical, but the serial row chain runs at LDS/FMA
// latency with the next tile's HBM round trip already in flight instead of
// stalling on each weight load. Probe:
// tools/bench/nem_decode_router_gemm_bench.hip (L2-cold 368x4096: 357us ->
// 73us, 4.9x; all shapes bitwise identical).
__global__ void GemmF32RingKernel(const float* __restrict__ x,
                                  const float* __restrict__ w,
                                  float* __restrict__ out, int n_out, int k) {
  const int t = blockIdx.y;
  const int r0 = blockIdx.x * kRingRows;
  const int tid = static_cast<int>(threadIdx.x);
  __shared__ float ws[kRingDepth][kRingRows][kRingStageK + 1];
  const float* xr = x + static_cast<int64_t>(t) * k;

  const bool adder = (tid < kRingRows) && (r0 + tid < n_out);
  float acc = 0.0F;
  const int g = tid / 32;
  const int kl = (tid % 32) * 4;
  const int loader_ok = (r0 + g) < n_out ? 1 : 0;
  const float* wrow = w + static_cast<int64_t>(r0 + g) * k + kl;
  const int npass = k / kRingStageK;
  float4 v[kRingDepth];
  v[0] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
#pragma unroll
  for (int d = 0; d < kRingDepth; ++d) {
    v[d] = (loader_ok && npass > d)
               ? *reinterpret_cast<const float4*>(wrow + d * kRingStageK)
               : v[0];
  }
  for (int p = 0; p < npass; ++p) {
    const int cur = p % kRingDepth;
    ws[cur][g][kl] = v[0].x;
    ws[cur][g][kl + 1] = v[0].y;
    ws[cur][g][kl + 2] = v[0].z;
    ws[cur][g][kl + 3] = v[0].w;
    __syncthreads();
    v[0] = v[1];
    if (p + kRingDepth < npass && loader_ok) {
      v[1] = *reinterpret_cast<const float4*>(wrow +
                                              (p + kRingDepth) * kRingStageK);
    }
    if (adder) {
      const float* wr = ws[cur][tid];
      const float* xrk = xr + p * kRingStageK;
#pragma unroll
      for (int q = 0; q < kRingStageK / 4; ++q) {
        const float4 xv = *reinterpret_cast<const float4*>(xrk + 4 * q);
        acc += xv.x * wr[4 * q];
        acc += xv.y * wr[4 * q + 1];
        acc += xv.z * wr[4 * q + 2];
        acc += xv.w * wr[4 * q + 3];
      }
    }
    __syncthreads();
  }
  if (adder) {
    out[static_cast<int64_t>(t) * n_out + r0 + tid] = acc;
  }
}

// Router GEMM: out[m, n_out] = x[m, k] w[n_out, k]^T with float32 inputs.
// A block covers 64 tokens x 64 experts with a 32-wide LDS k-stage, 256
// threads and 8 tokens x 2 experts of register accumulation per thread.
// This tile was the winner of a gfx1151 sweep (see
// tools/bench/nem_router_bench.hip): ~20x the thread-per-output mapping at
// pp16384, because each LDS load feeds 16 FMAs instead of one.
namespace {

constexpr int kGemmTileTokens = 64;
constexpr int kGemmTileExperts = 64;
constexpr int kGemmKStep = 32;
constexpr int kGemmThreads = 256;
constexpr int kGemmThreadGridR = 32;  // expert groups of 2 across 32 lanes
constexpr int kGemmThreadGridT = kGemmThreads / kGemmThreadGridR;
constexpr int kGemmThreadTokensN = kGemmTileTokens / kGemmThreadGridT;
constexpr int kGemmThreadExpertsN = kGemmTileExperts / kGemmThreadGridR;

}  // namespace

__global__ void GemmF32TiledKernel(const float* __restrict__ x,
                                   const float* __restrict__ w,
                                   float* __restrict__ out, int n_out, int k,
                                   int n_tokens) {
  const int t0 = blockIdx.x * kGemmTileTokens;
  const int r0 = blockIdx.y * kGemmTileExperts;
  const int tid = threadIdx.x;
  const int ty = tid / kGemmThreadGridR;
  const int tx = tid % kGemmThreadGridR;
  __shared__ float xs[kGemmTileTokens][kGemmKStep + 1];
  __shared__ float ws[kGemmTileExperts][kGemmKStep + 1];
  const float4* xp =
      reinterpret_cast<const float4*>(x + static_cast<int64_t>(t0) * k);
  const float4* wp =
      reinterpret_cast<const float4*>(w + static_cast<int64_t>(r0) * k);
  float acc[kGemmThreadTokensN][kGemmThreadExpertsN] = {};
  for (int kc = 0; kc < k; kc += kGemmKStep) {
    for (int i = tid; i < kGemmTileTokens * (kGemmKStep / 4);
         i += kGemmThreads) {
      const int row = i / (kGemmKStep / 4);
      const int c4 = i % (kGemmKStep / 4);
      float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      if (t0 + row < n_tokens) {
        v = *reinterpret_cast<const float4*>(
            xp + static_cast<int64_t>(row) * (k / 4) + (kc / 4) + c4);
      }
      xs[row][c4 * 4 + 0] = v.x;
      xs[row][c4 * 4 + 1] = v.y;
      xs[row][c4 * 4 + 2] = v.z;
      xs[row][c4 * 4 + 3] = v.w;
    }
    for (int i = tid; i < kGemmTileExperts * (kGemmKStep / 4);
         i += kGemmThreads) {
      const int row = i / (kGemmKStep / 4);
      const int c4 = i % (kGemmKStep / 4);
      float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      if (r0 + row < n_out) {
        v = *reinterpret_cast<const float4*>(
            wp + static_cast<int64_t>(row) * (k / 4) + (kc / 4) + c4);
      }
      ws[row][c4 * 4 + 0] = v.x;
      ws[row][c4 * 4 + 1] = v.y;
      ws[row][c4 * 4 + 2] = v.z;
      ws[row][c4 * 4 + 3] = v.w;
    }
    __syncthreads();
#pragma unroll 1
    for (int e = 0; e < kGemmKStep; ++e) {
      float ra[kGemmThreadTokensN], rb[kGemmThreadExpertsN];
#pragma unroll
      for (int i = 0; i < kGemmThreadTokensN; ++i) {
        ra[i] = xs[ty * kGemmThreadTokensN + i][e];
      }
#pragma unroll
      for (int j = 0; j < kGemmThreadExpertsN; ++j) {
        rb[j] = ws[tx * kGemmThreadExpertsN + j][e];
      }
#pragma unroll
      for (int i = 0; i < kGemmThreadTokensN; ++i) {
#pragma unroll
        for (int j = 0; j < kGemmThreadExpertsN; ++j) {
          acc[i][j] += ra[i] * rb[j];
        }
      }
    }
    __syncthreads();
  }
#pragma unroll 1
  for (int i = 0; i < kGemmThreadTokensN; ++i) {
    const int t = t0 + ty * kGemmThreadTokensN + i;
    if (t >= n_tokens) {
      break;
    }
#pragma unroll
    for (int j = 0; j < kGemmThreadExpertsN; ++j) {
      const int r = r0 + tx * kGemmThreadExpertsN + j;
      if (r < n_out) {
        out[static_cast<int64_t>(t) * n_out + r] = acc[i][j];
      }
    }
  }
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

// One block row per token: out[t] = embedding row ids[t].
__global__ void DequantQ8_0RowsKernel(const uint8_t* __restrict__ w,
                                      const int32_t* __restrict__ ids,
                                      float* __restrict__ out, int cols) {
  const int t = static_cast<int>(blockIdx.y);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cols) {
    return;
  }
  const int b = i >> 5;
  const int j = i & 31;
  const int64_t row = __ldg(ids + t);
  const BlockQ8_0* block = reinterpret_cast<const BlockQ8_0*>(
      w + row * (static_cast<int64_t>(cols) / 32 * sizeof(BlockQ8_0)));
  out[static_cast<int64_t>(t) * cols + i] =
      static_cast<float>(block[b].qs[j]) * __half2float(block[b].scale);
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
                                   float* __restrict__ out, int n_out, int k,
                                   int tokens) {
  const int row = blockIdx.x;
  const int token = static_cast<int>(blockIdx.y);
  const int s = static_cast<int>(threadIdx.x);
  const int n_sub = k / kNvfp4Sub;
  const float* xr = x + static_cast<int64_t>(token) * k;
  out += static_cast<int64_t>(token) * n_out;
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
// [aux] -> XOR of the grid word with its byte-negated word. With the
// 0xFF-byte sign-mask table below, the signed grid word is R = G ^ (m & X):
// a byte-lane select of g and -g with no per-lane subtraction or carry.
__device__ __constant__ uint64_t cIq2Xor[256];
// [7-bit sign index] -> {byte i = 0xFF iff element i (i=0..3) flips,
//                        byte i = 0xFF iff element i+4 flips}: spreads the
// packed uint8 sign bits of kSignsIq2Xs for the XOR-negate grid trick.
__device__ __constant__ uint2 cIq2SignMasks[128];

// ggml block_q8_1: half2 (d, sum) followed by 32 int8 values. Produced by
// nbl_quantize_q8_1 for the kNblGemvIq2XxsQ8_1 GEMV.
struct NblQ8_1 {
  __half2 ds;
  int8_t qs[kIq2Sub];
};
static_assert(sizeof(NblQ8_1) == kIq2Sub + 4, "q8_1 block must be 36 bytes");

// K-quant / IQ4_XS super-blocks (ggml block_q*_K / block_iq4_xs, 256 values
// per block) accepted by the F32-activation GEMV tiers for the requantized
// (Unsloth UD) artifacts. Layouts and value semantics mirror
// src/core/quant/ggml_dequant.hpp verbatim (verified against GGUF bytes).
constexpr int kKqElems = 256;
struct NblQ4_K {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[128];
};
struct NblQ5_K {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qh[32];
  uint8_t qs[128];
};
struct NblQ6_K {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  uint16_t d;
};
struct NblIq4Xs {
  uint16_t d;
  uint16_t scales_h;
  uint8_t scales_l[4];
  uint8_t qs[128];
};
static_assert(sizeof(NblQ4_K) == 144, "q4_K block must be 144 bytes");
static_assert(sizeof(NblQ5_K) == 176, "q5_K block must be 176 bytes");
static_assert(sizeof(NblQ6_K) == 210, "q6_K block must be 210 bytes");
static_assert(sizeof(NblIq4Xs) == 136, "iq4_xs block must be 136 bytes");

// Non-linear 4-bit codebook shared by IQ4_NL and IQ4_XS (ggml values_iq4_nl).
constexpr float kIq4Nl[16] = {-127.0F, -104.0F, -83.0F, -65.0F, -49.0F, -35.0F,
                              -22.0F,  -10.0F,  1.0F,   13.0F,  25.0F,  38.0F,
                              53.0F,   69.0F,   89.0F,  113.0F};

__device__ __forceinline__ float Fp16ToF32(uint16_t h) {
  return __half2float(*reinterpret_cast<__half*>(&h));
}

// Unpack the 6-bit sub-block scale/minimum of a Q4_K/Q5_K block (ggml
// get_scale_min_k4): index 0..7, two nibble fields plus the two-bit tail.
__device__ __forceinline__ void QScaleMin(int index, const uint8_t* packed,
                                          float d, float dmin, float& scale,
                                          float& minimum) {
  if (index < 4) {
    scale = d * static_cast<float>(packed[index] & 0x3FU);
    minimum = dmin * static_cast<float>(packed[index + 4] & 0x3FU);
    return;
  }
  scale = d * static_cast<float>((packed[index + 4] & 0x0FU) |
                                 ((packed[index - 4] >> 6U) << 4U));
  minimum = dmin * static_cast<float>((packed[index + 4] >> 4U) |
                                      ((packed[index] >> 6U) << 4U));
}

// One GEMV chunk is one 256-value super-block, thread-owned; the value
// extraction order follows the CPU oracle exactly.
__device__ __forceinline__ float GemvChunkDotQ4_K(const uint8_t* w,
                                                  const float* x, int chunk) {
  const NblQ4_K* b = reinterpret_cast<const NblQ4_K*>(w) + chunk;
  const float d = Fp16ToF32(b->d);
  const float dmin = Fp16ToF32(b->dmin);
  const float* xv = x + static_cast<int64_t>(chunk) * kKqElems;
  float acc = 0.0F;
  for (int g = 0; g < 8; ++g) {
    float s = 0.0F;
    float m = 0.0F;
    QScaleMin(g, b->scales, d, dmin, s, m);
    const uint8_t* q = b->qs + (g / 2) * 32;
    if ((g & 1) == 0) {
      for (int l = 0; l < 32; ++l) {
        acc += (s * static_cast<float>(q[l] & 0x0FU) - m) * xv[g * 32 + l];
      }
    } else {
      for (int l = 0; l < 32; ++l) {
        acc += (s * static_cast<float>(q[l] >> 4U) - m) * xv[g * 32 + l];
      }
    }
  }
  return acc;
}

__device__ __forceinline__ float GemvChunkDotQ5_K(const uint8_t* w,
                                                  const float* x, int chunk) {
  const NblQ5_K* b = reinterpret_cast<const NblQ5_K*>(w) + chunk;
  const float d = Fp16ToF32(b->d);
  const float dmin = Fp16ToF32(b->dmin);
  const float* xv = x + static_cast<int64_t>(chunk) * kKqElems;
  float acc = 0.0F;
  for (int gg = 0; gg < 4; ++gg) {
    for (int half = 0; half < 2; ++half) {
      float s = 0.0F;
      float m = 0.0F;
      QScaleMin(2 * gg + half, b->scales, d, dmin, s, m);
      const uint8_t* q = b->qs + gg * 32;
      for (int l = 0; l < 32; ++l) {
        const int qb = q[l];
        const int low = half == 0 ? (qb & 0x0F) : (qb >> 4U);
        const int hi = (b->qh[l] >> (2 * gg + half)) & 1;
        acc += (s * static_cast<float>(low + (hi << 4)) - m) *
               xv[gg * 64 + half * 32 + l];
      }
    }
  }
  return acc;
}

__device__ __forceinline__ float GemvChunkDotQ6_K(const uint8_t* w,
                                                  const float* x, int chunk) {
  const NblQ6_K* b = reinterpret_cast<const NblQ6_K*>(w) + chunk;
  const float d = Fp16ToF32(b->d);
  const float* xv = x + static_cast<int64_t>(chunk) * kKqElems;
  float acc = 0.0F;
  for (int half = 0; half < 2; ++half) {
    for (int lane = 0; lane < 32; ++lane) {
      const uint8_t qh = b->qh[half * 32 + lane];
      const uint8_t b0 = b->ql[half * 64 + lane];
      const uint8_t b1 = b->ql[half * 64 + 32 + lane];
      const int hi16 = lane >= 16 ? 1 : 0;
      const float s0 = d * static_cast<float>(b->scales[half * 8 + 0 + hi16]);
      const float s1 = d * static_cast<float>(b->scales[half * 8 + 2 + hi16]);
      const float s2 = d * static_cast<float>(b->scales[half * 8 + 4 + hi16]);
      const float s3 = d * static_cast<float>(b->scales[half * 8 + 6 + hi16]);
      const int base = half * 128 + lane;
      acc += s0 *
                 static_cast<float>(static_cast<int>((qh & 0x03U) << 4) +
                                    static_cast<int>(b0 & 0x0FU) - 32) *
                 xv[base] +
             s1 *
                 static_cast<float>(static_cast<int>((qh >> 2U & 0x03U) << 4) +
                                    static_cast<int>(b1 & 0x0FU) - 32) *
                 xv[base + 32] +
             s2 *
                 static_cast<float>(static_cast<int>((qh >> 4U & 0x03U) << 4) +
                                    static_cast<int>(b0 >> 4U) - 32) *
                 xv[base + 64] +
             s3 *
                 static_cast<float>(static_cast<int>((qh >> 6U) << 4) +
                                    static_cast<int>(b1 >> 4U) - 32) *
                 xv[base + 96];
    }
  }
  return acc;
}

__device__ __forceinline__ float GemvChunkDotIq4Xs(const uint8_t* w,
                                                   const float* x, int chunk) {
  const NblIq4Xs* b = reinterpret_cast<const NblIq4Xs*>(w) + chunk;
  const float d = Fp16ToF32(b->d);
  const float* xv = x + static_cast<int64_t>(chunk) * kKqElems;
  float acc = 0.0F;
  for (int ib = 0; ib < 8; ++ib) {
    const int ls =
        static_cast<int>((b->scales_l[ib / 2] >> (4U * (ib % 2))) & 0x0FU) |
        static_cast<int>(((b->scales_h >> (2U * ib)) & 0x03U) << 4U);
    const float s = d * static_cast<float>(ls - 32);
    const uint8_t* q = b->qs + ib * 16;
    for (int l = 0; l < 16; ++l) {
      const uint8_t p = q[l];
      acc += s * (kIq4Nl[p & 0x0FU] * xv[ib * 32 + l] +
                  kIq4Nl[p >> 4U] * xv[ib * 32 + 16 + l]);
    }
  }
  return acc;
}

// Warp-parallel super-block dots: lane owns the eight consecutive values
// v = 8*lane + p of one 256-value block, so the weight bytes of a lane are
// one contiguous 8-byte window and the activations two float4 loads. The
// thread-owned chunk dots above fetch every byte and value scalar-wise,
// which caps the wide decode rows (Q6_K ssm_in at ~16 GB/s) far below the
// measured ~240 GB/s ceiling. The per-warp accumulation order differs from
// the chunk dots; the deviation is ULP-scale, orders of magnitude inside
// the pinned device_step contract.
//
// LoadBytes8 re-aligns the lane window: the Q6_K ggml block stride is 210
// bytes, so block-relative offsets hit every 4-byte residue; the funnel
// shift reads the enclosing dword window (hipMalloc rounds allocations to
// pages, so the <=4-byte edge slack stays inside the buffer). The residue
// is warp-uniform because the per-lane offset is a multiple of 8.

// Fetch bytes w8[0..7] from a possibly dword-misaligned global pointer.
__device__ __forceinline__ uint2 LoadBytes8(const uint8_t* w8) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(w8);
  const uint32_t* const wd =
      reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
  const uint32_t sh = static_cast<uint32_t>(a & 3U) * 8U;
  const uint32_t d0 = wd[0];
  const uint32_t d1 = wd[1];
  if (sh == 0U) {
    return make_uint2(d0, d1);
  }
  const uint32_t d2 = wd[2];
  return make_uint2(__funnelshift_r(d0, d1, sh), __funnelshift_r(d1, d2, sh));
}

__device__ __forceinline__ float GemvWarpDotQ4_K(const uint8_t* w,
                                                 const float* x, int block,
                                                 int lane) {
  const NblQ4_K* b = reinterpret_cast<const NblQ4_K*>(w) + block;
  const float d = Fp16ToF32(b->d);
  const float dmin = Fp16ToF32(b->dmin);
  const int g = lane >> 2;
  float s = 0.0F;
  float m = 0.0F;
  QScaleMin(g, b->scales, d, dmin, s, m);
  const uint2 q8 = LoadBytes8(b->qs + (g >> 1) * 32 + (lane & 3) * 8);
  const int sh = (g & 1) * 4;
  const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
  const float4 xa = *reinterpret_cast<const float4*>(xv);
  const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
  const uint32_t qw[2] = {q8.x, q8.y};
  float acc = 0.0F;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const uint32_t byte = qw[p >> 2] >> (8 * (p & 3));
    const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
    acc += (s * static_cast<float>((byte >> sh) & 0x0FU) - m) * xv8;
  }
  return acc;
}

__device__ __forceinline__ float GemvWarpDotQ5_K(const uint8_t* w,
                                                 const float* x, int block,
                                                 int lane) {
  const NblQ5_K* b = reinterpret_cast<const NblQ5_K*>(w) + block;
  const float d = Fp16ToF32(b->d);
  const float dmin = Fp16ToF32(b->dmin);
  const int v0 = lane * 8;
  const int gg = v0 >> 6;
  const int half = (v0 >> 5) & 1;
  const int l0 = v0 & 31;
  float s = 0.0F;
  float m = 0.0F;
  QScaleMin(2 * gg + half, b->scales, d, dmin, s, m);
  const uint2 q8 = LoadBytes8(b->qs + gg * 32 + l0);
  const uint2 h8 = LoadBytes8(b->qh + l0);
  const int sh = 2 * gg + half;
  const uint32_t qw[2] = {q8.x, q8.y};
  const uint32_t hw[2] = {h8.x, h8.y};
  const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
  const float4 xa = *reinterpret_cast<const float4*>(xv);
  const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
  float acc = 0.0F;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const int b8 = 8 * (p & 3);
    const uint32_t q = (qw[p >> 2] >> b8) & 0xFFU;
    const uint32_t hi = (hw[p >> 2] >> (b8 + sh)) & 1U;
    const int v = static_cast<int>(half ? (q >> 4) : (q & 0x0FU)) +
                  static_cast<int>(hi << 4U);
    const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
    acc += (s * static_cast<float>(v) - m) * xv8;
  }
  return acc;
}

__device__ __forceinline__ float GemvWarpDotQ6_K(const uint8_t* w,
                                                 const float* x, int block,
                                                 int lane) {
  const NblQ6_K* b = reinterpret_cast<const NblQ6_K*>(w) + block;
  // Values v of a half are laid out as v = c*128 + k*32 + l (c = half,
  // sub-quant k, l 0..31); lane owns 8 consecutive l of one (c, k), so the
  // 6-bit term bits (2k..2k+1 of the qh byte), the ql nibble shift (k >= 2)
  // and the scale index c*8 + 2k + hi16 are lane constants.
  const int c = lane >> 4;
  const int k = (lane >> 2) & 3;
  const int hi16 = (lane >> 1) & 1;
  const int j0 = (lane & 3) * 8;
  const uint2 h8 = LoadBytes8(b->qh + c * 32 + j0);
  const uint2 q8 = LoadBytes8(b->ql + c * 64 + (k & 1) * 32 + j0);
  const float ds =
      Fp16ToF32(b->d) * static_cast<float>(b->scales[c * 8 + 2 * k + hi16]);
  const int sh_h = 2 * k;
  const int sh_q = (k >> 1) * 4;
  const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
  const float4 xa = *reinterpret_cast<const float4*>(xv);
  const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
  const uint32_t hw[2] = {h8.x, h8.y};
  const uint32_t qw[2] = {q8.x, q8.y};
  float acc = 0.0F;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const uint32_t h56 = hw[p >> 2] >> (8 * (p & 3));
    const uint32_t lo4 = qw[p >> 2] >> (8 * (p & 3));
    const int t = static_cast<int>(((h56 >> sh_h) & 0x3U) << 4U) +
                  static_cast<int>((lo4 >> sh_q) & 0x0FU) - 32;
    const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
    acc += ds * static_cast<float>(t) * xv8;
  }
  return acc;
}

__device__ __forceinline__ float GemvWarpDotIq4Xs(const uint8_t* w,
                                                  const float* x, int block,
                                                  int lane) {
  const NblIq4Xs* b = reinterpret_cast<const NblIq4Xs*>(w) + block;
  const float d = Fp16ToF32(b->d);
  // Values v = ib*32 + u: qs[ib*16 + (u % 16)] holds the nibble, low for
  // u < 16 and high above. Lane windows of 8 start at u = (lane&3)*8, so
  // the byte slot is the u%16 residue (lane&1) and the nibble half is
  // whether u reached 16 ((lane>>1)&1).
  const int ib = lane >> 2;
  const int ls =
      static_cast<int>((b->scales_l[ib / 2] >> (4U * (ib % 2))) & 0x0FU) |
      static_cast<int>(((b->scales_h >> (2U * ib)) & 0x03U) << 4U);
  const float s = d * static_cast<float>(ls - 32);
  const uint2 q8 = LoadBytes8(b->qs + ib * 16 + (lane & 1) * 8);
  const int sh = ((lane >> 1) & 1) * 4U;
  const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
  const float4 xa = *reinterpret_cast<const float4*>(xv);
  const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
  const uint32_t qw[2] = {q8.x, q8.y};
  float acc = 0.0F;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const uint32_t byte = qw[p >> 2] >> (8 * (p & 3));
    const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
    acc += s * (kIq4Nl[(byte >> sh) & 0x0FU] * xv8);
  }
  return acc;
}

template<int TYPE>
__device__ __forceinline__ float GemvWarpDot(const uint8_t* __restrict__ w,
                                             const float* __restrict__ x,
                                             int block, int lane) {
  if (TYPE == kNblGemvQ4_K) {
    return GemvWarpDotQ4_K(w, x, block, lane);
  }
  if (TYPE == kNblGemvQ5_K) {
    return GemvWarpDotQ5_K(w, x, block, lane);
  }
  if (TYPE == kNblGemvQ6_K) {
    return GemvWarpDotQ6_K(w, x, block, lane);
  }
  return GemvWarpDotIq4Xs(w, x, block, lane);
}

// Signed int8x4 dot via the native unsigned byte dot; the compiler inserts
// the sign fix-ups for the signed operands (on gfx1151 this lowers to
// v_dot4_i32_iu8, matching the vendored mmq iq2 path).
__device__ __forceinline__ int Dp4a(int a, int b, int c) {
#if defined(__GFX11__)
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
  const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
  const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    c += static_cast<int>(pa[i]) * static_cast<int>(pb[i]);
  }
  return c;
#endif
}

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
template<>
__device__ __forceinline__ int GemvChunk<5>() {
  return kKqElems;
}
template<>
__device__ __forceinline__ int GemvChunk<6>() {
  return kKqElems;
}
template<>
__device__ __forceinline__ int GemvChunk<7>() {
  return kKqElems;
}
template<>
__device__ __forceinline__ int GemvChunk<8>() {
  return kKqElems;
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
  } else if (TYPE == kNblGemvQ4_K) {
    acc = GemvChunkDotQ4_K(w, x, chunk);
  } else if (TYPE == kNblGemvQ5_K) {
    acc = GemvChunkDotQ5_K(w, x, chunk);
  } else if (TYPE == kNblGemvQ6_K) {
    acc = GemvChunkDotQ6_K(w, x, chunk);
  } else if (TYPE == kNblGemvIq4Xs) {
    acc = GemvChunkDotIq4Xs(w, x, chunk);
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

// IQ2_XXS x Q8_1 integer chunk dot (llama b11721 vec_dot_iq2_xxs_q8_1 on the
// production word layout). The signed grid word is built with the XOR blend
// (cIq2Xor/cIq2SignMasks) and contracted against the quantized activation
// with dot4 instructions; scale factors apply once per chunk. Probe:
// tools/bench/nem_decode_iq2_dp4a_bench.hip (1.44x-1.80x on the MoE decode
// shapes, per-GEMV rel <= 0.0063 vs the float path).
__device__ __forceinline__ float GemvChunkDotIq2Q8_1(
    const uint8_t* __restrict__ w, const NblQ8_1* __restrict__ xq, int chunk) {
  const uint8_t* blk = w + static_cast<int64_t>(chunk / 8) * 66;
  uint16_t dh = 0;
  uint32_t aux32[2] = {0, 0};
  __builtin_memcpy(&dh, blk, sizeof(dh));
  __builtin_memcpy(aux32, blk + 2 + (chunk % 8) * 8, sizeof(aux32));
  const float d = __half2float(*reinterpret_cast<__half*>(&dh));
  const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
  const NblQ8_1* xb = xq + chunk;
  const float da = __low2float(xb->ds);
  int sumi = 0;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const uint64_t grid = cIq2Grid[aux8[l]];
    const uint64_t xw = cIq2Xor[aux8[l]];
    const uint2 msk = cIq2SignMasks[(aux32[1] >> (7 * l)) & 127];
    const int lo = static_cast<int>(static_cast<uint32_t>(grid) ^
                                    (msk.x & static_cast<uint32_t>(xw)));
    const int hi = static_cast<int>(static_cast<uint32_t>(grid >> 32) ^
                                    (msk.y & static_cast<uint32_t>(xw >> 32)));
    const int8_t* q = xb->qs + l * 8;
    sumi = Dp4a(lo, *reinterpret_cast<const int*>(q), sumi);
    sumi = Dp4a(hi, *reinterpret_cast<const int*>(q + 4), sumi);
  }
  const float db = d * (0.5F + static_cast<float>(aux32[1] >> 28)) * 0.25F;
  return da * db * static_cast<float>(sumi);
}

template<int TYPE>
__global__ void GemvF32Kernel(const float* __restrict__ x,
                              const uint8_t* __restrict__ w,
                              float* __restrict__ out, int n_out, int k,
                              int expert_stride,
                              const int32_t* __restrict__ ids, int slots,
                              int64_t x_stride, int tokens) {
  const int row = static_cast<int>(blockIdx.x % static_cast<unsigned>(n_out));
  const int token = static_cast<int>(blockIdx.y);
  int64_t w_row = row;
  int64_t out_row = static_cast<int64_t>(token) * n_out + row;
  if (slots > 1) {
    const int slot = static_cast<int>(blockIdx.x / n_out);
    const int64_t s = static_cast<int64_t>(token) * slots + slot;
    w_row = static_cast<int64_t>(__ldg(ids + s)) * n_out + row;
    out_row = s * n_out + row;
    x += static_cast<int64_t>(slot) * x_stride;
  }
  // Shared expert inputs (x_stride == 0) are one vector per token of length k;
  // gathered per-slot inputs are a [tokens][slots][x_stride] block.
  x += static_cast<int64_t>(token) *
       ((slots > 1 && x_stride > 0) ? static_cast<int64_t>(slots) * x_stride
                                    : k);
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

// Row-packed variant of GemvF32Kernel for small-k decode rows. The
// one-row-per-block kernel activates only n_chunk of kBlock threads on decode
// shapes (k=1024 IQ2: 32 of 256) and launches a block per output row, so the
// MoE up-projection streams ~264 B per block through launch and reduce
// overhead. Here one block covers several rows: a row owns
// R = ceil(n_chunk / warpSize) consecutive warps and the block holds
// rows = blockDim / (warpSize * R) rows; warp i lane l of a row owns chunk
// warpSize * i + l -- the same chunk production thread warpSize * i + l
// owned -- and each lane holds at most one chunk while n_chunk <= kBlock. The
// per-warp shfl_down tree and the seeded ascending fold sum (zero-initialized
// entries reproduce the +0.0F padding warps) mirror BlockReduceSum bitwise,
// so every output is identical to GemvF32Kernel. Probe:
// tools/bench/nem_decode_iq2_bench.hip (1.2x-4.9x, bitwise across shapes).
template<int TYPE>
__global__ void GemvF32PackedKernel(const float* __restrict__ x,
                                    const uint8_t* __restrict__ w,
                                    float* __restrict__ out, int n_out, int k,
                                    int expert_stride,
                                    const int32_t* __restrict__ ids, int slots,
                                    int64_t x_stride, int tokens,
                                    int warps_per_row, int blocks_per_slot) {
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const int row_local = warp / warps_per_row;
  const int warp_in_row = warp % warps_per_row;
  __shared__ float fold[64];  // [row][fold entry], 8 entries per row
  if (static_cast<int>(threadIdx.x) < 64) {
    fold[threadIdx.x] = 0.0F;
  }
  __syncthreads();
  const int gid = static_cast<int>(blockIdx.x);
  const int slot = gid / blocks_per_slot;
  const int row = (gid % blocks_per_slot) * (static_cast<int>(blockDim.x) /
                                             warpSize / warps_per_row) +
                  row_local;
  const bool active = (row < n_out);
  const int token = static_cast<int>(blockIdx.y);
  float acc = 0.0F;
  int64_t out_row = 0;
  if (active) {
    // Slot and token addressing identical to GemvF32Kernel.
    const float* xr = x;
    int64_t w_row = row;
    out_row = static_cast<int64_t>(token) * n_out + row;
    if (slots > 1) {
      const int64_t s = static_cast<int64_t>(token) * slots + slot;
      w_row = static_cast<int64_t>(__ldg(ids + s)) * n_out + row;
      out_row = s * n_out + row;
      xr += static_cast<int64_t>(slot) * x_stride;
    }
    xr += static_cast<int64_t>(token) *
          ((slots > 1 && x_stride > 0) ? static_cast<int64_t>(slots) * x_stride
                                       : k);
    const uint8_t* wr = w + w_row * static_cast<int64_t>(expert_stride);
    const int c = lane + warpSize * warp_in_row;
    if (c < k / GemvChunk<TYPE>()) {
      acc = GemvChunkDot<TYPE>(wr, xr, c, k);
    }
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(~0ULL, acc, off);
  }
  if (lane == 0) {
    fold[row_local * 8 + warp_in_row] = acc;
  }
  __syncthreads();
  if (active && lane == 0 && warp_in_row == 0) {
    float sum = 0.0F;
    for (int j = 0; j < kBlock / warpSize; ++j) {
      sum += fold[row_local * 8 + j];
    }
    out[out_row] = sum;
  }
}

// Row-packed GEMV for K-quant / IQ4_XS weights with warp-owned
// super-blocks. Same block geometry, slot addressing and fold tree as
// GemvF32PackedKernel (whose thread-owned chunk reads every weight byte and
// activation scalar-wise); warp i of a row walks super-blocks
// i, i + warps_per_row, ... and a lane covers eight values per block. The
// x pointer must keep 16-byte alignment (enforced by the launcher).
template<int TYPE>
__global__ void GemvKqPackedKernel(const float* __restrict__ x,
                                   const uint8_t* __restrict__ w,
                                   float* __restrict__ out, int n_out, int k,
                                   int expert_stride,
                                   const int32_t* __restrict__ ids, int slots,
                                   int64_t x_stride, int tokens,
                                   int warps_per_row, int blocks_per_slot) {
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const int row_local = warp / warps_per_row;
  const int warp_in_row = warp % warps_per_row;
  __shared__ float fold[64];  // [row][fold entry], 8 entries per row
  if (static_cast<int>(threadIdx.x) < 64) {
    fold[threadIdx.x] = 0.0F;
  }
  __syncthreads();
  const int gid = static_cast<int>(blockIdx.x);
  const int slot = gid / blocks_per_slot;
  const int row = (gid % blocks_per_slot) * (static_cast<int>(blockDim.x) /
                                             warpSize / warps_per_row) +
                  row_local;
  const bool active = (row < n_out);
  const int token = static_cast<int>(blockIdx.y);
  float acc = 0.0F;
  int64_t out_row = 0;
  if (active) {
    // Slot and token addressing identical to GemvF32Kernel.
    const float* xr = x;
    int64_t w_row = row;
    out_row = static_cast<int64_t>(token) * n_out + row;
    if (slots > 1) {
      const int64_t s = static_cast<int64_t>(token) * slots + slot;
      w_row = static_cast<int64_t>(__ldg(ids + s)) * n_out + row;
      out_row = s * n_out + row;
      xr += static_cast<int64_t>(slot) * x_stride;
    }
    xr += static_cast<int64_t>(token) *
          ((slots > 1 && x_stride > 0) ? static_cast<int64_t>(slots) * x_stride
                                       : static_cast<int64_t>(k));
    const uint8_t* wr = w + w_row * static_cast<int64_t>(expert_stride);
    const int n_super = k / kKqElems;
    for (int s = warp_in_row; s < n_super; s += warps_per_row) {
      acc += GemvWarpDot<TYPE>(wr, xr, s, lane);
    }
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(~0ULL, acc, off);
  }
  if (lane == 0) {
    fold[row_local * 8 + warp_in_row] = acc;
  }
  __syncthreads();
  if (active && lane == 0 && warp_in_row == 0) {
    float sum = 0.0F;
    for (int j = 0; j < kBlock / warpSize; ++j) {
      sum += fold[row_local * 8 + j];
    }
    out[out_row] = sum;
  }
}

// Row-packed GEMV for IQ2 weights against nbl_quantize_q8_1 activations.
// Geometry and fold identical to GemvF32PackedKernel; the activation rows
// are 36-byte Q8_1 blocks, so x_stride is passed in blocks (not elements)
// and the shared-token step is k / kIq2Sub blocks.
__global__ void GemvIq2Q8_1PackedKernel(
    const NblQ8_1* __restrict__ x, const uint8_t* __restrict__ w,
    float* __restrict__ out, int n_out, int k, int expert_stride,
    const int32_t* __restrict__ ids, int slots, int64_t x_stride, int tokens,
    int warps_per_row, int blocks_per_slot) {
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const int row_local = warp / warps_per_row;
  const int warp_in_row = warp % warps_per_row;
  __shared__ float fold[64];  // [row][fold entry], 8 entries per row
  if (static_cast<int>(threadIdx.x) < 64) {
    fold[threadIdx.x] = 0.0F;
  }
  __syncthreads();
  const int gid = static_cast<int>(blockIdx.x);
  const int slot = gid / blocks_per_slot;
  const int row = (gid % blocks_per_slot) * (static_cast<int>(blockDim.x) /
                                             warpSize / warps_per_row) +
                  row_local;
  const bool active = (row < n_out);
  const int token = static_cast<int>(blockIdx.y);
  float acc = 0.0F;
  int64_t out_row = 0;
  if (active) {
    const NblQ8_1* xr = x;
    int64_t w_row = row;
    out_row = static_cast<int64_t>(token) * n_out + row;
    if (slots > 1) {
      const int64_t s = static_cast<int64_t>(token) * slots + slot;
      w_row = static_cast<int64_t>(__ldg(ids + s)) * n_out + row;
      out_row = s * n_out + row;
      xr += static_cast<int64_t>(slot) * x_stride;
    }
    xr += static_cast<int64_t>(token) *
          ((slots > 1 && x_stride > 0) ? static_cast<int64_t>(slots) * x_stride
                                       : static_cast<int64_t>(k) / kIq2Sub);
    const uint8_t* wr = w + w_row * static_cast<int64_t>(expert_stride);
    const int c = lane + warpSize * warp_in_row;
    if (c < k / kIq2Sub) {
      acc = GemvChunkDotIq2Q8_1(wr, xr, c);
    }
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(~0ULL, acc, off);
  }
  if (lane == 0) {
    fold[row_local * 8 + warp_in_row] = acc;
  }
  __syncthreads();
  if (active && lane == 0 && warp_in_row == 0) {
    float sum = 0.0F;
    for (int j = 0; j < kBlock / warpSize; ++j) {
      sum += fold[row_local * 8 + j];
    }
    out[out_row] = sum;
  }
}

// One warp per 32-value sub-block: d = amax / 127, q = round(x / d) (llama
// quantize_row_q8_1_generic parity), ds packs (d, sum(q)).
__global__ void QuantQ8_1Kernel(const float* __restrict__ x,
                                NblQ8_1* __restrict__ q, int64_t sub_blocks) {
  const int64_t gtid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t sub = gtid / warpSize;
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  if (sub >= sub_blocks) {
    return;
  }
  const float* xb = x + sub * kIq2Sub;
  const float v = xb[lane];
  float a = fabsf(v);
#pragma unroll
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    a = fmaxf(a, __shfl_xor_sync(~0ULL, a, off));
  }
  const float d = a / 127.0F;
  const float id = a != 0.0F ? 127.0F / a : 0.0F;
  const int8_t qv = a == 0.0F ? 0 : static_cast<int8_t>(roundf(v * id));
  int sum = static_cast<int>(qv);
#pragma unroll
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    sum += __shfl_xor_sync(~0ULL, sum, off);
  }
  q[sub].qs[lane] = qv;
  if (lane == 0) {
    q[sub].ds = __floats2half2_rn(d, static_cast<float>(sum));
  }
}

__global__ void RouterKernel(const float* __restrict__ logits,
                             const float* __restrict__ bias, int n_experts,
                             int top_k, int norm, float scale,
                             int32_t* __restrict__ ids,
                             float* __restrict__ weights) {
  const int token = blockIdx.x;
  const float* lg = logits + static_cast<int64_t>(token) * n_experts;
  __shared__ float score[kRouterMaxExperts];
  __shared__ float sigv[kRouterMaxExperts];
  __shared__ float wval[kRouterWarpCount];
  __shared__ int widx[kRouterWarpCount];
  __shared__ int sel[16];
  __shared__ float num[16];
  __shared__ int gidx;
  __shared__ int stop_k;
  const int tid = static_cast<int>(threadIdx.x);
  for (int e = tid; e < n_experts; e += blockDim.x) {
    const float sig = 1.0F / (1.0F + expf(-lg[e]));
    sigv[e] = sig;
    score[e] = sig + bias[e];
  }
  bool alive = tid < n_experts && score[tid] != -FLT_MAX;
  if (tid == 0) {
    stop_k = -1;
  }
  __syncthreads();
  for (int k = 0; k < top_k; ++k) {
    // Each round re-reads the immutable shared score and the per-thread
    // alive bit (the fold itself clobbers registers) and reduces the
    // (score, index) pair: strict '>' with descending lanes mirrors the
    // oracle's ascending scan whose first maximal element keeps the lowest
    // index, and masked candidates (idx < 0) never win.
    float val = alive ? score[tid] : -FLT_MAX;
    int idx = alive ? tid : -1;
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      const float oval = __shfl_down_sync(~0ULL, val, off);
      const int oidx = __shfl_down_sync(~0ULL, idx, off);
      const bool keep = idx >= 0 && (oidx < 0 || val > oval);
      val = keep ? val : oval;
      idx = keep ? idx : oidx;
    }
    if ((tid & (warpSize - 1)) == 0) {
      wval[tid / warpSize] = val;
      widx[tid / warpSize] = idx;
    }
    __syncthreads();
    if (tid < warpSize) {
      float mval = tid < kRouterWarpCount ? wval[tid] : -FLT_MAX;
      int midx = tid < kRouterWarpCount ? widx[tid] : -1;
      for (int off = warpSize / 2; off > 0; off >>= 1) {
        const float oval = __shfl_down_sync(~0ULL, mval, off);
        const int oidx = __shfl_down_sync(~0ULL, midx, off);
        const bool keep = midx >= 0 && (oidx < 0 || mval > oval);
        mval = keep ? mval : oval;
        midx = keep ? midx : oidx;
      }
      if (tid == 0) {
        gidx = midx;
        if (midx >= 0) {
          sel[k] = midx;
          num[k] = sigv[midx];  // same expf expression as the oracle recompute
        } else {
          stop_k = k;
        }
      }
    }
    __syncthreads();
    if (gidx < 0) {
      break;
    }
    if (tid == gidx) {
      alive = false;  // per-index mask; equal-valued peers stay eligible
    }
  }
  if (tid == 0) {
    const int total = stop_k >= 0 ? stop_k : top_k;
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
  // The tiled tile wins from 64 tokens up; below one tile the masked token
  // slots stream the same weights through the LDS pipeline for nothing. The
  // thread-per-output kernel folds each row in the same ascending-k order, so
  // switching here and below a full tile is bitwise identical (probe:
  // tools/bench/nem_decode_router_bench.hip, 280us -> 44us per decode router).
  if (k % kGemmKStep == 0 && n_tokens >= kGemmTileTokens) {
    const dim3 grid((n_tokens + kGemmTileTokens - 1) / kGemmTileTokens,
                    (n_out + kGemmTileExperts - 1) / kGemmTileExperts);
    GemmF32TiledKernel<<<grid, kGemmThreads, 0, stream>>>(x, w, out, n_out, k,
                                                          n_tokens);
  } else if (k % kRingStageK == 0) {
    const dim3 grid((n_out + kRingRows - 1) / kRingRows, n_tokens);
    GemmF32RingKernel<<<grid, kBlock, 0, stream>>>(x, w, out, n_out, k);
  } else {
    const dim3 grid((n_out + kBlock - 1) / kBlock, n_tokens);
    GemmF32Kernel<<<grid, kBlock, 0, stream>>>(x, w, out, n_out, k);
  }
  return CheckLaunch(hipGetLastError(), "nbl_gemm_f32");
}

extern "C" int nbl_gemv_nvfp4_f32(const float* x, const void* w, float* out,
                                  int n_out, int k, int tokens,
                                  hipStream_t stream) {
  if (!x || !w || !out || n_out <= 0 || k <= 0 || tokens <= 0 ||
      k % kNvfp4K != 0) {
    return -1;
  }
  const dim3 grid(static_cast<unsigned>(n_out), static_cast<unsigned>(tokens));
  GemvNvfp4F32Kernel<<<grid, kBlock, 0, stream>>>(
      x, static_cast<const uint8_t*>(w), out, n_out, k, tokens);
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
    case kNblGemvIq2XxsQ8_1:
      return k / kIq2Elems * 66;
    case kNblGemvQ4_K:
      return k / kKqElems * static_cast<int>(sizeof(NblQ4_K));
    case kNblGemvQ5_K:
      return k / kKqElems * static_cast<int>(sizeof(NblQ5_K));
    case kNblGemvQ6_K:
      return k / kKqElems * static_cast<int>(sizeof(NblQ6_K));
    case kNblGemvIq4Xs:
      return k / kKqElems * static_cast<int>(sizeof(NblIq4Xs));
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
    case kNblGemvIq2XxsQ8_1:
      return k % kIq2Elems == 0;
    case kNblGemvNvfp4:
      return k % kNvfp4K == 0;
    case kNblGemvQ4_K:
    case kNblGemvQ5_K:
    case kNblGemvQ6_K:
    case kNblGemvIq4Xs:
      return k % kKqElems == 0;
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
    uint64_t xors[256];
    uint2 masks[128];
    for (int i = 0; i < 256; ++i) {
      uint64_t neg = 0;
      for (int j = 0; j < 8; ++j) {
        const uint8_t v = static_cast<uint8_t>(cpu::kIq2XxsGrid[i] >> (8 * j));
        neg |= static_cast<uint64_t>(static_cast<uint8_t>(0u - v)) << (8 * j);
      }
      xors[i] = cpu::kIq2XxsGrid[i] ^ neg;
    }
    for (int i = 0; i < 128; ++i) {
      const uint32_t s = cpu::kSignsIq2Xs[i];
      uint32_t lo = 0;
      uint32_t hi = 0;
      for (int j = 0; j < 4; ++j) {
        lo |= (((s >> j) & 1u) != 0u ? 0xFFu : 0u) << (8 * j);
        hi |= (((s >> (j + 4)) & 1u) != 0u ? 0xFFu : 0u) << (8 * j);
      }
      masks[i] = make_uint2(lo, hi);
    }
    (void)hipMemcpyToSymbol(cIq2Xor, xors, sizeof(xors));
    (void)hipMemcpyToSymbol(cIq2SignMasks, masks, sizeof(masks));
    loaded = true;
  }
}

int GemvChunkHost(int type) {
  switch (type) {
    case kNblGemvF32:
      return 16;
    case kNblGemvQ8_0:
      return kQ8_0Elems;
    case kNblGemvIq2Xxs:
      return kIq2Sub;
    case kNblGemvIq2XxsQ8_1:
      return kIq2Sub;
    case kNblGemvQ4_K:
    case kNblGemvQ5_K:
    case kNblGemvQ6_K:
    case kNblGemvIq4Xs:
      return kKqElems;
    default:
      return kNvfp4Sub;
  }
}

int DeviceWarpSize() {
  static int warp = [] {
    int w = 32;
    (void)hipDeviceGetAttribute(&w, hipDeviceAttributeWarpSize, 0);
    return w > 0 ? w : 32;
  }();
  return warp;
}

template<int TYPE>
void LaunchGemvF32Type(bool packed, const dim3& grid, const dim3& block,
                       hipStream_t stream, const float* x, const uint8_t* wb,
                       float* out, int n_out, int k, int stride,
                       const int32_t* ids, int slots, int64_t x_stride,
                       int tokens, int warps_per_row, int blocks_per_slot) {
  if (packed) {
    // kNblGemvQ4_K..kNblGemvIq4Xs take the warp-dot kernel when x keeps the
    // per-lane float4 windows 16-byte aligned; every tier type still
    // compiles both branches, and non-K-quant tiers pass kq_vec == false.
    const bool kq_tier = (TYPE == kNblGemvQ4_K) || (TYPE == kNblGemvQ5_K) ||
                         (TYPE == kNblGemvQ6_K) || (TYPE == kNblGemvIq4Xs);
    const bool vec = kq_tier && (reinterpret_cast<uintptr_t>(x) & 15U) == 0 &&
                     x_stride % 4 == 0;
    if (vec) {
      GemvKqPackedKernel<TYPE><<<grid, block, 0, stream>>>(
          x, wb, out, n_out, k, stride, ids, slots, x_stride, tokens,
          warps_per_row, blocks_per_slot);
    } else {
      GemvF32PackedKernel<TYPE><<<grid, block, 0, stream>>>(
          x, wb, out, n_out, k, stride, ids, slots, x_stride, tokens,
          warps_per_row, blocks_per_slot);
    }
  } else {
    GemvF32Kernel<TYPE><<<grid, block, 0, stream>>>(
        x, wb, out, n_out, k, stride, ids, slots, x_stride, tokens);
  }
}

template<bool kMoe>
int LaunchGemv(int type, const void* w, const float* x, const int32_t* ids,
               float* out, int n_out, int k, int slots, int64_t x_stride,
               int tokens, hipStream_t stream) {
  const int stride = GemvRowBytes(type, k);
  const uint8_t* wb = static_cast<const uint8_t*>(w);
  // Small-k rows run the row-packed kernel: one row per R warps instead of
  // one row per 256-thread block. Bitwise identical fold (see the kernel).
  const int warp = DeviceWarpSize();
  const int n_chunk = k / GemvChunkHost(type);
  const bool packed = n_chunk <= kBlock;
  const int warps_per_row = packed ? (n_chunk + warp - 1) / warp : 1;
  const int rows_per_block =
      packed ? std::max(1, kBlock / (warp * warps_per_row)) : 1;
  const int blocks_per_slot =
      packed ? (n_out + rows_per_block - 1) / rows_per_block : n_out;
  const unsigned grid_x = kMoe ? static_cast<unsigned>(slots) *
                                     static_cast<unsigned>(blocks_per_slot)
                               : static_cast<unsigned>(blocks_per_slot);
  const dim3 grid(grid_x, static_cast<unsigned>(tokens));
  const dim3 block(static_cast<unsigned>(
                       packed ? warp * warps_per_row * rows_per_block : kBlock),
                   1);
  switch (type) {
    case kNblGemvF32:
      if (packed) {
        GemvF32PackedKernel<kNblGemvF32><<<grid, block, 0, stream>>>(
            x, wb, out, n_out, k, stride, ids, kMoe ? slots : 0, x_stride,
            tokens, warps_per_row, blocks_per_slot);
      } else {
        GemvF32Kernel<kNblGemvF32>
            <<<grid, block, 0, stream>>>(x, wb, out, n_out, k, stride, ids,
                                         kMoe ? slots : 0, x_stride, tokens);
      }
      break;
    case kNblGemvQ8_0:
      if (packed) {
        GemvF32PackedKernel<kNblGemvQ8_0><<<grid, block, 0, stream>>>(
            x, wb, out, n_out, k, stride, ids, kMoe ? slots : 0, x_stride,
            tokens, warps_per_row, blocks_per_slot);
      } else {
        GemvF32Kernel<kNblGemvQ8_0>
            <<<grid, block, 0, stream>>>(x, wb, out, n_out, k, stride, ids,
                                         kMoe ? slots : 0, x_stride, tokens);
      }
      break;
    case kNblGemvIq2Xxs:
      EnsureIq2Tables();
      if (packed) {
        GemvF32PackedKernel<kNblGemvIq2Xxs><<<grid, block, 0, stream>>>(
            x, wb, out, n_out, k, stride, ids, kMoe ? slots : 0, x_stride,
            tokens, warps_per_row, blocks_per_slot);
      } else {
        GemvF32Kernel<kNblGemvIq2Xxs>
            <<<grid, block, 0, stream>>>(x, wb, out, n_out, k, stride, ids,
                                         kMoe ? slots : 0, x_stride, tokens);
      }
      break;
    case kNblGemvIq2XxsQ8_1: {
      EnsureIq2Tables();
      if (!packed) {
        return -1;  // decode-only path; large-k rows must use kNblGemvIq2Xxs
      }
      // x holds nbl_quantize_q8_1 blocks; its row stride is in elements on
      // this API boundary and in blocks for the kernel.
      if (x_stride % kIq2Sub != 0) {
        return -1;
      }
      GemvIq2Q8_1PackedKernel<<<grid, block, 0, stream>>>(
          reinterpret_cast<const NblQ8_1*>(x), wb, out, n_out, k, stride, ids,
          kMoe ? slots : 0, x_stride / kIq2Sub, tokens, warps_per_row,
          blocks_per_slot);
      break;
    }
    case kNblGemvQ4_K:
      LaunchGemvF32Type<kNblGemvQ4_K>(
          packed, grid, block, stream, x, wb, out, n_out, k, stride, ids,
          kMoe ? slots : 0, x_stride, tokens, warps_per_row, blocks_per_slot);
      break;
    case kNblGemvQ5_K:
      LaunchGemvF32Type<kNblGemvQ5_K>(
          packed, grid, block, stream, x, wb, out, n_out, k, stride, ids,
          kMoe ? slots : 0, x_stride, tokens, warps_per_row, blocks_per_slot);
      break;
    case kNblGemvQ6_K:
      LaunchGemvF32Type<kNblGemvQ6_K>(
          packed, grid, block, stream, x, wb, out, n_out, k, stride, ids,
          kMoe ? slots : 0, x_stride, tokens, warps_per_row, blocks_per_slot);
      break;
    case kNblGemvIq4Xs:
      LaunchGemvF32Type<kNblGemvIq4Xs>(
          packed, grid, block, stream, x, wb, out, n_out, k, stride, ids,
          kMoe ? slots : 0, x_stride, tokens, warps_per_row, blocks_per_slot);
      break;
    default:
      if (packed) {
        GemvF32PackedKernel<kNblGemvNvfp4><<<grid, block, 0, stream>>>(
            x, wb, out, n_out, k, stride, ids, kMoe ? slots : 0, x_stride,
            tokens, warps_per_row, blocks_per_slot);
      } else {
        GemvF32Kernel<kNblGemvNvfp4>
            <<<grid, block, 0, stream>>>(x, wb, out, n_out, k, stride, ids,
                                         kMoe ? slots : 0, x_stride, tokens);
      }
      break;
  }
  return CheckLaunch(hipGetLastError(),
                     kMoe ? "nbl_gemv_moe_f32" : "nbl_gemv_f32");
}

extern "C" int nbl_gemv_f32(int type, const void* w, const float* x, float* out,
                            int n_out, int k, int tokens, hipStream_t stream) {
  if (!x || !w || !out || n_out <= 0 || k <= 0 || tokens <= 0 ||
      !GemvTypeValid(type, k)) {
    return -1;
  }
  return LaunchGemv<false>(type, w, x, nullptr, out, n_out, k, 0, 0, tokens,
                           stream);
}

extern "C" int nbl_gemv_moe_f32(int type, const void* w, const float* x,
                                const int32_t* ids, float* out, int n_out,
                                int k, int slots, int64_t x_stride, int tokens,
                                hipStream_t stream) {
  if (!x || !w || !ids || !out || n_out <= 0 || k <= 0 || slots <= 0 ||
      x_stride < 0 || tokens <= 0 || !GemvTypeValid(type, k)) {
    return -1;
  }
  return LaunchGemv<true>(type, w, x, ids, out, n_out, k, slots, x_stride,
                          tokens, stream);
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

extern "C" int nbl_quantize_q8_1(const float* x, void* xq, int64_t elems,
                                 hipStream_t stream) {
  const int warp = DeviceWarpSize();
  if (!x || !xq || elems <= 0 || elems % kIq2Sub != 0 || warp != kIq2Sub) {
    return -1;
  }
  const int64_t subs = elems / kIq2Sub;
  QuantQ8_1Kernel<<<BlockCount(subs * warp), kBlock, 0, stream>>>(
      x, static_cast<NblQ8_1*>(xq), subs);
  return CheckLaunch(hipGetLastError(), "nbl_quantize_q8_1");
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

extern "C" int nbl_dequant_rows_q8_0(const void* w, const int32_t* ids,
                                     int tokens, int cols, float* out,
                                     hipStream_t stream) {
  if (!w || !ids || !out || tokens <= 0 || cols <= 0 || cols % 32 != 0) {
    return -1;
  }
  const dim3 grid(BlockCount(cols), static_cast<unsigned>(tokens));
  DequantQ8_0RowsKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const uint8_t*>(w), ids, out, cols);
  return CheckLaunch(hipGetLastError(), "nbl_dequant_rows_q8_0");
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
