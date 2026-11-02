#include "src/models/flint/kernels/rocm/linear.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "flint_mmq.h"

namespace gufo::models::flint::rocm {
namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr std::uint32_t kQ8Block = 32;
constexpr std::size_t kQ8BlockBytes = 34;
constexpr int kTileTokens = 32;
// Multi-row Bf16 prefill tile (GemvBatchBf16TileRKernel): 8 rows x 8 tokens
// per warp, fastest bitwise-equal shape on gfx1151.
constexpr int kTileRows = 8;
constexpr int kTileTokR = 8;

[[nodiscard]] __device__ __forceinline__ float Bf16ToF16(std::uint16_t v) {
  union {
    std::uint32_t u;
    float f;
  } bits;
  bits.u = static_cast<std::uint32_t>(v) << 16;
  return bits.f;
}

/// The Q8_0 block header is a binary16 scale, then 32 signed bytes.
[[nodiscard]] __device__ __forceinline__ float Q8Scale(const std::uint8_t* bp) {
  return __half2float(*reinterpret_cast<const __half*>(bp));
}

__device__ __forceinline__ float WarpSum(float v) {
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    v += __shfl_down_sync(__activemask(), v, off);
  }
  return v;
}

// One Q8 block dot product. Loads x as float4 (block offset b*32 and the
// token pitch cols*4 are 128B-aligned for every Q8 shape) while keeping the
// scalar j-ascending add order, so results are unchanged.
__device__ __forceinline__ float Q8DotBlock(const std::int8_t* __restrict__ qs,
                                            const float* __restrict__ xk) {
  float part = 0.0F;
#pragma unroll
  for (std::uint32_t q = 0; q < kQ8Block / 4U; ++q) {
    const float4 v = *reinterpret_cast<const float4*>(xk + q * 4U);
    part += static_cast<float>(qs[q * 4U + 0U]) * v.x;
    part += static_cast<float>(qs[q * 4U + 1U]) * v.y;
    part += static_cast<float>(qs[q * 4U + 2U]) * v.z;
    part += static_cast<float>(qs[q * 4U + 3U]) * v.w;
  }
  return part;
}

__global__ void GemvQ8Kernel(const std::uint8_t* __restrict__ w,
                             std::uint32_t rows, std::uint32_t cols,
                             const float* __restrict__ x,
                             float* __restrict__ out) {
  const std::uint32_t row =
      blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (row >= rows) {
    return;
  }
  const std::uint8_t* __restrict__ wr =
      w + static_cast<std::size_t>(row) * (cols / kQ8Block) * kQ8BlockBytes;
  const std::uint32_t nblocks = cols / kQ8Block;
  float acc = 0.0F;
  for (std::uint32_t b = lane; b < nblocks; b += kWarp) {
    const std::uint8_t* __restrict__ bp = wr + b * kQ8BlockBytes;
    const float d = Q8Scale(bp);
    const std::int8_t* __restrict__ qs =
        reinterpret_cast<const std::int8_t*>(bp + 2);
    const float* __restrict__ xk = x + static_cast<std::size_t>(b) * kQ8Block;
    acc += d * Q8DotBlock(qs, xk);
  }
  acc = WarpSum(acc);
  if (lane == 0) {
    out[row] = acc;
  }
}

__global__ void GemvF32Kernel(const float* __restrict__ w, std::uint32_t rows,
                              std::uint32_t cols, const float* __restrict__ x,
                              float* __restrict__ out) {
  const std::uint32_t row =
      blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (row >= rows) {
    return;
  }
  const float* __restrict__ wr = w + static_cast<std::size_t>(row) * cols;
  float acc = 0.0F;
  for (std::uint32_t k = lane; k < cols; k += kWarp) {
    acc = fmaf(wr[k], x[k], acc);
  }
  acc = WarpSum(acc);
  if (lane == 0) {
    out[row] = acc;
  }
}

__global__ void GemvBf16Kernel(const std::uint16_t* __restrict__ w,
                               std::uint32_t rows, std::uint32_t cols,
                               const float* __restrict__ x,
                               float* __restrict__ out) {
  const std::uint32_t row =
      blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (row >= rows) {
    return;
  }
  const std::uint16_t* __restrict__ wr =
      w + static_cast<std::size_t>(row) * cols;
  float acc = 0.0F;
  // Unrolling keeps the per-lane k order (lane, lane+kWarp, ...) and the
  // dependent fma chain bitwise-identical; it only issues the scalar loads
  // several iterations ahead, which is what the 32-block router GEMV needs
  // (2 B/lane loads left it at ~36 GB/s; docs/models/flint/EXPERIMENTS.md C16).
#pragma unroll 4
  for (std::uint32_t k = lane; k < cols; k += kWarp) {
    acc = fmaf(Bf16ToF16(wr[k]), x[k], acc);
  }
  acc = WarpSum(acc);
  if (lane == 0) {
    out[row] = acc;
  }
}

__global__ void GemvBatchQ8Kernel(const std::uint8_t* __restrict__ w,
                                  std::uint32_t rows, std::uint32_t cols,
                                  const float* __restrict__ x,
                                  float* __restrict__ out) {
  const std::uint32_t token = blockIdx.y;
  const std::uint32_t row =
      blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (row >= rows) {
    return;
  }
  x += static_cast<std::size_t>(token) * cols;
  out += static_cast<std::size_t>(token) * rows;
  const std::uint8_t* __restrict__ wr =
      w + static_cast<std::size_t>(row) * (cols / kQ8Block) * kQ8BlockBytes;
  const std::uint32_t nblocks = cols / kQ8Block;
  float acc = 0.0F;
  for (std::uint32_t b = lane; b < nblocks; b += kWarp) {
    const std::uint8_t* __restrict__ bp = wr + b * kQ8BlockBytes;
    const float d = Q8Scale(bp);
    const std::int8_t* __restrict__ qs =
        reinterpret_cast<const std::int8_t*>(bp + 2);
    const float* __restrict__ xk = x + static_cast<std::size_t>(b) * kQ8Block;
    acc += d * Q8DotBlock(qs, xk);
  }
  acc = WarpSum(acc);
  if (lane == 0) {
    out[row] = acc;
  }
}

// Token-tiled mirror of GemvBatchQ8Kernel for prefill: one warp owns 32
// consecutive tokens of one row and loads each 34-byte Q8 block plus the
// exact float(qs) conversions once, then feeds 32 independent accumulator
// chains. Per token the b-stride order, float4 x loads with j-ascending dot
// order, d*x products and WarpSum tree match GemvBatchQ8Kernel exactly, so
// results are bit-identical while the weight bytes and i2f work are amortized
// over 32 tokens (measured 1.9-2.05x on gfx1151). grid.y must be batch/32;
// callers route the ragged tail through GemvBatchQ8Kernel.
__global__ void GemvBatchQ8TileKernel(const std::uint8_t* __restrict__ w,
                                      std::uint32_t rows, std::uint32_t cols,
                                      const float* __restrict__ x,
                                      float* __restrict__ out) {
  const std::uint32_t row =
      blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (row >= rows) {
    return;
  }
  const std::uint32_t t0 = blockIdx.y * kTileTokens;
  const std::uint8_t* __restrict__ wr =
      w + static_cast<std::size_t>(row) * (cols / kQ8Block) * kQ8BlockBytes;
  const float* __restrict__ xt = x + static_cast<std::size_t>(t0) * cols;
  float* __restrict__ ot =
      out + static_cast<std::size_t>(row) + static_cast<std::size_t>(t0) * rows;
  const std::uint32_t nblocks = cols / kQ8Block;
  float acc[kTileTokens];
#pragma unroll
  for (int t = 0; t < kTileTokens; ++t) {
    acc[t] = 0.0F;
  }
  for (std::uint32_t b = lane; b < nblocks; b += kWarp) {
    const std::uint8_t* __restrict__ bp = wr + b * kQ8BlockBytes;
    const float d = Q8Scale(bp);
    const std::int8_t* __restrict__ qs =
        reinterpret_cast<const std::int8_t*>(bp + 2);
    float qf[kQ8Block];
#pragma unroll
    for (std::uint32_t q = 0; q < kQ8Block; ++q) {
      qf[q] = static_cast<float>(qs[q]);
    }
#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
      const float* __restrict__ xk = xt + static_cast<std::size_t>(t) * cols +
                                     static_cast<std::size_t>(b) * kQ8Block;
      float part = 0.0F;
#pragma unroll
      for (std::uint32_t q = 0; q < kQ8Block / 4U; ++q) {
        const float4 v = *reinterpret_cast<const float4*>(xk + q * 4U);
        part += qf[q * 4U + 0U] * v.x;
        part += qf[q * 4U + 1U] * v.y;
        part += qf[q * 4U + 2U] * v.z;
        part += qf[q * 4U + 3U] * v.w;
      }
      acc[t] += d * part;
    }
  }
#pragma unroll
  for (int t = 0; t < kTileTokens; ++t) {
    const float a = WarpSum(acc[t]);
    if (lane == 0) {
      ot[static_cast<std::size_t>(t) * rows] = a;
    }
  }
}

// Batched mirrors of the Bf16/F32 GEMVs: token indexes blockIdx.y so one
// launch covers the whole prefill chunk instead of one launch per token.
// Each row keeps the single-token kernel's fmaf chain and warp reduction,
// so results are bit-identical to the per-token loop.
__global__ void GemvBatchBf16Kernel(const std::uint16_t* __restrict__ w,
                                    std::uint32_t rows, std::uint32_t cols,
                                    const float* __restrict__ x,
                                    float* __restrict__ out) {
  const std::uint32_t token = blockIdx.y;
  const std::uint32_t row =
      blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (row >= rows) {
    return;
  }
  x += static_cast<std::size_t>(token) * cols;
  out += static_cast<std::size_t>(token) * rows;
  const std::uint16_t* __restrict__ wr =
      w + static_cast<std::size_t>(row) * cols;
  float acc = 0.0F;
  for (std::uint32_t k = lane; k < cols; k += kWarp) {
    acc = fmaf(Bf16ToF16(wr[k]), x[k], acc);
  }
  acc = WarpSum(acc);
  if (lane == 0) {
    out[row] = acc;
  }
}

// Token-tiled mirror of GemvBatchBf16Kernel for prefill: one warp owns 32
// consecutive tokens of one row, loads each bf16 weight once and feeds 32
// independent accumulator chains. Per token the k-stride order, fmaf chain
// and WarpSum tree match GemvBatchBf16Kernel exactly (tools/
// bench/flint_gemv_bf16_tile_bench.hip: bitwise equal, ~7x on the 256x2048
// router shape), so router logits are bit-identical to the per-token loop.
// grid.y must be batch/32; callers route the ragged tail through
// GemvBatchBf16Kernel.
__global__ void GemvBatchBf16TileKernel(const std::uint16_t* __restrict__ w,
                                        std::uint32_t rows, std::uint32_t cols,
                                        const float* __restrict__ x,
                                        float* __restrict__ out) {
  const std::uint32_t row =
      blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (row >= rows) {
    return;
  }
  const std::uint32_t t0 = blockIdx.y * kTileTokens;
  const std::uint16_t* __restrict__ wr =
      w + static_cast<std::size_t>(row) * cols;
  const float* __restrict__ xt = x + static_cast<std::size_t>(t0) * cols;
  float* __restrict__ ot =
      out + static_cast<std::size_t>(row) + static_cast<std::size_t>(t0) * rows;
  float acc[kTileTokens];
#pragma unroll
  for (int t = 0; t < kTileTokens; ++t) {
    acc[t] = 0.0F;
  }
  // Pair-pipelined k steps: loads for k and k+warp are issued together
  // before the fmaf chains consume them, which raises memory-level
  // parallelism across the strided activation tile (~10% on the
  // router shape; tools/bench/flint_gemv_bf16_tile_bench.hip TilePF2).
  // Chain order per token stays k-ascending, so results remain bitwise
  // identical to the per-token kernel.
  std::uint32_t k = lane;
  for (; k + kWarp < cols; k += 2 * kWarp) {
    float xv0[kTileTokens];
    float xv1[kTileTokens];
#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
      xv0[t] = xt[static_cast<std::size_t>(t) * cols + k];
      xv1[t] = xt[static_cast<std::size_t>(t) * cols + k + kWarp];
    }
    const float w0 = Bf16ToF16(wr[k]);
    const float w1 = Bf16ToF16(wr[k + kWarp]);
#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
      acc[t] = fmaf(w0, xv0[t], acc[t]);
      acc[t] = fmaf(w1, xv1[t], acc[t]);
    }
  }
  if (k < cols) {
    const float wf = Bf16ToF16(wr[k]);
#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
      acc[t] = fmaf(wf, xt[static_cast<std::size_t>(t) * cols + k], acc[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < kTileTokens; ++t) {
    const float a = WarpSum(acc[t]);
    if (lane == 0) {
      ot[static_cast<std::size_t>(t) * rows] = a;
    }
  }
}

// Multi-row activation-tile variant: one warp owns kRows consecutive rows x
// kTok consecutive tokens, so each loaded x k-step feeds kRows accumulator
// chains and the block reads the token tile once per 8 rows instead of once
// per 8 warps. Per (row, token) the pair-pipelined loop keeps the
// k-ascending fmaf chain and WarpSum tree of GemvBatchBf16Kernel, so
// results stay bitwise identical (tools/bench/
// flint_gemv_tile2_bench.hip: 0 mismatches on the 256x2048 router shape
// for B in {256,1024,2048,8192}; t8x8 is ~2.4x over the 1-row 32-token
// warp at B=2048). Callers need rows % (8 * kRows) == 0 and route the
// ragged token tail through GemvBatchBf16Kernel.
template<int kRows, int kTok>
__global__ void GemvBatchBf16TileRKernel(const std::uint16_t* __restrict__ w,
                                         std::uint32_t rows, std::uint32_t cols,
                                         const float* __restrict__ x,
                                         float* __restrict__ out) {
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const std::uint32_t r0 = blockIdx.x * ((kThreads / kWarp) * kRows) +
                           warp * static_cast<std::uint32_t>(kRows);
  const std::uint32_t t0 = blockIdx.y * kTok;
  const std::uint16_t* __restrict__ wr =
      w + static_cast<std::size_t>(r0) * cols;
  const float* __restrict__ xt = x + static_cast<std::size_t>(t0) * cols;
  float* __restrict__ ot =
      out + static_cast<std::size_t>(r0) + static_cast<std::size_t>(t0) * rows;
  float acc[kRows][kTok];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
#pragma unroll
    for (int t = 0; t < kTok; ++t) {
      acc[r][t] = 0.0F;
    }
  }
  std::uint32_t k = lane;
  for (; k + kWarp < cols; k += 2 * kWarp) {
    float xv0[kTok];
    float xv1[kTok];
#pragma unroll
    for (int t = 0; t < kTok; ++t) {
      xv0[t] = xt[static_cast<std::size_t>(t) * cols + k];
      xv1[t] = xt[static_cast<std::size_t>(t) * cols + k + kWarp];
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const float w0 = Bf16ToF16(wr[static_cast<std::size_t>(r) * cols + k]);
      const float w1 =
          Bf16ToF16(wr[static_cast<std::size_t>(r) * cols + k + kWarp]);
#pragma unroll
      for (int t = 0; t < kTok; ++t) {
        acc[r][t] = fmaf(w0, xv0[t], acc[r][t]);
        acc[r][t] = fmaf(w1, xv1[t], acc[r][t]);
      }
    }
  }
  if (k < cols) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const float wf = Bf16ToF16(wr[static_cast<std::size_t>(r) * cols + k]);
#pragma unroll
      for (int t = 0; t < kTok; ++t) {
        acc[r][t] =
            fmaf(wf, xt[static_cast<std::size_t>(t) * cols + k], acc[r][t]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
#pragma unroll
    for (int t = 0; t < kTok; ++t) {
      const float a = WarpSum(acc[r][t]);
      if (lane == 0) {
        ot[static_cast<std::size_t>(r) + static_cast<std::size_t>(t) * rows] =
            a;
      }
    }
  }
}

__global__ void GemvBatchF32Kernel(const float* __restrict__ w,
                                   std::uint32_t rows, std::uint32_t cols,
                                   const float* __restrict__ x,
                                   float* __restrict__ out) {
  const std::uint32_t token = blockIdx.y;
  const std::uint32_t row =
      blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  if (row >= rows) {
    return;
  }
  x += static_cast<std::size_t>(token) * cols;
  out += static_cast<std::size_t>(token) * rows;
  const float* __restrict__ wr = w + static_cast<std::size_t>(row) * cols;
  float acc = 0.0F;
  for (std::uint32_t k = lane; k < cols; k += kWarp) {
    acc = fmaf(wr[k], x[k], acc);
  }
  acc = WarpSum(acc);
  if (lane == 0) {
    out[row] = acc;
  }
}

__global__ void DequantRowKernel(const std::uint8_t* __restrict__ w,
                                 core::GgmlType type, std::uint32_t cols,
                                 std::size_t row_bytes,
                                 float* __restrict__ out) {
  (void)row_bytes;
  if (type == core::GgmlType::kQ8_0) {
    for (std::uint32_t b = threadIdx.x; b < cols / kQ8Block; b += kThreads) {
      const std::uint8_t* __restrict__ bp = w + b * kQ8BlockBytes;
      const float d = Q8Scale(bp);
      const std::int8_t* __restrict__ qs =
          reinterpret_cast<const std::int8_t*>(bp + 2);
      for (std::uint32_t j = 0; j < kQ8Block; ++j) {
        out[b * kQ8Block + j] = d * static_cast<float>(qs[j]);
      }
    }
    return;
  }
  if (type == core::GgmlType::kF32) {
    const float* __restrict__ wf = reinterpret_cast<const float*>(w);
    for (std::uint32_t k = threadIdx.x; k < cols; k += kThreads) {
      out[k] = wf[k];
    }
    return;
  }
  const std::uint16_t* __restrict__ wb =
      reinterpret_cast<const std::uint16_t*>(w);
  for (std::uint32_t k = threadIdx.x; k < cols; k += kThreads) {
    out[k] = Bf16ToF16(wb[k]);
  }
}

}  // namespace

void Gemv(const void* base, core::GgmlType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          hipStream_t stream) {
  (void)row_bytes;
  const dim3 grid((rows + kThreads / kWarp - 1) / (kThreads / kWarp));
  switch (type) {
    case core::GgmlType::kQ8_0:
      GemvQ8Kernel<<<grid, kThreads, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), rows, cols, x, out);
      break;
    case core::GgmlType::kF32:
      GemvF32Kernel<<<grid, kThreads, 0, stream>>>(
          static_cast<const float*>(base), rows, cols, x, out);
      break;
    default:
      GemvBf16Kernel<<<grid, kThreads, 0, stream>>>(
          static_cast<const std::uint16_t*>(base), rows, cols, x, out);
      break;
  }
}

void GemvBatch(const void* base, core::GgmlType type, std::uint32_t rows,
               std::uint32_t cols, std::size_t row_bytes, const float* x,
               float* out, std::uint32_t batch, hipStream_t stream) {
  (void)row_bytes;
  const dim3 grid((rows + kThreads / kWarp - 1) / (kThreads / kWarp), batch);
  switch (type) {
    case core::GgmlType::kQ8_0: {
      const std::uint32_t tiles =
          batch / static_cast<std::uint32_t>(kTileTokens);
      const auto* const w8 = static_cast<const std::uint8_t*>(base);
      if (tiles > 0) {
        GemvBatchQ8TileKernel<<<dim3(grid.x, tiles), kThreads, 0, stream>>>(
            w8, rows, cols, x, out);
        const std::uint32_t done =
            tiles * static_cast<std::uint32_t>(kTileTokens);
        if (done < batch) {
          GemvBatchQ8Kernel<<<dim3(grid.x, batch - done), kThreads, 0,
                              stream>>>(
              w8, rows, cols, x + static_cast<std::size_t>(done) * cols,
              out + static_cast<std::size_t>(done) * rows);
        }
      } else {
        GemvBatchQ8Kernel<<<grid, kThreads, 0, stream>>>(w8, rows, cols, x,
                                                         out);
      }
      break;
    }
    case core::GgmlType::kF32:
      GemvBatchF32Kernel<<<grid, kThreads, 0, stream>>>(
          static_cast<const float*>(base), rows, cols, x, out);
      break;
    default: {
      const auto* const wb = static_cast<const std::uint16_t*>(base);
      // 8-row x 8-token tiles first (router shape): bitwise-identical and
      // ~2.4x over the 1-row warp at B=2048 (flint_gemv_tile2_bench).
      if (rows % (8U * kTileRows) == 0U) {
        const std::uint32_t tiles =
            batch / static_cast<std::uint32_t>(kTileTokR);
        if (tiles > 0) {
          GemvBatchBf16TileRKernel<kTileRows, kTileTokR>
              <<<dim3(rows / (kThreads / kWarp * kTileRows), tiles), kThreads,
                 0, stream>>>(wb, rows, cols, x, out);
          const std::uint32_t done =
              tiles * static_cast<std::uint32_t>(kTileTokR);
          if (done < batch) {
            GemvBatchBf16Kernel<<<dim3(grid.x, batch - done), kThreads, 0,
                                  stream>>>(
                wb, rows, cols, x + static_cast<std::size_t>(done) * cols,
                out + static_cast<std::size_t>(done) * rows);
          }
          break;
        }
      }
      const std::uint32_t tiles =
          batch / static_cast<std::uint32_t>(kTileTokens);
      if (tiles > 0) {
        GemvBatchBf16TileKernel<<<dim3(grid.x, tiles), kThreads, 0, stream>>>(
            wb, rows, cols, x, out);
        const std::uint32_t done =
            tiles * static_cast<std::uint32_t>(kTileTokens);
        if (done < batch) {
          GemvBatchBf16Kernel<<<dim3(grid.x, batch - done), kThreads, 0,
                                stream>>>(
              wb, rows, cols, x + static_cast<std::size_t>(done) * cols,
              out + static_cast<std::size_t>(done) * rows);
        }
      } else {
        GemvBatchBf16Kernel<<<grid, kThreads, 0, stream>>>(wb, rows, cols, x,
                                                           out);
      }
      break;
    }
  }
}

void Gemm(const void* base, core::GgmlType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          std::uint32_t batch, hipStream_t stream) {
  if (type == core::GgmlType::kQ8_0) {
    (void)flint_mmq_q8_0_dense(base, x, out, static_cast<int>(rows),
                               static_cast<int>(batch), static_cast<int>(cols),
                               stream);
    return;
  }
  GemvBatch(base, type, rows, cols, row_bytes, x, out, batch, stream);
}

void GemmMoe(const void* base, std::uint32_t rows, std::uint32_t cols,
             const float* x, const std::int32_t* ids, float* out,
             std::uint32_t n_tokens, std::uint32_t n_experts,
             std::uint32_t n_expert_used, hipStream_t stream,
             const unsigned int* h_counts) {
  // Routed launch bounds from the actual per-expert counts (flint_mmq
  // defaults cover all column tiles for all experts, which is whole-chunk
  // padding; tools/bench/flint_moe_mmq_bench.hip measured ~35 % off the
  // MoE tier with both knobs at B=2048). Grid-only: outputs unchanged.
  if (h_counts != nullptr) {
    unsigned int max_rows = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      max_rows = h_counts[e] > max_rows ? h_counts[e] : max_rows;
    }
    flint_mmq_set_routed_max_expert_rows(static_cast<int>(max_rows));
    flint_mmq_set_routed_tile_cols(flint_mmq_routed_tile_cols_for_counts(
        h_counts, static_cast<int>(n_experts)));
  }
  (void)flint_mmq_q8_0_moe_raw(
      base, x, ids, out, static_cast<int>(rows), static_cast<int>(cols),
      static_cast<int>(n_tokens), static_cast<int>(n_experts),
      static_cast<int>(n_expert_used), stream);
  if (h_counts != nullptr) {
    flint_mmq_set_routed_max_expert_rows(0);
    flint_mmq_set_routed_tile_cols(0);
  }
}

void DequantRow(const void* base, core::GgmlType type, std::uint32_t row,
                std::uint32_t cols, std::size_t row_bytes, float* out,
                hipStream_t stream) {
  const std::uint8_t* w = static_cast<const std::uint8_t*>(base) +
                          static_cast<std::size_t>(row) * row_bytes;
  DequantRowKernel<<<1, kThreads, 0, stream>>>(w, type, cols, row_bytes, out);
}

}  // namespace gufo::models::flint::rocm
