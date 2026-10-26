#include "src/models/flint/kernels/rocm/linear.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "qfn_mmq.h"

namespace gufo::models::flint::rocm {
namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr std::uint32_t kQ8Block = 32;
constexpr std::size_t kQ8BlockBytes = 34;

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
    float part = 0.0F;
    for (std::uint32_t j = 0; j < kQ8Block; ++j) {
      part += static_cast<float>(qs[j]) * xk[j];
    }
    acc += d * part;
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
    float part = 0.0F;
    for (std::uint32_t j = 0; j < kQ8Block; ++j) {
      part += static_cast<float>(qs[j]) * xk[j];
    }
    acc += d * part;
  }
  acc = WarpSum(acc);
  if (lane == 0) {
    out[row] = acc;
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
    case core::GgmlType::kQ8_0:
      GemvBatchQ8Kernel<<<grid, kThreads, 0, stream>>>(
          static_cast<const std::uint8_t*>(base), rows, cols, x, out);
      break;
    default:
      if (type == core::GgmlType::kF32) {
        GemvBatchF32Kernel<<<grid, kThreads, 0, stream>>>(
            static_cast<const float*>(base), rows, cols, x, out);
      } else {
        GemvBatchBf16Kernel<<<grid, kThreads, 0, stream>>>(
            static_cast<const std::uint16_t*>(base), rows, cols, x, out);
      }
      break;
  }
}

void Gemm(const void* base, core::GgmlType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          std::uint32_t batch, hipStream_t stream) {
  if (type == core::GgmlType::kQ8_0) {
    (void)qfn_mmq_q8_0_dense(base, x, out, static_cast<int>(rows),
                             static_cast<int>(batch), static_cast<int>(cols),
                             stream);
    return;
  }
  GemvBatch(base, type, rows, cols, row_bytes, x, out, batch, stream);
}

void GemmMoe(const void* base, std::uint32_t rows, std::uint32_t cols,
             const float* x, const std::int32_t* ids, float* out,
             std::uint32_t n_tokens, std::uint32_t n_experts,
             std::uint32_t n_expert_used, hipStream_t stream) {
  (void)qfn_mmq_q8_0_moe_raw(base, x, ids, out, static_cast<int>(rows),
                             static_cast<int>(cols), static_cast<int>(n_tokens),
                             static_cast<int>(n_experts),
                             static_cast<int>(n_expert_used), stream);
}

void DequantRow(const void* base, core::GgmlType type, std::uint32_t row,
                std::uint32_t cols, std::size_t row_bytes, float* out,
                hipStream_t stream) {
  const std::uint8_t* w = static_cast<const std::uint8_t*>(base) +
                          static_cast<std::size_t>(row) * row_bytes;
  DequantRowKernel<<<1, kThreads, 0, stream>>>(w, type, cols, row_bytes, out);
}

}  // namespace gufo::models::flint::rocm
