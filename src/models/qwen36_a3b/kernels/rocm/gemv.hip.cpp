#include "src/models/qwen36_a3b/kernels/rocm/gemv.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen36_a3b::rocm {
namespace {

// Q8_0 block: 32 elements, {half d; int8 qs[32]}, 34 bytes. The size must stay
// 34 (no alignas padding) to match the GGUF storage and the scalar oracle's
// block_q8_0; the half scale always lands on an even offset.
struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};

__device__ __forceinline__ float WarpReduceSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}

// One warp per output row. Each lane accumulates its strided slice of the
// reduction, then a single warp shuffle folds the row.
__global__ void GemvQ8_0(const Q8_0Block* __restrict__ w,
                         const float* __restrict__ x, float* __restrict__ out,
                         std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x;
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 32;
  const Q8_0Block* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x;
  float acc = 0.0f;
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const float d = __half2float(row[b].d);
    acc += d * static_cast<float>(row[b].qs[lane]) * x[b * 32 + lane];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

__global__ void GemvF32(const float* __restrict__ w,
                        const float* __restrict__ x, float* __restrict__ out,
                        std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x;
  if (r >= rows) {
    return;
  }
  const float* __restrict__ row = w + static_cast<std::size_t>(r) * cols;
  const std::uint32_t lane = threadIdx.x;
  float acc = 0.0f;
  for (std::uint32_t i = lane; i < cols; i += 32) {
    acc += row[i] * x[i];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

__global__ void GemvBf16(const hip_bfloat16* __restrict__ w,
                         const float* __restrict__ x, float* __restrict__ out,
                         std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x;
  if (r >= rows) {
    return;
  }
  const hip_bfloat16* __restrict__ row = w + static_cast<std::size_t>(r) * cols;
  const std::uint32_t lane = threadIdx.x;
  float acc = 0.0f;
  for (std::uint32_t i = lane; i < cols; i += 32) {
    acc += static_cast<float>(row[i]) * x[i];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

}  // namespace

void Gemv(const void* base, GemvType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          hipStream_t stream) {
  const dim3 grid(rows);
  const dim3 block(32);
  switch (type) {
    case GemvType::kQ8_0:
      GemvQ8_0<<<grid, block, 0, stream>>>(static_cast<const Q8_0Block*>(base),
                                           x, out, rows, cols);
      break;
    case GemvType::kF32:
      GemvF32<<<grid, block, 0, stream>>>(static_cast<const float*>(base), x,
                                          out, rows, cols);
      break;
    case GemvType::kBF16:
      GemvBf16<<<grid, block, 0, stream>>>(
          static_cast<const hip_bfloat16*>(base), x, out, rows, cols);
      break;
  }
}

}  // namespace gufo::models::qwen36_a3b::rocm