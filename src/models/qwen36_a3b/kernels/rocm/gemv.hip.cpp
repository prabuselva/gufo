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

// One block of four waves handles four consecutive output rows: wave w owns
// row blockIdx.x*4+w. Each wave walks its row with a four-block unrolled
// stream so several weight loads stay in flight, then reduces within the
// wave. The activation row is read from L1/L2 once per wave; splitting a
// single row across waves would multiply that traffic above the weight row.
__global__ void GemvQ8_0(const Q8_0Block* __restrict__ w,
                         const float* __restrict__ x, float* __restrict__ out,
                         std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 32;
  const Q8_0Block* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const float d = __half2float(row[b].d);
    acc += d * static_cast<float>(row[b].qs[lane]) * x[b * 32 + lane];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

// Grouped expert GEMV: a block of four waves folds four consecutive
// (slot, row) pairs of the selected-expert product. The expert id is read
// from device memory, so the whole routed MoE stays asynchronous.
__global__ void GemvGroupedQ8_0(const Q8_0Block* __restrict__ w,
                                const std::int32_t* __restrict__ ids,
                                std::size_t expert_stride_blocks,
                                const float* __restrict__ x,
                                std::uint32_t x_stride, float* __restrict__ out,
                                std::uint32_t used, std::uint32_t rows,
                                std::uint32_t cols) {
  const std::uint32_t pair = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (pair >= used * rows) {
    return;
  }
  const std::uint32_t s = pair / rows;
  const std::uint32_t r = pair - s * rows;
  const std::uint32_t nblocks = cols / 32;
  const Q8_0Block* __restrict__ row =
      w + static_cast<std::size_t>(ids[s]) * expert_stride_blocks +
      static_cast<std::size_t>(r) * nblocks;
  const float* __restrict__ xs = x + static_cast<std::size_t>(s) * x_stride;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const float d = __half2float(row[b].d);
    acc += d * static_cast<float>(row[b].qs[lane]) * xs[b * 32 + lane];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[pair] = acc;
  }
}

// Grouped expert GEMV for the dense (F32/BF16) expert stacks: same
// (slot, row) pairing as GemvGroupedQ8_0, element-strided within the row.
template <typename T>
__global__ void GemvGroupedDense(const T* __restrict__ w,
                                 const std::int32_t* __restrict__ ids,
                                 std::size_t expert_stride_elems,
                                 const float* __restrict__ x,
                                 std::uint32_t x_stride, float* __restrict__ out,
                                 std::uint32_t used, std::uint32_t rows,
                                 std::uint32_t cols) {
  const std::uint32_t pair = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (pair >= used * rows) {
    return;
  }
  const std::uint32_t s = pair / rows;
  const std::uint32_t r = pair - s * rows;
  const T* __restrict__ row =
      w + static_cast<std::size_t>(ids[s]) * expert_stride_elems +
      static_cast<std::size_t>(r) * cols;
  const float* __restrict__ xs = x + static_cast<std::size_t>(s) * x_stride;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t i = lane; i < cols; i += 32U) {
    acc += static_cast<float>(row[i]) * xs[i];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[pair] = acc;
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
#pragma unroll 4
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
#pragma unroll 4
  for (std::uint32_t i = lane; i < cols; i += 32) {
    acc += static_cast<float>(row[i]) * x[i];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

// Dequantize a single row of a quantized matrix (the token-embedding lookup).
// One warp per Q8_0 block; one thread per element for F32/BF16.
__global__ void EmbedRowQ8_0(const Q8_0Block* __restrict__ w,
                             float* __restrict__ out) {
  const std::uint32_t b = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const float d = __half2float(w[b].d);
  out[b * 32 + lane] = d * static_cast<float>(w[b].qs[lane]);
}

__global__ void EmbedRowF32(const float* __restrict__ w,
                            float* __restrict__ out, std::uint32_t cols) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < cols) {
    out[idx] = w[idx];
  }
}

__global__ void EmbedRowBf16(const hip_bfloat16* __restrict__ w,
                             float* __restrict__ out, std::uint32_t cols) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < cols) {
    out[idx] = static_cast<float>(w[idx]);
  }
}

}  // namespace

void EmbedRow(const void* base, GemvType type, std::uint32_t row,
              std::uint32_t cols, float* out, hipStream_t stream) {
  switch (type) {
    case GemvType::kQ8_0: {
      const std::uint32_t nblocks = cols / 32;
      const Q8_0Block* w = static_cast<const Q8_0Block*>(base) +
                           static_cast<std::size_t>(row) * nblocks;
      EmbedRowQ8_0<<<nblocks, 32, 0, stream>>>(w, out);
      break;
    }
    case GemvType::kF32: {
      const float* w = static_cast<const float*>(base) +
                       static_cast<std::size_t>(row) * cols;
      const std::size_t block = 256;
      const std::size_t grid =
          std::min<std::size_t>((cols + block - 1) / block, 65535);
      EmbedRowF32<<<grid, block, 0, stream>>>(w, out, cols);
      break;
    }
    case GemvType::kBF16: {
      const hip_bfloat16* w = static_cast<const hip_bfloat16*>(base) +
                              static_cast<std::size_t>(row) * cols;
      const std::size_t block = 256;
      const std::size_t grid =
          std::min<std::size_t>((cols + block - 1) / block, 65535);
      EmbedRowBf16<<<grid, block, 0, stream>>>(w, out, cols);
      break;
    }
  }
}

void Gemv(const void* base, GemvType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          hipStream_t stream) {
  const dim3 grid(rows);
  switch (type) {
    case GemvType::kQ8_0:
      GemvQ8_0<<<(rows + 3U) / 4U, dim3(128), 0, stream>>>(
          static_cast<const Q8_0Block*>(base), x, out, rows, cols);
      break;
    case GemvType::kF32:
      GemvF32<<<grid, dim3(32), 0, stream>>>(static_cast<const float*>(base),
                                             x, out, rows, cols);
      break;
    case GemvType::kBF16:
      GemvBf16<<<grid, dim3(32), 0, stream>>>(
          static_cast<const hip_bfloat16*>(base), x, out, rows, cols);
      break;
  }
}

void GemvGrouped(const void* base, GemvType type, std::size_t expert_stride,
                 const std::int32_t* ids, std::uint32_t used,
                 std::uint32_t rows, std::uint32_t cols, const float* x,
                 std::uint32_t x_stride, float* out, hipStream_t stream) {
  const std::size_t pairs = static_cast<std::size_t>(used) * rows;
  const std::uint32_t grid =
      static_cast<std::uint32_t>((pairs + 3U) / 4U);
  switch (type) {
    case GemvType::kQ8_0:
      GemvGroupedQ8_0<<<grid, dim3(128), 0, stream>>>(
          static_cast<const Q8_0Block*>(base), ids,
          expert_stride / sizeof(Q8_0Block), x, x_stride, out, used, rows,
          cols);
      break;
    case GemvType::kF32:
      GemvGroupedDense<float><<<grid, dim3(128), 0, stream>>>(
          static_cast<const float*>(base), ids,
          expert_stride / sizeof(float), x, x_stride, out, used, rows, cols);
      break;
    case GemvType::kBF16:
      GemvGroupedDense<hip_bfloat16><<<grid, dim3(128), 0, stream>>>(
          static_cast<const hip_bfloat16*>(base), ids,
          expert_stride / sizeof(hip_bfloat16), x, x_stride, out, used, rows,
          cols);
      break;
  }
}

}  // namespace gufo::models::qwen36_a3b::rocm