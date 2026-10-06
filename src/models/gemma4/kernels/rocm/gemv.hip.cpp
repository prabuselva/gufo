#include "src/models/gemma4/kernels/rocm/gemv.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::gemma4::rocm {
namespace {

// Q8_0 block: 32 elements, {half d; int8 qs[32]}, 34 bytes. The size must stay
// 34 (no alignas padding) to match the GGUF storage and the scalar oracle's
// block_q8_0; the half scale always lands on an even offset.
struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};

// Q4_K / Q5_K super-blocks (256 elements each), the routed-expert encodings of
// a Q4_K_XL artifact. Field order and size match the canonical GGUF layout
// (src/core/quant/ggml_dequant.hpp) and the WMMA path's copies.
struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q5KBlock) == 176, "block_q5_K must be 176 bytes");

__device__ __forceinline__ float WarpReduceSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}

// Sub-block (scale, minimum) unpack for the Q4_K / Q5_K super-block header, a
// device port of the oracle's GetQ4ScaleMin. `index` is the 32-element group
// (0..7) within the super-block.
__device__ __forceinline__ void Q4ScaleMin(int index,
                                           const std::uint8_t* packed,
                                           int& scale, int& minimum) {
  if (index < 4) {
    scale = packed[index] & 0x3F;
    minimum = packed[index + 4] & 0x3F;
  } else {
    scale = (packed[index + 4] & 0x0F) | ((packed[index - 4] >> 6) << 4);
    minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
  }
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

// Multi-row dense GEMV for the speculative verify pass: one wave owns an
// output row and dots the weight row against `Rows` activation rows in the
// same walk, so the weight matrix is read once for all tokens. Each per-row
// dot is bit-identical to GemvQ8_0.
template<std::uint32_t Rows>
__global__ void GemvRowsQ8_0(const Q8_0Block* __restrict__ w,
                             const float* __restrict__ x,
                             std::uint32_t x_stride, float* __restrict__ out,
                             std::uint32_t out_stride, std::uint32_t rows,
                             std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 32;
  const Q8_0Block* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
#pragma unroll 4
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const float d = __half2float(row[b].d);
    const float q = static_cast<float>(row[b].qs[lane]);
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      acc[o] +=
          d * q * x[static_cast<std::size_t>(o) * x_stride + b * 32 + lane];
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[static_cast<std::size_t>(o) * out_stride + r] = acc[o];
    }
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

// Grouped expert GEMV for the Q4_K routed stacks. Same (slot, row) pairing as
// GemvGroupedQ8_0, but a warp walks 256-element super-blocks: lane L takes the
// element at offset L of each of the eight 32-element groups (e = L + 32 j), so
// the nibble reads coalesce across the warp. The dequant mirrors the oracle's
// Q4Value: value = d * scale_j * q - dmin * min_j, with the nibble low for even
// j and high for odd j.
__global__ void GemvGroupedQ4K(const Q4KBlock* __restrict__ w,
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
  const std::uint32_t nblocks = cols / 256;
  const Q4KBlock* __restrict__ row =
      w + static_cast<std::size_t>(ids[s]) * expert_stride_blocks +
      static_cast<std::size_t>(r) * nblocks;
  const float* __restrict__ xs = x + static_cast<std::size_t>(s) * x_stride;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const Q4KBlock& blk = row[b];
    const float d = __half2float(blk.d);
    const float dmin = __half2float(blk.dmin);
    const float* xb = xs + b * 256 + lane;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint8_t packed = blk.qs[(j >> 1) * 32 + lane];
      const int quant = (j & 1) ? (packed >> 4) : (packed & 0xF);
      int sc = 0;
      int mn = 0;
      Q4ScaleMin(j, blk.scales, sc, mn);
      acc += (d * static_cast<float>(sc) * static_cast<float>(quant) -
              dmin * static_cast<float>(mn)) *
             xb[32 * j];
    }
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[pair] = acc;
  }
}

// Grouped expert GEMV for the Q5_K routed stacks. A 5-bit code is the Q4_K
// nibble plus a fifth bit from qh[lane] at position j; the sub-block scale
// index is j, matching the oracle's Q5Value.
__global__ void GemvGroupedQ5K(const Q5KBlock* __restrict__ w,
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
  const std::uint32_t nblocks = cols / 256;
  const Q5KBlock* __restrict__ row =
      w + static_cast<std::size_t>(ids[s]) * expert_stride_blocks +
      static_cast<std::size_t>(r) * nblocks;
  const float* __restrict__ xs = x + static_cast<std::size_t>(s) * x_stride;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const Q5KBlock& blk = row[b];
    const float d = __half2float(blk.d);
    const float dmin = __half2float(blk.dmin);
    const std::uint8_t qhb = blk.qh[lane];
    const float* xb = xs + b * 256 + lane;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint8_t qb = blk.qs[(j >> 1) * 32 + lane];
      const int quant4 = (j & 1) ? (qb >> 4) : (qb & 0xF);
      const int quant = quant4 + (((qhb >> j) & 1) ? 16 : 0);
      int sc = 0;
      int mn = 0;
      Q4ScaleMin(j, blk.scales, sc, mn);
      acc += (d * static_cast<float>(sc) * static_cast<float>(quant) -
              dmin * static_cast<float>(mn)) *
             xb[32 * j];
    }
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[pair] = acc;
  }
}

// Grouped gate+up pair for the routed experts: both expert stacks share the
// ids, the activation row and the shape, so one launch folds 2*used*rows
// (matrix, slot, row) triples. The per-row dot is bit-identical to
// GemvGroupedQ8_0.
__global__ void GemvGroupedQ8_0Pair(
    const Q8_0Block* __restrict__ wa, const Q8_0Block* __restrict__ wb,
    const std::int32_t* __restrict__ ids, std::size_t expert_stride_blocks,
    const float* __restrict__ x, std::uint32_t x_stride,
    float* __restrict__ out_a, float* __restrict__ out_b, std::uint32_t used,
    std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t total = used * rows;
  const std::uint32_t p = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (p >= 2U * total) {
    return;
  }
  const bool first = p < total;
  const std::uint32_t pair = first ? p : p - total;
  const std::uint32_t s = pair / rows;
  const std::uint32_t r = pair - s * rows;
  const std::uint32_t nblocks = cols / 32;
  const Q8_0Block* __restrict__ row =
      (first ? wa : wb) +
      static_cast<std::size_t>(ids[s]) * expert_stride_blocks +
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
    if (first) {
      out_a[pair] = acc;
    } else {
      out_b[pair] = acc;
    }
  }
}

// Fused multi projection: up to four Q8_0 matrices sharing one activation
// row. Wave handles global row blockIdx.x*4+w and picks the owning matrix
// from the cumulative row offsets; the per-row dot is bit-identical to
// GemvQ8_0. Folding the attention q/k/v projections into one launch
// measurably beats separate launches at cold L2 (bench
// tools/bench/moe_gemv_bench.hip).
struct Multi4Args {
  const Q8_0Block* w[4];
  float* out[4];
  std::uint32_t off[5];  // cumulative row offsets, off[0] = 0
  std::uint32_t nb[4];   // quant blocks per row
};

__global__ void GemvQ8_0Multi4(Multi4Args a, const float* __restrict__ x) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= a.off[4]) {
    return;
  }
  std::uint32_t t = 0;
  while (r >= a.off[t + 1]) {
    ++t;
  }
  const std::uint32_t row_index = r - a.off[t];
  const Q8_0Block* __restrict__ row =
      a.w[t] + static_cast<std::size_t>(row_index) * a.nb[t];
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t b = 0; b < a.nb[t]; ++b) {
    const float d = __half2float(row[b].d);
    acc += d * static_cast<float>(row[b].qs[lane]) * x[b * 32 + lane];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    a.out[t][row_index] = acc;
  }
}

// ---- vec4 Q8_0 variants (cols % 128 == 0) -------------------------------
//
// The scalar kernels load one int8 per lane per quant block (a 32-byte warp
// transaction every 34-byte stride). The vec4 variants load a uint32 per lane
// covering four contiguous codes of one block: for a group of four blocks,
// lane l reads block g*4 + l/8 at word (4l) % 32, so the warp pulls a full
// 128-byte qs span per instruction. Each block's per-block scale d is applied
// to the four codes the lane read, so dequantization stays exact; only the
// summation order differs from the scalar kernel (a dot-product reorder, well
// inside the GEMV tolerance). Requires cols % 128 == 0; the host dispatch
// falls back to the scalar kernel otherwise.

__global__ void GemvQ8_0Vec4(const Q8_0Block* __restrict__ w,
                             const float* __restrict__ x,
                             float* __restrict__ out, std::uint32_t rows,
                             std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t groups = cols / 128U;
  const Q8_0Block* __restrict__ row =
      w + static_cast<std::size_t>(r) * (cols / 32U);
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t g = 0; g < groups; ++g) {
    const std::uint32_t blk = g * 4U + lane / 8U;
    const std::uint32_t word = (4U * lane) % 32U;
    std::int32_t packed;
    __builtin_memcpy(&packed, &row[blk].qs[word], 4);
    const float d = __half2float(row[blk].d);
    const std::uint32_t k = blk * 32U + word;
    acc += d * static_cast<float>(static_cast<std::int8_t>(packed & 0xFF)) *
           x[k];
    acc += d *
           static_cast<float>(static_cast<std::int8_t>((packed >> 8) & 0xFF)) *
           x[k + 1];
    acc += d *
           static_cast<float>(static_cast<std::int8_t>((packed >> 16) & 0xFF)) *
           x[k + 2];
    acc += d *
           static_cast<float>(static_cast<std::int8_t>((packed >> 24) & 0xFF)) *
           x[k + 3];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

__global__ void GemvQ8_0Multi4Vec4(Multi4Args a, const float* __restrict__ x) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= a.off[4]) {
    return;
  }
  std::uint32_t t = 0;
  while (r >= a.off[t + 1]) {
    ++t;
  }
  const std::uint32_t row_index = r - a.off[t];
  const std::uint32_t groups = a.nb[t] / 4U;
  const Q8_0Block* __restrict__ row =
      a.w[t] + static_cast<std::size_t>(row_index) * a.nb[t];
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t g = 0; g < groups; ++g) {
    const std::uint32_t blk = g * 4U + lane / 8U;
    const std::uint32_t word = (4U * lane) % 32U;
    std::int32_t packed;
    __builtin_memcpy(&packed, &row[blk].qs[word], 4);
    const float d = __half2float(row[blk].d);
    const std::uint32_t k = blk * 32U + word;
    acc += d * static_cast<float>(static_cast<std::int8_t>(packed & 0xFF)) *
           x[k];
    acc += d *
           static_cast<float>(static_cast<std::int8_t>((packed >> 8) & 0xFF)) *
           x[k + 1];
    acc += d *
           static_cast<float>(static_cast<std::int8_t>((packed >> 16) & 0xFF)) *
           x[k + 2];
    acc += d *
           static_cast<float>(static_cast<std::int8_t>((packed >> 24) & 0xFF)) *
           x[k + 3];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    a.out[t][row_index] = acc;
  }
}

// Grouped expert GEMV for the dense (F32/BF16) expert stacks: same
// (slot, row) pairing as GemvGroupedQ8_0, element-strided within the row.
template<typename T>
__global__ void GemvGroupedDense(const T* __restrict__ w,
                                 const std::int32_t* __restrict__ ids,
                                 std::size_t expert_stride_elems,
                                 const float* __restrict__ x,
                                 std::uint32_t x_stride,
                                 float* __restrict__ out, std::uint32_t used,
                                 std::uint32_t rows, std::uint32_t cols) {
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

// Multi-row dense GEMV for the F32/BF16 stacks: one warp owns an output row
// and dots it against `Rows` activation rows in the same walk. Each per-row
// dot is bit-identical to GemvF32/GemvBf16.
template<typename T, std::uint32_t Rows>
__global__ void GemvRowsDense(const T* __restrict__ w,
                              const float* __restrict__ x,
                              std::uint32_t x_stride, float* __restrict__ out,
                              std::uint32_t out_stride, std::uint32_t rows,
                              std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x;
  if (r >= rows) {
    return;
  }
  const T* __restrict__ row = w + static_cast<std::size_t>(r) * cols;
  const std::uint32_t lane = threadIdx.x;
  float acc[Rows] = {};
#pragma unroll 4
  for (std::uint32_t i = lane; i < cols; i += 32) {
    const float wv = static_cast<float>(row[i]);
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      acc[o] += wv * x[static_cast<std::size_t>(o) * x_stride + i];
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[static_cast<std::size_t>(o) * out_stride + r] = acc[o];
    }
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
    // Q4_K/Q5_K are routed-expert encodings; ValidateTypes keeps them off the
    // embedding lookup, which only ever sees the Q8_0/F32/BF16 token table.
    case GemvType::kQ4_K:
    case GemvType::kQ5_K:
      break;
  }
}

void Gemv(const void* base, GemvType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          hipStream_t stream) {
  const dim3 grid(rows);
  switch (type) {
    case GemvType::kQ8_0:
      if (cols % 128U == 0U) {
        GemvQ8_0Vec4<<<(rows + 3U) / 4U, dim3(128), 0, stream>>>(
            static_cast<const Q8_0Block*>(base), x, out, rows, cols);
      } else {
        GemvQ8_0<<<(rows + 3U) / 4U, dim3(128), 0, stream>>>(
            static_cast<const Q8_0Block*>(base), x, out, rows, cols);
      }
      break;
    case GemvType::kF32:
      GemvF32<<<grid, dim3(32), 0, stream>>>(static_cast<const float*>(base), x,
                                             out, rows, cols);
      break;
    case GemvType::kBF16:
      GemvBf16<<<grid, dim3(32), 0, stream>>>(
          static_cast<const hip_bfloat16*>(base), x, out, rows, cols);
      break;
    // Routed-expert encodings never reach the dense single-row GEMV; they are
    // decoded by GemvGrouped (decode) and the WMMA routed GEMM (prefill).
    case GemvType::kQ4_K:
    case GemvType::kQ5_K:
      break;
  }
}

void GemvRows(const void* base, GemvType type, std::uint32_t tokens,
              std::uint32_t rows, std::uint32_t cols, std::size_t row_bytes,
              const float* x, std::uint32_t x_stride, float* out,
              std::uint32_t out_stride, hipStream_t stream) {
  (void)row_bytes;
  const std::uint32_t grid = (rows + 3U) / 4U;
#define GUFO_GEMV_ROWS(R)                                                      \
  case R:                                                                      \
    switch (type) {                                                            \
      case GemvType::kQ8_0:                                                    \
        GemvRowsQ8_0<R><<<grid, dim3(128), 0, stream>>>(                       \
            static_cast<const Q8_0Block*>(base), x, x_stride, out, out_stride, \
            rows, cols);                                                       \
        break;                                                                 \
      case GemvType::kF32:                                                     \
        GemvRowsDense<float, R><<<rows, dim3(32), 0, stream>>>(                \
            static_cast<const float*>(base), x, x_stride, out, out_stride,     \
            rows, cols);                                                       \
        break;                                                                 \
      case GemvType::kBF16:                                                    \
        GemvRowsDense<hip_bfloat16, R><<<rows, dim3(32), 0, stream>>>(         \
            static_cast<const hip_bfloat16*>(base), x, x_stride, out,          \
            out_stride, rows, cols);                                           \
        break;                                                                 \
      /* Routed-expert encodings never reach the dense row GEMV. */            \
      case GemvType::kQ4_K:                                                    \
      case GemvType::kQ5_K:                                                    \
        break;                                                                 \
    }                                                                          \
    break;
  switch (tokens) {
    GUFO_GEMV_ROWS(2U)
    GUFO_GEMV_ROWS(3U)
    GUFO_GEMV_ROWS(4U)
    GUFO_GEMV_ROWS(5U)
    default:
      break;
  }
#undef GUFO_GEMV_ROWS
}

void GemvGrouped(const void* base, GemvType type, std::size_t expert_stride,
                 const std::int32_t* ids, std::uint32_t used,
                 std::uint32_t rows, std::uint32_t cols, const float* x,
                 std::uint32_t x_stride, float* out, hipStream_t stream) {
  const std::size_t pairs = static_cast<std::size_t>(used) * rows;
  const std::uint32_t grid = static_cast<std::uint32_t>((pairs + 3U) / 4U);
  switch (type) {
    case GemvType::kQ8_0:
      // The grouped MoE GEMV keeps the scalar 1-byte/lane order: its contract
      // is 1e-4 (the expert outputs feed a weighted sum), and the vec4 reorder
      // lands at ~2e-4. The dense and fused-qkv GEMVs (2e-3 contract) use vec4.
      GemvGroupedQ8_0<<<grid, dim3(128), 0, stream>>>(
          static_cast<const Q8_0Block*>(base), ids,
          expert_stride / sizeof(Q8_0Block), x, x_stride, out, used, rows,
          cols);
      break;
    case GemvType::kQ4_K:
      GemvGroupedQ4K<<<grid, dim3(128), 0, stream>>>(
          static_cast<const Q4KBlock*>(base), ids,
          expert_stride / sizeof(Q4KBlock), x, x_stride, out, used, rows, cols);
      break;
    case GemvType::kQ5_K:
      GemvGroupedQ5K<<<grid, dim3(128), 0, stream>>>(
          static_cast<const Q5KBlock*>(base), ids,
          expert_stride / sizeof(Q5KBlock), x, x_stride, out, used, rows, cols);
      break;
    case GemvType::kF32:
      GemvGroupedDense<float><<<grid, dim3(128), 0, stream>>>(
          static_cast<const float*>(base), ids, expert_stride / sizeof(float),
          x, x_stride, out, used, rows, cols);
      break;
    case GemvType::kBF16:
      GemvGroupedDense<hip_bfloat16><<<grid, dim3(128), 0, stream>>>(
          static_cast<const hip_bfloat16*>(base), ids,
          expert_stride / sizeof(hip_bfloat16), x, x_stride, out, used, rows,
          cols);
      break;
  }
}

bool GemvGroupedPair(const void* wa, const void* wb, GemvType type,
                     std::size_t expert_stride, const std::int32_t* ids,
                     std::uint32_t used, std::uint32_t rows, std::uint32_t cols,
                     const float* x, std::uint32_t x_stride, float* out_a,
                     float* out_b, hipStream_t stream) {
  if (type != GemvType::kQ8_0) {
    return false;
  }
  const std::size_t pairs = 2ULL * used * rows;
  GemvGroupedQ8_0Pair<<<static_cast<std::uint32_t>((pairs + 3U) / 4U),
                        dim3(128), 0, stream>>>(
      static_cast<const Q8_0Block*>(wa), static_cast<const Q8_0Block*>(wb), ids,
      expert_stride / sizeof(Q8_0Block), x, x_stride, out_a, out_b, used, rows,
      cols);
  return true;
}

bool GemvMulti(const GemvMultiProj* projs, std::uint32_t n, const float* x,
               hipStream_t stream) {
  if (n == 0U || n > 4U) {
    return false;
  }
  // Fuse the Q8_0 subset into one Multi4 launch. Non-Q8_0 projections run as
  // plain Gemv calls, which is bit-identical to the unfused path; empty
  // projections are skipped.
  Multi4Args a{};
  std::uint32_t total = 0;
  std::uint32_t m = 0;
  bool vec4 = true;
  for (std::uint32_t i = 0; i < n; ++i) {
    if (projs[i].type != GemvType::kQ8_0 || projs[i].rows == 0U) {
      continue;
    }
    a.w[m] = static_cast<const Q8_0Block*>(projs[i].base);
    a.out[m] = projs[i].out;
    a.nb[m] = projs[i].cols / 32U;
    vec4 = vec4 && (a.nb[m] % 4U == 0U);
    a.off[m] = total;
    total += projs[i].rows;
    ++m;
  }
  for (std::uint32_t i = m; i < 4U; ++i) {
    a.off[i] = total;
  }
  a.off[4] = total;
  if (m > 1U) {
    if (vec4) {
      GemvQ8_0Multi4Vec4<<<(total + 3U) / 4U, dim3(128), 0, stream>>>(a, x);
    } else {
      GemvQ8_0Multi4<<<(total + 3U) / 4U, dim3(128), 0, stream>>>(a, x);
    }
  }
  for (std::uint32_t i = 0; i < n; ++i) {
    const bool fused =
        m > 1U && projs[i].type == GemvType::kQ8_0 && projs[i].rows != 0U;
    if (!fused && projs[i].rows != 0U) {
      Gemv(projs[i].base, projs[i].type, projs[i].rows, projs[i].cols, 0U, x,
           projs[i].out, stream);
    }
  }
  return true;
}

}  // namespace gufo::models::gemma4::rocm