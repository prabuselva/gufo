#include "src/models/ling3_flash/kernels/rocm/gemv.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::ling3_flash::rocm {
namespace {

// Q4_0 block: 32 elements, {half d; uint8 qs[16] with two 4-bit codes per
// byte, value = d * (code - 8)}, 18 bytes. The size must stay 18 (no alignas
// padding) to match the GGUF storage and the scalar oracle's Q4_0 decode; the
// whole MTP draft block uses this encoding.
struct Q4_0Block {
  __half d;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q4_0Block) == 18, "block_q4_0 must be 18 bytes");

// Q8_0 block: 32 elements, {half d; int8 qs[32]}, 34 bytes. The size must stay
// 34 (no alignas padding) to match the GGUF storage and the scalar oracle's
// block_q8_0; the half scale always lands on an even offset.
struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};

// Q4_K / Q5_K / Q6_K super-blocks (256 elements each). Field order and size
// match the canonical GGUF layout (src/core/quant/ggml_dequant.hpp).
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

struct Q6KBlock {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  __half d;
};
static_assert(sizeof(Q6KBlock) == 210, "block_q6_K must be 210 bytes");

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

__global__ void GemvQ4_0(const Q4_0Block* __restrict__ w,
                         const float* __restrict__ x, float* __restrict__ out,
                         std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 32;
  const Q4_0Block* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const float d = __half2float(row[b].d);
    const std::uint8_t packed = row[b].qs[lane & 15U];
    const int quant = (lane < 16U) ? (packed & 0xF) : (packed >> 4);
    acc += d * static_cast<float>(quant - 8) * x[b * 32 + lane];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

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

// ---- vec4 Q8_0 variant (cols % 128 == 0) --------------------------------
//
// The scalar kernel loads one int8 per lane per quant block (a 32-byte warp
// transaction every 34-byte stride). This variant loads a uint32 per lane
// covering four contiguous codes of one block: for a group of four blocks,
// lane l reads block g*4 + l/8 at word (4l) % 32, so the warp pulls a full
// 128-byte qs span per instruction. Each block's scale applies to the four
// codes the lane read, so dequantization stays exact; only the summation
// order differs from the scalar kernel (inside the 2e-3 GEMV tolerance, and
// the retained gemma4 form measures ~1.1e-3 worst case). Requires
// cols % 128 == 0; the host dispatch falls back to the scalar kernel
// otherwise. Ported from src/models/gemma4/kernels/rocm/gemv.hip.cpp.

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
    acc +=
        d * static_cast<float>(static_cast<std::int8_t>(packed & 0xFF)) * x[k];
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

// Q4_K walk: lane L takes the element at offset L of each of the eight
// 32-element groups (e = L + 32 j), so the nibble reads coalesce across the
// warp. The dequant mirrors the oracle's Q4Value: value = d*scale_j*q -
// dmin*min_j, with the nibble low for even j and high for odd j.
__global__ void GemvQ4K(const Q4KBlock* __restrict__ w,
                        const float* __restrict__ x, float* __restrict__ out,
                        std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 256;
  const Q4KBlock* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const Q4KBlock& blk = row[b];
    const float d = __half2float(blk.d);
    const float dmin = __half2float(blk.dmin);
    const float* xrow = x + b * 256 + lane;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint8_t packed = blk.qs[(j >> 1) * 32 + lane];
      const int quant = (j & 1) ? (packed >> 4) : (packed & 0xF);
      int sc = 0;
      int mn = 0;
      Q4ScaleMin(j, blk.scales, sc, mn);
      acc += (d * static_cast<float>(sc) * static_cast<float>(quant) -
              dmin * static_cast<float>(mn)) *
             xrow[32 * j];
    }
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

// Q5_K walk: a 5-bit code is the Q4_K nibble plus a fifth bit from qh[lane]
// at position j; the sub-block scale index is j, matching the oracle's
// Q5Value.
__global__ void GemvQ5K(const Q5KBlock* __restrict__ w,
                        const float* __restrict__ x, float* __restrict__ out,
                        std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 256;
  const Q5KBlock* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const Q5KBlock& blk = row[b];
    const float d = __half2float(blk.d);
    const float dmin = __half2float(blk.dmin);
    const std::uint8_t qhb = blk.qh[lane];
    const float* xrow = x + b * 256 + lane;
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
             xrow[32 * j];
    }
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

// Q6_K walk: lane L covers the eight (half, segment) groups at element
// half*128 + segment*32 + L. The 6-bit code is the ql nibble (low for
// segments 0/1, high for 2/3) plus two top bits from qh[half*32+lane]; the
// signed value subtracts 32 and scales by the int8 sub-block scale at
// half*8 + lane/16 + segment*2, matching the oracle's Q6Value.
__global__ void GemvQ6K(const Q6KBlock* __restrict__ w,
                        const float* __restrict__ x, float* __restrict__ out,
                        std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 256;
  const Q6KBlock* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const Q6KBlock& blk = row[b];
    const float d = __half2float(blk.d);
    const float* xrow = x + b * 256 + lane;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const std::uint8_t qhb = blk.qh[half * 32 + lane];
      const std::uint8_t q0 = blk.ql[half * 64 + lane];
      const std::uint8_t q1 = blk.ql[half * 64 + 32 + lane];
#pragma unroll
      for (int seg = 0; seg < 4; ++seg) {
        const std::uint8_t byte = (seg & 1) ? q1 : q0;
        const int low = (seg < 2) ? (byte & 0xF) : (byte >> 4);
        const int high = (qhb >> (2 * seg)) & 0x3;
        const auto quant = static_cast<std::int8_t>((high << 4) | low) - 32;
        const int scale = blk.scales[half * 8 + (lane >> 4) + seg * 2];
        acc += d * static_cast<float>(scale) * static_cast<float>(quant) *
               xrow[128 * half + 32 * seg];
      }
    }
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

// F32 output row dot for the router logits and the F32 stack weights. One
// wave owns one output row in the same strided chain as the CPU contract;
// four waves share a dispatch, mirroring the quantized Gemv tiers, so the
// router's thousands of tiny rows cost fewer blocks to schedule.
__global__ void GemvF32(const float* __restrict__ w,
                        const float* __restrict__ x, float* __restrict__ out,
                        std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const float* __restrict__ row = w + static_cast<std::size_t>(r) * cols;
  const std::uint32_t lane = threadIdx.x & 31U;
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

// Binary16 clone of GemvF32 for the upload-retyped router: the same strided
// per-lane chain (weights widened with the exact half -> float conversion),
// so the logits differ from the F32 tier only by the stored weight rounding.
__global__ void GemvF16(const __half* __restrict__ w,
                        const float* __restrict__ x, float* __restrict__ out,
                        std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const __half* __restrict__ row = w + static_cast<std::size_t>(r) * cols;
  const std::uint32_t lane = threadIdx.x & 31U;
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

// Multi-row dense GEMV for the speculative verify pass: one wave owns an
// output row and dots the weight row against `Rows` activation rows in the
// same walk, so the weight matrix is read once for all tokens. Each per-row
// dot is bit-identical to the matching single-row kernel.
template<std::uint32_t Rows>
__global__ void GemvRowsQ4_0(const Q4_0Block* __restrict__ w,
                             const float* __restrict__ x,
                             std::uint32_t x_stride, float* __restrict__ out,
                             std::uint32_t out_stride, std::uint32_t rows,
                             std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 32;
  const Q4_0Block* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
#pragma unroll 4
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const float d = __half2float(row[b].d);
    const std::uint8_t packed = row[b].qs[lane & 15U];
    const int quant = (lane < 16U) ? (packed & 0xF) : (packed >> 4);
    const float wv = d * static_cast<float>(quant - 8);
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      acc[o] += wv * x[static_cast<std::size_t>(o) * x_stride + b * 32 + lane];
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

// Vec4 multi-row variant, mirroring the host-side kernel selection of the
// single-row dispatcher: Q8_0 launches the vectorized walk whenever
// cols % 128 == 0, so the verify pass must use the same per-row summation
// order or verify rows drift from sequential steps (1e-5 per token, enough
// to break the bit-parity contract on near-tie logits). Each row keeps the
// exact GemvQ8_0Vec4 chain; the weight row and dequantized codes are shared
// by the Rows activation rows.
template<std::uint32_t Rows>
__global__ void GemvRowsQ8_0Vec4(const Q8_0Block* __restrict__ w,
                                 const float* __restrict__ x,
                                 std::uint32_t x_stride,
                                 float* __restrict__ out,
                                 std::uint32_t out_stride, std::uint32_t rows,
                                 std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t groups = cols / 128U;
  const Q8_0Block* __restrict__ row =
      w + static_cast<std::size_t>(r) * (cols / 32U);
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
#pragma unroll 4
  for (std::uint32_t g = 0; g < groups; ++g) {
    const std::uint32_t blk = g * 4U + lane / 8U;
    const std::uint32_t word = (4U * lane) % 32U;
    std::int32_t packed;
    __builtin_memcpy(&packed, &row[blk].qs[word], 4);
    const float d = __half2float(row[blk].d);
    const std::uint32_t k = blk * 32U + word;
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      const float* __restrict__ xr = x + static_cast<std::size_t>(o) * x_stride;
      acc[o] += d *
                static_cast<float>(static_cast<std::int8_t>(packed & 0xFF)) *
                xr[k];
      acc[o] +=
          d *
          static_cast<float>(static_cast<std::int8_t>((packed >> 8) & 0xFF)) *
          xr[k + 1];
      acc[o] +=
          d *
          static_cast<float>(static_cast<std::int8_t>((packed >> 16) & 0xFF)) *
          xr[k + 2];
      acc[o] +=
          d *
          static_cast<float>(static_cast<std::int8_t>((packed >> 24) & 0xFF)) *
          xr[k + 3];
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

template<std::uint32_t Rows>
__global__ void GemvRowsQ4K(const Q4KBlock* __restrict__ w,
                            const float* __restrict__ x, std::uint32_t x_stride,
                            float* __restrict__ out, std::uint32_t out_stride,
                            std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 256;
  const Q4KBlock* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const Q4KBlock& blk = row[b];
    const float d = __half2float(blk.d);
    const float dmin = __half2float(blk.dmin);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint8_t packed = blk.qs[(j >> 1) * 32 + lane];
      const int quant = (j & 1) ? (packed >> 4) : (packed & 0xF);
      int sc = 0;
      int mn = 0;
      Q4ScaleMin(j, blk.scales, sc, mn);
      const float wv = d * static_cast<float>(sc) * static_cast<float>(quant) -
                       dmin * static_cast<float>(mn);
#pragma unroll
      for (std::uint32_t o = 0; o < Rows; ++o) {
        acc[o] +=
            wv *
            x[static_cast<std::size_t>(o) * x_stride + b * 256 + 32 * j + lane];
      }
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

template<std::uint32_t Rows>
__global__ void GemvRowsQ5K(const Q5KBlock* __restrict__ w,
                            const float* __restrict__ x, std::uint32_t x_stride,
                            float* __restrict__ out, std::uint32_t out_stride,
                            std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 256;
  const Q5KBlock* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const Q5KBlock& blk = row[b];
    const float d = __half2float(blk.d);
    const float dmin = __half2float(blk.dmin);
    const std::uint8_t qhb = blk.qh[lane];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint8_t qb = blk.qs[(j >> 1) * 32 + lane];
      const int quant4 = (j & 1) ? (qb >> 4) : (qb & 0xF);
      const int quant = quant4 + (((qhb >> j) & 1) ? 16 : 0);
      int sc = 0;
      int mn = 0;
      Q4ScaleMin(j, blk.scales, sc, mn);
      const float wv = d * static_cast<float>(sc) * static_cast<float>(quant) -
                       dmin * static_cast<float>(mn);
#pragma unroll
      for (std::uint32_t o = 0; o < Rows; ++o) {
        acc[o] +=
            wv *
            x[static_cast<std::size_t>(o) * x_stride + b * 256 + 32 * j + lane];
      }
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

template<std::uint32_t Rows>
__global__ void GemvRowsQ6K(const Q6KBlock* __restrict__ w,
                            const float* __restrict__ x, std::uint32_t x_stride,
                            float* __restrict__ out, std::uint32_t out_stride,
                            std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 256;
  const Q6KBlock* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const Q6KBlock& blk = row[b];
    const float d = __half2float(blk.d);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const std::uint8_t qhb = blk.qh[half * 32 + lane];
      const std::uint8_t q0 = blk.ql[half * 64 + lane];
      const std::uint8_t q1 = blk.ql[half * 64 + 32 + lane];
#pragma unroll
      for (int seg = 0; seg < 4; ++seg) {
        const std::uint8_t byte = (seg & 1) ? q1 : q0;
        const int low = (seg < 2) ? (byte & 0xF) : (byte >> 4);
        const int high = (qhb >> (2 * seg)) & 0x3;
        const auto quant = static_cast<std::int8_t>((high << 4) | low) - 32;
        const int scale = blk.scales[half * 8 + (lane >> 4) + seg * 2];
        const float wv =
            d * static_cast<float>(scale) * static_cast<float>(quant);
#pragma unroll
        for (std::uint32_t o = 0; o < Rows; ++o) {
          acc[o] += wv * x[static_cast<std::size_t>(o) * x_stride + b * 256 +
                           128 * half + 32 * seg + lane];
        }
      }
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

// Multi-row dense GEMV for the F32 stacks: one warp owns an output row and
// dots it against `Rows` activation rows in the same walk. Each per-row dot
// is bit-identical to GemvF32.
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

// Grouped expert GEMV: a block of four waves folds four consecutive
// (slot, row-group) pairs of the selected-expert product. The expert id is
// read from device memory, so the whole routed MoE stays asynchronous. A
// wave walks `Rows` consecutive output rows of one slot together (Rows=1 is
// the old single (slot, row) pairing). Each row keeps the exact block walk
// of the single-pair kernel — same block order, same quant unpack, same x
// elements — so results are bit-identical while two weight streams
// interleave and the group costs half the blocks to schedule.
template<std::uint32_t Rows>
__global__ void GemvGroupedQ4_0(const Q4_0Block* __restrict__ w,
                                const std::int32_t* __restrict__ ids,
                                std::size_t expert_stride_blocks,
                                const float* __restrict__ x,
                                std::uint32_t slots_per_row,
                                std::uint32_t x_stride, float* __restrict__ out,
                                std::uint32_t used, std::uint32_t rows,
                                std::uint32_t cols) {
  static_assert(Rows == 1U || Rows == 2U, "one or two rows per wave");
  const std::uint32_t rows_per_slot = rows / Rows;
  const std::uint32_t g = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (g >= used * rows_per_slot) {
    return;
  }
  const std::uint32_t s = g / rows_per_slot;
  const std::uint32_t r = (g - s * rows_per_slot) * Rows;
  const std::uint32_t nblocks = cols / 32;
  const Q4_0Block* __restrict__ row =
      w + static_cast<std::size_t>(ids[s]) * expert_stride_blocks +
      static_cast<std::size_t>(r) * nblocks;
  const float* __restrict__ xs =
      x + static_cast<std::size_t>(s / slots_per_row) * x_stride;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
#pragma unroll 4
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const float xv = xs[b * 32 + lane];
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      const std::uint8_t packed = row[o * nblocks + b].qs[lane & 15U];
      const int quant = (lane < 16U) ? (packed & 0xF) : (packed >> 4);
      acc[o] += __half2float(row[o * nblocks + b].d) *
                static_cast<float>(quant - 8) * xv;
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[s * rows + r + o] = acc[o];
    }
  }
}

template<std::uint32_t Rows>
__global__ void GemvGroupedQ8_0(const Q8_0Block* __restrict__ w,
                                const std::int32_t* __restrict__ ids,
                                std::size_t expert_stride_blocks,
                                const float* __restrict__ x,
                                std::uint32_t slots_per_row,
                                std::uint32_t x_stride, float* __restrict__ out,
                                std::uint32_t used, std::uint32_t rows,
                                std::uint32_t cols) {
  static_assert(Rows == 1U || Rows == 2U, "one or two rows per wave");
  const std::uint32_t rows_per_slot = rows / Rows;
  const std::uint32_t g = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (g >= used * rows_per_slot) {
    return;
  }
  const std::uint32_t s = g / rows_per_slot;
  const std::uint32_t r = (g - s * rows_per_slot) * Rows;
  const std::uint32_t nblocks = cols / 32;
  const Q8_0Block* __restrict__ row =
      w + static_cast<std::size_t>(ids[s]) * expert_stride_blocks +
      static_cast<std::size_t>(r) * nblocks;
  const float* __restrict__ xs =
      x + static_cast<std::size_t>(s / slots_per_row) * x_stride;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
#pragma unroll 4
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const float xv = xs[b * 32 + lane];
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      acc[o] += __half2float(row[o * nblocks + b].d) *
                static_cast<float>(row[o * nblocks + b].qs[lane]) * xv;
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[s * rows + r + o] = acc[o];
    }
  }
}

// Grouped expert GEMV for the Q4_K routed gate/up stacks. A wave walks
// `Rows` consecutive output rows of one slot together (Rows=1 is the old
// single (slot, row) pairing). Each row keeps the exact super-block/j
// chain of the single-pair walk — same block order, same quant unpack,
// same x elements — so results are bit-identical while two weight streams
// interleave and the group costs half the blocks to schedule.
template<std::uint32_t Rows>
__global__ void GemvGroupedQ4K(const Q4KBlock* __restrict__ w,
                               const std::int32_t* __restrict__ ids,
                               std::size_t expert_stride_blocks,
                               const float* __restrict__ x,
                               std::uint32_t slots_per_row,
                               std::uint32_t x_stride, float* __restrict__ out,
                               std::uint32_t used, std::uint32_t rows,
                               std::uint32_t cols) {
  static_assert(Rows == 1U || Rows == 2U, "one or two rows per wave");
  const std::uint32_t rows_per_slot = rows / Rows;
  const std::uint32_t g = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (g >= used * rows_per_slot) {
    return;
  }
  const std::uint32_t s = g / rows_per_slot;
  const std::uint32_t r = (g - s * rows_per_slot) * Rows;
  const std::uint32_t nblocks = cols / 256;
  const Q4KBlock* __restrict__ row =
      w + static_cast<std::size_t>(ids[s]) * expert_stride_blocks +
      static_cast<std::size_t>(r) * nblocks;
  const float* __restrict__ xs =
      x + static_cast<std::size_t>(s / slots_per_row) * x_stride;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const float* __restrict__ xb = xs + b * 256 + lane;
    float xv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      xv[j] = xb[32 * j];
    }
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      const Q4KBlock& blk = row[o * nblocks + b];
      const float d = __half2float(blk.d);
      const float dmin = __half2float(blk.dmin);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint8_t packed = blk.qs[(j >> 1) * 32 + lane];
        const int quant = (j & 1) ? (packed >> 4) : (packed & 0xF);
        int sc = 0;
        int mn = 0;
        Q4ScaleMin(j, blk.scales, sc, mn);
        acc[o] += (d * static_cast<float>(sc) * static_cast<float>(quant) -
                   dmin * static_cast<float>(mn)) *
                  xv[j];
      }
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[s * rows + r + o] = acc[o];
    }
  }
}

// Grouped expert GEMV for the Q6_K routed down stacks; same templated
// consecutive-row walk and per-row bit-exactness as GemvGroupedQ4K.
template<std::uint32_t Rows>
__global__ void GemvGroupedQ6K(const Q6KBlock* __restrict__ w,
                               const std::int32_t* __restrict__ ids,
                               std::size_t expert_stride_blocks,
                               const float* __restrict__ x,
                               std::uint32_t slots_per_row,
                               std::uint32_t x_stride, float* __restrict__ out,
                               std::uint32_t used, std::uint32_t rows,
                               std::uint32_t cols) {
  static_assert(Rows == 1U || Rows == 2U, "one or two rows per wave");
  const std::uint32_t rows_per_slot = rows / Rows;
  const std::uint32_t g = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (g >= used * rows_per_slot) {
    return;
  }
  const std::uint32_t s = g / rows_per_slot;
  const std::uint32_t r = (g - s * rows_per_slot) * Rows;
  const std::uint32_t nblocks = cols / 256;
  const Q6KBlock* __restrict__ row =
      w + static_cast<std::size_t>(ids[s]) * expert_stride_blocks +
      static_cast<std::size_t>(r) * nblocks;
  const float* __restrict__ xs =
      x + static_cast<std::size_t>(s / slots_per_row) * x_stride;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc[Rows] = {};
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const float* __restrict__ xb = xs + b * 256 + lane;
    float xv[8];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int seg = 0; seg < 4; ++seg) {
        xv[4 * half + seg] = xb[128 * half + 32 * seg];
      }
    }
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      const Q6KBlock& blk = row[o * nblocks + b];
      const float d = __half2float(blk.d);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const std::uint8_t qhb = blk.qh[half * 32 + lane];
        const std::uint8_t q0 = blk.ql[half * 64 + lane];
        const std::uint8_t q1 = blk.ql[half * 64 + 32 + lane];
#pragma unroll
        for (int seg = 0; seg < 4; ++seg) {
          const std::uint8_t byte = (seg & 1) ? q1 : q0;
          const int low = (seg < 2) ? (byte & 0xF) : (byte >> 4);
          const int high = (qhb >> (2 * seg)) & 0x3;
          const auto quant = static_cast<std::int8_t>((high << 4) | low) - 32;
          const int scale = blk.scales[half * 8 + (lane >> 4) + seg * 2];
          acc[o] += d * static_cast<float>(scale) * static_cast<float>(quant) *
                    xv[4 * half + seg];
        }
      }
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    acc[o] = WarpReduceSum(acc[o]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      out[s * rows + r + o] = acc[o];
    }
  }
}

// Grouped expert GEMV for the dense (F32) expert stacks: same (slot, row)
// pairing as GemvGroupedQ8_0, element-strided within the row.
template<typename T>
__global__ void GemvGroupedDense(const T* __restrict__ w,
                                 const std::int32_t* __restrict__ ids,
                                 std::size_t expert_stride_elems,
                                 const float* __restrict__ x,
                                 std::uint32_t slots_per_row,
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
  const float* __restrict__ xs =
      x + static_cast<std::size_t>(s / slots_per_row) * x_stride;
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

// Dequantize a single row of a quantized matrix (the token-embedding lookup).
__global__ void EmbedRowQ4_0(const Q4_0Block* __restrict__ w,
                             float* __restrict__ out) {
  const std::uint32_t b = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const float d = __half2float(w[b].d);
  const std::uint8_t packed = w[b].qs[lane & 15U];
  const int quant = (lane < 16U) ? (packed & 0xF) : (packed >> 4);
  out[b * 32 + lane] = d * static_cast<float>(quant - 8);
}

__global__ void EmbedRowQ8_0(const Q8_0Block* __restrict__ w,
                             float* __restrict__ out) {
  const std::uint32_t b = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const float d = __half2float(w[b].d);
  out[b * 32 + lane] = d * static_cast<float>(w[b].qs[lane]);
}

// One warp dequantizes one Q4_K super-block: lane L writes the element at
// offset L of each of the eight 32-element groups.
__global__ void EmbedRowQ4K(const Q4KBlock* __restrict__ w,
                            float* __restrict__ out) {
  const std::uint32_t b = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const float d = __half2float(w[b].d);
  const float dmin = __half2float(w[b].dmin);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const std::uint8_t packed = w[b].qs[(j >> 1) * 32 + lane];
    const int quant = (j & 1) ? (packed >> 4) : (packed & 0xF);
    int sc = 0;
    int mn = 0;
    Q4ScaleMin(j, w[b].scales, sc, mn);
    out[b * 256 + j * 32 + lane] =
        d * static_cast<float>(sc) * static_cast<float>(quant) -
        dmin * static_cast<float>(mn);
  }
}

__global__ void EmbedRowQ5K(const Q5KBlock* __restrict__ w,
                            float* __restrict__ out) {
  const std::uint32_t b = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const float d = __half2float(w[b].d);
  const float dmin = __half2float(w[b].dmin);
  const std::uint8_t qhb = w[b].qh[lane];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const std::uint8_t qb = w[b].qs[(j >> 1) * 32 + lane];
    const int quant4 = (j & 1) ? (qb >> 4) : (qb & 0xF);
    const int quant = quant4 + (((qhb >> j) & 1) ? 16 : 0);
    int sc = 0;
    int mn = 0;
    Q4ScaleMin(j, w[b].scales, sc, mn);
    out[b * 256 + j * 32 + lane] =
        d * static_cast<float>(sc) * static_cast<float>(quant) -
        dmin * static_cast<float>(mn);
  }
}

__global__ void EmbedRowQ6K(const Q6KBlock* __restrict__ w,
                            float* __restrict__ out) {
  const std::uint32_t b = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const float d = __half2float(w[b].d);
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const std::uint8_t qhb = w[b].qh[half * 32 + lane];
    const std::uint8_t q0 = w[b].ql[half * 64 + lane];
    const std::uint8_t q1 = w[b].ql[half * 64 + 32 + lane];
#pragma unroll
    for (int seg = 0; seg < 4; ++seg) {
      const std::uint8_t byte = (seg & 1) ? q1 : q0;
      const int low = (seg < 2) ? (byte & 0xF) : (byte >> 4);
      const int high = (qhb >> (2 * seg)) & 0x3;
      const auto quant = static_cast<std::int8_t>((high << 4) | low) - 32;
      const int scale = w[b].scales[half * 8 + (lane >> 4) + seg * 2];
      out[b * 256 + half * 128 + seg * 32 + lane] =
          d * static_cast<float>(scale) * static_cast<float>(quant);
    }
  }
}

__global__ void EmbedRowF32(const float* __restrict__ w,
                            float* __restrict__ out, std::uint32_t cols) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < cols) {
    out[idx] = w[idx];
  }
}

__global__ void EmbedRowF16(const __half* __restrict__ w,
                            float* __restrict__ out, std::uint32_t cols) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < cols) {
    out[idx] = static_cast<float>(w[idx]);
  }
}

// Out-of-place F32 -> binary16 retype (round to nearest even) for the
// upload-time router conversion; 4 bytes shrink to 2 so it cannot be in
// place. Overflow (inf/NaN in the half result) is counted on the device and
// read back by the uploader, which fails the load on any hit.
__global__ void F32ToHalfKernel(const float* __restrict__ src,
                                __half* __restrict__ dst, std::size_t count,
                                std::uint32_t* __restrict__ overflow) {
  std::uint32_t outside = 0;
  for (std::size_t i =
           blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const __half h = __float2half_rn(src[i]);
    outside += __hisinf(h) != 0 || __hisnan(h) ? 1U : 0U;
    dst[i] = h;
  }
  if (outside != 0) {
    atomicAdd(overflow, outside);
  }
}

}  // namespace

void ConvertF32ToHalf(const float* src, __half* dst, std::size_t count,
                      std::uint32_t* overflow, hipStream_t stream) {
  constexpr int kThreads = 256;
  if (count == 0) {
    return;
  }
  const std::size_t blocks = (count + kThreads - 1) / kThreads < 4096
                                 ? (count + kThreads - 1) / kThreads
                                 : 4096;
  F32ToHalfKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      src, dst, count, overflow);
}

void EmbedRow(const void* base, GemvType type, std::uint32_t row,
              std::uint32_t cols, float* out, hipStream_t stream) {
  switch (type) {
    case GemvType::kQ4_0: {
      const std::uint32_t nblocks = cols / 32;
      const Q4_0Block* w = static_cast<const Q4_0Block*>(base) +
                           static_cast<std::size_t>(row) * nblocks;
      EmbedRowQ4_0<<<nblocks, 32, 0, stream>>>(w, out);
      break;
    }
    case GemvType::kQ8_0: {
      const std::uint32_t nblocks = cols / 32;
      const Q8_0Block* w = static_cast<const Q8_0Block*>(base) +
                           static_cast<std::size_t>(row) * nblocks;
      EmbedRowQ8_0<<<nblocks, 32, 0, stream>>>(w, out);
      break;
    }
    case GemvType::kQ4_K: {
      const std::uint32_t nblocks = cols / 256;
      const Q4KBlock* w = static_cast<const Q4KBlock*>(base) +
                          static_cast<std::size_t>(row) * nblocks;
      EmbedRowQ4K<<<nblocks, 32, 0, stream>>>(w, out);
      break;
    }
    case GemvType::kQ5_K: {
      const std::uint32_t nblocks = cols / 256;
      const Q5KBlock* w = static_cast<const Q5KBlock*>(base) +
                          static_cast<std::size_t>(row) * nblocks;
      EmbedRowQ5K<<<nblocks, 32, 0, stream>>>(w, out);
      break;
    }
    case GemvType::kQ6_K: {
      const std::uint32_t nblocks = cols / 256;
      const Q6KBlock* w = static_cast<const Q6KBlock*>(base) +
                          static_cast<std::size_t>(row) * nblocks;
      EmbedRowQ6K<<<nblocks, 32, 0, stream>>>(w, out);
      break;
    }
    case GemvType::kF32: {
      const float* w = static_cast<const float*>(base) +
                       static_cast<std::size_t>(row) * cols;
      const std::size_t block = 256;
      const std::size_t grid = (cols + block - 1) / block < 65535
                                   ? (cols + block - 1) / block
                                   : 65535;
      EmbedRowF32<<<grid, block, 0, stream>>>(w, out, cols);
      break;
    }
    case GemvType::kF16: {
      const __half* w = static_cast<const __half*>(base) +
                        static_cast<std::size_t>(row) * cols;
      const std::size_t block = 256;
      const std::size_t grid = (cols + block - 1) / block < 65535
                                   ? (cols + block - 1) / block
                                   : 65535;
      EmbedRowF16<<<grid, block, 0, stream>>>(w, out, cols);
      break;
    }
  }
}

void Gemv(const void* base, GemvType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          hipStream_t stream) {
  (void)row_bytes;
  switch (type) {
    case GemvType::kQ4_0:
      GemvQ4_0<<<(rows + 3U) / 4U, dim3(128), 0, stream>>>(
          static_cast<const Q4_0Block*>(base), x, out, rows, cols);
      break;
    case GemvType::kQ8_0:
      if (cols % 128U == 0U) {
        GemvQ8_0Vec4<<<(rows + 3U) / 4U, dim3(128), 0, stream>>>(
            static_cast<const Q8_0Block*>(base), x, out, rows, cols);
      } else {
        GemvQ8_0<<<(rows + 3U) / 4U, dim3(128), 0, stream>>>(
            static_cast<const Q8_0Block*>(base), x, out, rows, cols);
      }
      break;
    case GemvType::kQ4_K:
      GemvQ4K<<<(rows + 3U) / 4U, dim3(128), 0, stream>>>(
          static_cast<const Q4KBlock*>(base), x, out, rows, cols);
      break;
    case GemvType::kQ5_K:
      GemvQ5K<<<(rows + 3U) / 4U, dim3(128), 0, stream>>>(
          static_cast<const Q5KBlock*>(base), x, out, rows, cols);
      break;
    case GemvType::kQ6_K:
      GemvQ6K<<<(rows + 3U) / 4U, dim3(128), 0, stream>>>(
          static_cast<const Q6KBlock*>(base), x, out, rows, cols);
      break;
    case GemvType::kF32:
      GemvF32<<<(rows + 3U) / 4U, dim3(128), 0, stream>>>(
          static_cast<const float*>(base), x, out, rows, cols);
      break;
    case GemvType::kF16:
      GemvF16<<<(rows + 3U) / 4U, dim3(128), 0, stream>>>(
          static_cast<const __half*>(base), x, out, rows, cols);
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
      case GemvType::kQ4_0:                                                    \
        GemvRowsQ4_0<R><<<grid, dim3(128), 0, stream>>>(                       \
            static_cast<const Q4_0Block*>(base), x, x_stride, out, out_stride, \
            rows, cols);                                                       \
        break;                                                                 \
      case GemvType::kQ8_0:                                                    \
        if (cols % 128U == 0U) {                                               \
          GemvRowsQ8_0Vec4<R><<<grid, dim3(128), 0, stream>>>(                 \
              static_cast<const Q8_0Block*>(base), x, x_stride, out,           \
              out_stride, rows, cols);                                         \
        } else {                                                               \
          GemvRowsQ8_0<R><<<grid, dim3(128), 0, stream>>>(                     \
              static_cast<const Q8_0Block*>(base), x, x_stride, out,           \
              out_stride, rows, cols);                                         \
        }                                                                      \
        break;                                                                 \
      case GemvType::kQ4_K:                                                    \
        GemvRowsQ4K<R><<<grid, dim3(128), 0, stream>>>(                        \
            static_cast<const Q4KBlock*>(base), x, x_stride, out, out_stride,  \
            rows, cols);                                                       \
        break;                                                                 \
      case GemvType::kQ5_K:                                                    \
        GemvRowsQ5K<R><<<grid, dim3(128), 0, stream>>>(                        \
            static_cast<const Q5KBlock*>(base), x, x_stride, out, out_stride,  \
            rows, cols);                                                       \
        break;                                                                 \
      case GemvType::kQ6_K:                                                    \
        GemvRowsQ6K<R><<<grid, dim3(128), 0, stream>>>(                        \
            static_cast<const Q6KBlock*>(base), x, x_stride, out, out_stride,  \
            rows, cols);                                                       \
        break;                                                                 \
      case GemvType::kF32:                                                     \
        GemvRowsDense<float, R><<<rows, dim3(32), 0, stream>>>(                \
            static_cast<const float*>(base), x, x_stride, out, out_stride,     \
            rows, cols);                                                       \
        break;                                                                 \
      case GemvType::kF16:                                                     \
        GemvRowsDense<__half, R><<<rows, dim3(32), 0, stream>>>(               \
            static_cast<const __half*>(base), x, x_stride, out, out_stride,    \
            rows, cols);                                                       \
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
                 std::uint32_t slots_per_row, std::uint32_t rows,
                 std::uint32_t cols, const float* x, std::uint32_t x_stride,
                 float* out, hipStream_t stream) {
  const std::size_t pairs = static_cast<std::size_t>(used) * rows;
  const std::uint32_t grid = static_cast<std::uint32_t>((pairs + 3U) / 4U);
  switch (type) {
    case GemvType::kQ4_0:
      if (rows % 2U == 0U) {
        GemvGroupedQ4_0<2U><<<(pairs / 2U + 3U) / 4U, dim3(128), 0, stream>>>(
            static_cast<const Q4_0Block*>(base), ids,
            expert_stride / sizeof(Q4_0Block), x, slots_per_row, x_stride, out,
            used, rows, cols);
      } else {
        GemvGroupedQ4_0<1U><<<grid, dim3(128), 0, stream>>>(
            static_cast<const Q4_0Block*>(base), ids,
            expert_stride / sizeof(Q4_0Block), x, slots_per_row, x_stride, out,
            used, rows, cols);
      }
      break;
    case GemvType::kQ8_0:
      if (rows % 2U == 0U) {
        GemvGroupedQ8_0<2U><<<(pairs / 2U + 3U) / 4U, dim3(128), 0, stream>>>(
            static_cast<const Q8_0Block*>(base), ids,
            expert_stride / sizeof(Q8_0Block), x, slots_per_row, x_stride, out,
            used, rows, cols);
      } else {
        GemvGroupedQ8_0<1U><<<grid, dim3(128), 0, stream>>>(
            static_cast<const Q8_0Block*>(base), ids,
            expert_stride / sizeof(Q8_0Block), x, slots_per_row, x_stride, out,
            used, rows, cols);
      }
      break;
    case GemvType::kQ4_K:
      // Two rows per wave whenever the row count allows: the consecutive
      // pair shares one slot and halves the scheduled blocks; every row
      // keeps its single-pair chain, so the fold is bit-exact.
      if (rows % 2U == 0U) {
        GemvGroupedQ4K<2U><<<(pairs / 2U + 3U) / 4U, dim3(128), 0, stream>>>(
            static_cast<const Q4KBlock*>(base), ids,
            expert_stride / sizeof(Q4KBlock), x, slots_per_row, x_stride, out,
            used, rows, cols);
      } else {
        GemvGroupedQ4K<1U><<<grid, dim3(128), 0, stream>>>(
            static_cast<const Q4KBlock*>(base), ids,
            expert_stride / sizeof(Q4KBlock), x, slots_per_row, x_stride, out,
            used, rows, cols);
      }
      break;
    case GemvType::kQ6_K:
      if (rows % 2U == 0U) {
        GemvGroupedQ6K<2U><<<(pairs / 2U + 3U) / 4U, dim3(128), 0, stream>>>(
            static_cast<const Q6KBlock*>(base), ids,
            expert_stride / sizeof(Q6KBlock), x, slots_per_row, x_stride, out,
            used, rows, cols);
      } else {
        GemvGroupedQ6K<1U><<<grid, dim3(128), 0, stream>>>(
            static_cast<const Q6KBlock*>(base), ids,
            expert_stride / sizeof(Q6KBlock), x, slots_per_row, x_stride, out,
            used, rows, cols);
      }
      break;
    case GemvType::kF32:
      GemvGroupedDense<float><<<grid, dim3(128), 0, stream>>>(
          static_cast<const float*>(base), ids, expert_stride / sizeof(float),
          x, slots_per_row, x_stride, out, used, rows, cols);
      break;
    case GemvType::kF16:
      GemvGroupedDense<__half><<<grid, dim3(128), 0, stream>>>(
          static_cast<const __half*>(base), ids, expert_stride / sizeof(__half),
          x, slots_per_row, x_stride, out, used, rows, cols);
      break;
    // Q5_K only backs the dense attention-output projections; the routed
    // stacks are Q4_K/Q6_K (trunk) and Q4_0 (MTP).
    case GemvType::kQ5_K:
      break;
  }
}

}  // namespace gufo::models::ling3_flash::rocm