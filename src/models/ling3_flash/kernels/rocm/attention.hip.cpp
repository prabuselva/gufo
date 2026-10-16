#include "src/models/ling3_flash/kernels/rocm/attention.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <limits>

namespace gufo::models::ling3_flash::rocm {
namespace {

constexpr std::uint32_t kMaxTile = 1024U;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

__device__ __forceinline__ float WarpMaxFloat(float value) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    value = fmaxf(value, __shfl_down_sync(~0ULL, value, off));
  }
  return value;
}

__device__ __forceinline__ float WarpSumFloat(float value) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    value += __shfl_down_sync(~0ULL, value, off);
  }
  return value;
}

__device__ __forceinline__ double WarpSumDouble(double value) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    value += __shfl_xor_sync(~0ULL, value, off);
  }
  return value;
}

__device__ __forceinline__ float BlockMaxFloat(float value, float* red) {
  const float local = WarpMaxFloat(value);
  if ((threadIdx.x & 31U) == 0U) {
    red[threadIdx.x >> 5] = local;
  }
  __syncthreads();
  if (threadIdx.x < 32U) {
    const float v =
        (threadIdx.x < (blockDim.x >> 5)) ? red[threadIdx.x] : kNegInf;
    const float m = WarpMaxFloat(v);
    if (threadIdx.x == 0U) {
      red[0] = m;
    }
  }
  __syncthreads();
  return red[0];
}

__device__ __forceinline__ float BlockSumFloat(float value, float* red) {
  const float local = WarpSumFloat(value);
  if ((threadIdx.x & 31U) == 0U) {
    red[threadIdx.x >> 5] = local;
  }
  __syncthreads();
  if (threadIdx.x < 32U) {
    const float v = (threadIdx.x < (blockDim.x >> 5)) ? red[threadIdx.x] : 0.0F;
    const float s = WarpSumFloat(v);
    if (threadIdx.x == 0U) {
      red[0] = s;
    }
  }
  __syncthreads();
  return red[0];
}

__global__ void MlaDecodePartialKernel(
    const float* __restrict__ q_abs, const float* __restrict__ kv,
    const float* __restrict__ kpe, std::uint32_t n_pos, std::uint32_t tile,
    float scale, float* __restrict__ workspace, std::uint32_t heads,
    std::uint32_t kv_rank, std::uint32_t rotary) {
  __shared__ float s_q[640];
  __shared__ float s_scores[kMaxTile];
  __shared__ float s_red[32];
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t tile_idx = blockIdx.x;
  const std::uint32_t tile_begin = tile_idx * tile;
  const std::uint32_t tile_len = min(tile, n_pos - tile_begin);
  const std::uint32_t q_dim = kv_rank + rotary;
  for (std::uint32_t i = threadIdx.x; i < q_dim; i += blockDim.x) {
    s_q[i] = q_abs[static_cast<std::size_t>(h) * q_dim + i];
  }
  __syncthreads();

  // Pass A: scaled scores for this tile.
  for (std::uint32_t t = threadIdx.x; t < tile_len; t += blockDim.x) {
    const std::size_t row = static_cast<std::size_t>(tile_begin + t);
    double acc = 0.0;
    const float* kv_row = kv + row * kv_rank;
    for (std::uint32_t i = 0; i < kv_rank; ++i) {
      acc += static_cast<double>(s_q[i]) * kv_row[i];
    }
    const float* pe_row = kpe + row * rotary;
    for (std::uint32_t i = 0; i < rotary; ++i) {
      acc += static_cast<double>(s_q[kv_rank + i]) * pe_row[i];
    }
    s_scores[t] = static_cast<float>(acc * scale);
  }
  __syncthreads();

  // Tile max and sum of exponentials.
  float local_max = kNegInf;
  for (std::uint32_t t = threadIdx.x; t < tile_len; t += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[t]);
  }
  local_max = WarpMaxFloat(local_max);
  if ((threadIdx.x & 31U) == 0U) {
    s_red[threadIdx.x >> 5] = local_max;
  }
  __syncthreads();
  if (threadIdx.x < 32U) {
    const float v =
        (threadIdx.x < (blockDim.x >> 5)) ? s_red[threadIdx.x] : kNegInf;
    const float m = WarpMaxFloat(v);
    if (threadIdx.x == 0U) {
      s_red[0] = m;
    }
  }
  __syncthreads();
  const float tile_max = s_red[0];

  float local_sum = 0.0F;
  for (std::uint32_t t = threadIdx.x; t < tile_len; t += blockDim.x) {
    local_sum += ::expf(s_scores[t] - tile_max);
  }
  local_sum = WarpSumFloat(local_sum);
  __syncthreads();
  if ((threadIdx.x & 31U) == 0U) {
    s_red[threadIdx.x >> 5] = local_sum;
  }
  __syncthreads();
  if (threadIdx.x < 32U) {
    const float v =
        (threadIdx.x < (blockDim.x >> 5)) ? s_red[threadIdx.x] : 0.0F;
    const float s = WarpSumFloat(v);
    if (threadIdx.x == 0U) {
      s_red[0] = s;
    }
  }
  __syncthreads();

  // Pass B: unnormalized latent sum for this tile.
  float* out = workspace + (static_cast<std::size_t>(tile_idx) * heads + h) *
                               (kv_rank + 2U);
  for (std::uint32_t r = threadIdx.x; r < kv_rank; r += blockDim.x) {
    double acc = 0.0;
    for (std::uint32_t t = 0; t < tile_len; ++t) {
      acc += static_cast<double>(::expf(s_scores[t] - tile_max)) *
             kv[static_cast<std::size_t>(tile_begin + t) * kv_rank + r];
    }
    out[2U + r] = static_cast<float>(acc);
  }
  if (threadIdx.x == 0U) {
    out[0] = tile_max;
    out[1] = s_red[0];
  }
}

// Fast decode path: one block scores a four-head group over one position
// tile. Stages rows in short chunks with coalesced float4 copies, scores one
// (row, head) pair per warp from shared memory and reuses the staged rows,
// with their per-head exponentials, for the latent mix. Every reduced sum
// keeps at most a chunk-length run of fp32 fused products before widening to
// fp64 (RDNA vector fp64 is far slower), which stays well inside the fp64
// oracle tolerance. The scalar kernel above remains the fallback for shapes
// that are not float4-shaped or use tiles longer than kDecodeMaxTile.
constexpr std::uint32_t kDecodeGroup = 4U;
constexpr std::uint32_t kDecodeRows = 8U;
constexpr std::uint32_t kDecodeRowStride = 640U;
constexpr std::uint32_t kDecodeMaxTile = 256U;

__device__ __forceinline__ void StageDecodeRows(
    float* s_base, std::uint32_t rows, const float* __restrict__ kv,
    const float* __restrict__ kpe, std::size_t first_row, std::uint32_t kv_rank,
    std::uint32_t rotary) {
  const std::uint32_t kv4 = kv_rank >> 2U;
  const std::uint32_t half4 = kv4 + (rotary >> 2U);
  for (std::uint32_t p = threadIdx.x; p < rows * half4; p += blockDim.x) {
    const std::uint32_t t = p / half4;
    const std::uint32_t c = p - t * half4;
    const std::size_t row = first_row + t;
    const float4 v =
        (c < kv4)
            ? reinterpret_cast<const float4*>(kv + row * kv_rank)[c]
            : reinterpret_cast<const float4*>(kpe + row * rotary)[c - kv4];
    reinterpret_cast<float4*>(s_base + static_cast<std::size_t>(t) *
                                           kDecodeRowStride)[c] = v;
  }
}

__global__ void MlaDecodePartialGroupKernel(
    const float* __restrict__ q_abs, const float* __restrict__ kv,
    const float* __restrict__ kpe, std::uint32_t n_pos, std::uint32_t tile,
    float scale, float* __restrict__ workspace, std::uint32_t heads,
    std::uint32_t kv_rank, std::uint32_t rotary) {
  __shared__ float s_row[kDecodeRows][kDecodeRowStride];
  __shared__ float s_scores[kDecodeGroup][kDecodeMaxTile];
  __shared__ float s_red[kDecodeGroup][32];
  __shared__ float s_meta[kDecodeGroup][2];
  const std::uint32_t row_dim = kv_rank + rotary;
  const std::uint32_t row4 = row_dim >> 2U;
  const std::uint32_t h0 = blockIdx.x * kDecodeGroup;
  const std::uint32_t tile_idx = blockIdx.y;
  const std::uint32_t tile_begin = tile_idx * tile;
  const std::uint32_t tile_len = min(tile, n_pos - tile_begin);
  const std::uint32_t tid = threadIdx.x;

  // Every lane keeps its stripe of the four head queries in registers so a
  // staged row costs one LDS float4 chain for the whole group. Row-wise
  // scoring with one warp repeated per head made Pass A LDS-issue-bound.
  constexpr std::uint32_t kRow4Max = kDecodeRowStride / 4U / 32U;  // 5
  const uint32_t lane = tid & 31U;
  float4 q_reg[kDecodeGroup][kRow4Max];
  {
    const float4* q4 = reinterpret_cast<const float4*>(
        q_abs + static_cast<std::size_t>(h0) * row_dim);
#pragma unroll
    for (std::uint32_t g = 0U; g < kDecodeGroup; ++g) {
#pragma unroll
      for (std::uint32_t it = 0U; it < kRow4Max; ++it) {
        const std::uint32_t c = lane + it * 32U;
        q_reg[g][it] = (c < row4) ? q4[static_cast<std::size_t>(g) * row4 + c]
                                  : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      }
    }
  }

  // Pass A: scaled scores for this tile. Each warp reads a row float4 once
  // and dots it against all group queries; per-score term order matches the
  // one-warp-per-head layout, so results are bit-identical.
  for (std::uint32_t base = 0; base < tile_len; base += kDecodeRows) {
    const std::uint32_t rows =
        tile_len - base < kDecodeRows ? tile_len - base : kDecodeRows;
    __syncthreads();
    StageDecodeRows(&s_row[0][0], rows, kv, kpe,
                    static_cast<std::size_t>(tile_begin) + base, kv_rank,
                    rotary);
    __syncthreads();
    for (std::uint32_t t = tid >> 5U; t < rows; t += blockDim.x >> 5U) {
      const float4* row = reinterpret_cast<const float4*>(&s_row[t][0]);
      float pacc[kDecodeGroup];
#pragma unroll
      for (std::uint32_t g = 0U; g < kDecodeGroup; ++g) {
        pacc[g] = 0.0F;
      }
#pragma unroll
      for (std::uint32_t it = 0U; it < kRow4Max; ++it) {
        const std::uint32_t c = lane + it * 32U;
        if (c < row4) {
          const float4 a = row[c];
#pragma unroll
          for (std::uint32_t g = 0U; g < kDecodeGroup; ++g) {
            const float4 b = q_reg[g][it];
            pacc[g] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
          }
        }
      }
#pragma unroll
      for (std::uint32_t g = 0U; g < kDecodeGroup; ++g) {
        const double acc = WarpSumDouble(static_cast<double>(pacc[g]));
        if (lane == 0U) {
          s_scores[g][base + t] = static_cast<float>(acc * scale);
        }
      }
    }
  }
  __syncthreads();

  // Per-head tile maximum, then replace the scores with their exponentials.
  for (std::uint32_t g = 0; g < kDecodeGroup; ++g) {
    float local_max = kNegInf;
    for (std::uint32_t t = tid; t < tile_len; t += blockDim.x) {
      local_max = fmaxf(local_max, s_scores[g][t]);
    }
    const float tile_max = BlockMaxFloat(local_max, s_red[g]);
    float local_sum = 0.0F;
    for (std::uint32_t t = tid; t < tile_len; t += blockDim.x) {
      const float e = ::expf(s_scores[g][t] - tile_max);
      s_scores[g][t] = e;
      local_sum += e;
    }
    const float tile_sum = BlockSumFloat(local_sum, s_red[g]);
    if (tid == 0U) {
      s_meta[g][0] = tile_max;
      s_meta[g][1] = tile_sum;
    }
  }

  // Pass B: reuse every staged row for the latent mix of all four heads in
  // the same ascending position order as the scalar kernel. Each chunk
  // accumulates fp32 products per latent dimension and widens into a
  // tile-long fp64 accumulator.
  double acc[2][kDecodeGroup];
  for (std::uint32_t i = 0; i < 2U; ++i) {
#pragma unroll
    for (std::uint32_t g = 0; g < kDecodeGroup; ++g) {
      acc[i][g] = 0.0;
    }
  }
  for (std::uint32_t base = 0; base < tile_len; base += kDecodeRows) {
    const std::uint32_t rows =
        tile_len - base < kDecodeRows ? tile_len - base : kDecodeRows;
    __syncthreads();
    StageDecodeRows(&s_row[0][0], rows, kv, kpe,
                    static_cast<std::size_t>(tile_begin) + base, kv_rank,
                    rotary);
    __syncthreads();
    for (std::uint32_t i = 0; i < 2U; ++i) {
      const std::uint32_t r = tid + i * blockDim.x;
      if (r >= kv_rank) {
        continue;
      }
      float pacc[kDecodeGroup];
#pragma unroll
      for (std::uint32_t g = 0; g < kDecodeGroup; ++g) {
        pacc[g] = 0.0F;
      }
      for (std::uint32_t t = 0; t < rows; ++t) {
        const float v = s_row[t][r];
#pragma unroll
        for (std::uint32_t g = 0; g < kDecodeGroup; ++g) {
          pacc[g] += s_scores[g][base + t] * v;
        }
      }
#pragma unroll
      for (std::uint32_t g = 0; g < kDecodeGroup; ++g) {
        acc[i][g] += static_cast<double>(pacc[g]);
      }
    }
  }
  __syncthreads();
  for (std::uint32_t i = 0; i < 2U; ++i) {
    const std::uint32_t r = tid + i * blockDim.x;
    if (r >= kv_rank) {
      continue;
    }
#pragma unroll
    for (std::uint32_t g = 0; g < kDecodeGroup; ++g) {
      workspace[(static_cast<std::size_t>(tile_idx) * heads + h0 + g) *
                    (kv_rank + 2U) +
                2U + r] = static_cast<float>(acc[i][g]);
    }
  }
  if (tid < kDecodeGroup) {
    float* out =
        workspace + (static_cast<std::size_t>(tile_idx) * heads + h0 + tid) *
                        (kv_rank + 2U);
    out[0] = s_meta[tid][0];
    out[1] = s_meta[tid][1];
  }
}

__global__ void MlaDecodeCombineKernel(const float* __restrict__ workspace,
                                       std::uint32_t tiles, std::uint32_t heads,
                                       std::uint32_t kv_rank,
                                       float* __restrict__ latent) {
  __shared__ float s_red[32];
  __shared__ float s_scale[2];  // [global max, inverse sum]
  const std::uint32_t h = blockIdx.x;
  const float* ws = workspace + static_cast<std::size_t>(h) * (kv_rank + 2U);
  const std::size_t head_stride =
      static_cast<std::size_t>(heads) * (kv_rank + 2U);

  float local_max = kNegInf;
  for (std::uint32_t t = threadIdx.x; t < tiles; t += blockDim.x) {
    local_max = fmaxf(local_max, ws[static_cast<std::size_t>(t) * head_stride]);
  }
  local_max = WarpMaxFloat(local_max);
  if ((threadIdx.x & 31U) == 0U) {
    s_red[threadIdx.x >> 5] = local_max;
  }
  __syncthreads();
  if (threadIdx.x < 32U) {
    const float v =
        (threadIdx.x < (blockDim.x >> 5)) ? s_red[threadIdx.x] : kNegInf;
    const float m = WarpMaxFloat(v);
    if (threadIdx.x == 0U) {
      s_red[0] = m;
    }
  }
  __syncthreads();
  const float max_all = s_red[0];

  float local_sum = 0.0F;
  for (std::uint32_t t = threadIdx.x; t < tiles; t += blockDim.x) {
    local_sum +=
        ::expf(ws[static_cast<std::size_t>(t) * head_stride] - max_all) *
        ws[static_cast<std::size_t>(t) * head_stride + 1U];
  }
  local_sum = WarpSumFloat(local_sum);
  __syncthreads();
  if ((threadIdx.x & 31U) == 0U) {
    s_red[threadIdx.x >> 5] = local_sum;
  }
  __syncthreads();
  if (threadIdx.x < 32U) {
    const float v =
        (threadIdx.x < (blockDim.x >> 5)) ? s_red[threadIdx.x] : 0.0F;
    const float s = WarpSumFloat(v);
    if (threadIdx.x == 0U) {
      s_scale[1] = 1.0F / s;
    }
  }
  if (threadIdx.x == 0U) {
    s_scale[0] = max_all;
  }
  __syncthreads();

  float* dst = latent + static_cast<std::size_t>(h) * kv_rank;
  for (std::uint32_t r = threadIdx.x; r < kv_rank; r += blockDim.x) {
    double acc = 0.0;
    for (std::uint32_t t = 0; t < tiles; ++t) {
      const float* part = ws + static_cast<std::size_t>(t) * head_stride;
      acc += static_cast<double>(::expf(part[0] - s_scale[0])) * part[2U + r];
    }
    dst[r] = static_cast<float>(acc) * s_scale[1];
  }
}

// Online-softmax tiling constants: kQT queries and kKT cache rows per pass.
constexpr std::uint32_t kPrefillQT = 32;
constexpr std::uint32_t kPrefillKT = 8;
constexpr std::uint32_t kPrefillMaxKv = 512;
constexpr std::uint32_t kPrefillMaxRot = 64;

// Fast prefill path. Each block covers a 32-query tile for one head with
// eight warps; a warp owns four consecutive query rows. Query entries (16
// latent dims plus the two rope dims the lane covers) and the warp lane's
// 16 output accumulators live in registers for the whole loop, so key/value
// rows stream through a 16-row shared-memory tile exactly once and scores
// never touch shared memory. QK dots reduce with a butterfly, the online
// softmax state stays in registers, and each lane writes back the same
// accumulator slice it maintained. Ascending key order is preserved;
// the kernel above remains the fallback for other shapes.
constexpr std::uint32_t kPrefillFusedWarpQ = 2U;
constexpr std::uint32_t kPrefillFusedKT = 16U;
constexpr std::uint32_t kPrefillFusedWarps = 16U;
constexpr std::uint32_t kPrefillFusedQTile =
    kPrefillFusedWarpQ * kPrefillFusedWarps;
constexpr std::uint32_t kPrefillFusedKv = 512U;     // Ling3 MLA latent rank
constexpr std::uint32_t kPrefillFusedKvSlot = 16U;  // kPrefillFusedKv / 32

// KvSlice is a compile-time constant so the per-lane query and accumulator
// arrays stay in registers.
template<std::uint32_t KvSlice>
__global__ void MlaPrefillFusedKernel(
    const float* __restrict__ q_abs, const float* __restrict__ q_rope,
    const float* __restrict__ kv, const float* __restrict__ kpe,
    float* __restrict__ latent, std::uint32_t start_pos, std::uint32_t tokens,
    std::uint32_t heads, std::uint32_t kv_rank, std::uint32_t rotary,
    float scale) {
  constexpr std::uint32_t kStride = kPrefillMaxKv + kPrefillMaxRot;
  __shared__ float s_kv[kPrefillFusedKT][kStride];
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t q0 = blockIdx.y * kPrefillFusedQTile;
  const std::uint32_t q_last = min(q0 + kPrefillFusedQTile, tokens);
  const std::uint32_t row_dim = kv_rank + rotary;

  const std::uint32_t warp = tid >> 5U;
  const std::uint32_t wi0 = q0 + warp * kPrefillFusedWarpQ;
  // Beyond-tile warps must still stage shared rows: the staging loop strides
  // by blockDim, so an exited warp would leave its slice stale. They only
  // skip their own compute via the causal skip below.
  const bool warp_active = wi0 < q_last;
  // Per-lane query slices: dims lane + 32*i for the latent part and
  // lane + 32*(i - slice) for the rope part.
  float qreg[kPrefillFusedWarpQ][KvSlice];
  float qrot[kPrefillFusedWarpQ][2U];
  float acc[kPrefillFusedWarpQ][KvSlice];
  float rmax[kPrefillFusedWarpQ];
  float rsum[kPrefillFusedWarpQ];
  for (std::uint32_t i = 0; i < kPrefillFusedWarpQ; ++i) {
    const std::uint32_t token = wi0 + i;
    const bool valid = token < q_last;
    const std::size_t qo =
        (static_cast<std::size_t>(valid ? token : q0) * heads + h) * kv_rank;
    const std::size_t ro =
        (static_cast<std::size_t>(valid ? token : q0) * heads + h) * rotary;
    for (std::uint32_t d = 0; d < KvSlice; ++d) {
      qreg[i][d] = valid ? q_abs[qo + lane + 32U * d] : 0.0F;
      acc[i][d] = 0.0F;
    }
    qrot[i][0U] = valid ? q_rope[ro + lane] : 0.0F;
    qrot[i][1U] = valid ? q_rope[ro + lane + 32U] : 0.0F;
    rmax[i] = kNegInf;
    rsum[i] = 0.0F;
  }

  const std::uint32_t causal_max = start_pos + q_last;
  for (std::uint32_t base = 0; base < causal_max; base += kPrefillFusedKT) {
    __syncthreads();
    for (std::uint32_t idx = tid; idx < kPrefillFusedKT * row_dim;
         idx += blockDim.x) {
      const std::uint32_t k = idx / row_dim;
      const std::uint32_t d = idx % row_dim;
      const std::uint32_t j = base + k;
      s_kv[k][d] =
          (j < causal_max)
              ? ((d < kv_rank) ? kv[static_cast<std::size_t>(j) * kv_rank + d]
                               : kpe[static_cast<std::size_t>(j) * rotary +
                                     (d - kv_rank)])
              : 0.0F;
    }
    __syncthreads();

    const std::uint32_t rows = causal_max - base < kPrefillFusedKT
                                   ? causal_max - base
                                   : kPrefillFusedKT;
    const std::uint32_t qend =
        wi0 + kPrefillFusedWarpQ < q_last ? wi0 + kPrefillFusedWarpQ : q_last;
    if (!warp_active || base >= start_pos + qend) {
      continue;  // Every row of this warp is causally beyond the chunk.
    }
#pragma unroll 4
    for (std::uint32_t k = 0; k < rows; ++k) {
      // Stage the key/value row into registers once; the QK dot and the PV
      // update consume the same values, which halves shared-memory traffic
      // per (query, key) pair without changing any operand or order.
      float v[KvSlice];
      for (std::uint32_t d = 0; d < KvSlice; ++d) {
        v[d] = s_kv[k][lane + 32U * d];
      }
      for (std::uint32_t i = 0; i < kPrefillFusedWarpQ; ++i) {
        const std::uint32_t token = wi0 + i;
        // Two independent partial dots halve the serial fp32 adder chain;
        // each chain covers one half of the latent slice plus one rope dim.
        float pa = qrot[i][0U] * s_kv[k][kv_rank + lane];
        float pb = qrot[i][1U] * s_kv[k][kv_rank + lane + 32U];
        for (std::uint32_t d = 0; d < KvSlice / 2U; ++d) {
          pa += qreg[i][d] * v[d];
          pb += qreg[i][d + KvSlice / 2U] * v[d + KvSlice / 2U];
        }
        float p = pa + pb;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          p += __shfl_xor_sync(~0ULL, p, off);
        }
        float score = p * scale;
        if (token >= q_last || base + k > start_pos + token) {
          score = kNegInf;
        }
        const float m = fmaxf(rmax[i], score);
        // Hardware exp2 softmax: the shipped probe measured 1.19x on the
        // 16K-deep prefill call with a worst output delta of 3.5e-8 against
        // the accurate expf kernel, four orders below the attention quality
        // contract.
        const float corr = (rmax[i] == kNegInf) ? 0.0F : __expf(rmax[i] - m);
        const float e = (score == kNegInf) ? 0.0F : __expf(score - m);
        rsum[i] = rsum[i] * corr + e;
        rmax[i] = m;
        for (std::uint32_t d = 0; d < KvSlice; ++d) {
          acc[i][d] = acc[i][d] * corr + e * v[d];
        }
      }
    }
  }

  for (std::uint32_t i = 0; i < kPrefillFusedWarpQ; ++i) {
    const std::uint32_t token = wi0 + i;
    if (token >= q_last) {
      continue;
    }
    const std::size_t lo =
        (static_cast<std::size_t>(token) * heads + h) * kv_rank;
    for (std::uint32_t d = 0; d < KvSlice; ++d) {
      latent[lo + lane + 32U * d] = acc[i][d] / rsum[i];
    }
  }
}

__global__ void MlaPrefillKernel(const float* __restrict__ q_abs,
                                 const float* __restrict__ q_rope,
                                 const float* __restrict__ kv,
                                 const float* __restrict__ kpe,
                                 float* __restrict__ latent,
                                 std::uint32_t start_pos, std::uint32_t tokens,
                                 std::uint32_t heads, std::uint32_t kv_rank,
                                 std::uint32_t rotary, float scale) {
  // One staged row is kv_rank latents followed by the rotary key half; the
  // pad avoids bank conflicts in the vertical value-mix reads.
  constexpr std::uint32_t kStride = kPrefillMaxKv + kPrefillMaxRot + 1U;
  __shared__ float s_kv[kPrefillKT][kStride];
  __shared__ float s_score[kPrefillQT][kPrefillKT];
  __shared__ float rmax[kPrefillQT];
  __shared__ float rsum[kPrefillQT];
  __shared__ float corr[kPrefillQT];

  const std::uint32_t h = blockIdx.x;
  const std::uint32_t q0 = blockIdx.y * kPrefillQT;
  const std::uint32_t tid = threadIdx.x;
  if (tid < kPrefillQT) {
    rmax[tid] = kNegInf;
    rsum[tid] = 0.0F;
  }

  const std::uint32_t q_last = min(q0 + kPrefillQT, tokens);
  const std::uint32_t causal_max = start_pos + q_last;
  const std::uint32_t row_dim = kv_rank + rotary;
  float acc[kPrefillQT];
#pragma unroll
  for (std::uint32_t i = 0; i < kPrefillQT; ++i) {
    acc[i] = 0.0F;
  }

  for (std::uint32_t base = 0; base < causal_max; base += kPrefillKT) {
    __syncthreads();
    for (std::uint32_t idx = tid; idx < kPrefillKT * row_dim;
         idx += blockDim.x) {
      const std::uint32_t k = idx / row_dim;
      const std::uint32_t d = idx % row_dim;
      const std::uint32_t j = base + k;
      const bool valid = j < causal_max;
      s_kv[k][d] =
          valid ? ((d < kv_rank) ? kv[static_cast<std::size_t>(j) * kv_rank + d]
                                 : kpe[static_cast<std::size_t>(j) * rotary +
                                       (d - kv_rank)])
                : 0.0F;
    }
    __syncthreads();

    // Scores: one thread per (query, key) pair, causal mask as -inf.
    for (std::uint32_t p = tid; p < kPrefillQT * kPrefillKT; p += blockDim.x) {
      const std::uint32_t qi = p / kPrefillKT;
      const std::uint32_t k = p % kPrefillKT;
      const std::uint32_t token = q0 + qi;
      if (token >= tokens || base + k > start_pos + token) {
        s_score[qi][k] = kNegInf;
        continue;
      }
      const std::size_t qo =
          (static_cast<std::size_t>(token) * heads + h) * kv_rank;
      const std::size_t ro =
          (static_cast<std::size_t>(token) * heads + h) * rotary;
      float dot = 0.0F;
      for (std::uint32_t d = 0; d < kv_rank; ++d) {
        dot += q_abs[qo + d] * s_kv[k][d];
      }
      for (std::uint32_t d = 0; d < rotary; ++d) {
        dot += q_rope[ro + d] * s_kv[k][kv_rank + d];
      }
      s_score[qi][k] = dot * scale;
    }
    __syncthreads();

    if (tid < kPrefillQT) {
      float tile_max = kNegInf;
      for (std::uint32_t k = 0; k < kPrefillKT; ++k) {
        tile_max = fmaxf(tile_max, s_score[tid][k]);
      }
      const float new_max = fmaxf(rmax[tid], tile_max);
      corr[tid] = ::expf(rmax[tid] - new_max);
      float tile_sum = 0.0F;
      for (std::uint32_t k = 0; k < kPrefillKT; ++k) {
        const float e = ::expf(s_score[tid][k] - new_max);
        s_score[tid][k] = e;
        tile_sum += e;
      }
      rsum[tid] = rsum[tid] * corr[tid] + tile_sum;
      rmax[tid] = new_max;
    }
    __syncthreads();

    if (tid < kv_rank) {
      for (std::uint32_t qi = 0; qi < kPrefillQT; ++qi) {
        acc[qi] *= corr[qi];
      }
      for (std::uint32_t k = 0; k < kPrefillKT; ++k) {
        const float v = s_kv[k][tid];
        for (std::uint32_t qi = 0; qi < kPrefillQT; ++qi) {
          acc[qi] += s_score[qi][k] * v;
        }
      }
    }
  }

  if (tid < kv_rank) {
    for (std::uint32_t qi = 0; qi < kPrefillQT; ++qi) {
      const std::uint32_t token = q0 + qi;
      if (token >= tokens) {
        continue;
      }
      latent[(static_cast<std::size_t>(token) * heads + h) * kv_rank + tid] =
          acc[qi] / rsum[qi];
    }
  }
}

// WMMA absorbed-MLA prefill (port of the gemma4 3.4a WMMA attention
// skeleton, tools/bench/ling3_mla_prefill_bench candidate 4): 16-key tiles,
// two heads x 16 queries per row block, four waves slicing the 512-dim
// latent (128 dims + 16 rope each), three-term hi/lo fp16 QK with fp32
// accumulation (5.1a contract), fp16-P WMMA projection over the same tile's
// v^T staged from the K rows in registers -- no second KV pass.
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

constexpr std::uint32_t kKeys = 16;
constexpr std::uint32_t kSlice = 128;  // latent dims per wave
constexpr std::uint32_t kSliceSteps = kSlice / 16;
constexpr std::uint32_t kWavesPerRow = 4;  // 512 / kSlice
constexpr std::uint32_t kLatStride = 520;  // K latent halves rows (pad 8)
constexpr std::uint32_t kRopeStride = 72;  // K rope halves rows (pad 8)
constexpr std::uint32_t kVtStride = 16;    // V^T rows, exact (4-way LDS)
constexpr std::uint32_t kPStride = 24;     // P rows (16B-aligned)

constexpr std::uint32_t cdiv(std::uint32_t a, std::uint32_t b) {
  return (a + b - 1U) / b;
}

template<std::uint32_t kHeads>
__global__ void __launch_bounds__(32U * kWavesPerRow * kHeads)
    WmmaMlaPrefill(const float* __restrict__ q_abs,
                   const float* __restrict__ q_rope,
                   const float* __restrict__ kv, const float* __restrict__ kpe,
                   float* __restrict__ latent, std::uint32_t start_pos,
                   std::uint32_t tokens, std::uint32_t heads,
                   std::uint32_t kv_rank, std::uint32_t rotary, float scale) {
  constexpr std::uint32_t kRowBlocks = kHeads;
  constexpr std::uint32_t kThreads = 32U * kWavesPerRow * kHeads;
  __shared__ __align__(16) __half k_lat[kKeys][kLatStride];
  __shared__ __align__(16) __half k_lat_lo[kKeys][kLatStride];
  __shared__ __align__(16) __half k_rope[kKeys][kRopeStride];
  __shared__ __align__(16) __half k_rope_lo[kKeys][kRopeStride];
  __shared__ float s_lds[kRowBlocks][kWavesPerRow][16][16];
  __shared__ __align__(16) __half p_lds[kRowBlocks][2][16][kPStride];
  __shared__ float row_scale[kRowBlocks][16];
  __shared__ float row_sum[kRowBlocks][16];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t wave = __builtin_amdgcn_readfirstlane(tid >> 5U);
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;
  const std::uint32_t rb = wave / kWavesPerRow;
  const std::uint32_t part = wave % kWavesPerRow;
  const std::uint32_t dim0 = part * kSlice;
  const std::uint32_t rdim0 = part * 16U;

  const std::uint32_t q0 = blockIdx.x * 16U;
  const std::uint32_t head = blockIdx.y * kHeads + rb;
  const std::uint32_t q_last = min(q0 + 16U, tokens);
  const std::uint32_t causal_max = start_pos + q_last;

  uint4 q_hi_lat[kSliceSteps], q_lo_lat[kSliceSteps], q_hi_rope, q_lo_rope;

  const std::uint32_t row = q0 + sub;
  {
    const bool live = row < tokens;
    const std::size_t qo =
        (static_cast<std::size_t>(live ? row : q0) * heads + head) * kv_rank;
    const std::size_t ro =
        (static_cast<std::size_t>(live ? row : q0) * heads + head) * rotary;
#pragma unroll
    for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
      const float* src = q_abs + qo + dim0 + ks * 16U + half_id * 8U;
      const float4 f0 = live ? *reinterpret_cast<const float4*>(src)
                             : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      const float4 f1 = live ? *reinterpret_cast<const float4*>(src + 4)
                             : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      const __half2 h[4] = {
          __floats2half2_rn(f0.x, f0.y), __floats2half2_rn(f0.z, f0.w),
          __floats2half2_rn(f1.x, f1.y), __floats2half2_rn(f1.z, f1.w)};
      __builtin_memcpy(&q_hi_lat[ks], h, 16);
      const float2 d0 = __half22float2(h[0]);
      const float2 d1 = __half22float2(h[1]);
      const float2 d2 = __half22float2(h[2]);
      const float2 d3 = __half22float2(h[3]);
      const __half2 l[4] = {__floats2half2_rn(f0.x - d0.x, f0.y - d0.y),
                            __floats2half2_rn(f0.z - d1.x, f0.w - d1.y),
                            __floats2half2_rn(f1.x - d2.x, f1.y - d2.y),
                            __floats2half2_rn(f1.z - d3.x, f1.w - d3.y)};
      __builtin_memcpy(&q_lo_lat[ks], l, 16);
    }
    {
      const float* src = q_rope + ro + rdim0 + half_id * 8U;
      const float4 f0 = live ? *reinterpret_cast<const float4*>(src)
                             : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      const float4 f1 = live ? *reinterpret_cast<const float4*>(src + 4)
                             : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      const __half2 h[4] = {
          __floats2half2_rn(f0.x, f0.y), __floats2half2_rn(f0.z, f0.w),
          __floats2half2_rn(f1.x, f1.y), __floats2half2_rn(f1.z, f1.w)};
      __builtin_memcpy(&q_hi_rope, h, 16);
      const float2 d0 = __half22float2(h[0]);
      const float2 d1 = __half22float2(h[1]);
      const float2 d2 = __half22float2(h[2]);
      const float2 d3 = __half22float2(h[3]);
      const __half2 l[4] = {__floats2half2_rn(f0.x - d0.x, f0.y - d0.y),
                            __floats2half2_rn(f0.z - d1.x, f0.w - d1.y),
                            __floats2half2_rn(f1.x - d2.x, f1.y - d2.y),
                            __floats2half2_rn(f1.z - d3.x, f1.w - d3.y)};
      __builtin_memcpy(&q_lo_rope, l, 16);
    }
  }

  // Rebuild the duplicated-operand fragment from the owning 8 dims per lane
  // (gemma4 3.4a kGlobal trick, attention_wmma.hip.cpp:172).
  const auto q_frag = [&](const uint4& own) -> v16h {
    const auto swap = [](std::uint32_t v) {
      return static_cast<std::uint32_t>(__builtin_amdgcn_permlanex16(
          v, v, 0x76543210U, 0xFEDCBA98U, false, false));
    };
    const uint4 other =
        make_uint4(swap(own.x), swap(own.y), swap(own.z), swap(own.w));
    union {
      v16h f;
      uint4 u[2];
    } cvt;
    cvt.u[0] = half_id == 0 ? own : other;
    cvt.u[1] = half_id == 0 ? other : own;
    return cvt.f;
  };

  v8f o_acc[kSliceSteps] = {};

  // Online softmax: every wave runs its kWavesPerRow-th rows of the block.
  constexpr std::uint32_t kRowsPerPart = 16 / kWavesPerRow;
  constexpr std::uint32_t kSegLanes = 32 / kRowsPerPart;
  constexpr std::uint32_t kSegKeys = kKeys / kSegLanes;
  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::uint32_t sm_row = part * kRowsPerPart + lane / kSegLanes;
  const std::uint32_t sm_seg = lane % kSegLanes;
  const std::uint32_t sm_query = q0 + sm_row;
  const std::uint32_t sm_keys = start_pos + min(sm_query, tokens - 1U) + 1U;

  // Staging (one global pass per tile): a thread owns dims [d4, d4+4) of two
  // consecutive keys, so the transposed V word packs locally with the same
  // f32x2 -> f16x2 conversion as the K hi/lo split; no cross-lane transpose.
  // Latent chunk c = tid + j*kThreads maps pair = c % 8, dg = c / 8: the v^T
  // word at [dim][pair] keeps banks split 4-deep across the warp, while each
  // 32-byte sector is consumed whole by a lane pair. Rope: one chunk per
  // lane, key = tid % 16, float4 slot = (tid / 16).
  constexpr std::uint32_t kLatStage = 4;      // latent chunks per thread-tile
  float4 a_lat[kLatStage], b_lat[kLatStage];  // keys 2p and 2p+1, dims d4
  float4 r_own;                               // rope float4, key tid % 16

  const auto fetch = [&](std::uint32_t key0) {
#pragma unroll
    for (std::uint32_t j = 0; j < kLatStage; ++j) {
      const std::uint32_t c = tid + j * kThreads;
      const std::uint32_t p = c % 8U;   // key pair
      const std::uint32_t dg = c / 8U;  // float4 group within the row
      const std::uint32_t base_key = key0 + 2U * p;
      const bool v0 = base_key < causal_max;
      const bool v1 = base_key + 1U < causal_max;
      const std::size_t o0 =
          static_cast<std::size_t>(v0 ? base_key : 0) * kv_rank + dg * 4U;
      const std::size_t o1 =
          static_cast<std::size_t>(v1 ? base_key + 1U : 0) * kv_rank + dg * 4U;
      a_lat[j] = v0 ? *reinterpret_cast<const float4*>(kv + o0)
                    : make_float4(0, 0, 0, 0);
      b_lat[j] = v1 ? *reinterpret_cast<const float4*>(kv + o1)
                    : make_float4(0, 0, 0, 0);
    }
    const std::uint32_t rk = tid % 16U;
    const std::uint32_t rd = (tid / 16U) * 4U;  // 16 float4 slots over 64 dims
    const bool v = key0 + rk < causal_max;
    r_own = v ? *reinterpret_cast<const float4*>(
                    kpe + static_cast<std::size_t>(key0 + rk) * rotary + rd)
              : make_float4(0, 0, 0, 0);
  };

  const auto h2u = [](__half2 v) {
    return *reinterpret_cast<const std::uint32_t*>(&v);
  };
  const auto store_k = [&] {
#pragma unroll
    for (std::uint32_t j = 0; j < kLatStage; ++j) {
      const std::uint32_t c = tid + j * kThreads;
      const std::uint32_t keyp = 2U * (c % 8U);
      const std::uint32_t d4 = (c / 8U) * 4U;
      const __half2 ha[2] = {__floats2half2_rn(a_lat[j].x, a_lat[j].y),
                             __floats2half2_rn(a_lat[j].z, a_lat[j].w)};
      const __half2 hb[2] = {__floats2half2_rn(b_lat[j].x, b_lat[j].y),
                             __floats2half2_rn(b_lat[j].z, b_lat[j].w)};
      const float2 da0 = __half22float2(ha[0]);
      const float2 da1 = __half22float2(ha[1]);
      const float2 db0 = __half22float2(hb[0]);
      const float2 db1 = __half22float2(hb[1]);
      const __half2 la[2] = {
          __floats2half2_rn(a_lat[j].x - da0.x, a_lat[j].y - da0.y),
          __floats2half2_rn(a_lat[j].z - da1.x, a_lat[j].w - da1.y)};
      const __half2 lb[2] = {
          __floats2half2_rn(b_lat[j].x - db0.x, b_lat[j].y - db0.y),
          __floats2half2_rn(b_lat[j].z - db1.x, b_lat[j].w - db1.y)};
      *reinterpret_cast<uint2*>(&k_lat[keyp][d4]) =
          *reinterpret_cast<const uint2*>(ha);
      *reinterpret_cast<uint2*>(&k_lat_lo[keyp][d4]) =
          *reinterpret_cast<const uint2*>(la);
      *reinterpret_cast<uint2*>(&k_lat[keyp + 1][d4]) =
          *reinterpret_cast<const uint2*>(hb);
      *reinterpret_cast<uint2*>(&k_lat_lo[keyp + 1][d4]) =
          *reinterpret_cast<const uint2*>(lb);
    }
    const std::uint32_t rk = tid % 16U;
    const std::uint32_t rd = (tid / 16U) * 4U;
    const __half2 h[2] = {__floats2half2_rn(r_own.x, r_own.y),
                          __floats2half2_rn(r_own.z, r_own.w)};
    const float2 d0 = __half22float2(h[0]);
    const float2 d1 = __half22float2(h[1]);
    const __half2 l[2] = {__floats2half2_rn(r_own.x - d0.x, r_own.y - d0.y),
                          __floats2half2_rn(r_own.z - d1.x, r_own.w - d1.y)};
    *reinterpret_cast<uint2*>(&k_rope[rk][rd]) =
        *reinterpret_cast<const uint2*>(h);
    *reinterpret_cast<uint2*>(&k_rope_lo[rk][rd]) =
        *reinterpret_cast<const uint2*>(l);
  };
  // Scores (bar 1) retire the K region, so it doubles as hi/lo V^T staging
  // for the CURRENT tile straight from the registers; PV then reads it below.
  const auto store_vt = [&] {
    auto* vw_hi = reinterpret_cast<std::uint32_t*>(&k_lat[0][0]);
    auto* vw_lo = reinterpret_cast<std::uint32_t*>(&k_lat_lo[0][0]);
#pragma unroll
    for (std::uint32_t j = 0; j < kLatStage; ++j) {
      const std::uint32_t c = tid + j * kThreads;
      const std::uint32_t p = c % 8U;
      const std::uint32_t d4 = (c / 8U) * 4U;
      const __half2 ha0 = __floats2half2_rn(a_lat[j].x, b_lat[j].x);
      const __half2 ha1 = __floats2half2_rn(a_lat[j].y, b_lat[j].y);
      const __half2 ha2 = __floats2half2_rn(a_lat[j].z, b_lat[j].z);
      const __half2 ha3 = __floats2half2_rn(a_lat[j].w, b_lat[j].w);
      const float2 fa0 = __half22float2(ha0);
      const float2 fa1 = __half22float2(ha1);
      const float2 fa2 = __half22float2(ha2);
      const float2 fa3 = __half22float2(ha3);
      vw_hi[(d4 + 0U) * (kVtStride / 2U) + p] = h2u(ha0);
      vw_hi[(d4 + 1U) * (kVtStride / 2U) + p] = h2u(ha1);
      vw_hi[(d4 + 2U) * (kVtStride / 2U) + p] = h2u(ha2);
      vw_hi[(d4 + 3U) * (kVtStride / 2U) + p] = h2u(ha3);
      vw_lo[(d4 + 0U) * (kVtStride / 2U) + p] =
          h2u(__floats2half2_rn(a_lat[j].x - fa0.x, b_lat[j].x - fa0.y));
      vw_lo[(d4 + 1U) * (kVtStride / 2U) + p] =
          h2u(__floats2half2_rn(a_lat[j].y - fa1.x, b_lat[j].y - fa1.y));
      vw_lo[(d4 + 2U) * (kVtStride / 2U) + p] =
          h2u(__floats2half2_rn(a_lat[j].z - fa2.x, b_lat[j].z - fa2.y));
      vw_lo[(d4 + 3U) * (kVtStride / 2U) + p] =
          h2u(__floats2half2_rn(a_lat[j].w - fa3.x, b_lat[j].w - fa3.y));
    }
  };

  fetch(0);
  store_k();
  __syncthreads();

  for (std::uint32_t key0 = 0; key0 < causal_max; key0 += kKeys) {
    const bool more = key0 + kKeys < causal_max;

    // Scores over this wave's 9 slices: three terms per 16-dim step, one
    // accumulator per operand class so the tiny hi*lo and lo*hi corrections
    // never round against the large hi*hi running sum, then a two-add tree.
    {
      v8f acc_hh = {};
      v8f acc_hl = {};
      v8f acc_lh = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
        const v16h bh = LoadFrag(&k_lat[sub][dim0 + ks * 16U]);
        const v16h bl = LoadFrag(&k_lat_lo[sub][dim0 + ks * 16U]);
        const v16h ah = q_frag(q_hi_lat[ks]);
        const v16h al = q_frag(q_lo_lat[ks]);
        acc_hh = Wmma(ah, bh, acc_hh);
        acc_hl = Wmma(ah, bl, acc_hl);
        acc_lh = Wmma(al, bh, acc_lh);
      }
      {
        const v16h bh = LoadFrag(&k_rope[sub][rdim0]);
        const v16h bl = LoadFrag(&k_rope_lo[sub][rdim0]);
        const v16h ah = q_frag(q_hi_rope);
        const v16h al = q_frag(q_lo_rope);
        acc_hh = Wmma(ah, bh, acc_hh);
        acc_hl = Wmma(ah, bl, acc_hl);
        acc_lh = Wmma(al, bh, acc_lh);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[rb][part][2 * i + half_id][sub] =
            (acc_hh[i] + acc_hl[i]) + acc_lh[i];
      }
    }
    __syncthreads();

    // Online softmax for this wave's rows; P goes to p_lds in half.
    {
      float vals[kSegKeys];
      float tile_max = -INFINITY;
#pragma unroll
      for (std::uint32_t m = 0; m < kSegKeys; ++m) {
        const std::uint32_t col = sm_seg * kSegKeys + m;
        const std::uint32_t key = key0 + col;
        float s = 0.0F;
#pragma unroll
        for (std::uint32_t p = 0; p < kWavesPerRow; ++p) {
          s += s_lds[rb][p][sm_row][col];
        }
        const bool valid = sm_query < q_last && key < sm_keys;
        vals[m] = valid ? s * scale : -INFINITY;
        tile_max = fmaxf(tile_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSegLanes; off <<= 1) {
        tile_max =
            fmaxf(tile_max, __shfl_xor(tile_max, static_cast<int>(off), 32));
      }
      const float next_max = fmaxf(running_max, tile_max);
      const float corr =
          next_max == -INFINITY ? 1.0F : expf(running_max - next_max);
      float tile_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kSegKeys; ++m) {
        const float e = vals[m] == -INFINITY ? 0.0F : expf(vals[m] - next_max);
        tile_sum += e;
        const __half eh = __float2half(e);
        const float ef = __half2float(eh);
        p_lds[rb][0][sm_row][sm_seg * kSegKeys + m] = eh;
        p_lds[rb][1][sm_row][sm_seg * kSegKeys + m] = __float2half(e - ef);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSegLanes; off <<= 1) {
        tile_sum += __shfl_xor(tile_sum, static_cast<int>(off), 32);
      }
      running_max = next_max;
      running_sum = running_sum * corr + tile_sum;
      if (sm_seg == 0) {
        row_scale[rb][sm_row] = corr;
      }
    }
    // Scores (barrier above) retired the K region: V^T of the CURRENT tile
    // stages hi/lo from the registers into it, read after the next barrier.
    store_vt();
    if (more) {
      fetch(key0 + kKeys);
    }
    __syncthreads();

    float rsc[8];
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      rsc[i] = row_scale[rb][2 * i + half_id];
    }
    bool unchanged = true;
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      unchanged = unchanged && (rsc[i] == 1.0F);
    }
    if (!__all(unchanged)) {
#pragma unroll
      for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[t][i] *= rsc[i];
        }
      }
    }
    const v16h p_hi = LoadFrag(&p_lds[rb][0][sub][0]);
    const v16h p_lo = LoadFrag(&p_lds[rb][1][sub][0]);
#pragma unroll 4
    for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
      const v16h vh = LoadFrag(reinterpret_cast<const __half*>(&k_lat[0][0]) +
                               (dim0 + t * 16U + sub) * kVtStride);
      const v16h vl =
          LoadFrag(reinterpret_cast<const __half*>(&k_lat_lo[0][0]) +
                   (dim0 + t * 16U + sub) * kVtStride);
      o_acc[t] = Wmma(p_hi, vh, o_acc[t]);
      o_acc[t] = Wmma(p_lo, vh, o_acc[t]);
      o_acc[t] = Wmma(p_hi, vl, o_acc[t]);
    }
    __syncthreads();
    // PV retired this tile's V; the registers retired at store_vt above.
    // Prefetch, K-stage and barrier complete the 4-sync cadence.
    if (more) {
      store_k();
      __syncthreads();
    }
  }

  if (sm_seg == 0) {
    row_sum[rb][sm_row] = running_sum;
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    const std::uint32_t r = 2 * i + half_id;
    const std::uint32_t token = q0 + r;
    if (token >= tokens) {
      continue;
    }
    const float l = row_sum[rb][r];
    const float inv = l > 0.0F ? 1.0F / l : 0.0F;
    const std::size_t at =
        (static_cast<std::size_t>(token) * heads + head) * kv_rank + dim0 + sub;
#pragma unroll
    for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
      latent[at + t * 16U] = o_acc[t][i] * inv;
    }
  }
}

}  // namespace
std::uint32_t MlaDecodeTiles(std::uint32_t n_pos, std::uint32_t tile) {
  return (n_pos + tile - 1U) / tile;
}

void MlaDecodePartial(const float* q_abs, const float* kv, const float* kpe,
                      std::uint32_t n_pos, std::uint32_t tile, float scale,
                      float* workspace, std::uint32_t heads,
                      std::uint32_t kv_rank, std::uint32_t rotary,
                      hipStream_t stream) {
  const std::uint32_t tiles = MlaDecodeTiles(n_pos, tile);
  if ((heads % kDecodeGroup) == 0U && (kv_rank & 3U) == 0U &&
      (rotary & 3U) == 0U && kv_rank + rotary <= kDecodeRowStride &&
      kv_rank <= 2U * 256U && tile <= kDecodeMaxTile) {
    MlaDecodePartialGroupKernel<<<dim3(heads / kDecodeGroup, tiles), 256, 0,
                                  stream>>>(q_abs, kv, kpe, n_pos, tile, scale,
                                            workspace, heads, kv_rank, rotary);
    return;
  }
  MlaDecodePartialKernel<<<dim3(tiles, heads), 256, 0, stream>>>(
      q_abs, kv, kpe, n_pos, tile, scale, workspace, heads, kv_rank, rotary);
}

void MlaDecodeCombine(const float* workspace, std::uint32_t tiles,
                      std::uint32_t heads, std::uint32_t kv_rank, float* latent,
                      hipStream_t stream) {
  MlaDecodeCombineKernel<<<heads, 256, 0, stream>>>(workspace, tiles, heads,
                                                    kv_rank, latent);
}

void MlaPrefill(const float* q_abs, const float* q_rope, const float* kv,
                const float* kpe, float* latent, std::uint32_t start_pos,
                std::uint32_t tokens, std::uint32_t heads,
                std::uint32_t kv_rank, std::uint32_t rotary, float scale,
                hipStream_t stream) {
  if (tokens == 0U || kv_rank > kPrefillMaxKv || rotary > kPrefillMaxRot) {
    return;
  }
  if (kv_rank == kPrefillFusedKv && rotary == 2U * 32U && heads >= 2U &&
      (heads & 1U) == 0U) {
    WmmaMlaPrefill<2U>
        <<<dim3((tokens + 15U) / 16U, heads / 2U), 256U, 0, stream>>>(
            q_abs, q_rope, kv, kpe, latent, start_pos, tokens, heads, kv_rank,
            rotary, scale);
    return;
  }
  if (kv_rank == kPrefillFusedKv && rotary == 2U * 32U) {
    const std::uint32_t q_tiles =
        (tokens + kPrefillFusedQTile - 1U) / kPrefillFusedQTile;
    MlaPrefillFusedKernel<kPrefillFusedKvSlot>
        <<<dim3(heads, q_tiles), kPrefillFusedWarps * 32U, 0, stream>>>(
            q_abs, q_rope, kv, kpe, latent, start_pos, tokens, heads, kv_rank,
            rotary, scale);
    return;
  }
  const std::uint32_t q_tiles = (tokens + kPrefillQT - 1U) / kPrefillQT;
  MlaPrefillKernel<<<
      dim3(heads, q_tiles),
      (kv_rank > kPrefillQT * kPrefillKT ? kv_rank : kPrefillQT * kPrefillKT),
      0, stream>>>(q_abs, q_rope, kv, kpe, latent, start_pos, tokens, heads,
                   kv_rank, rotary, scale);
}

}  // namespace gufo::models::ling3_flash::rocm