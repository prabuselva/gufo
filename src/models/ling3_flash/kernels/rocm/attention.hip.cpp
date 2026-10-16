#include "src/models/ling3_flash/kernels/rocm/attention.hpp"

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