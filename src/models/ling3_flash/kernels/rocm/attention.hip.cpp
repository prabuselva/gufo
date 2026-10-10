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
  const std::uint32_t q_tiles = (tokens + kPrefillQT - 1U) / kPrefillQT;
  MlaPrefillKernel<<<
      dim3(heads, q_tiles),
      (kv_rank > kPrefillQT * kPrefillKT ? kv_rank : kPrefillQT * kPrefillKT),
      0, stream>>>(q_abs, q_rope, kv, kpe, latent, start_pos, tokens, heads,
                   kv_rank, rotary, scale);
}

}  // namespace gufo::models::ling3_flash::rocm