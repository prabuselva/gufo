#include "src/models/gemma4/kernels/rocm/attention.hpp"

#include <algorithm>
#include <cmath>

namespace gufo::models::gemma4::rocm {
namespace {

__device__ __forceinline__ float WarpReduceSumF(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}

__device__ __forceinline__ double WarpReduceSumD(double v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}

__device__ __forceinline__ double BlockReduceSumD(double v, double* shared) {
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int warps = (static_cast<int>(blockDim.x) + warpSize - 1) / warpSize;
  v = WarpReduceSumD(v);
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  if (warp == 0) {
    v = (lane < warps) ? shared[lane] : 0.0;
    v = WarpReduceSumD(v);
    if (lane == 0) {
      shared[0] = v;
    }
  }
  __syncthreads();
  return shared[0];
}

// RMSNorm of one head into `buf` (shared, >= head_dim floats).
__device__ __forceinline__ void NormHead(const float* x, const float* gamma,
                                         float* buf, std::uint32_t head_dim,
                                         float eps, double* reduce) {
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    const double v = x[i];
    ss += v * v;
  }
  const float scale =
      1.0F / sqrtf(static_cast<float>(BlockReduceSumD(ss, reduce) /
                                      static_cast<double>(head_dim)) +
                   eps);
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    buf[i] = x[i] * scale * (gamma != nullptr ? gamma[i] : 1.0F);
  }
}

// NEOX rope over the first `head_dim` elements of buf (rope_dim == head_dim
// for both layer classes); pair (i, i + half) with angle pos * inv_freq[i].
__device__ __forceinline__ void RopeHead(float* buf, std::uint32_t head_dim,
                                         const float* inv_freq,
                                         std::uint32_t pos) {
  const std::uint32_t half = head_dim / 2;
  for (std::uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
    const float angle = static_cast<float>(pos) * inv_freq[i];
    float s;
    float c;
    sincosf(angle, &s, &c);
    const float a = buf[i];
    const float b = buf[i + half];
    buf[i] = a * c - b * s;
    buf[i + half] = a * s + b * c;
  }
}

constexpr float kRmsEps = 1.0e-6F;

__global__ void QknormRopeKernel(const float* q, const float* gamma, float* out,
                                 std::uint32_t heads, std::uint32_t head_dim,
                                 const float* inv_freq, std::uint32_t start) {
  __shared__ float buf[512];
  __shared__ double reduce[32];
  const std::uint32_t token = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::size_t o =
      (static_cast<std::size_t>(token) * heads + head) * head_dim;
  NormHead(q + o, gamma, buf, head_dim, kRmsEps, reduce);
  __syncthreads();
  RopeHead(buf, head_dim, inv_freq, start + token);
  __syncthreads();
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    out[o + i] = buf[i];
  }
}

__global__ void KvNormRopeWriteKernel(const float* k, const float* v,
                                      const float* k_gamma, __half* k_cache,
                                      __half* v_cache, std::uint32_t kv_heads,
                                      std::uint32_t head_dim,
                                      const float* inv_freq,
                                      std::uint32_t start) {
  __shared__ float kb[512];
  __shared__ float vb[512];
  __shared__ double reduce[32];
  const std::uint32_t token = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::size_t o =
      (static_cast<std::size_t>(token) * kv_heads + head) * head_dim;
  NormHead(k + o, k_gamma, kb, head_dim, kRmsEps, reduce);
  __syncthreads();
  NormHead(v + o, nullptr, vb, head_dim, kRmsEps, reduce);
  __syncthreads();
  RopeHead(kb, head_dim, inv_freq, start + token);
  __syncthreads();
  const std::size_t c =
      (static_cast<std::size_t>(start) + token) * kv_heads * head_dim +
      static_cast<std::size_t>(head) * head_dim;
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    k_cache[c + i] = __float2half(kb[i]);
    v_cache[c + i] = __float2half(vb[i]);
  }
}

// Flash-decoding split pass. Grid (heads, splits): each block owns a
// contiguous chunk of [lo, n_keys) for one query head. A wave processes one
// key at a time with lane l holding element l + 32m of the query row in
// registers, keeping an online (max, sum, weighted-V) triple. Partials are
// [head][split][head_dim + 2] (accumulator, max, sum).
template<std::uint32_t HD>
__global__ void AttentionDecodeSplitKernel(
    const float* q, const __half* k_cache, const __half* v_cache, float* part,
    std::uint32_t lo, std::uint32_t n_keys, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t splits, std::uint32_t chunk) {
  constexpr std::uint32_t kWaves = 8;
  constexpr std::uint32_t kVec = HD / 32;
  __shared__ float s_acc[kWaves][HD];
  __shared__ float s_max[kWaves];
  __shared__ float s_sum[kWaves];
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t sp = blockIdx.y;
  const std::uint32_t kvh = h / (heads / kv_heads);
  const float* qh = q + static_cast<std::size_t>(h) * HD;
  const std::size_t kv_stride = static_cast<std::size_t>(kv_heads) * HD;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = threadIdx.x >> 5;
  float qv[kVec];
#pragma unroll
  for (std::uint32_t m = 0; m < kVec; ++m) {
    qv[m] = qh[lane + 32U * m];
  }
  float wmax = -INFINITY;
  float wsum = 0.0F;
  float acc[kVec];
#pragma unroll
  for (std::uint32_t m = 0; m < kVec; ++m) {
    acc[m] = 0.0F;
  }
  const std::uint32_t j0 = lo + sp * chunk;
  const std::uint32_t j1 = (n_keys < j0 + chunk) ? n_keys : j0 + chunk;
  for (std::uint32_t j = j0 + wave; j < j1; j += kWaves) {
    const __half* kj =
        k_cache + static_cast<std::size_t>(j) * kv_stride + kvh * HD;
    float dot = 0.0F;
#pragma unroll
    for (std::uint32_t m = 0; m < kVec; ++m) {
      dot += qv[m] * __half2float(kj[lane + 32U * m]);
    }
    const float score = __shfl(WarpReduceSumF(dot), 0);
    const float nm = fmaxf(wmax, score);
    const float r = expf(wmax - nm);
    const float e = expf(score - nm);
    wsum = wsum * r + e;
    const __half* vj =
        v_cache + static_cast<std::size_t>(j) * kv_stride + kvh * HD;
#pragma unroll
    for (std::uint32_t m = 0; m < kVec; ++m) {
      acc[m] = acc[m] * r + e * __half2float(vj[lane + 32U * m]);
    }
    wmax = nm;
  }
  if (lane == 0) {
    s_max[wave] = wmax;
    s_sum[wave] = wsum;
  }
  __syncthreads();
  float bmax = -INFINITY;
#pragma unroll
  for (std::uint32_t w = 0; w < kWaves; ++w) {
    bmax = fmaxf(bmax, s_max[w]);
  }
  const float r = (wmax == -INFINITY) ? 0.0F : expf(wmax - bmax);
#pragma unroll
  for (std::uint32_t m = 0; m < kVec; ++m) {
    s_acc[wave][lane + 32U * m] = acc[m] * r;
  }
  __syncthreads();
  float* pp = part + (static_cast<std::size_t>(h) * splits + sp) * (HD + 2U);
  for (std::uint32_t d = threadIdx.x; d < HD; d += blockDim.x) {
    float val = 0.0F;
#pragma unroll
    for (std::uint32_t w = 0; w < kWaves; ++w) {
      val += s_acc[w][d];
    }
    pp[d] = val;
  }
  if (threadIdx.x == 0) {
    float total = 0.0F;
#pragma unroll
    for (std::uint32_t w = 0; w < kWaves; ++w) {
      total +=
          (s_max[w] == -INFINITY) ? 0.0F : s_sum[w] * expf(s_max[w] - bmax);
    }
    pp[HD] = bmax;
    pp[HD + 1U] = total;
  }
}

__global__ void AttentionDecodeCombineKernel(const float* part, float* out,
                                             std::uint32_t heads,
                                             std::uint32_t splits,
                                             std::uint32_t head_dim) {
  const std::uint32_t h = blockIdx.x;
  const float* pp =
      part + static_cast<std::size_t>(h) * splits * (head_dim + 2U);
  float bmax = -INFINITY;
  for (std::uint32_t s = 0; s < splits; ++s) {
    bmax = fmaxf(bmax, pp[s * (head_dim + 2U) + head_dim]);
  }
  double sum = 0.0;
  for (std::uint32_t s = 0; s < splits; ++s) {
    const float m = pp[s * (head_dim + 2U) + head_dim];
    sum += (m == -INFINITY)
               ? 0.0
               : static_cast<double>(pp[s * (head_dim + 2U) + head_dim + 1U]) *
                     expf(m - bmax);
  }
  const float inv = 1.0F / static_cast<float>(sum);
  const std::uint32_t d = threadIdx.x;
  double acc = 0.0;
  for (std::uint32_t s = 0; s < splits; ++s) {
    const float m = pp[s * (head_dim + 2U) + head_dim];
    if (m == -INFINITY) {
      continue;
    }
    acc += static_cast<double>(pp[s * (head_dim + 2U) + d]) * expf(m - bmax);
  }
  out[static_cast<std::size_t>(h) * head_dim + d] =
      static_cast<float>(acc) * inv;
}

// Tiled causal windowed prefill on the F16 cache. One workgroup handles kQT
// queries of one head and streams KV tiles through shared memory; the online
// softmax keeps a running max/sum per query. Masked scores use -1e30 (not
// -inf) so a fully masked leading tile cannot poison the running state: the
// first tile with a valid key rescales it away exactly.
__global__ void AttentionPrefillTiledKernel(
    const float* q, const __half* k_cache, const __half* v_cache, float* out,
    std::uint32_t start, std::uint32_t tokens, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, std::uint32_t window) {
  constexpr std::uint32_t kQT = 32;
  constexpr std::uint32_t kKT = 8;
  constexpr std::uint32_t kMaxHD = 512;
  constexpr std::uint32_t kStride = kMaxHD + 1;
  constexpr float kMask = -1.0e30F;
  __shared__ float sK[kKT][kStride];
  __shared__ float sV[kKT][kStride];
  __shared__ float sScore[kQT][kKT];
  __shared__ float rmax[kQT];
  __shared__ float rsum[kQT];
  __shared__ float corr[kQT];

  const std::uint32_t h = blockIdx.x;
  const std::uint32_t q0 = blockIdx.y * kQT;
  const std::uint32_t kvh = h / (heads / kv_heads);
  const std::uint32_t tid = threadIdx.x;
  const std::size_t kv_stride = static_cast<std::size_t>(kv_heads) * head_dim;

  if (tid < kQT) {
    rmax[tid] = kMask;
    rsum[tid] = 0.0F;
  }

  const std::uint32_t q_last = q0 + kQT < tokens ? q0 + kQT : tokens;
  const std::uint32_t causal_max = start + q_last;
  const auto lo_of = [&](std::uint32_t token) {
    const std::uint32_t causal = start + token + 1;
    return (window > 0 && causal > window) ? causal - window : 0;
  };
  const std::uint32_t base_min = lo_of(q0);
  float acc[kQT];
  for (std::uint32_t i = 0; i < kQT; ++i) {
    acc[i] = 0.0F;
  }

  for (std::uint32_t base = base_min; base < causal_max; base += kKT) {
    __syncthreads();
    for (std::uint32_t idx = tid; idx < kKT * head_dim; idx += blockDim.x) {
      const std::uint32_t k = idx / head_dim;
      const std::uint32_t d = idx % head_dim;
      const std::uint32_t j = base + k;
      const std::size_t o =
          j * kv_stride + static_cast<std::size_t>(kvh) * head_dim + d;
      const bool valid = j < causal_max;
      sK[k][d] = valid ? __half2float(k_cache[o]) : 0.0F;
      sV[k][d] = valid ? __half2float(v_cache[o]) : 0.0F;
    }
    __syncthreads();

    for (std::uint32_t p = tid; p < kQT * kKT; p += blockDim.x) {
      const std::uint32_t qi = p / kKT;
      const std::uint32_t k = p % kKT;
      const std::uint32_t token = q0 + qi;
      const std::uint32_t j = base + k;
      if (token >= tokens || j > start + token || j < lo_of(token)) {
        sScore[qi][k] = kMask;
        continue;
      }
      const float* qh =
          q + (static_cast<std::size_t>(token) * heads + h) * head_dim;
      float dot = 0.0F;
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        dot += qh[d] * sK[k][d];
      }
      sScore[qi][k] = dot;
    }
    __syncthreads();

    if (tid < kQT) {
      float tile_max = kMask;
      for (std::uint32_t k = 0; k < kKT; ++k) {
        tile_max = fmaxf(tile_max, sScore[tid][k]);
      }
      const float new_max = fmaxf(rmax[tid], tile_max);
      corr[tid] = expf(rmax[tid] - new_max);
      float tile_sum = 0.0F;
      for (std::uint32_t k = 0; k < kKT; ++k) {
        const float e = expf(sScore[tid][k] - new_max);
        sScore[tid][k] = e;
        tile_sum += e;
      }
      rsum[tid] = rsum[tid] * corr[tid] + tile_sum;
      rmax[tid] = new_max;
    }
    __syncthreads();

    for (std::uint32_t qi = 0; qi < kQT; ++qi) {
      acc[qi] *= corr[qi];
    }
    for (std::uint32_t d = tid; d < head_dim; d += blockDim.x) {
      for (std::uint32_t k = 0; k < kKT; ++k) {
        const float sv = sV[k][d];
        for (std::uint32_t qi = 0; qi < kQT; ++qi) {
          acc[qi] += sScore[qi][k] * sv;
        }
      }
    }
  }

  for (std::uint32_t d = tid; d < head_dim; d += blockDim.x) {
    for (std::uint32_t qi = 0; qi < kQT; ++qi) {
      const std::uint32_t token = q0 + qi;
      if (token >= tokens) {
        continue;
      }
      const std::size_t o =
          (static_cast<std::size_t>(token) * heads + h) * head_dim + d;
      out[o] = acc[qi] / rsum[qi];
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// WMMA flash-attention prefill (matrix-core path for both layer classes).
//
// The scalar AttentionPrefillTiledKernel above stays the numerical oracle; this
// kernel replaces it on the production path. Every QK^T and PV product becomes
// a wave32 `wmma_f32_16x16x16_f16`. Fragment layout (verified against the CPU
// reference): A holds row L%16 and 16 contiguous k, B holds column L%16 and 16
// contiguous k (fed a transposed tile), C element i is row 2i + L/16, column
// L%16. Q never reaches LDS -- each wave keeps its A fragments in registers for
// the whole key loop. K and the transposed V share one staging buffer.
//
// The head dimension is a template parameter so the same tiling covers the full
// layers (512) and the SWA layers (256); the waves split the head dimension
// into `kWaves / kSTiles` halves summed through LDS, and each wave owns
// `kDimTilesPerWave = (kHeadDim/16)/kWaves` output dim tiles. The attention
// scale is 1.0 (pinned by the oracle), so Q is converted to F16 unscaled.
// ---------------------------------------------------------------------------

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f WmmaF16(v16h a, v16h b, v8f c) {
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

namespace {

constexpr std::uint32_t kWmmaHeads = 2;  // query heads per block, divides GQA

template<std::uint32_t kQueryRows, std::uint32_t kKeys, std::uint32_t kWaves,
         std::uint32_t kHeadDim>
__launch_bounds__(kWaves * 32, 1) __global__
    void AttentionPrefillWmmaKernel(const float* __restrict__ q,
                                    const __half* __restrict__ k_cache,
                                    const __half* __restrict__ v_cache,
                                    float* __restrict__ out,
                                    std::uint32_t start, std::uint32_t tokens,
                                    std::uint32_t heads, std::uint32_t kv_heads,
                                    std::uint32_t window) {
  constexpr std::uint32_t kRowBlocks = (kQueryRows / 16) * kWmmaHeads;
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kThreads = kWaves * 32;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kWmmaKSteps = kHeadDim / 16;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / (kWaves / kSTiles);
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kDimTilesPerWave = kWmmaKSteps / kWaves;
  constexpr std::uint32_t kSoftmaxLanes = kThreads / kRows;
  constexpr std::uint32_t kWmmaKStride = kHeadDim + 8;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  static_assert(kSTiles * 2 == kWaves,
                "the waves cover the S tiles in two head-dimension halves");
  static_assert(kKeys % 16 == 0 && kQueryRows % 16 == 0, "16-row WMMA tiles");
  static_assert(kKStepsPerWave * (kWaves / kSTiles) == kWmmaKSteps, "k split");
  static_assert(kWmmaKStride % 8 == 0 && kVtStride % 8 == 0,
                "fragment rows must start on a 16-byte boundary");
  static_assert(kSoftmaxLanes * (kKeys / kSoftmaxLanes) == kKeys, "softmax");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  const std::uint32_t gqa = heads / kv_heads;
  const std::size_t attn_width = static_cast<std::size_t>(heads) * kHeadDim;
  const std::size_t kv_width = static_cast<std::size_t>(kv_heads) * kHeadDim;

  const std::uint32_t query_start = blockIdx.x * kQueryRows;
  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (gqa / kWmmaHeads);
  const std::uint32_t pair_in_group = head_pair % (gqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      (kv_head * gqa) + (pair_in_group * kWmmaHeads);

  const auto row_block_head = [&](std::uint32_t rb) {
    return first_query_head + (rb % kWmmaHeads);
  };
  const auto row_block_offset = [&](std::uint32_t rb) {
    return (rb / kWmmaHeads) * 16;
  };

  constexpr std::uint32_t kKvLdsHalves =
      (kKeys * kWmmaKStride > kHeadDim * kVtStride) ? (kKeys * kWmmaKStride)
                                                    : (kHeadDim * kVtStride);
  __shared__ __half kv_lds[kKvLdsHalves];
  __shared__ float s_lds[2][kSTiles][16][16];
  __shared__ __half p_lds[kRows][kKeys];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  {
    const std::uint32_t local_query =
        query_start + row_block_offset(s_rb) + sub;
    const bool live = local_query < tokens;
    const float* q_row =
        q + (static_cast<std::size_t>(local_query) * attn_width) +
        (static_cast<std::size_t>(row_block_head(s_rb)) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w);
      }
    }
  }

  v8f o_acc[kRowBlocks][kDimTilesPerWave] = {};
  for (std::uint32_t r = tid; r < kRows; r += kThreads) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0F;
  }

  const std::uint32_t context_end = start + tokens;
  const std::uint32_t max_visible =
      min(context_end, start + query_start + kQueryRows);
  // The block's first query has the smallest window lower bound; keys below it
  // are invisible to every query in the block, so the key loop starts there.
  const std::uint32_t block_causal = start + query_start + 1u;
  const std::uint32_t key_begin =
      (window > 0 && block_causal > window)
          ? ((block_causal - window) / kKeys) * kKeys
          : 0u;

  for (std::uint32_t key_start = key_begin; key_start < max_visible;
       key_start += kKeys) {
    constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (kThreads * 8);
    uint4 v_raw[kVRegs];
    {
      const std::uint32_t v_key = lane % kKeys;
      const std::uint32_t v_slice = ((tid / kKeys) * (kVRegs * 8));
      const std::uint32_t key_position = key_start + v_key;
      const auto* src =
          v_cache + (static_cast<std::size_t>(key_position) * kv_width) +
          (static_cast<std::size_t>(kv_head) * kHeadDim) + v_slice;
      const bool live = key_position < context_end;
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        v_raw[j] = live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                        : make_uint4(0u, 0u, 0u, 0u);
      }
    }

    __syncthreads();
    for (std::uint32_t idx = tid; idx < kKeys * (kHeadDim / 8);
         idx += kThreads) {
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      const std::uint32_t key_position = key_start + key_row;
      __half packed[8] = {};
      if (key_position < context_end) {
        const auto* src = k_cache +
                          (static_cast<std::size_t>(key_position) * kv_width) +
                          (static_cast<std::size_t>(kv_head) * kHeadDim) + d8;
        *reinterpret_cast<uint4*>(packed) =
            *reinterpret_cast<const uint4*>(src);
      }
      *reinterpret_cast<uint4*>(&kv_lds[(key_row * kWmmaKStride) + d8]) =
          *reinterpret_cast<const uint4*>(packed);
    }
    __syncthreads();

    // --- S = Q K^T ---
    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const v16h k_frag =
            LoadFrag(&kv_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        s_acc = WmmaF16(q_frag[ks], k_frag, s_acc);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
    __syncthreads();

    // --- online softmax: kSoftmaxLanes threads per query row ---
    {
      const std::uint32_t rg = tid / kSoftmaxLanes;
      const std::uint32_t seg = tid % kSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + row;
      const std::uint32_t absolute_query = start + local_query;
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key_position = key_start + col;
        bool valid = local_query < tokens && key_position <= absolute_query &&
                     key_position < context_end;
        if (window > 0) {
          valid = valid && (absolute_query - key_position < window);
        }
        const std::uint32_t tile = ((col / 16) * kRowBlocks) + rb;
        vals[m] = valid ? (s_lds[0][tile][row][col % 16] +
                           s_lds[1][tile][row][col % 16])
                        : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      }
      const float prev_max = row_max[rg];
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p_lds[rg][(seg * kPerLane) + m] = static_cast<__half>(w);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_sum += __shfl_xor(part_sum, off);
      }
      if (seg == 0) {
        row_max[rg] = next_max;
        row_sum[rg] = (row_sum[rg] * prior_scale) + part_sum;
        row_scale[rg] = prior_scale;
      }
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
      float scale[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < kDimTilesPerWave; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale[i];
        }
      }
    }

    // --- stage V transposed ---
    {
      const std::uint32_t v_key = lane % kKeys;
      const std::uint32_t v_slice = ((tid / kKeys) * (kVRegs * 8));
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        const auto* packed = reinterpret_cast<const __half*>(&v_raw[j]);
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          kv_lds[((v_slice + (j * 8) + i) * kVtStride) + v_key] = packed[i];
        }
      }
    }
    __syncthreads();

    // --- O += P V ---
#pragma unroll
    for (std::uint32_t t = 0; t < kDimTilesPerWave; ++t) {
      const std::uint32_t dim_tile = wave + (t * kWaves);
      v16h v_frag[kKeyBlocks];
#pragma unroll
      for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
        v_frag[kb] = LoadFrag(
            &kv_lds[(((dim_tile * 16) + sub) * kVtStride) + (kb * 16)]);
      }
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          o_acc[rb][t] = WmmaF16(p_frag, v_frag[kb], o_acc[rb][t]);
        }
      }
    }
  }
  __syncthreads();

  // --- epilogue ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
    const std::uint32_t query_head = row_block_head(rb);
    const std::uint32_t row_offset = row_block_offset(rb);
#pragma unroll
    for (std::uint32_t t = 0; t < kDimTilesPerWave; ++t) {
      const std::uint32_t dim_tile = wave + (t * kWaves);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        const std::uint32_t local_query = query_start + row_offset + row;
        if (local_query >= tokens) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(local_query) * attn_width) +
            (static_cast<std::size_t>(query_head) * kHeadDim) +
            (dim_tile * 16) + sub;
        out[offset] =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
      }
    }
  }
}

}  // namespace

void QknormRope(const float* q, const float* gamma, float* out,
                std::uint32_t tokens, std::uint32_t heads,
                std::uint32_t head_dim, const float* inv_freq,
                std::uint32_t start, hipStream_t stream) {
  QknormRopeKernel<<<dim3(tokens, heads), 256, 0, stream>>>(
      q, gamma, out, heads, head_dim, inv_freq, start);
}

void KvNormRopeWrite(const float* k, const float* v, const float* k_gamma,
                     __half* k_cache, __half* v_cache, std::uint32_t start,
                     std::uint32_t tokens, std::uint32_t kv_heads,
                     std::uint32_t head_dim, const float* inv_freq,
                     hipStream_t stream) {
  KvNormRopeWriteKernel<<<dim3(tokens, kv_heads), 256, 0, stream>>>(
      k, v, k_gamma, k_cache, v_cache, kv_heads, head_dim, inv_freq, start);
}

void AttentionDecode(const float* q, const __half* k_cache,
                     const __half* v_cache, float* out, float* scratch,
                     std::uint32_t n_keys, std::uint32_t lo,
                     std::uint32_t heads, std::uint32_t kv_heads,
                     std::uint32_t head_dim, hipStream_t stream) {
  if (n_keys <= lo) {
    return;
  }
  const std::uint32_t range = n_keys - lo;
  std::uint32_t splits = (range + 63U) / 64U;
  if (splits > 32U) {
    splits = 32U;
  }
  std::uint32_t chunk = (range + splits - 1U) / splits;
  splits = (range + chunk - 1U) / chunk;
#define GUFO_GEMMA4_DECODE_SPLIT(HD)                                         \
  if (head_dim == HD) {                                                      \
    AttentionDecodeSplitKernel<HD><<<dim3(heads, splits), 256, 0, stream>>>( \
        q, k_cache, v_cache, scratch, lo, n_keys, heads, kv_heads, splits,   \
        chunk);                                                              \
  } else
  GUFO_GEMMA4_DECODE_SPLIT(256U)
  GUFO_GEMMA4_DECODE_SPLIT(512U) {
    return;
  }
#undef GUFO_GEMMA4_DECODE_SPLIT
  AttentionDecodeCombineKernel<<<heads, head_dim, 0, stream>>>(
      scratch, out, heads, splits, head_dim);
}

void AttentionPrefillScalar(const float* q, const __half* k_cache,
                            const __half* v_cache, float* out,
                            std::uint32_t start, std::uint32_t tokens,
                            std::uint32_t heads, std::uint32_t kv_heads,
                            std::uint32_t head_dim, std::uint32_t window,
                            hipStream_t stream) {
  if (tokens == 0 || head_dim > 512U) {
    return;
  }
  const dim3 grid(heads, (tokens + 31U) / 32U);
  // One workgroup thread per head dimension: the online accumulator is a
  // per-thread register, so blockDim.x must cover the whole head.
  AttentionPrefillTiledKernel<<<grid, head_dim, 0, stream>>>(
      q, k_cache, v_cache, out, start, tokens, heads, kv_heads, head_dim,
      window);
}

void AttentionPrefill(const float* q, const __half* k_cache,
                      const __half* v_cache, float* out, std::uint32_t start,
                      std::uint32_t tokens, std::uint32_t heads,
                      std::uint32_t kv_heads, std::uint32_t head_dim,
                      std::uint32_t window, hipStream_t stream) {
  if (tokens == 0 || head_dim > 512U) {
    return;
  }
  // The WMMA path covers the two Gemma-4 head dimensions; anything else keeps
  // the scalar oracle.
  if (head_dim != 256U && head_dim != 512U) {
    AttentionPrefillScalar(q, k_cache, v_cache, out, start, tokens, heads,
                           kv_heads, head_dim, window, stream);
    return;
  }
  constexpr std::uint32_t kQueryRows = 32;
  constexpr std::uint32_t kKeys = 16;
  constexpr std::uint32_t kWaves = 8;
  const dim3 grid((tokens + kQueryRows - 1U) / kQueryRows, heads / kWmmaHeads);
  if (head_dim == 512U) {
    AttentionPrefillWmmaKernel<kQueryRows, kKeys, kWaves, 512>
        <<<grid, kWaves * 32, 0, stream>>>(q, k_cache, v_cache, out, start,
                                           tokens, heads, kv_heads, window);
  } else {
    AttentionPrefillWmmaKernel<kQueryRows, kKeys, kWaves, 256>
        <<<grid, kWaves * 32, 0, stream>>>(q, k_cache, v_cache, out, start,
                                           tokens, heads, kv_heads, window);
  }
}

}  // namespace gufo::models::gemma4::rocm