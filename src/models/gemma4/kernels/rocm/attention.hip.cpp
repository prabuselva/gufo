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

void AttentionPrefill(const float* q, const __half* k_cache,
                      const __half* v_cache, float* out, std::uint32_t start,
                      std::uint32_t tokens, std::uint32_t heads,
                      std::uint32_t kv_heads, std::uint32_t head_dim,
                      std::uint32_t window, hipStream_t stream) {
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

}  // namespace gufo::models::gemma4::rocm