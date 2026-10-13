#include "src/models/gemma4/kernels/rocm/attention.hpp"

#include <algorithm>
#include <cmath>

#include "src/models/gemma4/kernels/rocm/attention_wmma.hpp"

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

// Butterfly tree: every lane ends with the full sum, so callers can use the
// result directly instead of broadcasting lane 0 afterwards.
__device__ __forceinline__ double WarpAllReduceSumD(double v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
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

// Warp-wide variants: one warp owns one token and loops its heads, so each
// lane computes its (head_dim/64) rope angles ONCE for the token — every head
// shares inv_freq — and reuses them in registers. Per head the NEOX pairs
// `(i, half+i)` and the double RMS sum stay in registers and the reduction is
// a bare butterfly shuffle. head_dim must be a multiple of 64; the block
// kernels below are the fallback for exotic widths. Per-element arithmetic is
// the same (norm scale in float over a double sum, then rope); only the sum's
// association order moves, and it stays in double.
// `pitch` is the float stride of one input token row: equal to
// heads * head_dim for a standalone q projection, or to the fused qkv row
// width when q is a column slice of a wider fused GEMM output. The output
// stays packed (heads * head_dim) because attention reads only q.
__global__ void QknormRopeWarpKernel(const float* q, const float* gamma,
                                     float* out, std::uint32_t tokens,
                                     std::uint32_t heads,
                                     std::uint32_t head_dim,
                                     const float* inv_freq, std::uint32_t start,
                                     std::uint32_t pitch) {
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t token =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (token >= tokens) {
    return;
  }
  const std::uint32_t half = head_dim / 2U;
  const std::uint32_t pairs = half / warpSize;
  float cs[16];
  float sn[16];
  const float pos = static_cast<float>(start + token);
  for (std::uint32_t m = 0; m < pairs; ++m) {
    const std::uint32_t i = lane + warpSize * m;
    sincosf(pos * inv_freq[i], &sn[m], &cs[m]);
  }
  float a[16];
  float b[16];
  const float* row = q + static_cast<std::size_t>(token) * pitch;
  float* orow = out + static_cast<std::size_t>(token) * heads * head_dim;
  for (std::uint32_t head = 0; head < heads; ++head) {
    double ss = 0.0;
    for (std::uint32_t m = 0; m < pairs; ++m) {
      const std::uint32_t i = lane + warpSize * m;
      a[m] = row[i];
      b[m] = row[half + i];
      ss += static_cast<double>(a[m]) * a[m] + static_cast<double>(b[m]) * b[m];
    }
    const float scale =
        1.0F / sqrtf(static_cast<float>(WarpAllReduceSumD(ss) /
                                        static_cast<double>(head_dim)) +
                     kRmsEps);
    for (std::uint32_t m = 0; m < pairs; ++m) {
      const std::uint32_t i = lane + warpSize * m;
      const float ya = a[m] * scale * (gamma != nullptr ? gamma[i] : 1.0F);
      const float yb =
          b[m] * scale * (gamma != nullptr ? gamma[half + i] : 1.0F);
      orow[i] = ya * cs[m] - yb * sn[m];
      orow[half + i] = ya * sn[m] + yb * cs[m];
    }
    row += head_dim;
    orow += head_dim;
  }
}

__global__ void KvNormRopeWriteWarpKernel(
    const float* k, const float* v, const float* k_gamma, __half* k_cache,
    __half* v_cache, std::uint32_t tokens, std::uint32_t kv_heads,
    std::uint32_t head_dim, const float* inv_freq, std::uint32_t start,
    std::uint32_t pitch) {
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t token =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (token >= tokens) {
    return;
  }
  const std::uint32_t half = head_dim / 2U;
  const std::uint32_t pairs = half / warpSize;
  float cs[16];
  float sn[16];
  const float pos = static_cast<float>(start + token);
  for (std::uint32_t m = 0; m < pairs; ++m) {
    const std::uint32_t i = lane + warpSize * m;
    sincosf(pos * inv_freq[i], &sn[m], &cs[m]);
  }
  float ka[16];
  float kb[16];
  float va[16];
  float vb[16];
  const float* krow = k + static_cast<std::size_t>(token) * pitch;
  const float* vrow = v + static_cast<std::size_t>(token) * pitch;
  __half* kc =
      k_cache + (static_cast<std::size_t>(start) + token) * kv_heads * head_dim;
  __half* vc =
      v_cache + (static_cast<std::size_t>(start) + token) * kv_heads * head_dim;
  for (std::uint32_t head = 0; head < kv_heads; ++head) {
    double kss = 0.0;
    double vss = 0.0;
    for (std::uint32_t m = 0; m < pairs; ++m) {
      const std::uint32_t i = lane + warpSize * m;
      ka[m] = krow[i];
      kb[m] = krow[half + i];
      va[m] = vrow[i];
      vb[m] = vrow[half + i];
      kss += static_cast<double>(ka[m]) * ka[m] +
             static_cast<double>(kb[m]) * kb[m];
      vss += static_cast<double>(va[m]) * va[m] +
             static_cast<double>(vb[m]) * vb[m];
    }
    const float kscale =
        1.0F / sqrtf(static_cast<float>(WarpAllReduceSumD(kss) /
                                        static_cast<double>(head_dim)) +
                     kRmsEps);
    const float vscale =
        1.0F / sqrtf(static_cast<float>(WarpAllReduceSumD(vss) /
                                        static_cast<double>(head_dim)) +
                     kRmsEps);
    for (std::uint32_t m = 0; m < pairs; ++m) {
      const std::uint32_t i = lane + warpSize * m;
      const float ya =
          ka[m] * kscale * (k_gamma != nullptr ? k_gamma[i] : 1.0F);
      const float yb =
          kb[m] * kscale * (k_gamma != nullptr ? k_gamma[half + i] : 1.0F);
      kc[i] = __float2half(ya * cs[m] - yb * sn[m]);
      kc[half + i] = __float2half(ya * sn[m] + yb * cs[m]);
      vc[i] = __float2half(va[m] * vscale);
      vc[half + i] = __float2half(vb[m] * vscale);
    }
    krow += head_dim;
    vrow += head_dim;
    kc += head_dim;
    vc += head_dim;
  }
}

// QKV post-processing in one launch: the q-head norm+RoPE jobs and the
// kv-head norm+RoPE/cache-write jobs of the two warp kernels above, dispatched
// round-robin over the block's warps (one block per token, one warp per head).
// every head passes through the verbatim load/reduce/store loops of
// QknormRopeWarpKernel and KvNormRopeWriteWarpKernel, so q and v are
// bit-identical to running the two kernels back to back and K differs only on
// rope rounding ties (see the store-loop note below). The RoPE sincos are
// computed once per warp and shared by all of its heads. HD selects a
// compile-time head_dim so the per-head loops and register arrays specialize
// and fully unroll; HD == 0 keeps the runtime head_dim for exotic shapes.
// Unrolling preserves the element and accumulation order, so every instance
// is bit-identical to the runtime one.
template<std::uint32_t HD>
__global__ void QkvPostWarpKernel(const float* q, const float* q_gamma,
                                  float* q_out, const float* k, const float* v,
                                  const float* k_gamma, __half* k_cache,
                                  __half* v_cache, std::uint32_t heads,
                                  std::uint32_t kv_heads,
                                  std::uint32_t head_dim, const float* inv_freq,
                                  std::uint32_t start, std::uint32_t q_pitch,
                                  std::uint32_t kv_pitch) {
  const std::uint32_t hdim = HD != 0U ? HD : head_dim;
  constexpr std::uint32_t kPairs = HD != 0U ? HD / 64U : 16U;
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t warp = threadIdx.x / warpSize;
  const std::uint32_t warps = blockDim.x / warpSize;
  const std::uint32_t token = blockIdx.x;
  const std::uint32_t total = heads + kv_heads;
  if (warp >= total) {
    return;
  }
  const std::uint32_t half = hdim / 2U;
  const std::uint32_t pairs = half / warpSize;
  float cs[kPairs];
  float sn[kPairs];
  const float pos = static_cast<float>(start + token);
  for (std::uint32_t m = 0; m < pairs; ++m) {
    const std::uint32_t i = lane + warpSize * m;
    sincosf(pos * inv_freq[i], &sn[m], &cs[m]);
  }
  const float* qrow = q + static_cast<std::size_t>(token) * q_pitch;
  const float* krow = k + static_cast<std::size_t>(token) * kv_pitch;
  const float* vrow = v + static_cast<std::size_t>(token) * kv_pitch;
  float* orow0 = q_out + static_cast<std::size_t>(token) * heads * hdim;
  __half* kc0 =
      k_cache + (static_cast<std::size_t>(start) + token) * kv_heads * hdim;
  __half* vc0 =
      v_cache + (static_cast<std::size_t>(start) + token) * kv_heads * hdim;
  for (std::uint32_t h = warp; h < total; h += warps) {
    if (h < heads) {
      float a[kPairs];
      float b[kPairs];
      const float* row = qrow + h * hdim;
      float* orow = orow0 + h * hdim;
      double ss = 0.0;
      for (std::uint32_t m = 0; m < pairs; ++m) {
        const std::uint32_t i = lane + warpSize * m;
        a[m] = row[i];
        b[m] = row[half + i];
        ss +=
            static_cast<double>(a[m]) * a[m] + static_cast<double>(b[m]) * b[m];
      }
      const float scale =
          1.0F / sqrtf(static_cast<float>(WarpAllReduceSumD(ss) /
                                          static_cast<double>(hdim)) +
                       kRmsEps);
      // Q rope stores: bit-identical to the standalone kernel's contraction
      // choice at the pinned toolchain, pinned by the QkvPost unit test.
      for (std::uint32_t m = 0; m < pairs; ++m) {
        const std::uint32_t i = lane + warpSize * m;
        const float ya =
            a[m] * scale * (q_gamma != nullptr ? q_gamma[i] : 1.0F);
        const float yb =
            b[m] * scale * (q_gamma != nullptr ? q_gamma[half + i] : 1.0F);
        orow[i] = ya * cs[m] - yb * sn[m];
        orow[half + i] = ya * sn[m] + yb * cs[m];
      }
      continue;
    }
    const std::uint32_t hq = h - heads;
    float ka[kPairs];
    float kb[kPairs];
    float va[kPairs];
    float vb[kPairs];
    const float* kr = krow + hq * hdim;
    const float* vr = vrow + hq * hdim;
    __half* kc = kc0 + hq * hdim;
    __half* vc = vc0 + hq * hdim;
    double kss = 0.0;
    double vss = 0.0;
    for (std::uint32_t m = 0; m < pairs; ++m) {
      const std::uint32_t i = lane + warpSize * m;
      ka[m] = kr[i];
      kb[m] = kr[half + i];
      va[m] = vr[i];
      vb[m] = vr[half + i];
      kss += static_cast<double>(ka[m]) * ka[m] +
             static_cast<double>(kb[m]) * kb[m];
      vss += static_cast<double>(va[m]) * va[m] +
             static_cast<double>(vb[m]) * vb[m];
    }
    const float kscale =
        1.0F / sqrtf(static_cast<float>(WarpAllReduceSumD(kss) /
                                        static_cast<double>(hdim)) +
                     kRmsEps);
    const float vscale =
        1.0F / sqrtf(static_cast<float>(WarpAllReduceSumD(vss) /
                                        static_cast<double>(hdim)) +
                     kRmsEps);
    // contract(off): the RoPE store must not contract into an FMA so this
    // kernel's arithmetic stays revision-stable. The standalone kernel's
    // contraction here is compiler-chosen, so K can differ from it by one
    // half ulp on rounding ties (measured 2/81920 and 1/40960 elements);
    // the prefill validator confirms the level is numerically inconsequential.
    {
#pragma clang fp contract(off)
      for (std::uint32_t m = 0; m < pairs; ++m) {
        const std::uint32_t i = lane + warpSize * m;
        const float ya =
            ka[m] * kscale * (k_gamma != nullptr ? k_gamma[i] : 1.0F);
        const float yb =
            kb[m] * kscale * (k_gamma != nullptr ? k_gamma[half + i] : 1.0F);
        kc[i] = __float2half(ya * cs[m] - yb * sn[m]);
        kc[half + i] = __float2half(ya * sn[m] + yb * cs[m]);
        vc[i] = __float2half(va[m] * vscale);
        vc[half + i] = __float2half(vb[m] * vscale);
      }
    }
  }
}

__global__ void QknormRopeKernel(const float* q, const float* gamma, float* out,
                                 std::uint32_t heads, std::uint32_t head_dim,
                                 const float* inv_freq, std::uint32_t start,
                                 std::uint32_t pitch) {
  __shared__ float buf[512];
  __shared__ double reduce[32];
  const std::uint32_t token = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::size_t o_in = static_cast<std::size_t>(token) * pitch +
                           static_cast<std::size_t>(head) * head_dim;
  const std::size_t o =
      (static_cast<std::size_t>(token) * heads + head) * head_dim;
  NormHead(q + o_in, gamma, buf, head_dim, kRmsEps, reduce);
  __syncthreads();
  RopeHead(buf, head_dim, inv_freq, start + token);
  __syncthreads();
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    out[o + i] = buf[i];
  }
}

__global__ void KvNormRopeWriteKernel(
    const float* k, const float* v, const float* k_gamma, __half* k_cache,
    __half* v_cache, std::uint32_t kv_heads, std::uint32_t head_dim,
    const float* inv_freq, std::uint32_t start, std::uint32_t pitch) {
  __shared__ float kb[512];
  __shared__ float vb[512];
  __shared__ double reduce[32];
  const std::uint32_t token = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::size_t o = static_cast<std::size_t>(token) * pitch +
                        static_cast<std::size_t>(head) * head_dim;
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
                std::uint32_t start, std::uint32_t pitch, hipStream_t stream) {
  // One warp per token only pays off once tokens provide their own
  // parallelism; at decode widths the block kernel's per-(token, head)
  // blocks finish sooner than a single warp looping every head.
  if (tokens >= 32U && head_dim % 64U == 0U && head_dim <= 1024U) {
    QknormRopeWarpKernel<<<(tokens + 3U) / 4U, 128, 0, stream>>>(
        q, gamma, out, tokens, heads, head_dim, inv_freq, start, pitch);
  } else {
    QknormRopeKernel<<<dim3(tokens, heads), 256, 0, stream>>>(
        q, gamma, out, heads, head_dim, inv_freq, start, pitch);
  }
}

void KvNormRopeWrite(const float* k, const float* v, const float* k_gamma,
                     __half* k_cache, __half* v_cache, std::uint32_t start,
                     std::uint32_t tokens, std::uint32_t kv_heads,
                     std::uint32_t head_dim, const float* inv_freq,
                     std::uint32_t pitch, hipStream_t stream) {
  if (tokens >= 32U && head_dim % 64U == 0U && head_dim <= 1024U) {
    KvNormRopeWriteWarpKernel<<<(tokens + 3U) / 4U, 128, 0, stream>>>(
        k, v, k_gamma, k_cache, v_cache, tokens, kv_heads, head_dim, inv_freq,
        start, pitch);
  } else {
    KvNormRopeWriteKernel<<<dim3(tokens, kv_heads), 256, 0, stream>>>(
        k, v, k_gamma, k_cache, v_cache, kv_heads, head_dim, inv_freq, start,
        pitch);
  }
}

void QkvPost(const float* q, const float* q_gamma, float* q_out, const float* k,
             const float* v, const float* k_gamma, __half* k_cache,
             __half* v_cache, std::uint32_t tokens, std::uint32_t heads,
             std::uint32_t kv_heads, std::uint32_t head_dim,
             const float* inv_freq, std::uint32_t start, std::uint32_t q_pitch,
             std::uint32_t kv_pitch, hipStream_t stream) {
  if (tokens >= 32U && head_dim % 64U == 0U && head_dim <= 1024U) {
    if (head_dim == 256U) {
      QkvPostWarpKernel<256U><<<tokens, 512, 0, stream>>>(
          q, q_gamma, q_out, k, v, k_gamma, k_cache, v_cache, heads, kv_heads,
          head_dim, inv_freq, start, q_pitch, kv_pitch);
    } else if (head_dim == 512U) {
      QkvPostWarpKernel<512U><<<tokens, 512, 0, stream>>>(
          q, q_gamma, q_out, k, v, k_gamma, k_cache, v_cache, heads, kv_heads,
          head_dim, inv_freq, start, q_pitch, kv_pitch);
    } else {
      QkvPostWarpKernel<0U><<<tokens, 512, 0, stream>>>(
          q, q_gamma, q_out, k, v, k_gamma, k_cache, v_cache, heads, kv_heads,
          head_dim, inv_freq, start, q_pitch, kv_pitch);
    }
  } else {
    QknormRope(q, q_gamma, q_out, tokens, heads, head_dim, inv_freq, start,
               q_pitch, stream);
    KvNormRopeWrite(k, v, k_gamma, k_cache, v_cache, start, tokens, kv_heads,
                    head_dim, inv_freq, kv_pitch, stream);
  }
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
  // the scalar oracle. Absolute causal positions over the full F16 cache:
  // the query row at index r sits at position start + r and every cache
  // position below start + tokens is resident (ring 0, no per-row key ends,
  // the whole head dimension is cached so no RoPE derivation applies).
  AttentionArgs args{};
  args.q = q;
  args.k_cache = reinterpret_cast<const std::uint16_t*>(k_cache);
  args.v_cache = reinterpret_cast<const std::uint16_t*>(v_cache);
  args.out = out;
  args.rows = tokens;
  args.heads = heads;
  args.kv_heads = kv_heads;
  args.head_dim = head_dim;
  args.first_position = start;
  args.shared_position = false;
  args.key_limit = start + tokens;
  args.window = window;
  args.ring = 0;
  args.rope_pairs = 0;
  args.key_ends = nullptr;
  args.out_half = nullptr;
  if (LaunchWmmaPrefillAttention(args, stream)) {
    return;
  }
  AttentionPrefillScalar(q, k_cache, v_cache, out, start, tokens, heads,
                         kv_heads, head_dim, window, stream);
}

}  // namespace gufo::models::gemma4::rocm