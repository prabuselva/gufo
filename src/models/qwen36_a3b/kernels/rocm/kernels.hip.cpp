#include "src/models/qwen36_a3b/kernels/rocm/kernels.hpp"

#include <algorithm>
#include <cmath>

namespace gufo::models::qwen36_a3b::rocm {
namespace {

// Elementwise helpers mirror the scalar oracle exactly (float math, same
// formulas) so a kernel and the CPU reference agree to rounding.
__device__ __forceinline__ float DSigmoid(float x) {
  return 1.0F / (1.0F + expf(-x));
}
__device__ __forceinline__ float DSilu(float x) {
  return x * DSigmoid(x);
}
__device__ __forceinline__ float DSoftplus(float x) {
  return x > 20.0F ? x : log1pf(expf(x));
}

__device__ __forceinline__ double WarpReduceSum(double v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}
// Butterfly reduction: the sum is valid in every lane of the wave.
__device__ __forceinline__ float WarpReduceSumF(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  return v;
}
__device__ __forceinline__ double WarpReduceSumD(double v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  return v;
}

// Sum over the whole workgroup; the result is valid in thread 0's slot and
// broadcast through shared[0].
__device__ __forceinline__ double BlockReduceSum(double v, double* shared) {
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int warps = (static_cast<int>(blockDim.x) + warpSize - 1) / warpSize;
  v = WarpReduceSum(v);
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  if (warp == 0) {
    v = (lane < warps) ? shared[lane] : 0.0;
    v = WarpReduceSum(v);
    if (lane == 0) {
      shared[0] = v;
    }
  }
  __syncthreads();
  return shared[0];
}

// Maximum over the workgroup with the lowest index winning ties; broadcast
// through shared[0] (value) and shared[1] (index).
__device__ __forceinline__ void BlockReduceMaxIndex(float value, int index,
                                                    float* shared_value,
                                                    int* shared_index) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    const float ov = __shfl_down(value, offset);
    const int oi = __shfl_down(index, offset);
    if (ov > value || (ov == value && oi < index)) {
      value = ov;
      index = oi;
    }
  }
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int warps = (static_cast<int>(blockDim.x) + warpSize - 1) / warpSize;
  if (lane == 0) {
    shared_value[warp] = value;
    shared_index[warp] = index;
  }
  __syncthreads();
  if (warp == 0) {
    value = (lane < warps) ? shared_value[lane] : -INFINITY;
    index = (lane < warps) ? shared_index[lane] : 0x7FFFFFFF;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      const float ov = __shfl_down(value, offset);
      const int oi = __shfl_down(index, offset);
      if (ov > value || (ov == value && oi < index)) {
        value = ov;
        index = oi;
      }
    }
    if (lane == 0) {
      shared_value[0] = value;
      shared_index[0] = index;
    }
  }
  __syncthreads();
}

__global__ void RmsNormKernel(const float* x, const float* gamma, float* out,
                              std::uint32_t dim, float eps) {
  __shared__ double reduce[32];
  const std::size_t row = blockIdx.x;
  const float* xr = x + row * dim;
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const double v = xr[i];
    ss += v * v;
  }
  const float scale =
      1.0F / sqrtf(static_cast<float>(BlockReduceSum(ss, reduce) /
                                      static_cast<double>(dim)) +
                   eps);
  float* orow = out + row * dim;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    orow[i] = xr[i] * scale * (gamma != nullptr ? gamma[i] : 1.0F);
  }
}

__global__ void RopeKernel(float* x, const std::uint32_t* pos,
                           std::uint32_t heads, std::uint32_t head_dim,
                           std::uint32_t rotary_dim, float theta) {
  const std::uint32_t row = blockIdx.x / heads;
  const std::uint32_t head = blockIdx.x % heads;
  const std::uint32_t half = rotary_dim / 2;
  if (threadIdx.x >= half) {
    return;
  }
  const std::uint32_t i = threadIdx.x;
  const float p = static_cast<float>(pos[row]);
  const float freq = powf(
      theta, -2.0F * static_cast<float>(i) / static_cast<float>(rotary_dim));
  const float angle = p * freq;
  const float c = cosf(angle);
  const float s = sinf(angle);
  float* v = x + (static_cast<std::size_t>(row) * heads + head) * head_dim;
  const float a = v[i];
  const float b = v[i + half];
  v[i] = a * c - b * s;
  v[i + half] = a * s + b * c;
}

// One thread per output channel: deinterleave the per-head [q | gate] pair.
__global__ void SplitQGateKernel(const float* qg, float* q, float* gate,
                                 std::uint32_t heads, std::uint32_t head_dim,
                                 std::uint32_t tokens) {
  const std::size_t row = static_cast<std::size_t>(heads) * head_dim;
  const std::size_t total = static_cast<std::size_t>(tokens) * row;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const std::size_t t = idx / row;
    const std::size_t rem = idx - t * row;
    const std::size_t head = rem / head_dim;
    const std::size_t i = rem - head * head_dim;
    const std::size_t base = (t * heads + head) * 2 * head_dim;
    q[idx] = qg[base + i];
    gate[idx] = qg[base + head_dim + i];
  }
}

__global__ void SwigluKernel(float* gate, const float* up, std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    gate[i] = DSilu(gate[i]) * up[i];
  }
}

__global__ void SigmoidMulKernel(float* x, const float* g, std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    x[i] *= DSigmoid(g[i]);
  }
}

__global__ void AddKernel(float* a, const float* b, std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    a[i] += b[i];
  }
}

__global__ void RouterTopKKernel(const float* logits, std::uint32_t stride,
                                 std::int32_t* ids, float* weights,
                                 std::uint32_t n_experts, std::uint32_t k) {
  extern __shared__ float prob[];
  __shared__ double reduce[32];
  __shared__ float max_buf[32];
  __shared__ int idx_buf[32];
  const std::uint32_t token = blockIdx.x;
  const float* row = logits + static_cast<std::size_t>(token) * stride;

  float value = -INFINITY;
  if (threadIdx.x < n_experts) {
    value = row[threadIdx.x];
  }
  float max_logit = -INFINITY;
  {
    float v = value;
    int dummy = 0;
    BlockReduceMaxIndex(v, dummy, max_buf, idx_buf);
    max_logit = max_buf[0];
  }
  double sum = 0.0;
  if (threadIdx.x < n_experts) {
    prob[threadIdx.x] = expf(value - max_logit);
    sum = prob[threadIdx.x];
  }
  sum = BlockReduceSum(sum, reduce);
  if (threadIdx.x < n_experts) {
    prob[threadIdx.x] /= static_cast<float>(sum);
  }
  __syncthreads();

  float selected = 0.0F;
  for (std::uint32_t slot = 0; slot < k; ++slot) {
    float v = (threadIdx.x < n_experts) ? prob[threadIdx.x] : -INFINITY;
    const int idx =
        (threadIdx.x < n_experts) ? static_cast<int>(threadIdx.x) : 0x7FFFFFFF;
    BlockReduceMaxIndex(v, idx, max_buf, idx_buf);
    const int winner = idx_buf[0];
    if (threadIdx.x == 0) {
      ids[token * k + slot] = winner;
      weights[token * k + slot] = max_buf[0];
      selected += max_buf[0];
    }
    if (static_cast<std::uint32_t>(winner) == threadIdx.x) {
      prob[threadIdx.x] = -INFINITY;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    selected = std::max(selected, 6.103515625e-5F);
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      weights[token * k + slot] /= static_cast<float>(selected);
    }
  }
}

__global__ void MoeEpilogueKernel(const float* expert_out, const float* weights,
                                  const float* shared, const float* gate,
                                  std::uint32_t gate_stride, float* out,
                                  std::uint32_t k, std::uint32_t dim,
                                  std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    const std::size_t token = index / dim;
    const std::size_t i = index % dim;
    double acc = 0.0;
    for (std::uint32_t s = 0; s < k; ++s) {
      const std::size_t slot = token * k + s;
      acc += static_cast<double>(weights[slot]) * expert_out[slot * dim + i];
    }
    acc += static_cast<double>(DSigmoid(gate[token * gate_stride])) *
           shared[token * dim + i];
    out[index] = static_cast<float>(acc);
  }
}

__global__ void GdnConvKernel(const float* qkv, const float* conv_w,
                              float* history, float* convolved,
                              std::uint32_t channels, std::uint32_t kernel) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t ch =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       ch < channels; ch += stride) {
    const float* w = conv_w + ch * kernel;
    float acc = w[kernel - 1] * qkv[ch];
    for (std::uint32_t t = 0; t + 1 < kernel; ++t) {
      acc += w[t] * history[static_cast<std::size_t>(t) * channels + ch];
    }
    convolved[ch] = DSilu(acc);
    for (std::uint32_t t = 0; t + 2 < kernel; ++t) {
      history[static_cast<std::size_t>(t) * channels + ch] =
          history[static_cast<std::size_t>(t + 1) * channels + ch];
    }
    history[static_cast<std::size_t>(kernel - 2) * channels + ch] = qkv[ch];
  }
}

__global__ void GdnNormQkKernel(const float* convolved, float* qn, float* kn,
                                std::uint32_t k_heads, std::uint32_t head_dim,
                                float eps) {
  __shared__ double reduce[32];
  const std::uint32_t head = blockIdx.x;
  const bool is_key = blockIdx.y != 0;
  // q heads occupy [0, k_heads); k heads occupy [k_heads, 2*k_heads).
  const float* src =
      convolved +
      static_cast<std::size_t>(is_key ? k_heads + head : head) * head_dim;
  float* dst = (is_key ? kn : qn) + static_cast<std::size_t>(head) * head_dim;
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    const double x = src[i];
    ss += x * x;
  }
  const float scale =
      1.0f / sqrtf(static_cast<float>(BlockReduceSum(ss, reduce)) + eps);
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    dst[i] = src[i] * scale;
  }
}

__global__ void GdnDeltaKernel(const float* qn, const float* kn, const float* v,
                               const float* alpha, const float* beta,
                               const float* a, const float* dt, float* state,
                               float* attn, std::uint32_t k_heads,
                               std::uint32_t head_dim) {
  extern __shared__ float smem[];
  float* sq = smem;
  float* sk = smem + head_dim;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t kh = h % k_heads;
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    sq[i] = qn[kh * head_dim + i];
    sk[i] = kn[kh * head_dim + i];
  }
  __syncthreads();
  const float decay = expf(a[h] * DSoftplus(alpha[h] + dt[h]));
  const float b = DSigmoid(beta[h]);
  const float q_scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  const std::uint32_t j = threadIdx.x;
  if (j >= head_dim) {
    return;
  }
  float* row = state + static_cast<std::size_t>(h) * head_dim * head_dim +
               static_cast<std::size_t>(j) * head_dim;
  double acc = 0.0;
  for (std::uint32_t i = 0; i < head_dim; ++i) {
    const float s = row[i] * decay;
    row[i] = s;
    acc += static_cast<double>(s) * sk[i];
  }
  const float u = static_cast<float>(acc);
  const float delta = (v[h * head_dim + j] - u) * b;
  acc = 0.0;
  for (std::uint32_t i = 0; i < head_dim; ++i) {
    const float s = row[i] + delta * sk[i];
    row[i] = s;
    acc += static_cast<double>(s) * sq[i];
  }
  attn[h * head_dim + j] = static_cast<float>(acc) * q_scale;
}

// Decode (single token) delta rule: one warp per (head, state row), the lane
// owns one float4 of the row. The state is read and written exactly once with
// fully coalesced 16 B lanes, and the q dot is folded through dot(k, q) so
// only the k dot needs a wave reduction before the row update.
__global__ void GdnDeltaDecodeKernel(const float* qn, const float* kn,
                                     const float* v, const float* alpha,
                                     const float* beta, const float* a,
                                     const float* dt, float* state, float* attn,
                                     std::uint32_t k_heads,
                                     std::uint32_t head_dim) {
  const std::uint32_t warp = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t j = blockIdx.x * 4U + warp;
  const std::uint32_t kh = h % k_heads;
  const float4* __restrict__ q4 =
      reinterpret_cast<const float4*>(qn) + kh * (head_dim / 4U);
  const float4* __restrict__ k4 =
      reinterpret_cast<const float4*>(kn) + kh * (head_dim / 4U);
  const float4 qv = q4[lane];
  const float4 kv = k4[lane];
  const float decay = expf(a[h] * DSoftplus(alpha[h] + dt[h]));
  const float b = DSigmoid(beta[h]);
  float4* __restrict__ r4 =
      reinterpret_cast<float4*>(state +
                                (static_cast<std::size_t>(h) * head_dim + j) *
                                    head_dim) +
      lane;
  float4 s = *r4;
  s.x *= decay;
  s.y *= decay;
  s.z *= decay;
  s.w *= decay;
  double du = static_cast<double>(s.x) * kv.x + static_cast<double>(s.y) * kv.y +
              static_cast<double>(s.z) * kv.z + static_cast<double>(s.w) * kv.w;
  du = WarpReduceSumD(du);
  const float delta = (v[h * head_dim + j] - static_cast<float>(du)) * b;
  float4 o;
  o.x = s.x + delta * kv.x;
  o.y = s.y + delta * kv.y;
  o.z = s.z + delta * kv.z;
  o.w = s.w + delta * kv.w;
  const double dq = static_cast<double>(o.x) * qv.x +
                    static_cast<double>(o.y) * qv.y +
                    static_cast<double>(o.z) * qv.z +
                    static_cast<double>(o.w) * qv.w;
  const float q_scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  attn[h * head_dim + j] = static_cast<float>(WarpReduceSumD(dq)) * q_scale;
  *r4 = o;
}

__global__ void GdnOutNormKernel(float* attn, const float* z,
                                 const float* norm_w, std::uint32_t head_dim,
                                 float eps) {
  __shared__ double reduce[32];
  const std::uint32_t h = blockIdx.x;
  float* row = attn + static_cast<std::size_t>(h) * head_dim;
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    const double x = row[i];
    ss += x * x;
  }
  const float scale =
      1.0F / sqrtf(static_cast<float>(BlockReduceSum(ss, reduce) /
                                      static_cast<double>(head_dim)) +
                   eps);
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    row[i] = row[i] * scale * norm_w[i] * DSilu(z[h * head_dim + i]);
  }
}

__global__ void AttentionDecodeKernel(const float* q, const float* k_cache,
                                      const float* v_cache, const float* gate,
                                      float* out, float* scratch,
                                      std::uint32_t n_kv, std::uint32_t heads,
                                      std::uint32_t kv_heads,
                                      std::uint32_t head_dim, float scale) {
  __shared__ double reduce[32];
  __shared__ float max_buf[32];
  __shared__ int idx_buf[32];
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t kvh = h / (heads / kv_heads);
  const float* qh = q + static_cast<std::size_t>(h) * head_dim;
  float* p = scratch + static_cast<std::size_t>(h) * n_kv;

  // Pass 1: scores and their maximum.
  float local_max = -INFINITY;
  for (std::uint32_t j = threadIdx.x; j < n_kv; j += blockDim.x) {
    const float* kj =
        k_cache + (static_cast<std::size_t>(j) * kv_heads + kvh) * head_dim;
    double dot = 0.0;
    for (std::uint32_t i = 0; i < head_dim; ++i) {
      dot += static_cast<double>(qh[i]) * kj[i];
    }
    const float score = static_cast<float>(dot) * scale;
    p[j] = score;
    local_max = fmaxf(local_max, score);
  }
  {
    float v = local_max;
    int dummy = 0;
    BlockReduceMaxIndex(v, dummy, max_buf, idx_buf);
    local_max = max_buf[0];
  }
  __syncthreads();

  // Pass 2: exponentials and their sum.
  double local_sum = 0.0;
  for (std::uint32_t j = threadIdx.x; j < n_kv; j += blockDim.x) {
    const float e = expf(p[j] - local_max);
    p[j] = e;
    local_sum += e;
  }
  const double sum = BlockReduceSum(local_sum, reduce);
  __syncthreads();

  // Pass 3: normalize, then accumulate the value mixture per output lane.
  const float inv = 1.0F / static_cast<float>(sum);
  const std::uint32_t i = threadIdx.x;
  double acc = 0.0;
  if (i < head_dim) {
    for (std::uint32_t j = 0; j < n_kv; ++j) {
      const float* vj =
          v_cache + (static_cast<std::size_t>(j) * kv_heads + kvh) * head_dim;
      acc += static_cast<double>(p[j] * inv) * vj[i];
    }
    out[static_cast<std::size_t>(h) * head_dim + i] =
        static_cast<float>(acc) *
        DSigmoid(gate[static_cast<std::size_t>(h) * head_dim + i]);
  }
}

// Flash-decoding split pass for head_dim == 256. Grid (heads, splits): each
// block owns a contiguous chunk of the KV range and one query head. A wave
// processes one token at a time — lane l holds element l + 32m of the query
// row in registers, so every K/V load is a coalesced 128 B wavefront — and
// keeps an online (max, sum, weighted-V) triple. The eight waves fold into
// per-block partials written to `part` as [head][split][head_dim + 2]
// (accumulator, max, sum); AttentionDecodeCombineKernel folds the splits.
__global__ void AttentionDecodeSplitKernel(
    const float* q, const float* k_cache, const float* v_cache, float* part,
    std::uint32_t n_kv, std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, float scale, std::uint32_t splits,
    std::uint32_t chunk) {
  constexpr std::uint32_t kWaves = 8;
  __shared__ float s_acc[kWaves][256];
  __shared__ float s_max[kWaves];
  __shared__ float s_sum[kWaves];
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t sp = blockIdx.y;
  const std::uint32_t kvh = h / (heads / kv_heads);
  const float* qh = q + static_cast<std::size_t>(h) * head_dim;
  const std::size_t kv_stride = static_cast<std::size_t>(kv_heads) * head_dim;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = threadIdx.x >> 5;
  float qv[8];
#pragma unroll
  for (std::uint32_t m = 0; m < 8U; ++m) {
    qv[m] = qh[lane + 32U * m];
  }
  float wmax = -INFINITY;
  float wsum = 0.0F;
  float acc[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  const std::uint32_t j0 = sp * chunk;
  const std::uint32_t j1 = (n_kv < j0 + chunk) ? n_kv : j0 + chunk;
  for (std::uint32_t j = j0 + wave; j < j1; j += kWaves) {
    const float* kj =
        k_cache + static_cast<std::size_t>(j) * kv_stride + kvh * head_dim;
    float dot = 0.0F;
#pragma unroll
    for (std::uint32_t m = 0; m < 8U; ++m) {
      dot += qv[m] * kj[lane + 32U * m];
    }
    const float score = WarpReduceSumF(dot) * scale;
    const float nm = fmaxf(wmax, score);
    const float r = expf(wmax - nm);
    const float e = expf(score - nm);
    wsum = wsum * r + e;
    const float* vj =
        v_cache + static_cast<std::size_t>(j) * kv_stride + kvh * head_dim;
#pragma unroll
    for (std::uint32_t m = 0; m < 8U; ++m) {
      acc[m] = acc[m] * r + e * vj[lane + 32U * m];
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
  const float r = expf(wmax - bmax);
#pragma unroll
  for (std::uint32_t m = 0; m < 8U; ++m) {
    s_acc[wave][lane + 32U * m] = acc[m] * r;
  }
  __syncthreads();
  float val = 0.0F;
#pragma unroll
  for (std::uint32_t w = 0; w < kWaves; ++w) {
    val += s_acc[w][threadIdx.x];
  }
  float* pp =
      part + (static_cast<std::size_t>(h) * splits + sp) * (head_dim + 2U);
  pp[threadIdx.x] = val;
  if (threadIdx.x == 0) {
    float total = 0.0F;
#pragma unroll
    for (std::uint32_t w = 0; w < kWaves; ++w) {
      total += s_sum[w] * expf(s_max[w] - bmax);
    }
    pp[head_dim] = bmax;
    pp[head_dim + 1U] = total;
  }
}

// Folds the AttentionDecodeSplitKernel partials per head and applies the
// sigmoid output gate.
__global__ void AttentionDecodeCombineKernel(const float* part,
                                             const float* gate, float* out,
                                             std::uint32_t heads,
                                             std::uint32_t splits,
                                             std::uint32_t head_dim) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t i = threadIdx.x;
  const float* pp = part + static_cast<std::size_t>(h) * splits * (head_dim + 2U);
  float m = -INFINITY;
  for (std::uint32_t s = 0; s < splits; ++s) {
    m = fmaxf(m, pp[static_cast<std::size_t>(s) * (head_dim + 2U) + head_dim]);
  }
  float num = 0.0F;
  float den = 0.0F;
  for (std::uint32_t s = 0; s < splits; ++s) {
    const float w =
        expf(pp[static_cast<std::size_t>(s) * (head_dim + 2U) + head_dim] - m);
    num += pp[static_cast<std::size_t>(s) * (head_dim + 2U) + i] * w;
    den += pp[static_cast<std::size_t>(s) * (head_dim + 2U) + head_dim + 1U] * w;
  }
  out[static_cast<std::size_t>(h) * head_dim + i] =
      (num / den) * DSigmoid(gate[static_cast<std::size_t>(h) * head_dim + i]);
}

// Fallback for head_dim > 256: one workgroup per (token, head). A tiled
// online softmax over the causal range [0, start + token] keeps only one tile
// of scores in shared memory, so the resident footprint does not grow with
// the context. Accumulation matches AttentionDecodeKernel (double sums, same
// sigmoid gate).
__global__ void AttentionPrefillNaive(const float* q, const float* k_cache,
                                      const float* v_cache, const float* gate,
                                      float* out, std::uint32_t start,
                                      std::uint32_t heads,
                                      std::uint32_t kv_heads,
                                      std::uint32_t head_dim, float scale) {
  constexpr std::uint32_t kTile = 128;
  __shared__ float score[kTile];
  __shared__ double reduce[32];
  __shared__ float max_buf[32];
  __shared__ int idx_buf[32];

  const std::uint32_t h = blockIdx.x % heads;
  const std::uint32_t token = blockIdx.x / heads;
  const std::uint32_t kvh = h / (heads / kv_heads);
  const float* qh =
      q + (static_cast<std::size_t>(token) * heads + h) * head_dim;
  const std::uint32_t causal = start + token + 1;

  const std::uint32_t i = threadIdx.x;
  double acc = 0.0;
  double run_sum = 0.0;
  float run_max = -INFINITY;

  for (std::uint32_t base = 0; base < causal; base += kTile) {
    const std::uint32_t span = causal - base;
    const std::uint32_t len = span < kTile ? span : kTile;
    __syncthreads();
    for (std::uint32_t t = threadIdx.x; t < len; t += blockDim.x) {
      const std::uint32_t j = base + t;
      const float* kj =
          k_cache + (static_cast<std::size_t>(j) * kv_heads + kvh) * head_dim;
      double dot = 0.0;
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        dot += static_cast<double>(qh[d]) * kj[d];
      }
      score[t] = static_cast<float>(dot) * scale;
    }
    __syncthreads();

    float tile_max = -INFINITY;
    for (std::uint32_t t = threadIdx.x; t < len; t += blockDim.x) {
      tile_max = fmaxf(tile_max, score[t]);
    }
    {
      int dummy = 0;
      BlockReduceMaxIndex(tile_max, dummy, max_buf, idx_buf);
      tile_max = max_buf[0];
    }
    const float new_max = fmaxf(run_max, tile_max);
    const float corr = expf(run_max - new_max);
    run_sum *= corr;
    if (i < head_dim) {
      acc *= corr;
    }

    double tile_sum = 0.0;
    for (std::uint32_t t = threadIdx.x; t < len; t += blockDim.x) {
      const float e = expf(score[t] - new_max);
      score[t] = e;
      tile_sum += e;
    }
    run_sum += BlockReduceSum(tile_sum, reduce);

    if (i < head_dim) {
      for (std::uint32_t t = 0; t < len; ++t) {
        const float* vj = v_cache +
                          (static_cast<std::size_t>(base + t) * kv_heads + kvh) *
                              head_dim;
        acc += static_cast<double>(score[t]) * vj[i];
      }
    }
    run_max = new_max;
  }

  const float inv = 1.0F / static_cast<float>(run_sum);
  if (i < head_dim) {
    const std::size_t o =
        (static_cast<std::size_t>(token) * heads + h) * head_dim + i;
    out[o] = static_cast<float>(acc * inv) * DSigmoid(gate[o]);
  }
}

// Tiled causal attention for head_dim <= 256. One workgroup handles kQT
// queries of one head and streams KV tiles through shared memory, so each
// K/V element is read once per kQT queries instead of once per query. The
// online softmax keeps a running max/sum per query; the score tile is
// exponentiated in place and consumed by the value mix.
__global__ void AttentionPrefillTiled(const float* q, const float* k_cache,
                                      const float* v_cache, const float* gate,
                                      float* out, std::uint32_t start,
                                      std::uint32_t tokens,
                                      std::uint32_t heads,
                                      std::uint32_t kv_heads,
                                      std::uint32_t head_dim, float scale) {
  constexpr std::uint32_t kQT = 32;
  constexpr std::uint32_t kKT = 8;
  constexpr std::uint32_t kMaxHD = 256;
  constexpr std::uint32_t kStride = kMaxHD + 1;
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

  if (tid < kQT) {
    rmax[tid] = -INFINITY;
    rsum[tid] = 0.0F;
  }

  const std::uint32_t q_last =
      q0 + kQT < tokens ? q0 + kQT : tokens;
  const std::uint32_t causal_max = start + q_last;
  float acc[kQT];
  for (std::uint32_t i = 0; i < kQT; ++i) {
    acc[i] = 0.0F;
  }

  for (std::uint32_t base = 0; base < causal_max; base += kKT) {
    __syncthreads();
    for (std::uint32_t idx = tid; idx < kKT * head_dim; idx += blockDim.x) {
      const std::uint32_t k = idx / head_dim;
      const std::uint32_t d = idx % head_dim;
      const std::uint32_t j = base + k;
      const std::size_t o =
          (static_cast<std::size_t>(j) * kv_heads + kvh) * head_dim + d;
      const bool valid = j < causal_max;
      sK[k][d] = valid ? k_cache[o] : 0.0F;
      sV[k][d] = valid ? v_cache[o] : 0.0F;
    }
    __syncthreads();

    // Scores: one thread per (query, key) pair, causal mask as -inf.
    for (std::uint32_t p = tid; p < kQT * kKT; p += blockDim.x) {
      const std::uint32_t qi = p / kKT;
      const std::uint32_t k = p % kKT;
      const std::uint32_t token = q0 + qi;
      if (token >= tokens || base + k > start + token) {
        sScore[qi][k] = -INFINITY;
        continue;
      }
      const float* qh =
          q + (static_cast<std::size_t>(token) * heads + h) * head_dim;
      float dot = 0.0F;
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        dot += qh[d] * sK[k][d];
      }
      sScore[qi][k] = dot * scale;
    }
    __syncthreads();

    if (tid < kQT) {
      float tile_max = -INFINITY;
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

    if (tid < head_dim) {
      for (std::uint32_t qi = 0; qi < kQT; ++qi) {
        acc[qi] *= corr[qi];
      }
      for (std::uint32_t k = 0; k < kKT; ++k) {
        const float sv = sV[k][tid];
        for (std::uint32_t qi = 0; qi < kQT; ++qi) {
          acc[qi] += sScore[qi][k] * sv;
        }
      }
    }
  }

  if (tid < head_dim) {
    for (std::uint32_t qi = 0; qi < kQT; ++qi) {
      const std::uint32_t token = q0 + qi;
      if (token >= tokens) {
        continue;
      }
      const std::size_t o =
          (static_cast<std::size_t>(token) * heads + h) * head_dim + tid;
      out[o] = (acc[qi] / rsum[qi]) * DSigmoid(gate[o]);
    }
  }
}

// Causal depthwise convolution over a chunk, SiLU applied. Token t reads the
// `kernel` inputs ending at t; those before the chunk come from `history`
// ([kernel-1][channels], oldest first). Accumulation order matches the decode
// kernel (newest tap first) so a chunk reproduces per-token decode.
__global__ void GdnConvPrefillKernel(const float* qkv, const float* conv_w,
                                     const float* history, float* convolved,
                                     std::uint32_t tokens,
                                     std::uint32_t channels,
                                     std::uint32_t kernel) {
  const std::size_t total = static_cast<std::size_t>(tokens) * channels;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const std::uint32_t t = static_cast<std::uint32_t>(idx / channels);
    const std::uint32_t ch = static_cast<std::uint32_t>(idx % channels);
    const float* w = conv_w + static_cast<std::size_t>(ch) * kernel;
    float acc = w[kernel - 1] * qkv[idx];
    for (std::uint32_t k = 0; k + 1 < kernel; ++k) {
      const std::int64_t s = static_cast<std::int64_t>(t) -
                             static_cast<std::int64_t>(kernel - 1) +
                             static_cast<std::int64_t>(k);
      const float xv = s >= 0
                           ? qkv[static_cast<std::size_t>(s) * channels + ch]
                           : history[static_cast<std::size_t>(
                                 static_cast<std::int64_t>(kernel - 1) + s) *
                                 channels +
                             ch];
      acc += w[k] * xv;
    }
    convolved[idx] = DSilu(acc);
  }
}

// Advances the rolling convolution state past a chunk: new row j is the input
// at absolute position (chunk_end - (kernel-1) + j). Reads the old `history`
// and the chunk `qkv`, writes a disjoint `out` so there is no aliasing.
__global__ void GdnHistoryUpdateKernel(const float* qkv, const float* history,
                                       float* out, std::uint32_t tokens,
                                       std::uint32_t channels,
                                       std::uint32_t kernel) {
  const std::size_t total =
      static_cast<std::size_t>(kernel - 1) * channels;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const std::uint32_t j = static_cast<std::uint32_t>(idx / channels);
    const std::uint32_t ch = static_cast<std::uint32_t>(idx % channels);
    const std::int64_t src = static_cast<std::int64_t>(tokens) -
                             static_cast<std::int64_t>(kernel - 1) +
                             static_cast<std::int64_t>(j);
    out[idx] = src >= 0 ? qkv[static_cast<std::size_t>(src) * channels + ch]
                        : history[static_cast<std::size_t>(
                              static_cast<std::int64_t>(kernel - 1) + src) *
                              channels +
                          ch];
  }
}

// Per-token RMS norm of the q and k heads, batched over the chunk.
__global__ void GdnNormQkPrefillKernel(const float* convolved, float* qn,
                                       float* kn, std::uint32_t tokens,
                                       std::uint32_t k_heads,
                                       std::uint32_t channels,
                                       std::uint32_t head_dim, float eps) {
  __shared__ double reduce[32];
  const std::uint32_t head = blockIdx.x % k_heads;
  const std::uint32_t token = blockIdx.x / k_heads;
  const bool is_key = blockIdx.y != 0;
  const float* src =
      convolved +
      static_cast<std::size_t>(token) * channels +
      static_cast<std::size_t>(is_key ? k_heads + head : head) * head_dim;
  float* dst = (is_key ? kn : qn) +
               (static_cast<std::size_t>(token) * k_heads + head) * head_dim;
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    const double x = src[i];
    ss += x * x;
  }
  const float scale =
      1.0f / sqrtf(static_cast<float>(BlockReduceSum(ss, reduce)) + eps);
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    dst[i] = src[i] * scale;
  }
}

// Gated delta-rule recurrence over a chunk, sequential over tokens but fully
// on-device. One warp owns kRows state rows of one value head; each lane
// keeps kCols state columns (head_dim = 64 * kCols) in registers across the
// whole chunk, so the per-token dot products reduce with warp shuffles only:
// no shared memory and no barriers on the token recurrence. o[r] reuses
// dot(s_new[r], q) = oq[r] + delta[r] * dot(k, q), and the next token's q/k
// elements are prefetched while the current one reduces.
template <std::uint32_t kRows, std::uint32_t kCols>
__global__ void GdnDeltaLoopKernel(const float* qn, const float* kn,
                                   const float* convolved, const float* alpha,
                                   const float* beta, const float* a,
                                   const float* dt, float* state, float* attn,
                                   std::uint32_t tokens, std::uint32_t k_heads,
                                   std::uint32_t v_heads,
                                   std::uint32_t head_dim,
                                   std::uint32_t channels) {
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t kh = h % k_heads;
  const std::uint32_t j0 = blockIdx.x * kRows;
  const std::uint32_t lane = threadIdx.x;
  const std::size_t attn_stride =
      static_cast<std::size_t>(v_heads) * head_dim;
  const std::size_t v_base = 2 * static_cast<std::size_t>(k_heads) * head_dim +
                             static_cast<std::size_t>(h) * head_dim;
  const float q_scale = 1.0F / sqrtf(static_cast<float>(head_dim));

  float s[kRows][kCols];
  for (std::uint32_t r = 0; r < kRows; ++r) {
    const float* row =
        state + (static_cast<std::size_t>(h) * head_dim + j0 + r) * head_dim;
    for (std::uint32_t c = 0; c < kCols; ++c) {
      s[r][c] = row[lane + c * blockDim.x];
    }
  }

  float kc[kCols];
  float qc[kCols];
  {
    const std::size_t off = static_cast<std::size_t>(kh) * head_dim + lane;
    for (std::uint32_t c = 0; c < kCols; ++c) {
      kc[c] = kn[off + c * blockDim.x];
      qc[c] = qn[off + c * blockDim.x];
    }
  }

  for (std::uint32_t t = 0; t < tokens; ++t) {
    float knx[kCols];
    float qnx[kCols];
    if (t + 1 < tokens) {
      const std::size_t off = (static_cast<std::size_t>(t + 1) * k_heads +
                               kh) * head_dim + lane;
      for (std::uint32_t c = 0; c < kCols; ++c) {
        knx[c] = kn[off + c * blockDim.x];
        qnx[c] = qn[off + c * blockDim.x];
      }
    }
    const float decay =
        expf(a[h] * DSoftplus(alpha[static_cast<std::size_t>(t) * v_heads + h] +
                              dt[h]));
    const float b = DSigmoid(beta[static_cast<std::size_t>(t) * v_heads + h]);
    float pu[kRows];
    float po[kRows];
    for (std::uint32_t r = 0; r < kRows; ++r) {
      float du = 0.0F;
      float dq = 0.0F;
      for (std::uint32_t c = 0; c < kCols; ++c) {
        s[r][c] *= decay;
        du += s[r][c] * kc[c];
        dq += s[r][c] * qc[c];
      }
      pu[r] = WarpReduceSumF(du);
      po[r] = WarpReduceSumF(dq);
    }
    float kq = 0.0F;
    for (std::uint32_t c = 0; c < kCols; ++c) {
      kq += kc[c] * qc[c];
    }
    kq = WarpReduceSumF(kq);
    for (std::uint32_t r = 0; r < kRows; ++r) {
      const float v_t =
          convolved[static_cast<std::size_t>(t) * channels + v_base + j0 + r];
      const float delta = (v_t - pu[r]) * b;
      for (std::uint32_t c = 0; c < kCols; ++c) {
        s[r][c] += delta * kc[c];
      }
      if (lane == r) {
        attn[static_cast<std::size_t>(t) * attn_stride +
             (static_cast<std::size_t>(h) * head_dim + j0 + r)] =
            (po[r] + delta * kq) * q_scale;
      }
    }
    for (std::uint32_t c = 0; c < kCols; ++c) {
      kc[c] = knx[c];
      qc[c] = qnx[c];
    }
  }
  for (std::uint32_t r = 0; r < kRows; ++r) {
    float* row =
        state + (static_cast<std::size_t>(h) * head_dim + j0 + r) * head_dim;
    for (std::uint32_t c = 0; c < kCols; ++c) {
      row[lane + c * blockDim.x] = s[r][c];
    }
  }
}

// Per-token gated RMS norm of the attention output, batched over the chunk.
__global__ void GdnOutNormPrefillKernel(float* attn, const float* z,
                                        const float* norm_w,
                                        std::uint32_t v_heads,
                                        std::uint32_t head_dim, float eps) {
  __shared__ double reduce[32];
  const std::uint32_t h = blockIdx.x % v_heads;
  const std::uint32_t token = blockIdx.x / v_heads;
  const std::size_t off =
      (static_cast<std::size_t>(token) * v_heads + h) * head_dim;
  float* row = attn + off;
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    const double x = row[i];
    ss += x * x;
  }
  const float scale =
      1.0F / sqrtf(static_cast<float>(BlockReduceSum(ss, reduce) /
                                      static_cast<double>(head_dim)) +
                   eps);
  for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    row[i] = row[i] * scale * norm_w[i] * DSilu(z[off + i]);
  }
}

}  // namespace

void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t rows, std::uint32_t dim, float eps,
                 hipStream_t stream) {
  const dim3 block(std::min<std::uint32_t>(dim, 256U));
  RmsNormKernel<<<rows, block, 0, stream>>>(x, gamma, out, dim, eps);
}

void Rope(float* x, const std::uint32_t* pos, std::uint32_t rows,
          std::uint32_t heads, std::uint32_t head_dim, std::uint32_t rotary_dim,
          float theta, hipStream_t stream) {
  const dim3 grid(rows * heads);
  const dim3 block(rotary_dim / 2);
  RopeKernel<<<grid, block, 0, stream>>>(x, pos, heads, head_dim, rotary_dim,
                                         theta);
}

void SplitQGate(const float* qg, float* q, float* gate, std::uint32_t heads,
                std::uint32_t head_dim, std::uint32_t tokens,
                hipStream_t stream) {
  const std::size_t count = static_cast<std::size_t>(tokens) * heads * head_dim;
  const std::size_t block = 256;
  const std::size_t grid =
      std::min<std::size_t>((count + block - 1) / block, 65535);
  SplitQGateKernel<<<grid, block, 0, stream>>>(qg, q, gate, heads, head_dim,
                                               tokens);
}

void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream) {
  if (count == 0) {
    return;
  }
  const std::size_t block = 256;
  const std::size_t grid =
      std::min<std::size_t>((count + block - 1) / block, 65535);
  SwigluKernel<<<grid, block, 0, stream>>>(gate, up, count);
}

void SigmoidMul(float* x, const float* g, std::size_t count,
                hipStream_t stream) {
  if (count == 0) {
    return;
  }
  const std::size_t block = 256;
  const std::size_t grid =
      std::min<std::size_t>((count + block - 1) / block, 65535);
  SigmoidMulKernel<<<grid, block, 0, stream>>>(x, g, count);
}

void Add(float* a, const float* b, std::size_t count, hipStream_t stream) {
  if (count == 0) {
    return;
  }
  const std::size_t block = 256;
  const std::size_t grid =
      std::min<std::size_t>((count + block - 1) / block, 65535);
  AddKernel<<<grid, block, 0, stream>>>(a, b, count);
}

void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream) {
  const dim3 block(std::max<std::uint32_t>(n_experts, 32U));
  const std::size_t shared = n_experts * sizeof(float);
  RouterTopKKernel<<<tokens, block, shared, stream>>>(logits, stride, ids,
                                                      weights, n_experts, k);
}

__global__ void ExpertCountsKernel(const std::int32_t* ids,
                                   std::uint32_t* counts, std::size_t slots) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < slots && ids[i] >= 0) {
    atomicAdd(counts + ids[i], 1u);
  }
}

void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream) {
  (void)hipMemsetAsync(counts, 0, n_experts * sizeof(std::uint32_t), stream);
  const std::size_t slots = static_cast<std::size_t>(tokens) * k;
  const std::size_t block = 256;
  const std::size_t grid = (slots + block - 1) / block;
  ExpertCountsKernel<<<grid, block, 0, stream>>>(ids, counts, slots);
}

void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream) {
  const std::size_t count = static_cast<std::size_t>(tokens) * dim;
  const std::size_t block = 256;
  const std::size_t grid = (count + block - 1) / block;
  MoeEpilogueKernel<<<grid, block, 0, stream>>>(
      expert_out, weights, shared, gate, gate_stride, out, k, dim, count);
}

void GdnConv(const float* qkv, const float* conv_w, float* history,
             float* convolved, std::uint32_t channels, std::uint32_t kernel,
             hipStream_t stream) {
  const std::size_t block = 256;
  const std::size_t grid =
      std::min<std::size_t>((channels + block - 1) / block, 65535);
  GdnConvKernel<<<grid, block, 0, stream>>>(qkv, conv_w, history, convolved,
                                            channels, kernel);
}

void GdnNormQk(const float* convolved, float* qn, float* kn,
               std::uint32_t k_heads, std::uint32_t head_dim, float eps,
               hipStream_t stream) {
  const dim3 grid(k_heads, 2);
  const dim3 block(head_dim);
  GdnNormQkKernel<<<grid, block, 0, stream>>>(convolved, qn, kn, k_heads,
                                              head_dim, eps);
}

void GdnDelta(const float* qn, const float* kn, const float* v,
              const float* alpha, const float* beta, const float* a,
              const float* dt, float* state, float* attn, std::uint32_t k_heads,
              std::uint32_t v_heads, std::uint32_t head_dim,
              hipStream_t stream) {
  if (head_dim == 128U) {
    // One warp per (head, row): 4 rows per 128-thread block.
    const dim3 grid(head_dim / 4U, v_heads);
    GdnDeltaDecodeKernel<<<grid, dim3(128), 0, stream>>>(
        qn, kn, v, alpha, beta, a, dt, state, attn, k_heads, head_dim);
    return;
  }
  const dim3 block(head_dim);
  const std::size_t shared = 2 * head_dim * sizeof(float);
  GdnDeltaKernel<<<v_heads, block, shared, stream>>>(
      qn, kn, v, alpha, beta, a, dt, state, attn, k_heads, head_dim);
}

void GdnOutNorm(float* attn, const float* z, const float* norm_w,
                std::uint32_t v_heads, std::uint32_t head_dim, float eps,
                hipStream_t stream) {
  const dim3 block(head_dim);
  GdnOutNormKernel<<<v_heads, block, 0, stream>>>(attn, z, norm_w, head_dim,
                                                  eps);
}

void GdnConvPrefill(const float* qkv, const float* conv_w, const float* history,
                    float* convolved, std::uint32_t tokens,
                    std::uint32_t channels, std::uint32_t kernel,
                    hipStream_t stream) {
  const std::size_t block = 256;
  const std::size_t total = static_cast<std::size_t>(tokens) * channels;
  const std::size_t grid =
      std::min<std::size_t>((total + block - 1) / block, 65535);
  GdnConvPrefillKernel<<<grid, block, 0, stream>>>(qkv, conv_w, history,
                                                   convolved, tokens, channels,
                                                   kernel);
}

void GdnHistoryUpdate(const float* qkv, const float* history, float* out,
                      std::uint32_t tokens, std::uint32_t channels,
                      std::uint32_t kernel, hipStream_t stream) {
  const std::size_t block = 256;
  const std::size_t total = static_cast<std::size_t>(kernel - 1) * channels;
  const std::size_t grid =
      std::min<std::size_t>((total + block - 1) / block, 65535);
  GdnHistoryUpdateKernel<<<grid, block, 0, stream>>>(qkv, history, out, tokens,
                                                     channels, kernel);
}

void GdnNormQkPrefill(const float* convolved, float* qn, float* kn,
                      std::uint32_t tokens, std::uint32_t k_heads,
                      std::uint32_t channels, std::uint32_t head_dim, float eps,
                      hipStream_t stream) {
  const dim3 grid(static_cast<std::uint32_t>(tokens) * k_heads, 2);
  const dim3 block(head_dim);
  GdnNormQkPrefillKernel<<<grid, block, 0, stream>>>(convolved, qn, kn, tokens,
                                                     k_heads, channels,
                                                     head_dim, eps);
}

void GdnDeltaLoop(const float* qn, const float* kn, const float* convolved,
                  const float* alpha, const float* beta, const float* a,
                  const float* dt, float* state, float* attn,
                  std::uint32_t tokens, std::uint32_t k_heads,
                  std::uint32_t v_heads, std::uint32_t head_dim,
                  std::uint32_t channels, hipStream_t stream) {
  static const std::uint32_t warp = [] {
    int ws = 64;
    hipDeviceGetAttribute(&ws, hipDeviceAttributeWarpSize, 0);
    return static_cast<std::uint32_t>(ws);
  }();
  constexpr std::uint32_t kRows = 8;
  const dim3 grid(head_dim / kRows, v_heads);
  const dim3 block(warp);
  switch (head_dim / warp) {
    case 8:
      GdnDeltaLoopKernel<kRows, 8><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels);
      break;
    case 7:
      GdnDeltaLoopKernel<kRows, 7><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels);
      break;
    case 6:
      GdnDeltaLoopKernel<kRows, 6><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels);
      break;
    case 5:
      GdnDeltaLoopKernel<kRows, 5><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels);
      break;
    case 4:
      GdnDeltaLoopKernel<kRows, 4><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels);
      break;
    case 3:
      GdnDeltaLoopKernel<kRows, 3><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels);
      break;
    case 2:
      GdnDeltaLoopKernel<kRows, 2><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels);
      break;
    default:
      GdnDeltaLoopKernel<kRows, 1><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels);
      break;
  }
}

void GdnOutNormPrefill(float* attn, const float* z, const float* norm_w,
                       std::uint32_t tokens, std::uint32_t v_heads,
                       std::uint32_t head_dim, float eps, hipStream_t stream) {
  const dim3 grid(static_cast<std::uint32_t>(tokens) * v_heads);
  const dim3 block(head_dim);
  GdnOutNormPrefillKernel<<<grid, block, 0, stream>>>(attn, z, norm_w, v_heads,
                                                      head_dim, eps);
}

void AttentionDecode(const float* q, const float* k_cache, const float* v_cache,
                     const float* gate, float* out, float* scratch,
                     float* part, std::uint32_t n_kv, std::uint32_t heads,
                     std::uint32_t kv_heads, std::uint32_t head_dim,
                     float scale, hipStream_t stream) {
  if (head_dim == 256U && n_kv > 0U) {
    std::uint32_t splits = (n_kv + 63U) / 64U;
    if (splits > 32U) {
      splits = 32U;
    }
    std::uint32_t chunk = (n_kv + splits - 1U) / splits;
    splits = (n_kv + chunk - 1U) / chunk;
    AttentionDecodeSplitKernel<<<dim3(heads, splits), 256, 0, stream>>>(
        q, k_cache, v_cache, part, n_kv, heads, kv_heads, head_dim, scale,
        splits, chunk);
    AttentionDecodeCombineKernel<<<heads, head_dim, 0, stream>>>(
        part, gate, out, heads, splits, head_dim);
    return;
  }
  const dim3 block(head_dim);
  AttentionDecodeKernel<<<heads, block, 0, stream>>>(q, k_cache, v_cache, gate,
                                                     out, scratch, n_kv, heads,
                                                     kv_heads, head_dim, scale);
}

void AttentionPrefill(const float* q, const float* k_cache,
                      const float* v_cache, const float* gate, float* out,
                      std::uint32_t start, std::uint32_t tokens,
                      std::uint32_t heads, std::uint32_t kv_heads,
                      std::uint32_t head_dim, float scale, hipStream_t stream) {
  if (head_dim <= 256U) {
    constexpr std::uint32_t kQT = 32;
    const dim3 grid(heads, (tokens + kQT - 1) / kQT);
    AttentionPrefillTiled<<<grid, 256, 0, stream>>>(
        q, k_cache, v_cache, gate, out, start, tokens, heads, kv_heads,
        head_dim, scale);
    return;
  }
  const dim3 grid(static_cast<std::uint32_t>(tokens) * heads);
  const dim3 block(std::max<std::uint32_t>(head_dim, 256U));
  AttentionPrefillNaive<<<grid, block, 0, stream>>>(
      q, k_cache, v_cache, gate, out, start, heads, kv_heads, head_dim, scale);
}

}  // namespace gufo::models::qwen36_a3b::rocm