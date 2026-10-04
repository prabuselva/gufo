#include "src/models/qwen36_a3b/kernels/rocm/kernels.hpp"

#include <hip/hip_fp16.h>

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

// Residual add fused with the following RMSNorm: x += addend, then
// out = rmsnorm(x) * gamma. Bit-identical to AddKernel followed by
// RmsNormKernel for a single row (grid == 1): the sum is formed as a float in
// the same order, accumulated as double, and reduced with the same block size.
__global__ void FusedAddRmsNormKernel(float* x, const float* addend,
                                      const float* gamma, float* out,
                                      std::uint32_t dim, float eps) {
  __shared__ double reduce[32];
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float sum = x[i] + addend[i];
    x[i] = sum;
    const double v = sum;
    ss += v * v;
  }
  const float scale =
      1.0F / sqrtf(static_cast<float>(BlockReduceSum(ss, reduce) /
                                      static_cast<double>(dim)) +
                   eps);
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    out[i] = x[i] * scale * (gamma != nullptr ? gamma[i] : 1.0F);
  }
}

__global__ void RopeKernel(float* x, const std::uint32_t* pos,
                           std::uint32_t heads, std::uint32_t head_dim,
                           std::uint32_t rotary_dim, float theta,
                           const qwen::vision::DeviceRope* rope) {
  const std::uint32_t row = blockIdx.x / heads;
  const std::uint32_t head = blockIdx.x % heads;
  const std::uint32_t half = rotary_dim / 2;
  if (threadIdx.x >= half) {
    return;
  }
  const std::uint32_t i = threadIdx.x;
  const float p = qwen::vision::RopePosition(rope, pos[row], i);
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

// Fused decode attention front-end: deinterleave the [q | gate] query
// projection, per-head RMSNorm and NEOX partial RoPE on q and k, and publish
// k/v into the FP32 KV planes and their FP16 mirror — all in one launch. Each
// block owns one head (query, key or value). The per-head norm reuses the exact
// RmsNormKernel reduction (double BlockReduceSum over a 256-thread block,
// scale = 1/sqrtf((float)(sum/dim) + eps), out = x*scale*gamma) and keeps the
// normalized value as a float in shared before RoPE, so the RoPE reads the same
// rounded floats the unfused chain round-trips through global memory. The RoPE
// expressions match RopeKernel bit-for-bit, and the cache mirror uses
// __float2half_rn like ConvertKvChunkF16Kernel. head_dim must be <= 256.
__global__ void FusedQKNormRoPEKvWriteKernel(
    const float* __restrict__ qg, const float* __restrict__ k_in,
    const float* __restrict__ v_in, const float* __restrict__ q_norm,
    const float* __restrict__ k_norm, float* __restrict__ q_out,
    float* __restrict__ gate_out, float* __restrict__ k_out,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::uint32_t pos, std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float theta, float eps,
    const qwen::vision::DeviceRope* rope) {
  __shared__ double reduce[32];
  __shared__ float normed[256];
  const std::uint32_t half = rotary_dim / 2;
  const std::uint32_t b = blockIdx.x;
  const std::size_t kv_row = static_cast<std::size_t>(kv_heads) * head_dim;
  const std::size_t cache_base = static_cast<std::size_t>(pos) * kv_row;

  if (b < heads) {
    // Query head: deinterleave q|gate, RMSNorm, RoPE the leading rotary_dim.
    const float* src = qg + static_cast<std::size_t>(b) * 2 * head_dim;
    const float* gate_src = src + head_dim;
    double ss = 0.0;
    for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
      const double v = src[i];
      ss += v * v;
    }
    const float scale = 1.0F / sqrtf(static_cast<float>(
                                       BlockReduceSum(ss, reduce) /
                                       static_cast<double>(head_dim)) +
                                   eps);
    for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
      normed[i] = src[i] * scale * (q_norm != nullptr ? q_norm[i] : 1.0F);
      gate_out[static_cast<std::size_t>(b) * head_dim + i] = gate_src[i];
    }
    __syncthreads();
    float* dst = q_out + static_cast<std::size_t>(b) * head_dim;
    if (threadIdx.x < half) {
      const std::uint32_t i = threadIdx.x;
      const float p = qwen::vision::RopePosition(rope, pos, i);
      const float freq = powf(theta, -2.0F * static_cast<float>(i) /
                                        static_cast<float>(rotary_dim));
      const float angle = p * freq;
      const float c = cosf(angle);
      const float s = sinf(angle);
      const float a = normed[i];
      const float bb = normed[i + half];
      dst[i] = a * c - bb * s;
      dst[i + half] = a * s + bb * c;
    }
    for (std::uint32_t i = rotary_dim + threadIdx.x; i < head_dim;
         i += blockDim.x) {
      dst[i] = normed[i];
    }
  } else if (b < heads + kv_heads) {
    // Key head: RMSNorm, RoPE, publish to k_out and the FP32/FP16 cache.
    const std::uint32_t hk = b - heads;
    const float* src = k_in + static_cast<std::size_t>(hk) * head_dim;
    double ss = 0.0;
    for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
      const double v = src[i];
      ss += v * v;
    }
    const float scale = 1.0F / sqrtf(static_cast<float>(
                                       BlockReduceSum(ss, reduce) /
                                       static_cast<double>(head_dim)) +
                                   eps);
    for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
      normed[i] = src[i] * scale * (k_norm != nullptr ? k_norm[i] : 1.0F);
    }
    __syncthreads();
    float* dst = k_out + static_cast<std::size_t>(hk) * head_dim;
    float* kc = k_cache + cache_base + static_cast<std::size_t>(hk) * head_dim;
    __half* kc16 = k_cache_f16 != nullptr
                       ? k_cache_f16 + cache_base +
                             static_cast<std::size_t>(hk) * head_dim
                       : nullptr;
    if (threadIdx.x < half) {
      const std::uint32_t i = threadIdx.x;
      const float p = qwen::vision::RopePosition(rope, pos, i);
      const float freq = powf(theta, -2.0F * static_cast<float>(i) /
                                        static_cast<float>(rotary_dim));
      const float angle = p * freq;
      const float c = cosf(angle);
      const float s = sinf(angle);
      const float a = normed[i];
      const float bb = normed[i + half];
      const float r0 = a * c - bb * s;
      const float r1 = a * s + bb * c;
      dst[i] = r0;
      dst[i + half] = r1;
      kc[i] = r0;
      kc[i + half] = r1;
      if (kc16 != nullptr) {
        kc16[i] = __float2half_rn(r0);
        kc16[i + half] = __float2half_rn(r1);
      }
    }
    for (std::uint32_t i = rotary_dim + threadIdx.x; i < head_dim;
         i += blockDim.x) {
      const float val = normed[i];
      dst[i] = val;
      kc[i] = val;
      if (kc16 != nullptr) {
        kc16[i] = __float2half_rn(val);
      }
    }
  } else {
    // Value head: publish to the FP32/FP16 cache (v_out is the projection).
    const std::uint32_t hv = b - heads - kv_heads;
    const float* src = v_in + static_cast<std::size_t>(hv) * head_dim;
    float* vc = v_cache + cache_base + static_cast<std::size_t>(hv) * head_dim;
    __half* vc16 = v_cache_f16 != nullptr
                       ? v_cache_f16 + cache_base +
                             static_cast<std::size_t>(hv) * head_dim
                       : nullptr;
    for (std::uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
      const float val = src[i];
      vc[i] = val;
      if (vc16 != nullptr) {
        vc16[i] = __float2half_rn(val);
      }
    }
  }
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

// out[(t*k + s)*cols + i] = x[t*cols + i]: replicate each of the `tokens`
// activation rows `k` times so the grouped expert GEMVs can index x by slot.
__global__ void DupRowsKernel(const float* x, std::uint32_t tokens,
                              std::uint32_t k, std::uint32_t cols, float* out) {
  const std::size_t total = static_cast<std::size_t>(tokens) * k * cols;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const std::size_t slot = i / cols;
    out[i] = x[(slot / k) * cols + (i - slot * cols)];
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

// Fused decode GDN front-end: the depthwise causal conv and the per-head RMSNorm
// of the q and k groups in one launch. Each block owns one head (q, k or v) and
// each thread owns one channel of that head, so the conv is the same per-channel
// dot over the kernel taps plus the in-place history shift as GdnConvKernel. The
// q and k blocks then reduce the head's convolved values with the identical
// BlockReduceSum and blockDim (head_dim) as GdnNormQkKernel, so qn/kn are
// bit-identical to the unfused conv + norm chain while the convolved q/k never
// round-trips through global memory. The v block writes the convolved value the
// delta recurrence consumes.
__global__ void GdnConvNormQkKernel(const float* qkv, const float* conv_w,
                                    float* history, float* convolved, float* qn,
                                    float* kn, std::uint32_t channels,
                                    std::uint32_t kernel,
                                    std::uint32_t k_heads,
                                    std::uint32_t head_dim, float eps) {
  __shared__ double reduce[32];
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t i = threadIdx.x;
  const std::size_t ch = static_cast<std::size_t>(head) * head_dim + i;
  const float* w = conv_w + ch * kernel;
  float acc = w[kernel - 1] * qkv[ch];
  for (std::uint32_t t = 0; t + 1 < kernel; ++t) {
    acc += w[t] * history[static_cast<std::size_t>(t) * channels + ch];
  }
  const float c = DSilu(acc);
  for (std::uint32_t t = 0; t + 2 < kernel; ++t) {
    history[static_cast<std::size_t>(t) * channels + ch] =
        history[static_cast<std::size_t>(t + 1) * channels + ch];
  }
  history[static_cast<std::size_t>(kernel - 2) * channels + ch] = qkv[ch];
  if (head < 2U * k_heads) {
    const bool is_key = head >= k_heads;
    const std::uint32_t local = is_key ? head - k_heads : head;
    float* dst = (is_key ? kn : qn) + static_cast<std::size_t>(local) * head_dim;
    const double ss = static_cast<double>(c) * c;
    const float scale =
        1.0f / sqrtf(static_cast<float>(BlockReduceSum(ss, reduce)) + eps);
    dst[i] = c * scale;
  } else {
    convolved[ch] = c;
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
      reinterpret_cast<float4*>(
          state + (static_cast<std::size_t>(h) * head_dim + j) * head_dim) +
      lane;
  float4 s = *r4;
  s.x *= decay;
  s.y *= decay;
  s.z *= decay;
  s.w *= decay;
  double du = static_cast<double>(s.x) * kv.x +
              static_cast<double>(s.y) * kv.y +
              static_cast<double>(s.z) * kv.z + static_cast<double>(s.w) * kv.w;
  du = WarpReduceSumD(du);
  const float delta = (v[h * head_dim + j] - static_cast<float>(du)) * b;
  float4 o;
  o.x = s.x + delta * kv.x;
  o.y = s.y + delta * kv.y;
  o.z = s.z + delta * kv.z;
  o.w = s.w + delta * kv.w;
  const double dq =
      static_cast<double>(o.x) * qv.x + static_cast<double>(o.y) * qv.y +
      static_cast<double>(o.z) * qv.z + static_cast<double>(o.w) * qv.w;
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

// Multi-row flash-decoding split for the speculative verify pass. One block
// per (head, split) folds all query rows over the same KV chunk so the
// cache is read once for the block; row o attends [0, n_kv0 + o). Partials
// use the AttentionDecodeSplitKernel layout per row:
// [row][head][split][hd + 2].
template<std::uint32_t Rows>
__global__ void AttentionDecodeSplitRowsKernel(
    const float* q, const float* k_cache, const float* v_cache, float* part,
    std::uint32_t n_kv0, std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, float scale, std::uint32_t splits,
    std::uint32_t chunk) {
  constexpr std::uint32_t kWaves = 8;
  __shared__ float s_acc[Rows][kWaves][256];
  __shared__ float s_max[Rows][kWaves];
  __shared__ float s_sum[Rows][kWaves];
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t sp = blockIdx.y;
  const std::uint32_t kvh = h / (heads / kv_heads);
  const std::size_t row_stride = static_cast<std::size_t>(heads) * head_dim;
  const float* q0 = q + static_cast<std::size_t>(h) * head_dim;
  const std::size_t kv_stride = static_cast<std::size_t>(kv_heads) * head_dim;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = threadIdx.x >> 5;
  float qv[Rows][8];
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
#pragma unroll
    for (std::uint32_t m = 0; m < 8U; ++m) {
      qv[o][m] = q0[o * row_stride + lane + 32U * m];
    }
  }
  float wmax[Rows];
  float wsum[Rows];
  float acc[Rows][8];
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    wmax[o] = -INFINITY;
    wsum[o] = 0.0F;
#pragma unroll
    for (std::uint32_t m = 0; m < 8U; ++m) {
      acc[o][m] = 0.0F;
    }
  }
  const std::uint32_t n_kv_last = n_kv0 + Rows - 1U;
  const std::uint32_t j0 = sp * chunk;
  const std::uint32_t j1 = (n_kv_last < j0 + chunk) ? n_kv_last : j0 + chunk;
  for (std::uint32_t j = j0 + wave; j < j1; j += kWaves) {
    const float* kj =
        k_cache + static_cast<std::size_t>(j) * kv_stride + kvh * head_dim;
    float dot[Rows];
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      dot[o] = 0.0F;
    }
#pragma unroll
    for (std::uint32_t m = 0; m < 8U; ++m) {
      const float kv = kj[lane + 32U * m];
#pragma unroll
      for (std::uint32_t o = 0; o < Rows; ++o) {
        dot[o] += qv[o][m] * kv;
      }
    }
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      dot[o] = WarpReduceSumF(dot[o]) * scale;
    }
    const float* vj =
        v_cache + static_cast<std::size_t>(j) * kv_stride + kvh * head_dim;
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      if (j >= n_kv0 + o) {
        continue;
      }
      const float nm = fmaxf(wmax[o], dot[o]);
      const float r = expf(wmax[o] - nm);
      const float e = expf(dot[o] - nm);
      wsum[o] = wsum[o] * r + e;
#pragma unroll
      for (std::uint32_t m = 0; m < 8U; ++m) {
        acc[o][m] = acc[o][m] * r + e * vj[lane + 32U * m];
      }
      wmax[o] = nm;
    }
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      s_max[o][wave] = wmax[o];
      s_sum[o][wave] = wsum[o];
    }
  }
  __syncthreads();
  float bmax[Rows];
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    bmax[o] = -INFINITY;
  }
#pragma unroll
  for (std::uint32_t w = 0; w < kWaves; ++w) {
#pragma unroll
    for (std::uint32_t o = 0; o < Rows; ++o) {
      bmax[o] = fmaxf(bmax[o], s_max[o][w]);
    }
  }
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    // A row whose whole chunk is causally masked keeps wmax = -INF; its
    // contribution is zero, not the NaN that expf(-INF - -INF) would give.
    const float r = (wmax[o] == -INFINITY) ? 0.0F : expf(wmax[o] - bmax[o]);
#pragma unroll
    for (std::uint32_t m = 0; m < 8U; ++m) {
      s_acc[o][wave][lane + 32U * m] = acc[o][m] * r;
    }
  }
  __syncthreads();
  // The block covers head_dim threads; each thread folds one element of
  // every row into its partial slot.
#pragma unroll
  for (std::uint32_t o = 0; o < Rows; ++o) {
    float val = 0.0F;
#pragma unroll
    for (std::uint32_t w = 0; w < kWaves; ++w) {
      val += s_acc[o][w][threadIdx.x];
    }
    const std::size_t base =
        (static_cast<std::size_t>(o) * heads + h) * splits + sp;
    part[base * (head_dim + 2U) + threadIdx.x] = val;
  }
  if (threadIdx.x < Rows) {
    float total = 0.0F;
#pragma unroll
    for (std::uint32_t w = 0; w < kWaves; ++w) {
      const float sm = s_max[threadIdx.x][w];
      total += (sm == -INFINITY)
                   ? 0.0F
                   : s_sum[threadIdx.x][w] * expf(sm - bmax[threadIdx.x]);
    }
    const std::size_t base =
        (static_cast<std::size_t>(threadIdx.x) * heads + h) * splits + sp;
    float* pp = part + base * (head_dim + 2U);
    pp[head_dim] = bmax[threadIdx.x];
    pp[head_dim + 1U] = total;
  }
}

// Folds the AttentionDecodeSplitRowsKernel partials per row and head and
// applies the sigmoid output gate. Grid (heads, rows), block head_dim.
__global__ void AttentionDecodeCombineRowsKernel(const float* part,
                                                 const float* gate, float* out,
                                                 std::uint32_t heads,
                                                 std::uint32_t splits,
                                                 std::uint32_t head_dim) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t o = blockIdx.y;
  const std::uint32_t i = threadIdx.x;
  const float* pp = part + (static_cast<std::size_t>(o) * heads + h) * splits *
                               (head_dim + 2U);
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
    den +=
        pp[static_cast<std::size_t>(s) * (head_dim + 2U) + head_dim + 1U] * w;
  }
  const std::size_t off =
      static_cast<std::size_t>(o) * heads * head_dim + h * head_dim + i;
  out[off] = (num / den) * DSigmoid(gate[off]);
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
  const float* pp =
      part + static_cast<std::size_t>(h) * splits * (head_dim + 2U);
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
    den +=
        pp[static_cast<std::size_t>(s) * (head_dim + 2U) + head_dim + 1U] * w;
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
        const float* vj =
            v_cache +
            (static_cast<std::size_t>(base + t) * kv_heads + kvh) * head_dim;
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
                                      std::uint32_t tokens, std::uint32_t heads,
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

  const std::uint32_t q_last = q0 + kQT < tokens ? q0 + kQT : tokens;
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

// ---------------------------------------------------------------------------
// WMMA causal prefill attention on the matrix cores (FP16 KV cache).
//
// Ported from the Qwen route's validated kernel
// (gufo_slimsami/src/models/qwen/hip/kernels/attention_wmma.hip), reduced to
// the unpacked (position-major FP16 KV) path: no head-major repack, no LSE
// output, and the key sweep always starts at position 0. Wave32 fragment
// layout, verified against the CPU double-precision reference: A holds row
// L%16 and 16 contiguous k, B holds column L%16 and 16 contiguous k (fed a
// transposed tile), and C element i is row 2i + L/16, column L%16.
//
// The scalar AttentionPrefillTiled above stays the oracle; this kernel reads a
// position-major FP16 mirror of the KV cache ([position][kv_head][head_dim],
// the same layout as the FP32 cache) and writes the same gated context
// ([token][head][head_dim]). attention_scale = 1/16 is 1/sqrt(head_dim) for
// head_dim 256, folded into Q before the FP16 conversion so the score stays in
// range; softmax is invariant to the constant.
constexpr std::uint32_t kWmmaHeadDim = 256;
constexpr std::uint32_t kWmmaHeads = 2;       // query heads per block
constexpr std::uint32_t kWmmaQueryRows = 32;  // queries per block (x2 heads)
constexpr std::uint32_t kWmmaKeys = 16;       // keys per tile
constexpr std::uint32_t kWmmaKSteps = kWmmaHeadDim / 16;
constexpr std::uint32_t kWmmaKStride = kWmmaHeadDim + 8;

using WmmaV16h =
    __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using WmmaV8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ WmmaV8f WmmaOp(WmmaV16h a, WmmaV16h b, WmmaV8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// A fragment is 16 contiguous halves = 32 bytes; two 16-byte loads and a
// bitcast replace 16 element loads. Every fragment base is 16-byte aligned.
__device__ __forceinline__ WmmaV16h WmmaLoadFrag(const __half* p) {
  union {
    WmmaV16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

// V^T is stored [dim][key]. A 16-half pad every 8 dims spreads the packed
// transpose writes and the fragment reads over all 32 LDS banks: without it the
// four lane groups differ only in dim, and 8 dims x kVtStride halves is 0 mod
// 32, so they collide. Both strides are multiples of 8, so every fragment base
// and every packed dword stays aligned.
template<std::uint32_t kVtStride, std::uint32_t kVtSwizzle>
__device__ __forceinline__ constexpr std::uint32_t WmmaVtRow(std::uint32_t d) {
  return (d * kVtStride) + ((d / 8) * kVtSwizzle);
}

template<std::uint32_t kQueryHeads, std::uint32_t kKvHeads, bool kSparse>
__launch_bounds__(256, 2) __global__
    void WmmaCausalAttentionKernel(const float* __restrict__ q,
                                   const float* __restrict__ gate,
                                   const __half* __restrict__ k_cache_f16,
                                   const __half* __restrict__ v_cache_f16,
                                   float* __restrict__ out_context,
                                   std::uint32_t start_pos,
                                   std::size_t batch_size, std::uint32_t window,
                                   std::uint32_t sink) {
  constexpr std::uint32_t kHeadDim = kWmmaHeadDim;
  constexpr std::uint32_t kGqa = kQueryHeads / kKvHeads;
  constexpr std::uint32_t kAttnWidth = kQueryHeads * kHeadDim;
  constexpr std::uint32_t kKvWidth = kKvHeads * kHeadDim;
  constexpr std::uint32_t kQueryRows = kWmmaQueryRows;
  constexpr std::uint32_t kKeys = kWmmaKeys;
  static_assert(kQueryHeads % kKvHeads == 0 && kGqa % kWmmaHeads == 0,
                "each block takes kWmmaHeads heads of one KV group");
  constexpr std::uint32_t kRowBlocks = (kQueryRows / 16) * kWmmaHeads;
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / (8 / kSTiles);
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kOTilesPerWave = (kRowBlocks * kWmmaKSteps) / 8;
  constexpr std::uint32_t kSoftmaxLanes = 256 / kRows;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  constexpr std::uint32_t kVtSwizzle = 16;
  static_assert(kSTiles == 4, "eight waves cover four S tiles in two halves");
  static_assert(kOTilesPerWave == 2 * kRowBlocks, "O tiles per wave");
  static_assert(kKeys % 16 == 0 && kQueryRows % 16 == 0, "16-row WMMA tiles");
  static_assert(kKStepsPerWave * 8 / kSTiles == kWmmaKSteps, "k split");
  static_assert(kKeys == 16 && kHeadDim == 256,
                "packed V mapping assumes 4 lanes x 32 dims per key");
  static_assert(
      kWmmaKStride % 8 == 0 && kVtStride % 8 == 0 && kVtSwizzle % 8 == 0,
      "fragment rows must start on a 16-byte boundary");
  static_assert(kSoftmaxLanes * (kKeys / kSoftmaxLanes) == kKeys, "softmax");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  const std::uint32_t query_start = blockIdx.x * kQueryRows;
  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (kGqa / kWmmaHeads);
  const std::uint32_t pair_in_group = head_pair % (kGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      (kv_head * kGqa) + (pair_in_group * kWmmaHeads);
  constexpr float attention_scale = 1.0F / 16.0F;

  const auto row_block_head = [&](std::uint32_t rb) {
    return first_query_head + (rb % kWmmaHeads);
  };
  const auto row_block_offset = [&](std::uint32_t rb) {
    return (rb / kWmmaHeads) * 16;
  };

  // The K tile needs kKeys * kWmmaKStride halves; the transposed V staging
  // needs kHeadDim * kVtStride. Size for the larger so the V transpose never
  // runs off the end and corrupts the neighbouring softmax buffers.
  constexpr std::uint32_t kVtHalves =
      WmmaVtRow<kVtStride, kVtSwizzle>(kHeadDim - 1) + kKeys;
  constexpr std::uint32_t kKvLdsHalves =
      (kKeys * kWmmaKStride > kVtHalves) ? kKeys * kWmmaKStride : kVtHalves;
  __shared__ __half kv_lds[kKvLdsHalves];
  __shared__ float s_lds[2][kSTiles][16][17];
  __shared__ __half p_lds[kRows][kKeys + 8];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  WmmaV16h q_frag[kKStepsPerWave];
  {
    const std::uint32_t local_query =
        query_start + row_block_offset(s_rb) + sub;
    const bool live = local_query < batch_size;
    const float* q_row =
        q + (static_cast<std::size_t>(local_query) * kAttnWidth) +
        (static_cast<std::size_t>(row_block_head(s_rb)) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x * attention_scale);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y * attention_scale);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z * attention_scale);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w * attention_scale);
      }
    }
  }

  WmmaV8f o_acc[kRowBlocks][2] = {};
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);

  // Opt-in sliding-window + attention-sink sparsity (kSparse). Each query keeps
  // the first `sink` keys and the last `window` keys; the middle key tiles are
  // never loaded, scored or multiplied. Tiles are 16 keys, so the two retained
  // ranges are tile-aligned and the softmax mask still drops the keys inside a
  // boundary tile that fall outside the window. sink_end never overlaps win_lo.
  std::uint32_t sink_end = 0;
  std::uint32_t win_lo = 0;
  if constexpr (kSparse) {
    win_lo = (start_pos + query_start + 1u > window)
                 ? ((start_pos + query_start + 1u - window) / kKeys) * kKeys
                 : 0u;
    if (win_lo > max_visible) {
      win_lo = max_visible;
    }
    sink_end = ((sink + kKeys - 1u) / kKeys) * kKeys;
    if (sink_end > win_lo) {
      sink_end = win_lo;
    }
  }
  const std::uint32_t first_key_start =
      (kSparse && sink_end == 0u) ? win_lo : 0u;

  constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
  constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
  static_assert(kKRegs * 256 == kKeys * (kHeadDim / 8), "K stages evenly");
  static_assert(kVRegs == 2, "packed V holds one uint4 per key of a pair");
  // Lane L holds dims [wave*32 + (L%4)*8, +8) of keys 2*(L/4) and 2*(L/4)+1, so
  // the two keys land in adjacent V^T columns and transpose to one dword.
  const std::uint32_t v_pair = lane >> 2u;
  const std::uint32_t v_dim = (wave * 32u) + ((lane & 3u) * 8u);
  const std::size_t kv_stride = kKvWidth;
  const std::size_t head_offset = static_cast<std::size_t>(kv_head) * kHeadDim;
  const auto* v_base = v_cache_f16 + head_offset + v_dim;
  const auto* k_base = k_cache_f16 + head_offset;

  const auto load_v = [&](std::uint32_t key_start, uint4* dst) {
#pragma unroll
    for (std::uint32_t j = 0; j < 2; ++j) {
      const std::uint32_t key_position = key_start + (2 * v_pair) + j;
      dst[j] = (key_position < context_end)
                   ? *reinterpret_cast<const uint4*>(
                         v_base +
                         (static_cast<std::size_t>(key_position) * kv_stride))
                   : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  const auto load_k = [&](std::uint32_t key_start, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_position = key_start + (idx / (kHeadDim / 8));
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      dst[n] =
          (key_position < context_end)
              ? *reinterpret_cast<const uint4*>(
                    k_base +
                    (static_cast<std::size_t>(key_position) * kv_stride) + d8)
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  uint4 k_cur[kKRegs];
  uint4 v_cur[kVRegs];
  uint4 k_pre[kKRegs] = {};
  uint4 v_pre[kVRegs] = {};
  load_k(first_key_start, k_cur);
  load_v(first_key_start, v_cur);

  for (std::uint32_t key_start = first_key_start; key_start < max_visible;) {
    // Next visited tile: the following tile, or jump over the dropped middle
    // range [sink_end, win_lo) when sparse so its KV is never fetched.
    std::uint32_t next_start = key_start + kKeys;
    if constexpr (kSparse) {
      if (next_start >= sink_end && next_start < win_lo) {
        next_start = win_lo;
      }
    }
    __syncthreads();
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&kv_lds[(key_row * kWmmaKStride) + d8]) =
          k_cur[n];
    }
    __syncthreads();

    if (next_start < max_visible) {
      load_k(next_start, k_pre);
      load_v(next_start, v_pre);
    }

    // --- S = Q K^T ---
    {
      WmmaV8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const WmmaV16h k_frag =
            WmmaLoadFrag(&kv_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        s_acc = WmmaOp(q_frag[ks], k_frag, s_acc);
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
      const std::uint32_t absolute_query = start_pos + local_query;
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key_position = key_start + col;
        bool valid = local_query < batch_size &&
                     key_position <= absolute_query &&
                     key_position < context_end;
        if constexpr (kSparse) {
          valid = valid && (key_position < sink ||
                            absolute_query - key_position < window);
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
      const float prev_max = running_max;
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
      running_max = next_max;
      running_sum = (running_sum * prior_scale) + part_sum;
      if (seg == 0) {
        row_scale[rg] = prior_scale;
      }
    }
    __syncthreads();

    // Rescale the running O across both dimension tiles (unpacked path).
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
      float scale[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale[i];
        }
      }
    }

    // --- transpose V into LDS: 8 packed dwords, two adjacent V^T columns each
    {
      const auto* lo = reinterpret_cast<const __half*>(&v_cur[0]);
      const auto* hi = reinterpret_cast<const __half*>(&v_cur[1]);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t packed =
            static_cast<std::uint32_t>(__half_as_ushort(lo[i])) |
            (static_cast<std::uint32_t>(__half_as_ushort(hi[i])) << 16);
        *reinterpret_cast<std::uint32_t*>(
            &kv_lds[WmmaVtRow<kVtStride, kVtSwizzle>(v_dim + i) +
                    (2 * v_pair)]) = packed;
      }
    }
    __syncthreads();

    // --- O += P V ---
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
        const std::uint32_t dim_tile = wave + t * 8;
        WmmaV16h v_frag[kKeyBlocks];
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          v_frag[kb] = WmmaLoadFrag(
              &kv_lds[WmmaVtRow<kVtStride, kVtSwizzle>((dim_tile * 16) + sub) +
                      (kb * 16)]);
        }
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const WmmaV16h p_frag =
              WmmaLoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          WmmaV8f next = WmmaOp(p_frag, v_frag[kb], o_acc[rb][t]);
          const auto first_key = key_start + kb * 16;
          if (first_key + 15 > start_pos + query_start) {
            WmmaV8f tail = o_acc[rb][t];
            const int relative_key = static_cast<int>(first_key) -
                                     static_cast<int>(start_pos + query_start +
                                                      row_block_offset(rb));
#pragma unroll
            for (std::uint32_t key = 0; key < 16; ++key) {
              const float value = static_cast<float>(v_frag[kb][key]);
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                if (static_cast<int>(half_id) <
                        relative_key + 15 - static_cast<int>(2 * i) &&
                    static_cast<int>(half_id) >= relative_key +
                                                     static_cast<int>(key) -
                                                     static_cast<int>(2 * i)) {
                  tail[i] =
                      fmaf(__half2float(
                               p_lds[rb * 16 + 2 * i + half_id][kb * 16 + key]),
                           value, tail[i]);
                }
              }
            }
#pragma unroll
            for (std::uint32_t i = 0; i < 8; ++i) {
              if (static_cast<int>(half_id) <
                  relative_key + 15 - static_cast<int>(2 * i)) {
                next[i] = tail[i];
              }
            }
          }
          o_acc[rb][t] = next;
        }
      }
    }

#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      k_cur[n] = k_pre[n];
    }
#pragma unroll
    for (std::uint32_t n = 0; n < kVRegs; ++n) {
      v_cur[n] = v_pre[n];
    }
    key_start = next_start;
  }
  if (tid % kSoftmaxLanes == 0) {
    row_max[tid / kSoftmaxLanes] = running_max;
    row_sum[tid / kSoftmaxLanes] = running_sum;
  }
  __syncthreads();

  // --- epilogue ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
    const std::uint32_t query_head = row_block_head(rb);
    const std::uint32_t row_offset = row_block_offset(rb);
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        const std::uint32_t local_query = query_start + row_offset + row;
        if (local_query >= batch_size) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(local_query) * kAttnWidth) +
            (static_cast<std::size_t>(query_head) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr) {
          value *= 1.0F / (1.0F + __expf(-gate[offset]));
        }
        out_context[offset] = value;
      }
    }
  }
}

// Convert one prefill chunk's FP32 keys/values to the FP16 KV mirror the WMMA
// kernel reads. `k`/`v` are the just-published FP32 cache rows at [start,
// start + tokens); the layout is [position][kv_head][head_dim] for both.
__global__ void ConvertKvChunkF16Kernel(const float* __restrict__ k,
                                        const float* __restrict__ v,
                                        __half* __restrict__ k_f16,
                                        __half* __restrict__ v_f16,
                                        std::size_t count) {
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    k_f16[i] = __float2half_rn(k[i]);
    v_f16[i] = __float2half_rn(v[i]);
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
      const float xv =
          s >= 0 ? qkv[static_cast<std::size_t>(s) * channels + ch]
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
  const std::size_t total = static_cast<std::size_t>(kernel - 1) * channels;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const std::uint32_t j = static_cast<std::uint32_t>(idx / channels);
    const std::uint32_t ch = static_cast<std::uint32_t>(idx % channels);
    const std::int64_t src = static_cast<std::int64_t>(tokens) -
                             static_cast<std::int64_t>(kernel - 1) +
                             static_cast<std::int64_t>(j);
    out[idx] = src >= 0
                   ? qkv[static_cast<std::size_t>(src) * channels + ch]
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
      convolved + static_cast<std::size_t>(token) * channels +
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

// Fused prefill GDN front-end: the batched depthwise causal conv and the
// per-head RMS norm of the q and k groups in one launch, mirroring the decode
// GdnConvNormQkKernel. Each block owns one (token, head); each thread owns one
// channel of that head. The conv is the same per-channel dot over the kernel
// taps as GdnConvPrefill (reading the halo from `qkv` for in-chunk positions
// and from `history` for the leading taps), so the convolved value is
// bit-identical. The q and k blocks then reduce the head's convolved values
// with the identical BlockReduceSum and blockDim (head_dim) as
// GdnNormQkPrefill, so qn/kn are bit-identical to the unfused conv + norm chain
// while the convolved q/k never round-trips through global memory. The v blocks
// write the convolved value the delta recurrence consumes. History is advanced
// separately by GdnHistoryUpdate, so this kernel leaves `history` untouched.
__global__ void GdnConvNormQkPrefillKernel(
    const float* qkv, const float* conv_w, const float* history,
    float* convolved, float* qn, float* kn, std::uint32_t tokens,
    std::uint32_t channels, std::uint32_t kernel, std::uint32_t k_heads,
    std::uint32_t v_heads, std::uint32_t head_dim, float eps) {
  __shared__ double reduce[32];
  const std::uint32_t heads = 2U * k_heads + v_heads;
  const std::uint32_t head = blockIdx.x % heads;
  const std::uint32_t token = blockIdx.x / heads;
  const std::uint32_t i = threadIdx.x;
  const std::size_t ch = static_cast<std::size_t>(head) * head_dim + i;
  const std::size_t idx = static_cast<std::size_t>(token) * channels + ch;
  const float* w = conv_w + ch * kernel;
  float acc = w[kernel - 1] * qkv[idx];
  for (std::uint32_t k = 0; k + 1 < kernel; ++k) {
    const std::int64_t s = static_cast<std::int64_t>(token) -
                           static_cast<std::int64_t>(kernel - 1) +
                           static_cast<std::int64_t>(k);
    const float xv =
        s >= 0 ? qkv[static_cast<std::size_t>(s) * channels + ch]
               : history[static_cast<std::size_t>(
                             static_cast<std::int64_t>(kernel - 1) + s) *
                             channels +
                         ch];
    acc += w[k] * xv;
  }
  const float c = DSilu(acc);
  if (head < 2U * k_heads) {
    const bool is_key = head >= k_heads;
    const std::uint32_t local = is_key ? head - k_heads : head;
    float* dst = (is_key ? kn : qn) +
                 (static_cast<std::size_t>(token) * k_heads + local) * head_dim;
    const double ss = static_cast<double>(c) * c;
    const float scale =
        1.0f / sqrtf(static_cast<float>(BlockReduceSum(ss, reduce)) + eps);
    dst[i] = c * scale;
  } else {
    convolved[idx] = c;
  }
}

// Gated delta-rule recurrence over a chunk, sequential over tokens but fully
// on-device. One warp owns kRows state rows of one value head; each lane
// keeps kCols state columns (head_dim = 64 * kCols) in registers across the
// whole chunk, so the per-token dot products reduce with warp shuffles only:
// no shared memory and no barriers on the token recurrence. o[r] reuses
// dot(s_new[r], q) = oq[r] + delta[r] * dot(k, q), and the next token's q/k
// elements are prefetched while the current one reduces.
template<std::uint32_t kRows, std::uint32_t kCols>
__global__ void GdnDeltaLoopKernel(
    const float* qn, const float* kn, const float* convolved,
    const float* alpha, const float* beta, const float* a, const float* dt,
    float* state, float* attn, std::uint32_t tokens, std::uint32_t k_heads,
    std::uint32_t v_heads, std::uint32_t head_dim, std::uint32_t channels,
    float* snap, std::uint32_t snap_rows) {
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t kh = h % k_heads;
  const std::uint32_t j0 = blockIdx.x * kRows;
  const std::uint32_t lane = threadIdx.x;
  const std::size_t attn_stride = static_cast<std::size_t>(v_heads) * head_dim;
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
      const std::size_t off =
          (static_cast<std::size_t>(t + 1) * k_heads + kh) * head_dim + lane;
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
    if (snap != nullptr && t < snap_rows) {
      for (std::uint32_t r = 0; r < kRows; ++r) {
        float* row =
            snap +
            ((static_cast<std::size_t>(t) * v_heads + h) * head_dim + j0 + r) *
                head_dim;
        for (std::uint32_t c = 0; c < kCols; ++c) {
          row[lane + c * blockDim.x] = s[r][c];
        }
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

// Four-lane butterfly via the ALU-path DPP row-xmask modifier (gfx10+): the xor
// masks stay inside an aligned 4-lane group, so the sum is valid in every lane
// of the group that cooperates on one state row.
__device__ __forceinline__ float RowXorAdd(float v) {
  const int y2 = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v),
                                             0x162, 0xF, 0xF, false);
  v += __builtin_bit_cast(float, y2);
  const int y1 = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v),
                                             0x161, 0xF, 0xF, false);
  v += __builtin_bit_cast(float, y1);
  return v;
}

// Per-token decay and beta, precomputed once per (token, value head) so the
// recurrence loop reads two scalars instead of running expf/softplus/sigmoid.
// The expressions match GdnDeltaLoopRowSplitKernel exactly, so every value is
// bit-identical; only the location of the work changes (out of the serial
// token loop, and computed once instead of redundantly by all 256 threads of
// every block that shares a head).
__global__ void GdnPrepAlphaBetaKernel(const float* alpha, const float* beta,
                                       const float* a, const float* dt,
                                       float* alpha_pre, float* beta_pre,
                                       std::uint32_t count,
                                       std::uint32_t v_heads) {
  const std::size_t i =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }
  const std::uint32_t h = static_cast<std::uint32_t>(i % v_heads);
  alpha_pre[i] = expf(a[h] * DSoftplus(alpha[i] + dt[h]));
  beta_pre[i] = DSigmoid(beta[i]);
}

// Per-token q.k dot for a key head, precomputed for the recurrence output term.
// The four cooperating lanes accumulate the same eight float4 in the same order
// as GdnDeltaLoopRowSplitKernel and reduce with the same RowXorAdd butterfly,
// so the stored value equals the in-loop kq bit for bit.
__global__ void GdnPrepKqKernel(const float* qn, const float* kn, float* kq_pre,
                                std::uint32_t pairs, std::uint32_t k_heads,
                                std::uint32_t head_dim) {
  constexpr int kVec = 8;  // 32 columns / 4 lanes
  const std::uint32_t pair =
      blockIdx.x * 8U + (static_cast<std::uint32_t>(threadIdx.x) >> 2);
  const int seg = static_cast<int>(threadIdx.x) & 3;
  if (pair >= pairs) {
    return;
  }
  const std::uint32_t kh = pair % k_heads;
  const std::uint32_t t = pair / k_heads;
  const std::size_t base =
      (static_cast<std::size_t>(t) * k_heads + kh) * head_dim;
  const float4* qp = reinterpret_cast<const float4*>(qn + base);
  const float4* kp = reinterpret_cast<const float4*>(kn + base);
  float kq = 0.0F;
#pragma unroll
  for (int vi = 0; vi < kVec; ++vi) {
    const float4 q4 = qp[seg * kVec + vi];
    const float4 k4 = kp[seg * kVec + vi];
    kq += k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
  }
  kq = RowXorAdd(kq) * (1.0F / sqrtf(static_cast<float>(head_dim)));
  if (seg == 0) {
    kq_pre[pair] = kq;
  }
}

// Row-split gated delta-rule recurrence for a prefill chunk (no per-token
// snapshots; the Verify path keeps GdnDeltaLoopKernel). head_dim == 128: four
// lanes cooperate on one 128-wide state row (32 columns each as eight float4),
// eight row groups fill a wave, and the two per-row reductions run as a
// four-lane butterfly instead of the full 32-lane shuffle. State stays in
// registers for the whole chunk and k/q are prefetched one token ahead. The
// arithmetic matches GdnDeltaLoopKernel on the pre-normalized qn/kn.
template<std::uint32_t kRowsPerLane>
__global__ void __launch_bounds__(256)
    GdnDeltaLoopRowSplitKernel(const float* qn, const float* kn,
                               const float* convolved, const float* alpha_pre,
                               const float* beta_pre, const float* kq_pre,
                               float* state, float* attn, std::uint32_t tokens,
                               std::uint32_t k_heads, std::uint32_t v_heads,
                               std::uint32_t head_dim, std::uint32_t channels) {
  constexpr int kLanesPerRow = 4;
  constexpr int kRowGroups = 32 / kLanesPerRow;            // 8
  constexpr int kRowsPerWave = kRowGroups * kRowsPerLane;  // 8 or 16
  constexpr int kVec = 8;                                  // 32 columns / 4
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t kh = h % k_heads;
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int seg = lane % kLanesPerRow;
  const int grp = lane / kLanesPerRow;
  const int row0 = static_cast<int>(blockIdx.x) * (kRowsPerWave * 8) +
                   wave * kRowsPerWave + grp;
  const std::size_t attn_stride = static_cast<std::size_t>(v_heads) * head_dim;
  const float q_scale = 1.0F / sqrtf(static_cast<float>(head_dim));

  float* s_head = state + static_cast<std::size_t>(h) * head_dim * head_dim;
  float4 s_reg[kRowsPerLane][kVec];
#pragma unroll
  for (int r = 0; r < kRowsPerLane; ++r) {
    const float4* src = reinterpret_cast<const float4*>(
        s_head + static_cast<std::size_t>(row0 + r * kRowGroups) * head_dim);
#pragma unroll
    for (int vi = 0; vi < kVec; ++vi) {
      s_reg[r][vi] = src[seg * kVec + vi];
    }
  }

  const float* q_base = qn + static_cast<std::size_t>(kh) * head_dim;
  const float* k_base = kn + static_cast<std::size_t>(kh) * head_dim;
  const float* v_base = convolved +
                        2 * static_cast<std::size_t>(k_heads) * head_dim +
                        static_cast<std::size_t>(h) * head_dim;
  const float* ap_base = alpha_pre + h;
  const float* bp_base = beta_pre + h;
  const float* kqp_base = kq_pre + kh;
  float* o_base = attn + static_cast<std::size_t>(h) * head_dim;

  float4 kv[kVec];
  float4 qv[kVec];
  {
    const float4* kp = reinterpret_cast<const float4*>(k_base);
    const float4* qp = reinterpret_cast<const float4*>(q_base);
#pragma unroll
    for (int vi = 0; vi < kVec; ++vi) {
      kv[vi] = kp[seg * kVec + vi];
      qv[vi] = qp[seg * kVec + vi];
    }
  }
  // Double-buffer the per-token scalars that sit on the recurrence's critical
  // path (decay, beta, k.q and the convolved v row) one iteration ahead,
  // exactly as k/q are prefetched, so their global-load latency hides behind
  // the current token's FMA chain. Pure load scheduling: the arithmetic and its
  // order are unchanged, so the result is bit-identical.
  float decay = *ap_base;
  float b = *bp_base;
  float kq = *kqp_base;
  float vcur[kRowsPerLane];
#pragma unroll
  for (int r = 0; r < kRowsPerLane; ++r) {
    vcur[r] = v_base[row0 + r * kRowGroups];
  }

  for (std::uint32_t t = 0; t < tokens; ++t) {
    float4 knx[kVec];
    float4 qnx[kVec];
    float decay_n = 0.0F;
    float b_n = 0.0F;
    float kq_n = 0.0F;
    float v_n[kRowsPerLane];
    if (t + 1 < tokens) {
      const float4* kp =
          reinterpret_cast<const float4*>(k_base + k_heads * head_dim);
      const float4* qp =
          reinterpret_cast<const float4*>(q_base + k_heads * head_dim);
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        knx[vi] = kp[seg * kVec + vi];
        qnx[vi] = qp[seg * kVec + vi];
      }
      decay_n = *(ap_base + v_heads);
      b_n = *(bp_base + v_heads);
      kq_n = *(kqp_base + k_heads);
#pragma unroll
      for (int r = 0; r < kRowsPerLane; ++r) {
        v_n[r] = v_base[channels + row0 + r * kRowGroups];
      }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
      float u = 0.0F;
      float p = 0.0F;
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        float4 s4 = s_reg[r][vi];
        s4.x *= decay;
        s4.y *= decay;
        s4.z *= decay;
        s4.w *= decay;
        s_reg[r][vi] = s4;
        u += s4.x * kv[vi].x + s4.y * kv[vi].y + s4.z * kv[vi].z +
             s4.w * kv[vi].w;
        p += s4.x * qv[vi].x + s4.y * qv[vi].y + s4.z * qv[vi].z +
             s4.w * qv[vi].w;
      }
      u = RowXorAdd(u);
      p = RowXorAdd(p);
      const int row = row0 + r * kRowGroups;
      const float delta = (vcur[r] - u) * b;
      if (seg == 0) {
        o_base[row] = p * q_scale + delta * kq;
      }
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        float4 s4 = s_reg[r][vi];
        s4.x += delta * kv[vi].x;
        s4.y += delta * kv[vi].y;
        s4.z += delta * kv[vi].z;
        s4.w += delta * kv[vi].w;
        s_reg[r][vi] = s4;
      }
    }
    q_base += k_heads * head_dim;
    k_base += k_heads * head_dim;
    v_base += channels;
    ap_base += v_heads;
    bp_base += v_heads;
    kqp_base += k_heads;
    o_base += attn_stride;
#pragma unroll
    for (int vi = 0; vi < kVec; ++vi) {
      kv[vi] = knx[vi];
      qv[vi] = qnx[vi];
    }
    decay = decay_n;
    b = b_n;
    kq = kq_n;
#pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
      vcur[r] = v_n[r];
    }
  }

#pragma unroll
  for (int r = 0; r < kRowsPerLane; ++r) {
    float4* dst = reinterpret_cast<float4*>(
        s_head + static_cast<std::size_t>(row0 + r * kRowGroups) * head_dim);
#pragma unroll
    for (int vi = 0; vi < kVec; ++vi) {
      dst[seg * kVec + vi] = s_reg[r][vi];
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

void FusedAddRmsNorm(float* x, const float* addend, const float* gamma,
                     float* out, std::uint32_t dim, float eps,
                     hipStream_t stream) {
  const dim3 block(std::min<std::uint32_t>(dim, 256U));
  FusedAddRmsNormKernel<<<1, block, 0, stream>>>(x, addend, gamma, out, dim,
                                                 eps);
}

void Rope(float* x, const std::uint32_t* pos, std::uint32_t rows,
          std::uint32_t heads, std::uint32_t head_dim, std::uint32_t rotary_dim,
          float theta, hipStream_t stream,
          const qwen::vision::DeviceRope* rope) {
  const dim3 grid(rows * heads);
  const dim3 block(rotary_dim / 2);
  RopeKernel<<<grid, block, 0, stream>>>(x, pos, heads, head_dim, rotary_dim,
                                         theta, rope);
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

bool FusedQKNormRoPEKvWrite(
    const float* qg, const float* k_in, const float* v_in, const float* q_norm,
    const float* k_norm, float* q_out, float* gate_out, float* k_out,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    std::uint32_t pos, std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float theta, float eps,
    hipStream_t stream, const qwen::vision::DeviceRope* rope) {
  if (head_dim == 0U || head_dim > 256U || rotary_dim % 2U != 0U ||
      rotary_dim > head_dim) {
    return false;
  }
  const dim3 grid(heads + 2U * kv_heads);
  const dim3 block(head_dim);
  FusedQKNormRoPEKvWriteKernel<<<grid, block, 0, stream>>>(
      qg, k_in, v_in, q_norm, k_norm, q_out, gate_out, k_out, k_cache, v_cache,
      static_cast<__half*>(k_cache_f16), static_cast<__half*>(v_cache_f16), pos,
      heads, kv_heads, head_dim, rotary_dim, theta, eps, rope);
  return true;
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

void DupRows(const float* x, std::uint32_t tokens, std::uint32_t k,
             std::uint32_t cols, float* out, hipStream_t stream) {
  const std::size_t count = static_cast<std::size_t>(tokens) * k * cols;
  const std::size_t block = 256;
  const std::size_t grid =
      std::min<std::size_t>((count + block - 1) / block, 65535);
  DupRowsKernel<<<grid, block, 0, stream>>>(x, tokens, k, cols, out);
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

void GdnConvNormQk(const float* qkv, const float* conv_w, float* history,
                   float* convolved, float* qn, float* kn,
                   std::uint32_t channels, std::uint32_t kernel,
                   std::uint32_t k_heads, std::uint32_t v_heads,
                   std::uint32_t head_dim, float eps, hipStream_t stream) {
  const std::uint32_t heads = 2U * k_heads + v_heads;
  GdnConvNormQkKernel<<<heads, head_dim, 0, stream>>>(
      qkv, conv_w, history, convolved, qn, kn, channels, kernel, k_heads,
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
  GdnConvPrefillKernel<<<grid, block, 0, stream>>>(
      qkv, conv_w, history, convolved, tokens, channels, kernel);
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
  GdnNormQkPrefillKernel<<<grid, block, 0, stream>>>(
      convolved, qn, kn, tokens, k_heads, channels, head_dim, eps);
}

void GdnConvNormQkPrefill(const float* qkv, const float* conv_w,
                          const float* history, float* convolved, float* qn,
                          float* kn, std::uint32_t tokens,
                          std::uint32_t channels, std::uint32_t kernel,
                          std::uint32_t k_heads, std::uint32_t v_heads,
                          std::uint32_t head_dim, float eps,
                          hipStream_t stream) {
  const dim3 grid(static_cast<std::uint32_t>(tokens) *
                  (2U * k_heads + v_heads));
  const dim3 block(head_dim);
  GdnConvNormQkPrefillKernel<<<grid, block, 0, stream>>>(
      qkv, conv_w, history, convolved, qn, kn, tokens, channels, kernel,
      k_heads, v_heads, head_dim, eps);
}

void GdnPrep(const float* alpha, const float* beta, const float* a,
             const float* dt, const float* qn, const float* kn,
             float* alpha_pre, float* beta_pre, float* kq_pre,
             std::uint32_t tokens, std::uint32_t k_heads, std::uint32_t v_heads,
             std::uint32_t head_dim, hipStream_t stream) {
  const std::uint32_t count = tokens * v_heads;
  GdnPrepAlphaBetaKernel<<<(count + 255U) / 256U, 256, 0, stream>>>(
      alpha, beta, a, dt, alpha_pre, beta_pre, count, v_heads);
  const std::uint32_t pairs = tokens * k_heads;
  GdnPrepKqKernel<<<(pairs + 7U) / 8U, 32, 0, stream>>>(qn, kn, kq_pre, pairs,
                                                        k_heads, head_dim);
}

void GdnDeltaLoop(const float* qn, const float* kn, const float* convolved,
                  const float* alpha, const float* beta, const float* a,
                  const float* dt, const float* alpha_pre,
                  const float* beta_pre, const float* kq_pre, float* state,
                  float* attn, std::uint32_t tokens, std::uint32_t k_heads,
                  std::uint32_t v_heads, std::uint32_t head_dim,
                  std::uint32_t channels, float* snap, std::uint32_t snap_rows,
                  hipStream_t stream) {
  static const std::uint32_t warp = [] {
    int ws = 64;
    hipDeviceGetAttribute(&ws, hipDeviceAttributeWarpSize, 0);
    return static_cast<std::uint32_t>(ws);
  }();
  // Prefill (no per-token snapshots) on the model's 128-wide state uses the
  // row-split recurrence: four lanes per state row and a four-lane butterfly
  // cut the reduction cost versus the 32-lane shuffle. The Verify path keeps
  // GdnDeltaLoopKernel, which emits the per-row state snapshots rollback needs.
  if (snap == nullptr && head_dim == 128U) {
    constexpr std::uint32_t kRowsPerWave = 8U;  // kRowsPerLane == 1
    const dim3 grid(head_dim / (kRowsPerWave * 8U), v_heads);
    GdnDeltaLoopRowSplitKernel<1><<<grid, dim3(256), 0, stream>>>(
        qn, kn, convolved, alpha_pre, beta_pre, kq_pre, state, attn, tokens,
        k_heads, v_heads, head_dim, channels);
    return;
  }
  constexpr std::uint32_t kRows = 8;
  const dim3 grid(head_dim / kRows, v_heads);
  const dim3 block(warp);
  switch (head_dim / warp) {
    case 8:
      GdnDeltaLoopKernel<kRows, 8><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels, snap, snap_rows);
      break;
    case 7:
      GdnDeltaLoopKernel<kRows, 7><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels, snap, snap_rows);
      break;
    case 6:
      GdnDeltaLoopKernel<kRows, 6><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels, snap, snap_rows);
      break;
    case 5:
      GdnDeltaLoopKernel<kRows, 5><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels, snap, snap_rows);
      break;
    case 4:
      GdnDeltaLoopKernel<kRows, 4><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels, snap, snap_rows);
      break;
    case 3:
      GdnDeltaLoopKernel<kRows, 3><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels, snap, snap_rows);
      break;
    case 2:
      GdnDeltaLoopKernel<kRows, 2><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels, snap, snap_rows);
      break;
    default:
      GdnDeltaLoopKernel<kRows, 1><<<grid, block, 0, stream>>>(
          qn, kn, convolved, alpha, beta, a, dt, state, attn, tokens, k_heads,
          v_heads, head_dim, channels, snap, snap_rows);
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

__global__ void MtpConcatKernel(const float* e, const float* h,
                                const float* h_prev, float* out,
                                std::uint32_t hidden) {
  const std::size_t t = blockIdx.x;
  const float* h_src = t == 0 ? h_prev : h + (t - 1) * hidden;
  float* row = out + t * 2 * static_cast<std::size_t>(hidden);
  const float* e_row = e + t * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    row[i] = e_row[i];
    row[hidden + i] = h_src[i];
  }
}

void MtpConcat(const float* e, const float* h, const float* h_prev, float* out,
               std::uint32_t tokens, std::uint32_t hidden, hipStream_t stream) {
  MtpConcatKernel<<<tokens, 256, 0, stream>>>(e, h, h_prev, out, hidden);
}

void AttentionDecode(const float* q, const float* k_cache, const float* v_cache,
                     const float* gate, float* out, float* scratch, float* part,
                     std::uint32_t n_kv, std::uint32_t heads,
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

void AttentionDecodeRows(const float* q, const float* k_cache,
                         const float* v_cache, const float* gate, float* out,
                         float* part, std::uint32_t n_kv0, std::uint32_t rows,
                         std::uint32_t heads, std::uint32_t kv_heads,
                         std::uint32_t head_dim, float scale,
                         hipStream_t stream) {
  const std::uint32_t n_kv_last = n_kv0 + rows - 1U;
  std::uint32_t splits = (n_kv_last + 63U) / 64U;
  if (splits > 32U) {
    splits = 32U;
  }
  std::uint32_t chunk = (n_kv_last + splits - 1U) / splits;
  splits = (n_kv_last + chunk - 1U) / chunk;
#define GUFO_ATTN_DECODE_ROWS(R)                                         \
  case R:                                                                \
    AttentionDecodeSplitRowsKernel<R>                                    \
        <<<dim3(heads, splits), 256, 0, stream>>>(                       \
            q, k_cache, v_cache, part, n_kv0, heads, kv_heads, head_dim, \
            scale, splits, chunk);                                       \
    break;
  switch (rows) {
    GUFO_ATTN_DECODE_ROWS(2U)
    GUFO_ATTN_DECODE_ROWS(3U)
    GUFO_ATTN_DECODE_ROWS(4U)
    GUFO_ATTN_DECODE_ROWS(5U)
    default:
      return;
  }
#undef GUFO_ATTN_DECODE_ROWS
  AttentionDecodeCombineRowsKernel<<<dim3(heads, rows), head_dim, 0, stream>>>(
      part, gate, out, heads, splits, head_dim);
}

void KvCacheWriteF16(const float* k, const float* v, void* k_cache_f16,
                     void* v_cache_f16, std::size_t start, std::size_t count,
                     hipStream_t stream) {
  if (k_cache_f16 == nullptr || v_cache_f16 == nullptr || count == 0) {
    return;
  }
  const float* k_src = k + start;
  const float* v_src = v + start;
  auto* k_dst = static_cast<__half*>(k_cache_f16) + start;
  auto* v_dst = static_cast<__half*>(v_cache_f16) + start;
  constexpr std::uint32_t kBlock = 256;
  const std::size_t want = (count + kBlock - 1) / kBlock;
  const std::uint32_t blocks =
      static_cast<std::uint32_t>(want < 4096 ? (want == 0 ? 1 : want) : 4096);
  ConvertKvChunkF16Kernel<<<blocks, kBlock, 0, stream>>>(k_src, v_src, k_dst,
                                                         v_dst, count);
}

void AttentionPrefill(const float* q, const float* k_cache,
                      const float* v_cache, void* k_cache_f16,
                      void* v_cache_f16, const float* gate, float* out,
                      std::uint32_t start, std::uint32_t tokens,
                      std::uint32_t heads, std::uint32_t kv_heads,
                      std::uint32_t head_dim, float scale, hipStream_t stream,
                      std::uint32_t window, std::uint32_t sink) {
  // Matrix-core route: the validated 16Q/2KV, head_dim 256 shape. The caller
  // has already published the chunk into both the FP32 cache and its FP16
  // mirror (KvCacheWriteF16) at [start, start + tokens); the WMMA kernel reads
  // the whole prefix causally from the FP16 planes. A positive `window` selects
  // the sliding-window + sink sparse instantiation; window 0 keeps the dense
  // kernel byte-for-byte.
  if (k_cache_f16 != nullptr && v_cache_f16 != nullptr && head_dim == 256U &&
      heads == 16U && kv_heads == 2U) {
    const dim3 grid(
        static_cast<unsigned>((tokens + kWmmaQueryRows - 1) / kWmmaQueryRows),
        kv_heads * ((heads / kv_heads) / kWmmaHeads));
    if (window > 0U) {
      WmmaCausalAttentionKernel<16, 2, true><<<grid, 256, 0, stream>>>(
          q, gate, static_cast<const __half*>(k_cache_f16),
          static_cast<const __half*>(v_cache_f16), out, start, tokens, window,
          sink);
    } else {
      WmmaCausalAttentionKernel<16, 2, false><<<grid, 256, 0, stream>>>(
          q, gate, static_cast<const __half*>(k_cache_f16),
          static_cast<const __half*>(v_cache_f16), out, start, tokens, 0U, 0U);
    }
    return;
  }
  if (head_dim <= 256U) {
    constexpr std::uint32_t kQT = 32;
    const dim3 grid(heads, (tokens + kQT - 1) / kQT);
    AttentionPrefillTiled<<<grid, 256, 0, stream>>>(q, k_cache, v_cache, gate,
                                                    out, start, tokens, heads,
                                                    kv_heads, head_dim, scale);
    return;
  }
  const dim3 grid(static_cast<std::uint32_t>(tokens) * heads);
  const dim3 block(std::max<std::uint32_t>(head_dim, 256U));
  AttentionPrefillNaive<<<grid, block, 0, stream>>>(
      q, k_cache, v_cache, gate, out, start, heads, kv_heads, head_dim, scale);
}

}  // namespace gufo::models::qwen36_a3b::rocm