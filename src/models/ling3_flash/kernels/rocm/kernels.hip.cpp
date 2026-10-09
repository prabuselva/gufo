#include "src/models/ling3_flash/kernels/rocm/kernels.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <limits>

namespace gufo::models::ling3_flash::rocm {
namespace {

constexpr float kWeightsSumEps = 6.103515625e-5F;
constexpr float kClampEps = 1e-6F;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

__device__ __forceinline__ float SigmoidD(float x) {
  return 1.0F / (1.0F + ::expf(-x));
}

__device__ __forceinline__ float SiluD(float x) {
  return x * SigmoidD(x);
}

/// Block-wide sum of a double partial. `smem` needs blockDim.x/32 slots.
__device__ __forceinline__ double BlockSumDouble(double value, double* smem) {
  const unsigned lane = threadIdx.x & 31U;
  const unsigned warp = threadIdx.x >> 5;
  const unsigned warps = (blockDim.x + 31U) >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    value += __shfl_down_sync(~0ULL, value, off);
  }
  if (lane == 0U) {
    smem[warp] = value;
  }
  __syncthreads();
  if (warp == 0U) {
    value = (lane < warps) ? smem[lane] : 0.0;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      value += __shfl_down_sync(~0ULL, value, off);
    }
    if (lane == 0U) {
      smem[0] = value;
    }
  }
  __syncthreads();
  return smem[0];
}

__global__ void RmsNormKernel(const float* __restrict__ x,
                              const float* __restrict__ w, float eps,
                              float* __restrict__ out, std::uint32_t n) {
  __shared__ double smem[8];
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const double v = static_cast<double>(x[i]);
    ss += v * v;
  }
  const float scale =
      1.0F / ::sqrtf(static_cast<float>(BlockSumDouble(ss, smem) /
                                        static_cast<double>(n)) +
                     eps);
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = x[i] * scale * (w != nullptr ? w[i] : 1.0F);
  }
}

__global__ void AddKernel(const float* __restrict__ a,
                          const float* __restrict__ b, float* __restrict__ out,
                          std::uint32_t n) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

__global__ void Concat2Kernel(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ out, std::uint32_t n) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i];
    out[n + i] = b[i];
  }
}

__global__ void SigmoidKernel(float* __restrict__ x, std::uint32_t n) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = SigmoidD(x[i]);
  }
}

__global__ void RopeKernel(float* __restrict__ x, std::uint32_t heads,
                           std::uint32_t stride, std::uint32_t rotary_dim,
                           std::uint32_t pos, float theta) {
  const std::uint32_t pairs = rotary_dim / 2U;
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= heads * pairs) {
    return;
  }
  const std::uint32_t h = i / pairs;
  const std::uint32_t p = i % pairs;
  float* v = x + static_cast<std::size_t>(h) * stride;
  const float freq = static_cast<float>(
      ::pow(static_cast<double>(theta),
            -2.0 * static_cast<double>(p) / static_cast<double>(rotary_dim)));
  const float angle = static_cast<float>(pos) * freq;
  double c = 0.0;
  double s = 0.0;
  ::sincos(static_cast<double>(angle), &s, &c);
  const float a = v[2 * p];
  const float b = v[2 * p + 1];
  v[2 * p] = a * c - b * s;
  v[2 * p + 1] = a * s + b * c;
}

__global__ void ConvSiluKernel(const float* __restrict__ conv_w,
                               const float* __restrict__ proj,
                               float* __restrict__ history,
                               float* __restrict__ out, std::uint32_t inner,
                               std::uint32_t kern) {
  const std::uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= inner) {
    return;
  }
  const float* cw = conv_w + static_cast<std::size_t>(ch) * kern;
  float acc = cw[kern - 1] * proj[ch];
  for (std::uint32_t t = 0; t + 1 < kern; ++t) {
    acc += cw[t] * history[static_cast<std::size_t>(t) * inner + ch];
  }
  const float value = SiluD(acc);
  out[ch] = value;
  for (std::uint32_t t = 0; t + 2 < kern; ++t) {
    history[static_cast<std::size_t>(t) * inner + ch] =
        history[static_cast<std::size_t>(t + 1) * inner + ch];
  }
  // The conv state holds the raw projected rows, not the activated output.
  if (kern > 1) {
    history[static_cast<std::size_t>(kern - 2) * inner + ch] = proj[ch];
  }
}

__global__ void KdaNormHeadsKernel(float* __restrict__ q, float* __restrict__ k,
                                   std::uint32_t d, float eps, float q_scale) {
  __shared__ double smem[8];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * d;
  for (int which = 0; which < 2; ++which) {
    float* v = (which == 0 ? q : k) + base;
    const double x = static_cast<double>(v[threadIdx.x]);
    double ss = BlockSumDouble(x * x, smem);
    const float scale = 1.0F / ::sqrtf(static_cast<float>(ss) + eps);
    v[threadIdx.x] *= scale * (which == 0 ? q_scale : 1.0F);
    __syncthreads();
  }
}

__global__ void KdaDeltaKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ f,
    const float* __restrict__ beta, const float* __restrict__ a,
    const float* __restrict__ norm_w, const float* __restrict__ g,
    float* __restrict__ state, float* __restrict__ out, std::uint32_t d,
    float eps, float gate_lb) {
  __shared__ double smem[8];
  __shared__ float s_k[128];
  __shared__ float s_q[128];
  __shared__ float s_u[128];
  __shared__ float s_o[128];
  __shared__ float s_decay[128];
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t j = threadIdx.x;
  const std::size_t hd = static_cast<std::size_t>(h) * d;
  s_k[j] = k[hd + j];
  s_q[j] = q[hd + j];
  // Decay addresses the key axis: element i of every row uses gate[i],
  // matching the ggml gated_delta_net KDA path.
  s_decay[j] = ::expf(gate_lb * SigmoidD(a[h] * f[hd + j]));
  __syncthreads();

  float* S = state + static_cast<std::size_t>(h) * d * d;
  float* row = S + static_cast<std::size_t>(j) * d;
  double acc = 0.0;
  for (std::uint32_t i = 0; i < d; ++i) {
    row[i] *= s_decay[i];
    acc += static_cast<double>(row[i]) * static_cast<double>(s_k[i]);
  }
  s_u[j] = static_cast<float>(acc);
  __syncthreads();

  const float b = SigmoidD(beta[h]);
  const float delta = (v[hd + j] - s_u[j]) * b;
  acc = 0.0;
  for (std::uint32_t i = 0; i < d; ++i) {
    row[i] += delta * s_k[i];
    acc += static_cast<double>(row[i]) * static_cast<double>(s_q[i]);
  }
  s_o[j] = static_cast<float>(acc);
  __syncthreads();

  const double o = static_cast<double>(s_o[j]);
  const double ss = BlockSumDouble(o * o, smem);
  const float scale =
      1.0F / ::sqrtf(static_cast<float>(ss / static_cast<double>(d)) + eps);
  out[hd + j] = s_o[j] * scale * norm_w[j] * SigmoidD(g[hd + j]);
}

__global__ void RouterTopKKernel(const float* __restrict__ logits,
                                 const float* __restrict__ bias,
                                 std::uint32_t n_exp, std::uint32_t groups,
                                 std::uint32_t groups_used, std::uint32_t used,
                                 float weights_scale,
                                 std::int32_t* __restrict__ ids,
                                 float* __restrict__ weights) {
  extern __shared__ float sm[];  // [n_exp] selection, [n_exp] probs, [groups]
  float* selection = sm;
  float* probs = sm + n_exp;
  float* group_scores = sm + 2 * n_exp;
  const std::uint32_t per_group = n_exp / groups;
  for (std::uint32_t e = threadIdx.x; e < n_exp; e += blockDim.x) {
    probs[e] = SigmoidD(logits[e]);
    selection[e] = probs[e] + bias[e];
  }
  __syncthreads();
  if (groups > 1U) {
    if (threadIdx.x < groups) {
      const std::uint32_t g = threadIdx.x;
      float best = kNegInf;
      float second = kNegInf;
      for (std::uint32_t i = 0; i < per_group; ++i) {
        const float value =
            selection[static_cast<std::size_t>(g) * per_group + i];
        if (value >= best) {
          second = best;
          best = value;
        } else if (value > second) {
          second = value;
        }
      }
      group_scores[g] = best + second;
    }
    __syncthreads();
    __shared__ bool keep[64];
    if (threadIdx.x == 0U) {
      for (std::uint32_t g = 0; g < groups; ++g) {
        keep[g] = false;
      }
      for (std::uint32_t s = 0; s < groups_used; ++s) {
        std::uint32_t best_g = 0U;
        float best = kNegInf;
        for (std::uint32_t g = 0; g < groups; ++g) {
          if (!keep[g] && group_scores[g] > best) {
            best = group_scores[g];
            best_g = g;
          }
        }
        keep[best_g] = true;
      }
    }
    __syncthreads();
    for (std::uint32_t g = threadIdx.x; g < groups; g += blockDim.x) {
      if (!keep[g]) {
        for (std::uint32_t i = 0; i < per_group; ++i) {
          selection[static_cast<std::size_t>(g) * per_group + i] = kNegInf;
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    double sum = 0.0;
    for (std::uint32_t s = 0; s < used; ++s) {
      std::uint32_t best_e = 0U;
      float best = kNegInf;
      for (std::uint32_t e = 0; e < n_exp; ++e) {
        if (selection[e] > best) {
          best = selection[e];
          best_e = e;
        }
      }
      selection[best_e] = kNegInf;
      ids[s] = static_cast<std::int32_t>(best_e);
      weights[s] = probs[best_e];
      sum += static_cast<double>(probs[best_e]);
    }
    const float norm = 1.0F / fmaxf(static_cast<float>(sum), kWeightsSumEps);
    for (std::uint32_t s = 0; s < used; ++s) {
      weights[s] *= norm * weights_scale;
    }
  }
}

__global__ void SwigluClampedKernel(const float* __restrict__ gate,
                                    const float* __restrict__ up,
                                    float* __restrict__ out, std::uint32_t n,
                                    float limit) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  if (limit > kClampEps) {
    const float u = fminf(fmaxf(up[i], -limit), limit);
    const float g = fminf(SiluD(gate[i]), limit);
    out[i] = g * u;
  } else {
    out[i] = SiluD(gate[i]) * up[i];
  }
}

__global__ void RowScaleKernel(float* __restrict__ x,
                               const float* __restrict__ scales,
                               std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * cols) {
    return;
  }
  x[i] *= scales[i / cols];
}

__global__ void RowWeightedSumKernel(const float* __restrict__ src,
                                     const float* __restrict__ weights,
                                     float* __restrict__ dst,
                                     std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cols) {
    return;
  }
  float acc = 0.0F;
  for (std::uint32_t r = 0; r < rows; ++r) {
    acc += weights[r] * src[static_cast<std::size_t>(r) * cols + i];
  }
  dst[i] = acc;
}

std::uint32_t Blocks(std::uint32_t n, std::uint32_t block) {
  return (n + block - 1U) / block;
}

}  // namespace

void RmsNorm(const float* x, const float* w, float eps, float* out,
             std::uint32_t n, hipStream_t stream) {
  RmsNormKernel<<<1, 256, 0, stream>>>(x, w, eps, out, n);
}

void Add(const float* a, const float* b, float* out, std::uint32_t n,
         hipStream_t stream) {
  AddKernel<<<Blocks(n, 256), 256, 0, stream>>>(a, b, out, n);
}

void Concat2(const float* a, const float* b, float* out, std::uint32_t n,
             hipStream_t stream) {
  Concat2Kernel<<<Blocks(n, 256), 256, 0, stream>>>(a, b, out, n);
}

void SigmoidInPlace(float* x, std::uint32_t n, hipStream_t stream) {
  SigmoidKernel<<<Blocks(n, 256), 256, 0, stream>>>(x, n);
}

void Rope(float* x, std::uint32_t heads, std::uint32_t head_dim,
          std::uint32_t rotary_dim, std::uint32_t pos, float theta,
          hipStream_t stream) {
  const std::uint32_t pairs = heads * (rotary_dim / 2U);
  RopeKernel<<<Blocks(pairs, 256), 256, 0, stream>>>(x, heads, head_dim,
                                                     rotary_dim, pos, theta);
}

void RopeStrided(float* x, std::uint32_t heads, std::uint32_t stride,
                 std::uint32_t rotary_dim, std::uint32_t pos, float theta,
                 hipStream_t stream) {
  const std::uint32_t pairs = heads * (rotary_dim / 2U);
  RopeKernel<<<Blocks(pairs, 256), 256, 0, stream>>>(x, heads, stride,
                                                     rotary_dim, pos, theta);
}

void ConvSilu(const float* conv_w, const float* proj, float* history,
              float* out, std::uint32_t inner, std::uint32_t kern,
              hipStream_t stream) {
  ConvSiluKernel<<<Blocks(inner, 256), 256, 0, stream>>>(conv_w, proj, history,
                                                         out, inner, kern);
}

void KdaNormHeads(float* q, float* k, std::uint32_t heads, std::uint32_t d,
                  float eps, float q_scale, hipStream_t stream) {
  KdaNormHeadsKernel<<<heads, d, 0, stream>>>(q, k, d, eps, q_scale);
}

void KdaDelta(const float* q, const float* k, const float* v, const float* f,
              const float* beta, const float* a, const float* norm_w,
              const float* g, float* state, float* out, std::uint32_t heads,
              std::uint32_t d, float eps, float gate_lb, hipStream_t stream) {
  KdaDeltaKernel<<<heads, d, 0, stream>>>(q, k, v, f, beta, a, norm_w, g, state,
                                          out, d, eps, gate_lb);
}

void RouterTopK(const float* logits, const float* bias, std::uint32_t n_exp,
                std::uint32_t groups, std::uint32_t groups_used,
                std::uint32_t used, float weights_scale, std::int32_t* ids,
                float* weights, hipStream_t stream) {
  const std::uint32_t smem = (2 * n_exp + groups) * sizeof(float);
  RouterTopKKernel<<<1, 512, smem, stream>>>(logits, bias, n_exp, groups,
                                             groups_used, used, weights_scale,
                                             ids, weights);
}

void SwigluClamped(const float* gate, const float* up, float* out,
                   std::uint32_t n, float limit, hipStream_t stream) {
  SwigluClampedKernel<<<Blocks(n, 256), 256, 0, stream>>>(gate, up, out, n,
                                                          limit);
}

void RowScale(float* x, const float* scales, std::uint32_t rows,
              std::uint32_t cols, hipStream_t stream) {
  RowScaleKernel<<<Blocks(rows * cols, 256), 256, 0, stream>>>(x, scales, rows,
                                                               cols);
}

void RowWeightedSum(const float* src, const float* weights, float* dst,
                    std::uint32_t rows, std::uint32_t cols,
                    hipStream_t stream) {
  RowWeightedSumKernel<<<Blocks(cols, 256), 256, 0, stream>>>(src, weights, dst,
                                                              rows, cols);
}

}  // namespace gufo::models::ling3_flash::rocm