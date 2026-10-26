#include "src/models/flint/kernels/rocm/kernels.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

namespace gufo::models::flint::rocm {
namespace {

constexpr int kNormThreads = 256;

__device__ __forceinline__ float BlockReduceSum(float v, float* sm) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(__activemask(), v, off);
  }
  if (lane == 0) {
    sm[warp] = v;
  }
  __syncthreads();
  const int warps = blockDim.x / 32;
  float total = 0.0F;
  if (threadIdx.x == 0) {
    for (int i = 0; i < warps; ++i) {
      total += sm[i];
    }
    sm[0] = total;
  }
  __syncthreads();
  return sm[0];
}

__global__ void RmsNormKernel(const float* __restrict__ x,
                              const float* __restrict__ gamma, float eps,
                              float* __restrict__ out, std::uint32_t cols) {
  __shared__ float sm[kNormThreads / 32];
  const float* __restrict__ xr =
      x + static_cast<std::size_t>(blockIdx.x) * cols;
  float* __restrict__ orow = out + static_cast<std::size_t>(blockIdx.x) * cols;
  float ss = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < cols; i += kNormThreads) {
    ss += xr[i] * xr[i];
  }
  ss = BlockReduceSum(ss, sm);
  const float scale = 1.0F / sqrtf(ss / static_cast<float>(cols) + eps);
  for (std::uint32_t i = threadIdx.x; i < cols; i += kNormThreads) {
    orow[i] = xr[i] * scale * gamma[i];
  }
}

__global__ void AddResidualKernel(float* __restrict__ x,
                                  const float* __restrict__ y,
                                  std::uint32_t cols) {
  float* __restrict__ xr = x + static_cast<std::size_t>(blockIdx.x) * cols;
  const float* __restrict__ yr =
      y + static_cast<std::size_t>(blockIdx.x) * cols;
  for (std::uint32_t i = threadIdx.x; i < cols; i += kNormThreads) {
    xr[i] += yr[i];
  }
}

__global__ void SiluMulKernel(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ out, std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    const float av = a[i];
    out[i] = av / (1.0F + expf(-av)) * b[i];
  }
}

__global__ void QkEpilogueKernel(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
    float eps, float theta, int rope, std::uint32_t pos0, std::uint32_t n_head,
    std::uint32_t n_kv, std::uint32_t head_dim, __half* __restrict__ k_cache,
    __half* __restrict__ v_cache, std::uint32_t kv_row) {
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t h = blockIdx.x;
  const bool is_q = h < n_head;
  const std::uint32_t d = threadIdx.x;
  float* __restrict__ x =
      is_q ? q + (static_cast<std::size_t>(t) * n_head + h) * head_dim
           : k + (static_cast<std::size_t>(t) * n_kv + (h - n_head)) * head_dim;
  __shared__ float sm_head[128];
  __shared__ float sm_part[8];
  const float xv = d < head_dim ? x[d] : 0.0F;
  float ss = xv * xv;
  ss = BlockReduceSum(ss, sm_part);
  const float scale = 1.0F / sqrtf(ss / static_cast<float>(head_dim) + eps);
  const float* gamma = is_q ? q_gamma : k_gamma;
  sm_head[d] = xv * scale * gamma[d];
  __syncthreads();
  const std::uint32_t pos = pos0 + t;
  if (rope != 0 && d < head_dim / 2) {
    const float freq = powf(
        theta, -2.0F * static_cast<float>(d) / static_cast<float>(head_dim));
    const float angle = static_cast<float>(pos) * freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const float a = sm_head[d];
    const float b = sm_head[d + head_dim / 2];
    sm_head[d] = a * c - b * s;
    sm_head[d + head_dim / 2] = a * s + b * c;
  }
  // The rope threads also rewrite the partner half of the head; the output
  // threads must not read those lanes until every rotation has landed.
  __syncthreads();
  if (is_q) {
    if (d < head_dim) {
      x[d] = sm_head[d];
    }
    return;
  }
  const std::uint32_t kvh = h - n_head;
  __half* __restrict__ krow =
      k_cache + static_cast<std::size_t>(pos) * kv_row + kvh * head_dim;
  const float* __restrict__ vrow =
      v + (static_cast<std::size_t>(t) * n_kv + kvh) * head_dim;
  __half* __restrict__ vwrite =
      v_cache + static_cast<std::size_t>(pos) * kv_row + kvh * head_dim;
  if (d < head_dim) {
    krow[d] = __float2half_rn(sm_head[d]);
    vwrite[d] = __float2half_rn(vrow[d]);
  }
}

__global__ void AttentionKernel(const float* __restrict__ q,
                                const __half* __restrict__ k_cache,
                                const __half* __restrict__ v_cache,
                                const float* __restrict__ sinks, float scale,
                                std::uint32_t pos0, std::uint32_t window,
                                std::uint32_t n_head, std::uint32_t n_kv,
                                std::uint32_t head_dim, std::uint32_t kv_row,
                                float* __restrict__ out) {
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t pos = pos0 + t;
  const std::uint32_t start =
      (window != 0 && pos + 1 > window) ? pos + 1 - window : 0;
  const std::uint32_t d = threadIdx.x;
  __shared__ float sm_q[128];
  __shared__ float sm_sc[4];
  sm_q[d] = q[(static_cast<std::size_t>(t) * n_head + h) * head_dim + d];
  __syncthreads();
  const std::uint32_t kvh = h * n_kv / n_head;
  // One warp per key slot of a 4-key chunk; head_dim 128 -> 4 dims per lane.
  const int slot = d / 32;
  const int lane = d % 32;
  float m = -INFINITY;
  float lacc = 0.0F;
  float o = 0.0F;
  for (std::uint32_t p = start; p <= pos; p += 4) {
    const std::uint32_t left = pos - p + 1;
    const int cnt = static_cast<int>(left < 4U ? left : 4U);
    if (slot < cnt) {
      const __half* __restrict__ kp =
          k_cache + static_cast<std::size_t>(p + slot) * kv_row +
          kvh * head_dim + lane * 4;
      float dot = 0.0F;
      for (int j = 0; j < 4; ++j) {
        dot += sm_q[lane * 4 + j] * __half2float(kp[j]);
      }
      for (int off = 16; off > 0; off >>= 1) {
        dot += __shfl_down_sync(__activemask(), dot, off);
      }
      if (lane == 0) {
        sm_sc[slot] = dot * scale;
      }
    }
    __syncthreads();
    float cm = -INFINITY;
    for (int c = 0; c < cnt; ++c) {
      cm = fmaxf(cm, sm_sc[c]);
    }
    const float m_new = fmaxf(m, cm);
    const float alpha = expf(m - m_new);
    float lnew = lacc * alpha;
    float add = 0.0F;
    for (int c = 0; c < cnt; ++c) {
      const float e = expf(sm_sc[c] - m_new);
      lnew += e;
      const __half* __restrict__ vp = v_cache +
                                      static_cast<std::size_t>(p + c) * kv_row +
                                      kvh * head_dim + d;
      add += e * __half2float(*vp);
    }
    o = o * alpha + add;
    m = m_new;
    lacc = lnew;
    __syncthreads();
  }
  lacc += expf(sinks[h] - m);
  out[(static_cast<std::size_t>(t) * n_head + h) * head_dim + d] = o / lacc;
}

constexpr float kWeightSumFloor = 6.103515625e-5F;

__global__ void RouterTopKKernel(const float* __restrict__ logits,
                                 std::int32_t* __restrict__ ids,
                                 float* __restrict__ weights,
                                 std::uint32_t used, int normalize,
                                 float weight_scale) {
  __shared__ float sp[256];
  const std::uint32_t t = blockIdx.x;
  if (threadIdx.x < 256) {
    const float x = logits[static_cast<std::size_t>(t) * 256 + threadIdx.x];
    sp[threadIdx.x] = 1.0F / (1.0F + expf(-x));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sel[8];
    float wsum = 0.0F;
    for (std::uint32_t r = 0; r < used; ++r) {
      int best = -1;
      float bv = -FLT_MAX;
      for (int i = 0; i < 256; ++i) {
        if (sp[i] > bv) {
          bv = sp[i];
          best = i;
        }
      }
      ids[static_cast<std::size_t>(t) * used + r] = best;
      sel[r] = bv;
      wsum += bv;
      sp[best] = -FLT_MAX;
    }
    if (normalize != 0) {
      if (wsum < kWeightSumFloor) {
        wsum = kWeightSumFloor;
      }
      for (std::uint32_t r = 0; r < used; ++r) {
        float w = sel[r] / wsum;
        if (weight_scale != 0.0F && weight_scale != 1.0F) {
          w *= weight_scale;
        }
        weights[static_cast<std::size_t>(t) * used + r] = w;
      }
    } else {
      for (std::uint32_t r = 0; r < used; ++r) {
        weights[static_cast<std::size_t>(t) * used + r] = sel[r];
      }
    }
  }
}

__global__ void SumExpertsKernel(float* __restrict__ x,
                                 const float* __restrict__ routed,
                                 const float* __restrict__ weights,
                                 const float* __restrict__ shared,
                                 std::uint32_t cols, std::uint32_t used) {
  const std::uint32_t t = blockIdx.y;
  float* __restrict__ xr = x + static_cast<std::size_t>(t) * cols;
  const float* __restrict__ rbase =
      routed + static_cast<std::size_t>(t) * used * cols;
  const float* __restrict__ wrow = weights + static_cast<std::size_t>(t) * used;
  for (std::uint32_t j = threadIdx.x; j < cols; j += kNormThreads) {
    float acc = 0.0F;
    for (std::uint32_t s = 0; s < used; ++s) {
      acc += wrow[s] * rbase[static_cast<std::size_t>(s) * cols + j];
    }
    if (shared != nullptr) {
      acc += shared[static_cast<std::size_t>(t) * cols + j];
    }
    xr[j] += acc;
  }
}

}  // namespace

void RmsNorm(const float* x, const float* gamma, float eps, float* out,
             std::uint32_t rows, std::uint32_t cols, hipStream_t stream) {
  RmsNormKernel<<<rows, kNormThreads, 0, stream>>>(x, gamma, eps, out, cols);
}

void AddResidual(float* x, const float* y, std::uint32_t rows,
                 std::uint32_t cols, hipStream_t stream) {
  AddResidualKernel<<<rows, kNormThreads, 0, stream>>>(x, y, cols);
}

void SiluMul(const float* a, const float* b, float* out, std::size_t count,
             hipStream_t stream) {
  const std::size_t block = 256;
  std::size_t grid = (count + block - 1) / block;
  if (grid > 65535) {
    grid = 65535;
  }
  SiluMulKernel<<<static_cast<unsigned int>(grid), block, 0, stream>>>(
      a, b, out, count);
}

void QkEpilogue(float* q, float* k, const float* v, const float* q_gamma,
                const float* k_gamma, float eps, float theta, bool rope,
                std::uint32_t pos0, std::uint32_t batch, std::uint32_t n_head,
                std::uint32_t n_kv, std::uint32_t head_dim, __half* k_cache,
                __half* v_cache, std::uint32_t kv_row_elements,
                hipStream_t stream) {
  const dim3 grid(n_head + n_kv, batch);
  QkEpilogueKernel<<<grid, 128, 0, stream>>>(
      q, k, v, q_gamma, k_gamma, eps, theta, rope ? 1 : 0, pos0, n_head, n_kv,
      head_dim, k_cache, v_cache, kv_row_elements);
}

void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const float* sinks, float scale, std::uint32_t pos0,
               std::uint32_t batch, std::uint32_t window, std::uint32_t n_head,
               std::uint32_t n_kv, std::uint32_t head_dim,
               std::uint32_t kv_row_elements, float* out, hipStream_t stream) {
  const dim3 grid(n_head, batch);
  AttentionKernel<<<grid, head_dim, 0, stream>>>(
      q, k_cache, v_cache, sinks, scale, pos0, window, n_head, n_kv, head_dim,
      kv_row_elements, out);
}

void RouterTopK(const float* probs_logits, std::int32_t* ids, float* weights,
                std::uint32_t batch, std::uint32_t used, bool normalize,
                float weight_scale, hipStream_t stream) {
  RouterTopKKernel<<<batch, 256, 0, stream>>>(probs_logits, ids, weights, used,
                                              normalize ? 1 : 0, weight_scale);
}

void SumExperts(float* x, const float* routed, const float* weights,
                const float* shared, std::uint32_t rows, std::uint32_t cols,
                std::uint32_t used, hipStream_t stream) {
  const dim3 grid(1, rows);
  SumExpertsKernel<<<grid, kNormThreads, 0, stream>>>(x, routed, weights,
                                                      shared, cols, used);
}

}  // namespace gufo::models::flint::rocm
