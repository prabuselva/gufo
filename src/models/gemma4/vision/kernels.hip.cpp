#include "src/models/gemma4/vision/kernels.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::gemma4::vision {
namespace {

constexpr std::uint32_t kPatch = 16;
constexpr std::uint32_t kPatchVec = kPatch * kPatch * 3;  // 768

__device__ __forceinline__ float GeluQuick(float x) {
  return x / (1.0F + expf(-1.702F * x));
}

__global__ void Im2ColKernel(const float* __restrict__ pixels,
                             float* __restrict__ patchmat, std::uint32_t nx,
                             std::uint32_t ny, std::uint32_t n_px,
                             std::uint32_t total) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const std::uint32_t p = static_cast<std::uint32_t>(idx / kPatchVec);
  const std::uint32_t col = static_cast<std::uint32_t>(idx % kPatchVec);
  const std::uint32_t ci = col / (kPatch * kPatch);
  const std::uint32_t rem = col % (kPatch * kPatch);
  const std::uint32_t ky = rem / kPatch;
  const std::uint32_t kx = rem % kPatch;
  const std::uint32_t ox = p % n_px;
  const std::uint32_t oy = p / n_px;
  const float raw =
      pixels[(static_cast<std::size_t>(ci) * ny + oy * kPatch + ky) * nx +
             ox * kPatch + kx];
  patchmat[idx] = raw * 2.0F - 1.0F;
}

__global__ void PosAddKernel(float* __restrict__ h,
                             const float* __restrict__ px,
                             const float* __restrict__ py, std::uint32_t n_px,
                             std::uint32_t emb, std::size_t total) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const std::uint32_t p = static_cast<std::uint32_t>(idx / emb);
  const std::uint32_t e = static_cast<std::uint32_t>(idx % emb);
  const float tx = px[static_cast<std::size_t>(p % n_px) * emb + e];
  const float ty = py[static_cast<std::size_t>(p / n_px) * emb + e];
  h[idx] += tx + ty;
}

// One thread per (patch, head, pair). head_dim = 4*quarter; pass 0 rotates
// dims [0,2q) by x, pass 1 rotates dims [2q,4q) by y.
__global__ void Rope2dKernel(float* __restrict__ qk,
                             const float* __restrict__ inv_freq,
                             std::uint32_t n_px, std::uint32_t heads,
                             std::uint32_t hd, std::uint32_t quarter,
                             std::size_t total) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const std::uint32_t pair = static_cast<std::uint32_t>(idx % (2 * quarter));
  const std::uint32_t rest = static_cast<std::uint32_t>(idx / (2 * quarter));
  const std::uint32_t pass = pair / quarter;
  const std::uint32_t i = pair % quarter;
  const std::uint32_t head = rest % heads;
  const std::uint32_t p = rest / heads;
  const std::uint32_t base = pass * (2 * quarter);
  const float pos = static_cast<float>(pass == 0 ? p % n_px : p / n_px);
  const float angle = pos * inv_freq[i];
  const float c = cosf(angle);
  const float s = sinf(angle);
  float* row = qk + static_cast<std::size_t>(p) * (heads * hd) +
               static_cast<std::size_t>(head) * hd;
  const float x0 = row[base + i];
  const float x1 = row[base + i + quarter];
  row[base + i] = x0 * c - x1 * s;
  row[base + i + quarter] = x0 * s + x1 * c;
}

__global__ void GegluQuickKernel(const float* __restrict__ gate,
                                 const float* __restrict__ up,
                                 float* __restrict__ act, std::size_t count) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  act[idx] = GeluQuick(gate[idx]) * up[idx];
}

__global__ void StdNormKernel(float* __restrict__ x,
                              const float* __restrict__ bias,
                              const float* __restrict__ scale,
                              std::uint32_t emb, std::size_t total) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const std::uint32_t e = static_cast<std::uint32_t>(idx % emb);
  x[idx] = (x[idx] - bias[e]) * scale[e];
}

__global__ void AvgPool3Kernel(const float* __restrict__ h,
                               float* __restrict__ pooled, std::uint32_t n_px,
                               std::uint32_t out_x, std::uint32_t emb,
                               float pool_scale, std::size_t total) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const std::uint32_t t = static_cast<std::uint32_t>(idx / emb);
  const std::uint32_t e = static_cast<std::uint32_t>(idx % emb);
  const std::uint32_t px = t % out_x;
  const std::uint32_t py = t / out_x;
  float acc = 0.0F;
  for (std::uint32_t dy = 0; dy < 3; ++dy) {
    for (std::uint32_t dx = 0; dx < 3; ++dx) {
      const std::uint32_t gx = px * 3 + dx;
      const std::uint32_t gy = py * 3 + dy;
      acc += h[(static_cast<std::size_t>(gy) * n_px + gx) * emb + e];
    }
  }
  pooled[idx] = (acc / 9.0F) * pool_scale;
}

// One block per (query, head); 72 threads (hd). Online softmax over keys, no
// 1/sqrt(d) scaling. Each thread owns one output feature.
__global__ void AttentionKernel(const float* __restrict__ q,
                                const float* __restrict__ k,
                                const float* __restrict__ v,
                                float* __restrict__ out, std::uint32_t n_pos,
                                std::uint32_t heads, std::uint32_t hd) {
  const std::uint32_t p = blockIdx.x;
  const std::uint32_t hh = blockIdx.y;
  const std::uint32_t tid = threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(heads) * hd;
  const float* qrow = q + static_cast<std::size_t>(p) * stride + hh * hd;
  const float qv = qrow[tid];
  __shared__ float red[128];
  __shared__ float sh_score;
  float acc = 0.0F;
  float m = -INFINITY;
  float l = 0.0F;
  for (std::uint32_t s = 0; s < n_pos; ++s) {
    const float* krow = k + static_cast<std::size_t>(s) * stride + hh * hd;
    red[tid] = qv * krow[tid];
    __syncthreads();
    if (tid == 0) {
      float d = 0.0F;
      for (std::uint32_t j = 0; j < hd; ++j) {
        d += red[j];
      }
      sh_score = d;
    }
    __syncthreads();
    const float score = sh_score;
    const float m_new = fmaxf(m, score);
    const float sc = expf(m - m_new);
    const float pp = expf(score - m_new);
    const float* vrow = v + static_cast<std::size_t>(s) * stride + hh * hd;
    acc = acc * sc + pp * vrow[tid];
    l = l * sc + pp;
    m = m_new;
    __syncthreads();
  }
  out[static_cast<std::size_t>(p) * stride + hh * hd + tid] = acc / l;
}

constexpr int kBlock = 256;

}  // namespace

void PatchEmbedIm2Col(const float* pixels, float* patchmat, std::uint32_t nx,
                      std::uint32_t ny, std::uint32_t n_px,
                      std::uint32_t n_patches, hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(n_patches) * kPatchVec;
  const std::uint32_t grid =
      static_cast<std::uint32_t>((total + kBlock - 1) / kBlock);
  Im2ColKernel<<<grid, kBlock, 0, stream>>>(pixels, patchmat, nx, ny, n_px,
                                            total);
}

void PosAdd(float* h, const float* pos_x, const float* pos_y,
            std::uint32_t n_px, std::uint32_t n_patches, std::uint32_t emb,
            hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(n_patches) * emb;
  const std::uint32_t grid =
      static_cast<std::uint32_t>((total + kBlock - 1) / kBlock);
  PosAddKernel<<<grid, kBlock, 0, stream>>>(h, pos_x, pos_y, n_px, emb, total);
}

void Rope2d(float* qk, const float* inv_freq, std::uint32_t n_px,
            std::uint32_t n_patches, std::uint32_t heads, std::uint32_t hd,
            hipStream_t stream) {
  const std::uint32_t quarter = hd / 4;
  const std::size_t total =
      static_cast<std::size_t>(n_patches) * heads * (2 * quarter);
  const std::uint32_t grid =
      static_cast<std::uint32_t>((total + kBlock - 1) / kBlock);
  Rope2dKernel<<<grid, kBlock, 0, stream>>>(qk, inv_freq, n_px, heads, hd,
                                            quarter, total);
}

void GegluQuick(const float* gate, const float* up, float* act,
                std::size_t count, hipStream_t stream) {
  const std::uint32_t grid =
      static_cast<std::uint32_t>((count + kBlock - 1) / kBlock);
  GegluQuickKernel<<<grid, kBlock, 0, stream>>>(gate, up, act, count);
}

void StdNorm(float* x, const float* bias, const float* scale,
             std::uint32_t n_tokens, std::uint32_t emb, hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(n_tokens) * emb;
  const std::uint32_t grid =
      static_cast<std::uint32_t>((total + kBlock - 1) / kBlock);
  StdNormKernel<<<grid, kBlock, 0, stream>>>(x, bias, scale, emb, total);
}

void AvgPool3(const float* h, float* pooled, std::uint32_t n_px,
              std::uint32_t n_py, std::uint32_t emb, float pool_scale,
              hipStream_t stream) {
  const std::uint32_t out_x = n_px / 3;
  const std::uint32_t out_y = n_py / 3;
  const std::size_t total = static_cast<std::size_t>(out_x) * out_y * emb;
  const std::uint32_t grid =
      static_cast<std::uint32_t>((total + kBlock - 1) / kBlock);
  AvgPool3Kernel<<<grid, kBlock, 0, stream>>>(h, pooled, n_px, out_x, emb,
                                              pool_scale, total);
}

void VisionAttention(const float* q, const float* k, const float* v, float* out,
                     std::uint32_t n_patches, std::uint32_t heads,
                     std::uint32_t hd, hipStream_t stream) {
  const dim3 grid(n_patches, heads);
  AttentionKernel<<<grid, hd, 0, stream>>>(q, k, v, out, n_patches, heads, hd);
}

}  // namespace gufo::models::gemma4::vision