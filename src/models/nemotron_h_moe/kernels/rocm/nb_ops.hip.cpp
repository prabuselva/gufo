#include "src/models/nemotron_h_moe/kernels/rocm/nb_ops.h"

#include <hip/hip_fp16.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kBlock = 256;
constexpr int kRouterBlock = 512;
constexpr int kRouterMaxExperts = 512;

// ggml block_q8_0 layout, kept local to avoid pulling the vendored headers.
struct BlockQ8_0 {
  __half scale;
  signed char qs[32];
};

inline int BlockCount(int64_t n) {
  return static_cast<int>((n + kBlock - 1) / kBlock);
}

__global__ void RmsNormKernel(const float* __restrict__ x,
                              const float* __restrict__ w,
                              float* __restrict__ out, int hidden, float eps) {
  const int row = blockIdx.x;
  const float* xr = x + static_cast<int64_t>(row) * hidden;
  float* orow = out + static_cast<int64_t>(row) * hidden;
  __shared__ float warp_sums[8];
  float acc = 0.0F;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = xr[i];
    acc += v * v;
  }
  // Warp reduce (wave-size agnostic), then across warps via shared memory.
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(~0ULL, acc, off);
  }
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0F;
    for (int i = 0; i < static_cast<int>(blockDim.x / warpSize); ++i) {
      total += warp_sums[i];
    }
    warp_sums[0] = rsqrtf(total / static_cast<float>(hidden) + eps);
  }
  __syncthreads();
  const float scale = warp_sums[0];
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    orow[i] = (xr[i] * scale) * w[i];
  }
}

__global__ void ScaleKernel(float* __restrict__ x, float s, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] *= s;
  }
}

__global__ void AddKernel(float* __restrict__ res,
                          const float* __restrict__ add, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    res[i] += add[i];
  }
}

__global__ void ReluSqrKernel(const float* __restrict__ src,
                              float* __restrict__ dst, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float v = src[i];
    dst[i] = v > 0.0F ? v * v : 0.0F;
  }
}

__global__ void DequantQ8_0RowKernel(const uint8_t* __restrict__ w,
                                     float* __restrict__ out, int cols) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cols) {
    return;
  }
  const int b = i >> 5;
  const int j = i & 31;
  const BlockQ8_0* block = reinterpret_cast<const BlockQ8_0*>(w);
  out[i] = static_cast<float>(block[b].qs[j]) * __half2float(block[b].scale);
}

__global__ void RouterKernel(const float* __restrict__ logits,
                             const float* __restrict__ bias, int n_experts,
                             int top_k, int norm, float scale,
                             int32_t* __restrict__ ids,
                             float* __restrict__ weights) {
  const int token = blockIdx.x;
  const float* lg = logits + static_cast<int64_t>(token) * n_experts;
  __shared__ float scores[kRouterMaxExperts];
  __shared__ int sel[16];
  __shared__ float num[16];
  for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
    scores[e] = 1.0F / (1.0F + expf(-lg[e])) + bias[e];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
    for (int k = 0; k < top_k; ++k) {
      // Serial top-k selection on one thread mirrors the oracle loop
      // exactly, including the tie-to-lowest-index rule.
      int best = -1;
      for (int e = 0; e < n_experts; ++e) {
        const float v = scores[e];
        if (v == -FLT_MAX) {
          continue;
        }
        if (best < 0 || v > scores[best]) {
          best = e;
        }
      }
      sel[k] = best;
      if (best < 0) {
        break;
      }
      total = k + 1;
      scores[best] = -FLT_MAX;
      num[k] = 1.0F / (1.0F + expf(-lg[best]));
    }
    float sum = 0.0F;
    for (int k = 0; k < total; ++k) {
      sum += num[k];
    }
    if (norm) {
      sum = sum < 6.103515625e-5F ? 6.103515625e-5F : sum;
    }
    const float inv = sum > 0.0F ? 1.0F / sum : 0.0F;
    for (int k = 0; k < top_k; ++k) {
      const bool chosen = k < total;
      ids[static_cast<int64_t>(token) * top_k + k] =
          chosen ? sel[k] : static_cast<int32_t>(-1);
      weights[static_cast<int64_t>(token) * top_k + k] =
          chosen ? num[k] * inv * scale : 0.0F;
    }
  }
}

__global__ void MoeCombineKernel(float* __restrict__ dst,
                                 const float* __restrict__ src,
                                 const float* __restrict__ w, int m,
                                 int top_k) {
  const int token = blockIdx.y;
  const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= m) {
    return;
  }
  float acc = 0.0F;
  for (int u = 0; u < top_k; ++u) {
    acc += w[static_cast<int64_t>(token) * top_k + u] *
           src[(static_cast<int64_t>(token) * top_k + u) * m + j];
  }
  dst[static_cast<int64_t>(token) * m + j] = acc;
}

int CheckLaunch(hipError_t err, const char* tag) {
  if (err != hipSuccess) {
    return -1;
  }
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

}  // namespace

extern "C" int nbl_rms_norm(const float* x, const float* w, float* out,
                            int rows, int hidden, float eps,
                            hipStream_t stream) {
  if (!x || !w || !out || rows <= 0 || hidden <= 0) {
    return -1;
  }
  RmsNormKernel<<<rows, kBlock, 0, stream>>>(x, w, out, hidden, eps);
  return CheckLaunch(hipGetLastError(), "nbl_rms_norm");
}

extern "C" int nbl_scale(float* x, float s, int64_t n, hipStream_t stream) {
  if (!x || n <= 0) {
    return -1;
  }
  ScaleKernel<<<BlockCount(n), kBlock, 0, stream>>>(x, s, n);
  return CheckLaunch(hipGetLastError(), "nbl_scale");
}

extern "C" int nbl_add(float* res, const float* add, int64_t n,
                       hipStream_t stream) {
  if (!res || !add || n <= 0) {
    return -1;
  }
  AddKernel<<<BlockCount(n), kBlock, 0, stream>>>(res, add, n);
  return CheckLaunch(hipGetLastError(), "nbl_add");
}

extern "C" int nbl_relu_sqr(const float* src, float* dst, int64_t n,
                            hipStream_t stream) {
  if (!src || !dst || n <= 0) {
    return -1;
  }
  ReluSqrKernel<<<BlockCount(n), kBlock, 0, stream>>>(src, dst, n);
  return CheckLaunch(hipGetLastError(), "nbl_relu_sqr");
}

extern "C" int nbl_dequant_row_q8_0(const void* w, int64_t row, int cols,
                                    float* out, hipStream_t stream) {
  if (!w || !out || row < 0 || cols <= 0 || cols % 32 != 0) {
    return -1;
  }
  const uint8_t* base = static_cast<const uint8_t*>(w) +
                        row * (static_cast<int64_t>(cols) / 32 *
                               static_cast<int64_t>(sizeof(BlockQ8_0)));
  DequantQ8_0RowKernel<<<BlockCount(cols), kBlock, 0, stream>>>(base, out,
                                                                cols);
  return CheckLaunch(hipGetLastError(), "nbl_dequant_row_q8_0");
}

extern "C" int nbl_router(const float* logits, const float* bias, int n_experts,
                          int top_k, int norm, float scale, int n_tokens,
                          int32_t* ids, float* weights, hipStream_t stream) {
  if (!logits || !bias || !ids || !weights || n_experts <= 0 ||
      n_experts > kRouterMaxExperts || top_k <= 0 || top_k > 16 ||
      n_tokens <= 0) {
    return -1;
  }
  RouterKernel<<<n_tokens, kRouterBlock, 0, stream>>>(
      logits, bias, n_experts, top_k, norm, scale, ids, weights);
  return CheckLaunch(hipGetLastError(), "nbl_router");
}

extern "C" int nbl_moe_combine(float* dst, const float* src, const float* w,
                               int m, int top_k, int n_tokens,
                               hipStream_t stream) {
  if (!dst || !src || !w || m <= 0 || top_k <= 0 || n_tokens <= 0) {
    return -1;
  }
  const dim3 grid(BlockCount(m), n_tokens);
  MoeCombineKernel<<<grid, kBlock, 0, stream>>>(dst, src, w, m, top_k);
  return CheckLaunch(hipGetLastError(), "nbl_moe_combine");
}
