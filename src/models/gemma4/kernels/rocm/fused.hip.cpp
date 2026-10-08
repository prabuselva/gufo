#include "src/models/gemma4/kernels/rocm/fused.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>

namespace gufo::models::gemma4::rocm {
namespace {

// Elementwise helpers mirror the scalar oracle exactly (float math, same
// formulas) so a kernel and the CPU reference agree to rounding.
__device__ __forceinline__ float DGelu(float x) {
  constexpr float kAlpha = 0.7978845608028654F;  // sqrt(2/pi)
  const float inner = kAlpha * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + tanhf(inner));
}

__device__ __forceinline__ double WarpReduceSum(double v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
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
  if ((dim & 3U) == 0U) {
    // Vectorized path: 16-byte loads/stores and a float per-thread partial
    // (the block reduce stays in double over the <=1024 partials). Measured
    // ~2x faster than the scalar double path at the decode shape (rows=1,
    // dim=2816); worst-relative error vs the double oracle is ~1.5e-7.
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    const std::uint32_t n4 = dim / 4U;
    float ss = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
      const float4 v = x4[i];
      ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    const float scale =
        1.0F / sqrtf(static_cast<float>(
                         BlockReduceSum(static_cast<double>(ss), reduce) /
                         static_cast<double>(dim)) +
                     eps);
    float4* orow = reinterpret_cast<float4*>(out + row * dim);
    if (gamma != nullptr) {
      const float4* g4 = reinterpret_cast<const float4*>(gamma);
      for (std::uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = x4[i];
        const float4 g = g4[i];
        orow[i] = {v.x * scale * g.x, v.y * scale * g.y, v.z * scale * g.z,
                   v.w * scale * g.w};
      }
    } else {
      for (std::uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = x4[i];
        orow[i] = {v.x * scale, v.y * scale, v.z * scale, v.w * scale};
      }
    }
    return;
  }
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

// Binary16-output RMSNorm. The sum-of-squares reduction and the float scale are
// identical to RmsNormKernel (double accumulator over the row, float rsqrt);
// only the store is F16. dim is a multiple of 4 for every gemma4 norm, so the
// vectorized float4 load path covers it; the scalar path is kept for safety.
__global__ void RmsNormKernelHalf(const float* x, const float* gamma,
                                  __half* out, std::uint32_t dim, float eps) {
  __shared__ double reduce[32];
  const std::size_t row = blockIdx.x;
  const float* xr = x + row * dim;
  if ((dim & 3U) == 0U) {
    const float4* x4 = reinterpret_cast<const float4*>(xr);
    const std::uint32_t n4 = dim / 4U;
    float ss = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
      const float4 v = x4[i];
      ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    const float scale =
        1.0F / sqrtf(static_cast<float>(
                         BlockReduceSum(static_cast<double>(ss), reduce) /
                         static_cast<double>(dim)) +
                     eps);
    __half* orow = out + row * dim;
    // Materialize the product as a float before the F16 store: the compiler
    // otherwise fuses mul.f32 -> cvt.f16.f32 (one rounding), while the float
    // RmsNormKernel + narrow() round twice, diverging at half-ULP ties. The
    // volatile temp forces the float round-trip; applies to all stores below.
    if (gamma != nullptr) {
      const float4* g4 = reinterpret_cast<const float4*>(gamma);
      for (std::uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = x4[i];
        const float4 g = g4[i];
        volatile float rx = v.x * scale * g.x;
        volatile float ry = v.y * scale * g.y;
        volatile float rz = v.z * scale * g.z;
        volatile float rw = v.w * scale * g.w;
        orow[4 * i] = __float2half(rx);
        orow[4 * i + 1] = __float2half(ry);
        orow[4 * i + 2] = __float2half(rz);
        orow[4 * i + 3] = __float2half(rw);
      }
    } else {
      for (std::uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = x4[i];
        volatile float rx = v.x * scale;
        volatile float ry = v.y * scale;
        volatile float rz = v.z * scale;
        volatile float rw = v.w * scale;
        orow[4 * i] = __float2half(rx);
        orow[4 * i + 1] = __float2half(ry);
        orow[4 * i + 2] = __float2half(rz);
        orow[4 * i + 3] = __float2half(rw);
      }
    }
    return;
  }
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const double v = xr[i];
    ss += v * v;
  }
  const float scale =
      1.0F / sqrtf(static_cast<float>(BlockReduceSum(ss, reduce) /
                                      static_cast<double>(dim)) +
                   eps);
  __half* orow = out + row * dim;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float g = gamma != nullptr ? gamma[i] : 1.0F;
    volatile float r = xr[i] * scale * g;
    orow[i] = __float2half(r);
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
  if ((dim & 3U) == 0U) {
    const float4* a4 = reinterpret_cast<const float4*>(addend);
    float4* x4 = reinterpret_cast<float4*>(x);
    const std::uint32_t n4 = dim / 4U;
    float ss = 0.0F;
    for (std::uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
      const float4 xv = x4[i];
      const float4 av = a4[i];
      const float4 sum = {xv.x + av.x, xv.y + av.y, xv.z + av.z, xv.w + av.w};
      x4[i] = sum;
      ss += sum.x * sum.x + sum.y * sum.y + sum.z * sum.z + sum.w * sum.w;
    }
    const float scale =
        1.0F / sqrtf(static_cast<float>(
                         BlockReduceSum(static_cast<double>(ss), reduce) /
                         static_cast<double>(dim)) +
                     eps);
    float4* orow = reinterpret_cast<float4*>(out);
    if (gamma != nullptr) {
      const float4* g4 = reinterpret_cast<const float4*>(gamma);
      for (std::uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = x4[i];
        const float4 g = g4[i];
        orow[i] = {v.x * scale * g.x, v.y * scale * g.y, v.z * scale * g.z,
                   v.w * scale * g.w};
      }
    } else {
      for (std::uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = x4[i];
        orow[i] = {v.x * scale, v.y * scale, v.z * scale, v.w * scale};
      }
    }
    return;
  }
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

__global__ void ExpertCountsKernel(const std::int32_t* ids,
                                   std::uint32_t* counts, std::size_t slots) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < slots && ids[i] >= 0) {
    atomicAdd(counts + ids[i], 1u);
  }
}

__global__ void MoeEpilogueKernel(const float* expert_out, const float* weights,
                                  const std::int32_t* ids,
                                  const float* expert_scale, float* out,
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
      const float scale =
          expert_scale != nullptr ? expert_scale[ids[slot]] : 1.0F;
      acc += static_cast<double>(weights[slot]) * scale *
             expert_out[slot * dim + i];
    }
    out[index] = static_cast<float>(acc);
  }
}

template<typename T>
__device__ __forceinline__ T GegluPair(const T* gu, std::uint32_t ff);

template<>
__device__ __forceinline__ float GegluPair<float>(const float* gu,
                                                  std::uint32_t ff) {
  return DGelu(gu[0]) * gu[ff];
}

template<>
__device__ __forceinline__ __half GegluPair<__half>(const __half* gu,
                                                    std::uint32_t ff) {
  const float gate = __half2float(gu[0]);
  const float up = __half2float(gu[ff]);
  return __float2half(DGelu(gate) * up);
}

template<typename T>
__global__ void GegluKernel(const T* gu, T* act, std::uint32_t ff,
                            std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    const std::size_t row = i / ff;
    const std::size_t col = i % ff;
    act[i] = GegluPair<T>(gu + row * (2U * ff) + col, ff);
  }
}

__global__ void ScaleKernel(float* x, float k, std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    x[i] *= k;
  }
}

__global__ void MulKernel(float* x, const float* scale, std::size_t count,
                          std::size_t period) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    x[i] *= scale[i % period];
  }
}

__global__ void SoftcapKernel(float* x, float cap, std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    x[i] = cap * tanhf(x[i] / cap);
  }
}

__global__ void GegluSeparateKernel(const float* gate, const float* up,
                                    float* act, std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    act[i] = DGelu(gate[i]) * up[i];
  }
}

// Binary16-output shared-FFN geglu: the same gelu_tanh(gate) * up in float,
// stored as F16 so the down GEMM consumes the act row directly. The volatile
// temp is the same float-materialization barrier as RmsNormKernelHalf: without
// it the compiler fuses the final mul into the cvt and single-rounds, whereas
// the FP32 path stores the float act and NarrowHalf rounds it again.
__global__ void GegluSeparateHalfKernel(const float* gate, const float* up,
                                        __half* act, std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    volatile float r = DGelu(gate[i]) * up[i];
    act[i] = __float2half(r);
  }
}

}  // namespace

void ScaleInPlace(float* x, float k, std::size_t count, hipStream_t stream) {
  if (count == 0) {
    return;
  }
  const std::size_t block = 256;
  const std::size_t grid =
      std::min<std::size_t>((count + block - 1) / block, 65535);
  ScaleKernel<<<grid, block, 0, stream>>>(x, k, count);
}

void MulInPlace(float* x, const float* scale, std::size_t count,
                std::size_t period, hipStream_t stream) {
  if (count == 0) {
    return;
  }
  const std::size_t block = 256;
  const std::size_t grid =
      std::min<std::size_t>((count + block - 1) / block, 65535);
  MulKernel<<<grid, block, 0, stream>>>(x, scale, count, period);
}

void SoftcapInPlace(float* x, float cap, std::size_t count,
                    hipStream_t stream) {
  if (count == 0) {
    return;
  }
  const std::size_t block = 256;
  const std::size_t grid =
      std::min<std::size_t>((count + block - 1) / block, 65535);
  SoftcapKernel<<<grid, block, 0, stream>>>(x, cap, count);
}

void GegluF32Separate(const float* gate, const float* up, float* act,
                      std::size_t count, hipStream_t stream) {
  if (count == 0) {
    return;
  }
  const std::size_t block = 256;
  const std::size_t grid =
      std::min<std::size_t>((count + block - 1) / block, 65535);
  GegluSeparateKernel<<<grid, block, 0, stream>>>(gate, up, act, count);
}

void GegluF16Separate(const float* gate, const float* up, __half* act,
                      std::size_t count, hipStream_t stream) {
  if (count == 0) {
    return;
  }
  const std::size_t block = 256;
  const std::size_t grid =
      std::min<std::size_t>((count + block - 1) / block, 65535);
  GegluSeparateHalfKernel<<<grid, block, 0, stream>>>(gate, up, act, count);
}

void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t rows, std::uint32_t dim, float eps,
                 hipStream_t stream) {
  const dim3 block(std::min<std::uint32_t>(dim, 256U));
  RmsNormKernel<<<rows, block, 0, stream>>>(x, gamma, out, dim, eps);
}

void RmsNormRowsHalf(const float* x, const float* gamma, __half* out,
                     std::uint32_t rows, std::uint32_t dim, float eps,
                     hipStream_t stream) {
  const dim3 block(std::min<std::uint32_t>(dim, 256U));
  RmsNormKernelHalf<<<rows, block, 0, stream>>>(x, gamma, out, dim, eps);
}

void FusedAddRmsNorm(float* x, const float* addend, const float* gamma,
                     float* out, std::uint32_t dim, float eps,
                     hipStream_t stream) {
  const dim3 block(std::min<std::uint32_t>(dim, 256U));
  FusedAddRmsNormKernel<<<1, block, 0, stream>>>(x, addend, gamma, out, dim,
                                                 eps);
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
                 const std::int32_t* ids, const float* expert_scale, float* out,
                 std::uint32_t tokens, std::uint32_t k, std::uint32_t dim,
                 hipStream_t stream) {
  const std::size_t count = static_cast<std::size_t>(tokens) * dim;
  const std::size_t block = 256;
  const std::size_t grid = (count + block - 1) / block;
  MoeEpilogueKernel<<<grid, block, 0, stream>>>(
      expert_out, weights, ids, expert_scale, out, k, dim, count);
}

void GegluF32(const float* gu, float* act, std::size_t count, std::uint32_t ff,
              hipStream_t stream) {
  if (count == 0) {
    return;
  }
  const std::size_t block = 256;
  const std::size_t grid =
      std::min<std::size_t>((count + block - 1) / block, 65535);
  GegluKernel<float><<<grid, block, 0, stream>>>(gu, act, ff, count);
}

void GegluF16(const __half* gu, __half* act, std::size_t count,
              std::uint32_t ff, hipStream_t stream) {
  if (count == 0) {
    return;
  }
  const std::size_t block = 256;
  const std::size_t grid =
      std::min<std::size_t>((count + block - 1) / block, 65535);
  GegluKernel<__half><<<grid, block, 0, stream>>>(gu, act, ff, count);
}

}  // namespace gufo::models::gemma4::rocm