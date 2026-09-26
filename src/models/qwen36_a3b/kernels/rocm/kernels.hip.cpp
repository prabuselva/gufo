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
    for (std::uint32_t t = 0; t + 1 < kernel; ++t) {
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

void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream) {
  const dim3 block(std::max<std::uint32_t>(n_experts, 32U));
  const std::size_t shared = n_experts * sizeof(float);
  RouterTopKKernel<<<tokens, block, shared, stream>>>(logits, stride, ids,
                                                      weights, n_experts, k);
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

void AttentionDecode(const float* q, const float* k_cache, const float* v_cache,
                     const float* gate, float* out, float* scratch,
                     std::uint32_t n_kv, std::uint32_t heads,
                     std::uint32_t kv_heads, std::uint32_t head_dim,
                     float scale, hipStream_t stream) {
  const dim3 block(head_dim);
  AttentionDecodeKernel<<<heads, block, 0, stream>>>(q, k_cache, v_cache, gate,
                                                     out, scratch, n_kv, heads,
                                                     kv_heads, head_dim, scale);
}

}  // namespace gufo::models::qwen36_a3b::rocm