#include "src/models/ling3_flash/kernels/rocm/kernels.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
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

/// Block-wide argmax over one candidate per thread, choosing the lowest index
/// on exact-value ties; reproduces a first-strict-max ascending scan of an
/// array by a single thread. `vsm`/`ism` need (blockDim.x + 31) / 32 + 1
/// entries and the caller must sync before the next call on the same arrays.
__device__ __forceinline__ std::uint32_t BlockArgMax(float v, std::uint32_t idx,
                                                     float* vsm,
                                                     std::uint32_t* ism) {
  const unsigned lane = threadIdx.x & 31U;
  const unsigned warp = threadIdx.x >> 5;
  const unsigned warps = (blockDim.x + 31U) >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(~0ULL, v, off);
    const std::uint32_t oi = __shfl_down_sync(~0ULL, idx, off);
    if (ov > v || (ov == v && oi < idx)) {
      v = ov;
      idx = oi;
    }
  }
  if (lane == 0U) {
    vsm[warp] = v;
    ism[warp] = idx;
  }
  __syncthreads();
  if (warp == 0U) {
    const bool owned = lane < warps;
    v = owned ? vsm[lane] : kNegInf;
    idx = owned ? ism[lane] : 0xFFFFFFFFU;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(~0ULL, v, off);
      const std::uint32_t oi = __shfl_down_sync(~0ULL, idx, off);
      if (ov > v || (ov == v && oi < idx)) {
        v = ov;
        idx = oi;
      }
    }
    if (lane == 0U) {
      ism[warps] = idx;
    }
  }
  __syncthreads();
  return ism[warps];
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

// ---------------------------------------------------------------------------
// Prefill tier: the same KDA recurrence over a token chunk, sequential over
// tokens but fully on-device with the state resident in registers. The
// front-end kernels are written so every value feeding the recurrence is
// bit-identical to what the single-token decode chain produces for the same
// inputs (same tap order in the conv, same reductions in the norms); the
// recurrence reorganizes the dot products into register stripes plus
// cross-lane reductions, so the attention outputs match the per-token decode
// chain up to float reassociation, and the state matches column-block by
// column-block.

/// Batched causal depthwise conv + SiLU over a chunk for one q/k/v stream.
/// `history` holds the `kern-1` raw projected rows before the chunk and is
/// left untouched (KdaHistoryUpdate advances it); taps read in-chunk rows of
/// `proj` and fall back to `history` for the leading taps.
__global__ void KdaConvSiluPrefillKernel(
    const float* __restrict__ proj, const float* __restrict__ conv_w,
    const float* __restrict__ history, float* __restrict__ out,
    std::uint32_t tokens, std::uint32_t inner, std::uint32_t kern) {
  const std::size_t total = static_cast<std::size_t>(tokens) * inner;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const std::uint32_t t = static_cast<std::uint32_t>(idx / inner);
    const std::uint32_t ch = static_cast<std::uint32_t>(idx % inner);
    const float* cw = conv_w + static_cast<std::size_t>(ch) * kern;
    float acc = cw[kern - 1] * proj[idx];
    for (std::uint32_t k = 0; k + 1 < kern; ++k) {
      const std::int64_t s = static_cast<std::int64_t>(t) -
                             static_cast<std::int64_t>(kern - 1) +
                             static_cast<std::int64_t>(k);
      const float xv =
          s >= 0 ? proj[static_cast<std::size_t>(s) * inner + ch]
                 : history[static_cast<std::size_t>(
                               static_cast<std::int64_t>(kern - 1) + s) *
                               inner +
                           ch];
      acc += cw[k] * xv;
    }
    out[idx] = SiluD(acc);
  }
}

/// Advances one conv history past a chunk: new row j is the raw projected row
/// at absolute position (chunk_end - (kern-1) + j). Reads the old `history`
/// and the chunk `proj`, writes a disjoint `out` so there is no aliasing.
__global__ void KdaHistoryUpdateKernel(const float* __restrict__ proj,
                                       const float* __restrict__ history,
                                       float* __restrict__ out,
                                       std::uint32_t tokens,
                                       std::uint32_t inner,
                                       std::uint32_t kern) {
  const std::size_t total = static_cast<std::size_t>(kern - 1) * inner;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const std::uint32_t j = static_cast<std::uint32_t>(idx / inner);
    const std::uint32_t ch = static_cast<std::uint32_t>(idx % inner);
    const std::int64_t src = static_cast<std::int64_t>(tokens) -
                             static_cast<std::int64_t>(kern - 1) +
                             static_cast<std::int64_t>(j);
    out[idx] = src >= 0
                   ? proj[static_cast<std::size_t>(src) * inner + ch]
                   : history[static_cast<std::size_t>(
                                 static_cast<std::int64_t>(kern - 1) + src) *
                                 inner +
                             ch];
  }
}

/// decay[i] = exp(gate_lb * sigmoid(a[h] * f[i])) over all (token, head, key),
/// hoisted out of the serial token loop; same expression as the decode
/// KdaDeltaKernel, so bit-identical.
__global__ void KdaPrepDecayKernel(const float* __restrict__ f,
                                   const float* __restrict__ a,
                                   float* __restrict__ decay,
                                   std::uint32_t heads, std::uint32_t d,
                                   float gate_lb, std::size_t total) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       i < total; i += stride) {
    const std::uint32_t h = static_cast<std::uint32_t>((i / d) % heads);
    decay[i] = ::expf(gate_lb * SigmoidD(a[h] * f[i]));
  }
}

/// beta_pre[t*heads+h] = sigmoid(beta[t*heads+h]), hoisted out of the loop.
__global__ void KdaPrepBetaKernel(const float* __restrict__ beta,
                                  float* __restrict__ beta_pre,
                                  std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       i < count; i += stride) {
    beta_pre[i] = SigmoidD(beta[i]);
  }
}

// Four-lane butterfly via the ALU-path DPP row-xmask modifier (gfx10+): the
// xor masks stay inside an aligned 4-lane group, so the sum is valid in every
// lane of the group that cooperates on one state row.
__device__ __forceinline__ float RowXorAdd(float v) {
  const int y2 = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v),
                                             0x162, 0xF, 0xF, false);
  v += __builtin_bit_cast(float, y2);
  const int y1 = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v),
                                             0x161, 0xF, 0xF, false);
  v += __builtin_bit_cast(float, y1);
  return v;
}

/// kq_pre[t*heads+h] = dot(kn, qn) per (token, head) for the recurrence
/// output term. qn already carries q_scale, so the loop stores
/// attn = po + delta * kq with no extra scale.
__global__ void KdaPrepKqKernel(const float* __restrict__ qn,
                                const float* __restrict__ kn,
                                float* __restrict__ kq_pre, std::uint32_t pairs,
                                std::uint32_t heads, std::uint32_t d) {
  constexpr int kVec = 8;  // 32 columns / 4 lanes
  const std::uint32_t pair =
      blockIdx.x * 8U + (static_cast<std::uint32_t>(threadIdx.x) >> 2);
  const int seg = static_cast<int>(threadIdx.x) & 3;
  if (pair >= pairs) {
    return;
  }
  const std::uint32_t h = pair % heads;
  const std::uint32_t t = pair / heads;
  const std::size_t base = (static_cast<std::size_t>(t) * heads + h) * d;
  const float4* qp = reinterpret_cast<const float4*>(qn + base);
  const float4* kp = reinterpret_cast<const float4*>(kn + base);
  float kq = 0.0F;
#pragma unroll
  for (int vi = 0; vi < kVec; ++vi) {
    const float4 q4 = qp[seg * kVec + vi];
    const float4 k4 = kp[seg * kVec + vi];
    kq += k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
  }
  kq = RowXorAdd(kq);
  if (seg == 0) {
    kq_pre[pair] = kq;
  }
}

/// 32-lane full butterfly sum (valid in every lane; blocks are single warps).
__device__ __forceinline__ float WarpSumF(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(~0ULL, v, off);
  }
  return v;
}

/// Warp-resident KDA delta-rule recurrence over a chunk with per-token state
/// snapshots. One warp owns kRows state rows of one head; each lane keeps
/// kCols state columns (head_dim = 32 * kCols) in registers across the whole
/// chunk. The per-channel key-axis decay is read from the precomputed
/// `decay` stripe, and attn reuses dot(s_new, q) = po + delta * dot(k, q).
template<std::uint32_t kRows, std::uint32_t kCols>
__global__ void KdaDeltaLoopKernel(
    const float* __restrict__ qn, const float* __restrict__ kn,
    const float* __restrict__ v, const float* __restrict__ decay,
    const float* __restrict__ beta_pre, const float* __restrict__ kq_pre,
    float* __restrict__ state, float* __restrict__ attn, std::uint32_t tokens,
    std::uint32_t heads, std::uint32_t head_dim, float* __restrict__ snap,
    std::uint32_t snap_rows) {
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t j0 = blockIdx.x * kRows;
  const std::uint32_t lane = threadIdx.x;
  const std::size_t inner = static_cast<std::size_t>(heads) * head_dim;

  float s[kRows][kCols];
  for (std::uint32_t r = 0; r < kRows; ++r) {
    const float* row =
        state + (static_cast<std::size_t>(h) * head_dim + j0 + r) * head_dim;
    for (std::uint32_t c = 0; c < kCols; ++c) {
      s[r][c] = row[lane + c * 32U];
    }
  }

  for (std::uint32_t t = 0; t < tokens; ++t) {
    const std::size_t off =
        (static_cast<std::size_t>(t) * heads + h) * head_dim + lane;
    float kc[kCols];
    float qc[kCols];
    float dc[kCols];
    for (std::uint32_t c = 0; c < kCols; ++c) {
      kc[c] = kn[off + c * 32U];
      qc[c] = qn[off + c * 32U];
      dc[c] = decay[off + c * 32U];
    }
    const float b = beta_pre[static_cast<std::size_t>(t) * heads + h];
    const float kq = kq_pre[static_cast<std::size_t>(t) * heads + h];
    float pu[kRows];
    float po[kRows];
    for (std::uint32_t r = 0; r < kRows; ++r) {
      float du = 0.0F;
      float dq = 0.0F;
      for (std::uint32_t c = 0; c < kCols; ++c) {
        s[r][c] *= dc[c];
        du += s[r][c] * kc[c];
        dq += s[r][c] * qc[c];
      }
      pu[r] = WarpSumF(du);
      po[r] = WarpSumF(dq);
    }
    for (std::uint32_t r = 0; r < kRows; ++r) {
      const float v_t =
          v[(static_cast<std::size_t>(t) * heads + h) * head_dim + j0 + r];
      const float delta = (v_t - pu[r]) * b;
      for (std::uint32_t c = 0; c < kCols; ++c) {
        s[r][c] += delta * kc[c];
      }
      if (lane == r) {
        attn[(static_cast<std::size_t>(t) * heads + h) * head_dim + j0 + r] =
            po[r] + delta * kq;
      }
    }
    if (snap != nullptr && t < snap_rows) {
      for (std::uint32_t r = 0; r < kRows; ++r) {
        float* row =
            snap +
            ((static_cast<std::size_t>(t) * heads + h) * head_dim + j0 + r) *
                head_dim;
        for (std::uint32_t c = 0; c < kCols; ++c) {
          row[lane + c * 32U] = s[r][c];
        }
      }
    }
  }
  for (std::uint32_t r = 0; r < kRows; ++r) {
    float* row =
        state + (static_cast<std::size_t>(h) * head_dim + j0 + r) * head_dim;
    for (std::uint32_t c = 0; c < kCols; ++c) {
      row[lane + c * 32U] = s[r][c];
    }
  }
}

/// Row-split KDA delta-rule recurrence for a prefill chunk (no snapshots).
/// head_dim == 128: four lanes cooperate on one 128-wide state row
/// (32 columns each as eight float4), the per-row reductions run as a
/// four-lane butterfly, and the key-axis decay is consumed as a per-lane
/// float4 stripe so the exp/sigmoid work is fully hoisted to KdaPrep.
/// qn carries q_scale, so attn = po + delta * kq.
template<std::uint32_t kRowsPerLane>
__global__ void __launch_bounds__(256) KdaDeltaLoopRowSplitKernel(
    const float* __restrict__ qn, const float* __restrict__ kn,
    const float* __restrict__ v, const float* __restrict__ decay,
    const float* __restrict__ beta_pre, const float* __restrict__ kq_pre,
    float* __restrict__ state, float* __restrict__ attn, std::uint32_t tokens,
    std::uint32_t heads, std::uint32_t head_dim) {
  constexpr int kLanesPerRow = 4;
  constexpr int kRowGroups = 32 / kLanesPerRow;            // 8
  constexpr int kRowsPerWave = kRowGroups * kRowsPerLane;  // 8 or 16
  constexpr int kVec = 8;                                  // 32 columns / 4
  const std::uint32_t h = blockIdx.y;
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int seg = lane % kLanesPerRow;
  const int grp = lane / kLanesPerRow;
  const int row0 = static_cast<int>(blockIdx.x) * (kRowsPerWave * 8) +
                   wave * kRowsPerWave + grp;
  const std::size_t inner = static_cast<std::size_t>(heads) * head_dim;

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

  const float* q_base = qn + static_cast<std::size_t>(h) * head_dim;
  const float* k_base = kn + static_cast<std::size_t>(h) * head_dim;
  const float* v_base = v + static_cast<std::size_t>(h) * head_dim;
  const float* d_base = decay + static_cast<std::size_t>(h) * head_dim;
  const float* bp_base = beta_pre + h;
  const float* kqp_base = kq_pre + h;
  float* o_base = attn + static_cast<std::size_t>(h) * head_dim;

  float4 kv[kVec];
  float4 qv[kVec];
  float4 dv[kVec];
  {
    const float4* kp = reinterpret_cast<const float4*>(k_base);
    const float4* qp = reinterpret_cast<const float4*>(q_base);
    const float4* dp = reinterpret_cast<const float4*>(d_base);
#pragma unroll
    for (int vi = 0; vi < kVec; ++vi) {
      kv[vi] = kp[seg * kVec + vi];
      qv[vi] = qp[seg * kVec + vi];
      dv[vi] = dp[seg * kVec + vi];
    }
  }
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
    float4 dnx[kVec];
    float b_n = 0.0F;
    float kq_n = 0.0F;
    float v_n[kRowsPerLane];
    if (t + 1 < tokens) {
      const float4* kp = reinterpret_cast<const float4*>(k_base + inner);
      const float4* qp = reinterpret_cast<const float4*>(q_base + inner);
      const float4* dp = reinterpret_cast<const float4*>(d_base + inner);
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        knx[vi] = kp[seg * kVec + vi];
        qnx[vi] = qp[seg * kVec + vi];
        dnx[vi] = dp[seg * kVec + vi];
      }
      b_n = *(bp_base + heads);
      kq_n = *(kqp_base + heads);
#pragma unroll
      for (int r = 0; r < kRowsPerLane; ++r) {
        v_n[r] = v_base[inner + row0 + r * kRowGroups];
      }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
      float u = 0.0F;
      float p = 0.0F;
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        float4 s4 = s_reg[r][vi];
        const float4 d4 = dv[vi];
        s4.x *= d4.x;
        s4.y *= d4.y;
        s4.z *= d4.z;
        s4.w *= d4.w;
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
        o_base[row] = p + delta * kq;
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
    q_base += inner;
    k_base += inner;
    v_base += inner;
    d_base += inner;
    bp_base += heads;
    kqp_base += heads;
    o_base += inner;
#pragma unroll
    for (int vi = 0; vi < kVec; ++vi) {
      kv[vi] = knx[vi];
      qv[vi] = qnx[vi];
      dv[vi] = dnx[vi];
    }
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

/// Per-token gated RMS norm of the attention output, batched over the chunk.
/// Matches the output-norm tail of the decode KdaDeltaKernel (same
/// BlockSumDouble reduction, same element formula), so with bit-identical
/// attention rows the result is bit-identical.
__global__ void KdaOutNormPrefillKernel(float* __restrict__ attn,
                                        const float* __restrict__ g,
                                        const float* __restrict__ norm_w,
                                        std::uint32_t d, float eps) {
  __shared__ double smem[8];
  const std::size_t off = static_cast<std::size_t>(blockIdx.x) * d;
  float* row = attn + off;
  const double x = static_cast<double>(row[threadIdx.x]);
  const double ss = BlockSumDouble(x * x, smem);
  const float scale =
      1.0F / ::sqrtf(static_cast<float>(ss / static_cast<double>(d)) + eps);
  row[threadIdx.x] = row[threadIdx.x] * scale * norm_w[threadIdx.x] *
                     SigmoidD(g[off + threadIdx.x]);
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
  // One block per token row; the decode launcher keeps grid == 1 with a
  // row-offset base pointer, so row == 0 reproduces the single-row path.
  const std::uint32_t row = blockIdx.x;
  logits += static_cast<std::size_t>(row) * n_exp;
  ids += static_cast<std::size_t>(row) * used;
  weights += static_cast<std::size_t>(row) * used;
  const std::uint32_t per_group = n_exp / groups;
  __shared__ bool keep[64];
  __shared__ float red_v[20];
  __shared__ std::uint32_t red_i[20];
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
    if (threadIdx.x < groups) {
      keep[threadIdx.x] = false;
    }
    __syncthreads();
    for (std::uint32_t s = 0; s < groups_used; ++s) {
      const std::uint32_t g = threadIdx.x;
      const std::uint32_t best_g =
          BlockArgMax((g < groups && !keep[g]) ? group_scores[g] : kNegInf, g,
                      red_v, red_i);
      if (threadIdx.x == best_g) {
        keep[best_g] = true;
      }
      __syncthreads();
    }
    for (std::uint32_t g = threadIdx.x; g < groups; g += blockDim.x) {
      if (!keep[g]) {
        for (std::uint32_t i = 0; i < per_group; ++i) {
          selection[static_cast<std::size_t>(g) * per_group + i] = kNegInf;
        }
      }
    }
    __syncthreads();
  }
  // Greedy top-k over experts: one block-wide (max, lowest-index) argmax per
  // round replaces the former thread-0 serial scan chain.
  double sum = 0.0;
  for (std::uint32_t s = 0; s < used; ++s) {
    float v = kNegInf;
    std::uint32_t idx = 0xFFFFFFFFU;
    for (std::uint32_t e = threadIdx.x; e < n_exp; e += blockDim.x) {
      const float value = selection[e];
      if (value > v) {
        v = value;
        idx = e;
      }
    }
    const std::uint32_t best_e = BlockArgMax(v, idx, red_v, red_i);
    if (threadIdx.x == best_e) {
      selection[best_e] = kNegInf;
    }
    if (threadIdx.x == 0U) {
      ids[s] = static_cast<std::int32_t>(best_e);
      weights[s] = probs[best_e];
      sum += static_cast<double>(probs[best_e]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
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

__global__ void RmsNormRowsKernel(const float* __restrict__ x,
                                  const float* __restrict__ w, float eps,
                                  float* __restrict__ out, std::uint32_t n) {
  __shared__ double smem[8];
  const std::size_t off = static_cast<std::size_t>(blockIdx.x) * n;
  const float* xr = x + off;
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const double v = static_cast<double>(xr[i]);
    ss += v * v;
  }
  const float scale =
      1.0F / ::sqrtf(static_cast<float>(BlockSumDouble(ss, smem) /
                                        static_cast<double>(n)) +
                     eps);
  float* orow = out + off;
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    orow[i] = xr[i] * scale * (w != nullptr ? w[i] : 1.0F);
  }
}

__global__ void AddBiasRowsKernel(float* __restrict__ f,
                                  const float* __restrict__ bias,
                                  std::uint32_t n, std::size_t total) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < total) {
    f[i] += bias[i - (i / n) * n];
  }
}

__global__ void ExpertCountsKernel(const std::int32_t* __restrict__ ids,
                                   std::uint32_t* __restrict__ counts,
                                   std::size_t total) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < total && ids[i] >= 0) {
    atomicAdd(&counts[ids[i]], 1U);
  }
}

__global__ void FillModuloIdsKernel(std::int32_t* __restrict__ ids,
                                    std::size_t total, std::uint32_t pattern) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < total) {
    ids[i] = static_cast<std::int32_t>(i % pattern);
  }
}

__global__ void RowWeightedSumTokensKernel(const float* __restrict__ src,
                                           const float* __restrict__ weights,
                                           float* __restrict__ dst,
                                           std::uint32_t used,
                                           std::uint32_t cols,
                                           std::size_t total) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= total) {
    return;
  }
  const std::size_t t = i / cols;
  const std::uint32_t c = static_cast<std::uint32_t>(i - t * cols);
  const float* srow = src + t * static_cast<std::size_t>(used) * cols;
  const float* wrow = weights + t * used;
  float acc = 0.0F;
  for (std::uint32_t r = 0; r < used; ++r) {
    acc += wrow[r] * srow[static_cast<std::size_t>(r) * cols + c];
  }
  dst[i] = acc;
}

__global__ void MlaQRopeRowsKernel(
    float* __restrict__ q, float* __restrict__ qrp, std::uint32_t total_pairs,
    std::uint32_t heads, std::uint32_t head_stride, std::uint32_t nope,
    std::uint32_t rotary_dim, std::uint32_t start_pos, float theta) {
  const std::uint32_t rp = rotary_dim / 2U;
  const std::uint32_t per_token = heads * rp;
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_pairs) {
    return;
  }
  const std::uint32_t t = i / per_token;
  const std::uint32_t rem = i - t * per_token;
  const std::uint32_t h = rem / rp;
  const std::uint32_t p = rem - h * rp;
  float* v = q + static_cast<std::size_t>(t) * heads * head_stride +
             static_cast<std::size_t>(h) * head_stride + nope;
  const float freq = static_cast<float>(
      ::pow(static_cast<double>(theta),
            -2.0 * static_cast<double>(p) / static_cast<double>(rotary_dim)));
  const float angle = static_cast<float>(start_pos + t) * freq;
  double c = 0.0;
  double s = 0.0;
  ::sincos(static_cast<double>(angle), &s, &c);
  const float a = v[2 * p];
  const float b = v[2 * p + 1];
  const float lo = a * c - b * s;
  const float hi = a * s + b * c;
  v[2 * p] = lo;
  v[2 * p + 1] = hi;
  float* qr = qrp + (static_cast<std::size_t>(t) * heads + h) * rotary_dim;
  qr[2 * p] = lo;
  qr[2 * p + 1] = hi;
}

__global__ void MlaKvANormRopeRowsKernel(
    const float* __restrict__ a, const float* __restrict__ gamma, float eps,
    float* __restrict__ kv_cache, float* __restrict__ kpe_cache,
    std::uint32_t kv_rank, std::uint32_t rotary_dim, std::uint32_t start_pos,
    float theta) {
  __shared__ double smem[8];
  const std::uint32_t t = blockIdx.x;
  const float* ar = a + static_cast<std::size_t>(t) * (kv_rank + rotary_dim);
  double ss = 0.0;
  for (std::uint32_t i = threadIdx.x; i < kv_rank; i += blockDim.x) {
    const double v = static_cast<double>(ar[i]);
    ss += v * v;
  }
  const float scale =
      1.0F / ::sqrtf(static_cast<float>(BlockSumDouble(ss, smem) /
                                        static_cast<double>(kv_rank)) +
                     eps);
  float* kv = kv_cache + (static_cast<std::size_t>(start_pos) + t) * kv_rank;
  for (std::uint32_t i = threadIdx.x; i < kv_rank; i += blockDim.x) {
    kv[i] = ar[i] * scale * gamma[i];
  }
  const std::uint32_t rp = rotary_dim / 2U;
  if (threadIdx.x < rp) {
    const std::uint32_t p = threadIdx.x;
    const float* v = ar + kv_rank;
    const float freq = static_cast<float>(
        ::pow(static_cast<double>(theta),
              -2.0 * static_cast<double>(p) / static_cast<double>(rotary_dim)));
    const float angle = static_cast<float>(start_pos + t) * freq;
    double c = 0.0;
    double s = 0.0;
    ::sincos(static_cast<double>(angle), &s, &c);
    const float a0 = v[2 * p];
    const float b0 = v[2 * p + 1];
    float* kpe =
        kpe_cache + (static_cast<std::size_t>(start_pos) + t) * rotary_dim;
    kpe[2 * p] = a0 * c - b0 * s;
    kpe[2 * p + 1] = a0 * s + b0 * c;
  }
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

void KdaConvSiluPrefill(const float* conv_w, const float* proj,
                        const float* history, float* out, std::uint32_t tokens,
                        std::uint32_t inner, std::uint32_t kern,
                        hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(tokens) * inner;
  const std::uint32_t grid = static_cast<std::uint32_t>(
      std::min<std::size_t>((total + 255U) / 256U, 65535U));
  KdaConvSiluPrefillKernel<<<grid, 256, 0, stream>>>(proj, conv_w, history, out,
                                                     tokens, inner, kern);
}

void KdaHistoryUpdate(const float* proj, const float* history, float* out,
                      std::uint32_t tokens, std::uint32_t inner,
                      std::uint32_t kern, hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(kern - 1) * inner;
  const std::uint32_t grid = static_cast<std::uint32_t>(
      std::min<std::size_t>((total + 255U) / 256U, 65535U));
  KdaHistoryUpdateKernel<<<grid, 256, 0, stream>>>(proj, history, out, tokens,
                                                   inner, kern);
}

void KdaNormHeadsPrefill(float* q, float* k, std::uint32_t tokens,
                         std::uint32_t heads, std::uint32_t d, float eps,
                         float q_scale, hipStream_t stream) {
  // Same kernel as decode with one block per (token, head): the body already
  // derives the row from blockIdx.x, so the batched run is bit-identical.
  KdaNormHeadsKernel<<<tokens * heads, d, 0, stream>>>(q, k, d, eps, q_scale);
}

void KdaPrep(const float* f, const float* a, const float* beta, const float* qn,
             const float* kn, float gate_lb, float* decay_pre, float* beta_pre,
             float* kq_pre, std::uint32_t tokens, std::uint32_t heads,
             std::uint32_t d, hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(tokens) * heads * d;
  const std::uint32_t decay_grid = static_cast<std::uint32_t>(
      std::min<std::size_t>((total + 255U) / 256U, 65535U));
  KdaPrepDecayKernel<<<decay_grid, 256, 0, stream>>>(f, a, decay_pre, heads, d,
                                                     gate_lb, total);
  const std::size_t pairs = static_cast<std::size_t>(tokens) * heads;
  const std::uint32_t beta_grid = static_cast<std::uint32_t>(
      std::min<std::size_t>((pairs + 255U) / 256U, 65535U));
  KdaPrepBetaKernel<<<beta_grid, 256, 0, stream>>>(beta, beta_pre, pairs);
  KdaPrepKqKernel<<<static_cast<std::uint32_t>((pairs + 7U) / 8U), 32, 0,
                    stream>>>(qn, kn, kq_pre, static_cast<std::uint32_t>(pairs),
                              heads, d);
}

void KdaDeltaLoop(const float* qn, const float* kn, const float* v,
                  const float* decay_pre, const float* beta_pre,
                  const float* kq_pre, float* state, float* attn,
                  std::uint32_t tokens, std::uint32_t heads,
                  std::uint32_t head_dim, float* snap, std::uint32_t snap_rows,
                  hipStream_t stream) {
  // gfx1151 runs wave32; the recurrences hard-code aligned 32-lane groups.
  constexpr std::uint32_t kWarp = 32U;
  if (snap == nullptr && head_dim == 128U) {
    constexpr std::uint32_t kRowsPerWave = 8U;  // kRowsPerLane == 1
    const dim3 grid(head_dim / (kRowsPerWave * 8U), heads);
    KdaDeltaLoopRowSplitKernel<1>
        <<<grid, dim3(256), 0, stream>>>(qn, kn, v, decay_pre, beta_pre, kq_pre,
                                         state, attn, tokens, heads, head_dim);
    return;
  }
  constexpr std::uint32_t kRows = 8;
  const dim3 grid(head_dim / kRows, heads);
  const dim3 block(kWarp);
  switch (head_dim / kWarp) {
    case 8:
      KdaDeltaLoopKernel<kRows, 8><<<grid, block, 0, stream>>>(
          qn, kn, v, decay_pre, beta_pre, kq_pre, state, attn, tokens, heads,
          head_dim, snap, snap_rows);
      break;
    case 5:
      KdaDeltaLoopKernel<kRows, 5><<<grid, block, 0, stream>>>(
          qn, kn, v, decay_pre, beta_pre, kq_pre, state, attn, tokens, heads,
          head_dim, snap, snap_rows);
      break;
    case 4:
      KdaDeltaLoopKernel<kRows, 4><<<grid, block, 0, stream>>>(
          qn, kn, v, decay_pre, beta_pre, kq_pre, state, attn, tokens, heads,
          head_dim, snap, snap_rows);
      break;
    case 2:
      KdaDeltaLoopKernel<kRows, 2><<<grid, block, 0, stream>>>(
          qn, kn, v, decay_pre, beta_pre, kq_pre, state, attn, tokens, heads,
          head_dim, snap, snap_rows);
      break;
    default:
      KdaDeltaLoopKernel<kRows, 1><<<grid, block, 0, stream>>>(
          qn, kn, v, decay_pre, beta_pre, kq_pre, state, attn, tokens, heads,
          head_dim, snap, snap_rows);
      break;
  }
}

void KdaOutNormPrefill(float* attn, const float* g, const float* norm_w,
                       std::uint32_t tokens, std::uint32_t heads,
                       std::uint32_t d, float eps, hipStream_t stream) {
  KdaOutNormPrefillKernel<<<tokens * heads, d, 0, stream>>>(attn, g, norm_w, d,
                                                            eps);
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

void RmsNormRows(const float* x, const float* w, float eps, float* out,
                 std::uint32_t rows, std::uint32_t n, hipStream_t stream) {
  RmsNormRowsKernel<<<rows, 256, 0, stream>>>(x, w, eps, out, n);
}

void AddBiasRows(float* f, const float* bias, std::uint32_t rows,
                 std::uint32_t n, hipStream_t stream) {
  // The 65535-block cap covers a chunk of up to 2048 rows for every stream
  // width used by the executor (inner 4096 -> 32768 blocks).
  const std::size_t total = static_cast<std::size_t>(rows) * n;
  const std::uint32_t grid = static_cast<std::uint32_t>(
      std::min<std::size_t>((total + 255U) / 256U, 65535U));
  AddBiasRowsKernel<<<grid, 256, 0, stream>>>(f, bias, n, total);
}

void RouterTopKRows(const float* logits, const float* bias, std::uint32_t n_exp,
                    std::uint32_t groups, std::uint32_t groups_used,
                    std::uint32_t used, float weights_scale, std::int32_t* ids,
                    float* weights, std::uint32_t tokens, hipStream_t stream) {
  const std::uint32_t smem = (2 * n_exp + groups) * sizeof(float);
  RouterTopKKernel<<<tokens, 512, smem, stream>>>(logits, bias, n_exp, groups,
                                                  groups_used, used,
                                                  weights_scale, ids, weights);
}

void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n, std::uint32_t n_exp, hipStream_t stream) {
  (void)hipMemsetAsync(counts, 0, n_exp * sizeof(std::uint32_t), stream);
  const std::size_t total = n;
  const std::uint32_t grid = static_cast<std::uint32_t>(
      std::min<std::size_t>((total + 255U) / 256U, 65535U));
  ExpertCountsKernel<<<grid, 256, 0, stream>>>(ids, counts, total);
}

void FillModuloIds(std::int32_t* ids, std::uint32_t n, std::uint32_t pattern,
                   hipStream_t stream) {
  const std::uint32_t grid = static_cast<std::uint32_t>(std::min<std::size_t>(
      (static_cast<std::size_t>(n) + 255U) / 256U, 65535U));
  FillModuloIdsKernel<<<grid, 256, 0, stream>>>(ids, n, pattern);
}

void RowWeightedSumTokens(const float* src, const float* weights, float* dst,
                          std::uint32_t tokens, std::uint32_t used,
                          std::uint32_t cols, hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(tokens) * cols;
  const std::uint32_t grid = static_cast<std::uint32_t>(
      std::min<std::size_t>((total + 255U) / 256U, 65535U));
  RowWeightedSumTokensKernel<<<grid, 256, 0, stream>>>(src, weights, dst, used,
                                                       cols, total);
}

void MlaQRopeRows(float* q, float* qrp, std::uint32_t start_pos,
                  std::uint32_t tokens, std::uint32_t heads,
                  std::uint32_t head_stride, std::uint32_t nope,
                  std::uint32_t rotary_dim, float theta, hipStream_t stream) {
  const std::size_t total =
      static_cast<std::size_t>(tokens) * heads * (rotary_dim / 2U);
  const std::uint32_t grid = static_cast<std::uint32_t>(
      std::min<std::size_t>((total + 255U) / 256U, 65535U));
  MlaQRopeRowsKernel<<<grid, 256, 0, stream>>>(
      q, qrp, static_cast<std::uint32_t>(total), heads, head_stride, nope,
      rotary_dim, start_pos, theta);
}

void MlaKvANormRopeRows(const float* a, const float* gamma, float eps,
                        float* kv_cache, float* kpe_cache,
                        std::uint32_t start_pos, std::uint32_t tokens,
                        std::uint32_t kv_rank, std::uint32_t rotary_dim,
                        float theta, hipStream_t stream) {
  MlaKvANormRopeRowsKernel<<<tokens, 256, 0, stream>>>(
      a, gamma, eps, kv_cache, kpe_cache, kv_rank, rotary_dim, start_pos,
      theta);
}

}  // namespace gufo::models::ling3_flash::rocm