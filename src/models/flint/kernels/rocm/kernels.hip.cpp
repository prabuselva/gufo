#include "src/models/flint/kernels/rocm/kernels.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

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

// opt-c7v-norm-quant: RMSNorm (optionally after an in-place residual add) and
// the tiled Q8_1 activation copy in one launch.
//
// The output loop strides kNormThreads, so with cols % 256 == 0 each iteration
// leaves warp w holding exactly the 32 elements of quantization block
// `iteration * (kNormThreads / 32) + w`: the shipped per-wave quantizer's
// shape matches the norm loop with no staging. The tree, d/id/roundf
// expressions and StoreQ8ActLane are copied from the shipped quantizer and run
// on the register value the norm stored, so the q bytes, scales and
// activation-sum sidecars are bit-identical to quantizing a second read of the
// normed row (probe tools/bench/flint_norm_probe.hip: zero mismatches on
// 9 batches, +5 % to +21 % for rows >= 3072). Measured callers gate the batch;
// shorter rows keep the chain.
template<bool AddResidual>
__global__ void RmsNormQuantizeQ8_1Kernel(
    float* __restrict__ x, const float* __restrict__ y,
    const float* __restrict__ gamma, float eps, float* __restrict__ out,
    void* __restrict__ q8, std::size_t batch, std::uint32_t cols) {
  __shared__ float sm[kNormThreads / 32];
  float* __restrict__ xr = x + static_cast<std::size_t>(blockIdx.x) * cols;
  const float* __restrict__ yr =
      AddResidual ? y + static_cast<std::size_t>(blockIdx.x) * cols : nullptr;
  float* __restrict__ orow = out + static_cast<std::size_t>(blockIdx.x) * cols;
  float ss = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < cols; i += kNormThreads) {
    float v = xr[i];
    if constexpr (AddResidual) {
      v += yr[i];
      xr[i] = v;
    }
    ss += v * v;
  }
  ss = BlockReduceSum(ss, sm);
  const float scale = 1.0F / sqrtf(ss / static_cast<float>(cols) + eps);
  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp = threadIdx.x >> 5u;
  const std::size_t num_blocks = cols / 32u;
  const std::size_t tok = blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < cols; i += kNormThreads) {
    const float val = xr[i] * scale * gamma[i];
    orow[i] = val;
    const std::size_t blk =
        (static_cast<std::size_t>(i) / kNormThreads) * (kNormThreads / 32u) +
        warp;
    if (blk >= num_blocks) {
      continue;
    }
    float max_abs = fabsf(val);
    for (int off = 16; off > 0; off >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
    }
    const float d = max_abs / 127.0F;
    const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
    const std::int8_t q = static_cast<std::int8_t>(roundf(val * id));
    gufo::hip::StoreQ8ActLane<true>(q8, batch, num_blocks, tok, blk, lane, d,
                                    q);
  }
}

__global__ void AddResidualKernel(float* __restrict__ x,
                                  const float* __restrict__ y,
                                  std::uint32_t cols) {
  float* __restrict__ xr = x + static_cast<std::size_t>(blockIdx.x) * cols;
  const float* __restrict__ yr =
      y + static_cast<std::size_t>(blockIdx.x) * cols;
  // Vector path: cols % 4 == 0 keeps every row 16-byte aligned and every
  // output element's arithmetic identical to the scalar loop.
  if ((cols & 3U) == 0U) {
    const std::uint32_t cols4 = cols / 4U;
    float4* __restrict__ xv =
        reinterpret_cast<float4*>(xr);  // IWYU pragma: ignore
    const float4* __restrict__ yv =
        reinterpret_cast<const float4*>(yr);  // IWYU pragma: ignore
    for (std::uint32_t i = threadIdx.x; i < cols4; i += kNormThreads) {
      float4 a = xv[i];
      const float4 b = yv[i];
      a.x += b.x;
      a.y += b.y;
      a.z += b.z;
      a.w += b.w;
      xv[i] = a;
    }
    return;
  }
  for (std::uint32_t i = threadIdx.x; i < cols; i += kNormThreads) {
    xr[i] += yr[i];
  }
}

__global__ void SiluMulKernel(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ out, std::size_t count) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t n4 = count / 4U;  // pointers are 16-byte aligned bases
  const float4* __restrict__ a4 =
      reinterpret_cast<const float4*>(a);  // IWYU pragma: ignore
  const float4* __restrict__ b4 =
      reinterpret_cast<const float4*>(b);  // IWYU pragma: ignore
  float4* __restrict__ out4 =
      reinterpret_cast<float4*>(out);  // IWYU pragma: ignore
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4; i += stride) {
    float4 o;
    const float4 av = a4[i];
    const float4 bv = b4[i];
    const float xs[4] = {av.x, av.y, av.z, av.w};
    const float ys[4] = {bv.x, bv.y, bv.z, bv.w};
    float os[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      os[c] = xs[c] / (1.0F + expf(-xs[c])) * ys[c];
    }
    o.x = os[0];
    o.y = os[1];
    o.z = os[2];
    o.w = os[3];
    out4[i] = o;
  }
  const std::size_t tail = n4 * 4U;
  if (blockIdx.x == 0 && threadIdx.x < count - tail) {
    const std::size_t i = tail + threadIdx.x;
    const float av = a[i];
    out[i] = av / (1.0F + expf(-av)) * b[i];
  }
}

// opt-c10-qk: the epilogue walks kTokensPerBlock tokens per block instead of
// one, hoisting block-uniform setup (gamma load, rope frequency, row base
// pointers) out of the token loop and replacing the paired cosf/sinf with
// sincosf. Every per-token FP operation, the 128-thread reduce tree and the
// rounding stay identical: tools/bench/flint_qk_bench.hip audits sincosf
// against cosf/sinf over the full (pos, d, theta) rope domain (0 mismatches
// for theta 1e4..5e6, pos < 131072) and verifies bit-identical whole-buffer
// output against the previous kernel at the pp2048/pp4096 and decode shapes,
// while the 8-token walk cuts blocks per launch 8x and reaches 230 GB/s of
// the modelled traffic (155 GB/s before; ~239 GB/s measured DRAM ceiling).
__global__ void QkEpilogueKernel(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
    float eps, float theta, int rope, std::uint32_t pos0, std::uint32_t n_head,
    std::uint32_t n_kv, std::uint32_t head_dim, __half* __restrict__ k_cache,
    __half* __restrict__ v_cache, std::uint32_t kv_row, std::uint32_t batch) {
  constexpr std::uint32_t kTokensPerBlock = 8U;
  const std::uint32_t h = blockIdx.x;
  const bool is_q = h < n_head;
  const std::uint32_t d = threadIdx.x;
  __shared__ float sm_head[128];
  __shared__ float sm_part[8];

  float* __restrict__ xbase =
      is_q ? q + static_cast<std::size_t>(h) * head_dim
           : k + static_cast<std::size_t>(h - n_head) * head_dim;
  const std::size_t x_row = is_q ? static_cast<std::size_t>(n_head) * head_dim
                                 : static_cast<std::size_t>(n_kv) * head_dim;
  const float* __restrict__ gamma = is_q ? q_gamma : k_gamma;
  const float gam = d < head_dim ? gamma[d] : 0.0F;
  float freq = 0.0F;
  if (rope != 0 && d < head_dim / 2) {
    freq = powf(theta,
                -2.0F * static_cast<float>(d) / static_cast<float>(head_dim));
  }

  const std::uint32_t t0 = blockIdx.y * kTokensPerBlock;
#pragma unroll
  for (std::uint32_t i = 0; i < kTokensPerBlock; ++i) {
    const std::uint32_t t = t0 + i;
    if (t >= batch) {
      break;  // block-uniform: never skips a barrier
    }
    float* __restrict__ x = xbase + static_cast<std::size_t>(t) * x_row;
    const float xv = d < head_dim ? x[d] : 0.0F;
    float ss = xv * xv;
    ss = BlockReduceSum(ss, sm_part);
    const float scale = 1.0F / sqrtf(ss / static_cast<float>(head_dim) + eps);
    sm_head[d] = xv * scale * gam;
    __syncthreads();
    const std::uint32_t pos = pos0 + t;
    if (rope != 0 && d < head_dim / 2) {
      const float angle = static_cast<float>(pos) * freq;
      float c;
      float s;
      sincosf(angle, &s, &c);
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
      continue;
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
    // sm_head carries over between rows exactly like the previous kernel
    // across blocks: every lane rewrites its slot before the next use.
  }
}

// Tiled attention: TR query rows per block, KV staged into LDS with K
// transposed (skT[d][j + 1]; the +1 pad avoids the bank conflict of
// k-row-major reads in the score loop). Each LDS K word feeds two FFMAs
// through the 2x2 register tile; score chains are d-ascending per output.
// Ported from the nemotron_h_moe LDS-tile kernel (measured 3.2x at pp16384
// shapes with TR=24, kTile=16 on gfx1151), adapted to Flint's separate
// fp16 K/V cache rows, sliding-window masking and attention sinks.
constexpr int kHeadDim = 128;  // Flint rotary/noPE head dimension.

template<int TR, int kTile>
__global__ void AttentionKernel(const float* __restrict__ q,
                                const __half* __restrict__ k_cache,
                                const __half* __restrict__ v_cache,
                                const float* __restrict__ sinks, float scale,
                                std::uint32_t pos0, std::uint32_t batch,
                                std::uint32_t window, std::uint32_t n_head,
                                std::uint32_t n_kv, std::uint32_t head_dim,
                                std::uint32_t kv_row, float* __restrict__ out) {
  constexpr float kNegInf = -3.4e38F;
  const std::uint32_t h = blockIdx.y;
  const std::uint32_t row0 = blockIdx.x * TR;
  const std::uint32_t tokens = batch;
  const std::uint32_t rows =
      tokens - row0 < TR ? tokens - row0 : static_cast<std::uint32_t>(TR);
  const std::uint32_t kvh = h * n_kv / n_head;
  const int tid = static_cast<int>(threadIdx.x);

  __shared__ float sq[TR][kHeadDim];
  __shared__ float skT[kHeadDim][kTile + 1];
  __shared__ float sv[kTile][kHeadDim];
  __shared__ float se[TR][kTile];
  __shared__ float sm[TR];
  __shared__ float sl[TR];
  __shared__ float sa[TR];

  for (int i = tid; i < TR * kHeadDim; i += kHeadDim) {
    const int r = i / kHeadDim;
    const int d = i % kHeadDim;
    sq[r][d] =
        r < static_cast<int>(rows)
            ? q[(static_cast<std::size_t>(row0 + r) * n_head + h) * head_dim +
                d]
            : 0.0F;
  }
  if (tid < TR) {
    sm[tid] = kNegInf;
    sl[tid] = 0.0F;
  }
  __syncthreads();

  const __half* __restrict__ kv_k = k_cache + kvh * head_dim;
  const __half* __restrict__ kv_v = v_cache + kvh * head_dim;
  const std::uint32_t nmax = pos0 + row0 + rows;  // exclusive causal bound
  const uint32_t pmin =
      window != 0 && pos0 + row0 + 1 > window ? pos0 + row0 + 1 - window : 0u;
  const std::uint32_t p0_start = pmin / kTile * kTile;

  float o[TR] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  for (std::uint32_t p0 = p0_start; p0 < nmax; p0 += kTile) {
    const int nj = static_cast<int>(
        nmax - p0 < static_cast<std::uint32_t>(kTile) ? nmax - p0 : kTile);
    for (int i = tid; i < nj * kHeadDim; i += kHeadDim) {
      const int j = i / kHeadDim;
      const int d = i % kHeadDim;
      skT[d][j + 1] =
          __half2float(kv_k[(static_cast<std::size_t>(p0 + j)) * kv_row + d]);
    }
    for (int i = tid; i < nj * kHeadDim; i += kHeadDim) {
      const int j = i / kHeadDim;
      const int d = i % kHeadDim;
      sv[j][d] =
          __half2float(kv_v[(static_cast<std::size_t>(p0 + j)) * kv_row + d]);
    }
    __syncthreads();

    for (int i = tid; i < (TR / 2) * (kTile / 2); i += kHeadDim) {
      const int r0 = (i / (kTile / 2)) * 2;
      const int j0 = (i % (kTile / 2)) * 2;
      float a00 = 0.0F;
      float a01 = 0.0F;
      float a10 = 0.0F;
      float a11 = 0.0F;
      for (int d = 0; d < kHeadDim; ++d) {
        const float k0 = skT[d][j0 + 1];
        const float k1 = skT[d][j0 + 2];
        const float q0 = sq[r0][d];
        a00 += q0 * k0;
        a01 += q0 * k1;
        const float q1 = sq[r0 + 1][d];
        a10 += q1 * k0;
        a11 += q1 * k1;
      }
      const std::uint32_t rp = pos0 + row0;
      const std::uint32_t p00 = p0 + j0;
      const std::uint32_t p01 = p0 + j0 + 1;
      const bool ok00 = r0 < rows && p00 <= rp + r0 && p00 >= pmin;
      const bool ok01 = r0 < rows && p01 <= rp + r0 && p01 >= pmin;
      const bool ok10 = r0 + 1 < rows && p00 <= rp + r0 + 1 && p00 >= pmin;
      const bool ok11 = r0 + 1 < rows && p01 <= rp + r0 + 1 && p01 >= pmin;
      se[r0][j0] = ok00 ? a00 * scale : kNegInf;
      se[r0][j0 + 1] = ok01 ? a01 * scale : kNegInf;
      se[r0 + 1][j0] = ok10 ? a10 * scale : kNegInf;
      se[r0 + 1][j0 + 1] = ok11 ? a11 * scale : kNegInf;
    }
    __syncthreads();

    if (tid < TR) {
      float mnew = sm[tid];
      for (int j = 0; j < kTile; ++j) {
        mnew = se[tid][j] > mnew ? se[tid][j] : mnew;
      }
      sa[tid] = expf(sm[tid] - mnew);
      sm[tid] = mnew;
    }
    __syncthreads();

    for (int i = tid; i < TR * kTile; i += kHeadDim) {
      const int r = i / kTile;
      const int j = i % kTile;
      se[r][j] = expf(se[r][j] - sm[r]);
    }
    __syncthreads();

    if (tid < TR) {
      float ssum = 0.0F;
      for (int j = 0; j < kTile; ++j) {
        ssum += se[tid][j];
      }
      sl[tid] = sl[tid] * sa[tid] + ssum;
    }
    for (int r = 0; r < TR; ++r) {
      o[r] *= sa[r];
    }
    for (int j = 0; j < nj; ++j) {
      for (int r = 0; r < TR; ++r) {
        o[r] += se[r][j] * sv[j][tid];
      }
    }
    __syncthreads();
  }

  if (tid < TR) {
    sl[tid] += expf(sinks[h] - sm[tid]);
  }
  __syncthreads();
  for (std::uint32_t r = 0; r < rows; ++r) {
    out[(static_cast<std::size_t>(row0 + r) * n_head + h) * head_dim + tid] =
        o[r] / sl[r];
  }
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

__device__ inline float4 LoadRouted4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}

__device__ inline float4 LoadRouted4(const __half* p) {
  union {
    uint2 u;
    __half2 h[2];
  } v;
  v.u = *reinterpret_cast<const uint2*>(p);
  return make_float4(__half2float(v.h[0].x), __half2float(v.h[0].y),
                     __half2float(v.h[1].x), __half2float(v.h[1].y));
}

template<typename RoutedT>
__global__ void SumExpertsKernel(float* __restrict__ x,
                                 const RoutedT* __restrict__ routed,
                                 const float* __restrict__ weights,
                                 const float* __restrict__ shared,
                                 std::uint32_t cols, std::uint32_t used) {
  const std::uint32_t t = blockIdx.y;
  float* __restrict__ xr = x + static_cast<std::size_t>(t) * cols;
  const RoutedT* __restrict__ rbase =
      routed + static_cast<std::size_t>(t) * used * cols;
  const float* __restrict__ wrow = weights + static_cast<std::size_t>(t) * used;
  if ((cols & 3U) == 0U) {
    // Vector path; each output channel keeps the same per-s add order.
    const std::uint32_t cols4 = cols / 4U;
    float4* __restrict__ xv = reinterpret_cast<float4*>(xr);
    const RoutedT* __restrict__ rv = rbase;
    const float4* __restrict__ sv =
        shared != nullptr ? reinterpret_cast<const float4*>(shared) : nullptr;
    const std::size_t row4 = static_cast<std::size_t>(t) * cols4;
    for (std::uint32_t i = threadIdx.x; i < cols4; i += kNormThreads) {
      float4 acc = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      for (std::uint32_t s = 0; s < used; ++s) {
        const float w = wrow[s];
        const float4 r =
            LoadRouted4(rv + (static_cast<std::size_t>(s) * cols4 + i) * 4U);
        acc.x += w * r.x;
        acc.y += w * r.y;
        acc.z += w * r.z;
        acc.w += w * r.w;
      }
      if (sv != nullptr) {
        const float4 sh = sv[row4 + i];
        acc.x += sh.x;
        acc.y += sh.y;
        acc.z += sh.z;
        acc.w += sh.w;
      }
      float4 a = xv[i];
      a.x += acc.x;
      a.y += acc.y;
      a.z += acc.z;
      a.w += acc.w;
      xv[i] = a;
    }
    return;
  }
  for (std::uint32_t j = threadIdx.x; j < cols; j += kNormThreads) {
    float acc = 0.0F;
    for (std::uint32_t s = 0; s < used; ++s) {
      acc += wrow[s] *
             static_cast<float>(rbase[static_cast<std::size_t>(s) * cols + j]);
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

void RmsNormQuantizeQ8_1(float* x, const float* residual, const float* gamma,
                         float eps, float* out, void* q8_1_out,
                         std::uint32_t rows, std::uint32_t cols,
                         hipStream_t stream) {
  if (residual != nullptr) {
    RmsNormQuantizeQ8_1Kernel<true><<<rows, kNormThreads, 0, stream>>>(
        x, residual, gamma, eps, out, q8_1_out, rows, cols);
  } else {
    RmsNormQuantizeQ8_1Kernel<false><<<rows, kNormThreads, 0, stream>>>(
        x, nullptr, gamma, eps, out, q8_1_out, rows, cols);
  }
  gufo::hip::LaunchZeroQ8ActTailFor(q8_1_out, rows, cols, stream);
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
  constexpr std::uint32_t kTokensPerBlock = 8U;
  const dim3 grid(n_head + n_kv,
                  (batch + kTokensPerBlock - 1U) / kTokensPerBlock);
  QkEpilogueKernel<<<grid, 128, 0, stream>>>(
      q, k, v, q_gamma, k_gamma, eps, theta, rope ? 1 : 0, pos0, n_head, n_kv,
      head_dim, k_cache, v_cache, kv_row_elements, batch);
}

void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const float* sinks, float scale, std::uint32_t pos0,
               std::uint32_t batch, std::uint32_t window, std::uint32_t n_head,
               std::uint32_t n_kv, std::uint32_t head_dim,
               std::uint32_t kv_row_elements, float* out, hipStream_t stream) {
  if (head_dim == kHeadDim &&
      LaunchWmmaAttention(q, k_cache, v_cache, sinks, scale, pos0, batch,
                          window, n_head, n_kv, kv_row_elements, out, stream)) {
    return;
  }
  const dim3 grid((batch + 23) / 24, n_head);
  AttentionKernel<24, 16><<<grid, kHeadDim, 0, stream>>>(
      q, k_cache, v_cache, sinks, scale, pos0, batch, window, n_head, n_kv,
      head_dim, kv_row_elements, out);
}

void RouterTopK(const float* probs_logits, std::int32_t* ids, float* weights,
                std::uint32_t batch, std::uint32_t used, bool normalize,
                float weight_scale, hipStream_t stream) {
  RouterTopKKernel<<<batch, 256, 0, stream>>>(probs_logits, ids, weights, used,
                                              normalize ? 1 : 0, weight_scale);
}

namespace {
__global__ void ExpertCountsKernel(const std::int32_t* __restrict__ ids,
                                   std::uint32_t n_ids,
                                   unsigned int* __restrict__ counts) {
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_ids;
       i += gridDim.x * blockDim.x) {
    atomicAdd(&counts[ids[i]], 1U);
  }
}
}  // namespace

void ExpertCounts(const std::int32_t* ids, std::uint32_t n_ids,
                  std::uint32_t n_experts, unsigned int* counts,
                  hipStream_t stream) {
  (void)hipMemsetAsync(counts, 0, static_cast<std::size_t>(n_experts) * 4,
                       stream);
  const std::uint32_t threads = 256;
  const std::uint32_t blocks = (n_ids + threads - 1) / threads;
  ExpertCountsKernel<<<blocks, threads, 0, stream>>>(ids, n_ids, counts);
}

void SumExperts(float* x, const float* routed, const float* weights,
                const float* shared, std::uint32_t rows, std::uint32_t cols,
                std::uint32_t used, hipStream_t stream) {
  const dim3 grid(1, rows);
  SumExpertsKernel<float><<<grid, kNormThreads, 0, stream>>>(
      x, routed, weights, shared, cols, used);
}

void SumExperts(float* x, const __half* routed, const float* weights,
                const float* shared, std::uint32_t rows, std::uint32_t cols,
                std::uint32_t used, hipStream_t stream) {
  const dim3 grid(1, rows);
  SumExpertsKernel<__half><<<grid, kNormThreads, 0, stream>>>(
      x, routed, weights, shared, cols, used);
}

}  // namespace gufo::models::flint::rocm
