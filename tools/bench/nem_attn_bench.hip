// Standalone microbenchmark for the Nemotron-3.5-Bolt-74B prefill attention.
//
// Self-contained (no repository headers) so it compiles directly with hipcc on
// the supported gfx1151 host for fast kernel-design iteration.
//
//   tools/bench/build.sh nem_attn_bench
//   /tmp/nem_attn_bench -c                     // correctness vs FP64 CPU ref
//   /tmp/nem_attn_bench -b 8192 -s 24576 -i 3  // pp32768 last-chunk shape
//
// Shape is the production attention layer: 32 query heads over 2 KV heads
// (GQA 16), head_dim 128, an FP32 KV cache with one token-interleaved row
// [position][2*kvdim] (K of head g at g*128, V at kvdim + g*128), and FP32
// queries/output laid out [token][head][dim].
//
// Variants:
//   base   AttnKernel<24, 16> copied verbatim from nb_attention.hip.cpp.
//   plain  WMMA flash attention with a single F16 product per dot; shows the
//          matrix-core ceiling but its FP16 rounding cannot meet the 2e-5
//          model contract (reference point only).
//   split  WMMA flash attention with the FP32 operands carried as hi+lo FP16
//          pairs (Ootomo fp16x3: ah*bh + ah*bl + al*bh), which keeps the
//          per-product error near 2^-22 and inside the contract.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    hipError_t err_ = (expr);                                           \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

constexpr std::uint32_t kQueryHeads = 32;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kGqa = kQueryHeads / kKvHeads;
constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kAttnWidth = kQueryHeads * kHeadDim;
constexpr std::uint32_t kKvdim = kKvHeads * kHeadDim;
constexpr std::uint32_t kKvRow = 2 * kKvdim;  // floats per KV cache row
constexpr float kScale = 0.08838834764831845F;  // 1 / sqrt(128)

// ---------------------------------------------------------------------------
// Baseline: the production tiled kernel (src/models/nemotron_h_moe/kernels/
// rocm/nb_attention.hip.cpp, AttnKernel<24, 16>) copied verbatim.
// ---------------------------------------------------------------------------

constexpr int kDim = 128;
constexpr float kNegInf = -3.4e38F;

template<int TR, int kTile>
__global__ void AttnKernel(const float* __restrict__ q,
                           const float* __restrict__ kv, int pos0, int tokens,
                           float* __restrict__ out, int heads, int hd, int hdv,
                           int kvh, int kvdim, float scale) {
  const int h = static_cast<int>(blockIdx.y);
  const int row0 = static_cast<int>(blockIdx.x) * TR;
  const int rows = tokens - row0 < TR ? tokens - row0 : TR;
  const int g = h * kvh / heads;
  const int tid = static_cast<int>(threadIdx.x);

  __shared__ float sq[TR][kDim];
  __shared__ float skT[kDim][kTile + 1];
  __shared__ float sv[kTile][kDim];
  __shared__ float se[TR][kTile];
  __shared__ float sm[TR];
  __shared__ float sl[TR];
  __shared__ float sa[TR];

  for (int i = tid; i < TR * kDim; i += kDim) {
    const int r = i / kDim;
    const int d = i % kDim;
    sq[r][d] = r < rows && d < hd
                   ? q[static_cast<int64_t>(row0 + r) * heads * hd + h * hd + d]
                   : 0.0F;
  }
  if (tid < TR) {
    sm[tid] = kNegInf;
    sl[tid] = 0.0F;
  }
  __syncthreads();

  float o[TR] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  const int64_t kvrow = 2 * kvdim;
  const float* kv_k = kv + g * hd;
  const float* kv_v = kv + kvdim + g * hdv;
  const int nmax = pos0 + row0 + rows;  // exclusive causal bound for this tile

  for (int p0 = 0; p0 < nmax; p0 += kTile) {
    const int nj = nmax - p0 < kTile ? nmax - p0 : kTile;
    for (int i = tid; i < nj * kDim; i += kDim) {
      const int j = i / kDim;
      const int d = i % kDim;
      skT[d][j + 1] =
          d < hd ? kv_k[(static_cast<int64_t>(p0 + j)) * kvrow + d] : 0.0F;
    }
    for (int i = tid; i < nj * kDim; i += kDim) {
      const int j = i / kDim;
      const int d = i % kDim;
      sv[j][d] =
          d < hdv ? kv_v[(static_cast<int64_t>(p0 + j)) * kvrow + d] : 0.0F;
    }
    __syncthreads();

    for (int i = tid; i < (TR / 2) * (kTile / 2); i += kDim) {
      const int r0 = (i / (kTile / 2)) * 2;
      const int j0 = (i % (kTile / 2)) * 2;
      float a00 = 0.0F;
      float a01 = 0.0F;
      float a10 = 0.0F;
      float a11 = 0.0F;
      for (int d = 0; d < kDim; ++d) {
        const float k0 = skT[d][j0 + 1];
        const float k1 = skT[d][j0 + 2];
        const float q0 = sq[r0][d];
        a00 += q0 * k0;
        a01 += q0 * k1;
        const float q1 = sq[r0 + 1][d];
        a10 += q1 * k0;
        a11 += q1 * k1;
      }
      const int p00 = p0 + j0;
      const int p01 = p0 + j0 + 1;
      se[r0][j0] = r0 < rows && p00 <= pos0 + row0 + r0 && p00 < nmax
                       ? a00 * scale
                       : kNegInf;
      se[r0][j0 + 1] = r0 < rows && p01 <= pos0 + row0 + r0 && p01 < nmax
                           ? a01 * scale
                           : kNegInf;
      se[r0 + 1][j0] =
          r0 + 1 < rows && p00 <= pos0 + row0 + r0 + 1 && p00 < nmax
              ? a10 * scale
              : kNegInf;
      se[r0 + 1][j0 + 1] =
          r0 + 1 < rows && p01 <= pos0 + row0 + r0 + 1 && p01 < nmax
              ? a11 * scale
              : kNegInf;
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

    for (int i = tid; i < TR * kTile; i += kDim) {
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

  if (tid < hdv) {
    for (int r = 0; r < rows; ++r) {
      out[static_cast<int64_t>(row0 + r) * heads * hdv + h * hdv + tid] =
          o[r] / sl[r];
    }
  }
}

// ---------------------------------------------------------------------------
// Candidate: masked flash attention on the WMMA matrix cores.
//
// Ported from tools/bench/attn_causal_bench_v2.hip (WmmaCausalAttention<32,
// 16, 8>) to the Nemotron geometry: 32x over 2 KV heads, head_dim 128 and the
// FP32 token-interleaved KV rows. Fragment layout (verified against the CPU
// reference there): A holds row L%16 and 16 contiguous k, B holds column
// L%16 and 16 contiguous k (so B is fed a transposed tile), C element i is
// row 2i + L/16, column L%16.
//
// kSplit selects the numerics: false keeps a single F16 product per dot;
// true carries Q, K, P and V as hi+lo FP16 pairs and chains three WMMA
// accumulations per 16-step (ah*bh + ah*bl + al*bh), so the FP32 operands
// stay inside the 2e-5 model contract.
// ---------------------------------------------------------------------------

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

constexpr std::uint32_t kWmmaHeads = 2;  // query heads per block
constexpr std::uint32_t kWmmaKStride = kHeadDim + 8;

template<bool kSplit>
__device__ __forceinline__ void Split8(const float4 lo_in, const float4 hi_in,
                                       __half* hi, __half* lo) {
  const float f[8] = {lo_in.x, lo_in.y, lo_in.z, lo_in.w,
                      hi_in.x, hi_in.y, hi_in.z, hi_in.w};
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    const __half h = __float2half_rn(f[i]);
    hi[i] = h;
    if constexpr (kSplit) {
      lo[i] = __float2half_rn(f[i] - __half2float(h));
    }
  }
}

template<std::uint32_t kQueryRows, std::uint32_t kKeys, std::uint32_t kWaves,
         bool kSplitQK, bool kSplitPV>
__launch_bounds__(kWaves * 32, 1) __global__
    void WmmaCausalAttention(const float* __restrict__ q,
                             const float* __restrict__ kv,
                             float* __restrict__ out, std::uint32_t start_pos,
                             std::uint32_t batch_size) {
  constexpr bool kSplit = kSplitQK || kSplitPV;
  constexpr std::uint32_t kRowBlocks = (kQueryRows / 16) * kWmmaHeads;
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kThreads = kWaves * 32;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kWmmaKSteps = kHeadDim / 16;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / (kWaves / kSTiles);
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kDimTilesPerWave = kWmmaKSteps / kWaves;
  constexpr std::uint32_t kSoftmaxLanes = kThreads / kRows;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  static_assert(kSTiles * 2 == kWaves,
                "the waves cover the S tiles in two head-dimension halves");
  static_assert(kKeys % 16 == 0 && kQueryRows % 16 == 0, "16-row WMMA tiles");
  static_assert(kKStepsPerWave * (kWaves / kSTiles) == kWmmaKSteps, "k split");
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

  const auto row_block_head = [&](std::uint32_t rb) {
    return first_query_head + (rb % kWmmaHeads);
  };
  const auto row_block_offset = [&](std::uint32_t rb) {
    return (rb / kWmmaHeads) * 16;
  };

  // K lives [key][dim] and V transposed [dim][key], each bank in a hi and a lo
  // FP16 copy; S is complete before PV starts, so both phases share the
  // staging space.
  constexpr std::uint32_t kKvLdsHalves =
      (kKeys * kWmmaKStride > kHeadDim * kVtStride) ? (kKeys * kWmmaKStride)
                                                    : (kHeadDim * kVtStride);
  __shared__ __half kv_hi[kKvLdsHalves];
  __shared__ __half kv_lo[kSplit ? kKvLdsHalves : 1];  // K lo if kSplitQK,
                                                       // V^T lo if kSplitPV
  __shared__ float s_lds[2][kSTiles][16][16];
  // +8 half row pad: keeps P fragment rows 16-byte aligned and makes the
  // 16-row LDS128 read bank-rotation cover 16 distinct banks.
  __shared__ __half p_hi[kRows][kKeys + 8];
  __shared__ __half p_lo[kSplitPV ? kRows : 1][kKeys + 8];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h qh[kKStepsPerWave];
  v16h ql[kSplitQK ? kKStepsPerWave : 1];
  {
    const std::uint32_t local_query =
        query_start + row_block_offset(s_rb) + sub;
    const bool live = local_query < batch_size;
    const float* q_row =
        q + (static_cast<std::size_t>(local_query) * kAttnWidth) +
        (static_cast<std::uint32_t>(row_block_head(s_rb)) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 2; ++v) {
        const float4 fa = live ? qp[v * 2] : make_float4(0, 0, 0, 0);
        const float4 fb = live ? qp[v * 2 + 1] : make_float4(0, 0, 0, 0);
        const float f[8] = {fa.x * kScale, fa.y * kScale, fa.z * kScale,
                            fa.w * kScale, fb.x * kScale, fb.y * kScale,
                            fb.z * kScale, fb.w * kScale};
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          const _Float16 h = _Float16(f[i]);
          qh[ks][(v * 8) + i] = h;
          if constexpr (kSplitQK) {
            ql[ks][(v * 8) + i] = _Float16(f[i] - static_cast<float>(h));
          }
        }
      }
    }
  }

  v8f o_acc[kRowBlocks][kDimTilesPerWave] = {};
  for (std::uint32_t r = tid; r < kRows; r += kThreads) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0F;
  }

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);

  constexpr std::uint32_t kVF32 = (kKeys * kHeadDim) / (kThreads * 8);
  for (std::uint32_t key_start = 0; key_start < max_visible;
       key_start += kKeys) {
    // Prefetch V (FP32) before the S phase; parked in registers until the
    // transpose (mirrors attn_causal_bench_v2).
    uint4 v_raw[kVF32 * 2];
    {
      const std::uint32_t v_key = lane % kKeys;
      const std::uint32_t v_slice = (tid / kKeys) * (kVF32 * 8);
      const std::uint32_t key_position = key_start + v_key;
      const float* src = kv + kKvdim + (static_cast<std::size_t>(kv_head) *
                                        kHeadDim) +
                         v_slice +
                         (static_cast<std::size_t>(key_position) * kKvRow);
      const bool live = key_position < context_end;
#pragma unroll
      for (std::uint32_t j = 0; j < kVF32; ++j) {
        v_raw[j * 2] =
            live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                 : make_uint4(0u, 0u, 0u, 0u);
        v_raw[j * 2 + 1] =
            live ? *reinterpret_cast<const uint4*>(src + (j * 8) + 4)
                 : make_uint4(0u, 0u, 0u, 0u);
      }
    }

    // --- stage K [key][dim] with the FP32->F16 (hi[,lo]) split.
    __syncthreads();
    for (std::uint32_t idx = tid; idx < kKeys * (kHeadDim / 8);
         idx += kThreads) {
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      const std::uint32_t key_position = key_start + key_row;
      const bool live = key_position < context_end;
      const float* src = kv + (static_cast<std::size_t>(kv_head) * kHeadDim) +
                         d8 + (static_cast<std::size_t>(key_position) * kKvRow);
      float4 fa = live ? *reinterpret_cast<const float4*>(src)
                       : make_float4(0, 0, 0, 0);
      float4 fb = live ? *reinterpret_cast<const float4*>(src + 4)
                       : make_float4(0, 0, 0, 0);
      __half hi[8];
      __half lo[8] = {};
      Split8<kSplitQK>(fa, fb, hi, lo);
      *reinterpret_cast<uint4*>(&kv_hi[(key_row * kWmmaKStride) + d8]) =
          *reinterpret_cast<const uint4*>(hi);
      if constexpr (kSplitQK) {
        *reinterpret_cast<uint4*>(&kv_lo[(key_row * kWmmaKStride) + d8]) =
            *reinterpret_cast<const uint4*>(lo);
      }
    }
    __syncthreads();

    // --- S = Q K^T
    {
      v8f s_acc = {};
      v8f s_lo_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const uint32_t koff = (((s_kb * 16) + sub) * kWmmaKStride) + d0;
        if constexpr (kSplitQK) {
          const v16h bh = LoadFrag(&kv_hi[koff]);
          const v16h bl = LoadFrag(&kv_lo[koff]);
          s_acc = Wmma(qh[ks], bh, s_acc);
          s_lo_acc = Wmma(qh[ks], bl, s_lo_acc);
          s_lo_acc = Wmma(ql[ks], bh, s_lo_acc);
        } else {
          const v16h bh = LoadFrag(&kv_hi[koff]);
          s_acc = Wmma(qh[ks], bh, s_acc);
        }
      }
      if constexpr (kSplitQK) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) s_acc[i] += s_lo_acc[i];
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
    __syncthreads();

    // --- online softmax: kSoftmaxLanes threads per query row
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
        const bool valid = local_query < batch_size &&
                           key_position <= absolute_query &&
                           key_position < context_end;
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
      const float prev_max = row_max[rg];
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w =
            isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        const __half h = __float2half_rn(w);
        p_hi[rg][(seg * kPerLane) + m] = h;
        if constexpr (kSplitPV) {
          p_lo[rg][(seg * kPerLane) + m] = __float2half_rn(w - __half2float(h));
        }
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_sum += __shfl_xor(part_sum, off);
      }
      if (seg == 0) {
        row_max[rg] = next_max;
        row_sum[rg] = (row_sum[rg] * prior_scale) + part_sum;
        row_scale[rg] = prior_scale;
      }
    }
    __syncthreads();

    // --- rescale the running O by the new maximum.
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
      float scale8[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        scale8[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < kDimTilesPerWave; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale8[i];
        }
      }
    }

    // --- stage V transposed [dim][key] with the FP16 (hi[,lo]) split.
    {
      const std::uint32_t v_key = lane % kKeys;
      const std::uint32_t v_slice = (tid / kKeys) * (kVF32 * 8);
#pragma unroll
      for (std::uint32_t j = 0; j < kVF32; ++j) {
        const float4 fa = *reinterpret_cast<const float4*>(&v_raw[j * 2]);
        const float4 fb = *reinterpret_cast<const float4*>(&v_raw[j * 2 + 1]);
        const float f[8] = {fa.x, fa.y, fa.z, fa.w, fb.x, fb.y, fb.z, fb.w};
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          const std::uint32_t d = v_slice + (j * 8) + i;
          const __half h = __float2half_rn(f[i]);
          kv_hi[(d * kVtStride) + v_key] = h;
          if constexpr (kSplitPV) {
            kv_lo[(d * kVtStride) + v_key] =
                __float2half_rn(f[i] - __half2float(h));
          }
        }
      }
    }
    __syncthreads();

    // --- O += P V
#pragma unroll
    for (std::uint32_t t = 0; t < kDimTilesPerWave; ++t) {
      const std::uint32_t dim_tile = wave + (t * kWaves);
      v16h vh[kKeyBlocks];
      v16h vl[kSplitPV ? kKeyBlocks : 1];
#pragma unroll
      for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
        vh[kb] = LoadFrag(&kv_hi[((dim_tile * 16) + sub) * kVtStride + kb * 16]);
        if constexpr (kSplitPV) {
          vl[kb] = LoadFrag(&kv_lo[((dim_tile * 16) + sub) * kVtStride + kb * 16]);
        }
      }
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
        v8f o_lo = {};
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h ph = LoadFrag(&p_hi[(rb * 16) + sub][kb * 16]);
          if constexpr (kSplitPV) {
            const v16h pl = LoadFrag(&p_lo[(rb * 16) + sub][kb * 16]);
            o_lo = Wmma(ph, vl[kb], o_lo);
            o_lo = Wmma(pl, vh[kb], o_lo);
          }
          o_acc[rb][t] = Wmma(ph, vh[kb], o_acc[rb][t]);
        }
        if constexpr (kSplitPV) {
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) o_acc[rb][t][i] += o_lo[i];
        }
      }
    }
  }
  __syncthreads();

  // --- epilogue
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
    const std::uint32_t query_head = row_block_head(rb);
    const std::uint32_t row_offset = row_block_offset(rb);
#pragma unroll
    for (std::uint32_t t = 0; t < kDimTilesPerWave; ++t) {
      const std::uint32_t dim_tile = wave + (t * kWaves);
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
            (static_cast<std::uint32_t>(query_head) * kHeadDim) +
            (dim_tile * 16) + sub;
        out[offset] =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

template<typename Fn>
float TimeMs(int iters, Fn&& fn) {
  fn();
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());
  hipEvent_t a = nullptr;
  hipEvent_t b = nullptr;
  HIP_OK(hipEventCreate(&a));
  HIP_OK(hipEventCreate(&b));
  HIP_OK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  HIP_OK(hipEventRecord(b));
  HIP_OK(hipEventSynchronize(b));
  float ms = 0.0F;
  HIP_OK(hipEventElapsedTime(&ms, a, b));
  HIP_OK(hipEventDestroy(a));
  HIP_OK(hipEventDestroy(b));
  return ms / static_cast<float>(iters);
}

// CPU FP64 reference, multithreaded over (token chunk, head): the candidate
// gate is the production rocm_attention_test contract (relative <= 2e-5 vs
// FP64 over the whole output tensor).
void ReferenceAttention(const std::vector<float>& q,
                        const std::vector<float>& kv, std::uint32_t start_pos,
                        std::uint32_t batch, std::vector<double>& out) {
  std::vector<std::thread> threads;
  constexpr std::uint32_t kChunks = 8;
  const std::uint32_t chunk = (batch + kChunks - 1) / kChunks;
  for (std::uint32_t c = 0; c < kChunks; ++c) {
    threads.emplace_back([&, c] {
      const std::uint32_t t0 = c * chunk;
      const std::uint32_t t1 = std::min(batch, t0 + chunk);
      for (std::uint32_t t = t0; t < t1; ++t) {
        const std::uint32_t absolute = start_pos + t;
        for (std::uint32_t h = 0; h < kQueryHeads; ++h) {
          const std::uint32_t g = h / kGqa;
          const float* qrow =
              q.data() + std::size_t(t) * kAttnWidth + h * kHeadDim;
          std::vector<double> scores;
          scores.reserve(absolute + 1);
          for (std::uint32_t kp = 0; kp <= absolute; ++kp) {
            double dot = 0.0;
            const float* krow =
                kv.data() + std::size_t(kp) * kKvRow + g * kHeadDim;
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
              dot += static_cast<double>(qrow[d]) *
                     static_cast<double>(krow[d]);
            }
            scores.push_back(dot * static_cast<double>(kScale));
          }
          double m = -INFINITY;
          for (double s : scores) {
            m = std::fmax(m, s);
          }
          std::vector<double> w(scores.size());
          double denom = 0.0;
          for (std::size_t i = 0; i < scores.size(); ++i) {
            w[i] = std::exp(scores[i] - m);
            denom += w[i];
          }
          for (std::uint32_t d = 0; d < kHeadDim; ++d) {
            double acc = 0.0;
            for (std::size_t i = 0; i < w.size(); ++i) {
              const float* vrow = kv.data() + std::size_t(i) * kKvRow +
                                  kKvdim + g * kHeadDim;
              acc += w[i] * static_cast<double>(vrow[d]);
            }
            out[std::size_t(t) * kAttnWidth + h * kHeadDim + d] =
                (denom > 0.0) ? acc / denom : 0.0;
          }
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
}

int main(int argc, char** argv) {
  std::uint32_t batch = 2048;
  std::uint32_t start_pos = 0;
  int iters = 3;
  bool check = false;
  std::string only;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-b" && i + 1 < argc) {
      batch = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "-s" && i + 1 < argc) {
      start_pos =
          static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "-i" && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (arg == "-v" && i + 1 < argc) {
      only = argv[++i];
    } else if (arg == "-c") {
      check = true;
    }
  }

  const std::uint32_t context_end = start_pos + batch;
  std::vector<float> h_q(static_cast<std::size_t>(batch) * kAttnWidth);
  std::vector<float> h_kv(static_cast<std::size_t>(context_end) * kKvRow);
  for (std::size_t i = 0; i < h_q.size(); ++i) {
    h_q[i] = 0.9F * std::sin(0.0131F * static_cast<float>(i % 811)) +
             0.2F * std::cos(0.0027F * static_cast<float>(i % 53));
  }
  for (std::size_t i = 0; i < h_kv.size(); ++i) {
    const float kvv = 0.8F * std::sin(0.0173F * static_cast<float>(i % 907)) +
                      0.3F * std::cos(0.0091F * static_cast<float>(i % 173));
    const float vvv = 0.7F * std::cos(0.0119F * static_cast<float>(i % 643)) +
                      0.25F * std::sin(0.0071F * static_cast<float>(i % 211));
    const std::size_t slot = i % kKvRow;
    h_kv[i] = slot < kKvdim ? kvv : vvv;
  }

  float* d_q = nullptr;
  float* d_kv = nullptr;
  float* d_out = nullptr;
  HIP_OK(hipMalloc(&d_q, h_q.size() * sizeof(float)));
  HIP_OK(hipMalloc(&d_kv, h_kv.size() * sizeof(float)));
  HIP_OK(hipMalloc(&d_out, h_q.size() * sizeof(float)));
  HIP_OK(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float),
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_kv, h_kv.data(), h_kv.size() * sizeof(float),
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemset(d_out, 0, h_q.size() * sizeof(float)));

  const dim3 base_grid((batch + 23) / 24, kQueryHeads);
  const dim3 wmma_grid((batch + 31) / 32, kQueryHeads / kWmmaHeads);
  const auto run_base = [&] {
    AttnKernel<24, 16><<<base_grid, kDim>>>(d_q, d_kv,
                                            static_cast<int>(start_pos),
                                            static_cast<int>(batch), d_out,
                                            kQueryHeads, kHeadDim, kHeadDim,
                                            kKvHeads, kKvdim, kScale);
  };
  const auto run_plain = [&] {
    WmmaCausalAttention<32, 16, 8, false, false><<<wmma_grid, 256>>>(
        d_q, d_kv, d_out, start_pos, batch);
  };
  const auto run_split = [&] {
    WmmaCausalAttention<32, 16, 8, true, true><<<wmma_grid, 256>>>(
        d_q, d_kv, d_out, start_pos, batch);
  };
  const auto run_qkonly = [&] {
    WmmaCausalAttention<32, 16, 8, true, false><<<wmma_grid, 256>>>(
        d_q, d_kv, d_out, start_pos, batch);
  };
  const auto run_pvonly = [&] {
    WmmaCausalAttention<32, 16, 8, false, true><<<wmma_grid, 256>>>(
        d_q, d_kv, d_out, start_pos, batch);
  };

  struct Variant {
    const char* name;
    const char* role;
    bool run_check_ok;  // expected to meet 2e-5 vs FP64?
    const std::function<void()>* run;
  };
  const std::function<void()> f_base = run_base;
  const std::function<void()> f_plain = run_plain;
  const std::function<void()> f_split = run_split;
  const std::function<void()> f_qk = run_qkonly;
  const std::function<void()> f_pv = run_pvonly;
  const Variant variants[] = {
      {"base", "production AttnKernel<24,16>", true, &f_base},
      {"plain", "WMMA single F16 product (ceiling, fails contract)", false,
       &f_plain},
      {"split", "WMMA fp16x3 hi/lo split (contract candidate)", true, &f_split},
      {"qkonly", "ablation: split S only, plain PV", false, &f_qk},
      {"pvonly", "ablation: plain S, split PV", false, &f_pv},
  };

  std::vector<float> got(static_cast<std::size_t>(batch) * kAttnWidth);

  std::fprintf(stderr, "shape: batch=%u start_pos=%u context_end=%u\n", batch,
               start_pos, context_end);

  if (check) {
    std::vector<double> ref(static_cast<std::size_t>(batch) * kAttnWidth, 0.0);
    ReferenceAttention(h_q, h_kv, start_pos, batch, ref);
    double ref_norm = 0.0;
    for (double r : ref) {
      ref_norm = std::max(ref_norm, std::abs(r));
    }
    for (const auto& v : variants) {
      if (!only.empty() && only != v.name) {
        continue;
      }
      HIP_OK(hipMemset(d_out, 0, h_q.size() * sizeof(float)));
      (*v.run)();
      HIP_OK(hipGetLastError());
      HIP_OK(hipDeviceSynchronize());
      HIP_OK(hipMemcpy(got.data(), d_out, got.size() * sizeof(float),
                       hipMemcpyDeviceToHost));
      double max_abs = 0.0;
      double num = 0.0;
      double den = 0.0;
      std::size_t worst = 0;
      for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = std::abs(static_cast<double>(got[i]) - ref[i]);
        if (!std::isfinite(got[i]) || std::isnan(ref[i])) {
          std::fprintf(stderr, "[%s] non-finite at %zu\n", v.name, i);
          return 1;
        }
        if (d > max_abs) {
          max_abs = d;
          worst = i;
        }
        num += d;
        den += std::abs(ref[i]);
      }
      {
        const std::uint32_t t = static_cast<std::uint32_t>(worst / kAttnWidth);
        const std::uint32_t h =
            static_cast<std::uint32_t>((worst / kHeadDim) % kQueryHeads);
        std::fprintf(stderr,
                     "[%s] worst t=%u h=%u d=%lu got=%.9g ref=%.9g\n", v.name,
                     t, h, static_cast<unsigned long>(worst % kHeadDim),
                     static_cast<double>(got[worst]), ref[worst]);
      }
      const double rel = num / den;
      const char* verdict = (rel <= 2e-5)
                                ? "PASS"
                                : (v.run_check_ok ? "FAIL" : "REF");
      std::fprintf(stderr,
                   "[%s] max_elem_rel=%.3e overall_rel=%.3e %s (gate 2e-5)\n",
                   v.name, ref_norm > 0 ? max_abs / ref_norm : max_abs, rel,
                   verdict);
    }
  }

  if (!check || only.empty()) {
    const std::uint64_t pairs = static_cast<std::uint64_t>(batch) *
                                    static_cast<uint64_t>(start_pos) +
                                static_cast<uint64_t>(batch) * (batch + 1) / 2;
    const double flops = 4.0 * static_cast<double>(pairs) * kQueryHeads *
                         kHeadDim;
    for (const auto& v : variants) {
      if (!only.empty() && only != v.name) {
        continue;
      }
      if (only.empty() && (std::string(v.name) == "qkonly" ||
                           std::string(v.name) == "pvonly")) {
        continue;  // ablations are correctness-only unless selected
      }
      const float ms = TimeMs(iters, *v.run);
      std::fprintf(stderr, "[%s] %.3f ms  %.2f TFLOP/s  %s\n", v.name, ms,
                   flops / (static_cast<double>(ms) * 1e-3) / 1e12, v.role);
    }
  }
  return 0;
}
