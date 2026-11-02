#include "src/models/nemotron_h_moe/kernels/rocm/nb_attention.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int kDim = 128;  // max head dimension; one thread per dimension.
constexpr float kNegInf = -3.4e38F;

__global__ void AttnAppendKernel(const float* __restrict__ k_src,
                                 const float* __restrict__ v_src,
                                 float* __restrict__ kv, int pos0, int tokens,
                                 int kvdim) {
  const int64_t n = static_cast<int64_t>(tokens) * (2 * kvdim);
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int pos = pos0 + static_cast<int>(i / (2 * kvdim));
    const int64_t slot = i % (2 * kvdim);
    const int64_t t = (i - slot) / (2 * kvdim);
    const float* src = slot < kvdim ? k_src + t * kvdim + slot
                                    : v_src + t * kvdim + (slot - kvdim);
    kv[static_cast<int64_t>(pos) * (2 * kvdim) + slot] = *src;
  }
}

// Flash-style online-softmax causal attention, one block per (query head,
// row tile). threads are dimension lanes; o[r] accumulates output element
// `tid` of tile row r across the growing context.
// K rows live in LDS transposed (skT[d][j + 1]; the +1 pad avoids the 32-way
// bank conflict seen with sk[j][d] when j is lane-distributed in the score
// loop). Measured 3.2x at pp16384 shapes with TR=24, kTile=16 (gfx1151).
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

    // 2x2 register tile: each thread computes rows (r0, r0+1) x cols
    // (j0, j0+1) so every LDS word feeds two FFMAs; the dot-product chain
    // per output stays d-ascending and matches the scalar-loop result.
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

// Split eight FP32 values into rounded FP16 hi parts and FP16 residuals.
__device__ __forceinline__ void Split8(const float4 lo_in, const float4 hi_in,
                                       __half* hi, __half* lo) {
  const float f[8] = {lo_in.x, lo_in.y, lo_in.z, lo_in.w,
                      hi_in.x, hi_in.y, hi_in.z, hi_in.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const __half h = __float2half_rn(f[i]);
    hi[i] = h;
    lo[i] = __float2half_rn(f[i] - __half2float(h));
  }
}

// WMMA prefill kernel for the production shape (head_dim == head_dim_v ==
// 128, even GQA groups): 32 queries x 2 query heads per block, 16 keys per
// tile, 8 waves across head_dim. FP32 Q/K/V/P ride the FP16 tensor cores as
// hi/lo pairs: S and P V accumulate hi*hi in the main accumulator and the
// correction products hi*lo + lo*hi in a separate one, combined once per
// head-dimension stretch (S) or per key tile (P V). Chaining the corrections
// into the large accumulator instead loses them below its ulp and drifts to
// ~1e-5 over 5000-key contexts (tools/bench/nem_attn_bench.hip). Fragment
// layout: A holds row L % 16 and 16 contiguous k, B a transposed tile
// (column L % 16, 16 contiguous k), C element i is row 2i + L / 16, column
// L % 16. K stages as [key][dim] and V transposed as [dim][key] sharing the
// same LDS banks (hi and lo each), so S completes before PV rewrites them.
constexpr int kWmmaQueries = 32;
constexpr int kWmmaKeys = 16;
constexpr int kWmmaWaves = 8;
constexpr int kWmmaThreads = kWmmaWaves * 32;
constexpr int kWmmaHeadDim = 128;
constexpr int kWmmaHeadsPerBlock = 2;

__launch_bounds__(kWmmaThreads, 1) __global__
    void WmmaAttentionKernel(const float* __restrict__ q,
                             const float* __restrict__ kv, int pos0, int tokens,
                             float* __restrict__ out, int heads, int kvh,
                             int kvdim, float scale) {
  constexpr int kRowBlocks =
      (kWmmaQueries / 16) * kWmmaHeadsPerBlock;  // 16 rows x 2 heads
  constexpr int kRows = kRowBlocks * 16;
  constexpr int kKSteps = kWmmaHeadDim / 16;  // 8 16-wide k steps
  constexpr int kSTiles = kRowBlocks;         // 16 keys per tile
  constexpr int kKStepsPerWave = kKSteps / (kWmmaWaves / kSTiles);  // 4
  constexpr int kDimTilesPerWave = 1;
  constexpr int kSoftmaxLanes = kWmmaThreads / kRows;  // 4
  constexpr int kStride = kWmmaHeadDim + 8;  // K rows / P rows element stride
  constexpr int kVtStride = kWmmaKeys + 8;
  constexpr int kKvLdsHalves = kWmmaHeadDim * kVtStride > kWmmaKeys * kStride
                                   ? kWmmaHeadDim * kVtStride
                                   : kWmmaKeys * kStride;
  constexpr int kVF32 = (kWmmaKeys * kWmmaHeadDim) / (kWmmaThreads * 8);

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int sub = lane & 15;
  const int half_id = lane >> 4;
  const int gqa = heads / kvh;

  const int query_start = static_cast<int>(blockIdx.x) * kWmmaQueries;
  const int kv_head = static_cast<int>(blockIdx.y) / (gqa / kWmmaHeadsPerBlock);
  const int first_query_head = kv_head * gqa + (static_cast<int>(blockIdx.y) %
                                                (gqa / kWmmaHeadsPerBlock)) *
                                                   kWmmaHeadsPerBlock;

  __shared__ __half kv_hi[kKvLdsHalves];
  __shared__ __half kv_lo[kKvLdsHalves];
  __shared__ float s_lds[2][kSTiles][16][16];
  __shared__ __half p_hi[kRows][kVtStride];
  __shared__ __half p_lo[kRows][kVtStride];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const int s_tile = wave % kSTiles;
  const int s_kh = wave / kSTiles;
  const int s_rb = s_tile % kRowBlocks;
  const auto row_block_head = [&](int rb) {
    return first_query_head + (rb % kWmmaHeadsPerBlock);
  };
  const auto row_block_offset = [&](int rb) {
    return (rb / kWmmaHeadsPerBlock) * 16;
  };

  // Q (scaled) lives in registers as hi/lo, k steps distributed 4 waves per
  // S tile half; scale folds into the operand split, matching AttnKernel.
  v16h qh[kKStepsPerWave];
  v16h ql[kKStepsPerWave];
  {
    const int local_query = query_start + row_block_offset(s_rb) + sub;
    const bool live = local_query < tokens;
    const float* q_row =
        q + (static_cast<int64_t>(local_query) * heads + row_block_head(s_rb)) *
                kWmmaHeadDim;
#pragma unroll
    for (int ks = 0; ks < kKStepsPerWave; ++ks) {
      const int d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (int v = 0; v < 2; ++v) {
        const float4 fa = live ? qp[v * 2] : make_float4(0, 0, 0, 0);
        const float4 fb = live ? qp[v * 2 + 1] : make_float4(0, 0, 0, 0);
        const float4 sa = {fa.x * scale, fa.y * scale, fa.z * scale,
                           fa.w * scale};
        const float4 sb = {fb.x * scale, fb.y * scale, fb.z * scale,
                           fb.w * scale};
        const float f[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const __half h = __float2half_rn(f[i]);
          qh[ks][(v * 8) + i] = h;
          ql[ks][(v * 8) + i] = __float2half_rn(f[i] - __half2float(h));
        }
      }
    }
  }

  v8f o_acc[kRowBlocks][kDimTilesPerWave] = {};
  for (int r = tid; r < kRows; r += kWmmaThreads) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0F;
  }

  const int64_t kvrow = 2 * kvdim;
  const float* kv_k = kv + kv_head * kWmmaHeadDim;
  const float* kv_v = kv + kvdim + kv_head * kWmmaHeadDim;
  const int context_end = pos0 + tokens;
  const int max_visible = context_end < pos0 + query_start + kWmmaQueries
                              ? context_end
                              : pos0 + query_start + kWmmaQueries;

  for (int key_start = 0; key_start < max_visible; key_start += kWmmaKeys) {
    // Prefetch V (FP32) before the S phase; parked in registers until the
    // transpose so the LDS staging slot stays free for K.
    uint4 v_raw[kVF32 * 2];
    {
      const int v_key = lane % kWmmaKeys;
      const int v_slice = (tid / kWmmaKeys) * (kVF32 * 8);
      const int key_position = key_start + v_key;
      const float* src =
          kv_v + (static_cast<int64_t>(key_position)) * kvrow + v_slice;
      const bool live = key_position < context_end;
#pragma unroll
      for (int j = 0; j < kVF32; ++j) {
        v_raw[j * 2] = live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                            : make_uint4(0u, 0u, 0u, 0u);
        v_raw[j * 2 + 1] =
            live ? *reinterpret_cast<const uint4*>(src + (j * 8) + 4)
                 : make_uint4(0u, 0u, 0u, 0u);
      }
    }

    // --- stage K as [key][dim] with the FP32->FP16 hi/lo split.
    __syncthreads();
    for (int idx = tid; idx < kWmmaKeys * (kWmmaHeadDim / 8);
         idx += kWmmaThreads) {
      const int key_row = idx / (kWmmaHeadDim / 8);
      const int d8 = (idx % (kWmmaHeadDim / 8)) * 8;
      const int key_position = key_start + key_row;
      const bool live = key_position < context_end;
      const float* src =
          kv_k + (static_cast<int64_t>(key_position)) * kvrow + d8;
      const float4 fa = live ? *reinterpret_cast<const float4*>(src)
                             : make_float4(0, 0, 0, 0);
      const float4 fb = live ? *reinterpret_cast<const float4*>(src + 4)
                             : make_float4(0, 0, 0, 0);
      __half hi[8];
      __half lo[8];
      Split8(fa, fb, hi, lo);
      *reinterpret_cast<uint4*>(&kv_hi[(key_row * kStride) + d8]) =
          *reinterpret_cast<const uint4*>(hi);
      *reinterpret_cast<uint4*>(&kv_lo[(key_row * kStride) + d8]) =
          *reinterpret_cast<const uint4*>(lo);
    }
    __syncthreads();

    // --- S = Q K^T; correction products accumulate in their own register
    // sum so they never vanish into the main accumulator's rounding.
    {
      v8f s_acc = {};
      v8f s_lo_acc = {};
#pragma unroll
      for (int ks = 0; ks < kKStepsPerWave; ++ks) {
        const int d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const int koff = (sub * kStride) + d0;
        const v16h bh = LoadFrag(&kv_hi[koff]);
        const v16h bl = LoadFrag(&kv_lo[koff]);
        s_acc = Wmma(qh[ks], bh, s_acc);
        s_lo_acc = Wmma(qh[ks], bl, s_lo_acc);
        s_lo_acc = Wmma(ql[ks], bh, s_lo_acc);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i)
        s_acc[i] += s_lo_acc[i];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
    __syncthreads();

    // --- online softmax: kSoftmaxLanes threads per query row.
    {
      const int rg = tid / kSoftmaxLanes;
      const int seg = tid % kSoftmaxLanes;
      constexpr int kPerLane = kWmmaKeys / kSoftmaxLanes;
      const int rb = rg / 16;
      const int row = rg % 16;
      const int local_query = query_start + row_block_offset(rb) + row;
      const int absolute_query = pos0 + local_query;
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (int m = 0; m < kPerLane; ++m) {
        const int col = (seg * kPerLane) + m;
        const int key_position = key_start + col;
        const bool valid = local_query < tokens &&
                           key_position <= absolute_query &&
                           key_position < context_end;
        vals[m] = valid ? (s_lds[0][rb][row][col] + s_lds[1][rb][row][col])
                        : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (int off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      }
      const float prev_max = row_max[rg];
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (int m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        const __half h = __float2half_rn(w);
        p_hi[rg][(seg * kPerLane) + m] = h;
        p_lo[rg][(seg * kPerLane) + m] = __float2half_rn(w - __half2float(h));
      }
#pragma unroll
      for (int off = 1; off < kSoftmaxLanes; off <<= 1) {
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
    for (int rb = 0; rb < kRowBlocks; ++rb) {
      float scale8[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        scale8[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
      for (int t = 0; t < kDimTilesPerWave; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale8[i];
        }
      }
    }

    // --- stage V transposed as [dim][key], hi/lo, reusing the K LDS banks.
    {
      const int v_key = lane % kWmmaKeys;
      const int v_slice = (tid / kWmmaKeys) * (kVF32 * 8);
#pragma unroll
      for (int j = 0; j < kVF32; ++j) {
        const float4 fa = *reinterpret_cast<const float4*>(&v_raw[j * 2]);
        const float4 fb = *reinterpret_cast<const float4*>(&v_raw[j * 2 + 1]);
        __half hi[8];
        __half lo[8];
        Split8(fa, fb, hi, lo);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int d = v_slice + (j * 8) + i;
          kv_hi[(d * kVtStride) + v_key] = hi[i];
          kv_lo[(d * kVtStride) + v_key] = lo[i];
        }
      }
    }
    __syncthreads();

    // --- O += P V: the corrections per (row block, dim tile) accumulate
    // separately before one combine into the running output.
#pragma unroll
    for (int t = 0; t < kDimTilesPerWave; ++t) {
      const int dim_tile = wave + (t * kWmmaWaves);
      const v16h vh = LoadFrag(&kv_hi[(dim_tile * 16 + sub) * kVtStride]);
      const v16h vl = LoadFrag(&kv_lo[(dim_tile * 16 + sub) * kVtStride]);
#pragma unroll
      for (int rb = 0; rb < kRowBlocks; ++rb) {
        const v16h ph = LoadFrag(&p_hi[(rb * 16) + sub][0]);
        const v16h pl = LoadFrag(&p_lo[(rb * 16) + sub][0]);
        v8f o_lo = {};
        o_lo = Wmma(ph, vl, o_lo);
        o_lo = Wmma(pl, vh, o_lo);
        o_acc[rb][t] = Wmma(ph, vh, o_acc[rb][t]);
        o_acc[rb][t] += o_lo;
      }
    }
  }
  __syncthreads();

  // --- epilogue: divide by the online denominator.
#pragma unroll
  for (int rb = 0; rb < kRowBlocks; ++rb) {
    const int query_head = row_block_head(rb);
    const int row_offset = row_block_offset(rb);
    const int dim_tile = wave;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = (2 * i) + half_id;
      const int local_query = query_start + row_offset + row;
      if (local_query >= tokens) {
        continue;
      }
      const float denominator = row_sum[(rb * 16) + row];
      const int64_t offset =
          (static_cast<int64_t>(local_query) * heads + query_head) *
              kWmmaHeadDim +
          (dim_tile * 16) + sub;
      out[offset] =
          (denominator > 0.0F) ? (o_acc[rb][0][i] / denominator) : 0.0F;
    }
  }
}

}  // namespace

extern "C" int nbl_attn_append(const float* k_src, const float* v_src,
                               float* kv, int pos0, int tokens, int kvdim,
                               hipStream_t stream) {
  if (!k_src || !v_src || !kv || pos0 < 0 || tokens <= 0 || kvdim <= 0) {
    return -1;
  }
  const int64_t n = static_cast<int64_t>(tokens) * (2 * kvdim);
  const int grid = static_cast<int>((n + 255) / 256);
  const int launch = grid < 1024 ? grid : 1024;
  AttnAppendKernel<<<launch, 256, 0, stream>>>(k_src, v_src, kv, pos0, tokens,
                                               kvdim);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

extern "C" int nbl_attn(const float* q, const float* kv, int pos0, int tokens,
                        float* out, int heads, int hd, int hdv, int kvh,
                        int kvdim, float scale, hipStream_t stream) {
  if (!q || !kv || !out || pos0 < 0 || tokens <= 0 || heads <= 0 || kvh <= 0 ||
      heads % kvh != 0 || hd <= 0 || hd > kDim || hdv <= 0 || hdv > kDim ||
      kvdim < kvh * hd) {
    return -1;
  }
  // Prefill chunks on the production shape take the WMMA hi/lo split path
  // (6.5x at deep pp32768 tiles); partial rows and causal tails are handled
  // inside the kernel. Decode and other shapes keep AttnKernel.
  if (hd == kWmmaHeadDim && hdv == kWmmaHeadDim && tokens >= kWmmaQueries &&
      kvdim == kvh * hd && ((heads / kvh) % 2) == 0) {
    const dim3 wmma_grid(
        static_cast<unsigned>((tokens + kWmmaQueries - 1) / kWmmaQueries),
        static_cast<unsigned>(heads / kWmmaHeadsPerBlock));
    WmmaAttentionKernel<<<wmma_grid, kWmmaThreads, 0, stream>>>(
        q, kv, pos0, tokens, out, heads, kvh, kvdim, scale);
    return hipGetLastError() == hipSuccess ? 0 : -2;
  }
  // Partial tiles (including decode with tokens == 1) take the guarded path.
  const dim3 grid(static_cast<unsigned>((tokens + 23) / 24),
                  static_cast<unsigned>(heads));
  AttnKernel<24, 16><<<grid, kDim, 0, stream>>>(q, kv, pos0, tokens, out, heads,
                                                hd, hdv, kvh, kvdim, scale);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}
