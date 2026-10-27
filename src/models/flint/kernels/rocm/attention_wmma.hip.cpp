// Prefill attention for Flint on the gfx1151 WMMA matrix cores.
//
// Ported from the gemma4 WMMA prefill kernel, specialized to Flint's head
// dimension 128 (one wave per 16-row block holds its full Q/O slice) and
// adapted to Flint's separate fp16 K/V cache rows, attention sinks and the
// post-dot score scale. The scalar AttentionKernel stays the fallback for
// other head dimensions.
//
// Tiling. A block of kHeads heads sharing one KV head times kQueryBlocks
// 16-query blocks covers kRowBlocks row blocks; every staged K/V tile serves
// all of them. Per 16-key tile: stage K ([key][dim]) and V transposed
// ([dim][key]) in LDS; the wave adds its partial scores to LDS, runs the
// online softmax over them (window, causal and key-limit masks per element)
// and writes the binary16 probabilities; then it rescales its O registers
// and accumulates P V.
//
// Fragment layout (wave32 v_wmma_f32_16x16x16_f16): A holds row L%16 and 16
// contiguous k, B holds column L%16 and 16 contiguous k, and C element i is
// row 2i + L/16, column L%16; both half-waves carry the same operands.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "src/models/flint/kernels/rocm/kernels.hpp"

namespace gufo::models::flint::rocm {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr std::uint32_t kKeys = 16;
constexpr std::uint32_t kSlice = 128;  // head dims per wave
constexpr std::uint32_t kSliceSteps = 8;

struct WmmaArgs {
  const float* q;
  const __half* k_cache;
  const __half* v_cache;
  const float* sinks;
  float* out;
  float scale;
  std::uint32_t rows;
  std::uint32_t first_position;
  std::uint32_t key_limit;
  std::uint32_t window;
  std::uint32_t heads;
  std::uint32_t kv_heads;
  std::uint32_t kv_stride;  // elements per cache row
};

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

struct Bounds {
  std::uint32_t lo;
  std::uint32_t hi;
};

__device__ inline Bounds KeyBounds(const WmmaArgs& a, std::uint32_t position) {
  const std::uint32_t hi = min(position + 1, a.key_limit);
  std::uint32_t lo = 0;
  if (a.window != 0 && position + 1 > a.window) {
    lo = position + 1 - a.window;
  }
  return {min(lo, hi), hi};
}

template<std::uint32_t kHeads, std::uint32_t kQueryBlocks>
__global__ void __launch_bounds__(32 * kHeads * kQueryBlocks)
    WmmaAttentionKernel(const WmmaArgs a) {
  constexpr std::uint32_t kD = 128;
  constexpr std::uint32_t kRowBlocks = kHeads * kQueryBlocks;
  constexpr std::uint32_t kThreads = 32 * kRowBlocks;
  constexpr std::uint32_t kKStride = kD + 8;      // K rows, halves
  constexpr std::uint32_t kVtStride = kKeys + 8;  // V^T rows, halves
  constexpr std::uint32_t kQueries = kQueryBlocks * 16;
  // One wave owns a whole 16-row block: 16 rows spread over the 32 lanes,
  // each row's keys split over kSegLanes lanes of kSegKeys keys.
  constexpr std::uint32_t kSegLanes = 2;
  constexpr std::uint32_t kSegKeys = kKeys / kSegLanes;

  __shared__ __align__(16) __half k_lds[kKeys * kKStride];
  __shared__ __align__(16) __half vt_lds[kD * kVtStride];
  __shared__ float s_lds[kRowBlocks][16][17];
  __shared__ __align__(16) __half p_lds[kRowBlocks][16][kKeys + 8];
  __shared__ __align__(16) __half pl_lds[kRowBlocks][16][kKeys + 8];
  __shared__ float row_scale[kRowBlocks][16];
  __shared__ float row_sum[kRowBlocks][16];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  // Wave-uniform: the row block, head and slice offsets stay scalar.
  const std::uint32_t rb = __builtin_amdgcn_readfirstlane(tid >> 5U);
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;

  const std::uint32_t query_start = blockIdx.x * kQueries;
  const std::uint32_t head_base = blockIdx.y * kHeads;
  const std::uint32_t gqa = a.heads / a.kv_heads;
  const std::uint32_t kv_head = head_base / gqa;
  const std::uint32_t head = head_base + rb % kHeads;
  const std::uint32_t row0 = query_start + (rb / kHeads) * 16;  // this rb

  // Keys the whole block may need.
  const std::uint32_t last_row = min(query_start + kQueries, a.rows) - 1;
  const std::uint32_t position_of_last =
      a.first_position + min(last_row, a.rows - 1);
  const std::uint32_t block_lo =
      KeyBounds(a, a.first_position + min(query_start, a.rows - 1)).lo &
      ~(kKeys - 1);
  const std::uint32_t block_hi = KeyBounds(a, position_of_last).hi;

  // Q fragment: row `sub` of this row block, all 128 dims (8 steps). The
  // FP32 values split into binary16 hi parts plus binary16 residuals (the
  // nemotron nb_attention pattern): S accumulates hi*K in the main
  // accumulator and lo*K in a correction one, so the Q operand keeps ~22
  // mantissa bits against the exact f16-K, which the chaotic MoE routing
  // downstream needs for oracle-argmax parity.
  v16h qh[kSliceSteps];
  v16h ql[kSliceSteps];
  {
    const std::uint32_t row = row0 + sub;
    const bool live = row < a.rows;
    const float* q =
        a.q + (static_cast<std::size_t>(row) * a.heads + head) * kD;
#pragma unroll
    for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
#pragma unroll
      for (std::uint32_t v = 0; v < 16; v += 4) {
        const float4 f = live
                             ? *reinterpret_cast<const float4*>(q + ks * 16 + v)
                             : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        const float x[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
        for (std::uint32_t i = 0; i < 4; ++i) {
          const __half h = __float2half_rn(x[i]);
          qh[ks][v + i] = static_cast<_Float16>(h);
          ql[ks][v + i] =
              static_cast<_Float16>(__float2half_rn(x[i] - __half2float(h)));
        }
      }
    }
  }
  v8f o_acc[kSliceSteps] = {};

  // Online softmax state: lanes 2r and 2r + 1 own row r's key segments.
  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::uint32_t sm_row = lane / kSegLanes;
  const std::uint32_t sm_seg = lane % kSegLanes;
  const std::uint32_t sm_query = row0 + sm_row;
  const std::uint32_t sm_clamped = min(sm_query, a.rows - 1);
  const Bounds sm_keys = KeyBounds(a, a.first_position + sm_clamped);

  const auto* k_cache = a.k_cache;
  const auto* v_cache = a.v_cache;
  const std::size_t kv_stride = a.kv_stride;

  // Each thread stages one 8-half chunk of K and V per tile (kChunks ==
  // kThreads for D = 128); the next tile's chunks load into registers while
  // the current one computes. K chunk: key tid % 16, dims 8 (tid / 16).
  const std::uint32_t chunk_key = tid & (kKeys - 1);
  const std::uint32_t chunk_d8 = (tid >> 4) * 8;
  const __half* const key_base = k_cache + kv_head * kD + chunk_d8;
  uint4 k_next;
  uint4 v_next;
  const auto fetch = [&](std::uint32_t key0) {
    const std::uint32_t position = key0 + chunk_key;
    k_next = make_uint4(0U, 0U, 0U, 0U);
    v_next = make_uint4(0U, 0U, 0U, 0U);
    if (position < block_hi) {
      k_next = *reinterpret_cast<const uint4*>(
          key_base + static_cast<std::size_t>(position) * kv_stride);
      v_next = *reinterpret_cast<const uint4*>(
          v_cache + static_cast<std::size_t>(position) * kv_stride +
          static_cast<std::size_t>(kv_head) * kD + chunk_d8);
    }
  };
  const auto store_keys = [&] {
    *reinterpret_cast<uint4*>(&k_lds[chunk_key * kKStride + chunk_d8]) = k_next;
  };
  // V transposed: keys 2m and 2m + 1 pair into 32-bit words. Even and odd
  // lanes hold keys 2m, 2m + 1 of dims 8 (tid/16); swapping one half of each
  // dword (dpp) gives every lane a dim's even key and the odd lane's even
  // key, so lane 2m writes column key/2 of dims 0..3 and lane 2m + 1 dims
  // 4..7.
  const auto store_values = [&] {
    const bool even = (tid & 1U) == 0;
    const uint4 v = v_next;
    const std::uint32_t send0 = even ? v.z : v.x;
    const std::uint32_t send1 = even ? v.w : v.y;
    const std::uint32_t recv0 =
        static_cast<std::uint32_t>(__builtin_amdgcn_mov_dpp(
            static_cast<int>(send0), 0xB1, 0xF, 0xF, true));
    const std::uint32_t recv1 =
        static_cast<std::uint32_t>(__builtin_amdgcn_mov_dpp(
            static_cast<int>(send1), 0xB1, 0xF, 0xF, true));
    const std::uint32_t own[2] = {even ? v.x : v.z, even ? v.y : v.w};
    const std::uint32_t other[2] = {recv0, recv1};
    auto* rows = reinterpret_cast<std::uint32_t*>(vt_lds);
    const std::uint32_t dim = chunk_d8 + (even ? 0U : 4U);
#pragma unroll
    for (std::uint32_t i = 0; i < 4; ++i) {
      const std::uint32_t mine = (own[i / 2] >> (16 * (i % 2))) & 0xFFFFU;
      const std::uint32_t theirs = (other[i / 2] >> (16 * (i % 2))) & 0xFFFFU;
      const std::uint32_t pair =
          even ? (mine | (theirs << 16)) : (theirs | (mine << 16));
      rows[(dim + i) * (kVtStride / 2) + chunk_key / 2] = pair;
    }
  };
  if (block_lo < block_hi) {
    fetch(block_lo);
    store_keys();
    store_values();
  }
  __syncthreads();

  // Three barriers per tile: after the scores (the next keys may replace
  // this tile's), after the softmax (probabilities and next keys are
  // staged), after the values (the next values may replace this tile's).
  // WMMA may only fuse a tile when every row of this wave attends every
  // column of it. A masked probability is an exact binary16 zero, yet real
  // values sharing the B tile with that zero still shift the f32 accumulators
  // by one unit in the last place (measured on gfx1151), and the zero-filled
  // tail keys make the tile contents launch-specific. Outside the interior
  // the products accumulate per element, where a zero is exact.
  //
  // The interior bound must not depend on a.rows: a row that exists in every
  // prefix length has to take the same arithmetic path for a tile in all of
  // them, and the fused product and the per-element sum round differently.
  // Keyed on the wave's own rows only, a tile whose first column is attended
  // by the wave's last row and whose last column is the wave's first row's
  // own position is interior for the row itself and for every longer launch.
  const std::uint32_t full_hi = a.first_position + row0 + 1U;
  const std::int64_t full_lo =
      a.window == 0U ? 0
                     : static_cast<std::int64_t>(a.first_position) + row0 + 16 -
                           static_cast<std::int64_t>(a.window);
  for (std::uint32_t key0 = block_lo; key0 < block_hi; key0 += kKeys) {
    const bool more = key0 + kKeys < block_hi;
    if (more) {
      fetch(key0 + kKeys);
    }

    // Partial scores over this wave's dim slice: hi*K in the main chain,
    // lo*K in a correction chain combined once before the store.
    {
      v8f s_acc = {};
      v8f s_corr = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
        const v16h k_frag = LoadFrag(&k_lds[sub * kKStride + ks * 16]);
        s_acc = Wmma(qh[ks], k_frag, s_acc);
        s_corr = Wmma(ql[ks], k_frag, s_corr);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[rb][2 * i + half_id][sub] = s_acc[i] + s_corr[i];
      }
    }
    __syncthreads();

    {
      const std::uint32_t position_row = sm_query;
      float vals[kSegKeys];
      float tile_max = -INFINITY;
#pragma unroll
      for (std::uint32_t m = 0; m < kSegKeys; ++m) {
        const std::uint32_t col = sm_seg * kSegKeys + m;
        const std::uint32_t key = key0 + col;
        float s = 0.0F;
        s += s_lds[rb][sm_row][col];
        const bool valid =
            position_row < a.rows && key >= sm_keys.lo && key < sm_keys.hi;
        vals[m] = valid ? s * a.scale : -INFINITY;
        tile_max = fmaxf(tile_max, vals[m]);
      }
      tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 1, 32));
      const float next_max = fmaxf(running_max, tile_max);
      const float scale =
          next_max == -INFINITY ? 1.0F : expf(running_max - next_max);
      float tile_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kSegKeys; ++m) {
        const float w = vals[m] == -INFINITY ? 0.0F : expf(vals[m] - next_max);
        tile_sum += w;
        const __half ph = __float2half_rn(w);
        p_lds[rb][sm_row][sm_seg * kSegKeys + m] = static_cast<_Float16>(ph);
        pl_lds[rb][sm_row][sm_seg * kSegKeys + m] =
            static_cast<_Float16>(__float2half_rn(w - __half2float(ph)));
      }
      tile_sum += __shfl_xor(tile_sum, 1, 32);
      running_max = next_max;
      running_sum = running_sum * scale + tile_sum;
      if (sm_seg == 0) {
        row_scale[rb][sm_row] = scale;
      }
    }
    // The scores are done with this tile's keys: stage the next ones.
    if (more) {
      store_keys();
    }
    __syncthreads();

    float scale2[8];
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      scale2[i] = row_scale[rb][2 * i + half_id];
    }
    const bool full_tile =
        key0 + kKeys <= full_hi &&
        static_cast<std::int64_t>(key0) >= (full_lo > 0 ? full_lo : 0);
    // Rescale the accumulators before the matrix products, and only when a
    // row's running maximum moved (a factor of 1 is exact): after the first
    // tiles almost none do, and the products then issue back to back.
    bool unchanged = true;
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      unchanged = unchanged && scale2[i] == 1.0F;
    }
    if (!__all(unchanged)) {
#pragma unroll
      for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[t][i] *= scale2[i];
        }
      }
    }
    if (full_tile) {
      const v16h p_frag = LoadFrag(&p_lds[rb][sub][0]);
      const v16h pl_frag = LoadFrag(&pl_lds[rb][sub][0]);
#pragma unroll
      for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
        const v16h v_frag = LoadFrag(&vt_lds[(t * 16 + sub) * kVtStride]);
        o_acc[t] = Wmma(p_frag, v_frag, o_acc[t]);
        o_acc[t] = Wmma(pl_frag, v_frag, o_acc[t]);
      }
    } else {
      // Exact scalar accumulation for the boundary tile: a masked zero
      // contributes nothing whatever the values in the same tile are. The
      // element mapping matches the C fragment: row 2i + half_id owns entry
      // i, and lane sub owns dimension t * 16 + sub.
#pragma unroll
      for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
        const __half* const v_row = &vt_lds[(t * 16U + sub) * kVtStride];
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          const std::uint32_t r = 2U * i + half_id;
          float s = o_acc[t][i];
#pragma unroll
          for (std::uint32_t m = 0; m < kKeys; ++m) {
            const float v = static_cast<float>(v_row[m]);
            s += static_cast<float>(p_lds[rb][r][m]) * v;
            s += static_cast<float>(pl_lds[rb][r][m]) * v;
          }
          o_acc[t][i] = s;
        }
      }
    }
    // Every wave is done with this tile's values and probabilities before
    // the next values replace them.
    __syncthreads();
    if (more) {
      store_values();
    }
  }

  if (sm_seg == 0) {
    // The attention sink joins the softmax denominator exactly as in the
    // scalar path: weight exp(sink - running max), never masked.
    row_sum[rb][sm_row] = running_sum + expf(a.sinks[head] - running_max);
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    const std::uint32_t r = 2 * i + half_id;
    const std::uint32_t row = row0 + r;
    if (row >= a.rows) {
      continue;
    }
    const float l = row_sum[rb][r];
    const float inv = l > 0.0F ? 1.0F / l : 0.0F;
    const std::size_t at =
        (static_cast<std::size_t>(row) * a.heads + head) * kD + sub;
#pragma unroll
    for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
      a.out[at + t * 16] = o_acc[t][i] * inv;
    }
  }
}

template<std::uint32_t kHeads, std::uint32_t kQueryBlocks>
bool Launch(const WmmaArgs& a, hipStream_t stream) {
  if ((a.heads / a.kv_heads) % kHeads != 0) {
    return false;
  }
  constexpr std::uint32_t kQueries = 16 * kQueryBlocks;
  const dim3 grid((a.rows + kQueries - 1) / kQueries, a.heads / kHeads);
  WmmaAttentionKernel<kHeads, kQueryBlocks>
      <<<grid, 32 * kHeads * kQueryBlocks, 0, stream>>>(a);
  return true;
}

}  // namespace

bool LaunchWmmaAttention(const float* q, const __half* k_cache,
                         const __half* v_cache, const float* sinks, float scale,
                         std::uint32_t pos0, std::uint32_t batch,
                         std::uint32_t window, std::uint32_t n_head,
                         std::uint32_t n_kv, std::uint32_t kv_row_elements,
                         float* out, hipStream_t stream) {
  if (n_kv == 0 || n_head % n_kv != 0) {
    return false;
  }
  const WmmaArgs a{q,      k_cache, v_cache,        sinks,        out,
                   scale,  batch,   pos0,           pos0 + batch, window,
                   n_head, n_kv,    kv_row_elements};
  // Eight heads x one 16-query block: the staged K/V tile serves the whole
  // GQA group of Flint's 8:1 layout, quartering the staging traffic per
  // score work. Swept bitwise against <2, 4>, <4, 2> and <1, 8> on the
  // measured windowed-513 and full-causal pp2048 shapes
  // (tools/bench/flint_attn_wmma_bench): identical outputs, 31 % faster
  // windowed, 10 % faster full. <2, 4> remains the fallback for head
  // counts whose GQA group is smaller than 8.
  return Launch<8, 1>(a, stream) || Launch<2, 4>(a, stream);
}

}  // namespace gufo::models::flint::rocm
