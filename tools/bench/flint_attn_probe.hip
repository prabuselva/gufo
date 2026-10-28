// C7r attention correction-chain probe for the shipped WMMA attention
// <8, 1>. The probe kernels are generated copies of WmmaAttentionKernel
// with the hi/lo Q-correction machinery varied; all variants except
// ProbeKernelRebuildQ (bit-identical check) and the shipped kernel have
// INVALID numerics and exist only to time the correction chain.
// Measured on gfx1151 (full shape, exclusive GPU) against the shipped
// kernel in the same process: dropping the lo chain runs -44 %, every
// bitwise-preserving rework (per-tile rebuild, occupancy copies,
// dead-lo sinking, split accumulator chains) stays within +/- 2 %.
// Conclusion recorded in docs/models/flint/STATUS.md (C7r): no adoptable
// lever; the lo correction is quality-required (router argmax parity).
// Launch grids must cover all heads: dim3((kRows + 15) / 16, kHeadsN / 8).
//
// Build:
//   PATH=/opt/rocm/lib/llvm/bin:$PATH clang++ --offload-arch=gfx1151 \
//     --driver-mode=g++ -L/opt/rocm/lib --hip-link -O3 -std=c++20 -x hip \
//     -I . tools/bench/flint_attn_probe.hip -o /tmp/flint_attn_probe
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/flint/kernels/rocm/attention_wmma.hip.cpp"

// --- probe kernel (generated copy of the shipped kernel, correction stripped)
namespace gufo::models::flint::rocm {
namespace {
template<std::uint32_t kHeads, std::uint32_t kQueryBlocks>
__global__ void __launch_bounds__(32 * kHeads * kQueryBlocks)
    ProbeKernelNoQl(const WmmaArgs a) {
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
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
        const v16h k_frag = LoadFrag(&k_lds[sub * kKStride + ks * 16]);
        s_acc = Wmma(qh[ks], k_frag, s_acc);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[rb][2 * i + half_id][sub] = s_acc[i];
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

}  // namespace
}  // namespace gufo::models::flint::rocm

// --- probe kernel 2: Q fragments rebuilt per tile (bitwise target)
namespace gufo::models::flint::rocm {
namespace {
template<std::uint32_t kHeads, std::uint32_t kQueryBlocks>
__global__ void __launch_bounds__(32 * kHeads * kQueryBlocks)
    ProbeKernelRebuildQ(const WmmaArgs a) {
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
  const std::uint32_t q_row = row0 + sub;
  const bool q_live = q_row < a.rows;
  const float* const q_base =
      a.q + (static_cast<std::size_t>(q_row) * a.heads + head) * kD;
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
        v16h qh;
        v16h ql;
#pragma unroll
        for (std::uint32_t v = 0; v < 16; v += 4) {
          const float4 f =
              q_live ? *reinterpret_cast<const float4*>(q_base + ks * 16 + v)
                     : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
          const float x[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
          for (std::uint32_t i = 0; i < 4; ++i) {
            const __half h = __float2half_rn(x[i]);
            qh[v + i] = static_cast<_Float16>(h);
            ql[v + i] =
                static_cast<_Float16>(__float2half_rn(x[i] - __half2float(h)));
          }
        }
        const v16h k_frag = LoadFrag(&k_lds[sub * kKStride + ks * 16]);
        s_acc = Wmma(qh, k_frag, s_acc);
        s_corr = Wmma(ql, k_frag, s_corr);
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

}  // namespace
}  // namespace gufo::models::flint::rocm

// --- probe kernels 3/4: launch_bounds(256, 2) occupancy variants
namespace gufo::models::flint::rocm {
namespace {
template<std::uint32_t kHeads, std::uint32_t kQueryBlocks>
__global__ void __launch_bounds__(32 * kHeads * kQueryBlocks, 2)
    ProbeKernelLb2(const WmmaArgs a) {
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
__global__ void __launch_bounds__(32 * kHeads * kQueryBlocks, 2)
    ProbeKernelRebuildQ2b(const WmmaArgs a) {
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
  const std::uint32_t q_row = row0 + sub;
  const bool q_live = q_row < a.rows;
  const float* const q_base =
      a.q + (static_cast<std::size_t>(q_row) * a.heads + head) * kD;
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
        v16h qh;
        v16h ql;
#pragma unroll
        for (std::uint32_t v = 0; v < 16; v += 4) {
          const float4 f =
              q_live ? *reinterpret_cast<const float4*>(q_base + ks * 16 + v)
                     : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
          const float x[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
          for (std::uint32_t i = 0; i < 4; ++i) {
            const __half h = __float2half_rn(x[i]);
            qh[v + i] = static_cast<_Float16>(h);
            ql[v + i] =
                static_cast<_Float16>(__float2half_rn(x[i] - __half2float(h)));
          }
        }
        const v16h k_frag = LoadFrag(&k_lds[sub * kKStride + ks * 16]);
        s_acc = Wmma(qh, k_frag, s_acc);
        s_corr = Wmma(ql, k_frag, s_corr);
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

}  // namespace
}  // namespace gufo::models::flint::rocm

// --- probe kernel 5: rebuild frags per tile, lo WMMA NOT executed
namespace gufo::models::flint::rocm {
namespace {
template<std::uint32_t kHeads, std::uint32_t kQueryBlocks>
__global__ void __launch_bounds__(32 * kHeads * kQueryBlocks)
    ProbeKernelBuildNoLo(const WmmaArgs a) {
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
  const std::uint32_t q_row = row0 + sub;
  const bool q_live = q_row < a.rows;
  const float* const q_base =
      a.q + (static_cast<std::size_t>(q_row) * a.heads + head) * kD;
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
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
        v16h qh;
        v16h ql;
#pragma unroll
        for (std::uint32_t v = 0; v < 16; v += 4) {
          const float4 f =
              q_live ? *reinterpret_cast<const float4*>(q_base + ks * 16 + v)
                     : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
          const float x[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
          for (std::uint32_t i = 0; i < 4; ++i) {
            const __half h = __float2half_rn(x[i]);
            qh[v + i] = static_cast<_Float16>(h);
            ql[v + i] =
                static_cast<_Float16>(__float2half_rn(x[i] - __half2float(h)));
          }
        }
        const v16h k_frag = LoadFrag(&k_lds[sub * kKStride + ks * 16]);
        s_acc = Wmma(qh, k_frag, s_acc);
        if (ql[0] == static_cast<_Float16>(65504.0F)) {
          s_acc[0] += float{0x1p-30F};
        }
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[rb][2 * i + half_id][sub] = s_acc[i];
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

}  // namespace
}  // namespace gufo::models::flint::rocm

// --- probe kernels C/D (bisect of the correction chain; numerics differ)
namespace gufo::models::flint::rocm {
namespace {
template<std::uint32_t kHeads, std::uint32_t kQueryBlocks>
__global__ void __launch_bounds__(32 * kHeads * kQueryBlocks)
    ProbeKernelPersistNoSCorr(const WmmaArgs a) {
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
        if (s_corr[0] == -1e38F) s_corr[1] = 1.0F;
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[rb][2 * i + half_id][sub] = s_acc[i];
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
__global__ void __launch_bounds__(32 * kHeads * kQueryBlocks)
    ProbeKernelHiPersistLoRebuild(const WmmaArgs a) {
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
        }
      }
    }
  }
  const std::uint32_t q_row = row0 + sub;
  const bool q_live = q_row < a.rows;
  const float* const q_base =
      a.q + (static_cast<std::size_t>(q_row) * a.heads + head) * kD;
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
        v16h ql;
#pragma unroll
        for (std::uint32_t v = 0; v < 16; v += 4) {
          const float4 f =
              q_live ? *reinterpret_cast<const float4*>(q_base + ks * 16 + v)
                     : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
          const float x[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
          for (std::uint32_t i = 0; i < 4; ++i) {
            ql[v + i] = static_cast<_Float16>(
                __float2half_rn(x[i] - __half2float(__float2half_rn(x[i]))));
          }
        }
        const v16h k_frag = LoadFrag(&k_lds[sub * kKStride + ks * 16]);
        s_acc = Wmma(qh[ks], k_frag, s_acc);
        s_corr = Wmma(ql, k_frag, s_corr);
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

}  // namespace
}  // namespace gufo::models::flint::rocm

// --- probe kernel E: per-tile hi-only rebuild (no ql code; numerics differ)
namespace gufo::models::flint::rocm {
namespace {
template<std::uint32_t kHeads, std::uint32_t kQueryBlocks>
__global__ void __launch_bounds__(32 * kHeads * kQueryBlocks)
    ProbeKernelRebuildHiOnly(const WmmaArgs a) {
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
  const std::uint32_t q_row = row0 + sub;
  const bool q_live = q_row < a.rows;
  const float* const q_base =
      a.q + (static_cast<std::size_t>(q_row) * a.heads + head) * kD;
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
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
        v16h qh;
#pragma unroll
        for (std::uint32_t v = 0; v < 16; v += 4) {
          const float4 f =
              q_live ? *reinterpret_cast<const float4*>(q_base + ks * 16 + v)
                     : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
          const float x[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
          for (std::uint32_t i = 0; i < 4; ++i) {
            qh[v + i] = static_cast<_Float16>(__float2half_rn(x[i]));
          }
        }
        const v16h k_frag = LoadFrag(&k_lds[sub * kKStride + ks * 16]);
        s_acc = Wmma(qh, k_frag, s_acc);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[rb][2 * i + half_id][sub] = s_acc[i];
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

}  // namespace
}  // namespace gufo::models::flint::rocm

// --- probe kernel F: split accumulator chains (reassociates; not bitwise)
namespace gufo::models::flint::rocm {
namespace {
template<std::uint32_t kHeads, std::uint32_t kQueryBlocks>
__global__ void __launch_bounds__(32 * kHeads * kQueryBlocks)
    ProbeKernelSplitChain(const WmmaArgs a) {
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
      v8f s_acc_a = {};
      v8f s_acc_b = {};
      v8f s_corr_a = {};
      v8f s_corr_b = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSliceSteps / 2; ++ks) {
        const v16h k_frag = LoadFrag(&k_lds[sub * kKStride + ks * 16]);
        s_acc_a = Wmma(qh[ks], k_frag, s_acc_a);
        s_corr_a = Wmma(ql[ks], k_frag, s_corr_a);
      }
#pragma unroll
      for (std::uint32_t ks = kSliceSteps / 2; ks < kSliceSteps; ++ks) {
        const v16h k_frag = LoadFrag(&k_lds[sub * kKStride + ks * 16]);
        s_acc_b = Wmma(qh[ks], k_frag, s_acc_b);
        s_corr_b = Wmma(ql[ks], k_frag, s_corr_b);
      }
      const v8f s_acc = s_acc_a + s_acc_b;
      const v8f s_corr = s_corr_a + s_corr_b;
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

// --- C7t candidate: two-barrier schedule with double-buffered K/V staging.
// Arithmetic and tile values are byte-identical to the shipped kernel; only
// the loop placement changes. The shipped loop runs three __syncthreads per
// 16-key tile: after the scores, after softmax+next-keys, after the values
// prefetch+before the next tile. Double-buffering K and V^T lets each
// "readers are done" barrier gate a store into the *other* buffer, so the
// third barrier drops:
//   fetch(next) -> scores(k[slot]) -> SYNC_A -> store_keys(other); softmax
//   -> SYNC_B -> store_values(other) -> rescale + PV(vt[slot]).
// The register prefetch stays at the top of the iteration, exactly as in
// the shipped kernel: each iteration's fetch feeds that same iteration's
// stores. SYNC_A proves every wave finished reading k[slot], so the next keys may
// stage into the other buffer, and SYNC_B proves the previous tile's PV
// retired before it in each wave's program order (PV precedes the next
// iteration's SYNC_A/SYNC_B), so vt[other] is safe to write. p/pl/row_scale
// stay single-buffered: they are overwritten only after the next SYNC_B,
// once every wave has consumed them in softmax and PV.
template<std::uint32_t kHeads, std::uint32_t kQueryBlocks>
__global__ void __launch_bounds__(32 * kHeads * kQueryBlocks)
    ProbeKernelTwoSyncs(const WmmaArgs a) {
  constexpr std::uint32_t kD = 128;
  constexpr std::uint32_t kRowBlocks = kHeads * kQueryBlocks;
  constexpr std::uint32_t kKStride = kD + 8;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  constexpr std::uint32_t kQueries = kQueryBlocks * 16;
  constexpr std::uint32_t kSegLanes = 2;
  constexpr std::uint32_t kSegKeys = kKeys / kSegLanes;

  __shared__ __align__(16) __half k_lds[2][kKeys * kKStride];
  __shared__ __align__(16) __half vt_lds[2][kD * kVtStride];
  __shared__ float s_lds[kRowBlocks][16][17];
  __shared__ __align__(16) __half p_lds[kRowBlocks][16][kKeys + 8];
  __shared__ __align__(16) __half pl_lds[kRowBlocks][16][kKeys + 8];
  __shared__ float row_scale[kRowBlocks][16];
  __shared__ float row_sum[kRowBlocks][16];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t rb = __builtin_amdgcn_readfirstlane(tid >> 5U);
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;

  const std::uint32_t query_start = blockIdx.x * kQueries;
  const std::uint32_t head_base = blockIdx.y * kHeads;
  const std::uint32_t gqa = a.heads / a.kv_heads;
  const std::uint32_t kv_head = head_base / gqa;
  const std::uint32_t head = head_base + rb % kHeads;
  const std::uint32_t row0 = query_start + (rb / kHeads) * 16;

  const std::uint32_t last_row = min(query_start + kQueries, a.rows) - 1;
  const std::uint32_t position_of_last =
      a.first_position + min(last_row, a.rows - 1);
  const std::uint32_t block_lo =
      KeyBounds(a, a.first_position + min(query_start, a.rows - 1)).lo &
      ~(kKeys - 1);
  const std::uint32_t block_hi = KeyBounds(a, position_of_last).hi;

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
  const auto store_keys = [&](int slot) {
    *reinterpret_cast<uint4*>(&k_lds[slot][chunk_key * kKStride + chunk_d8]) =
        k_next;
  };
  const auto store_values = [&](int slot) {
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
    auto* rows = reinterpret_cast<std::uint32_t*>(vt_lds[slot]);
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
    store_keys(0);
    store_values(0);
  }
  __syncthreads();

  const std::uint32_t full_hi = a.first_position + row0 + 1U;
  const std::int64_t full_lo =
      a.window == 0U ? 0
                     : static_cast<std::int64_t>(a.first_position) + row0 + 16 -
                           static_cast<std::int64_t>(a.window);
  int slot = 0;
  for (std::uint32_t key0 = block_lo; key0 < block_hi;
       key0 += kKeys, slot ^= 1) {
    const bool more = key0 + kKeys < block_hi;
    const int nslot = slot ^ 1;
    if (more) {
      fetch(key0 + kKeys);
    }

    {
      v8f s_acc = {};
      v8f s_corr = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
        const v16h k_frag = LoadFrag(&k_lds[slot][sub * kKStride + ks * 16]);
        s_acc = Wmma(qh[ks], k_frag, s_acc);
        s_corr = Wmma(ql[ks], k_frag, s_corr);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[rb][2 * i + half_id][sub] = s_acc[i] + s_corr[i];
      }
    }
    __syncthreads();
    if (more) {
      store_keys(nslot);
    }

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
    __syncthreads();
    if (more) {
      store_values(nslot);
    }

    float sc2[8];
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      sc2[i] = row_scale[rb][2 * i + half_id];
    }
    const bool full_tile =
        key0 + kKeys <= full_hi &&
        static_cast<std::int64_t>(key0) >= (full_lo > 0 ? full_lo : 0);
    bool unchanged = true;
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      unchanged = unchanged && sc2[i] == 1.0F;
    }
    if (!__all(unchanged)) {
#pragma unroll
      for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[t][i] *= sc2[i];
        }
      }
    }
    if (full_tile) {
      const v16h p_frag = LoadFrag(&p_lds[rb][sub][0]);
      const v16h pl_frag = LoadFrag(&pl_lds[rb][sub][0]);
#pragma unroll
      for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
        const v16h v_frag =
            LoadFrag(&vt_lds[slot][(t * 16 + sub) * kVtStride]);
        o_acc[t] = Wmma(p_frag, v_frag, o_acc[t]);
        o_acc[t] = Wmma(pl_frag, v_frag, o_acc[t]);
      }
    } else {
#pragma unroll
      for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
        const __half* const v_row =
            &vt_lds[slot][(t * 16U + sub) * kVtStride];
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
  }

  if (sm_seg == 0) {
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

}  // namespace
}  // namespace gufo::models::flint::rocm

namespace fr = gufo::models::flint::rocm;
using fr::WmmaArgs;
using gufo::models::flint::rocm::ProbeKernelNoQl;
using gufo::models::flint::rocm::ProbeKernelRebuildQ;
using gufo::models::flint::rocm::ProbeKernelLb2;
using gufo::models::flint::rocm::ProbeKernelBuildNoLo;
using gufo::models::flint::rocm::ProbeKernelPersistNoSCorr;
using gufo::models::flint::rocm::ProbeKernelRebuildHiOnly;
using gufo::models::flint::rocm::ProbeKernelSplitChain;
using gufo::models::flint::rocm::ProbeKernelHiPersistLoRebuild;
using gufo::models::flint::rocm::ProbeKernelRebuildQ2b;
using gufo::models::flint::rocm::ProbeKernelTwoSyncs;
constexpr std::uint32_t kRows = 2048;
constexpr std::uint32_t kHeadsN = 32;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kDim = 128;
constexpr std::uint32_t kPos0 = 512;
constexpr std::size_t kQn =
    static_cast<std::size_t>(kRows) * kHeadsN * kDim;

#define CHECK(cmd)                                                   \
  do {                                                               \
    const hipError_t err__ = (cmd);                                  \
    if (err__ != hipSuccess) {                                       \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                \
                   hipGetErrorString(err__), __FILE__, __LINE__);     \
      std::exit(1);                                                  \
    }                                                                \
  } while (0)

WmmaArgs MakeArgs(const float* q, const __half* k, const __half* v,
                      const float* sinks, float* out, std::uint32_t window) {
  fr::WmmaArgs a{};
  a.q = q;
  a.k_cache = k;
  a.v_cache = v;
  a.sinks = sinks;
  a.out = out;
  a.scale = 1.0F / std::sqrt(static_cast<float>(kDim));
  a.rows = kRows;
  a.first_position = kPos0;
  a.key_limit = kPos0 + kRows;
  a.window = window;
  a.heads = kHeadsN;
  a.kv_heads = kKvHeads;
  a.kv_stride = kKvHeads * kDim;
  return a;
}

void LaunchOne(int cfg, const WmmaArgs& a) {
  if (cfg == 2 || cfg == 4) {
    ProbeKernelRebuildQ<8, 1><<<dim3((kRows + 15) / 16, kHeadsN / 8), 256>>>(a);
  } else {
    fr::WmmaAttentionKernel<8, 1><<<dim3((kRows + 15) / 16, kHeadsN / 8), 256>>>(a);
  }
}

double TimeIt(int cfg, const WmmaArgs& a, int iters) {
  const auto launch = [&] {
    if (cfg == 1) {
      ProbeKernelNoQl<8, 1><<<dim3((kRows + 15) / 16, kHeadsN / 8), 256>>>(a);
    } else if (cfg == 2) {
      ProbeKernelRebuildQ<8, 1><<<dim3((kRows + 15) / 16, kHeadsN / 8), 256>>>(a);
    } else if (cfg == 3) {
      ProbeKernelLb2<8, 1><<<dim3((kRows + 15) / 16, kHeadsN / 8), 256>>>(a);
    } else if (cfg == 4) {
      ProbeKernelRebuildQ2b<8, 1><<<dim3((kRows + 15) / 16, kHeadsN / 8), 256>>>(a);
    } else if (cfg == 5) {
      ProbeKernelBuildNoLo<8, 1><<<dim3((kRows + 15) / 16, kHeadsN / 8), 256>>>(a);
    } else if (cfg == 6) {
      ProbeKernelPersistNoSCorr<8, 1><<<dim3((kRows + 15) / 16, kHeadsN / 8), 256>>>(a);
    } else if (cfg == 7) {
      ProbeKernelHiPersistLoRebuild<8, 1><<<dim3((kRows + 15) / 16, kHeadsN / 8), 256>>>(a);
    } else if (cfg == 8) {
      ProbeKernelRebuildHiOnly<8, 1><<<dim3((kRows + 15) / 16, kHeadsN / 8), 256>>>(a);
    } else if (cfg == 9) {
      ProbeKernelSplitChain<8, 1><<<dim3((kRows + 15) / 16, kHeadsN / 8), 256>>>(a);
    } else {
      fr::WmmaAttentionKernel<8, 1><<<dim3((kRows + 15) / 16, kHeadsN / 8), 256>>>(a);
    }
  };
  for (int i = 0; i < 3; ++i) launch();
  CHECK(hipDeviceSynchronize());
  hipEvent_t beg, end;
  CHECK(hipEventCreate(&beg));
  CHECK(hipEventCreate(&end));
  CHECK(hipEventRecord(beg));
  for (int i = 0; i < iters; ++i) launch();
  CHECK(hipEventRecord(end));
  CHECK(hipEventSynchronize(end));
  float ms = 0.0F;
  CHECK(hipEventElapsedTime(&ms, beg, end));
  CHECK(hipEventDestroy(beg));
  CHECK(hipEventDestroy(end));
  return ms / iters;
}


// C7t long-context configuration sweep. C7g picked <8, 1> at pp2048 shapes
// where one launch's KV fits the MALL, so KV re-reads across the 16-row
// blocks are nearly free. At pp32768 a full dispatch serves ~18K average
// keys: each extra row-block column re-reads the whole KV span, so wider
// kQueries configs (<4, 2>, <1, 8>) cut KV traffic 2x/8x and may win where
// they lost at short context. Bitwise-equal per row block by the same
// per-row invariants as the C7g sweep; mismatches are printed to prove it.
namespace {

WmmaArgs MakeCfgArgs(const float* q, const __half* k, const __half* v,
                     const float* sink, float* out, std::uint32_t rows,
                     std::uint32_t pos0, std::uint32_t window) {
  WmmaArgs a{};
  a.q = q;
  a.k_cache = k;
  a.v_cache = v;
  a.sinks = sink;
  a.out = out;
  a.scale = 1.0F / std::sqrt(static_cast<float>(kDim));
  a.rows = rows;
  a.first_position = pos0;
  a.key_limit = pos0 + rows;
  a.window = window;
  a.heads = kHeadsN;
  a.kv_heads = kKvHeads;
  a.kv_stride = kKvHeads * kDim;
  return a;
}

template <typename LaunchFn>
double TimeLaunch(LaunchFn launch, int iters) {
  for (int i = 0; i < 2; ++i) {
    launch();
  }
  CHECK(hipDeviceSynchronize());
  hipEvent_t beg, end;
  CHECK(hipEventCreate(&beg));
  CHECK(hipEventCreate(&end));
  CHECK(hipEventRecord(beg));
  for (int i = 0; i < iters; ++i) {
    launch();
  }
  CHECK(hipEventRecord(end));
  CHECK(hipEventSynchronize(end));
  float ms = 0.0F;
  CHECK(hipEventElapsedTime(&ms, beg, end));
  CHECK(hipEventDestroy(beg));
  CHECK(hipEventDestroy(end));
  return ms / iters;
}

template <std::uint32_t KH, std::uint32_t QB>
double TimeCfg(const float* q, const __half* k, const __half* v,
               const float* sink, float* out, std::uint32_t rows,
               std::uint32_t pos0, std::uint32_t window, int iters) {
  WmmaArgs a = MakeCfgArgs(q, k, v, sink, out, rows, pos0, window);
  constexpr std::uint32_t kQueries = 16 * QB;
  return TimeLaunch(
      [&] {
        fr::WmmaAttentionKernel<KH, QB>
            <<<dim3((rows + kQueries - 1) / kQueries, kHeadsN / KH),
               32 * KH * QB>>>(a);
      },
      iters);
}

// C7t: two-barrier double-buffered variant at the shipped <8, 1> shape.
double TimeTwoSyncs(const float* q, const __half* k, const __half* v,
                    const float* sink, float* out, std::uint32_t rows,
                    std::uint32_t pos0, std::uint32_t window, int iters) {
  WmmaArgs a = MakeCfgArgs(q, k, v, sink, out, rows, pos0, window);
  return TimeLaunch(
      [&] {
        ProbeKernelTwoSyncs<8, 1>
            <<<dim3((rows + 15) / 16, kHeadsN / 8), 256>>>(a);
      },
      iters);
}

struct CfgShape {
  const char* label;
  std::uint32_t rows;
  std::uint32_t pos0;
  int iters;
};

std::size_t Mismatches(const float* ref, const float* got, std::size_t n) {
  std::vector<float> r(n), g(n);
  CHECK(hipMemcpy(r.data(), ref, n * sizeof(float), hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(g.data(), got, n * sizeof(float), hipMemcpyDeviceToHost));
  std::size_t mism = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (std::memcmp(&r[i], &g[i], sizeof(float)) != 0) {
      ++mism;
    }
  }
  return mism;
}

int RunConfigs() {
  const CfgShape shapes[] = {
      {"long32k", 4096, 28672, 8},
      {"mid16k", 4096, 12288, 12},
      {"ship2k", 2048, 512, 20},
  };
  std::uint32_t state = 987654321U;
  auto rnd = [&] {
    state = state * 1664525U + 1013904223U;
    return (static_cast<float>(state >> 8) / 8388608.0F) - 1.0F;
  };
  for (const CfgShape& s : shapes) {
    const std::size_t qn = std::size_t(s.rows) * kHeadsN * kDim;
    const std::size_t kcn = std::size_t(s.pos0 + s.rows) * kKvHeads * kDim;
    std::vector<float> h_q(qn), h_sink(kHeadsN, 0.75F);
    std::vector<__half> h_k(kcn), h_v(kcn);
    for (auto& x : h_q) {
      x = rnd();
    }
    for (std::size_t i = 0; i < kcn; ++i) {
      h_k[i] = __float2half_rn(2.0F * rnd());
      h_v[i] = __float2half_rn(2.0F * rnd());
    }
    float* q = nullptr;
    float* sink = nullptr;
    __half *k = nullptr, *v = nullptr;
    float* oa = nullptr;
    float* ob = nullptr;
    float* oc = nullptr;
    CHECK(hipMalloc(&q, qn * sizeof(float)));
    CHECK(hipMalloc(&sink, kHeadsN * sizeof(float)));
    CHECK(hipMalloc(&k, kcn * sizeof(__half)));
    CHECK(hipMalloc(&v, kcn * sizeof(__half)));
    CHECK(hipMalloc(&oa, qn * sizeof(float)));
    CHECK(hipMalloc(&ob, qn * sizeof(float)));
    CHECK(hipMalloc(&oc, qn * sizeof(float)));
    CHECK(hipMemcpy(q, h_q.data(), qn * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(sink, h_sink.data(), kHeadsN * sizeof(float),
                    hipMemcpyHostToDevice));
    CHECK(hipMemcpy(k, h_k.data(), kcn * sizeof(__half),
                    hipMemcpyHostToDevice));
    CHECK(hipMemcpy(v, h_v.data(), kcn * sizeof(__half),
                    hipMemcpyHostToDevice));
    for (const std::uint32_t w : {513U, 0U}) {
      const char* wl = w == 0U ? "full" : "win513";
      const double t81 = TimeCfg<8, 1>(q, k, v, sink, oa, s.rows, s.pos0, w, 1);
      (void)t81;
      const double a81 = TimeCfg<8, 1>(q, k, v, sink, oa, s.rows, s.pos0, w,
                                       s.iters);
      const double a24 = TimeCfg<2, 4>(q, k, v, sink, ob, s.rows, s.pos0, w,
                                       s.iters);
      const std::size_t m24 = Mismatches(oa, ob, qn);
      const double a42 = TimeCfg<4, 2>(q, k, v, sink, ob, s.rows, s.pos0, w,
                                       s.iters);
      const std::size_t m42 = Mismatches(oa, ob, qn);
      const double a18 = TimeCfg<1, 8>(q, k, v, sink, ob, s.rows, s.pos0, w,
                                       s.iters);
      const std::size_t m18 = Mismatches(oa, ob, qn);
      const double a2s = TimeTwoSyncs(q, k, v, sink, ob, s.rows, s.pos0, w,
                                      s.iters);
      const std::size_t m2s = Mismatches(oa, ob, qn);
      const double a2s2 = TimeTwoSyncs(q, k, v, sink, oc, s.rows, s.pos0, w,
                                       s.iters);
      const std::size_t m2s2 = Mismatches(ob, oc, qn);
      std::printf(
          "%-8s %-7s h8q1 %8.3f ms | h2q4 %8.3f ms (%+6.1f%% m=%zu) | "
          "h4q2 %8.3f ms (%+6.1f%% m=%zu) | h1q8 %8.3f ms (%+6.1f%% m=%zu)\n",
          s.label, wl, a81, a24, 100.0 * (a24 / a81 - 1.0), m24, a42,
          100.0 * (a42 / a81 - 1.0), m42, a18, 100.0 * (a18 / a81 - 1.0), m18);
      std::printf("%-8s %-7s 2sync %8.3f ms (%+6.1f%% m=%zu self=%zu)\n",
                  s.label, wl, a2s, 100.0 * (a2s / a81 - 1.0), m2s, m2s2);
    }
    hipFree(q);
    hipFree(sink);
    hipFree(k);
    hipFree(v);
    hipFree(oa);
    hipFree(ob);
    hipFree(oc);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "configs") == 0) {
    return RunConfigs();
  }
  std::vector<float> h_q(kQn);
  std::vector<float> h_sink(kHeadsN, 0.75F);
  const std::size_t kcn =
      static_cast<std::size_t>(kPos0 + kRows) * kKvHeads * kDim;
  std::vector<__half> h_k(kcn), h_v(kcn);
  std::uint32_t state = 12345U;
  auto rnd = [&] {
    state = state * 1664525U + 1013904223U;
    return (static_cast<float>(state >> 8) / 8388608.0F) - 1.0F;
  };
  for (auto& x : h_q) x = rnd();
  for (std::size_t i = 0; i < kcn; ++i) {
    h_k[i] = __float2half_rn(2.0F * rnd());
    h_v[i] = __float2half_rn(2.0F * rnd());
  }
  float* q = nullptr;
  float* sink = nullptr;
  __half *k = nullptr, *v = nullptr;
  float* out[4] = {nullptr, nullptr, nullptr, nullptr};
  
  CHECK(hipMalloc(&q, kQn * sizeof(float)));
  CHECK(hipMalloc(&sink, kHeadsN * sizeof(float)));
  CHECK(hipMalloc(&k, kcn * sizeof(__half)));
  CHECK(hipMalloc(&v, kcn * sizeof(__half)));
  for (auto& o : out) CHECK(hipMalloc(&o, kQn * sizeof(float)));
  CHECK(hipMemcpy(q, h_q.data(), kQn * sizeof(float), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(sink, h_sink.data(), kHeadsN * sizeof(float),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(k, h_k.data(), kcn * sizeof(__half), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(v, h_v.data(), kcn * sizeof(__half), hipMemcpyHostToDevice));

  for (const char* label : {"windowed513", "full"}) {
    const std::uint32_t window = label[0] == 'w' ? 513U : 0U;
    const WmmaArgs as = MakeArgs(q, k, v, sink, out[0], window);
    const WmmaArgs ap = MakeArgs(q, k, v, sink, out[1], window);
    LaunchOne(0, as);
    LaunchOne(2, ap);
    CHECK(hipDeviceSynchronize());
    std::size_t mism_r = 0;
    {
      std::vector<float> ref(kQn), got(kQn);
      CHECK(hipMemcpy(ref.data(), out[0], kQn * sizeof(float),
                      hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(got.data(), out[1], kQn * sizeof(float),
                      hipMemcpyDeviceToHost));
      for (std::size_t i = 0; i < kQn; ++i) {
        if (std::memcmp(&ref[i], &got[i], sizeof(float)) != 0) ++mism_r;
      }
    }
    const double t0 = TimeIt(0, as, 20);
    const double t1 = TimeIt(1, ap, 20);
    const double t2 = TimeIt(2, ap, 20);
    const double t3 = TimeIt(3, as, 20);
    const double t4 = TimeIt(4, ap, 20);
    const double t5 = TimeIt(5, ap, 20);
    const double t6 = TimeIt(6, ap, 20);
    const double t7 = TimeIt(7, ap, 20);
    const double t8 = TimeIt(8, ap, 20);
    const double t9 = TimeIt(9, ap, 20);
    std::printf("%-12s ship      %8.3f ms\n", label, t0);
    std::printf("%-12s noQl      %8.3f ms  (%+.1f%%)\n", label, t1,
                100.0 * (t1 / t0 - 1.0));
    std::printf("%-12s rebuildQ  %8.3f ms  (%+.1f%%)  mismatches=%zu\n", label,
                t2, 100.0 * (t2 / t0 - 1.0), mism_r);
    std::printf("%-12s ship2b    %8.3f ms  (%+.1f%%)\n", label, t3,
                100.0 * (t3 / t0 - 1.0));
    std::printf("%-12s rebuild2b %8.3f ms  (%+.1f%%)\n", label, t4,
                100.0 * (t4 / t0 - 1.0));
    std::printf("%-12s buildNoLo %8.3f ms  (%+.1f%%)\n", label, t5,
                100.0 * (t5 / t0 - 1.0));
    std::printf("%-12s persistNoS %8.3f ms  (%+.1f%%)\n", label, t6,
                100.0 * (t6 / t0 - 1.0));
    std::printf("%-12s hiPersist  %8.3f ms  (%+.1f%%)\n", label, t7,
                100.0 * (t7 / t0 - 1.0));
    std::printf("%-12s rebuildHi  %8.3f ms  (%+.1f%%)\n", label, t8,
                100.0 * (t8 / t0 - 1.0));
    std::printf("%-12s splitChain %8.3f ms  (%+.1f%%)\n", label, t9,
                100.0 * (t9 / t0 - 1.0));
  }
  return 0;
}
