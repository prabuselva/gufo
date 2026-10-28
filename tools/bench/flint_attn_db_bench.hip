// WmmaAttentionKernelDB: double-buffered K/V staging with one __syncthreads
// per 16-key tile (the shipped kernel runs three). Every other LDS array
// (s/p/pl/row_scale/row_sum) is wave-private because each row block is one
// wave, so only K and V staging impose block barriers; writing the next
// tile into the opposite buffer removes the write-after-read barriers and
// the prologue keeps one barrier before the first tile. Outputs must be
// bitwise identical to the shipped <8, 1> (cfg 3 in the base sweep): the
// arithmetic is copied unchanged, only the LDS buffering and barrier
// placement move. Occupancy note: the second K/V buffer raises LDS from
// ~31.7 KiB to ~42.2 KiB per block (2 blocks -> 1 block per CU at
// kHeads=8); the sweep measures whether the barrier saving wins anyway.
//
// Build:
//   clang++ --offload-arch=gfx1151 --driver-mode=g++ -L/opt/rocm/lib \
//     --hip-link -O3 -std=c++20 -x hip -I . \
//     tools/bench/flint_attn_db_bench.hip -o /tmp/flint_attn_db_bench
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/flint/kernels/rocm/attention_wmma.hip.cpp"

namespace gufo::models::flint::rocm {
namespace {

template<std::uint32_t kHeads, std::uint32_t kQueryBlocks,
         std::uint32_t kBufs>
__global__ void __launch_bounds__(32 * kHeads * kQueryBlocks)
    WmmaAttentionKernelDB(const WmmaArgs a) {
  constexpr std::uint32_t kD = 128;
  constexpr std::uint32_t kRowBlocks = kHeads * kQueryBlocks;
  constexpr std::uint32_t kKStride = kD + 8;      // K rows, halves
  constexpr std::uint32_t kVtStride = kKeys + 8;  // V^T rows, halves
  constexpr std::uint32_t kQueries = kQueryBlocks * 16;
  constexpr std::uint32_t kSegLanes = 2;
  constexpr std::uint32_t kSegKeys = kKeys / kSegLanes;

  __shared__ __align__(16) __half kbuf[kBufs][kKeys * kKStride];
  __shared__ __align__(16) __half vbuf[kBufs][kD * kVtStride];
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
  const auto store_keys = [&](std::uint32_t par) {
    *reinterpret_cast<uint4*>(&kbuf[par][chunk_key * kKStride + chunk_d8]) =
        k_next;
  };
  const auto store_values = [&](std::uint32_t par) {
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
    auto* rows = reinterpret_cast<std::uint32_t*>(vbuf[par]);
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
  std::uint32_t par = 0;
  for (std::uint32_t key0 = block_lo; key0 < block_hi; key0 += kKeys) {
    const std::uint32_t npar = par + 1U == kBufs ? 0U : par + 1U;
    const bool more = key0 + kKeys < block_hi;
    if (more) {
      fetch(key0 + kKeys);
    }

    {
      v8f s_acc = {};
      v8f s_corr = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
        const v16h k_frag =
            LoadFrag(&kbuf[par][sub * kKStride + ks * 16]);
        s_acc = Wmma(qh[ks], k_frag, s_acc);
        s_corr = Wmma(ql[ks], k_frag, s_corr);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[rb][2 * i + half_id][sub] = s_acc[i] + s_corr[i];
      }
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
    if constexpr (kBufs == 2) {
      // Orders s_lds/row_scale producers (cross-lane within a wave needs
      // no wait, but kbuf[par] must be safe) and keeps every wave out of
      // the softmax until the score matrix is block-complete.
      __syncthreads();
    }

    float scale2[8];
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      scale2[i] = row_scale[rb][2 * i + half_id];
    }
    const bool full_tile =
        key0 + kKeys <= full_hi &&
        static_cast<std::int64_t>(key0) >= (full_lo > 0 ? full_lo : 0);
    bool unchanged = true;
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      unchanged = unchanged && scale2[i] == 1.0F;
    }
    if (more) {
      store_keys(npar);
      store_values(npar);
    }
    // The stage of tile t+1 precedes this barrier, so every wave's
    // score/PV of t+1 (past the barrier) sees it; with kBufs == 2 the
    // WAR against tile t-1's reads is covered by the softmax barrier
    // above (stage runs after it), with kBufs == 3 by the previous
    // iteration's barrier.
    __syncthreads();
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
        const v16h v_frag = LoadFrag(&vbuf[par][(t * 16 + sub) * kVtStride]);
        o_acc[t] = Wmma(p_frag, v_frag, o_acc[t]);
        o_acc[t] = Wmma(pl_frag, v_frag, o_acc[t]);
      }
    } else {
#pragma unroll
      for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
        const __half* const v_row = &vbuf[par][(t * 16U + sub) * kVtStride];
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
    par = npar;
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

namespace {

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
                   hipGetErrorString(err__), __FILE__, __LINE__);    \
      std::exit(1);                                                  \
    }                                                                \
  } while (0)

fr::WmmaArgs MakeArgs(const float* q, const __half* k, const __half* v,
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

template<std::uint32_t kH, std::uint32_t kQ, std::uint32_t kBufs>
void LaunchCfg(const fr::WmmaArgs& a) {
  const dim3 grid((kRows + kQ * 16 - 1) / (kQ * 16), kHeadsN / kH);
  if constexpr (kBufs == 0) {
    fr::WmmaAttentionKernel<kH, kQ><<<grid, 32 * kH * kQ>>>(a);
  } else {
    fr::WmmaAttentionKernelDB<kH, kQ, kBufs><<<grid, 32 * kH * kQ>>>(a);
  }
}

double TimeCfg(int cfg, const fr::WmmaArgs& a, int iters) {
  const auto launch = [&] {
    switch (cfg) {
      case 0: LaunchCfg<8, 1, 0>(a); break;
      case 1: LaunchCfg<8, 1, 2>(a); break;
      case 2: LaunchCfg<8, 1, 3>(a); break;
      case 3: LaunchCfg<2, 4, 0>(a); break;
      case 4: LaunchCfg<2, 4, 2>(a); break;
      default: LaunchCfg<2, 4, 3>(a); break;
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

}  // namespace

int main() {
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
  constexpr int kCfg = 6;
  float* out[kCfg] = {};
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

  const char* names[kCfg] = {"ship<8,1>",  "db2<8,1>", "db3<8,1>",
                             "ship<2,4>",  "db2<2,4>", "db3<2,4>"};
  int rc = 0;
  for (const char* label : {"windowed513", "full"}) {
    const std::uint32_t window = label[0] == 'w' ? 513U : 0U;
    std::fprintf(stderr, "=== %s ===\n", label);
    for (int c = 0; c < kCfg; ++c) {
      const fr::WmmaArgs a = MakeArgs(q, k, v, sink, out[c], window);
      switch (c) {
        case 0: LaunchCfg<8, 1, 0>(a); break;
        case 1: LaunchCfg<8, 1, 2>(a); break;
        case 2: LaunchCfg<8, 1, 3>(a); break;
        case 3: LaunchCfg<2, 4, 0>(a); break;
        case 4: LaunchCfg<2, 4, 2>(a); break;
        default: LaunchCfg<2, 4, 3>(a); break;
      }
    }
    CHECK(hipDeviceSynchronize());
    std::vector<float> ref(kQn), got(kQn);
    CHECK(hipMemcpy(ref.data(), out[0], kQn * sizeof(float),
                    hipMemcpyDeviceToHost));
    for (int c = 0; c < kCfg; ++c) {
      std::size_t mismatch = 0;
      if (c != 0) {
        CHECK(hipMemcpy(got.data(), out[c], kQn * sizeof(float),
                        hipMemcpyDeviceToHost));
        for (std::size_t i = 0; i < kQn; ++i) {
          if (std::memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
            ++mismatch;
          }
        }
      }
      const fr::WmmaArgs a = MakeArgs(q, k, v, sink, out[c], window);
      const double ms = TimeCfg(c, a, 20);
      std::printf("%-12s cfg %-10s %8.3f ms  mismatches=%zu\n", label, names[c],
                  ms, mismatch);
      if (mismatch != 0) rc = 1;
    }
  }
  std::printf("%s\n", rc == 0 ? "BITWISE OK" : "BITWISE MISMATCH");
  return rc;
}
