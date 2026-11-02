// C17b step 1: test the LDS-occupancy hypothesis for WmmaAttentionKernel
// <8, 2> on gfx1151 (heads 32, kv 4, head 128, B 2048, the measured pp
// shapes). The shipped kernel reserves s_lds + p_lds + pl_lds = 42 KB of its
// 54.5 KB block LDS although every slice is wave-private; that caps the
// block at one resident block per CU (64 KB). UnionRowKernel merges the
// three arrays into one 64 B row per query row (scores f32[16] in place,
// then p f16[16] and pl f16[16] overwriting them under __syncwarp, since
// only the owning wave reads and writes its rows), and drops the loop to
// two block barriers: K staging only needs to wait for every wave's QK
// reads, V staging only for every wave's PV reads; the score/probability
// handoffs become __syncwarp. Same arithmetic order everywhere -> outputs
// must be bitwise identical to the shipped <8, 2>. Also prints the register
// count, scratch spill size and occupancy API limits for every variant: the
// shipped kernel is register-capped at one block per CU, a two-block
// launch bound does not move the allocator, and a variant that reloads and
// re-splits the Q fragments per tile (kReloadQ) trades spills for VALU at
// the same occupancy. Result (EXPERIMENTS C17b): no bit-safe variant
// reaches the acceptance floor; attention stays as shipped.
//
// Build:
//   clang++ --offload-arch=gfx1151 --driver-mode=g++ -L/opt/rocm/lib \
//     --hip-link -O3 -std=c++20 -x hip -I . \
//     tools/bench/flint_attn_lds2_bench.hip -o /tmp/flint_attn_lds2_bench
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/flint/kernels/rocm/attention_wmma.hip.cpp"

namespace fr = gufo::models::flint::rocm;

namespace {

constexpr std::uint32_t kRows = 2048;
constexpr std::uint32_t kHeadsN = 32;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kDim = 128;
constexpr std::uint32_t kPos0 = 512;
constexpr std::size_t kQn =
    static_cast<std::size_t>(kRows) * kHeadsN * kDim;  // 8.4M floats

#define CHECK(cmd)                                                  \
  do {                                                              \
    const hipError_t err__ = (cmd);                                 \
    if (err__ != hipSuccess) {                                      \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",               \
                   hipGetErrorString(err__), __FILE__, __LINE__);   \
      std::exit(1);                                                 \
    }                                                               \
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

template<std::uint32_t kH, std::uint32_t kQ>
void LaunchCfg(const fr::WmmaArgs& a) {
  fr::WmmaAttentionKernel<kH, kQ>
      <<<dim3((kRows + kQ * 16 - 1) / (kQ * 16), kHeadsN / kH),
         32 * kH * kQ>>>(a);
}

using namespace fr;

// Wave-private softmax rows: score s f32[16] at offset 0 (64 B), then p
// f16[16] at 0 and pl f16[16] at 32 overwrite it after the softmax read.
__device__ __forceinline__ __half* RowP(unsigned char* base,
                                        std::uint32_t r) {
  return reinterpret_cast<__half*>(&base[r * 64U]);
}

// kMinBlocks=1 mirrors the shipped launch bound; kMinBlocks=2 forces the
// allocator under the two-resident-block register budget, trading more
// scratch for the free LDS headroom of the merged rows. kReloadQ drops the
// 128-register qh/ql cache entirely and re-splits the Q row from global
// (L2-hot) every key tile, to test whether removing the spilled operands
// from the allocator's live set beats paying the split cost.
template<int kMinBlocks, bool kReloadQ = false>
__global__ void __launch_bounds__(32 * 8 * 2, kMinBlocks == 0 ? 1
                                                              : kMinBlocks)
    UnionRowKernel(const WmmaArgs a) {
  constexpr std::uint32_t kD = 128;
  constexpr std::uint32_t kRowBlocks = 16;
  constexpr std::uint32_t kThreads = 32 * kRowBlocks;
  constexpr std::uint32_t kKStride = kD + 8;      // K rows, halves
  constexpr std::uint32_t kVtStride = kKeys + 8;  // V^T rows, halves
  constexpr std::uint32_t kQueries = 32;
  constexpr std::uint32_t kSegLanes = 2;
  constexpr std::uint32_t kSegKeys = kKeys / kSegLanes;

  __shared__ __align__(16) __half k_lds[kKeys * kKStride];
  __shared__ __align__(16) __half vt_lds[kD * kVtStride];
  __shared__ __align__(16) unsigned char sw_lds[kRowBlocks][16 * 64];
  __shared__ float row_scale[kRowBlocks][16];
  __shared__ float row_sum[kRowBlocks][16];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t rb = __builtin_amdgcn_readfirstlane(tid >> 5U);
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;

  const std::uint32_t query_start = blockIdx.x * kQueries;
  const std::uint32_t head_base = blockIdx.y * 8U;
  const std::uint32_t gqa = a.heads / a.kv_heads;
  const std::uint32_t kv_head = head_base / gqa;
  const std::uint32_t head = head_base + rb % 8U;
  const std::uint32_t row0 = query_start + (rb / 8U) * 16;  // this rb

  const std::uint32_t last_row = min(query_start + kQueries, a.rows) - 1;
  const std::uint32_t position_of_last =
      a.first_position + min(last_row, a.rows - 1);
  const std::uint32_t block_lo =
      KeyBounds(a, a.first_position + min(query_start, a.rows - 1)).lo &
      ~(kKeys - 1);
  const std::uint32_t block_hi = KeyBounds(a, position_of_last).hi;

  const std::uint32_t q_row = row0 + sub;
  const bool live_q = q_row < a.rows;
  const float* const q_row_ptr =
      a.q + (static_cast<std::size_t>(q_row) * a.heads + head) * kD;
  v16h qh[kReloadQ ? 1 : kSliceSteps];
  v16h ql[kReloadQ ? 1 : kSliceSteps];
  if constexpr (!kReloadQ) {
    const float* q = q_row_ptr;
    const bool live = live_q;
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
  unsigned char* const sw = &sw_lds[rb][0];

  const auto* k_cache = a.k_cache;
  const auto* v_cache = a.v_cache;
  const std::size_t kv_stride = a.kv_stride;

  const std::uint32_t chunk_key = tid & (kKeys - 1);
  const std::uint32_t chunk_d8 = ((tid >> 4) & (kD / 8 - 1)) * 8;
  const bool staging = kThreads == 256U ? true : tid < 256U;
  const auto* key_base = k_cache + kv_head * kD + chunk_d8;
  uint4 k_next;
  uint4 v_next;
  const auto fetch = [&](std::uint32_t key0) {
    const std::uint32_t position = key0 + chunk_key;
    k_next = make_uint4(0U, 0U, 0U, 0U);
    v_next = make_uint4(0U, 0U, 0U, 0U);
    if (staging && position < block_hi) {
      k_next = *reinterpret_cast<const uint4*>(
          key_base + static_cast<std::size_t>(position) * kv_stride);
      v_next = *reinterpret_cast<const uint4*>(
          v_cache + static_cast<std::size_t>(position) * kv_stride +
          static_cast<std::size_t>(kv_head) * kD + chunk_d8);
    }
  };
  const auto store_keys = [&] {
    if (!staging) {
      return;
    }
    *reinterpret_cast<uint4*>(&k_lds[chunk_key * kKStride + chunk_d8]) = k_next;
  };
  const auto store_values = [&] {
    if (!staging) {
      return;
    }
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

    {
      v8f s_acc = {};
      v8f s_corr = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
        v16h a_h;
        v16h a_l;
        if constexpr (kReloadQ) {
#pragma unroll
          for (std::uint32_t v = 0; v < 16; v += 4) {
            const float4 f =
                live_q ? *reinterpret_cast<const float4*>(q_row_ptr +
                                                          ks * 16 + v)
                       : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            const float x[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
            for (std::uint32_t i = 0; i < 4; ++i) {
              const __half h = __float2half_rn(x[i]);
              a_h[v + i] = static_cast<_Float16>(h);
              a_l[v + i] = static_cast<_Float16>(
                  __float2half_rn(x[i] - __half2float(h)));
            }
          }
        } else {
          a_h = qh[ks];
          a_l = ql[ks];
        }
        const v16h k_frag = LoadFrag(&k_lds[sub * kKStride + ks * 16]);
        s_acc = Wmma(a_h, k_frag, s_acc);
        s_corr = Wmma(a_l, k_frag, s_corr);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t r = 2 * i + half_id;
        reinterpret_cast<float*>(&sw[r * 64U])[sub] = s_acc[i] + s_corr[i];
      }
    }
    // One block barrier: every wave is done reading this tile's K, so the
    // next keys may be staged. It also orders the score writes for the
    // wave-private softmax read below.
    __syncthreads();
    if (more) {
      store_keys();
    }

    {
      const std::uint32_t position_row = sm_query;
      float vals[kSegKeys];
      float tile_max = -INFINITY;
      const float* const sp = reinterpret_cast<const float*>(&sw[sm_row * 64U]);
#pragma unroll
      for (std::uint32_t m = 0; m < kSegKeys; ++m) {
        const std::uint32_t col = sm_seg * kSegKeys + m;
        const std::uint32_t key = key0 + col;
        float s = 0.0F;
        s += sp[col];
        const bool valid =
            position_row < a.rows && key >= sm_keys.lo && key < sm_keys.hi;
        vals[m] = valid ? s * a.scale : -INFINITY;
        tile_max = fmaxf(tile_max, vals[m]);
      }
      tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 1, 32));
      const float next_max = fmaxf(running_max, tile_max);
      const float scale =
          next_max == -INFINITY ? 1.0F : expf(running_max - next_max);
      // The pair's second half may still be reading the score bytes this
      // lane is about to overwrite with p/pl.
      __syncwarp();
      float tile_sum = 0.0F;
      __half* const pp = RowP(sw, sm_row);
#pragma unroll
      for (std::uint32_t m = 0; m < kSegKeys; ++m) {
        const float w = vals[m] == -INFINITY ? 0.0F : expf(vals[m] - next_max);
        tile_sum += w;
        const __half ph = __float2half_rn(w);
        pp[sm_seg * kSegKeys + m] = static_cast<_Float16>(ph);
        pp[16 + sm_seg * kSegKeys + m] =
            static_cast<_Float16>(__float2half_rn(w - __half2float(ph)));
      }
      tile_sum += __shfl_xor(tile_sum, 1, 32);
      running_max = next_max;
      running_sum = running_sum * scale + tile_sum;
      if (sm_seg == 0) {
        row_scale[rb][sm_row] = scale;
      }
    }
    // Wave-private handoff: p/pl and row_scale were written by this wave.
    __syncwarp();

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
      const v16h p_frag = LoadFrag(RowP(sw, sub));
      const v16h pl_frag = LoadFrag(RowP(sw, sub) + 16);
#pragma unroll
      for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
        const v16h v_frag = LoadFrag(&vt_lds[(t * 16 + sub) * kVtStride]);
        o_acc[t] = Wmma(p_frag, v_frag, o_acc[t]);
        o_acc[t] = Wmma(pl_frag, v_frag, o_acc[t]);
      }
    } else {
#pragma unroll
      for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
        const __half* const v_row = &vt_lds[(t * 16U + sub) * kVtStride];
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          const std::uint32_t r = 2U * i + half_id;
          const __half* const pr = RowP(sw, r);
          float s = o_acc[t][i];
#pragma unroll
          for (std::uint32_t m = 0; m < kKeys; ++m) {
            const float v = static_cast<float>(v_row[m]);
            s += static_cast<float>(pr[m]) * v;
            s += static_cast<float>(pr[16 + m]) * v;
          }
          o_acc[t][i] = s;
        }
      }
    }
    // Every wave is done with this tile's values before the next replace
    // them; the same barrier publishes the staged keys for the next QK.
    __syncthreads();
    if (more) {
      store_values();
    }
  }

  if (sm_seg == 0) {
    row_sum[rb][sm_row] = running_sum + expf(a.sinks[head] - running_max);
  }
  __syncwarp();
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

void LaunchUnion(const fr::WmmaArgs& a, int variant) {
  switch (variant) {
    case 2:
      UnionRowKernel<2><<<dim3((kRows + 31) / 32, kHeadsN / 8), 512>>>(a);
      break;
    case 3:
      UnionRowKernel<1, true><<<dim3((kRows + 31) / 32, kHeadsN / 8), 512>>>(a);
      break;
    default:
      UnionRowKernel<1><<<dim3((kRows + 31) / 32, kHeadsN / 8), 512>>>(a);
      break;
  }
}

double TimeCfg(int cfg, const fr::WmmaArgs& a, int iters) {
  const auto launch = [&] {
    if (cfg == 0) {
      LaunchCfg<8, 2>(a);
    } else {
      LaunchUnion(a, cfg);
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

void PrintRes(const char* name, const void* fn) {
  hipFuncAttributes attr{};
  if (hipFuncGetAttributes(&attr, fn) != hipSuccess) {
    std::printf("%-8s attributes unavailable\n", name);
    return;
  }
  int blocks = 0;
  CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, 512, 0));
  std::printf(
      "%-8s regs/thread=%d spill=%zu B lds=%zu B occupancy=%d "
      "blocks/CU\n",
      name, attr.numRegs, attr.localSizeBytes,
      attr.sharedSizeBytes, blocks);
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
  float* out[4] = {};
  CHECK(hipMalloc(&q, kQn * sizeof(float)));
  CHECK(hipMalloc(&sink, kHeadsN * sizeof(float)));
  CHECK(hipMalloc(&k, kcn * sizeof(__half)));
  CHECK(hipMalloc(&v, kcn * sizeof(__half)));
  for (auto& o : out) CHECK(hipMalloc(&o, kQn * sizeof(float)));
  CHECK(hipMemcpy(q, h_q.data(), kQn * sizeof(float),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(sink, h_sink.data(), kHeadsN * sizeof(float),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(k, h_k.data(), kcn * sizeof(__half),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(v, h_v.data(), kcn * sizeof(__half),
                  hipMemcpyHostToDevice));

  PrintRes("shipped", (const void*)&fr::WmmaAttentionKernel<8, 2>);
  PrintRes("union1", (const void*)&UnionRowKernel<1>);
  PrintRes("union2", (const void*)&UnionRowKernel<2>);
  PrintRes("reloadq", (const void*)&UnionRowKernel<1, true>);

  const char* names[4] = {"shipped", "union1", "union2", "reloadq"};
  int rc = 0;
  for (const char* label : {"windowed513", "full"}) {
    const std::uint32_t window = label[0] == 'w' ? 513U : 0U;
    std::fprintf(stderr, "=== %s ===\n", label);
    LaunchCfg<8, 2>(MakeArgs(q, k, v, sink, out[0], window));
    for (int c = 1; c < 4; ++c) {
      LaunchUnion(MakeArgs(q, k, v, sink, out[c], window), c);
    }
    CHECK(hipDeviceSynchronize());
    std::vector<float> ref(kQn), got(kQn);
    CHECK(hipMemcpy(ref.data(), out[0], kQn * sizeof(float),
                    hipMemcpyDeviceToHost));
    for (int c = 0; c < 4; ++c) {
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
      std::printf("%-12s cfg %-6s %8.3f ms  mismatches=%zu\n", label,
                  names[c], ms, mismatch);
      if (mismatch != 0) rc = 1;
    }
  }
  std::printf("%s\n", rc == 0 ? "BITWISE OK" : "BITWISE MISMATCH");
  return rc;
}
