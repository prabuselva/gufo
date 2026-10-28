// C7s probe for the Flint grouped routed GEMM stage (RoutedHalfDownKernel,
// src/models/flint/kernels/rocm/moe/moe_prefill.hip.cpp) at the production
// chunk shapes: gate/up geglu (m=1024 via [gate|up], K=2048, half slot rows
// out) and down (m=H=2048, K=expert_ff=512, half slot rows out). The
// production TU is included so the probe runs the shipped kernels and their
// internals unchanged; each configuration also reports the modelled DRAM
// traffic (weights + activations + output) as GB/s.
//
//   PATH=/opt/rocm/lib/llvm/bin:$PATH clang++ --offload-arch=gfx1151 \
//     --driver-mode=g++ -L/opt/rocm/lib --hip-link -O3 -std=c++20 \
//     -I . -I src/models/flint/kernels/rocm/mmq -x hip \
//     tools/bench/flint_moe_down_probe.hip -x none \
//     build/release/src/models/flint/libgufo_flint_moe.a \
//     build/release/src/models/flint/libgufo_flint_mmq.a \
//     -lhipblas -o /tmp/flint_moe_down_probe
//
// Usage: /tmp/flint_moe_down_probe [reps]
#include "src/models/flint/kernels/rocm/moe/moe_prefill.hip.cpp"

#include "src/models/flint/kernels/rocm/moe/routed_compact.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace flint = gufo::models::flint::rocm;

#define CHECK(call)                                                     \
  do {                                                                  \
    const hipError_t err__ = (call);                                    \
    if (err__ != hipSuccess) {                                          \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                   \
                   hipGetErrorString(err__), __FILE__, __LINE__);       \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

namespace {

struct Xorshift {
  std::uint32_t s = 0x2545F491U;
  std::uint32_t Next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  float Uniform() {  // [-1, 1)
    return static_cast<float>(Next() >> 8) / 8388608.0f - 1.0f;
  }
};

constexpr int kChunks[] = {4096, 2048};
constexpr int kExperts = 256;
constexpr int kUsed = 6;
constexpr int kGateM = 1024;   // 2 * expert_ff, [gate | up] rows
constexpr int kDownM = 2048;   // hidden size
constexpr int kK = 2048;       // gate/up reduction
constexpr int kKDown = 512;    // down reduction (expert_ff)

std::size_t Q8_0Bytes(int rows, int k) {
  return static_cast<std::size_t>(rows) * (k / 32) * 34;
}

void FillQ8(std::vector<std::uint8_t>& w, int rows, int k, Xorshift& rng,
            bool sane_scale = false) {
  w.resize(Q8_0Bytes(rows, k));
  const int n_blocks = rows * (k / 32);
  for (int b = 0; b < n_blocks; ++b) {
    // sane_scale keeps the binary16 scale in a normal mid-range window;
    // the plain fill draws arbitrary bits, i.e. ~1/8 denormal scales.
    const unsigned short raw = static_cast<unsigned short>(rng.Next() & 0x3fff);
    const unsigned short d_bits = sane_scale
        ? static_cast<unsigned short>(0x2c00U + (raw & 0x300U))
        : raw;
    std::memcpy(&w[static_cast<std::size_t>(b) * 34], &d_bits, 2);
    for (int q = 0; q < 32; ++q) w[b * 34 + 2 + q] = rng.Next() & 0xff;
  }
}

void FillIds(std::vector<std::int32_t>& ids, int n_tokens, Xorshift& rng) {
  ids.resize(static_cast<std::size_t>(n_tokens) * kUsed);
  for (int t = 0; t < n_tokens; ++t) {
    const int base = static_cast<int>(rng.Next() % kExperts);
    for (int i = 0; i < kUsed; ++i) {
      ids[static_cast<std::size_t>(t) * kUsed + i] = (base + i * 7) % kExperts;
    }
  }
}

float Median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <typename L>
float TimeLaunch(L&& launch, int reps) {
  for (int i = 0; i < 3; ++i) launch();
  CHECK(hipDeviceSynchronize());
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  std::vector<float> ts;
  for (int i = 0; i < reps; ++i) {
    CHECK(hipEventRecord(a));
    launch();
    CHECK(hipEventRecord(b));
    CHECK(hipEventSynchronize(b));
    float ms = 0.0F;
    CHECK(hipEventElapsedTime(&ms, a, b));
    ts.push_back(ms);
  }
  CHECK(hipEventDestroy(a));
  CHECK(hipEventDestroy(b));
  return Median(ts);
}

void Report(const char* name, float ms, double bytes) {
  std::printf("  %-16s %8.3f ms  %7.1f GB/s modelled\n", name, ms,
              bytes / (ms * 1e-3) / 1e9);
}

// Bit-identical copy of the kQ8_0 geglu pass of RoutedHalfDownKernel with
// two A/B knobs: kMinB (__launch_bounds__ min blocks/CU, 1 = production
// default) and kPFN (weight prefetch ring depth: kPFN LDS row buffers plus
// register buffers, fetched kPFN-1 stages ahead; 1 = production depth).
template <int kMinB, int kPFN>
__global__ void __launch_bounds__(256, kMinB == 0 ? 1 : kMinB)
    ProbeGateup(const std::uint8_t* __restrict__ w,
                const __half* __restrict__ x,
                const std::int32_t* __restrict__ tiles,
                const std::int32_t* __restrict__ pad_bounds,
                const std::int32_t* __restrict__ rows_in,
                const std::int32_t* __restrict__ rows_out,
                __half* __restrict__ out_half, std::uint32_t m,
                std::uint32_t k, const std::uint8_t* __restrict__ w_up) {
  using flint::kQ80Words;
  constexpr flint::ExpertFormat kF = flint::ExpertFormat::kQ8_0;
  constexpr int kStride = flint::kDownStride<kF>;
  constexpr int kStageBytes = 2 * flint::kDownBlockBytes<kF>;
  constexpr int kTokTiles = 8;
  constexpr int kActChunks = 128 * flint::kDownStageK / 8;
  constexpr int kActPer = (kActChunks + 255) / 256;
  constexpr int kActBytes = 128 * flint::kDownActStride * 2;
  constexpr int kOutBytes = 16 * flint::kOutStride * 4;
  constexpr int kWords = 128 * kQ80Words;
  constexpr int kWordsPer = (kWords + 255) / 256;
  constexpr int kNB = kPFN;
  __shared__ __attribute__((aligned(16))) std::uint8_t
      s_rows[kNB][128 * kStride];
  __shared__ __attribute__((aligned(16))) std::uint8_t
      s_act_bytes[kActBytes > kOutBytes ? kActBytes : kOutBytes];
  auto* s_act = reinterpret_cast<__half*>(s_act_bytes);
  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * 128;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const unsigned row_block = blockIdx.x;
  if (t_local >= bucket_rows) {
    return;
  }
  const int live_tiles = std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const auto source_row = [&](int t) { return rows_in[bucket_begin + t]; };
  const auto dest_row = [&](int t) { return rows_out[bucket_begin + t]; };
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % 32;
  const int wave = tid / 32;
  const int sub = lane & 15;
  const int half = lane >> 4;
  const std::uint32_t half_m = m / 2;
  const std::size_t row_bytes = std::size_t{k} / 32 * flint::kDownBlockBytes<kF>;
  const std::uint8_t* w_expert =
      w + std::size_t{static_cast<std::uint32_t>(expert)} * half_m * row_bytes;
  const std::uint8_t* wu_expert =
      w_up + std::size_t{static_cast<std::uint32_t>(expert)} * half_m *
             row_bytes;
  const auto row_live = [&](int l) {
    return row_block * 64 + static_cast<std::uint32_t>(l % 64) < half_m;
  };
  const auto row_src = [&](int r_local) {
    const bool up = r_local >= 64;
    const std::uint32_t local =
        row_block * 64 + static_cast<std::uint32_t>(r_local % 64);
    return (up ? wu_expert : w_expert) +
           std::size_t{row_live(r_local) ? local : half_m - 1} * row_bytes;
  };
  const auto word_of = [&](int i) {
    const int c = tid + i * 256;
    return c < kWords ? c : kWords - 1;
  };
  const int stages = static_cast<int>(k / flint::kDownStageK);
  std::uint32_t fbuf[kPFN][kWordsPer];
  const auto fetch_w = [&](int stage, std::uint32_t (&d)[kWordsPer]) {
#pragma unroll
    for (int i = 0; i < kWordsPer; ++i) {
      const int c = word_of(i);
      d[i] = *reinterpret_cast<const std::uint32_t*>(
          row_src(c / kQ80Words) + stage * kStageBytes + (c % kQ80Words) * 4);
    }
  };
  const auto commit_w = [&](const std::uint32_t (&d)[kWordsPer], int buf) {
#pragma unroll
    for (int i = 0; i < kWordsPer; ++i) {
      const int c = tid + i * 256;
      if (c < kWords) {
        *reinterpret_cast<std::uint32_t*>(s_rows[buf] +
                                          (c / kQ80Words) * kStride +
                                          (c % kQ80Words) * 4) = d[i];
      }
    }
  };
  std::int32_t a_src[kActPer];
#pragma unroll
  for (int i = 0; i < kActPer; ++i) {
    const int chunk = tid + i * 256;
    const int t = chunk / 8;
    a_src[i] = -1;
    if (chunk < kActChunks && t_local + t < bucket_rows) {
      const std::int32_t src = source_row(t_local + t);
      if (src >= 0) {
        a_src[i] =
            src * static_cast<std::int32_t>(k) + (chunk % 8) * 8;
      }
    }
  }
  uint4 a_data[kActPer];
  const auto fetch_act = [&](int stage) {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      a_data[i] = a_src[i] >= 0
                      ? *reinterpret_cast<const uint4*>(x + a_src[i] +
                                                        stage * flint::kDownStageK)
                      : make_uint4(0U, 0U, 0U, 0U);
    }
  };
  const auto commit_act = [&] {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      const int chunk = tid + i * 256;
      if (chunk < kActChunks) {
        *reinterpret_cast<uint4*>(s_act + (chunk / 8) * flint::kDownActStride +
                                  (chunk % 8) * 8) = a_data[i];
      }
    }
  };
  flint::v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    acc[j] = flint::v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  }
  const bool live = row_live(wave * 16 + sub);
#pragma unroll
  for (int i = 0; i < kPFN; ++i) {
    fetch_w(i, fbuf[i]);
  }
  fetch_act(0);
  // Unrolled by the ring depth so every fbuf/s_rows slot index is constant.
  // rounds stay uniform across the block, so the guarded tail cannot
  // diverge around the barriers.
  const int rounds = ((stages + kPFN - 1) / kPFN) * kPFN;
  for (int s0 = 0; s0 < rounds; s0 += kPFN) {
#pragma unroll
    for (int q = 0; q < kPFN; ++q) {
      const int stage = s0 + q;
      const bool run = stage < stages;
      // Stage the weights fetched kPFN iterations earlier and the activation
      // tile fetched after the previous syncthreads.
      if (run) {
        commit_w(fbuf[q], q);
        commit_act();
      }
      __syncthreads();
      if (stage + kPFN < stages) {
        fetch_w(stage + kPFN, fbuf[q]);
      }
      if (stage + 1 < stages) {
        fetch_act(stage + 1);
      }
      if (run) {
        const std::uint8_t* row = s_rows[q] + (wave * 16 + sub) * kStride;
      std::uint32_t words[kQ80Words];
      const auto* r4 = reinterpret_cast<const uint4*>(row);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 v = r4[i];
        words[4 * i] = v.x;
        words[4 * i + 1] = v.y;
        words[4 * i + 2] = v.z;
        words[4 * i + 3] = v.w;
      }
      words[16] = reinterpret_cast<const std::uint32_t*>(row)[16];
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        flint::v16h a_lo;
        flint::v16h a_hi;
        flint::DecodeQ80(words, s, live, &a_lo, &a_hi);
#pragma unroll
        for (int j = 0; j < kTokTiles; ++j) {
          if (j < live_tiles) {
            const __half* b =
                s_act + (j * 16 + sub) * flint::kDownActStride + s * 32;
            flint::v16h b_lo;
            flint::v16h b_hi;
            __builtin_memcpy(&b_lo, b, 32);
            __builtin_memcpy(&b_hi, b + 16, 32);
            acc[j] = flint::Wmma(a_lo, b_lo, acc[j]);
            acc[j] = flint::Wmma(a_hi, b_hi, acc[j]);
          }
        }
      }
      }
      __syncthreads();
    }
  }
  // geglu epilogue, as production: up waves publish FP32 rows through LDS.
  constexpr int kJT = kTokTiles / 2;
  auto* s_up = reinterpret_cast<float*>(s_act_bytes);
  bool up_done = false;
  for (int jc = 0; jc < 2; ++jc) {
    if (wave >= 4) {
#pragma unroll
      for (int j = 0; j < kJT; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          s_up[(((wave - 4) * kJT + j) * 8 + i) * 32 + lane] =
              acc[jc * kJT + j][i];
        }
      }
    }
    __syncthreads();
    if (wave >= 4) {
      up_done = true;
    }
    if (!up_done) {
#pragma unroll
      for (int jj = 0; jj < kJT; ++jj) {
        const int j = jc * kJT + jj;
        const int t = t_local + j * 16 + sub;
        if (t >= bucket_rows) {
          continue;
        }
        const std::int32_t dst = dest_row(t);
        if (dst < 0) {
          continue;
        }
        const std::size_t base =
            std::size_t{static_cast<std::uint32_t>(dst)} * half_m;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const std::uint32_t r =
              row_block * 64 +
              static_cast<std::uint32_t>(wave * 16 + 2 * i + half);
          if (r < half_m) {
            const float up = s_up[((wave * kJT + jj) * 8 + i) * 32 + lane];
            out_half[base + r] = flint::SaturatedHalf(
                acc[j][i] / (1.0F + expf(-acc[j][i])) * up);
          }
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace

bool g_sane = false;

int main(int argc, char** argv) {
  int reps = 10;
  if (argc > 1) reps = std::atoi(argv[1]);
  if (argc > 2) g_sane = std::atoi(argv[2]) != 0;
  CHECK(hipSetDevice(0));

  for (const int n_tokens : kChunks) {
    const std::size_t slots = static_cast<std::size_t>(n_tokens) * kUsed;
    Xorshift rng;
    std::vector<std::int32_t> ids;
    FillIds(ids, n_tokens, rng);
    std::vector<std::uint32_t> counts(kExperts, 0);
    for (const std::int32_t e : ids) ++counts[e];
    std::vector<std::uint8_t> w_gate, w_up, w_down;
    FillQ8(w_gate, kExperts * (kGateM / 2), kK, rng, g_sane);
    FillQ8(w_up, kExperts * (kGateM / 2), kK, rng, g_sane);
    FillQ8(w_down, kExperts * kDownM, kKDown, rng, g_sane);
    std::vector<float> x(static_cast<std::size_t>(n_tokens) * kK);
    for (auto& v : x) v = rng.Uniform() * 0.25F;
    std::vector<float> x_slots(slots * kKDown);
    for (auto& v : x_slots) v = rng.Uniform() * 0.25F;

    // Fused [gate | up] copy for the single-tensor control.
    std::vector<std::uint8_t> w_fused(w_gate.size() * 2);
    const std::size_t half_bytes = Q8_0Bytes(kGateM / 2, kK);  // per expert
    for (int e = 0; e < kExperts; ++e) {
      const std::size_t we = static_cast<std::size_t>(e);
      std::memcpy(w_fused.data() + we * 2 * half_bytes,
                  w_gate.data() + we * half_bytes, half_bytes);
      std::memcpy(w_fused.data() + we * 2 * half_bytes + half_bytes,
                  w_up.data() + we * half_bytes, half_bytes);
    }

    void *d_gate, *d_up, *d_down, *d_fused, *d_xh, *d_xsh;
    __half *d_gout, *d_out;
    std::int32_t *d_ids, *d_pad, *d_cur, *d_rows_tok, *d_rows_slot, *d_tiles;
    std::uint32_t* d_counts;
    CHECK(hipMalloc(&d_gate, w_gate.size()));
    CHECK(hipMalloc(&d_up, w_up.size()));
    CHECK(hipMalloc(&d_down, w_down.size()));
    CHECK(hipMalloc(&d_fused, w_fused.size()));
    CHECK(hipMemcpy(d_gate, w_gate.data(), w_gate.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_up, w_up.data(), w_up.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_down, w_down.data(), w_down.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_fused, w_fused.data(), w_fused.size(),
                    hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_xh, static_cast<std::size_t>(n_tokens) * kK * 2));
    float* d_x = nullptr;
    CHECK(hipMalloc(&d_x, x.size() * sizeof(float)));
    CHECK(hipMemcpy(d_x, x.data(), x.size() * sizeof(float),
                    hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_xsh, slots * kKDown * 2));
    CHECK(hipMalloc(&d_gout, slots * (kGateM / 2) * 2));
    CHECK(hipMalloc(&d_out, slots * kDownM * 2));
    CHECK(hipMalloc(&d_ids, ids.size() * sizeof(std::int32_t)));
    CHECK(hipMemcpy(d_ids, ids.data(), ids.size() * sizeof(std::int32_t),
                    hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_counts, counts.size() * sizeof(std::uint32_t)));
    CHECK(hipMemcpy(d_counts, counts.data(), counts.size() *
                        sizeof(std::uint32_t), hipMemcpyHostToDevice));
    const std::size_t compact_rows = flint::RoutedCompactRows(slots, kExperts);
    CHECK(hipMalloc(&d_pad, (kExperts + 1) * sizeof(std::int32_t)));
    CHECK(hipMalloc(&d_cur, kExperts * sizeof(std::int32_t)));
    CHECK(hipMalloc(&d_rows_tok, compact_rows * sizeof(std::int32_t)));
    CHECK(hipMalloc(&d_rows_slot, compact_rows * sizeof(std::int32_t)));
    CHECK(hipMemset(d_rows_tok, 0, compact_rows * sizeof(std::int32_t)));
    CHECK(hipMemset(d_rows_slot, 0, compact_rows * sizeof(std::int32_t)));
    const std::uint32_t rows_tile = 128U;
    const std::uint32_t n_tiles = static_cast<std::uint32_t>(
        flint::RoutedTileCapacity(static_cast<std::uint32_t>(slots),
                                  static_cast<std::uint32_t>(kExperts),
                                  rows_tile));
    CHECK(hipMalloc(&d_tiles, n_tiles * sizeof(std::int32_t)));
    flint::BuildRoutedTiles(d_counts, kExperts, rows_tile, n_tiles, d_tiles,
                            rows_tile, n_tiles, d_tiles, 0);
    flint::RoutedCompact(d_ids, d_counts, d_pad, d_cur, d_rows_tok,
                         d_rows_slot, n_tokens, kUsed, kExperts, 0);
    flint::NarrowActivations(d_x, static_cast<__half *>(d_xh), static_cast<std::size_t>(n_tokens) *
                           kK, 0);
    // NarrowActivations takes device sources in the grouped path; stage the
    // slot activations through the device too.
    float* d_x_slots = nullptr;
    CHECK(hipMalloc(&d_x_slots, x_slots.size() * sizeof(float)));
    CHECK(hipMemcpy(d_x_slots, x_slots.data(),
                    x_slots.size() * sizeof(float), hipMemcpyHostToDevice));
    flint::NarrowActivations(d_x_slots, static_cast<__half *>(d_xsh), slots * kKDown, 0);

    std::printf("chunk %d tokens, %zu slots, %u tiles\n", n_tokens, slots,
                n_tiles);

    const dim3 geglu_grid((kGateM / 2 + 63) / 64, n_tiles);
    const dim3 down_grid((kDownM + 127) / 128, n_tiles);
    const auto* gb = static_cast<const std::uint8_t*>(d_gate);
    const auto* ub = static_cast<const std::uint8_t*>(d_up);
    const auto* db = static_cast<const std::uint8_t*>(d_down);
    const auto* fb = static_cast<const std::uint8_t*>(d_fused);
    const auto* xh = static_cast<const __half*>(d_xh);
    const auto* xsh = static_cast<const __half*>(d_xsh);
    auto* go = static_cast<__half*>(d_gout);
    auto* o = static_cast<__half*>(d_out);

    // gate/up geglu, separate tensors (production).
    const double w_act = 0.0;
    const double gw_bytes = 2.0 * static_cast<double>(w_gate.size()) +
                            n_tokens * kK * 2.0 + slots * (kGateM / 2) * 2.0 +
                            w_act;
    const float t_pro = TimeLaunch(
        [&] {
          flint::RoutedHalfDownKernel<flint::ExpertFormat::kQ8_0, 128, true>
              <<<geglu_grid, 256>>>(gb, xh, d_tiles, d_pad, d_rows_tok,
                                    d_rows_slot, nullptr, go, kGateM, kK, 0, 0,
                                    ub);
        },
        reps);
    // Bit stability of the production configuration across launches.
    CHECK(hipMemsetAsync(go, 0, slots * (kGateM / 2) * 2));
    flint::RoutedHalfDownKernel<flint::ExpertFormat::kQ8_0, 128, true>
        <<<geglu_grid, 256>>>(gb, xh, d_tiles, d_pad, d_rows_tok, d_rows_slot,
                              nullptr, go, kGateM, kK, 0, 0, ub);
    std::vector<__half> g1(slots * (kGateM / 2));
    CHECK(hipMemcpy(g1.data(), go, g1.size() * 2, hipMemcpyDeviceToHost));
    flint::RoutedHalfDownKernel<flint::ExpertFormat::kQ8_0, 128, true>
        <<<geglu_grid, 256>>>(gb, xh, d_tiles, d_pad, d_rows_tok, d_rows_slot,
                              nullptr, go, kGateM, kK, 0, 0, ub);
    std::vector<__half> g2(g1.size());
    CHECK(hipMemcpy(g2.data(), go, g2.size() * 2, hipMemcpyDeviceToHost));
    std::size_t mism = 0;
    for (std::size_t i = 0; i < g1.size(); ++i) {
      mism += std::memcmp(&g1[i], &g2[i], 2) != 0 ? 1U : 0U;
    }
    std::printf("  pro-vs-pro bit mismatches: %zu\n", mism);
    Report("gateup w+up", t_pro, gw_bytes);

    // A/B the knobbed copies against the bit-exact production reference g1.
    {
      __half* d_vout = nullptr;
      CHECK(hipMalloc(&d_vout, g1.size() * 2));
      const auto time_check = [&](const char* name, auto&& fn) {
        const float ms = TimeLaunch(fn, reps);
        std::vector<__half> gv(g1.size());
        CHECK(hipMemcpy(gv.data(), d_vout, g1.size() * 2,
                        hipMemcpyDeviceToHost));
        std::size_t bad = 0;
        for (std::size_t i = 0; i < gv.size(); ++i) {
          bad += std::memcmp(&gv[i], &g1[i], 2) != 0 ? 1U : 0U;
        }
        std::printf("  %-16s %8.3f ms  %7.1f GB/s  bit-diff %zu\n", name, ms,
                    gw_bytes / (ms * 1e-3) / 1e9, bad);
      };
#define PROBE_VARIANT(NAME, MINB, PFN)                                        \
      do {                                                                     \
        CHECK(hipMemsetAsync(d_vout, 0, g1.size() * 2));                       \
        ProbeGateup<MINB, PFN><<<geglu_grid, 256>>>(                            \
            gb, xh, d_tiles, d_pad, d_rows_tok, d_rows_slot, d_vout,            \
            kGateM, kK, ub);                                                    \
        if (hipDeviceSynchronize() != hipSuccess) {                             \
          (void)hipGetLastError();                                              \
          std::printf("  %-16s launch failed (LDS)\n", NAME);                   \
        } else {                                                                \
          time_check(NAME, [&] {                                               \
            ProbeGateup<MINB, PFN><<<geglu_grid, 256>>>(                         \
                gb, xh, d_tiles, d_pad, d_rows_tok, d_rows_slot, d_vout,        \
                kGateM, kK, ub);                                                \
          });                                                                   \
        }                                                                       \
      } while (0)
      PROBE_VARIANT("v min1 pf1", 1, 1);
      PROBE_VARIANT("v min1 pf2", 1, 2);
      PROBE_VARIANT("v min1 pf3", 1, 3);
      PROBE_VARIANT("v min1 pf4", 1, 4);
#undef PROBE_VARIANT
      CHECK(hipFree(d_vout));
    }

    // gate/up geglu, fused single tensor (control: two-pointer row_src cost).
    const float t_fused = TimeLaunch(
        [&] {
          flint::RoutedHalfDownKernel<flint::ExpertFormat::kQ8_0, 128, true>
              <<<geglu_grid, 256>>>(fb, xh, d_tiles, d_pad, d_rows_tok,
                                    d_rows_slot, nullptr, go, kGateM, kK);
        },
        reps);
    Report("gateup fused", t_fused, gw_bytes);

    // down, half slot rows out (production).
    const double dw_bytes = static_cast<double>(w_down.size()) +
                            slots * kKDown * 2.0 + slots * kDownM * 2.0;
    const float t_down = TimeLaunch(
        [&] {
          flint::RoutedHalfDownKernel<flint::ExpertFormat::kQ8_0, 128, false,
                                      false, true>
              <<<down_grid, 256>>>(db, xsh, d_tiles, d_pad, d_rows_slot,
                                   d_rows_slot, nullptr, o, kDownM, kKDown);
        },
        reps);
    Report("down half", t_down, dw_bytes);
    const double d2_bytes = static_cast<double>(w_down.size()) +
                            slots * kKDown * 2.0 + slots * kDownM * 4.0;
    std::printf("      (if down output were f32 modelled, GB/s shown below)\n");

    // Control: down with binary32 token-major output via geglu=false,
    // kHalfOut=false writes `out` float at [dst][m] (larger traffic).
    float* d_out_f = nullptr;
    CHECK(hipMalloc(&d_out_f, slots * kDownM * sizeof(float)));
    const float t_down_f = TimeLaunch(
        [&] {
          flint::RoutedHalfDownKernel<flint::ExpertFormat::kQ8_0, 128, false,
                                      false, false>
              <<<down_grid, 256>>>(db, xsh, d_tiles, d_pad, d_rows_slot,
                                   d_rows_slot, d_out_f, nullptr, kDownM,
                                   kKDown);
        },
        reps);
    Report("down ctrl f32", t_down_f, d2_bytes);

    CHECK(hipFree(d_out_f));
    CHECK(hipFree(d_gate));
    CHECK(hipFree(d_up));
    CHECK(hipFree(d_down));
    CHECK(hipFree(d_fused));
    CHECK(hipFree(d_xh));
    CHECK(hipFree(d_xsh));
    CHECK(hipFree(d_x_slots));
    CHECK(hipFree(d_gout));
    CHECK(hipFree(d_out));
    CHECK(hipFree(d_ids));
    CHECK(hipFree(d_counts));
    CHECK(hipFree(d_pad));
    CHECK(hipFree(d_cur));
    CHECK(hipFree(d_rows_tok));
    CHECK(hipFree(d_rows_slot));
    CHECK(hipFree(d_tiles));
  }
  return 0;
}
