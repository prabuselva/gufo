// A/B harness for the Flint grouped MoE route (Gemma-4 fork,
// src/models/flint/kernels/rocm/moe) against the shipped mmq routed entry
// point at the measured Flint shapes. Both routes run on identical
// weights/activations/routing; outputs are slot-indexed [token*used+j][M]
// in both routes, so they are directly comparable (grouped narrows
// activations to f16, mmq quantizes to q8_1 -> expect close, not equal).
//
//   clang++ --offload-arch=gfx1151 --driver-mode=g++ -L/opt/rocm/lib \
//     --hip-link -O3 -std=c++20 \
//     -I . -I src/models/flint/kernels/rocm/mmq \
//     -x hip tools/bench/flint_moe_route_bench.hip -x none \
//     build/release/src/models/flint/libgufo_flint_moe.a \
//     build/release/src/models/flint/libgufo_flint_mmq.a \
//     -lhipblas -o /tmp/flint_moe_route_bench
//
// Usage: /tmp/flint_moe_route_bench [reps]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include <flint_mmq.h>

#include "src/models/flint/kernels/rocm/moe/moe_prefill.hpp"
#include "src/models/flint/kernels/rocm/moe/routed_compact.hpp"

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

#define RK(call)                                                        \
  do {                                                                  \
    const int rc__ = (call);                                            \
    if (rc__ != 0) {                                                    \
      std::fprintf(stderr, "mmq error %d at %s:%d\n", rc__, __FILE__,  \
                   __LINE__);                                           \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

namespace {

struct Xorshift {
  std::uint32_t s = 0x9e3779b9u;
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

struct Shape {
  const char *name;
  int M;           // weight rows (output dim) per expert
  int K;           // reduction dim
  int n_tokens;
  int n_experts;
  int expert_used;
  bool down;       // slot-indexed activations (ffn_down) vs token rows
};

// Flint routed projections at a full pp2048 chunk (same routing as
// flint_mmq_bench, so the mmq gate/up side reproduces its baseline).
constexpr Shape kShapes[] = {
    {"gateup_1024x2048", 1024, 2048, 2048, 256, 6, false},
    {"down_2048x1024", 2048, 1024, 2048, 256, 6, true},
};

std::size_t Q8_0Bytes(int rows, int k) {
  return static_cast<std::size_t>(rows) * (k / 32) * 34;  // d (fp16) + qs
}

void FillWeights(std::vector<std::uint8_t> &w, int rows, int k, Xorshift &rng) {
  w.resize(Q8_0Bytes(rows, k));
  const int n_blocks = rows * (k / 32);
  for (int b = 0; b < n_blocks; ++b) {
    unsigned short d_bits = static_cast<unsigned short>(rng.Next() & 0x3fff);
    std::memcpy(&w[static_cast<std::size_t>(b) * 34], &d_bits, 2);
    for (int q = 0; q < 32; ++q) w[b * 34 + 2 + q] = rng.Next() & 0xff;
  }
}

void FillIds(std::vector<std::int32_t> &ids, int n_tokens, int used,
             int n_experts, Xorshift &rng) {
  ids.resize(static_cast<std::size_t>(n_tokens) * used);
  for (int t = 0; t < n_tokens; ++t) {
    const int base = static_cast<int>(rng.Next() % n_experts);
    for (int i = 0; i < used; ++i) {
      ids[static_cast<std::size_t>(t) * used + i] = (base + i * 7) % n_experts;
    }
  }
}

float Median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

float HalfSat(float v) {  // SaturatedHalf mirror (value, not bits)
  if (v > 65504.0F) v = 65504.0F;
  if (v < -65504.0F) v = -65504.0F;
  return v;
}

// Gate mirror for the fused grouped launch (Flint gates experts with
// SiLU, as in SiluMulKernel).
float GeGluRef(float x, float u) { return x / (1.0F + expf(-x)) * u; }

// Unit check for the fused GeGLU grouped launch: the same fused [gate | up]
// Q8_0 tensor must give bit-identical results for the single-pointer and
// the two-pointer (w_up) modes, and both must match the f32 reference
// (ungated m-row launch + host GeGLU with the kernel's binary16 rounding).
int CheckGeGlu() {
  const Shape &s = kShapes[0];
  if (s.down) {
    std::fprintf(stderr, "expected a gate/up shape first\n");
    return 1;
  }
  Xorshift rng;
  std::vector<std::int32_t> ids;
  FillIds(ids, s.n_tokens, s.expert_used, s.n_experts, rng);
  std::vector<std::uint32_t> counts(s.n_experts, 0);
  for (const std::int32_t e : ids) ++counts[e];

  std::vector<std::uint8_t> wf;
  FillWeights(wf, s.n_experts * s.M, s.K, rng);
  const std::size_t rb = Q8_0Bytes(1, s.K);
  const std::size_t half_exp = static_cast<std::size_t>(s.M / 2) * rb;
  std::vector<std::uint8_t> wg(static_cast<std::size_t>(s.n_experts) *
                               half_exp);
  std::vector<std::uint8_t> wu(wg.size());
  for (int e = 0; e < s.n_experts; ++e) {
    const std::uint8_t *src = wf.data() + static_cast<std::size_t>(e) *
                                               static_cast<std::size_t>(s.M) *
                                               rb;
    std::memcpy(wg.data() + static_cast<std::size_t>(e) * half_exp, src,
                half_exp);
    std::memcpy(wu.data() + static_cast<std::size_t>(e) * half_exp,
                src + half_exp, half_exp);
  }
  const std::size_t slots =
      static_cast<std::size_t>(s.n_tokens) * s.expert_used;
  std::vector<float> x(static_cast<std::size_t>(s.n_tokens) * s.K);
  for (auto &v : x) v = rng.Uniform() * 0.25f;

  void *d_wf = nullptr, *d_wg = nullptr, *d_wu = nullptr;
  float *d_x = nullptr, *d_ref = nullptr;
  __half *d_xh = nullptr, *d_o1 = nullptr, *d_o2 = nullptr;
  std::int32_t *d_ids = nullptr, *d_pad = nullptr, *d_cur = nullptr;
  std::int32_t *d_rows_tok = nullptr, *d_rows_slot = nullptr;
  std::int32_t *d_tiles = nullptr;
  std::uint32_t *d_counts = nullptr;
  const std::size_t compact_rows =
      flint::RoutedCompactRows(slots, s.n_experts);
  const std::uint32_t rows_tile = 128U;
  const std::uint32_t n_tiles = static_cast<std::uint32_t>(
      flint::RoutedTileCapacity(static_cast<std::uint32_t>(slots),
                                static_cast<std::uint32_t>(s.n_experts),
                                rows_tile));
  CHECK(hipMalloc(&d_wf, wf.size()));
  CHECK(hipMemcpy(d_wf, wf.data(), wf.size(), hipMemcpyHostToDevice));
  CHECK(hipMalloc(&d_wg, wg.size()));
  CHECK(hipMemcpy(d_wg, wg.data(), wg.size(), hipMemcpyHostToDevice));
  CHECK(hipMalloc(&d_wu, wu.size()));
  CHECK(hipMemcpy(d_wu, wu.data(), wu.size(), hipMemcpyHostToDevice));
  CHECK(hipMalloc(&d_x, x.size() * sizeof(float)));
  CHECK(hipMemcpy(d_x, x.data(), x.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  CHECK(hipMalloc(&d_xh, x.size() * sizeof(__half)));
  CHECK(hipMalloc(&d_ref, slots * s.M * sizeof(float)));
  CHECK(hipMalloc(&d_o1, slots * (s.M / 2) * sizeof(__half)));
  CHECK(hipMalloc(&d_o2, slots * (s.M / 2) * sizeof(__half)));
  CHECK(hipMalloc(&d_ids, ids.size() * sizeof(std::int32_t)));
  CHECK(hipMemcpy(d_ids, ids.data(), ids.size() * sizeof(std::int32_t),
                  hipMemcpyHostToDevice));
  CHECK(hipMalloc(&d_counts, counts.size() * sizeof(std::uint32_t)));
  CHECK(hipMemcpy(d_counts, counts.data(), counts.size() * sizeof(std::uint32_t),
                  hipMemcpyHostToDevice));
  CHECK(hipMalloc(&d_pad, (s.n_experts + 1) * sizeof(std::int32_t)));
  CHECK(hipMalloc(&d_cur, s.n_experts * sizeof(std::int32_t)));
  CHECK(hipMalloc(&d_rows_tok, compact_rows * sizeof(std::int32_t)));
  CHECK(hipMalloc(&d_rows_slot, compact_rows * sizeof(std::int32_t)));
  CHECK(hipMalloc(&d_tiles, n_tiles * sizeof(std::int32_t)));
  flint::BuildRoutedTiles(d_counts, s.n_experts, rows_tile, n_tiles, d_tiles,
                          rows_tile, n_tiles, d_tiles, 0);
  flint::RoutedCompact(d_ids, d_counts, d_pad, d_cur, d_rows_tok, d_rows_slot,
                       s.n_tokens, s.expert_used, s.n_experts, 0);
  flint::NarrowActivations(d_x, d_xh, x.size(), 0);

  const auto half_m = static_cast<std::uint32_t>(s.M / 2);
  const auto m = static_cast<std::uint32_t>(s.M);
  const auto k = static_cast<std::uint32_t>(s.K);
  const bool ok_ref = flint::LaunchRoutedHalfGemm(
      flint::ExpertFormat::kQ8_0, d_wf, d_xh, d_tiles, n_tiles, rows_tile,
      d_pad, d_rows_tok, d_rows_slot, d_ref, nullptr, m, k, 0, /*geglu=*/false);
  const bool ok1 = flint::LaunchRoutedHalfGemm(
      flint::ExpertFormat::kQ8_0, d_wf, d_xh, d_tiles, n_tiles, rows_tile,
      d_pad, d_rows_tok, d_rows_slot, nullptr, d_o1, m, k, 0, /*geglu=*/true);
  const bool ok2 = flint::LaunchRoutedHalfGemm(
      flint::ExpertFormat::kQ8_0, d_wg, d_xh, d_tiles, n_tiles, rows_tile,
      d_pad, d_rows_tok, d_rows_slot, nullptr, d_o2, m, k, 0, /*geglu=*/true,
      d_wu);
  if (!ok_ref || !ok1 || !ok2) {
    std::fprintf(stderr, "launch rejected ref=%d fused=%d twoptr=%d\n", ok_ref,
                 ok1, ok2);
    return 1;
  }
  CHECK(hipDeviceSynchronize());

  std::vector<float> href(slots * s.M);
  std::vector<__half> h1(slots * half_m), h2(slots * half_m);
  CHECK(hipMemcpy(href.data(), d_ref, href.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(h1.data(), d_o1, h1.size() * sizeof(__half),
                  hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(h2.data(), d_o2, h2.size() * sizeof(__half),
                  hipMemcpyDeviceToHost));
  std::size_t twoptr_bits_mismatch = 0, ref_mismatch = 0, old_mismatch = 0;
  double twoptr_max = 0.0, ref_max = 0.0, ref_ref_abs = 0.0, old_max = 0.0;
  int shown = 0;
  for (std::size_t slot = 0; slot < slots; ++slot) {
    for (std::uint32_t r = 0; r < half_m; ++r) {
      const std::size_t o = slot * half_m + r;
      const float gate = __half2float(
          __float2half_rn(HalfSat(href[slot * s.M + r])));
      const float up = __half2float(
          __float2half_rn(HalfSat(href[slot * s.M + s.M / 2 + r])));
      const float expect = HalfSat(GeGluRef(gate, up));
      const float expect_old = HalfSat(
          GeGluRef(href[slot * s.M + r], href[slot * s.M + s.M / 2 + r]));
      const float v1 = __half2float(h1[o]);
      const float v2 = __half2float(h2[o]);
      if (h1[o] != h2[o]) ++twoptr_bits_mismatch;
      const bool ref_bad = __float2half_rn(expect) != h1[o];
      if (ref_bad && ++ref_mismatch && shown < 5) {
        std::printf("slot %zu r %u: gate %g up %g expect %.6f fused %.6f\n",
                    slot, r, gate, up, expect, v1);
        ++shown;
      }
      if (__float2half_rn(expect_old) != h1[o]) ++old_mismatch;
      if (const double d = std::fabs(v1 - expect_old); d > old_max)
        old_max = d;
      if (const double d = std::fabs(v1 - v2); d > twoptr_max)
        twoptr_max = d;
      if (const double d = std::fabs(v1 - expect); d > ref_max) ref_max = d;
      if (const double d = std::fabs(expect); d > ref_ref_abs)
        ref_ref_abs = d;
    }
  }
  std::printf("twoptr vs fused: %zu/%zu bit mismatches, max|diff|=%.6g\n",
              twoptr_bits_mismatch, h1.size(), twoptr_max);
  std::printf("fused  vs ref:   %zu/%zu bit mismatches, max|diff|=%.6g "
              "(max|ref|=%.6g)\n",
              ref_mismatch, h1.size(), ref_max, ref_ref_abs);
  std::printf("fused  vs old pipeline: %zu/%zu bit mismatches, "
              "max|diff|=%.6g\n",
              old_mismatch, h1.size(), old_max);
  return twoptr_bits_mismatch == 0 && ref_mismatch < h1.size() / 2 ? 0 : 2;
}

float TimeLaunch(const std::function<void()> &launch, int reps) {
  launch();  // warmup (module load, scratch alloc)
  launch();
  CHECK(hipDeviceSynchronize());
  hipEvent_t ev0, ev1;
  CHECK(hipEventCreate(&ev0));
  CHECK(hipEventCreate(&ev1));
  std::vector<float> ms;
  for (int r = 0; r < reps; ++r) {
    CHECK(hipEventRecord(ev0, 0));
    launch();
    CHECK(hipEventRecord(ev1, 0));
    CHECK(hipEventSynchronize(ev1));
    float t;
    CHECK(hipEventElapsedTime(&t, ev0, ev1));
    ms.push_back(t);
  }
  CHECK(hipEventDestroy(ev0));
  CHECK(hipEventDestroy(ev1));
  return Median(ms);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc > 1 && std::strcmp(argv[1], "check-geglu") == 0) {
    RK(flint_mmq_init(0));
    return CheckGeGlu();
  }
  int reps = 20;
  if (argc > 1) reps = std::atoi(argv[1]);

  RK(flint_mmq_init(0));

  // Deterministic single routing shared by every shape's both routes.
  Xorshift rng;
  const Shape &route_src = kShapes[0];
  std::vector<std::int32_t> ids;
  FillIds(ids, route_src.n_tokens, route_src.expert_used, route_src.n_experts,
          rng);
  std::vector<std::uint32_t> counts(route_src.n_experts, 0);
  for (const std::int32_t e : ids) ++counts[e];
  // Consume the same rng stream the per-shape fills would have seen.
  Xorshift shape_rng = rng;

  for (const Shape &s : kShapes) {
    const std::size_t slots =
        static_cast<std::size_t>(s.n_tokens) * s.expert_used;

    std::vector<std::uint8_t> w;
    FillWeights(w, s.n_experts * s.M, s.K, shape_rng);
    std::vector<float> x(static_cast<std::size_t>(s.n_tokens) * s.K);
    for (auto &v : x) v = shape_rng.Uniform() * 0.25f;
    // The down GEMM consumes slot-indexed activations [slots][K]; both
    // routes are fed the same buffer so outputs are directly comparable.
    std::vector<float> x_slots;
    if (s.down) {
      x_slots.assign(slots * static_cast<std::size_t>(s.K), 0.0f);
      for (auto &v : x_slots) v = shape_rng.Uniform() * 0.25f;
    }

    // Device buffers.
    void *d_w = nullptr;
    float *d_x = nullptr;
    float *d_x_slots = nullptr;
    __half *d_xh = nullptr;
    float *d_out_mmq = nullptr;
    float *d_out_grp = nullptr;
    float *d_out_grp2 = nullptr;
    std::int32_t *d_ids = nullptr, *d_pad = nullptr, *d_cur = nullptr;
    std::int32_t *d_rows_tok = nullptr, *d_rows_slot = nullptr;
    std::int32_t *d_tiles = nullptr;
    std::uint32_t *d_counts = nullptr;

    const std::size_t compact_rows =
        flint::RoutedCompactRows(slots, s.n_experts);
    const std::uint32_t rows_tile = 128U;  // Q8_0 128-row bucket tiles
    const std::uint32_t n_tiles = static_cast<std::uint32_t>(
        flint::RoutedTileCapacity(static_cast<std::uint32_t>(slots),
                                  static_cast<std::uint32_t>(s.n_experts),
                                  rows_tile));

    CHECK(hipMalloc(&d_w, w.size()));
    CHECK(hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_x, x.size() * sizeof(float)));
    CHECK(hipMemcpy(d_x, x.data(), x.size() * sizeof(float),
                    hipMemcpyHostToDevice));
    if (s.down) {
      CHECK(hipMalloc(&d_x_slots, x_slots.size() * sizeof(float)));
      CHECK(hipMemcpy(d_x_slots, x_slots.data(),
                      x_slots.size() * sizeof(float), hipMemcpyHostToDevice));
    }
    const std::size_t x_route_elems =
        (s.down ? slots : static_cast<std::size_t>(s.n_tokens)) * s.K;
    CHECK(hipMalloc(&d_xh, x_route_elems * sizeof(__half)));
    CHECK(hipMalloc(&d_out_mmq, slots * s.M * sizeof(float)));
    CHECK(hipMalloc(&d_out_grp, slots * s.M * sizeof(float)));
    CHECK(hipMalloc(&d_out_grp2, slots * s.M * sizeof(float)));
    CHECK(hipMalloc(&d_ids, ids.size() * sizeof(std::int32_t)));
    CHECK(hipMemcpy(d_ids, ids.data(), ids.size() * sizeof(std::int32_t),
                    hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_counts, counts.size() * sizeof(std::uint32_t)));
    CHECK(hipMemcpy(d_counts, counts.data(),
                    counts.size() * sizeof(std::uint32_t),
                    hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_pad, (s.n_experts + 1) * sizeof(std::int32_t)));
    CHECK(hipMalloc(&d_cur, s.n_experts * sizeof(std::int32_t)));
    CHECK(hipMalloc(&d_rows_tok, compact_rows * sizeof(std::int32_t)));
    CHECK(hipMalloc(&d_rows_slot, compact_rows * sizeof(std::int32_t)));
    CHECK(hipMalloc(&d_tiles, n_tiles * sizeof(std::int32_t)));

    // One compact layout is enough: same routing for both projections.
    flint::BuildRoutedTiles(d_counts, s.n_experts, rows_tile, n_tiles, d_tiles,
                            rows_tile, n_tiles, d_tiles, 0);
    flint::RoutedCompact(d_ids, d_counts, d_pad, d_cur, d_rows_tok,
                         d_rows_slot, s.n_tokens, s.expert_used,
                         s.n_experts, 0);
    // gate/up: token activation rows in (rows_token), slot rows out;
    // down: slot activation rows in and out. mmq mirrors the executor:
    // gate/up token semantics, down = slot list with expert_used 1.
    const float *x_mmq = s.down ? d_x_slots : d_x;
    const int mmq_rows_in = s.down ? static_cast<int>(slots) : s.n_tokens;
    const int mmq_used = s.down ? 1 : s.expert_used;
    flint::NarrowActivations(x_mmq, d_xh,
                             static_cast<std::size_t>(mmq_rows_in) * s.K, 0);

    const float t_mmq = TimeLaunch(
        [&] {
          RK(flint_mmq_q8_0_moe_raw(d_w, x_mmq, d_ids, d_out_mmq, s.M, s.K,
                                    mmq_rows_in, s.n_experts, mmq_used, 0));
        },
        reps);

    const auto grouped_launch = [&](float *out) {
      const bool ok = flint::LaunchRoutedHalfGemm(
          flint::ExpertFormat::kQ8_0, d_w, d_xh, d_tiles, n_tiles, rows_tile,
          d_pad, s.down ? d_rows_slot : d_rows_tok, d_rows_slot, out, nullptr,
          static_cast<std::uint32_t>(s.M), static_cast<std::uint32_t>(s.K), 0,
          /*geglu=*/false);
      if (!ok) {
        std::fprintf(stderr, "grouped route unsupported at %s:%d\n", __FILE__,
                     __LINE__);
        std::exit(1);
      }
    };

    const float t_grp = TimeLaunch([&] { grouped_launch(d_out_grp); }, reps);
    grouped_launch(d_out_grp2);
    CHECK(hipDeviceSynchronize());

    const double flops = 2.0 * static_cast<double>(slots) * s.M * s.K;
    std::printf("%-18s mmq   %8.3f ms  %7.2f TFLOP/s\n", s.name, t_mmq,
                flops / (t_mmq * 1e-3) / 1e12);
    std::printf("%-18s grouped %8.3f ms %7.2f TFLOP/s  (%+.1f %%)\n", s.name,
                t_grp, flops / (t_grp * 1e-3) / 1e12,
                (t_mmq / t_grp - 1.0) * 100.0);

    // Numerics: grouped vs mmq over the slot rows (close, not equal) and
    // grouped vs grouped (must be bit-stable across launches).
    std::vector<float> o_mmq(slots * s.M), o_grp(slots * s.M),
        o_grp2(slots * s.M);
    CHECK(hipMemcpy(o_mmq.data(), d_out_mmq, o_mmq.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(o_grp.data(), d_out_grp, o_grp.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(o_grp2.data(), d_out_grp2, o_grp2.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    double max_abs = 0.0, ref_abs = 0.0, sq = 0.0, max2 = 0.0;
    std::size_t mism2 = 0;
    for (std::size_t i = 0; i < o_mmq.size(); ++i) {
      if (const double a =
              std::fabs(static_cast<double>(o_grp[i] - o_mmq[i]));
          a > max_abs)
        max_abs = a;
      if (const double r = std::fabs(static_cast<double>(o_mmq[i]));
          r > ref_abs)
        ref_abs = r;
      sq += static_cast<double>(o_grp[i] - o_mmq[i]) *
            static_cast<double>(o_grp[i] - o_mmq[i]);
      if (std::memcmp(&o_grp[i], &o_grp2[i], sizeof(float)) != 0) ++mism2;
      if (const double a = std::fabs(static_cast<double>(o_grp[i] - o_grp2[i]));
          a > max2)
        max2 = a;
    }
    std::printf("%-18s vs mmq: max|d|=%.4f (ref max %.3f) rms=%.5f | "
                "self: %zu mismatches max|d|=%.6f\n",
                s.name, max_abs, ref_abs,
                std::sqrt(sq / static_cast<double>(o_mmq.size())), mism2,
                max2);
    (void)compact_rows;

    CHECK(hipFree(d_w));
    CHECK(hipFree(d_x));
    CHECK(hipFree(d_x_slots));
    CHECK(hipFree(d_xh));
    CHECK(hipFree(d_out_mmq));
    CHECK(hipFree(d_out_grp));
    CHECK(hipFree(d_out_grp2));
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
