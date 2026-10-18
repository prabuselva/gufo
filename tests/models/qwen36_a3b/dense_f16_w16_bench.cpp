// Standalone microbenchmark for the native-F16 weight route of the shared
// core dense F16 WMMA GEMM (`gufo::core::hip::gemm::DenseF16Gemm` with
// `weights_f16`, the wide-batch path Qwen3.6 attention projections take on
// their binary16 weights). The 256-row tiles spill their accumulators on this
// route (the verbatim half fetch holds four uint4 per unit where the Q8_0
// dequant needs scale + two uint4), so the tile plans are compared against
// each other and against the same kernel on Q8_0 bytes, which is the route's
// speed ceiling. No model artifact is required.
//
// Weights are rotated across several buffers so the streamed working set
// exceeds the 32 MiB MALL cache; a single reused weight would look
// unrealistically fast. See docs/models/gemma4/EXPERIMENTS.md.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "src/core/hip/gemm/dense_f16_gemm.hpp"
#include "tests/models/qwen36_a3b/hip_test.hpp"

namespace core = gufo::core::hip::gemm;
namespace t = gufo::tests::qwen36_a3b;

namespace {

// Synthetic block_q8_0 matrix (half scale + 32 int8 codes per 32-block) with
// its exactly dequantized values, re-rounded to binary16: the same logical
// weight is uploaded once as Q8_0 bytes (dequant route) and once as F16
// (verbatim route), so both are compared to one double reference.
struct WeightMatrix {
  std::vector<std::uint8_t> bytes;
  std::vector<__half> halves;
  std::vector<double> values;
};

WeightMatrix MakeWeight(std::size_t rows, std::uint32_t cols,
                        std::uint32_t seed) {
  WeightMatrix out;
  const std::size_t nblocks = cols / 32;
  out.bytes.reserve(rows * nblocks * 34);
  out.halves.resize(rows * cols);
  out.values.resize(rows * cols);
  for (std::size_t idx = 0; idx < rows * nblocks; ++idx) {
    const float scale =
        0.5F *
        static_cast<float>(static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) -
                           32768) /
        32768.0F;
    const __half d = __float2half(scale);
    const auto* scale_bytes = reinterpret_cast<const std::uint8_t*>(&d);
    out.bytes.push_back(scale_bytes[0]);
    out.bytes.push_back(scale_bytes[1]);
    for (int i = 0; i < 32; ++i) {
      const std::int8_t code =
          static_cast<std::int8_t>(t::NextRandom(&seed) & 0xFFU);
      out.bytes.push_back(static_cast<std::uint8_t>(code));
      const double exact = static_cast<double>(__half2float(d)) * code;
      const std::size_t off = idx * 32 + static_cast<std::size_t>(i);
      out.halves[off] = __float2half(static_cast<float>(exact));
      out.values[off] = exact;
    }
  }
  return out;
}

// One projection shape the trunk runs on binary16 weights inside the
// wide-batch kF16 window (gated Q 2*4096 rows, O-proj 2048x4096, shared-
// expert down 2048x512). Production prefill chunks at 2048 tokens; the
// 8192-batch rows bound a hypothetical whole-prompt chunk.
struct Row {
  const char* name;
  std::uint32_t m;  // output rows
  std::uint32_t k;  // reduction / input cols
  std::uint32_t batch;
};

constexpr Row kRows[] = {
    {"attn_q", 8192, 2048, 8192},   {"attn_q", 8192, 2048, 2048},
    {"attn_out", 2048, 4096, 8192}, {"attn_out", 2048, 4096, 2048},
    {"shexp_dn", 2048, 512, 2048},
};

constexpr int kWeightCopies = 4;  // rotate to exceed the 32 MiB MALL cache
constexpr int kWarmup = 3;
constexpr int kReps = 15;

double Median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  if (n == 0) {
    return 0.0;
  }
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// The tile plans, all launched with weights_f16 = true. The kAuto 256x128
// 4x2-wave instantiation is the current production route (spills 160 B of
// register to scratch); kStaged128/kTile64 are the half-accumulator shapes
// meant to fit the VGPR cap.
constexpr core::DenseF16Plan kPlans[] = {
    core::DenseF16Plan::kAuto,
    core::DenseF16Plan::kRowGroups8,
    core::DenseF16Plan::kStagedRowGroups8,
    core::DenseF16Plan::kStaged256,
    core::DenseF16Plan::kTile128,
    core::DenseF16Plan::kStaged128,
    core::DenseF16Plan::kTile64,
};
constexpr int kNumPlans = 7;
constexpr const char* const kPlanNames[kNumPlans] = {
    "auto", "rg8", "staged", "staged256", "tile128", "staged128", "tile64"};

struct Result {
  double q8_us = 0.0;
  double us[kNumPlans] = {0.0};
  double err[kNumPlans] = {0.0};
  double q8_err = 0.0;
  bool any = false;
};

Result BenchRow(const Row& row) {
  Result r;
  const std::size_t wbytes =
      static_cast<std::size_t>(row.m) * (row.k / 32) * 34;
  const std::size_t whcount = static_cast<std::size_t>(row.m) * row.k;

  // Rotating weight copies (distinct data) so the streamed set exceeds MALL.
  std::vector<std::unique_ptr<t::HipBuffer<std::uint8_t>>> q8;
  std::vector<std::unique_ptr<t::HipBuffer<__half>>> wh;
  std::vector<WeightMatrix> host_w;
  for (int c = 0; c < kWeightCopies; ++c) {
    host_w.push_back(MakeWeight(row.m, row.k, 0x1234ABCDU + c * 0x9E3779B1U));
    q8.push_back(std::make_unique<t::HipBuffer<std::uint8_t>>(wbytes));
    t::CheckHip(hipMemcpy(q8[c]->get(), host_w[c].bytes.data(), wbytes,
                          hipMemcpyHostToDevice),
                "upload q8 weight");
    wh.push_back(std::make_unique<t::HipBuffer<__half>>(whcount));
    t::CheckHip(hipMemcpy(wh[c]->get(), host_w[c].halves.data(), wh[c]->bytes(),
                          hipMemcpyHostToDevice),
                "upload f16 weight");
  }

  // Binary16 activations, exactly as the production NarrowHalf output feeds
  // the kernel.
  const std::size_t xcount = static_cast<std::size_t>(row.batch) * row.k;
  const std::vector<float> x = t::MakeValues(xcount, 0x6E6E6E6EU, 1.0F);
  std::vector<__half> xh(xcount);
  for (std::size_t i = 0; i < xcount; ++i) {
    xh[i] = __float2half(x[i]);
  }
  std::vector<double> xd(xcount);
  for (std::size_t i = 0; i < xcount; ++i) {
    xd[i] = static_cast<double>(__half2float(xh[i]));
  }
  t::HipBuffer<__half> d_xh(xcount);
  t::CheckHip(
      hipMemcpy(d_xh.get(), xh.data(), d_xh.bytes(), hipMemcpyHostToDevice),
      "upload f16 activation");

  const std::size_t ocount = static_cast<std::size_t>(row.batch) * row.m;
  t::HipBuffer<float> d_out(ocount);

  hipEvent_t ev0, ev1;
  hipEventCreate(&ev0);
  hipEventCreate(&ev1);

  const auto time_launch = [&](const auto& launch) {
    for (int i = 0; i < kWarmup; ++i) {
      launch(i % kWeightCopies);
    }
    std::vector<float> times;
    for (int i = 0; i < kReps; ++i) {
      hipEventRecord(ev0);
      launch(i % kWeightCopies);
      hipEventRecord(ev1);
      hipEventSynchronize(ev1);
      float ms = 0.0F;
      hipEventElapsedTime(&ms, ev0, ev1);
      times.push_back(ms * 1000.0F);
    }
    return Median(times);
  };

  // Q8_0 dequant route at kAuto: the ceiling for the same kernel and tiles.
  r.q8_us = time_launch([&](int c) {
    (void)core::DenseF16Gemm(q8[c]->get(), d_xh.get(), d_out.get(), row.batch,
                             row.m, row.k, nullptr);
  });

  // Native-F16 route, one plan at a time.
  bool plan_ok[kNumPlans] = {false};
  for (int p = 0; p < kNumPlans; ++p) {
    plan_ok[p] =
        core::DenseF16Gemm(wh[0]->get(), d_xh.get(), d_out.get(), row.batch,
                           row.m, row.k, nullptr, kPlans[p], true);
    if (!plan_ok[p]) {
      continue;
    }
    r.any = true;
    r.us[p] = time_launch([&](int c) {
      (void)core::DenseF16Gemm(wh[c]->get(), d_xh.get(), d_out.get(), row.batch,
                               row.m, row.k, nullptr, kPlans[p], true);
    });
  }

  // --- correctness against a double reference on weight copy 0 ---
  const auto check = [&](bool ok, bool weights_f16, core::DenseF16Plan plan,
                         const void* w, double* err) {
    if (!ok) {
      *err = 0.0;
      return;
    }
    (void)core::DenseF16Gemm(w, d_xh.get(), d_out.get(), row.batch, row.m,
                             row.k, nullptr, plan, weights_f16);
    t::CheckHip(hipDeviceSynchronize(), "sync");
    const std::vector<float> out = t::Download(&d_out, ocount);
    // Sampled rows to bound host cost: the reduction is O(batch*m*k).
    std::vector<float> ref;
    std::vector<float> got;
    const std::size_t stride = std::max<std::size_t>(1, ocount / 4096);
    for (std::size_t i = 0; i < ocount; i += stride) {
      const std::size_t tok = i / row.m;
      const std::size_t r0 = i % row.m;
      double acc = 0.0;
      for (std::uint32_t j = 0; j < row.k; ++j) {
        acc += host_w[0].values[r0 * row.k + j] * xd[tok * row.k + j];
      }
      ref.push_back(static_cast<float>(acc));
      got.push_back(out[i]);
    }
    *err = t::WorstRelativeToScale(ref, got);
  };
  check(true, false, core::DenseF16Plan::kAuto, q8[0]->get(), &r.q8_err);
  for (int p = 0; p < kNumPlans; ++p) {
    check(plan_ok[p], true, kPlans[p], wh[0]->get(), &r.err[p]);
  }

  hipEventDestroy(ev0);
  hipEventDestroy(ev1);
  return r;
}

}  // namespace

int main() {
  std::printf(
      "core dense F16 GEMM native-F16 weights: tile plans vs the Q8_0 "
      "dequant ceiling (4 weight copies, median of %d reps)\n\n",
      kReps);
  std::printf("%-9s %6s %9s", "shape", "batch", "q8_us");
  for (const char* name : kPlanNames) {
    std::printf("%11s_us", name);
  }
  std::printf("%10s", "best/q8");
  for (int p = 1; p < kNumPlans; ++p) {
    std::printf("%11s/auto", kPlanNames[p]);
  }
  std::printf("%11s\n", "worst_err");
  int failures = 0;
  for (const Row& row : kRows) {
    const Result r = BenchRow(row);
    if (!r.any) {
      std::printf("%-9s %6u %9.1f  (route unsupported)\n", row.name, row.batch,
                  r.q8_us);
      continue;
    }
    double best = 0.0;
    double worst_err = r.q8_err;
    for (int p = 0; p < kNumPlans; ++p) {
      if (r.us[p] > 0.0 && (best == 0.0 || r.us[p] < best)) {
        best = r.us[p];
      }
      worst_err = std::max(worst_err, r.err[p]);
    }
    std::printf("%-9s %6u %9.1f", row.name, row.batch, r.q8_us);
    for (int p = 0; p < kNumPlans; ++p) {
      if (r.us[p] == 0.0) {
        std::printf("%11s", "n/a");
      } else {
        std::printf("%11.1f", r.us[p]);
      }
    }
    std::printf("%9.2fx", best / r.q8_us);
    for (int p = 1; p < kNumPlans; ++p) {
      if (r.us[p] == 0.0 || r.us[0] == 0.0) {
        std::printf("%11s", "n/a");
      } else {
        std::printf("%10.2fx", r.us[p] / r.us[0]);
      }
    }
    std::printf(" %10.3e\n", worst_err);
    if (worst_err > 2e-2) {
      ++failures;
    }
  }
  if (failures > 0) {
    std::printf("FAIL: %d row(s) above 2e-2 scale-relative error\n", failures);
    return 1;
  }
  return 0;
}
