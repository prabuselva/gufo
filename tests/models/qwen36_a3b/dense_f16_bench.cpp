// Standalone prefill-GEMM microbenchmark for Qwen3.6-35B-A3B dense
// projections: the production int8 path (`qfn_mmq_q8_0_dense`, FP32
// activations requantized to Q8_1 in-kernel) against the binary16 WMMA path
// (`qwen36_a3b::rocm::DenseF16Gemm`, Q8_0 weights dequantized to F16 in LDS,
// F16 activations). Both consume the same block_q8_0 weight bytes.
//
// For each dense shape the trunk runs at pp2048 it reports the median kernel
// time of each path and the worst scale-relative error of each against a
// double-precision reference (dequantized weights x FP32 activations). This
// confirms the F16 route's winning window (batch >= 96, rows >= 2048,
// cols <= 4096) on this trunk's real projections. No model artifact is
// required.
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

#include "src/models/qwen36_a3b/kernels/rocm/dense_f16_gemm.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "tests/models/qwen36_a3b/hip_test.hpp"

namespace fn = gufo::models::qwen36_a3b::rocm;
namespace t = gufo::tests::qwen36_a3b;

namespace {

// Synthetic block_q8_0 matrix (half scale + 32 int8 codes per 32-block), with
// the exactly dequantized rows returned for the double reference. Mirrors
// rocm_gemm_test.cpp MakeQ8.
struct Q8Matrix {
  std::vector<std::uint8_t> bytes;
  std::vector<float> values;
};

Q8Matrix MakeQ8(std::size_t rows, std::uint32_t cols, std::uint32_t seed,
                float scale_unit = 0.5F) {
  Q8Matrix out;
  const std::size_t nblocks = cols / 32;
  out.bytes.reserve(rows * nblocks * 34);
  out.values.resize(rows * cols);
  for (std::size_t idx = 0; idx < rows * nblocks; ++idx) {
    const float scale =
        scale_unit *
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
      out.values[idx * 32 + static_cast<std::size_t>(i)] =
          __half2float(d) * code;
    }
  }
  return out;
}

// One dense projection shape: out[m][batch] = W[m][k] * x[batch][k].
struct Shape {
  const char* name;
  std::uint32_t m;  // output rows
  std::uint32_t k;  // reduction / input cols
};

// The dense projections the trunk runs per layer at pp2048 (hidden 2048, 16
// heads x 256, gated Q so attn_q is 2*4096 rows, SSM conv channels 8192, SSM
// value dim 4096, shared-expert FFN 512). attn_k/attn_v (512 rows) and the
// shared-expert gate/up (512 rows) stay below the 2048-row gate and remain on
// int8.
constexpr Shape kShapes[] = {
    {"attn_q", 8192, 2048},   {"ssm_qkv", 8192, 2048},
    {"ssm_gate", 4096, 2048}, {"attn_out", 2048, 4096},
    {"ssm_out", 2048, 4096},  {"shexp_down", 2048, 512},
};

constexpr std::uint32_t kBatch = 2048;
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

// The wide-batch tile plans under comparison: kAuto (the trunk's measured
// selection), the single-stage row-groups plan (waves 8x1), the two-K-block
// staged row-groups plan now kept by kAuto for k >= 2048, the 4x2 waves
// with a two-K-block stage and a half-row tile.
constexpr gufo::core::hip::gemm::DenseF16Plan kPlans[] = {
    gufo::core::hip::gemm::DenseF16Plan::kAuto,
    gufo::core::hip::gemm::DenseF16Plan::kRowGroups8,
    gufo::core::hip::gemm::DenseF16Plan::kStagedRowGroups8,
    gufo::core::hip::gemm::DenseF16Plan::kStaged256,
    gufo::core::hip::gemm::DenseF16Plan::kTile128,
};
constexpr int kNumPlans = 5;

struct Result {
  double int8_us = 0.0;
  double f16_us[kNumPlans] = {0.0};
  double int8_err = 0.0;
  double f16_err[kNumPlans] = {0.0};
  bool f16_supported = false;
};

Result BenchShape(const Shape& s) {
  Result r;
  const std::size_t wbytes = static_cast<std::size_t>(s.m) * (s.k / 32) * 34;

  // Rotating weight copies (distinct data) so the streamed set exceeds MALL.
  std::vector<std::unique_ptr<t::HipBuffer<std::uint8_t>>> weights;
  std::vector<Q8Matrix> host_w;
  weights.reserve(kWeightCopies);
  host_w.reserve(kWeightCopies);
  for (int c = 0; c < kWeightCopies; ++c) {
    host_w.push_back(MakeQ8(s.m, s.k, 0x1234ABCDU + c * 0x9E3779B1U));
    weights.push_back(std::make_unique<t::HipBuffer<std::uint8_t>>(wbytes));
    t::CheckHip(hipMemcpy(weights[c]->get(), host_w[c].bytes.data(), wbytes,
                          hipMemcpyHostToDevice),
                "upload weight");
  }

  // Activations: FP32 for the int8 path, its binary16 copy for the WMMA path.
  const std::size_t xcount = static_cast<std::size_t>(kBatch) * s.k;
  const std::vector<float> x = t::MakeValues(xcount, 0x6E6E6E6EU, 1.0F);
  std::vector<__half> xh(xcount);
  for (std::size_t i = 0; i < xcount; ++i) {
    xh[i] = __float2half(x[i]);
  }
  t::HipBuffer<float> d_x(xcount);
  t::Upload(&d_x, x);
  t::HipBuffer<__half> d_xh(xcount);
  t::CheckHip(
      hipMemcpy(d_xh.get(), xh.data(), d_xh.bytes(), hipMemcpyHostToDevice),
      "upload f16 activation");

  const std::size_t ocount = static_cast<std::size_t>(kBatch) * s.m;
  t::HipBuffer<float> d_out(ocount);

  hipEvent_t ev0, ev1;
  hipEventCreate(&ev0);
  hipEventCreate(&ev1);

  // --- int8 MMQ path (FP32 activations) ---
  for (int i = 0; i < kWarmup; ++i) {
    qfn_mmq_q8_0_dense(weights[i % kWeightCopies]->get(), d_x.get(),
                       d_out.get(), static_cast<int>(s.m),
                       static_cast<int>(kBatch), static_cast<int>(s.k),
                       nullptr);
  }
  std::vector<float> int8_times;
  for (int i = 0; i < kReps; ++i) {
    hipEventRecord(ev0);
    qfn_mmq_q8_0_dense(weights[i % kWeightCopies]->get(), d_x.get(),
                       d_out.get(), static_cast<int>(s.m),
                       static_cast<int>(kBatch), static_cast<int>(s.k),
                       nullptr);
    hipEventRecord(ev1);
    hipEventSynchronize(ev1);
    float ms = 0.0F;
    hipEventElapsedTime(&ms, ev0, ev1);
    int8_times.push_back(ms * 1000.0F);
  }
  r.int8_us = Median(int8_times);

  // --- binary16 WMMA path (F16 activations), one plan at a time ---
  bool plan_ok[kNumPlans] = {false};
  for (int p = 0; p < kNumPlans; ++p) {
    const bool ok = fn::DenseF16Gemm(weights[0]->get(), d_xh.get(), d_out.get(),
                                     kBatch, s.m, s.k, nullptr, kPlans[p]);
    plan_ok[p] = ok;
    r.f16_supported = r.f16_supported || ok;
    if (!ok) {
      continue;
    }
    for (int i = 0; i < kWarmup; ++i) {
      fn::DenseF16Gemm(weights[i % kWeightCopies]->get(), d_xh.get(),
                       d_out.get(), kBatch, s.m, s.k, nullptr, kPlans[p]);
    }
    std::vector<float> f16_times;
    for (int i = 0; i < kReps; ++i) {
      hipEventRecord(ev0);
      fn::DenseF16Gemm(weights[i % kWeightCopies]->get(), d_xh.get(),
                       d_out.get(), kBatch, s.m, s.k, nullptr, kPlans[p]);
      hipEventRecord(ev1);
      hipEventSynchronize(ev1);
      float ms = 0.0F;
      hipEventElapsedTime(&ms, ev0, ev1);
      f16_times.push_back(ms * 1000.0F);
    }
    r.f16_us[p] = Median(f16_times);
  }

  // --- correctness against a double reference on weight copy 0 ---
  // int8 path output.
  qfn_mmq_q8_0_dense(weights[0]->get(), d_x.get(), d_out.get(),
                     static_cast<int>(s.m), static_cast<int>(kBatch),
                     static_cast<int>(s.k), nullptr);
  t::CheckHip(hipDeviceSynchronize(), "int8 sync");
  const std::vector<float> int8_out = t::Download(&d_out, ocount);
  // binary16 path output (plan 0 below, alongside its reference compare).
  if (plan_ok[0]) {
    fn::DenseF16Gemm(weights[0]->get(), d_xh.get(), d_out.get(), kBatch, s.m,
                     s.k, nullptr, kPlans[0]);
    t::CheckHip(hipDeviceSynchronize(), "f16 sync");
  }

  // Double reference (dequantized weights x FP32 activations). Sampled rows to
  // bound host cost: the reduction is O(batch*m*k) which is ~1e11 for the
  // largest shape, so check a strided subset of output entries.
  std::vector<float> ref;
  ref.reserve(ocount);
  const std::size_t stride = std::max<std::size_t>(1, ocount / 4096);
  std::vector<std::size_t> idxs;
  for (std::size_t i = 0; i < ocount; i += stride) {
    const std::size_t tok = i / s.m;
    const std::size_t row = i % s.m;
    double acc = 0.0;
    for (std::uint32_t j = 0; j < s.k; ++j) {
      acc += static_cast<double>(host_w[0].values[row * s.k + j]) *
             static_cast<double>(x[tok * s.k + j]);
    }
    ref.push_back(static_cast<float>(acc));
    idxs.push_back(i);
  }
  std::vector<float> int8_sub;
  int8_sub.reserve(idxs.size());
  for (const std::size_t i : idxs) {
    int8_sub.push_back(int8_out[i]);
  }
  r.int8_err = t::WorstRelativeToScale(ref, int8_sub);
  for (int p = 0; p < kNumPlans; ++p) {
    if (!plan_ok[p]) {
      continue;
    }
    if (p > 0) {
      fn::DenseF16Gemm(weights[0]->get(), d_xh.get(), d_out.get(), kBatch, s.m,
                       s.k, nullptr, kPlans[p]);
      t::CheckHip(hipDeviceSynchronize(), "f16 plan sync");
    }
    const std::vector<float> f16_plan_out = t::Download(&d_out, ocount);
    std::vector<float> f16_sub;
    f16_sub.reserve(idxs.size());
    for (const std::size_t i : idxs) {
      f16_sub.push_back(f16_plan_out[i]);
    }
    r.f16_err[p] = t::WorstRelativeToScale(ref, f16_sub);
  }

  hipEventDestroy(ev0);
  hipEventDestroy(ev1);
  return r;
}

}  // namespace

int main() {
  if (qfn_mmq_init(0) != 0) {
    std::fprintf(stderr, "qfn_mmq_init failed\n");
    return 1;
  }
  std::printf(
      "qwen36_a3b dense prefill GEMM: int8 mmq vs binary16 WMMA "
      "(batch=%u, %d weight copies, median of %d reps)\n\n",
      kBatch, kWeightCopies, kReps);
  static const char* const kPlanNames[kNumPlans] = {"auto", "rg8", "staged",
                                                    "staged256", "tile128"};
  std::printf("%-14s %9s", "shape", "int8_us");
  for (const char* name : kPlanNames) {
    std::printf("%11s_us", name);
  }
  for (int p = 0; p < kNumPlans; ++p) {
    std::printf("%10s", (p == 0) ? "auto/int8" : "ratio/auto");
  }
  std::printf("%12s%12s\n", "int8_err", "f16_err");
  for (const Shape& s : kShapes) {
    const Result r = BenchShape(s);
    if (!r.f16_supported) {
      std::printf("%-14s %9.1f", s.name, r.int8_us);
      for (int p = 0; p < kNumPlans; ++p) {
        std::printf("%11s", "n/a");
      }
      for (int p = 0; p < kNumPlans; ++p) {
        std::printf("%10s", "n/a");
      }
      std::printf(" %11.3e %11s\n", r.int8_err, "unsupported");
      continue;
    }
    double worst_err = 0.0;
    for (double e : r.f16_err) {
      worst_err = std::max(worst_err, e);
    }
    std::printf("%-14s %9.1f", s.name, r.int8_us);
    for (int p = 0; p < kNumPlans; ++p) {
      std::printf("%11.1f", r.f16_us[p]);
    }
    for (int p = 0; p < kNumPlans; ++p) {
      std::printf("%9.2fx", (p == 0) ? r.int8_us / r.f16_us[0]
                                     : r.f16_us[p] / r.f16_us[0]);
    }
    std::printf(" %11.3e %11.3e\n", r.int8_err, worst_err);
  }
  return 0;
}