// Operator-level checks for the ROCm attention kernels without a model.
//
// A CPU fp64 oracle pins the sink + window causal semantics for the WMMA and
// scalar kernels within a fixed absolute error, and the WMMA kernel must be
// bit-exact for the same row across batch sizes: every row's arithmetic is
// private, so the engine's "replay equals step" bitwise contract holds only
// if per-row results never depend on the batch or the padded tail.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "src/models/flint/kernels/rocm/kernels.hpp"

namespace {

constexpr std::uint32_t kHeads = 32;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kRowElems = kKvHeads * kHeadDim;
constexpr std::uint32_t kCacheRows = 4352;
constexpr float kScale = 0.08838834764831845F;  // 1 / sqrt(128)
// Absolute tolerance: f16 probabilities and Q residuals plus tensor-core
// accumulation give ~1e-5 here; anything structural (a wrong mask or window
// edge) moves whole probability mass, an order of magnitude larger.
constexpr double kTolerance = 1.5e-3;
constexpr std::size_t kNoGroup = ~static_cast<std::size_t>(0);

std::uint64_t g_rng = 0x243F6A8885A308D3ULL;

double Urng() {  // splitmix64 finalizer, [-1, 1)
  g_rng += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = g_rng;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return static_cast<double>(z >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

int g_failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

struct Case {
  std::uint32_t rows;
  std::uint32_t pos0;
  std::uint32_t window;  // 0 selects all keys
};

void Check(const std::vector<float>& q, std::size_t q_offset,
           const std::vector<__half>& kc, const std::vector<__half>& vc,
           const std::vector<float>& sinks, const Case& c,
           const std::vector<float>& got, const char* label) {
  double max_abs = 0.0;
  std::uint32_t worst = 0;
  std::vector<double> logits;
  std::vector<double> w;
  double acc_ref[kHeadDim];
  for (std::uint32_t t = 0; t < c.rows; ++t) {
    const std::uint32_t pos = c.pos0 + t;
    std::uint32_t lo = 0;
    if (c.window != 0 && pos + 1 > c.window) {
      lo = pos + 1 - c.window;
    }
    const std::uint32_t hi = pos + 1;
    for (std::uint32_t head = 0; head < kHeads; ++head) {
      const float* qr =
          q.data() + q_offset +
          (static_cast<std::size_t>(t) * kHeads + head) * kHeadDim;
      const std::size_t kv = head / (kHeads / kKvHeads);
      logits.assign(hi, 0.0);
      double max_l = sinks[head];
      for (std::uint32_t key = lo; key < hi; ++key) {
        const __half* krow = kc.data() +
                             static_cast<std::size_t>(key) * kRowElems +
                             kv * kHeadDim;
        double acc = 0.0;
        for (std::uint32_t d = 0; d < kHeadDim; ++d) {
          acc += static_cast<double>(qr[d]) * __half2float(krow[d]);
        }
        const double s = acc * kScale;
        logits[key] = s;
        max_l = s > max_l ? s : max_l;
      }
      w.assign(hi - lo + 1, 0.0);
      w[0] = std::exp(sinks[head] - max_l);
      double sum = w[0];
      for (std::uint32_t key = lo; key < hi; ++key) {
        w[key - lo + 1] = std::exp(logits[key] - max_l);
        sum += w[key - lo + 1];
      }
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        acc_ref[d] = 0.0;
      }
      for (std::uint32_t key = lo; key < hi; ++key) {
        const __half* vrow = vc.data() +
                             static_cast<std::size_t>(key) * kRowElems +
                             kv * kHeadDim;
        for (std::uint32_t d = 0; d < kHeadDim; ++d) {
          acc_ref[d] += w[key - lo + 1] * __half2float(vrow[d]);
        }
      }
      const float* got_row =
          got.data() + (static_cast<std::size_t>(t) * kHeads + head) * kHeadDim;
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        const double err =
            std::fabs(static_cast<double>(got_row[d]) - acc_ref[d] / sum);
        if (err > max_abs) {
          max_abs = err;
          worst = t * 1000000 + head * 1000 + d;
        }
      }
    }
  }
  char line[256];
  std::snprintf(line, sizeof(line),
                "%s rows=%u pos0=%u window=%u max|diff|=%.3e"
                " (row%u head%u d%u)",
                label, c.rows, c.pos0, c.window, max_abs, worst / 1000000,
                (worst / 1000) % 1000, worst % 1000);
  std::printf("%s\n", line);
  Expect(max_abs <= kTolerance, line);
}

// Magnitude and raw words of the first bitwise mismatch (±0 and tiny real
// drift look identical to memcmp, so the test reports both).
void ReportBitDiff(const std::vector<float>& a, const std::vector<float>& b,
                   std::size_t rows) {
  std::size_t t = 0;
  for (; t < rows; ++t) {
    if (std::memcmp(a.data() + t * kHeads * kHeadDim,
                    b.data() + t * kHeads * kHeadDim,
                    kHeads * kHeadDim * sizeof(float)) != 0) {
      break;
    }
  }
  float max_abs = 0.0F;
  std::uint32_t aw = 0;
  std::uint32_t bw = 0;
  std::size_t n_bad = 0;
  std::size_t bad_heads = 0;
  std::size_t dim_lo = kHeadDim;
  std::size_t dim_hi = 0;
  for (std::size_t r = t; r < rows; ++r) {
    const float* x = a.data() + r * kHeads * kHeadDim;
    const float* y = b.data() + r * kHeads * kHeadDim;
    for (std::size_t i = 0; i < kHeads * kHeadDim; ++i) {
      const float d = std::fabs(x[i] - y[i]);
      if (d > 0.0F) {
        ++n_bad;
        const std::size_t head = (i / kHeadDim) % kHeads;
        bad_heads |= std::size_t{1} << head;
        const std::size_t d_idx = i % kHeadDim;
        dim_lo = std::min(dim_lo, d_idx);
        dim_hi = std::max(dim_hi, d_idx);
      }
      if (d > max_abs) {
        max_abs = d;
        std::memcpy(&aw, &x[i], 4);
        std::memcpy(&bw, &y[i], 4);
      }
    }
  }
  std::printf(
      "  first differing row %zu of %zu | max|delta|=%.3e (%08x vs"
      " %08x)\n",
      t, rows, max_abs, aw, bw);
  std::printf("  %zu differing floats | heads mask %08x%08x | dims %zu..%zu\n",
              n_bad, static_cast<unsigned>(bad_heads >> 32),
              static_cast<unsigned>(bad_heads & 0xFFFFFFFFU), dim_lo, dim_hi);
}

}  // namespace

int main() {
  // Cases sharing pos0/window (groups below) must agree bit-for-bit on the
  // rows they have in common, whatever the batch size or padded tail.
  const std::vector<Case> cases = {
      /* 0 */ {1, 480, 513},      {4, 480, 513},   {33, 480, 513},
      /* 3 */ {65, 480, 513},     {96, 480, 513},
      /* 5 */ {1, 4200, 0},       {16, 4200, 0},   {96, 4200, 0},
      /* 8 */ {1, 0, 0},          {64, 0, 0},      {65, 0, 0},
      /*11 */ {1, 800, 513},      {33, 700, 513},  {80, 1024, 513},
      /*15 */ {65, 2000, 0},
      /* idx15 */ {32, 480, 513}, {112, 480, 513}, {2, 480, 513},
  };
  const std::vector<std::vector<std::size_t>> groups = {
      {0, 1, 2, 3, 4, 15, 16, 17}, {5, 6, 7}, {8, 9, 10}};
  // The fp64 oracle is cubic-cheap only for a bounded row/key budget; the
  // invariance checks below then cover every remaining case.
  const std::vector<std::size_t> oracle = {2, 4, 6, 7, 10, 13};

  // Shared Q: every case in a group must see the same row-t query vector for
  // any rows t it has with the group's largest case, or the bitwise
  // invariance comparisons below would compare different inputs.
  std::vector<std::size_t> q_group(cases.size(), kNoGroup);
  std::vector<std::vector<float>> pool(groups.size());
  for (std::size_t g = 0; g < groups.size(); ++g) {
    std::size_t max_rows = 0;
    for (std::size_t i : groups[g]) {
      max_rows = std::max(max_rows, static_cast<std::size_t>(cases[i].rows));
    }
    pool[g].assign(max_rows * kHeads * kHeadDim, 0.0F);
    for (std::size_t i : groups[g]) {
      q_group[i] = g;
    }
  }
  std::vector<std::size_t> offset(cases.size(), 0);
  std::size_t q_total = 0;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    offset[i] = q_total;
    if (q_group[i] != kNoGroup) {
      continue;  // group cases share one pool at the end of q
    }
    q_total += static_cast<std::size_t>(cases[i].rows) * kHeads * kHeadDim;
  }
  std::vector<std::size_t> group_base(groups.size());
  for (std::size_t g = 0; g < groups.size(); ++g) {
    group_base[g] = q_total;
    q_total += pool[g].size();
  }
  for (std::size_t i = 0; i < cases.size(); ++i) {
    if (q_group[i] != kNoGroup) {
      offset[i] = group_base[q_group[i]];
    }
  }

  g_rng = 0x243F6A8885A308D3ULL;
  std::vector<float> q(q_total);
  for (auto& v : q) {
    v = static_cast<float>(Urng());
  }
  std::vector<__half> kc(kCacheRows * kRowElems), vc(kCacheRows * kRowElems);
  for (auto& h : kc) {
    h = __float2half_rn(static_cast<float>(Urng()));
  }
  for (auto& h : vc) {
    h = __float2half_rn(static_cast<float>(Urng()) * 0.5F);
  }
  std::vector<float> sinks(kHeads);
  for (auto& s : sinks) {
    s = static_cast<float>(Urng()) * 2.0F;
  }

  float* dq;
  __half* dkc;
  __half* dvc;
  float* dsinks;
  float* dout;
  if (hipMalloc(&dq, q_total * sizeof(float)) != hipSuccess ||
      hipMalloc(&dkc, kc.size() * sizeof(__half)) != hipSuccess ||
      hipMalloc(&dvc, vc.size() * sizeof(__half)) != hipSuccess ||
      hipMalloc(&dsinks, sinks.size() * sizeof(float)) != hipSuccess ||
      hipMalloc(&dout, q_total * sizeof(float)) != hipSuccess) {
    std::printf("SKIP: hipMalloc failed\n");
    return 77;
  }
  assert(hipMemcpy(dq, q.data(), q_total * sizeof(float),
                   hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(dkc, kc.data(), kc.size() * sizeof(__half),
                   hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(dvc, vc.data(), vc.size() * sizeof(__half),
                   hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(dsinks, sinks.data(), sinks.size() * sizeof(float),
                   hipMemcpyHostToDevice) == hipSuccess);

  // Diagnostic: zero the K/V rows in the group-A key range beyond row 0's
  // bound so the masked-past-hi data (real cache rows in the 112 launch,
  // fetch zero-fill in the 1 launch) is bit-identical in every group-A
  // launch. If the group-A drift disappears, masked p * V products with
  // launch-dependent V content are the bitwise channel.
  const std::size_t z_from = 481 * kRowElems;
  const std::size_t z_count = (592 - 481) * kRowElems * sizeof(__half);
  std::memset(kc.data() + z_from, 0, z_count);
  std::memset(vc.data() + z_from, 0, z_count);
  assert(hipMemcpy(dkc, kc.data(), kc.size() * sizeof(__half),
                   hipMemcpyHostToDevice) == hipSuccess);
  assert(hipMemcpy(dvc, vc.data(), vc.size() * sizeof(__half),
                   hipMemcpyHostToDevice) == hipSuccess);

  std::vector<std::vector<float>> wmma_out(cases.size());
  std::vector<std::vector<float>> scalar_out(cases.size());
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const Case& c = cases[i];
    const float* cq = dq + offset[i];
    const std::size_t n = static_cast<std::size_t>(c.rows) * kHeads * kHeadDim;
    const bool launched = gufo::models::flint::rocm::LaunchWmmaAttention(
        cq, dkc, dvc, dsinks, kScale, c.pos0, c.rows, c.window, kHeads,
        kKvHeads, kRowElems, dout, nullptr);
    Expect(launched, "LaunchWmmaAttention accepted the shape");
    wmma_out[i].assign(n, 0.0F);
    assert(hipMemcpy(wmma_out[i].data(), dout, n * sizeof(float),
                     hipMemcpyDeviceToHost) == hipSuccess);
    gufo::models::flint::rocm::Attention(cq, dkc, dvc, dsinks, kScale, c.pos0,
                                         c.rows, c.window, kHeads, kKvHeads,
                                         kHeadDim, kRowElems, dout, nullptr);
    scalar_out[i].assign(n, 0.0F);
    assert(hipMemcpy(scalar_out[i].data(), dout, n * sizeof(float),
                     hipMemcpyDeviceToHost) == hipSuccess);
  }

  for (std::size_t i : oracle) {
    Check(q, offset[i], kc, vc, sinks, cases[i], wmma_out[i], "wmma");
    Check(q, offset[i], kc, vc, sinks, cases[i], scalar_out[i], "scalar");
    const bool routed = std::memcmp(wmma_out[i].data(), scalar_out[i].data(),
                                    wmma_out[i].size() * sizeof(float)) == 0;
    Expect(routed, "dispatched Attention equals the direct WMMA launch");
  }

  // Bitwise row-invariance: the same row must produce identical bits
  // regardless of the batch size and padded tail around it.
  for (const auto& group : groups) {
    std::size_t base = group.front();
    for (std::size_t i : group) {
      if (cases[i].rows > cases[base].rows) {
        base = i;
      }
    }
    for (std::size_t i : group) {
      if (i == base) {
        continue;
      }
      const std::size_t n = cases[i].rows;  // base is the largest
      const bool same_w =
          std::memcmp(wmma_out[base].data(), wmma_out[i].data(),
                      n * kHeads * kHeadDim * sizeof(float)) == 0;
      const bool same_s =
          std::memcmp(scalar_out[base].data(), scalar_out[i].data(),
                      n * kHeads * kHeadDim * sizeof(float)) == 0;
      char msg[160];
      std::snprintf(msg, sizeof(msg),
                    "bitwise prefix rows %zu (case %zu vs %zu pos0=%u"
                    " window=%u)",
                    n, base, i, cases[base].pos0, cases[base].window);
      Expect(same_w, (std::string(msg) + " wmma").c_str());
      Expect(same_s, (std::string(msg) + " routed").c_str());
      if (!same_w) {
        ReportBitDiff(wmma_out[base], wmma_out[i], n);
      }
      if (!same_s) {
        ReportBitDiff(scalar_out[base], scalar_out[i], n);
      }
    }
  }

  assert(hipFree(dq) == hipSuccess);
  assert(hipFree(dkc) == hipSuccess);
  assert(hipFree(dvc) == hipSuccess);
  assert(hipFree(dsinks) == hipSuccess);
  assert(hipFree(dout) == hipSuccess);
  if (g_failures != 0) {
    std::printf("flint attention_op_test: %d FAILURES\n", g_failures);
    return 1;
  }
  std::printf("flint attention_op_test: OK\n");
  return 0;
}
