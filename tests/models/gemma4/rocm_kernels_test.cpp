// GPU kernel-tier parity for Gemma-4-26B-A4B: the quantized GEMV tier, the
// mmq/hipBLAS GEMM tier, the fused routing operators and the routed F16 WMMA
// MoE pipeline (compact -> gate/up GEMM -> geglu -> down GEMM -> weighted
// combine) run on synthetic tensors at the artifact's geometry (hidden 2816,
// expert_ff 704, 128 experts top-8) and are compared against double-precision
// host references that dequantize the same encodings. No model artifact is
// required; the real tensors are exercised by the forward check.
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/gemma4/kernels/rocm/attention.hpp"
#include "src/models/gemma4/kernels/rocm/fused.hpp"
#include "src/models/gemma4/kernels/rocm/gemm.hpp"
#include "src/models/gemma4/kernels/rocm/gemv.hpp"
#include "src/models/gemma4/kernels/rocm/routed_f16.hpp"
#include "tests/models/gemma4/hip_test.hpp"

namespace q = gufo::models::gemma4::rocm;
namespace t = gufo::tests::gemma4;

namespace {

constexpr std::uint32_t kHidden = 2816;
constexpr std::uint32_t kExpertFf = 704;

bool Check(const char* name, double worst, double tolerance) {
  std::cout << name << " worst error " << worst << '\n';
  if (!(worst <= tolerance)) {
    std::cerr << name << " exceeded tolerance " << tolerance << '\n';
    return false;
  }
  return true;
}

// BF16 is the top 16 bits of a float32; encode round-to-nearest-even and
// decode exactly so the host reference matches the kernel bit-for-bit.
std::uint16_t FloatToBf16Bits(float f) {
  std::uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>(bits >> 16);
}

float Bf16BitsToFloat(std::uint16_t bits) {
  const std::uint32_t fbits = static_cast<std::uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &fbits, sizeof(f));
  return f;
}

// Synthetic Q8_0 matrix: per 32-element block a half scale and 32 int8 codes.
// `values` receives the exactly dequantized rows for the host reference.
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

std::vector<std::int32_t> MakeIds(std::size_t count, std::uint32_t n_experts,
                                  std::uint32_t seed) {
  std::vector<std::int32_t> ids(count);
  for (auto& id : ids) {
    id = static_cast<std::int32_t>(t::NextRandom(&seed) % n_experts);
  }
  return ids;
}

bool TestGemv() {
  constexpr std::uint32_t kRows = 64;
  const auto x = t::MakeValues(kHidden, 0x11112222U, 1.0F);
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_out(kRows);
  bool ok = true;

  const auto w8 = MakeQ8(kRows, kHidden, 0x0BADF00DU);
  t::HipBuffer<std::uint8_t> d_w8(w8.bytes.size());
  t::CheckHip(hipMemcpy(d_w8.get(), w8.bytes.data(), d_w8.bytes(),
                        hipMemcpyHostToDevice),
              "upload q8_0 weights");
  q::Gemv(d_w8.get(), q::GemvType::kQ8_0, kRows, kHidden, 34 * (kHidden / 32),
          d_x.get(), d_out.get(), nullptr);
  t::CheckHip(hipDeviceSynchronize(), "Gemv Q8_0 synchronization");
  auto got = t::Download(&d_out, kRows);
  std::vector<float> ref(kRows);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    double acc = 0.0;
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      acc += static_cast<double>(
                 w8.values[static_cast<std::size_t>(r) * kHidden + i]) *
             x[i];
    }
    ref[r] = static_cast<float>(acc);
  }
  ok = Check("Gemv Q8_0", t::WorstRelative(ref, got), 2e-3) && ok;

  std::vector<std::uint16_t> wb(static_cast<std::size_t>(kRows) * kHidden);
  std::uint32_t seed = 0x9A8B7C6DU;
  for (auto& value : wb) {
    value = FloatToBf16Bits(
        static_cast<float>(static_cast<int>(t::NextRandom(&seed) & 0xFFFFU) -
                           32768) /
        32768.0F);
  }
  t::HipBuffer<std::uint16_t> d_wb(wb.size());
  t::CheckHip(
      hipMemcpy(d_wb.get(), wb.data(), d_wb.bytes(), hipMemcpyHostToDevice),
      "upload bf16 weights");
  q::Gemv(d_wb.get(), q::GemvType::kBF16, kRows, kHidden, 2, d_x.get(),
          d_out.get(), nullptr);
  t::CheckHip(hipDeviceSynchronize(), "Gemv BF16 synchronization");
  got = t::Download(&d_out, kRows);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    double acc = 0.0;
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      acc += static_cast<double>(Bf16BitsToFloat(
                 wb[static_cast<std::size_t>(r) * kHidden + i])) *
             x[i];
    }
    ref[r] = static_cast<float>(acc);
  }
  ok = Check("Gemv BF16", t::WorstRelative(ref, got), 1e-4) && ok;
  return ok;
}

bool TestEmbedRow() {
  constexpr std::uint32_t kRows = 8;
  const auto w8 = MakeQ8(kRows, kHidden, 0x5EED1234U);
  t::HipBuffer<std::uint8_t> d_w(w8.bytes.size());
  t::CheckHip(hipMemcpy(d_w.get(), w8.bytes.data(), w8.bytes.size(),
                        hipMemcpyHostToDevice),
              "upload embedding");
  t::HipBuffer<float> d_row(kHidden);
  q::EmbedRow(d_w.get(), q::GemvType::kQ8_0, 5, kHidden, d_row.get(), nullptr);
  t::CheckHip(hipDeviceSynchronize(), "EmbedRow synchronization");
  const auto got = t::Download(&d_row, kHidden);
  std::vector<float> ref(w8.values.begin() + 5 * kHidden,
                         w8.values.begin() + 6 * kHidden);
  return Check("EmbedRow Q8_0", t::WorstRelative(ref, got), 1e-6);
}

bool TestGemvGrouped() {
  constexpr std::uint32_t kExperts = 4;
  constexpr std::uint32_t kRows = 64;
  constexpr std::uint32_t kSlots = 8;
  const auto ids = MakeIds(kSlots, kExperts, 0x17171717U);
  const auto x = t::MakeValues(kSlots * kHidden, 0x23232323U, 1.0F);
  const auto w8 =
      MakeQ8(static_cast<std::size_t>(kExperts) * kRows, kHidden, 0x31313131U);
  const std::size_t row_bytes = 34 * (kHidden / 32);

  t::HipBuffer<std::uint8_t> d_w(w8.bytes.size());
  t::CheckHip(hipMemcpy(d_w.get(), w8.bytes.data(), w8.bytes.size(),
                        hipMemcpyHostToDevice),
              "upload expert weights");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<std::int32_t> d_ids(ids.size());
  t::CheckHip(
      hipMemcpy(d_ids.get(), ids.data(), d_ids.bytes(), hipMemcpyHostToDevice),
      "upload ids");
  t::HipBuffer<float> d_out(static_cast<std::size_t>(kSlots) * kRows);

  q::GemvGrouped(d_w.get(), q::GemvType::kQ8_0, kRows * row_bytes, d_ids.get(),
                 kSlots, kRows, kHidden, d_x.get(), kHidden, d_out.get(),
                 nullptr);
  t::CheckHip(hipDeviceSynchronize(), "GemvGrouped synchronization");
  const auto got =
      t::Download(&d_out, static_cast<std::size_t>(kSlots) * kRows);
  std::vector<float> ref(static_cast<std::size_t>(kSlots) * kRows);
  for (std::uint32_t s = 0; s < kSlots; ++s) {
    const std::size_t expert = static_cast<std::size_t>(ids[s]) * kRows;
    for (std::uint32_t r = 0; r < kRows; ++r) {
      double acc = 0.0;
      for (std::uint32_t i = 0; i < kHidden; ++i) {
        acc += static_cast<double>(w8.values[(expert + r) * kHidden + i]) *
               x[static_cast<std::size_t>(s) * kHidden + i];
      }
      ref[static_cast<std::size_t>(s) * kRows + r] = static_cast<float>(acc);
    }
  }
  bool ok = Check("GemvGrouped Q8_0", t::WorstRelative(ref, got), 1e-4);

  // Fused gate/up layout: one [experts][2 * kExpertFf][kHidden] tensor with
  // the gate rows first; the pair kernel reads both halves with a custom
  // expert stride.
  constexpr std::uint32_t kFusedRows = 2 * kExpertFf;
  const auto gu = MakeQ8(static_cast<std::size_t>(kExperts) * kFusedRows,
                         kHidden, 0x45454545U);
  t::HipBuffer<std::uint8_t> d_gu(gu.bytes.size());
  t::CheckHip(hipMemcpy(d_gu.get(), gu.bytes.data(), gu.bytes.size(),
                        hipMemcpyHostToDevice),
              "upload fused gate_up");
  t::HipBuffer<float> d_gate(static_cast<std::size_t>(kSlots) * kExpertFf);
  t::HipBuffer<float> d_up(static_cast<std::size_t>(kSlots) * kExpertFf);
  const std::size_t fused_stride = kFusedRows * row_bytes;
  if (!q::GemvGroupedPair(d_gu.get(), d_gu.get() + kExpertFf * row_bytes,
                          q::GemvType::kQ8_0, fused_stride, d_ids.get(), kSlots,
                          kExpertFf, kHidden, d_x.get(), kHidden, d_gate.get(),
                          d_up.get(), nullptr)) {
    std::cerr << "GemvGroupedPair unsupported for Q8_0\n";
    return false;
  }
  t::CheckHip(hipDeviceSynchronize(), "GemvGroupedPair synchronization");
  const auto gate =
      t::Download(&d_gate, static_cast<std::size_t>(kSlots) * kExpertFf);
  const auto up =
      t::Download(&d_up, static_cast<std::size_t>(kSlots) * kExpertFf);
  std::vector<float> ref_gate(static_cast<std::size_t>(kSlots) * kExpertFf);
  std::vector<float> ref_up(static_cast<std::size_t>(kSlots) * kExpertFf);
  for (std::uint32_t s = 0; s < kSlots; ++s) {
    const std::size_t base = static_cast<std::size_t>(ids[s]) * kFusedRows;
    for (std::uint32_t r = 0; r < kExpertFf; ++r) {
      double g = 0.0;
      double u = 0.0;
      for (std::uint32_t i = 0; i < kHidden; ++i) {
        const double xv = x[static_cast<std::size_t>(s) * kHidden + i];
        g += static_cast<double>(gu.values[(base + r) * kHidden + i]) * xv;
        u += static_cast<double>(
                 gu.values[(base + kExpertFf + r) * kHidden + i]) *
             xv;
      }
      ref_gate[static_cast<std::size_t>(s) * kExpertFf + r] =
          static_cast<float>(g);
      ref_up[static_cast<std::size_t>(s) * kExpertFf + r] =
          static_cast<float>(u);
    }
  }
  ok = Check("GemvGroupedPair fused", t::WorstRelative(ref_gate, gate), 2e-3) &&
       ok;
  ok = Check("GemvGroupedPair fused up", t::WorstRelative(ref_up, up), 2e-3) &&
       ok;
  return ok;
}

bool TestGeglu() {
  constexpr std::uint32_t kRows = 4;
  const auto gu = t::MakeValues(static_cast<std::size_t>(kRows) * 2 * kExpertFf,
                                0x67676767U, 4.0F);
  std::vector<float> ref(static_cast<std::size_t>(kRows) * kExpertFf);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    for (std::uint32_t i = 0; i < kExpertFf; ++i) {
      ref[static_cast<std::size_t>(r) * kExpertFf + i] = static_cast<float>(
          t::GeluD(gu[static_cast<std::size_t>(r) * 2 * kExpertFf + i]) *
          gu[static_cast<std::size_t>(r) * 2 * kExpertFf + kExpertFf + i]);
    }
  }
  t::HipBuffer<float> d_gu(gu.size());
  t::Upload(&d_gu, gu);
  t::HipBuffer<float> d_act(ref.size());
  q::GegluF32(d_gu.get(), d_act.get(), ref.size(), kExpertFf, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "GegluF32 synchronization");
  bool ok = Check("Geglu F32",
                  t::WorstRelative(ref, t::Download(&d_act, ref.size())), 1e-3);

  std::vector<__half> gu_h(gu.size());
  for (std::size_t i = 0; i < gu.size(); ++i) {
    gu_h[i] = __float2half(gu[i]);
  }
  std::vector<__half> act_h(ref.size());
  t::HipBuffer<__half> d_gu_h(gu_h.size());
  t::CheckHip(hipMemcpy(d_gu_h.get(), gu_h.data(), d_gu_h.bytes(),
                        hipMemcpyHostToDevice),
              "upload gu half");
  t::HipBuffer<__half> d_act_h(act_h.size());
  q::GegluF16(d_gu_h.get(), d_act_h.get(), act_h.size(), kExpertFf, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "GegluF16 synchronization");
  t::CheckHip(hipMemcpy(act_h.data(), d_act_h.get(), d_act_h.bytes(),
                        hipMemcpyDeviceToHost),
              "download act half");
  std::vector<float> got(act_h.size());
  for (std::size_t i = 0; i < act_h.size(); ++i) {
    got[i] = __half2float(act_h[i]);
  }
  return Check("Geglu F16", t::WorstRelative(ref, got), 1e-2) && ok;
}

bool TestRouterTopK() {
  constexpr std::uint32_t kTokens = 4;
  constexpr std::uint32_t kExperts = 128;
  constexpr std::uint32_t kTop = 8;
  const auto logits = t::MakeValues(
      static_cast<std::size_t>(kTokens) * kExperts, 0x89898989U, 6.0F);
  t::HipBuffer<float> d_logits(logits.size());
  t::Upload(&d_logits, logits);
  t::HipBuffer<std::int32_t> d_ids(static_cast<std::size_t>(kTokens) * kTop);
  t::HipBuffer<float> d_weights(static_cast<std::size_t>(kTokens) * kTop);
  q::RouterTopK(d_logits.get(), kExperts, d_ids.get(), d_weights.get(), kTokens,
                kExperts, kTop, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "RouterTopK synchronization");
  std::vector<std::int32_t> got_ids(static_cast<std::size_t>(kTokens) * kTop);
  t::CheckHip(hipMemcpy(got_ids.data(), d_ids.get(), d_ids.bytes(),
                        hipMemcpyDeviceToHost),
              "download ids");
  const auto weights =
      t::Download(&d_weights, static_cast<std::size_t>(kTokens) * kTop);

  bool ok = true;
  for (std::uint32_t tok = 0; tok < kTokens; ++tok) {
    std::vector<double> prob(kExperts);
    double max_logit = -INFINITY;
    for (std::uint32_t e = 0; e < kExperts; ++e) {
      max_logit = std::max(
          max_logit, static_cast<double>(
                         logits[static_cast<std::size_t>(tok) * kExperts + e]));
    }
    double sum = 0.0;
    for (std::uint32_t e = 0; e < kExperts; ++e) {
      prob[e] =
          std::exp(static_cast<double>(
                       logits[static_cast<std::size_t>(tok) * kExperts + e]) -
                   max_logit);
      sum += prob[e];
    }
    for (auto& p : prob) {
      p /= sum;
    }
    std::vector<std::uint32_t> order(kExperts);
    for (std::uint32_t e = 0; e < kExperts; ++e) {
      order[e] = e;
    }
    std::stable_sort(
        order.begin(), order.end(),
        [&](std::uint32_t a, std::uint32_t b) { return prob[a] > prob[b]; });
    double sel_sum = 0.0;
    for (std::uint32_t s = 0; s < kTop; ++s) {
      sel_sum += prob[order[s]];
    }
    sel_sum = std::max(sel_sum, 6.103515625e-5);
    for (std::uint32_t s = 0; s < kTop; ++s) {
      const std::size_t slot = static_cast<std::size_t>(tok) * kTop + s;
      if (got_ids[slot] != static_cast<std::int32_t>(order[s])) {
        std::cerr << "RouterTopK id mismatch token " << tok << " slot " << s
                  << " want " << order[s] << " got " << got_ids[slot] << '\n';
        ok = false;
      }
      const double want = prob[order[s]] / sel_sum;
      const double scale = std::max(1e-3, want);
      if (std::abs(want - weights[slot]) / scale > 1e-4) {
        std::cerr << "RouterTopK weight mismatch token " << tok << " slot " << s
                  << " want " << want << " got " << weights[slot] << '\n';
        ok = false;
      }
    }
  }
  std::cout << "RouterTopK " << (ok ? "matches" : "MISMATCH") << '\n';
  return ok;
}

bool TestNormsAndAdd() {
  constexpr std::uint32_t kRows = 3;
  const auto x = t::MakeValues(static_cast<std::size_t>(kRows) * kHidden,
                               0x3B3B3B3BU, 2.0F);
  const auto gamma = t::MakeValues(kHidden, 0x4C4C4C4CU, 1.0F, 1.0F);
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_gamma(gamma.size());
  t::Upload(&d_gamma, gamma);
  t::HipBuffer<float> d_out(x.size());
  q::RmsNormRows(d_x.get(), d_gamma.get(), d_out.get(), kRows, kHidden, 1e-6F,
                 nullptr);
  t::CheckHip(hipDeviceSynchronize(), "RmsNormRows synchronization");
  std::vector<float> ref(x.size());
  for (std::uint32_t r = 0; r < kRows; ++r) {
    double ss = 0.0;
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      const double v = x[static_cast<std::size_t>(r) * kHidden + i];
      ss += v * v;
    }
    const float scale =
        static_cast<float>(1.0 / std::sqrt(ss / kHidden + 1e-6));
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      ref[static_cast<std::size_t>(r) * kHidden + i] =
          x[static_cast<std::size_t>(r) * kHidden + i] * scale * gamma[i];
    }
  }
  bool ok = Check("RmsNormRows",
                  t::WorstRelative(ref, t::Download(&d_out, x.size())), 1e-5);

  const auto addend = t::MakeValues(kHidden, 0x5D5D5D5DU, 2.0F);
  std::vector<float> single(x.begin(), x.begin() + kHidden);
  t::HipBuffer<float> d_single(single.size());
  t::Upload(&d_single, single);
  t::HipBuffer<float> d_addend(addend.size());
  t::Upload(&d_addend, addend);
  t::HipBuffer<float> d_norm(kHidden);
  q::FusedAddRmsNorm(d_single.get(), d_addend.get(), d_gamma.get(),
                     d_norm.get(), kHidden, 1e-6F, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "FusedAddRmsNorm synchronization");
  std::vector<float> ref_add(kHidden);
  double ss = 0.0;
  for (std::uint32_t i = 0; i < kHidden; ++i) {
    const double v = static_cast<double>(single[i]) + addend[i];
    ss += v * v;
  }
  const float scale = static_cast<float>(1.0 / std::sqrt(ss / kHidden + 1e-6));
  for (std::uint32_t i = 0; i < kHidden; ++i) {
    ref_add[i] =
        static_cast<float>(static_cast<double>(single[i]) + addend[i]) * scale *
        gamma[i];
  }
  ok = Check("FusedAddRmsNorm",
             t::WorstRelative(ref_add, t::Download(&d_norm, kHidden)), 1e-5) &&
       ok;

  t::Upload(&d_single, single);
  q::Add(d_single.get(), d_addend.get(), kHidden, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "Add synchronization");
  for (std::uint32_t i = 0; i < kHidden; ++i) {
    ref_add[i] = single[i] + addend[i];
  }
  return Check("Add",
               t::WorstRelative(ref_add, t::Download(&d_single, kHidden)),
               1e-6) &&
         ok;
}

bool TestGemm() {
  constexpr std::uint32_t kRows = 64;
  constexpr std::uint32_t kBatch = 8;
  const auto x = t::MakeValues(static_cast<std::size_t>(kBatch) * kHidden,
                               0x6E6E6E6EU, 1.0F);
  const auto w8 = MakeQ8(kRows, kHidden, 0x7F7F7F7FU);
  t::HipBuffer<std::uint8_t> d_w(w8.bytes.size());
  t::CheckHip(hipMemcpy(d_w.get(), w8.bytes.data(), w8.bytes.size(),
                        hipMemcpyHostToDevice),
              "upload q8_0 matrix");
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);
  t::HipBuffer<float> d_out(static_cast<std::size_t>(kBatch) * kRows);
  q::Gemm(d_w.get(), q::GemvType::kQ8_0, kRows, kHidden, 34 * (kHidden / 32),
          d_x.get(), d_out.get(), kBatch, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "Gemm synchronization");
  const auto got =
      t::Download(&d_out, static_cast<std::size_t>(kBatch) * kRows);
  std::vector<float> ref(static_cast<std::size_t>(kBatch) * kRows);
  for (std::uint32_t tok = 0; tok < kBatch; ++tok) {
    for (std::uint32_t r = 0; r < kRows; ++r) {
      double acc = 0.0;
      for (std::uint32_t i = 0; i < kHidden; ++i) {
        acc += static_cast<double>(
                   w8.values[static_cast<std::size_t>(r) * kHidden + i]) *
               x[static_cast<std::size_t>(tok) * kHidden + i];
      }
      ref[static_cast<std::size_t>(tok) * kRows + r] = static_cast<float>(acc);
    }
  }
  return Check("Gemm Q8_0 batch8", t::WorstRelativeToScale(ref, got), 2e-2);
}

// Full routed MoE prefill pipeline at the artifact geometry: fused gate/up
// [experts][1408][2816] Q8_0 -> geglu -> down [experts][2816][704] Q8_0 ->
// weighted combine with per-expert down scales, against a double reference.
bool TestRoutedPipeline() {
  constexpr std::uint32_t kExperts = 8;
  constexpr std::uint32_t kTokens = 16;
  constexpr std::uint32_t kTop = 8;
  constexpr std::uint32_t kSlots = kTokens * kTop;
  const auto ids = MakeIds(kSlots, kExperts, 0x91919191U);
  const auto x = t::MakeValues(static_cast<std::size_t>(kTokens) * kHidden,
                               0xA2A2A2A2U, 1.0F);
  std::vector<float> weights(kSlots);
  std::uint32_t weight_seed = 0xD5D5D5D5U;
  for (auto& w : weights) {
    w = 0.1F + 0.9F *
                   static_cast<float>(t::NextRandom(&weight_seed) & 0xFFFFU) /
                   65535.0F;
  }
  std::vector<float> down_scale(kExperts);
  for (auto& s : down_scale) {
    s = 0.5F +
        static_cast<float>(t::NextRandom(&weight_seed) & 0xFFFFU) / 65535.0F;
  }
  // Realistic weight magnitude: the artifact's projections act on
  // RMS-normalized O(1) activations, so gate/up outputs stay O(10) and the F16
  // geglu results stay far below the half-precision ceiling.
  const auto gu = MakeQ8(static_cast<std::size_t>(kExperts) * 2 * kExpertFf,
                         kHidden, 0xB3B3B3B3U, 0.008F);
  const auto dn = MakeQ8(static_cast<std::size_t>(kExperts) * kHidden,
                         kExpertFf, 0xC4C4C4C4U, 0.008F);

  t::HipBuffer<std::uint8_t> d_gu(gu.bytes.size());
  t::CheckHip(hipMemcpy(d_gu.get(), gu.bytes.data(), gu.bytes.size(),
                        hipMemcpyHostToDevice),
              "upload gate_up");
  t::HipBuffer<std::uint8_t> d_dn(dn.bytes.size());
  t::CheckHip(hipMemcpy(d_dn.get(), dn.bytes.data(), dn.bytes.size(),
                        hipMemcpyHostToDevice),
              "upload down");
  t::HipBuffer<std::int32_t> d_ids(ids.size());
  t::CheckHip(
      hipMemcpy(d_ids.get(), ids.data(), d_ids.bytes(), hipMemcpyHostToDevice),
      "upload ids");
  t::HipBuffer<float> d_weights(weights.size());
  t::Upload(&d_weights, weights);
  t::HipBuffer<float> d_scale(down_scale.size());
  t::Upload(&d_scale, down_scale);
  t::HipBuffer<float> d_x(x.size());
  t::Upload(&d_x, x);

  t::HipBuffer<std::uint32_t> d_counts(kExperts);
  q::ExpertCounts(d_ids.get(), d_counts.get(), kTokens, kExperts, kTop,
                  nullptr);
  std::vector<std::uint32_t> counts(kExperts);
  t::CheckHip(hipMemcpy(counts.data(), d_counts.get(), d_counts.bytes(),
                        hipMemcpyDeviceToHost),
              "download counts");

  // Host tile map: one entry per 64-row tile of each 16-padded bucket.
  std::vector<std::int32_t> tiles;
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    const std::uint32_t padded = (counts[e] + 15U) / 16U * 16U;
    for (std::uint32_t j = 0; j < (padded + 63U) / 64U; ++j) {
      tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
    }
  }
  if (tiles.empty()) {
    std::cerr << "routed test produced no tiles\n";
    return false;
  }
  t::HipBuffer<std::int32_t> d_tiles(tiles.size());
  t::CheckHip(hipMemcpy(d_tiles.get(), tiles.data(), d_tiles.bytes(),
                        hipMemcpyHostToDevice),
              "upload tiles");

  const std::size_t compact_rows = q::RoutedCompactRows(kSlots, kExperts);
  t::HipBuffer<std::int32_t> d_pad_bounds(kExperts + 1);
  t::HipBuffer<std::int32_t> d_cursors(kExperts);
  t::HipBuffer<std::int32_t> d_rows_token(compact_rows);
  t::HipBuffer<std::int32_t> d_rows_slot(compact_rows);
  q::RoutedCompact(d_ids.get(), d_counts.get(), d_pad_bounds.get(),
                   d_cursors.get(), d_rows_token.get(), d_rows_slot.get(),
                   kTokens, kTop, kExperts, nullptr);

  t::HipBuffer<__half> d_x_half(x.size());
  q::NarrowActivations(d_x.get(), d_x_half.get(), false, x.size(), nullptr);
  t::HipBuffer<__half> d_gu_half(static_cast<std::size_t>(kSlots) * 2 *
                                 kExpertFf);
  if (!q::RoutedF16Gemm(d_gu.get(), q::WeightType::kQ8_0, d_x_half.get(),
                        d_tiles.get(), static_cast<std::uint32_t>(tiles.size()),
                        64, d_pad_bounds.get(), d_rows_token.get(),
                        d_rows_slot.get(), nullptr, d_gu_half.get(),
                        2 * kExpertFf, kHidden, nullptr)) {
    std::cerr << "RoutedF16Gemm gate_up unsupported\n";
    return false;
  }
  t::HipBuffer<__half> d_act_half(static_cast<std::size_t>(kSlots) * kExpertFf);
  q::GegluF16(d_gu_half.get(), d_act_half.get(),
              static_cast<std::size_t>(kSlots) * kExpertFf, kExpertFf, nullptr);
  t::HipBuffer<float> d_expert_out(static_cast<std::size_t>(kSlots) * kHidden);
  if (!q::RoutedF16Gemm(d_dn.get(), q::WeightType::kQ8_0, d_act_half.get(),
                        d_tiles.get(), static_cast<std::uint32_t>(tiles.size()),
                        64, d_pad_bounds.get(), d_rows_slot.get(),
                        d_rows_slot.get(), d_expert_out.get(), nullptr, kHidden,
                        kExpertFf, nullptr)) {
    std::cerr << "RoutedF16Gemm down unsupported\n";
    return false;
  }
  t::HipBuffer<float> d_out(static_cast<std::size_t>(kTokens) * kHidden);
  q::MoeEpilogue(d_expert_out.get(), d_weights.get(), d_ids.get(),
                 d_scale.get(), d_out.get(), kTokens, kTop, kHidden, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "routed pipeline synchronization");
  const auto got =
      t::Download(&d_out, static_cast<std::size_t>(kTokens) * kHidden);

  std::vector<float> ref(static_cast<std::size_t>(kTokens) * kHidden, 0.0F);
  std::vector<double> guv(2 * kExpertFf);
  std::vector<double> act(kExpertFf);
  for (std::uint32_t tok = 0; tok < kTokens; ++tok) {
    for (std::uint32_t s = 0; s < kTop; ++s) {
      const std::size_t slot = static_cast<std::size_t>(tok) * kTop + s;
      const std::size_t expert = static_cast<std::size_t>(ids[slot]);
      for (std::uint32_t r = 0; r < 2 * kExpertFf; ++r) {
        double acc = 0.0;
        for (std::uint32_t i = 0; i < kHidden; ++i) {
          acc += static_cast<double>(
                     gu.values[(expert * 2 * kExpertFf + r) * kHidden + i]) *
                 x[static_cast<std::size_t>(tok) * kHidden + i];
        }
        guv[r] = acc;
      }
      for (std::uint32_t i = 0; i < kExpertFf; ++i) {
        act[i] = t::GeluD(guv[i]) * guv[kExpertFf + i];
      }
      const double w = static_cast<double>(weights[slot]) * down_scale[expert];
      for (std::uint32_t r = 0; r < kHidden; ++r) {
        double acc = 0.0;
        for (std::uint32_t i = 0; i < kExpertFf; ++i) {
          acc += static_cast<double>(
                     dn.values[(expert * kHidden + r) * kExpertFf + i]) *
                 act[i];
        }
        ref[static_cast<std::size_t>(tok) * kHidden + r] +=
            static_cast<float>(w * acc);
      }
    }
  }
  return Check("Routed MoE pipeline", t::WorstRelativeToScale(ref, got), 2e-2);
}

}  // namespace

// Attention front-end (per-head RMSNorm + NEOX rope) and back-end (GQA
// decode over the F16 cache with a sliding window, causal windowed tiled
// prefill) for both layer classes: SWA (head_dim 256, 8 kv heads) and full
// (head_dim 512, 2 kv heads). References are double-precision over the same
// F16-rounded cache contents.
bool TestAttention() {
  bool ok = true;
  constexpr std::uint32_t kHeads = 16;
  struct ClassCfg {
    std::uint32_t head_dim;
    std::uint32_t kv_heads;
    std::uint32_t window;
    double theta;
  };
  for (const ClassCfg cfg :
       {ClassCfg{256, 8, 1024, 1.0e4}, ClassCfg{512, 2, 1024, 1.0e6}}) {
    const std::uint32_t hd = cfg.head_dim;
    const std::uint32_t kvh = cfg.kv_heads;
    const std::string tag = cfg.head_dim == 256 ? "swa" : "full";
    std::vector<float> inv_freq(hd / 2);
    for (std::uint32_t i = 0; i < hd / 2; ++i) {
      inv_freq[i] = static_cast<float>(std::pow(
          cfg.theta, -2.0 * static_cast<double>(i) / static_cast<double>(hd)));
    }
    t::HipBuffer<float> d_inv(inv_freq.size());
    t::Upload(&d_inv, inv_freq);

    // QknormRope: 3 tokens x kHeads.
    {
      const std::uint32_t tokens = 3;
      const auto q = t::MakeValues(
          static_cast<std::size_t>(tokens) * kHeads * hd, 7, 1.0F);
      const auto gamma = t::MakeValues(hd, 11, 1.0F);
      t::HipBuffer<float> d_q(q.size()), d_g(gamma.size()), d_o(q.size());
      t::Upload(&d_q, q);
      t::Upload(&d_g, gamma);
      q::QknormRope(d_q.get(), d_g.get(), d_o.get(), tokens, kHeads, hd,
                    d_inv.get(), 17, nullptr);
      t::CheckHip(hipDeviceSynchronize(), "QknormRope synchronization");
      const auto got = t::Download(&d_o, q.size());
      std::vector<float> ref(q.size());
      for (std::uint32_t tok = 0; tok < tokens; ++tok) {
        for (std::uint32_t h = 0; h < kHeads; ++h) {
          const float* x =
              q.data() + (static_cast<std::size_t>(tok) * kHeads + h) * hd;
          float* r =
              ref.data() + (static_cast<std::size_t>(tok) * kHeads + h) * hd;
          double ss = 0.0;
          for (std::uint32_t i = 0; i < hd; ++i) {
            ss += static_cast<double>(x[i]) * x[i];
          }
          const float scale =
              static_cast<float>(1.0 / std::sqrt(ss / hd + 1.0e-6));
          for (std::uint32_t i = 0; i < hd; ++i) {
            r[i] = x[i] * scale * gamma[i];
          }
          for (std::uint32_t i = 0; i < hd / 2; ++i) {
            const double angle = static_cast<double>(17 + tok) *
                                 static_cast<double>(inv_freq[i]);
            const float c = static_cast<float>(std::cos(angle));
            const float s = static_cast<float>(std::sin(angle));
            const float a = r[i];
            const float b = r[i + hd / 2];
            r[i] = a * c - b * s;
            r[i + hd / 2] = a * s + b * c;
          }
        }
      }
      ok = Check(("QknormRope " + tag).c_str(), t::WorstAbsolute(ref, got),
                 1e-5) &&
           ok;
    }

    // KvNormRopeWrite: 3 tokens x kvh heads into an F16 cache at start=9.
    {
      const std::uint32_t tokens = 3;
      const std::uint32_t start = 9;
      const auto k =
          t::MakeValues(static_cast<std::size_t>(tokens) * kvh * hd, 13, 1.0F);
      const auto v =
          t::MakeValues(static_cast<std::size_t>(tokens) * kvh * hd, 17, 1.0F);
      const auto gamma = t::MakeValues(hd, 19, 1.0F);
      t::HipBuffer<float> d_k(k.size()), d_v(v.size()), d_g(gamma.size());
      t::Upload(&d_k, k);
      t::Upload(&d_v, v);
      t::Upload(&d_g, gamma);
      const std::size_t cache =
          static_cast<std::size_t>(start + tokens) * kvh * hd;
      t::HipBuffer<__half> d_kc(cache), d_vc(cache);
      q::KvNormRopeWrite(d_k.get(), d_v.get(), d_g.get(), d_kc.get(),
                         d_vc.get(), start, tokens, kvh, hd, d_inv.get(),
                         nullptr);
      t::CheckHip(hipDeviceSynchronize(), "KvNormRopeWrite synchronization");
      std::vector<__half> kc(cache), vc(cache);
      t::CheckHip(
          hipMemcpy(kc.data(), d_kc.get(), d_kc.bytes(), hipMemcpyDeviceToHost),
          "KvNormRopeWrite K download");
      t::CheckHip(
          hipMemcpy(vc.data(), d_vc.get(), d_vc.bytes(), hipMemcpyDeviceToHost),
          "KvNormRopeWrite V download");
      std::vector<float> ref_k(k.size()), ref_v(v.size());
      for (std::uint32_t tok = 0; tok < tokens; ++tok) {
        for (std::uint32_t h = 0; h < kvh; ++h) {
          const std::size_t o = (static_cast<std::size_t>(tok) * kvh + h) * hd;
          for (int which = 0; which < 2; ++which) {
            const float* x = which == 0 ? k.data() + o : v.data() + o;
            float* r = which == 0 ? ref_k.data() + o : ref_v.data() + o;
            double ss = 0.0;
            for (std::uint32_t i = 0; i < hd; ++i) {
              ss += static_cast<double>(x[i]) * x[i];
            }
            const float scale =
                static_cast<float>(1.0 / std::sqrt(ss / hd + 1.0e-6));
            for (std::uint32_t i = 0; i < hd; ++i) {
              r[i] = x[i] * scale * (which == 0 ? gamma[i] : 1.0F);
            }
            if (which == 0) {
              for (std::uint32_t i = 0; i < hd / 2; ++i) {
                const double angle = static_cast<double>(start + tok) *
                                     static_cast<double>(inv_freq[i]);
                const float c = static_cast<float>(std::cos(angle));
                const float s = static_cast<float>(std::sin(angle));
                const float a = r[i];
                const float b = r[i + hd / 2];
                r[i] = a * c - b * s;
                r[i + hd / 2] = a * s + b * c;
              }
            }
            __half* dst = which == 0 ? kc.data() : vc.data();
            const std::size_t co =
                (static_cast<std::size_t>(start) + tok) * kvh * hd +
                static_cast<std::size_t>(h) * hd;
            for (std::uint32_t i = 0; i < hd; ++i) {
              dst[co + i] = __float2half(r[i]);
              r[i] = __half2float(dst[co + i]);
            }
          }
        }
      }
      std::vector<float> got_k(k.size()), got_v(v.size());
      for (std::size_t i = 0; i < k.size(); ++i) {
        const std::size_t co = (static_cast<std::size_t>(start) +
                                i / (static_cast<std::size_t>(kvh) * hd)) *
                                   kvh * hd +
                               i % (static_cast<std::size_t>(kvh) * hd);
        got_k[i] = __half2float(kc[co]);
        got_v[i] = __half2float(vc[co]);
      }
      ok = Check(("KvNormRopeWrite K " + tag).c_str(),
                 t::WorstRelative(ref_k, got_k), 1e-6) &&
           ok;
      ok = Check(("KvNormRopeWrite V " + tag).c_str(),
                 t::WorstRelative(ref_v, got_v), 1e-6) &&
           ok;
    }

    // Decode attention over an F16 cache: dense and windowed.
    {
      const std::uint32_t n_keys = 1300;
      const auto q =
          t::MakeValues(static_cast<std::size_t>(kHeads) * hd, 23, 1.0F);
      const auto kvals =
          t::MakeValues(static_cast<std::size_t>(n_keys) * kvh * hd, 29, 1.0F);
      const auto vvals =
          t::MakeValues(static_cast<std::size_t>(n_keys) * kvh * hd, 31, 1.0F);
      std::vector<__half> k16(kvals.size()), v16(vvals.size());
      for (std::size_t i = 0; i < kvals.size(); ++i) {
        k16[i] = __float2half(kvals[i]);
        v16[i] = __float2half(vvals[i]);
      }
      t::HipBuffer<float> d_q(q.size());
      t::Upload(&d_q, q);
      t::HipBuffer<__half> d_kc(k16.size()), d_vc(v16.size());
      t::CheckHip(hipMemcpy(d_kc.get(), k16.data(), d_kc.bytes(),
                            hipMemcpyHostToDevice),
                  "decode K upload");
      t::CheckHip(hipMemcpy(d_vc.get(), v16.data(), d_vc.bytes(),
                            hipMemcpyHostToDevice),
                  "decode V upload");
      t::HipBuffer<float> d_out(q.size()),
          d_scratch(static_cast<std::size_t>(kHeads) * 32 * (hd + 2));
      for (const std::uint32_t lo : {0U, n_keys - cfg.window}) {
        q::AttentionDecode(d_q.get(), d_kc.get(), d_vc.get(), d_out.get(),
                           d_scratch.get(), n_keys, lo, kHeads, kvh, hd,
                           nullptr);
        t::CheckHip(hipDeviceSynchronize(), "AttentionDecode synchronization");
        const auto got = t::Download(&d_out, q.size());
        std::vector<float> ref(q.size());
        for (std::uint32_t h = 0; h < kHeads; ++h) {
          const std::uint32_t g = h / (kHeads / kvh);
          const float* qh = q.data() + static_cast<std::size_t>(h) * hd;
          std::vector<double> scores(n_keys - lo);
          double max_s = -INFINITY;
          for (std::uint32_t j = lo; j < n_keys; ++j) {
            const std::size_t o = static_cast<std::size_t>(j) * kvh * hd +
                                  static_cast<std::size_t>(g) * hd;
            double dot = 0.0;
            for (std::uint32_t i = 0; i < hd; ++i) {
              dot += static_cast<double>(qh[i]) * __half2float(k16[o + i]);
            }
            scores[j - lo] = dot;
            max_s = std::max(max_s, dot);
          }
          double sum = 0.0;
          for (auto& s : scores) {
            s = std::exp(s - max_s);
            sum += s;
          }
          for (std::uint32_t j = lo; j < n_keys; ++j) {
            const std::size_t o = static_cast<std::size_t>(j) * kvh * hd +
                                  static_cast<std::size_t>(g) * hd;
            const double w = scores[j - lo] / sum;
            for (std::uint32_t i = 0; i < hd; ++i) {
              ref[static_cast<std::size_t>(h) * hd + i] +=
                  static_cast<float>(w * __half2float(v16[o + i]));
            }
          }
        }
        ok = Check(("AttentionDecode " + tag + (lo ? " windowed" : "")).c_str(),
                   t::WorstRelativeToScale(ref, got), 1e-3) &&
             ok;
      }
    }

    // Causal windowed prefill: 40 tokens starting at 5, window 64.
    {
      const std::uint32_t start = 5;
      const std::uint32_t tokens = 40;
      const std::uint32_t window = 64;
      const auto q = t::MakeValues(
          static_cast<std::size_t>(tokens) * kHeads * hd, 37, 1.0F);
      const auto kvals = t::MakeValues(
          static_cast<std::size_t>(start + tokens) * kvh * hd, 41, 1.0F);
      const auto vvals = t::MakeValues(
          static_cast<std::size_t>(start + tokens) * kvh * hd, 43, 1.0F);
      std::vector<__half> k16(kvals.size()), v16(vvals.size());
      for (std::size_t i = 0; i < kvals.size(); ++i) {
        k16[i] = __float2half(kvals[i]);
        v16[i] = __float2half(vvals[i]);
      }
      t::HipBuffer<float> d_q(q.size());
      t::Upload(&d_q, q);
      t::HipBuffer<__half> d_kc(k16.size()), d_vc(v16.size());
      t::CheckHip(hipMemcpy(d_kc.get(), k16.data(), d_kc.bytes(),
                            hipMemcpyHostToDevice),
                  "prefill K upload");
      t::CheckHip(hipMemcpy(d_vc.get(), v16.data(), v16.size() * sizeof(__half),
                            hipMemcpyHostToDevice),
                  "prefill V upload");
      t::HipBuffer<float> d_out(q.size());
      q::AttentionPrefill(d_q.get(), d_kc.get(), d_vc.get(), d_out.get(), start,
                          tokens, kHeads, kvh, hd, window, nullptr);
      t::CheckHip(hipDeviceSynchronize(), "AttentionPrefill synchronization");
      const auto got = t::Download(&d_out, q.size());
      std::vector<float> ref(q.size());
      for (std::uint32_t tok = 0; tok < tokens; ++tok) {
        const std::uint32_t causal = start + tok + 1;
        const std::uint32_t lo = causal > window ? causal - window : 0;
        for (std::uint32_t h = 0; h < kHeads; ++h) {
          const std::uint32_t g = h / (kHeads / kvh);
          const float* qh =
              q.data() + (static_cast<std::size_t>(tok) * kHeads + h) * hd;
          std::vector<double> scores(causal - lo);
          double max_s = -INFINITY;
          for (std::uint32_t j = lo; j < causal; ++j) {
            const std::size_t o = static_cast<std::size_t>(j) * kvh * hd +
                                  static_cast<std::size_t>(g) * hd;
            double dot = 0.0;
            for (std::uint32_t i = 0; i < hd; ++i) {
              dot += static_cast<double>(qh[i]) * __half2float(k16[o + i]);
            }
            scores[j - lo] = dot;
            max_s = std::max(max_s, dot);
          }
          double sum = 0.0;
          for (auto& s : scores) {
            s = std::exp(s - max_s);
            sum += s;
          }
          for (std::uint32_t j = lo; j < causal; ++j) {
            const std::size_t o = static_cast<std::size_t>(j) * kvh * hd +
                                  static_cast<std::size_t>(g) * hd;
            const double w = scores[j - lo] / sum;
            for (std::uint32_t i = 0; i < hd; ++i) {
              ref[(static_cast<std::size_t>(tok) * kHeads + h) * hd + i] +=
                  static_cast<float>(w * __half2float(v16[o + i]));
            }
          }
        }
      }
      ok = Check(("AttentionPrefill " + tag).c_str(),
                 t::WorstRelativeToScale(ref, got), 1e-3) &&
           ok;
    }
  }
  return ok;
}

int main() {
  bool ok = true;
  ok = TestGemv() && ok;
  ok = TestEmbedRow() && ok;
  ok = TestGemvGrouped() && ok;
  ok = TestGeglu() && ok;
  ok = TestRouterTopK() && ok;
  ok = TestNormsAndAdd() && ok;
  ok = TestGemm() && ok;
  ok = TestRoutedPipeline() && ok;
  ok = TestAttention() && ok;
  if (!ok) {
    std::cerr << "gemma4 kernel parity FAILED\n";
    return 1;
  }
  std::cout << "gemma4 kernel parity OK\n";
  return 0;
}