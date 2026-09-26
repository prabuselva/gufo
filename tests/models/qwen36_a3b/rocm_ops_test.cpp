#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "src/models/qwen36_a3b/kernels/rocm/kernels.hpp"
#include "tests/models/qwen36_a3b/hip_test.hpp"

namespace q = gufo::models::qwen36_a3b::rocm;
namespace t = gufo::tests::qwen36_a3b;

namespace {

// Model geometry for the elementwise, router and MoE-combine operators.
constexpr std::uint32_t kHidden = 2048;
constexpr std::uint32_t kHeads = 16;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kRotary = 64;
constexpr std::uint32_t kExperts = 256;
constexpr std::uint32_t kTopK = 8;
constexpr float kRmsEps = 1e-6F;
constexpr float kTheta = 1.0e7F;

bool Check(const char* name, double worst, double tolerance) {
  std::cout << name << " worst error " << worst << '\n';
  if (!(worst <= tolerance)) {
    std::cerr << name << " exceeded tolerance " << tolerance << '\n';
    return false;
  }
  return true;
}

bool TestRmsNorm() {
  constexpr std::uint32_t rows = 5;
  const auto x = t::MakeValues(static_cast<std::size_t>(rows) * kHidden,
                               0x1234ABCDU, 2.0F);
  const auto gamma = t::MakeValues(kHidden, 0xBEEF0001U, 0.5F, 1.0F);
  t::HipBuffer<float> d_x(x.size());
  t::HipBuffer<float> d_gamma(gamma.size());
  t::HipBuffer<float> d_out(x.size());
  t::Upload(&d_x, x);
  t::Upload(&d_gamma, gamma);
  q::RmsNormRows(d_x.get(), d_gamma.get(), d_out.get(), rows, kHidden, kRmsEps,
                 nullptr);
  t::CheckHip(hipDeviceSynchronize(), "RmsNorm synchronization");
  const auto got = t::Download(&d_out, x.size());
  std::vector<float> ref(x.size());
  for (std::uint32_t r = 0; r < rows; ++r) {
    const float* xr = x.data() + static_cast<std::size_t>(r) * kHidden;
    double ss = 0.0;
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      ss += static_cast<double>(xr[i]) * xr[i];
    }
    const float scale =
        1.0F / std::sqrt(static_cast<float>(ss / kHidden) + kRmsEps);
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      ref[static_cast<std::size_t>(r) * kHidden + i] = xr[i] * scale * gamma[i];
    }
  }
  return Check("RmsNormRows", t::WorstRelative(ref, got), 1e-5);
}

bool TestRope() {
  constexpr std::uint32_t rows = 3;
  const auto x = t::MakeValues(
      static_cast<std::size_t>(rows) * kHeads * kHeadDim, 0x0BADF00DU, 1.0F);
  const std::vector<std::uint32_t> pos = {0U, 1U, 4096U};
  t::HipBuffer<float> d_x(x.size());
  t::HipBuffer<std::uint32_t> d_pos(pos.size());
  t::Upload(&d_x, x);
  t::CheckHip(
      hipMemcpy(d_pos.get(), pos.data(), d_pos.bytes(), hipMemcpyHostToDevice),
      "upload positions");
  q::Rope(d_x.get(), d_pos.get(), rows, kHeads, kHeadDim, kRotary, kTheta,
          nullptr);
  t::CheckHip(hipDeviceSynchronize(), "Rope synchronization");
  const auto got = t::Download(&d_x, x.size());
  std::vector<float> ref = x;
  const std::uint32_t half = kRotary / 2;
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      float* v =
          ref.data() + (static_cast<std::size_t>(r) * kHeads + h) * kHeadDim;
      for (std::uint32_t i = 0; i < half; ++i) {
        const float freq = std::pow(kTheta, -2.0F * static_cast<float>(i) /
                                                static_cast<float>(kRotary));
        const float angle = static_cast<float>(pos[r]) * freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        const float a = v[i];
        const float b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
      }
    }
  }
  // The kernel rotates with float sincosf/powf (the production convention); the
  // oracle uses double. Float argument reduction at large positions bounds the
  // agreement to ~1e-4 absolute on O(1) outputs.
  return Check("Rope", t::WorstAbsolute(ref, got), 1e-3);
}

bool TestSwiglu() {
  constexpr std::size_t count = 4096;
  const auto gate = t::MakeValues(count, 0xC0FFEE11U, 3.0F);
  const auto up = t::MakeValues(count, 0xFEEDFACEU, 2.0F);
  t::HipBuffer<float> d_gate(gate.size());
  t::HipBuffer<float> d_up(up.size());
  t::Upload(&d_gate, gate);
  t::Upload(&d_up, up);
  q::Swiglu(d_gate.get(), d_up.get(), count, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "Swiglu synchronization");
  const auto got = t::Download(&d_gate, count);
  std::vector<float> ref(count);
  for (std::size_t i = 0; i < count; ++i) {
    ref[i] = static_cast<float>(t::SiluD(gate[i]) * up[i]);
  }
  return Check("Swiglu", t::WorstRelative(ref, got), 1e-5);
}

bool TestSigmoidMul() {
  constexpr std::size_t count = 4096;
  const auto x = t::MakeValues(count, 0x13579BDFU, 1.0F);
  const auto g = t::MakeValues(count, 0x2468ACE0U, 4.0F);
  t::HipBuffer<float> d_x(x.size());
  t::HipBuffer<float> d_g(g.size());
  t::Upload(&d_x, x);
  t::Upload(&d_g, g);
  q::SigmoidMul(d_x.get(), d_g.get(), count, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "SigmoidMul synchronization");
  const auto got = t::Download(&d_x, count);
  std::vector<float> ref(count);
  for (std::size_t i = 0; i < count; ++i) {
    ref[i] = static_cast<float>(x[i] * t::SigmoidD(g[i]));
  }
  return Check("SigmoidMul", t::WorstRelative(ref, got), 1e-5);
}

bool TestRouterTopK() {
  constexpr std::uint32_t tokens = 6;
  const auto logits = t::MakeValues(static_cast<std::size_t>(tokens) * kExperts,
                                    0xDEADBEEFU, 4.0F);
  t::HipBuffer<float> d_logits(logits.size());
  t::HipBuffer<std::int32_t> d_ids(static_cast<std::size_t>(tokens) * kTopK);
  t::HipBuffer<float> d_weights(static_cast<std::size_t>(tokens) * kTopK);
  t::Upload(&d_logits, logits);
  q::RouterTopK(d_logits.get(), kExperts, d_ids.get(), d_weights.get(), tokens,
                kExperts, kTopK, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "RouterTopK synchronization");
  std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens) * kTopK);
  t::CheckHip(
      hipMemcpy(ids.data(), d_ids.get(), d_ids.bytes(), hipMemcpyDeviceToHost),
      "download ids");
  const auto weights = t::Download(&d_weights, ids.size());

  std::vector<std::int32_t> ref_ids(ids.size());
  std::vector<float> ref_weights(ids.size());
  for (std::uint32_t token = 0; token < tokens; ++token) {
    const float* row =
        logits.data() + static_cast<std::size_t>(token) * kExperts;
    float max_logit = -std::numeric_limits<float>::infinity();
    for (std::uint32_t e = 0; e < kExperts; ++e) {
      max_logit = std::max(max_logit, row[e]);
    }
    std::vector<float> prob(kExperts);
    double denom = 0.0;
    for (std::uint32_t e = 0; e < kExperts; ++e) {
      prob[e] = static_cast<float>(std::exp(row[e] - max_logit));
      denom += prob[e];
    }
    for (float& p : prob) {
      p = static_cast<float>(p / denom);
    }
    // Repeated argmax with the lowest index winning ties, matching the kernel.
    std::vector<float> pool = prob;
    float sum = 0.0F;
    for (std::uint32_t slot = 0; slot < kTopK; ++slot) {
      int winner = 0;
      for (std::uint32_t e = 1; e < kExperts; ++e) {
        if (pool[e] > pool[winner]) {
          winner = static_cast<int>(e);
        }
      }
      ref_ids[static_cast<std::size_t>(token) * kTopK + slot] = winner;
      ref_weights[static_cast<std::size_t>(token) * kTopK + slot] =
          pool[winner];
      sum += pool[winner];
      pool[winner] = -std::numeric_limits<float>::infinity();
    }
    sum = std::max(sum, 6.103515625e-5F);
    for (std::uint32_t slot = 0; slot < kTopK; ++slot) {
      ref_weights[static_cast<std::size_t>(token) * kTopK + slot] /=
          static_cast<float>(sum);
    }
  }
  bool ok = true;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] != ref_ids[i]) {
      std::cerr << "RouterTopK id mismatch at " << i << ": " << ids[i] << " vs "
                << ref_ids[i] << '\n';
      ok = false;
    }
  }
  return Check("RouterTopK weights", t::WorstRelative(ref_weights, weights),
               1e-5) &&
         ok;
}

bool TestMoeEpilogue() {
  constexpr std::uint32_t tokens = 4;
  constexpr std::uint32_t gate_stride = kExperts;
  const std::size_t expert_count =
      static_cast<std::size_t>(tokens) * kTopK * kHidden;
  const std::size_t out_count = static_cast<std::size_t>(tokens) * kHidden;
  const auto expert_out = t::MakeValues(expert_count, 0x1234ABCDU, 1.0F);
  const auto weights = t::MakeValues(static_cast<std::size_t>(tokens) * kTopK,
                                     0xBADC0FFEU, 0.5F, 0.5F);
  const auto shared = t::MakeValues(out_count, 0xDEADBEEFU, 1.0F);
  const auto gate = t::MakeValues(
      static_cast<std::size_t>(tokens) * gate_stride, 0xC0FFEE11U, 3.0F);
  t::HipBuffer<float> d_expert(expert_count);
  t::HipBuffer<float> d_weights(weights.size());
  t::HipBuffer<float> d_shared(out_count);
  t::HipBuffer<float> d_gate(gate.size());
  t::HipBuffer<float> d_out(out_count);
  t::Upload(&d_expert, expert_out);
  t::Upload(&d_weights, weights);
  t::Upload(&d_shared, shared);
  t::Upload(&d_gate, gate);
  q::MoeEpilogue(d_expert.get(), d_weights.get(), d_shared.get(), d_gate.get(),
                 gate_stride, d_out.get(), tokens, kTopK, kHidden, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "MoeEpilogue synchronization");
  const auto got = t::Download(&d_out, out_count);
  std::vector<float> ref(out_count);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    const float sg = static_cast<float>(
        t::SigmoidD(gate[static_cast<std::size_t>(token) * gate_stride]));
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      double acc = 0.0;
      for (std::uint32_t s = 0; s < kTopK; ++s) {
        const std::size_t slot = static_cast<std::size_t>(token) * kTopK + s;
        acc +=
            static_cast<double>(weights[slot]) * expert_out[slot * kHidden + i];
      }
      acc += static_cast<double>(sg) *
             shared[static_cast<std::size_t>(token) * kHidden + i];
      ref[static_cast<std::size_t>(token) * kHidden + i] =
          static_cast<float>(acc);
    }
  }
  return Check("MoeEpilogue", t::WorstRelative(ref, got), 1e-4);
}

}  // namespace

int main() {
  try {
    bool ok = true;
    ok = TestRmsNorm() && ok;
    ok = TestRope() && ok;
    ok = TestSwiglu() && ok;
    ok = TestSigmoidMul() && ok;
    ok = TestRouterTopK() && ok;
    ok = TestMoeEpilogue() && ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}