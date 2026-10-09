// Fused single-token operator checks: every kernel in kernels.hpp is run on
// synthetic tensors and compared against a host reference that mirrors the
// CPU oracle formulas (reference.cpp / cpu_ops.cpp) in double. No model
// artifact is required; the real-artifact wiring is exercised by the forward
// check.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "src/models/ling3_flash/kernels/rocm/kernels.hpp"
#include "tests/models/ling3_flash/hip_test.hpp"

namespace q = gufo::models::ling3_flash::rocm;
namespace t = gufo::tests::ling3_flash;

namespace {

constexpr float kEps = 1e-6F;
constexpr float kWeightsSumEps = 6.103515625e-5F;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

bool Check(const char* name, double worst, double tolerance) {
  std::cout << name << " worst error " << worst << '\n';
  if (!(worst <= tolerance)) {
    std::cout << name << " FAILED (tolerance " << tolerance << ")\n";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------- RmsNorm

void HostRmsNorm(const std::vector<float>& x, const std::vector<float>* w,
                 float eps, std::vector<double>* out) {
  double ss = 0.0;
  for (float v : x) {
    ss += static_cast<double>(v) * v;
  }
  const double scale = 1.0 / std::sqrt(ss / static_cast<double>(x.size()) +
                                       static_cast<double>(eps));
  out->resize(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    (*out)[i] = static_cast<double>(x[i]) * scale *
                (w != nullptr ? static_cast<double>((*w)[i]) : 1.0);
  }
}

bool TestRmsNorm() {
  constexpr std::uint32_t n = 2560;
  const auto x = t::MakeValues(n, 0x11110001U, 2.0F);
  const auto w = t::MakeValues(n, 0x11110002U, 1.0F, 1.0F);
  std::vector<double> ref;
  HostRmsNorm(x, &w, kEps, &ref);
  t::HipBuffer<float> d_x(n), d_w(n), d_out(n);
  t::Upload(&d_x, x);
  t::Upload(&d_w, w);
  q::RmsNorm(d_x.get(), d_w.get(), kEps, d_out.get(), n, nullptr);
  const auto got = t::Download(&d_out, n);
  std::vector<float> ref_f(ref.begin(), ref.end());
  return Check("RmsNorm", t::WorstRelative(ref_f, got), 2e-5);
}

// ------------------------------------------------------- Add/Concat/Sigmoid

bool TestAddConcatSigmoid() {
  constexpr std::uint32_t n = 2560;
  const auto a = t::MakeValues(n, 0x22220001U, 3.0F);
  const auto b = t::MakeValues(n, 0x22220002U, 3.0F);
  t::HipBuffer<float> d_a(n), d_b(n), d_add(n), d_cat(2 * n);
  t::Upload(&d_a, a);
  t::Upload(&d_b, b);
  bool ok = true;

  q::Add(d_a.get(), d_b.get(), d_add.get(), n, nullptr);
  std::vector<float> ref(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    ref[i] = a[i] + b[i];
  }
  ok = Check("Add", t::WorstRelative(ref, t::Download(&d_add, n)), 0.0) && ok;

  q::Concat2(d_a.get(), d_b.get(), d_cat.get(), n, nullptr);
  auto got = t::Download(&d_cat, 2 * n);
  std::vector<float> ref_cat(2 * n);
  std::copy(a.begin(), a.end(), ref_cat.begin());
  std::copy(b.begin(), b.end(), ref_cat.begin() + n);
  ok = Check("Concat2", t::WorstRelative(ref_cat, got), 0.0) && ok;

  q::SigmoidInPlace(d_a.get(), n, nullptr);
  std::vector<float> ref_sig(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    ref_sig[i] = static_cast<float>(t::SigmoidD(a[i]));
  }
  ok = Check("SigmoidInPlace", t::WorstAbsolute(ref_sig, t::Download(&d_a, n)),
             1e-6) &&
       ok;
  return ok;
}

// --------------------------------------------------------------------- Rope

bool TestRopeOne(std::uint32_t heads, std::uint32_t head_dim, std::uint32_t rot,
                 std::uint32_t pos, float theta, std::uint32_t seed) {
  const std::size_t count = static_cast<std::size_t>(heads) * head_dim;
  const auto x = t::MakeValues(count, seed, 2.0F);
  std::vector<float> ref = x;
  const std::uint32_t pairs = rot / 2U;
  for (std::uint32_t h = 0; h < heads; ++h) {
    for (std::uint32_t i = 0; i < pairs; ++i) {
      const float freq = static_cast<float>(
          std::pow(static_cast<double>(theta),
                   -2.0 * static_cast<double>(i) / static_cast<double>(rot)));
      const float angle = static_cast<float>(pos) * freq;
      const float c = static_cast<float>(std::cos(static_cast<double>(angle)));
      const float s = static_cast<float>(std::sin(static_cast<double>(angle)));
      float* v = ref.data() + static_cast<std::size_t>(h) * head_dim;
      const float a = v[2 * i];
      const float b = v[2 * i + 1];
      v[2 * i] = a * c - b * s;
      v[2 * i + 1] = a * s + b * c;
    }
  }
  t::HipBuffer<float> d_x(count);
  t::Upload(&d_x, x);
  q::Rope(d_x.get(), heads, head_dim, rot, pos, theta, nullptr);
  const auto got = t::Download(&d_x, count);
  const double worst = t::WorstRelative(ref, got);
  const char* name = heads > 1 ? "RopeQ" : "RopeK";
  // The oracle quantizes each frequency to float; one float ulp in freq is
  // amplified by the position into the angle, so agreement with any
  // independent pow implementation is bounded near 1e-4 at large positions.
  return Check(name, worst, 2e-4);
}

bool TestRope() {
  bool ok = true;
  ok = TestRopeOne(32, 64, 64, 1234, 6.0e6F, 0x33330001U) && ok;
  ok = TestRopeOne(1, 64, 64, 262143, 6.0e6F, 0x33330002U) && ok;
  ok = TestRopeOne(32, 192, 64, 7, 6.0e6F, 0x33330003U) && ok;
  return ok;
}

// ----------------------------------------------------------------- ConvSilu

bool TestConvSilu() {
  constexpr std::uint32_t inner = 4096;
  constexpr std::uint32_t kern = 4;
  const auto cw = t::MakeValues(inner * kern, 0x44440001U, 1.0F);
  const auto proj = t::MakeValues(inner, 0x44440002U, 2.0F);
  const auto hist = t::MakeValues(static_cast<std::size_t>(kern - 1) * inner,
                                  0x44440003U, 2.0F);
  std::vector<float> ref_out(inner);
  std::vector<float> ref_hist = hist;
  for (std::uint32_t ch = 0; ch < inner; ++ch) {
    float acc = cw[ch * kern + kern - 1] * proj[ch];
    for (std::uint32_t tt = 0; tt + 1 < kern; ++tt) {
      acc +=
          cw[ch * kern + tt] * hist[static_cast<std::size_t>(tt) * inner + ch];
    }
    const float value = acc / (1.0F + std::exp(-acc));
    ref_out[ch] = value;
    for (std::uint32_t tt = 0; tt + 2 < kern; ++tt) {
      ref_hist[static_cast<std::size_t>(tt) * inner + ch] =
          hist[static_cast<std::size_t>(tt + 1) * inner + ch];
    }
    ref_hist[static_cast<std::size_t>(kern - 2) * inner + ch] = value;
  }
  t::HipBuffer<float> d_cw(inner * kern), d_proj(inner),
      d_hist(static_cast<std::size_t>(kern - 1) * inner), d_out(inner);
  t::Upload(&d_cw, cw);
  t::Upload(&d_proj, proj);
  t::Upload(&d_hist, hist);
  q::ConvSilu(d_cw.get(), d_proj.get(), d_hist.get(), d_out.get(), inner, kern,
              nullptr);
  bool ok = Check("ConvSilu out",
                  t::WorstAbsolute(ref_out, t::Download(&d_out, inner)), 1e-5);
  ok = Check("ConvSilu history",
             t::WorstAbsolute(ref_hist, t::Download(&d_hist, ref_hist.size())),
             1e-5) &&
       ok;
  return ok;
}

// ------------------------------------------------------------ KdaNormHeads

bool TestKdaNormHeads() {
  constexpr std::uint32_t heads = 32;
  constexpr std::uint32_t d = 128;
  const std::size_t n = static_cast<std::size_t>(heads) * d;
  const auto q_in = t::MakeValues(n, 0x55550001U, 2.0F);
  const auto k_in = t::MakeValues(n, 0x55550002U, 2.0F);
  const float q_scale = 1.0F / std::sqrt(static_cast<float>(d));
  auto l2 = [q_scale](const std::vector<float>& in, bool scale) {
    std::vector<float> out = in;
    for (std::uint32_t h = 0; h < heads; ++h) {
      float* v = out.data() + static_cast<std::size_t>(h) * d;
      double ss = 0.0;
      for (std::uint32_t i = 0; i < d; ++i) {
        ss += static_cast<double>(v[i]) * v[i];
      }
      const float s = 1.0F / std::sqrt(static_cast<float>(ss) + kEps);
      for (std::uint32_t i = 0; i < d; ++i) {
        v[i] *= s * (scale ? q_scale : 1.0F);
      }
    }
    return out;
  };
  const auto ref_q = l2(q_in, true);
  const auto ref_k = l2(k_in, false);
  t::HipBuffer<float> d_q(n), d_k(n);
  t::Upload(&d_q, q_in);
  t::Upload(&d_k, k_in);
  q::KdaNormHeads(d_q.get(), d_k.get(), heads, d, kEps, q_scale, nullptr);
  bool ok = Check("KdaNormHeads q",
                  t::WorstRelative(ref_q, t::Download(&d_q, n)), 1e-5);
  ok = Check("KdaNormHeads k", t::WorstRelative(ref_k, t::Download(&d_k, n)),
             1e-5) &&
       ok;
  return ok;
}

// -------------------------------------------------------------- KdaDelta

bool TestKdaDelta() {
  constexpr std::uint32_t heads = 32;
  constexpr std::uint32_t d = 128;
  constexpr float kGateLb = -5.0F;
  const std::size_t inner = static_cast<std::size_t>(heads) * d;
  const auto q_in = t::MakeValues(inner, 0x66660001U, 1.0F);
  const auto k_in = t::MakeValues(inner, 0x66660002U, 1.0F);
  const auto v_in = t::MakeValues(inner, 0x66660003U, 2.0F);
  const auto f_in = t::MakeValues(inner, 0x66660004U, 3.0F);
  const auto beta = t::MakeValues(heads, 0x66660005U, 3.0F);
  const auto a = t::MakeValues(heads, 0x66660006U, 1.0F, -1.0F);
  const auto norm_w = t::MakeValues(d, 0x66660007U, 1.0F, 1.0F);
  const auto g = t::MakeValues(inner, 0x66660008U, 3.0F);
  const auto state_in =
      t::MakeValues(static_cast<std::size_t>(heads) * d * d, 0x66660009U, 1.0F);

  // Host reference: float state storage, double accumulators, exactly the
  // oracle's KdaAttention recurrence plus the per-head output norm.
  std::vector<float> state_ref = state_in;
  std::vector<float> ref_out(inner);
  for (std::uint32_t h = 0; h < heads; ++h) {
    const float* qh = q_in.data() + static_cast<std::size_t>(h) * d;
    const float* kh = k_in.data() + static_cast<std::size_t>(h) * d;
    const float* vh = v_in.data() + static_cast<std::size_t>(h) * d;
    float* S = state_ref.data() + static_cast<std::size_t>(h) * d * d;
    const float b = static_cast<float>(t::SigmoidD(beta[h]));
    std::vector<float> u(d);
    // The decay gate addresses the key axis i, broadcast across value rows.
    std::vector<float> decay(d);
    for (std::uint32_t i = 0; i < d; ++i) {
      decay[i] = static_cast<float>(
          std::exp(static_cast<double>(kGateLb) *
                   t::SigmoidD(static_cast<double>(a[h]) *
                               f_in[static_cast<std::size_t>(h) * d + i])));
    }
    for (std::uint32_t j = 0; j < d; ++j) {
      float* row = S + static_cast<std::size_t>(j) * d;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < d; ++i) {
        row[i] *= decay[i];
        acc += static_cast<double>(row[i]) * kh[i];
      }
      u[j] = static_cast<float>(acc);
    }
    std::vector<float> o(d);
    for (std::uint32_t j = 0; j < d; ++j) {
      const float delta = (vh[j] - u[j]) * b;
      float* row = S + static_cast<std::size_t>(j) * d;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < d; ++i) {
        row[i] += delta * kh[i];
        acc += static_cast<double>(row[i]) * qh[i];
      }
      o[j] = static_cast<float>(acc);
    }
    double ss = 0.0;
    for (std::uint32_t j = 0; j < d; ++j) {
      ss += static_cast<double>(o[j]) * o[j];
    }
    const float scale =
        1.0F /
        std::sqrt(static_cast<float>(ss / static_cast<double>(d)) + kEps);
    for (std::uint32_t j = 0; j < d; ++j) {
      ref_out[static_cast<std::size_t>(h) * d + j] =
          o[j] * scale * norm_w[j] *
          static_cast<float>(
              t::SigmoidD(g[static_cast<std::size_t>(h) * d + j]));
    }
  }

  t::HipBuffer<float> d_q(inner), d_k(inner), d_v(inner), d_f(inner),
      d_beta(heads), d_a(heads), d_norm(d), d_g(inner),
      d_state(static_cast<std::size_t>(heads) * d * d), d_out(inner);
  t::Upload(&d_q, q_in);
  t::Upload(&d_k, k_in);
  t::Upload(&d_v, v_in);
  t::Upload(&d_f, f_in);
  t::Upload(&d_beta, beta);
  t::Upload(&d_a, a);
  t::Upload(&d_norm, norm_w);
  t::Upload(&d_g, g);
  t::Upload(&d_state, state_in);
  q::KdaDelta(d_q.get(), d_k.get(), d_v.get(), d_f.get(), d_beta.get(),
              d_a.get(), d_norm.get(), d_g.get(), d_state.get(), d_out.get(),
              heads, d, kEps, kGateLb, nullptr);
  bool ok = Check("KdaDelta out",
                  t::WorstRelative(ref_out, t::Download(&d_out, inner)), 2e-4);
  ok = Check(
           "KdaDelta state",
           t::WorstRelative(state_ref, t::Download(&d_state, state_ref.size())),
           2e-4) &&
       ok;
  return ok;
}

// ------------------------------------------------------------- RouterTopK

bool TestRouterTopK() {
  constexpr std::uint32_t n_exp = 512;
  constexpr std::uint32_t groups = 8;
  constexpr std::uint32_t groups_used = 4;
  constexpr std::uint32_t used = 8;
  constexpr float kScale = 2.5F;
  const auto logits = t::MakeValues(n_exp, 0x77770001U, 4.0F);
  const auto bias = t::MakeValues(n_exp, 0x77770002U, 0.5F);

  std::vector<float> probs(n_exp);
  std::vector<float> selection(n_exp);
  for (std::uint32_t e = 0; e < n_exp; ++e) {
    probs[e] = static_cast<float>(t::SigmoidD(logits[e]));
    selection[e] = probs[e] + bias[e];
  }
  const std::uint32_t per_group = n_exp / groups;
  std::vector<float> group_scores(groups);
  for (std::uint32_t g = 0; g < groups; ++g) {
    float best = kNegInf;
    float second = kNegInf;
    for (std::uint32_t i = 0; i < per_group; ++i) {
      const float value =
          selection[static_cast<std::size_t>(g) * per_group + i];
      if (value >= best) {
        second = best;
        best = value;
      } else if (value > second) {
        second = value;
      }
    }
    group_scores[g] = best + second;
  }
  std::vector<bool> keep(groups, false);
  for (std::uint32_t s = 0; s < groups_used; ++s) {
    std::uint32_t best_g = 0;
    float best = kNegInf;
    for (std::uint32_t g = 0; g < groups; ++g) {
      if (!keep[g] && group_scores[g] > best) {
        best = group_scores[g];
        best_g = g;
      }
    }
    keep[best_g] = true;
  }
  for (std::uint32_t g = 0; g < groups; ++g) {
    if (!keep[g]) {
      for (std::uint32_t i = 0; i < per_group; ++i) {
        selection[static_cast<std::size_t>(g) * per_group + i] = kNegInf;
      }
    }
  }
  std::vector<std::int32_t> ref_ids(used);
  std::vector<float> ref_weights(used);
  double sum = 0.0;
  for (std::uint32_t s = 0; s < used; ++s) {
    std::uint32_t best_e = 0;
    float best = kNegInf;
    for (std::uint32_t e = 0; e < n_exp; ++e) {
      if (selection[e] > best) {
        best = selection[e];
        best_e = e;
      }
    }
    selection[best_e] = kNegInf;
    ref_ids[s] = static_cast<std::int32_t>(best_e);
    ref_weights[s] = probs[best_e];
    sum += probs[best_e];
  }
  const float norm = 1.0F / std::max(static_cast<float>(sum), kWeightsSumEps);
  for (float& w : ref_weights) {
    w *= norm * kScale;
  }

  t::HipBuffer<float> d_logits(n_exp), d_bias(n_exp);
  t::HipBuffer<std::int32_t> d_ids(used);
  t::HipBuffer<float> d_weights(used);
  t::Upload(&d_logits, logits);
  t::Upload(&d_bias, bias);
  q::RouterTopK(d_logits.get(), d_bias.get(), n_exp, groups, groups_used, used,
                kScale, d_ids.get(), d_weights.get(), nullptr);
  std::vector<std::int32_t> ids(used);
  t::CheckHip(
      hipMemcpy(ids.data(), d_ids.get(), d_ids.bytes(), hipMemcpyDeviceToHost),
      "download ids");
  std::vector<float> weights(used);
  t::CheckHip(hipMemcpy(weights.data(), d_weights.get(), d_weights.bytes(),
                        hipMemcpyDeviceToHost),
              "download weights");
  bool ids_ok = ids == ref_ids;
  std::cout << "RouterTopK ids " << (ids_ok ? "match" : "MISMATCH") << '\n';
  bool ok = ids_ok;
  ok = Check("RouterTopK weights", t::WorstRelative(ref_weights, weights),
             1e-5) &&
       ok;
  return ok;
}

// ---------------------------------------------------------- SwigluClamped

bool TestSwigluOne(float limit, std::uint32_t seed) {
  constexpr std::uint32_t n = 6144;
  const auto gate = t::MakeValues(n, seed, 8.0F);
  const auto up = t::MakeValues(n, seed + 1U, 8.0F);
  std::vector<float> ref(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    if (limit > 1e-6F) {
      const float u = std::min(std::max(up[i], -limit), limit);
      const float g = std::min(static_cast<float>(t::SiluD(gate[i])), limit);
      ref[i] = g * u;
    } else {
      ref[i] = static_cast<float>(t::SiluD(gate[i])) * up[i];
    }
  }
  t::HipBuffer<float> d_gate(n), d_up(n), d_out(n);
  t::Upload(&d_gate, gate);
  t::Upload(&d_up, up);
  q::SwigluClamped(d_gate.get(), d_up.get(), d_out.get(), n, limit, nullptr);
  const char* name = limit > 0.0F ? "SwigluClamped" : "Swiglu";
  return Check(name, t::WorstRelative(ref, t::Download(&d_out, n)), 1e-5);
}

bool TestSwiglu() {
  return TestSwigluOne(7.0F, 0x88880001U) && TestSwigluOne(0.0F, 0x88880002U);
}

}  // namespace

int main() {
  try {
    bool ok = true;
    ok = TestRmsNorm() && ok;
    ok = TestAddConcatSigmoid() && ok;
    ok = TestRope() && ok;
    ok = TestConvSilu() && ok;
    ok = TestKdaNormHeads() && ok;
    ok = TestKdaDelta() && ok;
    ok = TestRouterTopK() && ok;
    ok = TestSwiglu() && ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}