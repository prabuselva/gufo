#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "src/models/qwen36_a3b/kernels/rocm/kernels.hpp"
#include "tests/models/qwen36_a3b/hip_test.hpp"

namespace q = gufo::models::qwen36_a3b::rocm;
namespace t = gufo::tests::qwen36_a3b;

namespace {

// The model's linear-attention geometry: 16 key heads, 32 value heads,
// 128-wide state, kernel-4 causal depthwise conv.
constexpr std::uint32_t kKHeads = 16;
constexpr std::uint32_t kVHeads = 32;
constexpr std::uint32_t kDim = 128;
constexpr std::uint32_t kKernel = 4;
constexpr std::uint32_t kKeyDim = kKHeads * kDim;
constexpr std::uint32_t kValueDim = kVHeads * kDim;
constexpr std::uint32_t kChannels = 2 * kKeyDim + kValueDim;
constexpr float kEps = 1e-6F;

bool Check(const char* name, double worst, double tolerance) {
  std::cout << name << " worst relative error " << worst << '\n';
  if (!(worst <= tolerance)) {
    std::cerr << name << " exceeded tolerance " << tolerance << '\n';
    return false;
  }
  return true;
}

bool RunCase(bool extreme_gates) {
  const std::size_t state_count =
      static_cast<std::size_t>(kVHeads) * kDim * kDim;
  const std::size_t conv_state =
      static_cast<std::size_t>(kKernel - 1) * kChannels;

  const auto qkv = t::MakeValues(kChannels, 0x1234ABCDU, 1.0F);
  const auto conv_w = t::MakeValues(
      static_cast<std::size_t>(kChannels) * kKernel, 0xDEADBEEFU, 0.5F);
  const auto z = t::MakeValues(kValueDim, 0x0BADF00DU, 2.0F);
  auto alpha = t::MakeValues(kVHeads, 0xBADC0FFEU, 2.0F);
  const auto beta = t::MakeValues(kVHeads, 0x600DCAFEU, 3.0F);
  auto a = t::MakeValues(kVHeads, 0xC0FFEE11U, 1.0F, -1.5F);
  auto dt = t::MakeValues(kVHeads, 0xFEEDFACEU, 1.0F);
  const auto norm_w = t::MakeValues(kDim, 0x13579BDFU, 0.5F, 1.0F);
  const auto history = t::MakeValues(conv_state, 0x2468ACE0U, 1.0F);
  const auto state = t::MakeValues(state_count, 0x0F1E2D3CU, 0.1F);
  if (extreme_gates) {
    // Finite gates on both sides of the softplus threshold and expf overflow.
    constexpr float gates[] = {-100.0F, 19.9F, 20.1F, 88.8F, 100.0F, 1000.0F};
    for (std::uint32_t h = 0; h < kVHeads; ++h) {
      a[h] = -0.01F;
      dt[h] = 0.0F;
      alpha[h] = gates[h % 6];
    }
  }

  t::HipBuffer<float> d_qkv(qkv.size());
  t::HipBuffer<float> d_conv_w(conv_w.size());
  t::HipBuffer<float> d_z(z.size());
  t::HipBuffer<float> d_alpha(alpha.size());
  t::HipBuffer<float> d_beta(beta.size());
  t::HipBuffer<float> d_a(a.size());
  t::HipBuffer<float> d_dt(dt.size());
  t::HipBuffer<float> d_norm_w(norm_w.size());
  t::HipBuffer<float> d_history(history.size());
  t::HipBuffer<float> d_convolved(kChannels);
  t::HipBuffer<float> d_qn(kKeyDim);
  t::HipBuffer<float> d_kn(kKeyDim);
  t::HipBuffer<float> d_attn(kValueDim);
  t::HipBuffer<float> d_state(state_count);
  t::Upload(&d_qkv, qkv);
  t::Upload(&d_conv_w, conv_w);
  t::Upload(&d_z, z);
  t::Upload(&d_alpha, alpha);
  t::Upload(&d_beta, beta);
  t::Upload(&d_a, a);
  t::Upload(&d_dt, dt);
  t::Upload(&d_norm_w, norm_w);
  t::Upload(&d_history, history);
  t::Upload(&d_state, state);

  q::GdnConv(d_qkv.get(), d_conv_w.get(), d_history.get(), d_convolved.get(),
             kChannels, kKernel, nullptr);
  q::GdnNormQk(d_convolved.get(), d_qn.get(), d_kn.get(), kKHeads, kDim, kEps,
               nullptr);
  q::GdnDelta(d_qn.get(), d_kn.get(), d_convolved.get() + 2 * kKeyDim,
              d_alpha.get(), d_beta.get(), d_a.get(), d_dt.get(), d_state.get(),
              d_attn.get(), kKHeads, kVHeads, kDim, nullptr);
  q::GdnOutNorm(d_attn.get(), d_z.get(), d_norm_w.get(), kVHeads, kDim, kEps,
                nullptr);
  t::CheckHip(hipDeviceSynchronize(), "GDN synchronization");

  const auto convolved = t::Download(&d_convolved, kChannels);
  const auto qn = t::Download(&d_qn, kKeyDim);
  const auto kn = t::Download(&d_kn, kKeyDim);
  const auto attn = t::Download(&d_attn, kValueDim);
  const auto final_state = t::Download(&d_state, state_count);

  // CPU reference mirroring the scalar oracle's LinearAttention body.
  std::vector<float> conv(kChannels);
  std::vector<float> ref_history = history;
  for (std::uint32_t ch = 0; ch < kChannels; ++ch) {
    float acc =
        conv_w[static_cast<std::size_t>(ch) * kKernel + kKernel - 1] * qkv[ch];
    for (std::uint32_t k = 0; k + 1 < kKernel; ++k) {
      acc += conv_w[static_cast<std::size_t>(ch) * kKernel + k] *
             ref_history[static_cast<std::size_t>(k) * kChannels + ch];
    }
    conv[ch] = static_cast<float>(t::SiluD(acc));
  }
  for (std::uint32_t k = 0; k + 1 < kKernel; ++k) {
    std::copy(
        ref_history.begin() + static_cast<std::ptrdiff_t>((k + 1) * kChannels),
        ref_history.begin() + static_cast<std::ptrdiff_t>((k + 2) * kChannels),
        ref_history.begin() + static_cast<std::ptrdiff_t>(k * kChannels));
  }
  std::copy(qkv.begin(), qkv.end(), ref_history.end() - kChannels);

  std::vector<float> ref_qn(kKeyDim);
  std::vector<float> ref_kn(kKeyDim);
  for (std::uint32_t h = 0; h < kKHeads; ++h) {
    for (int which = 0; which < 2; ++which) {
      const float* src = conv.data() + (which == 0 ? h : kKHeads + h) * kDim;
      double ss = 0.0;
      for (std::uint32_t i = 0; i < kDim; ++i) {
        ss += static_cast<double>(src[i]) * src[i];
      }
      const float scale = 1.0F / std::sqrt(static_cast<float>(ss) + kEps);
      float* dst = (which == 0 ? ref_qn.data() : ref_kn.data()) +
                   static_cast<std::size_t>(h) * kDim;
      for (std::uint32_t i = 0; i < kDim; ++i) {
        dst[i] = src[i] * scale;
      }
    }
  }

  std::vector<float> ref_attn(kValueDim);
  std::vector<float> ref_state = state;
  const float q_scale = 1.0F / std::sqrt(static_cast<float>(kDim));
  std::vector<float> u(kDim);
  for (std::uint32_t h = 0; h < kVHeads; ++h) {
    const std::uint32_t kh = h % kKHeads;
    const float* qh = ref_qn.data() + static_cast<std::size_t>(kh) * kDim;
    const float* khn = ref_kn.data() + static_cast<std::size_t>(kh) * kDim;
    const float* vh =
        conv.data() + 2 * kKeyDim + static_cast<std::size_t>(h) * kDim;
    float* S = ref_state.data() + static_cast<std::size_t>(h) * kDim * kDim;
    const float decay =
        static_cast<float>(std::exp(a[h] * t::SoftplusD(alpha[h] + dt[h])));
    const float b = static_cast<float>(t::SigmoidD(beta[h]));
    for (std::uint32_t j = 0; j < kDim; ++j) {
      float* row = S + static_cast<std::size_t>(j) * kDim;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < kDim; ++i) {
        row[i] *= decay;
        acc += static_cast<double>(row[i]) * khn[i];
      }
      u[j] = static_cast<float>(acc);
    }
    float* o = ref_attn.data() + static_cast<std::size_t>(h) * kDim;
    for (std::uint32_t j = 0; j < kDim; ++j) {
      const float delta = (vh[j] - u[j]) * b;
      float* row = S + static_cast<std::size_t>(j) * kDim;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < kDim; ++i) {
        row[i] += delta * khn[i];
        acc += static_cast<double>(row[i]) * qh[i];
      }
      o[j] = static_cast<float>(acc) * q_scale;
    }
    double ss = 0.0;
    for (std::uint32_t j = 0; j < kDim; ++j) {
      ss += static_cast<double>(o[j]) * o[j];
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(ss / kDim) + kEps);
    for (std::uint32_t j = 0; j < kDim; ++j) {
      o[j] = o[j] * scale * norm_w[j] *
             static_cast<float>(
                 t::SiluD(z[static_cast<std::size_t>(h) * kDim + j]));
    }
  }

  bool ok = true;
  ok = Check("GdnConv", t::WorstRelative(conv, convolved), 1e-4) && ok;
  ok = Check("GdnNormQk q", t::WorstRelative(ref_qn, qn), 1e-4) && ok;
  ok = Check("GdnNormQk k", t::WorstRelative(ref_kn, kn), 1e-4) && ok;
  ok = Check("GdnDelta attn", t::WorstRelative(ref_attn, attn), 2e-4) && ok;
  ok =
      Check("GdnDelta state", t::WorstRelative(ref_state, final_state), 2e-4) &&
      ok;
  for (float v : attn) {
    if (!std::isfinite(v)) {
      std::cerr << "GDN attn is not finite\n";
      return false;
    }
  }
  return ok;
}

}  // namespace

int main() {
  try {
    bool ok = RunCase(false);
    ok = RunCase(true) && ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}