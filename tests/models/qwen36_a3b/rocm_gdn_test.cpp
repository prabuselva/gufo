#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
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

bool Check(const std::string& name, double worst, double tolerance) {
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

// The batched prefill block must reproduce the decode kernels run token by
// token: same per-token attention, same final recurrent state, same advanced
// convolution history. The decode path is the already-verified ground truth.
bool RunPrefillCase(std::uint32_t tokens) {
  const std::size_t state_count =
      static_cast<std::size_t>(kVHeads) * kDim * kDim;
  const std::size_t conv_state =
      static_cast<std::size_t>(kKernel - 1) * kChannels;
  const std::size_t qkv_count = static_cast<std::size_t>(tokens) * kChannels;
  const std::size_t attn_count =
      static_cast<std::size_t>(tokens) * kValueDim;

  const auto qkv = t::MakeValues(qkv_count, 0x1234ABCDU, 1.0F);
  const auto conv_w = t::MakeValues(
      static_cast<std::size_t>(kChannels) * kKernel, 0xDEADBEEFU, 0.5F);
  const auto z = t::MakeValues(attn_count, 0x0BADF00DU, 2.0F);
  const auto alpha = t::MakeValues(static_cast<std::size_t>(tokens) * kVHeads,
                                   0xBADC0FFEU, 2.0F);
  const auto beta = t::MakeValues(static_cast<std::size_t>(tokens) * kVHeads,
                                  0x600DCAFEU, 3.0F);
  const auto a = t::MakeValues(kVHeads, 0xC0FFEE11U, 1.0F, -1.5F);
  const auto dt = t::MakeValues(kVHeads, 0xFEEDFACEU, 1.0F);
  const auto norm_w = t::MakeValues(kDim, 0x13579BDFU, 0.5F, 1.0F);
  const auto history = t::MakeValues(conv_state, 0x2468ACE0U, 1.0F);
  const auto state = t::MakeValues(state_count, 0x0F1E2D3CU, 0.1F);

  // Prefill buffers.
  t::HipBuffer<float> d_qkv(qkv_count);
  t::HipBuffer<float> d_conv_w(conv_w.size());
  t::HipBuffer<float> d_z(attn_count);
  t::HipBuffer<float> d_alpha(alpha.size());
  t::HipBuffer<float> d_beta(beta.size());
  t::HipBuffer<float> d_a(a.size());
  t::HipBuffer<float> d_dt(dt.size());
  t::HipBuffer<float> d_norm_w(norm_w.size());
  t::HipBuffer<float> d_convolved(qkv_count);
  t::HipBuffer<float> d_qn(static_cast<std::size_t>(tokens) * kKeyDim);
  t::HipBuffer<float> d_kn(static_cast<std::size_t>(tokens) * kKeyDim);
  t::HipBuffer<float> d_attn(attn_count);
  t::HipBuffer<float> d_state(state_count);
  t::HipBuffer<float> d_history(conv_state);
  t::HipBuffer<float> d_hist_new(conv_state);
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

  q::GdnConvPrefill(d_qkv.get(), d_conv_w.get(), d_history.get(),
                    d_convolved.get(), tokens, kChannels, kKernel, nullptr);
  q::GdnHistoryUpdate(d_qkv.get(), d_history.get(), d_hist_new.get(), tokens,
                      kChannels, kKernel, nullptr);
  q::GdnNormQkPrefill(d_convolved.get(), d_qn.get(), d_kn.get(), tokens,
                      kKHeads, kChannels, kDim, kEps, nullptr);
  q::GdnDeltaLoop(d_qn.get(), d_kn.get(), d_convolved.get(), d_alpha.get(),
                  d_beta.get(), d_a.get(), d_dt.get(), d_state.get(),
                  d_attn.get(), tokens, kKHeads, kVHeads, kDim, kChannels,
                  nullptr, 0, nullptr);
  q::GdnOutNormPrefill(d_attn.get(), d_z.get(), d_norm_w.get(), tokens, kVHeads,
                       kDim, kEps, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "GDN prefill synchronization");
  const auto pf_attn = t::Download(&d_attn, attn_count);
  const auto pf_state = t::Download(&d_state, state_count);
  const auto pf_hist = t::Download(&d_hist_new, conv_state);

  // Decode reference: the existing kernels, one token at a time, rolling the
  // same history and state.
  t::HipBuffer<float> r_conv(kChannels);
  t::HipBuffer<float> r_qn(kKeyDim);
  t::HipBuffer<float> r_kn(kKeyDim);
  t::HipBuffer<float> r_attn(kValueDim);
  t::HipBuffer<float> r_state(state_count);
  t::HipBuffer<float> r_history(conv_state);
  t::HipBuffer<float> r_attn_all(attn_count);
  t::Upload(&r_state, state);
  t::Upload(&r_history, history);
  std::vector<std::vector<float>> dc_states;
  for (std::uint32_t tk = 0; tk < tokens; ++tk) {
    const float* qkv_t = d_qkv.get() + static_cast<std::size_t>(tk) * kChannels;
    q::GdnConv(qkv_t, d_conv_w.get(), r_history.get(), r_conv.get(), kChannels,
               kKernel, nullptr);
    q::GdnNormQk(r_conv.get(), r_qn.get(), r_kn.get(), kKHeads, kDim, kEps,
                 nullptr);
    q::GdnDelta(r_qn.get(), r_kn.get(), r_conv.get() + 2 * kKeyDim,
                d_alpha.get() + static_cast<std::size_t>(tk) * kVHeads,
                d_beta.get() + static_cast<std::size_t>(tk) * kVHeads,
                d_a.get(), d_dt.get(), r_state.get(), r_attn.get(), kKHeads,
                kVHeads, kDim, nullptr);
    q::GdnOutNorm(r_attn.get(), d_z.get() + static_cast<std::size_t>(tk) * kValueDim,
                  d_norm_w.get(), kVHeads, kDim, kEps, nullptr);
    (void)hipMemcpyAsync(r_attn_all.get() +
                             static_cast<std::size_t>(tk) * kValueDim,
                         r_attn.get(), kValueDim * sizeof(float),
                         hipMemcpyDeviceToDevice, nullptr);
    dc_states.push_back(t::Download(&r_state, state_count));
  }
  t::CheckHip(hipDeviceSynchronize(), "GDN decode reference synchronization");
  const auto dc_attn = t::Download(&r_attn_all, attn_count);
  const auto dc_state = t::Download(&r_state, state_count);
  const auto dc_hist = t::Download(&r_history, conv_state);

  bool ok = true;
  ok = Check("GdnPrefill attn tokens=" + std::to_string(tokens),
             t::WorstRelativeToScale(dc_attn, pf_attn), 1e-4) &&
       ok;
  ok = Check("GdnPrefill state tokens=" + std::to_string(tokens),
             t::WorstRelativeToScale(dc_state, pf_state), 1e-4) &&
       ok;
  ok = Check("GdnPrefill history tokens=" + std::to_string(tokens),
             t::WorstRelativeToScale(dc_hist, pf_hist), 1e-4) &&
       ok;

  // Speculative verify rolls the recurrent state back to the last accepted
  // row, so slot t of the snapshot must equal the decode state after
  // consuming tokens [0, t].
  if (tokens > 1) {
    t::HipBuffer<float> d_snap(static_cast<std::size_t>(tokens - 1) *
                               state_count);
    t::Upload(&d_state, state);
    q::GdnDeltaLoop(d_qn.get(), d_kn.get(), d_convolved.get(), d_alpha.get(),
                    d_beta.get(), d_a.get(), d_dt.get(), d_state.get(),
                    d_attn.get(), tokens, kKHeads, kVHeads, kDim, kChannels,
                    d_snap.get(), tokens - 1, nullptr);
    t::CheckHip(hipDeviceSynchronize(), "GDN snapshot synchronization");
    const auto snap =
        t::Download(&d_snap, static_cast<std::size_t>(tokens - 1) * state_count);
    double worst = 0.0;
    for (std::uint32_t tk = 0; tk + 1 < tokens; ++tk) {
      const std::vector<float> slot(
          snap.begin() + static_cast<std::size_t>(tk) * state_count,
          snap.begin() + static_cast<std::size_t>(tk + 1) * state_count);
      worst = std::max(worst, t::WorstRelativeToScale(dc_states[tk], slot));
    }
    ok = Check("GdnPrefill state snapshots tokens=" + std::to_string(tokens),
               worst, 1e-4) &&
         ok;
  }
  return ok;
}

}  // namespace

int main() {
  try {
    bool ok = RunCase(false);
    ok = RunCase(true) && ok;
    for (std::uint32_t tokens : {1U, 2U, 5U, 17U, 64U}) {
      ok = RunPrefillCase(tokens) && ok;
    }
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}