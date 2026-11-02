// Mamba-2 scan contract: nbl_mamba_scan over chunked prefill and single
// decode steps versus an independent CPU fixture that mirrors the oracle
// StepMamba recurrence (conv window shift, per-head Euler update, z gate).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "src/models/nemotron_h_moe/kernels/rocm/nb_mamba2.h"
#include "tests/models/nemotron_h_moe/hip_test.hpp"

namespace {

using namespace gufo::tests::nemotron_h_moe;

struct Shapes {
  int inner{256};
  int heads{16};
  int hd{16};
  int state{64};
  int groups{2};
  int taps{4};
  [[nodiscard]] int xbc() const { return inner + 2 * groups * state; }
  [[nodiscard]] int proj() const {
    return 2 * inner + 2 * groups * state + heads;
  }
};

float SiluD(float x) {
  return x / (1.0F + std::exp(-x));
}

float SoftplusD(float x) {
  return x > 20.0F ? x : std::log1p(std::exp(x));
}

// Independent fixture: one token through conv + SSM + gate, updating
// window/state in place. No grouped norm.
void ReferenceStep(const Shapes& s, const float* zxbc, const float* cw,
                   const float* cb, const float* a, const float* d,
                   const float* dtb, float* conv, float* sp, float* y) {
  const int inner = s.inner;
  const int xbc = s.xbc();
  std::vector<float> cx(static_cast<std::size_t>(xbc));
  for (int ch = 0; ch < xbc; ++ch) {
    float acc = cb != nullptr ? cb[ch] : 0.0F;
    for (int k = 0; k + 1 < s.taps; ++k) {
      acc += cw[ch * s.taps + k] * conv[static_cast<std::size_t>(k) * xbc + ch];
    }
    const float nv = zxbc[inner + ch];
    acc += cw[ch * s.taps + s.taps - 1] * nv;
    cx[static_cast<std::size_t>(ch)] = SiluD(acc);
    for (int k = 0; k + 2 < s.taps; ++k) {
      conv[static_cast<std::size_t>(k) * xbc + ch] =
          conv[static_cast<std::size_t>(k + 1) * xbc + ch];
    }
    conv[static_cast<std::size_t>(s.taps - 2) * xbc + ch] = nv;
  }
  for (int h = 0; h < s.heads; ++h) {
    const int g = h / (s.heads / s.groups);
    const float dt_sp =
        SoftplusD(zxbc[2 * inner + 2 * s.groups * s.state + h] + dtb[h]);
    const float dA = std::exp(dt_sp * a[h]);
    for (int i = 0; i < s.hd; ++i) {
      const int q = h * s.hd + i;
      const float xv = cx[static_cast<std::size_t>(q)] * dt_sp;
      float sum = 0.0F;
      for (int st = 0; st < s.state; ++st) {
        const std::size_t k = static_cast<std::size_t>(q) * s.state +
                              static_cast<std::size_t>(st);
        const float ns =
            sp[k] * dA +
            cx[static_cast<std::size_t>(inner + g * s.state + st)] * xv;
        sp[k] = ns;
        sum +=
            ns *
            cx[static_cast<std::size_t>(inner + (s.groups + g) * s.state + st)];
      }
      y[q] = (sum + cx[static_cast<std::size_t>(q)] * d[h]) *
             SiluD(zxbc[q]);  // z channels [0, inner)
    }
  }
}

bool Check(const char* name, const std::vector<float>& reference,
           const std::vector<float>& gpu, double tolerance) {
  double scale = 0.0;
  double worst = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    scale = std::max(scale, std::abs(static_cast<double>(reference[i])));
    worst = std::max(worst, std::abs(static_cast<double>(reference[i]) -
                                     static_cast<double>(gpu[i])));
  }
  const double rel = worst / std::max(scale, 1e-30);
  const bool ok = rel <= tolerance;
  std::printf("%-36s %s  rel=%.3e\n", name, ok ? "PASS" : "FAIL", rel);
  return ok;
}

void ReferenceStepD(const Shapes& s, const float* zxbc, const float* cw,
                    const float* cb, const float* a, const float* d,
                    const float* dtb, double* conv, double* sp, float* y);

bool RunChunk(const Shapes& sh, int tokens, int iters) {
  std::uint32_t seed = 0x51A200u + static_cast<std::uint32_t>(tokens);
  const int xbc = sh.xbc();
  const int proj = sh.proj();
  auto zxbc =
      MakeValues(static_cast<std::size_t>(iters) * tokens * proj, seed, 0.6F);
  auto cw = MakeValues(static_cast<std::size_t>(xbc) * sh.taps, seed ^ 1, 1.0F);
  auto cb = MakeValues(static_cast<std::size_t>(xbc), seed ^ 2, 0.25F);
  auto a = MakeValues(static_cast<std::size_t>(sh.heads), seed ^ 3, 1.0F);
  auto d = MakeValues(static_cast<std::size_t>(sh.heads), seed ^ 4, 1.0F);
  auto dtb = MakeValues(static_cast<std::size_t>(sh.heads), seed ^ 5, 0.5F);
  for (float& v : a) {
    v = -std::abs(v);  // the checkpoint stores -exp(A_log)
  }

  std::vector<float> ref_conv(static_cast<std::size_t>(sh.taps - 1) * xbc,
                              0.0F);
  std::vector<float> ref_ssm(
      static_cast<std::size_t>(sh.heads) * sh.hd * sh.state, 0.0F);
  std::vector<float> ref_y(static_cast<std::size_t>(iters) * tokens * sh.inner);
  std::vector<float> ref_yd(static_cast<std::size_t>(iters) * tokens *
                            sh.inner);
  std::vector<double> ref_conv_d(ref_conv.size(), 0.0);
  std::vector<double> ref_ssm_d(ref_ssm.size(), 0.0);
  const std::vector<float> conv0 = ref_conv;  // zero-initialized inputs
  const std::vector<float> ssm0 = ref_ssm;
  for (int it = 0; it < iters; ++it) {
    for (int t = 0; t < tokens; ++t) {
      const float* z =
          zxbc.data() + static_cast<std::size_t>(it * tokens + t) * proj;
      float* y =
          ref_y.data() + static_cast<std::size_t>(it * tokens + t) * sh.inner;
      float* yd =
          ref_yd.data() + static_cast<std::size_t>(it * tokens + t) * sh.inner;
      ReferenceStep(sh, z, cw.data(), cb.data(), a.data(), d.data(), dtb.data(),
                    ref_conv.data(), ref_ssm.data(), y);
      ReferenceStepD(sh, z, cw.data(), cb.data(), a.data(), d.data(),
                     dtb.data(), ref_conv_d.data(), ref_ssm_d.data(), yd);
    }
  }

  HipBuffer<float> z_dev(zxbc.size());
  HipBuffer<float> cw_dev(cw.size());
  HipBuffer<float> cb_dev(cb.size());
  HipBuffer<float> a_dev(a.size());
  HipBuffer<float> d_dev(d.size());
  HipBuffer<float> dtb_dev(dtb.size());
  HipBuffer<float> conv_dev(ref_conv.size());
  HipBuffer<float> ssm_dev(ref_ssm.size());
  HipBuffer<float> y_dev(ref_y.size());
  Upload(&z_dev, zxbc);
  Upload(&cw_dev, cw);
  Upload(&cb_dev, cb);
  Upload(&a_dev, a);
  Upload(&d_dev, d);
  Upload(&dtb_dev, dtb);
  Upload(&conv_dev, conv0);  // zero-initialized running state
  Upload(&ssm_dev, ssm0);
  for (int it = 0; it < iters; ++it) {
    if (nbl_mamba_scan(
            z_dev.get() + static_cast<std::size_t>(it) * tokens * proj,
            cw_dev.get(), cb_dev.get(), a_dev.get(), d_dev.get(), dtb_dev.get(),
            conv_dev.get(), ssm_dev.get(),
            y_dev.get() + static_cast<std::size_t>(it) * tokens * sh.inner,
            sh.inner, sh.heads, sh.hd, sh.state, sh.groups, sh.taps, xbc, proj,
            tokens, nullptr) != 0) {
      std::printf("mamba_scan tokens=%d rc!=0\n", tokens);
      return false;
    }
  }
  char name[64];
  std::snprintf(name, sizeof name, "mamba_scan T=%d x%d", tokens, iters);
  const std::vector<float> got_y = Download(&y_dev);
  const bool y_ok = Check(name, ref_y, got_y, 2e-4);
  std::snprintf(name, sizeof name, "mamba_scanD T=%d x%d", tokens, iters);
  const bool yd_ok = Check(name, ref_yd, got_y, 2e-4);
  std::snprintf(name, sizeof name, "mamba_DvsF T=%d x%d", tokens, iters);
  const bool d_ok = Check(name, ref_yd, ref_y, 2e-4);
  (void)yd_ok;
  (void)d_ok;
  std::snprintf(name, sizeof name, "mamba_state T=%d x%d", tokens, iters);
  const std::vector<float> got_conv = Download(&conv_dev);
  const std::vector<float> got_ssm = Download(&ssm_dev);
  const bool conv_ok = Check(name, ref_conv, got_conv, 0.0) && y_ok;
  const bool ssm_ok = Check("ssm_state", ref_ssm, got_ssm, 2e-4) && conv_ok;
  return ssm_ok;
}

// Double-precision y oracle isolating fp32 summation/cancellation noise.
void ReferenceStepD(const Shapes& s, const float* zxbc, const float* cw,
                    const float* cb, const float* a, const float* d,
                    const float* dtb, double* conv, double* sp, float* y) {
  const int inner = s.inner;
  const int xbc = s.xbc();
  std::vector<double> cx(static_cast<std::size_t>(xbc));
  for (int ch = 0; ch < xbc; ++ch) {
    double acc = cb != nullptr ? cb[ch] : 0.0;
    for (int k = 0; k + 1 < s.taps; ++k) {
      acc += static_cast<double>(cw[ch * s.taps + k]) *
             conv[static_cast<std::size_t>(k) * xbc + ch];
    }
    const double nv = zxbc[inner + ch];
    acc += static_cast<double>(cw[ch * s.taps + s.taps - 1]) * nv;
    const float sil = static_cast<float>(acc / (1.0 + std::exp(-acc)));
    cx[static_cast<std::size_t>(ch)] = sil;
    for (int k = 0; k + 2 < s.taps; ++k) {
      conv[static_cast<std::size_t>(k) * xbc + ch] =
          conv[static_cast<std::size_t>(k + 1) * xbc + ch];
    }
    conv[static_cast<std::size_t>(s.taps - 2) * xbc + ch] = nv;
  }
  for (int h = 0; h < s.heads; ++h) {
    const int g = h / (s.heads / s.groups);
    const double dt_sp =
        zxbc[2 * inner + 2 * s.groups * s.state + h] + dtb[h] > 20.0F
            ? zxbc[2 * inner + 2 * s.groups * s.state + h] + dtb[h]
            : std::log1p(std::exp(static_cast<double>(
                  zxbc[2 * inner + 2 * s.groups * s.state + h] + dtb[h])));
    const double dA = std::exp(dt_sp * static_cast<double>(a[h]));
    for (int i = 0; i < s.hd; ++i) {
      const int q = h * s.hd + i;
      const double xv = cx[static_cast<std::size_t>(q)] * dt_sp;
      double sum = 0.0;
      for (int st = 0; st < s.state; ++st) {
        const std::size_t k = static_cast<std::size_t>(q) * s.state +
                              static_cast<std::size_t>(st);
        const double ns =
            sp[k] * dA +
            cx[static_cast<std::size_t>(inner + g * s.state + st)] * xv;
        sp[k] = ns;
        sum +=
            ns *
            cx[static_cast<std::size_t>(inner + (s.groups + g) * s.state + st)];
      }
      const double zz = zxbc[q];
      y[q] = static_cast<float>((sum + cx[static_cast<std::size_t>(q)] * d[h]) *
                                zz / (1.0 + std::exp(-zz)));
    }
  }
}

}  // namespace

int main() {
  Shapes sh;
  bool ok = true;
  ok = RunChunk(sh, 1, 6) && ok;    // decode
  ok = RunChunk(sh, 8, 2) && ok;    // small prefill chunks
  ok = RunChunk(sh, 128, 1) && ok;  // larger chunk
  std::printf("rocm_mamba: %s\n", ok ? "ALL PASS" : "FAILURES");
  return ok ? 0 : 1;
}
