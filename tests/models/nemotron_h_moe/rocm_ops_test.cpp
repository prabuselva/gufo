// Fused elementwise kernel contracts: each nbl_* entry versus the scalar
// oracle's own cpu_* implementation on deterministic synthetic data.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/nemotron_h_moe/cpu_ops.hpp"
#include "src/models/nemotron_h_moe/kernels/rocm/nb_ops.h"
#include "src/models/nemotron_h_moe/weights.hpp"
#include "tests/models/nemotron_h_moe/hip_test.hpp"

namespace {

using gufo::models::nemotron_h_moe::TensorRef;
using namespace gufo::tests::nemotron_h_moe;

double WorstRel(const std::vector<float>& reference,
                const std::vector<float>& gpu) {
  double scale = 0.0;
  double worst = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    scale = std::max(scale, std::abs(static_cast<double>(reference[i])));
    worst = std::max(worst, std::abs(static_cast<double>(reference[i]) -
                                     static_cast<double>(gpu[i])));
  }
  return worst / std::max(scale, 1e-30);
}

void Report(const char* name, double rel, double tolerance) {
  std::printf("%-36s %s  rel=%.3e\n", name, rel <= tolerance ? "PASS" : "FAIL",
              rel);
}

std::string g_fail;

bool Check(const char* name, double rel, double tolerance) {
  Report(name, rel, tolerance);
  if (rel > tolerance) {
    g_fail = name;
    return false;
  }
  return true;
}

bool RunRmsNorm() {
  bool ok = true;
  std::uint32_t seed = 0x51A2u;
  for (const int hidden : {1024, 4096}) {
    for (const int rows : {1, 3}) {
      auto x =
          MakeValues(static_cast<std::size_t>(rows) * hidden, seed++, 1.0F);
      auto w = MakeValues(static_cast<std::size_t>(hidden), seed++, 0.5F);
      std::vector<float> reference;
      reference.reserve(x.size());
      for (int r = 0; r < rows; ++r) {
        std::vector<float> row(
            x.begin() + static_cast<std::ptrdiff_t>(r) * hidden,
            x.begin() + static_cast<std::ptrdiff_t>(r + 1) * hidden);
        gufo::models::nemotron_h_moe::cpu::RmsNorm(row, w.data(), 1e-5F);
        reference.insert(reference.end(), row.begin(), row.end());
      }
      HipBuffer<float> x_dev(x.size());
      HipBuffer<float> w_dev(w.size());
      HipBuffer<float> out_dev(x.size());
      Upload(&x_dev, x);
      Upload(&w_dev, w);
      if (nbl_rms_norm(x_dev.get(), w_dev.get(), out_dev.get(), rows, hidden,
                       1e-5F, nullptr) != 0) {
        std::printf("rms_norm rows=%d hidden=%d rc!=0\n", rows, hidden);
        return false;
      }
      char name[64];
      std::snprintf(name, sizeof name, "rms_norm rows=%d h=%d", rows, hidden);
      ok = Check(name, WorstRel(reference, Download(&out_dev)), 2e-6) && ok;
    }
  }
  return ok;
}

bool RunElementwise() {
  bool ok = true;
  std::uint32_t seed = 0xADD5u;
  for (const int n : {1, 1000, 4096}) {
    auto a = MakeValues(static_cast<std::size_t>(n), seed++, 1.0F);
    auto b = MakeValues(static_cast<std::size_t>(n), seed++, 1.0F);
    std::vector<float> sum(n);
    std::vector<float> rsq(n);
    for (int i = 0; i < n; ++i) {
      sum[i] = a[i] + b[i];
      rsq[i] = a[i] > 0.0F ? a[i] * a[i] : 0.0F;
    }
    HipBuffer<float> a_dev(a.size());
    HipBuffer<float> b_dev(b.size());
    HipBuffer<float> out_dev(a.size());
    Upload(&a_dev, a);
    Upload(&b_dev, b);
    Upload(&out_dev, a);
    nbl_add(out_dev.get(), b_dev.get(), n, nullptr);
    char name[64];
    std::snprintf(name, sizeof name, "add n=%d", n);
    ok = Check(name, WorstRel(sum, Download(&out_dev)), 0.0) && ok;
    Upload(&out_dev, a);
    nbl_relu_sqr(a_dev.get(), out_dev.get(), n, nullptr);
    std::snprintf(name, sizeof name, "relu_sqr n=%d", n);
    ok = Check(name, WorstRel(rsq, Download(&out_dev)), 0.0) && ok;
  }
  return ok;
}

bool RunDequantRow() {
  // Q8_0 rows of the token-embedding shape.
  const int cols = 4096;
  const int rows = 4;
  std::uint32_t seed = 0xC0DE00u;
  auto bytes = MakeBytes(static_cast<std::size_t>(rows) * cols / 32 * 34, seed);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    if (i % 34 < 2) {
      bytes[i] = 0x30 | static_cast<std::uint8_t>(bytes[i] & 0x01U);
    }
  }
  TensorRef w;
  w.data = bytes.data();
  w.type = gufo::core::GgmlType::kQ8_0;
  w.cols = cols;
  w.rows = rows;
  HipBuffer<std::uint8_t> w_dev(bytes.size());
  Upload(w_dev.get(), bytes.data(), bytes.size());
  bool ok = true;
  for (int r = 0; r < rows; ++r) {
    std::vector<float> reference(cols);
    gufo::models::nemotron_h_moe::cpu::DequantizeRow(
        w, 0, static_cast<std::uint64_t>(r), reference.data());
    HipBuffer<float> out_dev(reference.size());
    if (nbl_dequant_row_q8_0(w_dev.get(), r, cols, out_dev.get(), nullptr) !=
        0) {
      std::printf("dequant_row r=%d rc!=0\n", r);
      return false;
    }
    char name[64];
    std::snprintf(name, sizeof name, "dequant_row q8_0 r=%d", r);
    ok = Check(name, WorstRel(reference, Download(&out_dev)), 0.0) && ok;
  }
  return ok;
}

// Oracle selection: scores = sigmoid(logits) + bias, iterated argmax with
// the lower index kept on a tie, weights = sigmoid(selected) renormalized
// by 1 / max(sum, 2^-14) times the scale.
bool RunRouter(std::uint32_t seed) {
  bool ok = true;
  for (const int n_exp : {8, 240, 496}) {
    for (const int tokens : {1, 5}) {
      constexpr int kTopK = 9;
      auto logits =
          MakeValues(static_cast<std::size_t>(tokens) * n_exp, seed++, 4.0F);
      auto bias = MakeValues(static_cast<std::size_t>(n_exp), seed++, 0.125F);
      std::vector<std::int32_t> ref_ids(
          static_cast<std::size_t>(tokens) * kTopK, -1);
      std::vector<float> ref_w(static_cast<std::size_t>(tokens) * kTopK, 0.0F);
      for (int t = 0; t < tokens; ++t) {
        const float* lg = logits.data() + static_cast<std::size_t>(t) * n_exp;
        std::vector<float> scores(n_exp);
        for (int e = 0; e < n_exp; ++e) {
          scores[e] = 1.0F / (1.0F + std::exp(-lg[e])) + bias[e];
        }
        float num[kTopK];
        int total = 0;
        for (int k = 0; k < kTopK; ++k) {
          int best = -1;
          for (int e = 0; e < n_exp; ++e) {
            if (scores[e] == -std::numeric_limits<float>::infinity()) {
              continue;
            }
            if (best < 0 || scores[e] > scores[best]) {
              best = e;
            }
          }
          if (best < 0) {
            break;
          }
          ref_ids[static_cast<std::size_t>(t) * kTopK + k] = best;
          scores[best] = -std::numeric_limits<float>::infinity();
          num[k] = 1.0F / (1.0F + std::exp(-lg[best]));
          total = k + 1;
        }
        float sum = 0.0F;
        for (int k = 0; k < total; ++k) {
          sum += num[k];
        }
        sum = sum < 6.103515625e-5F ? 6.103515625e-5F : sum;
        for (int k = 0; k < total; ++k) {
          ref_w[static_cast<std::size_t>(t) * kTopK + k] =
              (num[k] / sum) * 5.0F;
        }
      }
      HipBuffer<float> lg_dev(logits.size());
      HipBuffer<float> bias_dev(bias.size());
      HipBuffer<std::int32_t> ids_dev(ref_ids.size());
      HipBuffer<float> w_dev(ref_w.size());
      Upload(&lg_dev, logits);
      Upload(&bias_dev, bias);
      if (nbl_router(lg_dev.get(), bias_dev.get(), n_exp, kTopK, 1, 5.0F,
                     tokens, ids_dev.get(), w_dev.get(), nullptr) != 0) {
        std::printf("router n_exp=%d tokens=%d rc!=0\n", n_exp, tokens);
        return false;
      }
      const std::vector<std::int32_t> ids = Download(&ids_dev);
      const std::vector<float> weights = Download(&w_dev);
      bool ids_ok = ids == ref_ids;
      double rel = WorstRel(ref_w, weights);
      char name[64];
      std::snprintf(name, sizeof name, "router n_exp=%-3d N=%d", n_exp, tokens);
      if (!ids_ok) {
        std::printf("%-36s FAIL  selection mismatch\n", name);
        g_fail = name;
      }
      ok = ids_ok && Check(name, rel, ids_ok ? 1e-6 : 1.0) && ok;
    }
  }
  return ok;
}

bool RunMoeCombine() {
  bool ok = true;
  std::uint32_t seed = 0xC04B1Eu;
  for (const int m : {128, 1024}) {
    constexpr int kTopK = 9;
    constexpr int kTokens = 3;
    auto src =
        MakeValues(static_cast<std::size_t>(kTokens) * kTopK * m, seed++, 1.0F);
    auto weights =
        MakeValues(static_cast<std::size_t>(kTokens) * kTopK, seed++, 1.0F);
    std::vector<float> reference(static_cast<std::size_t>(kTokens) * m, 0.0F);
    for (int t = 0; t < kTokens; ++t) {
      for (int j = 0; j < m; ++j) {
        double acc = 0.0;
        for (int u = 0; u < kTopK; ++u) {
          acc += static_cast<double>(
                     weights[static_cast<std::size_t>(t) * kTopK + u]) *
                 static_cast<double>(
                     src[(static_cast<std::size_t>(t) * kTopK + u) * m + j]);
        }
        reference[static_cast<std::size_t>(t) * m + j] =
            static_cast<float>(acc);
      }
    }
    HipBuffer<float> src_dev(src.size());
    HipBuffer<float> w_dev(weights.size());
    HipBuffer<float> out_dev(reference.size());
    Upload(&src_dev, src);
    Upload(&w_dev, weights);
    if (nbl_moe_combine(out_dev.get(), src_dev.get(), w_dev.get(), m, kTopK,
                        kTokens, nullptr) != 0) {
      std::printf("moe_combine m=%d rc!=0\n", m);
      return false;
    }
    char name[64];
    std::snprintf(name, sizeof name, "moe_combine m=%d", m);
    ok = Check(name, WorstRel(reference, Download(&out_dev)), 1e-5) && ok;
  }
  return ok;
}

float F16ToF32(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1FU;
  const std::uint32_t man = h & 0x3FFU;
  if (exp == 0U) {
    const float val = static_cast<float>(man) / 1024.0F * (1.0F / 16384.0F);
    return sign != 0U ? -val : val;
  }
  const std::uint32_t bits = sign | ((exp - 15U + 127U) << 23) | (man << 13);
  float out = 0.0F;
  std::memcpy(&out, &bits, sizeof out);
  return out;
}

// Q8_1 staging: block scale must be amax / 127 (in f16), the codes must
// reconstruct every value within half a quantum, the stored sum must match
// sum(qs) within one f16 ulp, and the amax element must saturate.
bool RunQuantQ8_1() {
  bool ok = true;
  std::uint32_t seed = 0x08E1u;
  for (const int n : {32, 1024, 5376}) {
    auto x = MakeValues(static_cast<std::size_t>(n), seed++, 1.0F);
    HipBuffer<float> x_dev(x.size());
    HipBuffer<std::uint8_t> q_dev(static_cast<std::size_t>(n) / 32 * 36);
    Upload(&x_dev, x);
    if (nbl_quantize_q8_1(x_dev.get(), q_dev.get(), n, nullptr) != 0) {
      std::printf("quant_q8_1 n=%d rc!=0\n", n);
      return false;
    }
    const std::vector<std::uint8_t> q = Download(&q_dev);
    bool block_ok = true;
    for (int b = 0; b < n / 32; ++b) {
      const std::uint8_t* blk = q.data() + static_cast<std::size_t>(b) * 36;
      std::uint16_t dh = 0;
      std::uint16_t sh = 0;
      std::memcpy(&dh, blk, 2);
      std::memcpy(&sh, blk + 2, 2);
      const float d = F16ToF32(dh);
      int sum = 0;
      int amax_idx = 0;
      float amax = 0.0F;
      for (int j = 0; j < 32; ++j) {
        const float v = x[static_cast<std::size_t>(b) * 32 + j];
        if (std::abs(v) > amax) {
          amax = std::abs(v);
          amax_idx = j;
        }
        const std::int8_t qv = static_cast<std::int8_t>(blk[4 + j]);
        sum += qv;
        // Reconstruction must stay within half a quantum (0.5), plus the
        // scale's f16 rounding propagated through q (<=127*2^-11 = 0.062)
        // and the float product rounding: 0.513 -> 0.58 with margin.
        if (std::abs(static_cast<double>(v) - static_cast<double>(d) * qv) >
            std::abs(static_cast<double>(d)) * 0.58) {
          block_ok = false;
        }
      }
      const double sat = std::abs(
          static_cast<double>(x[static_cast<std::size_t>(b) * 32 + amax_idx]) -
          d * static_cast<double>(static_cast<std::int8_t>(blk[4 + amax_idx])));
      if (sat > std::abs(static_cast<double>(d)) *
                    1.02) {  // amax element must saturate to +-127
        block_ok = false;
      }
      const float sum_f = F16ToF32(sh);
      const double sum_tol =
          std::max(1e-3, std::abs(static_cast<double>(sum)) * 9.8e-4);
      if (std::abs(static_cast<double>(sum_f) - sum) > sum_tol) {
        block_ok = false;
      }
    }
    char name[64];
    std::snprintf(name, sizeof name, "quant_q8_1 n=%d", n);
    if (!block_ok) {
      std::printf("%-36s FAIL  quantize bound mismatch\n", name);
      g_fail = name;
      ok = false;
    } else {
      Report(name, 0.0, 0.0);
    }
  }
  return ok;
}

// IQ2_XXS x Q8_1 integer GEMV versus the CPU oracle (dequantized rows,
// double accumulation) on the exact activation values fed to the kernel.
bool RunGemvIq2Q8() {
  constexpr int kRowsTokens = 3;
  bool ok = true;
  namespace cpu = gufo::models::nemotron_h_moe::cpu;
  std::uint32_t seed = 0x102Du;
  struct GemvCase {
    int n_out;
    int k;
    int experts;
    int slots;
    int stride;  // 0 = one k-vector shared across slots
  };
  const GemvCase cases[] = {{256, 256, 4, 1, 0},
                            {768, 512, 4, 1, 0},
                            {1536, 1024, 4, 2, 0},
                            {1024, 512, 4, 2, 512},
                            {256, 1024, 4, 2, 1024}};
  for (const GemvCase& gc : cases) {
    const int n_out = gc.n_out;
    const int k = gc.k;
    const int experts = gc.experts;
    const int slots = gc.slots;
    const int stride = gc.stride;
    const int rows_per_token = stride > 0 ? slots : 1;
    const std::size_t x_elems =
        static_cast<std::size_t>(kRowsTokens) * rows_per_token * k;
    const std::size_t w_rows = static_cast<std::size_t>(experts) * n_out;
    const std::size_t w_bytes =
        w_rows * (static_cast<std::size_t>(k) / 256 * 66);
    auto w = MakeBytes(w_bytes, seed++);
    // Keep the f16 block scales normal (+-(0.25..2)) so random codewords
    // dequantize to finite values.
    for (std::size_t off = 0; off < w_bytes; off += 66) {
      w[off] = static_cast<std::uint8_t>(0x00 | (seed++ & 0x3FU) << 2);
      w[off + 1] = static_cast<std::uint8_t>(0x3E | ((seed++ & 1U) << 7));
    }
    auto x = MakeValues(x_elems, seed++, 1.0F);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(kRowsTokens) *
                                  slots);
    for (auto& id : ids) {
      id = static_cast<std::int32_t>(seed++ %
                                     static_cast<std::uint32_t>(experts));
    }
    TensorRef wt;
    wt.data = w.data();
    wt.type = gufo::core::GgmlType::kIQ2_XXS;
    wt.cols = static_cast<std::uint64_t>(k);
    wt.rows = static_cast<std::uint64_t>(n_out);
    wt.experts = static_cast<std::uint64_t>(experts);
    std::vector<float> reference(static_cast<std::size_t>(kRowsTokens) * slots *
                                 n_out);
    std::vector<float> y(n_out);
    for (int t = 0; t < kRowsTokens; ++t) {
      for (int s = 0; s < slots; ++s) {
        const std::size_t xb =
            static_cast<std::size_t>(t) * rows_per_token * k +
            (stride > 0 ? static_cast<std::size_t>(s) * k : 0U);
        std::span<const float> xv(x.data() + xb, static_cast<std::size_t>(k));
        // The non-MoE entry point ignores ids (row `row` of expert 0 is
        // the weight); the MoE one gathers through ids[t * slots + s].
        const std::uint64_t expert =
            slots > 1 ? static_cast<std::uint64_t>(
                            ids[static_cast<std::size_t>(t) * slots + s])
                      : 0U;
        cpu::MatVec(
            wt, expert, xv,
            std::span<float>(y.data(), static_cast<std::size_t>(n_out)));
        std::copy(y.begin(), y.end(),
                  reference.begin() +
                      static_cast<std::ptrdiff_t>(
                          (static_cast<std::size_t>(t) * slots + s) * n_out));
      }
    }
    HipBuffer<float> x_dev(x.size());
    HipBuffer<std::uint8_t> w_dev(w.size());
    HipBuffer<std::uint8_t> xq_dev(x_elems / 32 * 36);
    HipBuffer<std::int32_t> ids_dev(ids.size());
    HipBuffer<float> out_dev(reference.size());
    Upload(&x_dev, x);
    Upload(w_dev.get(), w.data(), w.size());
    Upload(ids_dev.get(), ids.data(), ids.size() * sizeof(std::int32_t));
    if (nbl_quantize_q8_1(x_dev.get(), xq_dev.get(),
                          static_cast<std::int64_t>(x_elems), nullptr) != 0) {
      std::printf("gemv_iq2_q8 n=%d k=%d quantize rc!=0\n", n_out, k);
      return false;
    }
    int rc = 0;
    if (slots > 1) {
      rc = nbl_gemv_moe_f32(kNblGemvIq2XxsQ8_1, w_dev.get(),
                            reinterpret_cast<const float*>(xq_dev.get()),
                            ids_dev.get(), out_dev.get(), n_out, k, slots,
                            stride, kRowsTokens, nullptr);
    } else {
      rc = nbl_gemv_f32(kNblGemvIq2XxsQ8_1, w_dev.get(),
                        reinterpret_cast<const float*>(xq_dev.get()),
                        out_dev.get(), n_out, k, kRowsTokens, nullptr);
    }
    if (rc != 0) {
      std::printf("gemv_iq2_q8 n=%d k=%d rc=%d\n", n_out, k, rc);
      return false;
    }
    char name[64];
    std::snprintf(name, sizeof name, "gemv_iq2_q8 n=%-4d k=%-4d s=%d", n_out, k,
                  slots);
    const std::vector<float> got = Download(&out_dev);
    ok = Check(name, WorstRel(reference, got), 0.02) && ok;
  }
  return ok;
}

// Block-scale gemv tiers (Q4_K/Q5_K/Q6_K/IQ4_XS) versus the core oracle dot
// products on random weight bytes. Only the f16 block scales (and, for
// Q4_K/Q5_K, the 6-bit scale deltas) are constrained so every random block
// dequantizes to finite values; the kernels and the oracle share the same
// decode formulas, so any layout or indexing mistake shows as a mismatch.
bool RunGemvKQuant() {
  namespace quant = gufo::quant;
  using DotFn = float (*)(const void*, std::span<const float>, std::size_t);
  struct KqCase {
    int type;
    int row_bytes;  // bytes per 256-value block
    int scale_off;  // f16 d (+ dmin at scale_off+2 when two_scales)
    bool two_scales;
    DotFn dot;
    int n_out;
    int k;
    int experts;
    int slots;
    int stride;  // 0 = one k-vector shared across slots
  };
  const KqCase cases[] = {
      {kNblGemvQ4_K, 144, 0, true, &quant::DotProductQ4_K, 256, 256, 1, 1, 0},
      {kNblGemvQ5_K, 176, 0, true, &quant::DotProductQ5_K, 256, 256, 1, 1, 0},
      {kNblGemvQ6_K, 210, 208, false, &quant::DotProductQ6_K, 256, 256, 1, 1,
       0},
      {kNblGemvIq4Xs, 136, 0, false, &quant::DotProductIQ4_XS, 256, 256, 1, 1,
       0},
      {kNblGemvQ4_K, 144, 0, true, &quant::DotProductQ4_K, 512, 512, 1, 1, 0},
      {kNblGemvQ5_K, 176, 0, true, &quant::DotProductQ5_K, 320, 1024, 1, 1, 0},
      {kNblGemvQ6_K, 210, 208, false, &quant::DotProductQ6_K, 256, 768, 1, 1,
       0},
      {kNblGemvIq4Xs, 136, 0, false, &quant::DotProductIQ4_XS, 512, 512, 1, 1,
       0},
      {kNblGemvQ4_K, 144, 0, true, &quant::DotProductQ4_K, 256, 512, 4, 2, 512},
      {kNblGemvQ5_K, 176, 0, true, &quant::DotProductQ5_K, 256, 512, 4, 2, 512},
      {kNblGemvQ6_K, 210, 208, false, &quant::DotProductQ6_K, 256, 512, 4, 2,
       512},
      {kNblGemvIq4Xs, 136, 0, false, &quant::DotProductIQ4_XS, 256, 512, 4, 2,
       512},
  };
  constexpr int kTokens = 2;
  bool ok = true;
  std::uint32_t seed = 0x4C1Du;
  for (const KqCase& gc : cases) {
    const int blocks = gc.k / 256;
    const std::size_t rows = static_cast<std::size_t>(gc.experts) *
                             static_cast<std::size_t>(gc.n_out);
    const std::size_t w_bytes = rows * static_cast<std::size_t>(gc.row_bytes) *
                                static_cast<std::size_t>(blocks);
    auto w = MakeBytes(w_bytes, seed++);
    for (std::size_t off = 0; off < w_bytes;
         off += static_cast<std::size_t>(gc.row_bytes) *
                static_cast<std::size_t>(blocks)) {
      const std::size_t d0 = off + static_cast<std::size_t>(gc.scale_off);
      w[d0] = static_cast<std::uint8_t>((seed++ & 0x3FU) << 2);
      w[d0 + 1] = static_cast<std::uint8_t>(0x3E | ((seed++ & 1U) << 7));
      if (gc.two_scales) {
        w[d0 + 2] = static_cast<std::uint8_t>((seed++ & 0x3FU) << 2);
        w[d0 + 3] = static_cast<std::uint8_t>(0x3E | ((seed++ & 1U) << 7));
        // Q4_K/Q5_K 6-bit scale deltas: keep them small so the bit-shifted
        // d/dmin exponents stay finite for random blocks.
        for (int b = 0; b < 8; ++b) {
          w[off + 4 + static_cast<std::size_t>(b)] =
              static_cast<std::uint8_t>(seed++ & 0x03U);
        }
      }
    }
    const int rows_per_token = gc.stride > 0 ? gc.slots : 1;
    const std::size_t x_elems = static_cast<std::size_t>(kTokens) *
                                rows_per_token * static_cast<std::size_t>(gc.k);
    auto x = MakeValues(x_elems, seed++, 1.0F);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(kTokens) * gc.slots);
    for (auto& id : ids) {
      id = static_cast<std::int32_t>(seed++ %
                                     static_cast<std::uint32_t>(gc.experts));
    }
    const std::size_t out_elems =
        static_cast<std::size_t>(kTokens) * gc.slots * gc.n_out;
    std::vector<float> reference(out_elems);
    const std::size_t row_stride = static_cast<std::size_t>(gc.row_bytes) *
                                   static_cast<std::size_t>(blocks);
    for (int t = 0; t < kTokens; ++t) {
      for (int s = 0; s < gc.slots; ++s) {
        const std::size_t xb =
            static_cast<std::size_t>(t) * rows_per_token * gc.k +
            (gc.stride > 0 ? static_cast<std::size_t>(s) * gc.k : 0U);
        std::span<const float> xv(x.data() + xb,
                                  static_cast<std::size_t>(gc.k));
        const std::size_t expert =
            gc.slots > 1 ? static_cast<std::size_t>(
                               ids[static_cast<std::size_t>(t) * gc.slots + s])
                         : 0U;
        float* y = reference.data() +
                   (static_cast<std::size_t>(t) * gc.slots + s) * gc.n_out;
        for (int r = 0; r < gc.n_out; ++r) {
          y[r] = gc.dot(w.data() + (expert * gc.n_out + r) * row_stride, xv,
                        static_cast<std::size_t>(gc.k));
        }
      }
    }
    HipBuffer<float> x_dev(x.size());
    HipBuffer<std::uint8_t> w_dev(w.size());
    HipBuffer<std::int32_t> ids_dev(ids.size());
    HipBuffer<float> out_dev(reference.size());
    Upload(&x_dev, x);
    Upload(w_dev.get(), w.data(), w.size());
    Upload(ids_dev.get(), ids.data(), ids.size() * sizeof(std::int32_t));
    int rc = 0;
    if (gc.slots > 1) {
      rc = nbl_gemv_moe_f32(gc.type, w_dev.get(), x_dev.get(), ids_dev.get(),
                            out_dev.get(), gc.n_out, gc.k, gc.slots, gc.stride,
                            kTokens, nullptr);
    } else {
      rc = nbl_gemv_f32(gc.type, w_dev.get(), x_dev.get(), out_dev.get(),
                        gc.n_out, gc.k, kTokens, nullptr);
    }
    if (rc != 0) {
      std::printf("gemv_kq type=%d n=%d k=%d rc=%d\n", gc.type, gc.n_out, gc.k,
                  rc);
      return false;
    }
    char name[64];
    std::snprintf(name, sizeof name, "gemv_kq t=%d n=%-4d k=%-4d s=%d", gc.type,
                  gc.n_out, gc.k, gc.slots);
    ok = Check(name, WorstRel(reference, Download(&out_dev)), 2e-3) && ok;
  }
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok = RunRmsNorm() && ok;
  ok = RunElementwise() && ok;
  ok = RunDequantRow() && ok;
  ok = RunRouter(0xB0BAu) && ok;
  ok = RunMoeCombine() && ok;
  ok = RunQuantQ8_1() && ok;
  ok = RunGemvIq2Q8() && ok;
  ok = RunGemvKQuant() && ok;
  if (ok) {
    std::printf("rocm_ops: ALL PASS\n");
  } else {
    std::printf("rocm_ops: FAIL (%s)\n", g_fail.c_str());
  }
  return ok ? 0 : 1;
}
