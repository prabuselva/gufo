// Fused elementwise kernel contracts: each nbl_* entry versus the scalar
// oracle's own cpu_* implementation on deterministic synthetic data.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

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

float DotRow(const std::vector<float>& w, const float* x) {
  double acc = 0.0;
  for (std::size_t i = 0; i < w.size(); ++i) {
    acc += static_cast<double>(w[i]) * static_cast<double>(x[i]);
  }
  return static_cast<float>(acc);
}

// Scale/suffix bytes masked so no half or UE4M3 code decodes to NaN, which
// would defeat WorstRel even when kernel and oracle agree.
std::vector<std::uint8_t> MaskScales(std::vector<std::uint8_t> bytes,
                                     int type) {
  const int span = type == kNblGemvQ8_0 ? 34 : type == kNblGemvIq2Xxs ? 66 : 36;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    if (i % static_cast<std::size_t>(span) < 4) {
      bytes[i] = type == kNblGemvNvfp4
                     ? static_cast<std::uint8_t>(bytes[i] & 0x2FU)
                     : static_cast<std::uint8_t>(0x30U | (bytes[i] & 0x01U));
    }
  }
  return bytes;
}

std::size_t RowBytes(int type, int k) {
  return type == kNblGemvQ8_0     ? static_cast<std::size_t>(k) / 32 * 34
         : type == kNblGemvIq2Xxs ? static_cast<std::size_t>(k) / 256 * 66
                                  : static_cast<std::size_t>(k) / 64 * 36;
}

gufo::core::GgmlType GgmlFor(int type) {
  return type == kNblGemvQ8_0     ? gufo::core::GgmlType::kQ8_0
         : type == kNblGemvIq2Xxs ? gufo::core::GgmlType::kIQ2_XXS
                                  : gufo::core::GgmlType::kNVFP4;
}

// Dense row-batched GEMV versus dequantize + double-dot, covering token
// tails (len < NBL_ROW_NT tiles) and partial output-row blocks.
bool RunGemvRowsDense() {
  bool ok = true;
  std::uint32_t seed = 0xB00C0Au;
  struct Case {
    int type;
    int n_out;
    int k;
    int tokens;
  };
  for (const Case& c :
       {Case{kNblGemvQ8_0, 130, 4096, 1}, Case{kNblGemvQ8_0, 130, 1024, 65},
        Case{kNblGemvIq2Xxs, 257, 1024, 17},
        Case{kNblGemvNvfp4, 300, 1024, 33}}) {
    auto bytes = MaskScales(
        MakeBytes(static_cast<std::size_t>(c.n_out) * RowBytes(c.type, c.k),
                  seed++),
        c.type);
    auto x = MakeValues(static_cast<std::size_t>(c.tokens) * c.k, seed++, 1.0F);
    TensorRef w;
    w.data = bytes.data();
    w.type = GgmlFor(c.type);
    w.cols = c.k;
    w.rows = c.n_out;
    std::vector<float> wq(static_cast<std::size_t>(c.k));
    std::vector<float> reference(static_cast<std::size_t>(c.tokens) * c.n_out);
    for (int n = 0; n < c.n_out; ++n) {
      gufo::models::nemotron_h_moe::cpu::DequantizeRow(
          w, 0, static_cast<std::uint64_t>(n), wq.data());
      for (int t = 0; t < c.tokens; ++t) {
        reference[static_cast<std::size_t>(t) * c.n_out + n] =
            DotRow(wq, x.data() + static_cast<std::size_t>(t) * c.k);
      }
    }
    HipBuffer<std::uint8_t> w_dev(bytes.size());
    HipBuffer<float> x_dev(x.size());
    HipBuffer<float> out_dev(reference.size());
    Upload(w_dev.get(), bytes.data(), bytes.size());
    Upload(&x_dev, x);
    if (nbl_gemv_rows_f32(c.type, w_dev.get(), x_dev.get(), out_dev.get(),
                          c.n_out, c.k, c.k, c.tokens, nullptr) != 0) {
      std::printf("gemv_rows type=%d rc!=0\n", c.type);
      return false;
    }
    char name[64];
    std::snprintf(name, sizeof name, "gemv_rows t=%-4d n=%-4d N=%d", c.type,
                  c.n_out, c.tokens);
    ok = Check(name, WorstRel(reference, Download(&out_dev)), 1e-4) && ok;
  }
  return ok;
}

// Expert index table versus its contract, then the LIST row-batched GEMV
// against dequantize + double-dot with expert gather for both xdiv modes.
bool RunExpertIndexAndMoeRows() {
  bool ok = true;
  std::uint32_t seed = 0x1DEA0Bu;
  const std::vector<int> counts = {7, 0, 9, 4, 0, 3, 1, 12};
  std::vector<std::int32_t> ids;
  for (std::size_t e = 0; e < counts.size(); ++e) {
    for (int i = 0; i < counts[e]; ++i) {
      ids.push_back(static_cast<std::int32_t>(e));
    }
  }
  std::shuffle(ids.begin(), ids.end(), std::mt19937(0x2BE1u));
  const int n_exp = static_cast<int>(counts.size());
  const int slots = static_cast<int>(ids.size());
  const int grid_tiles = slots / NBL_ROW_NT + n_exp;
  HipBuffer<std::int32_t> ids_dev(ids.size());
  HipBuffer<std::int32_t> list_dev(static_cast<std::size_t>(slots));
  HipBuffer<std::int32_t> desc_dev(3ULL * static_cast<std::size_t>(grid_tiles));
  HipBuffer<std::int32_t> ndesc_dev(1);
  Upload(ids_dev.get(), ids.data(), ids.size() * sizeof(std::int32_t));
  if (nbl_expert_index(ids_dev.get(), n_exp, slots, NBL_ROW_NT, list_dev.get(),
                       desc_dev.get(), ndesc_dev.get(), nullptr) != 0) {
    std::printf("expert_index rc!=0\n");
    return false;
  }
  const std::vector<std::int32_t> list = Download(&list_dev);
  const std::vector<std::int32_t> desc = Download(&desc_dev);
  const std::vector<std::int32_t> n_desc = Download(&ndesc_dev);
  bool table_ok = !n_desc.empty() && n_desc[0] >= 0;
  if (table_ok) {
    std::vector<std::int32_t> seen(static_cast<std::size_t>(slots), 0);
    for (std::size_t r = 0; r < static_cast<std::size_t>(slots) && table_ok;
         ++r) {
      table_ok = list[r] >= 0 && list[r] < slots &&
                 ++seen[static_cast<std::size_t>(list[r])] == 1;
    }
    std::vector<int> placed(static_cast<std::size_t>(n_exp), 0);
    std::int64_t tiles = 0;
    for (int i = 0; i < n_desc[0] && table_ok; ++i) {
      const int e = desc[3 * i];
      const int begin = desc[3 * i + 1];
      const int len = desc[3 * i + 2];
      table_ok = e >= 0 && e < n_exp && len >= 1 && len <= NBL_ROW_NT;
      for (int j = 0; j < len && table_ok; ++j) {
        table_ok = ids[static_cast<std::size_t>(
                       list[static_cast<std::size_t>(begin + j)])] == e;
      }
      placed[static_cast<std::size_t>(e)] += len;
      tiles += (len + NBL_ROW_NT - 1) / NBL_ROW_NT;
    }
    for (int e = 0; e < n_exp && table_ok; ++e) {
      table_ok = placed[static_cast<std::size_t>(e)] ==
                 counts[static_cast<std::size_t>(e)];
    }
    table_ok = table_ok && tiles == n_desc[0];
  }
  std::printf("%-36s %s\n", "expert_index table", table_ok ? "PASS" : "FAIL");
  if (!table_ok) {
    g_fail = "expert_index table";
    return false;
  }
  const int k = 1024;
  for (const int xdiv : {0, 4}) {
    const int n_out = xdiv == 0 ? 257 : 1024;
    const int type = kNblGemvIq2Xxs;
    auto bytes = MaskScales(
        MakeBytes(static_cast<std::size_t>(n_exp) * n_out * RowBytes(type, k),
                  seed++),
        type);
    const int x_tokens = xdiv > 0 ? slots / xdiv : slots;
    auto x = MakeValues(static_cast<std::size_t>(x_tokens) * k, seed++, 1.0F);
    TensorRef w;
    w.data = bytes.data();
    w.type = GgmlFor(type);
    w.cols = k;
    w.rows = n_out;
    std::vector<float> wq(static_cast<std::size_t>(k));
    std::vector<float> reference(static_cast<std::size_t>(slots) * n_out);
    for (int i = 0; i < n_desc[0]; ++i) {
      const int e = desc[3 * i];
      const int begin = desc[3 * i + 1];
      const int len = desc[3 * i + 2];
      for (int j = 0; j < len; ++j) {
        const std::int32_t slot = list[static_cast<std::size_t>(begin + j)];
        const float* xv =
            x.data() +
            static_cast<std::size_t>(xdiv > 0 ? slot / xdiv : slot) * k;
        for (int n = 0; n < n_out; ++n) {
          gufo::models::nemotron_h_moe::cpu::DequantizeRow(
              w, static_cast<std::uint64_t>(e), static_cast<std::uint64_t>(n),
              wq.data());
          reference[static_cast<std::size_t>(slot) * n_out + n] =
              DotRow(wq, xv);
        }
      }
    }
    HipBuffer<std::uint8_t> w_dev(bytes.size());
    HipBuffer<float> x_dev(x.size());
    HipBuffer<float> out_dev(reference.size());
    Upload(w_dev.get(), bytes.data(), bytes.size());
    Upload(&x_dev, x);
    if (nbl_gemv_rows_moe(type, w_dev.get(), x_dev.get(), list_dev.get(),
                          desc_dev.get(), ndesc_dev.get(), grid_tiles, xdiv,
                          out_dev.get(), n_out, k, nullptr) != 0) {
      std::printf("gemv_rows_moe rc!=0\n");
      return false;
    }
    char name[64];
    std::snprintf(name, sizeof name, "gemv_rows_moe iq2 xdiv=%d", xdiv);
    ok = Check(name, WorstRel(reference, Download(&out_dev)), 1e-4) && ok;
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
  ok = RunGemvRowsDense() && ok;
  ok = RunExpertIndexAndMoeRows() && ok;
  if (ok) {
    std::printf("rocm_ops: ALL PASS\n");
  } else {
    std::printf("rocm_ops: FAIL (%s)\n", g_fail.c_str());
  }
  return ok ? 0 : 1;
}
