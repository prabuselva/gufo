// Attention contract: nbl_attn_append + nbl_attn (NoPE GQA, online softmax)
// versus an independent CPU fixture over the same KV cache contents for
// causal prefill chunks, mid-cache chunks and decode steps.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "src/models/nemotron_h_moe/kernels/rocm/nb_attention.h"
#include "tests/models/nemotron_h_moe/hip_test.hpp"

namespace {

using namespace gufo::tests::nemotron_h_moe;

struct Cfg {
  int heads;
  int hd;
  int hdv;
  int kvh;
  double scale;
  [[nodiscard]] int kvdim() const { return kvh * hd; }
};

// Independent causal GQA SDPA over the cache rows [0, pos0 + tokens).
void ReferenceAttn(const Cfg& c, const std::vector<float>& q,
                   const std::vector<float>& kv, int pos0, int tokens,
                   std::vector<float>& out) {
  const int kvdim = c.kvdim();
  const int64_t kvrow = 2 * kvdim;
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < c.heads; ++h) {
      const int g = h * c.kvh / c.heads;
      const float* qh =
          q.data() + static_cast<std::size_t>(t) * c.heads * c.hd + h * c.hd;
      const int n = pos0 + t + 1;
      std::vector<double> scores(static_cast<std::size_t>(n));
      double mx = -1e300;
      for (int p = 0; p < n; ++p) {
        const float* kp =
            kv.data() + static_cast<std::size_t>(p) * kvrow + g * c.hd;
        double acc = 0.0;
        for (int d = 0; d < c.hd; ++d) {
          acc += static_cast<double>(qh[d]) * static_cast<double>(kp[d]);
        }
        acc *= c.scale;
        scores[static_cast<std::size_t>(p)] = acc;
        mx = std::max(mx, acc);
      }
      double sum = 0.0;
      for (int p = 0; p < n; ++p) {
        scores[static_cast<std::size_t>(p)] =
            std::exp(scores[static_cast<std::size_t>(p)] - mx);
        sum += scores[static_cast<std::size_t>(p)];
      }
      for (int d = 0; d < c.hdv; ++d) {
        double acc = 0.0;
        for (int p = 0; p < n; ++p) {
          const float* vp = kv.data() + static_cast<std::size_t>(p) * kvrow +
                            kvdim + g * c.hdv + d;
          acc += scores[static_cast<std::size_t>(p)] * static_cast<double>(*vp);
        }
        out[static_cast<std::size_t>(t) * c.heads * c.hdv + h * c.hdv + d] =
            static_cast<float>(acc / sum);
      }
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
  std::printf("%-40s %s  rel=%.3e\n", name, ok ? "PASS" : "FAIL", rel);
  return ok;
}

bool RunChunk(const Cfg& c, std::uint32_t& seed, std::vector<float>& cache,
              int pos0, int tokens, const char* label) {
  const int kvdim = c.kvdim();
  auto q = MakeValues(static_cast<std::size_t>(tokens) * c.heads * c.hd, seed++,
                      1.0F);
  auto k = MakeValues(static_cast<std::size_t>(tokens) * kvdim, seed++, 1.0F);
  auto v = MakeValues(static_cast<std::size_t>(tokens) * kvdim, seed++, 1.0F);

  std::vector<float> ref(static_cast<std::size_t>(tokens) * c.heads * c.hdv);
  // Append into the host cache first; the fixture reads post-append rows.
  for (int t = 0; t < tokens; ++t) {
    float* row = cache.data() + static_cast<std::size_t>(pos0 + t) * 2 * kvdim;
    for (int d = 0; d < kvdim; ++d) {
      row[d] = k[static_cast<std::size_t>(t) * kvdim + d];
      row[kvdim + d] = v[static_cast<std::size_t>(t) * kvdim + d];
    }
  }
  ReferenceAttn(c, q, cache, pos0, tokens, ref);

  HipBuffer<float> q_dev(q.size());
  HipBuffer<float> k_dev(k.size());
  HipBuffer<float> v_dev(v.size());
  HipBuffer<float> kv_dev(cache.size());
  HipBuffer<float> out_dev(ref.size());
  Upload(&q_dev, q);
  Upload(&k_dev, k);
  Upload(&v_dev, v);
  Upload(&kv_dev, cache);
  if (nbl_attn_append(k_dev.get(), v_dev.get(), kv_dev.get(), pos0, tokens,
                      kvdim, nullptr) != 0) {
    std::printf("%s: append rc!=0\n", label);
    return false;
  }
  if (nbl_attn(q_dev.get(), kv_dev.get(), pos0, tokens, out_dev.get(), c.heads,
               c.hd, c.hdv, c.kvh, kvdim, static_cast<float>(c.scale),
               nullptr) != 0) {
    std::printf("%s: attn rc!=0\n", label);
    return false;
  }
  const std::vector<float> got = Download(&out_dev);
  // The appended cache rows must match the freshly uploaded projection rows.
  const std::vector<float> got_cache = Download(&kv_dev);
  char name[96];
  std::snprintf(name, sizeof name, "%s append", label);
  bool ok = Check(name, cache, got_cache, 0.0);
  std::snprintf(name, sizeof name, "%s", label);
  ok = Check(name, ref, got, 2e-5) && ok;
  return ok;
}

bool RunShape(const Cfg& c, int capacity) {
  std::uint32_t seed = 0xA770ULL;
  std::vector<float> cache(static_cast<std::size_t>(capacity) * 2 * c.kvdim(),
                           0.0F);
  bool ok = true;
  char label[96];
  std::snprintf(label, sizeof label, "attn h%d kv%d T5 prefill", c.heads,
                c.kvh);
  ok = RunChunk(c, seed, cache, 0, 5, label) && ok;
  std::snprintf(label, sizeof label, "attn h%d kv%d decode x4", c.heads, c.kvh);
  for (int i = 0; i < 4; ++i) {
    std::snprintf(label, sizeof label, "attn h%d kv%d step%d", c.heads, c.kvh,
                  i);
    ok = RunChunk(c, seed, cache, 5 + i, 1, label) && ok;
  }
  std::snprintf(label, sizeof label, "attn h%d kv%d T6 midcache", c.heads,
                c.kvh);
  ok = RunChunk(c, seed, cache, 9, 6, label) && ok;
  std::snprintf(label, sizeof label, "attn h%d kv%d T17 unaligned", c.heads,
                c.kvh);
  ok = RunChunk(c, seed, cache, 15, 17, label) && ok;
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok = RunShape(Cfg{8, 64, 64, 2, 1.0 / 8.0}, 200) && ok;
  ok = RunShape(Cfg{4, 32, 32, 1, 1.0 / std::sqrt(32.0)}, 200) && ok;
  ok = RunShape(Cfg{6, 64, 32, 3, 0.125}, 200) && ok;  // hd != hdv
  std::printf("rocm_attention: %s\n", ok ? "ALL PASS" : "FAILURES");
  return ok ? 0 : 1;
}
