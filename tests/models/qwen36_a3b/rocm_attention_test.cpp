#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "src/models/qwen36_a3b/kernels/rocm/kernels.hpp"
#include "tests/models/qwen36_a3b/hip_test.hpp"

namespace q = gufo::models::qwen36_a3b::rocm;
namespace t = gufo::tests::qwen36_a3b;

namespace {

// The model's full-attention geometry: 16 query heads over 2 key/value heads,
// 256-wide heads, with a sigmoid output gate.
constexpr std::uint32_t kHeads = 16;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kGroup = kHeads / kKvHeads;

bool Check(const std::string& name, double worst, double tolerance) {
  std::cout << name << " worst relative error " << worst << '\n';
  if (!(worst <= tolerance)) {
    std::cerr << name << " exceeded tolerance " << tolerance << '\n';
    return false;
  }
  return true;
}

bool RunCase(std::uint32_t n_kv) {
  const std::size_t q_count = static_cast<std::size_t>(kHeads) * kHeadDim;
  const std::size_t cache_count =
      static_cast<std::size_t>(n_kv) * kKvHeads * kHeadDim;
  const auto query = t::MakeValues(q_count, 0x1234ABCDU, 1.0F);
  const auto k_cache = t::MakeValues(cache_count, 0xDEADBEEFU, 1.0F);
  const auto v_cache = t::MakeValues(cache_count, 0x0BADF00DU, 1.0F);
  const auto gate = t::MakeValues(q_count, 0xC0FFEE11U, 3.0F);

  t::HipBuffer<float> d_q(query.size());
  t::HipBuffer<float> d_k(k_cache.size());
  t::HipBuffer<float> d_v(v_cache.size());
  t::HipBuffer<float> d_gate(gate.size());
  t::HipBuffer<float> d_out(q_count);
  t::HipBuffer<float> d_scratch(static_cast<std::size_t>(kHeads) * n_kv);
  t::HipBuffer<float> d_part(static_cast<std::size_t>(kHeads) * 32U *
                             (kHeadDim + 2U));
  t::Upload(&d_q, query);
  t::Upload(&d_k, k_cache);
  t::Upload(&d_v, v_cache);
  t::Upload(&d_gate, gate);
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  q::AttentionDecode(d_q.get(), d_k.get(), d_v.get(), d_gate.get(), d_out.get(),
                     d_scratch.get(), d_part.get(), n_kv, kHeads, kKvHeads,
                     kHeadDim, scale, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "Attention synchronization");
  const auto got = t::Download(&d_out, q_count);

  std::vector<float> ref(q_count, 0.0F);
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    const std::uint32_t kvh = h / kGroup;
    const float* qh = query.data() + static_cast<std::size_t>(h) * kHeadDim;
    std::vector<double> scores(n_kv);
    double max_score = -INFINITY;
    for (std::uint32_t j = 0; j < n_kv; ++j) {
      const float* kj =
          k_cache.data() +
          (static_cast<std::size_t>(j) * kKvHeads + kvh) * kHeadDim;
      double dot = 0.0;
      for (std::uint32_t i = 0; i < kHeadDim; ++i) {
        dot += static_cast<double>(qh[i]) * kj[i];
      }
      scores[j] = dot * scale;
      max_score = std::max(max_score, scores[j]);
    }
    double denom = 0.0;
    for (double& sc : scores) {
      sc = std::exp(sc - max_score);
      denom += sc;
    }
    float* oh = ref.data() + static_cast<std::size_t>(h) * kHeadDim;
    for (std::uint32_t j = 0; j < n_kv; ++j) {
      const float p = static_cast<float>(scores[j] / denom);
      const float* vj =
          v_cache.data() +
          (static_cast<std::size_t>(j) * kKvHeads + kvh) * kHeadDim;
      for (std::uint32_t i = 0; i < kHeadDim; ++i) {
        oh[i] += p * vj[i];
      }
    }
    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
      oh[i] *= static_cast<float>(
          t::SigmoidD(gate[static_cast<std::size_t>(h) * kHeadDim + i]));
    }
  }
  return Check("AttentionDecode n_kv=" + std::to_string(n_kv),
               t::WorstRelative(ref, got), 1e-4);
}

// Causal gated attention reference for `tokens` rows starting at absolute
// position `start`; token `t` attends positions [0, start + t].
std::vector<float> Reference(const std::vector<float>& query,
                             const std::vector<float>& gate,
                             const std::vector<float>& k_cache,
                             const std::vector<float>& v_cache,
                             std::uint32_t start, std::uint32_t tokens) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  std::vector<float> ref(static_cast<std::size_t>(tokens) * kHeads * kHeadDim,
                         0.0F);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    const std::uint32_t n_kv = start + token + 1;
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::uint32_t kvh = h / kGroup;
      const float* qh = query.data() +
                        (static_cast<std::size_t>(token) * kHeads + h) *
                            kHeadDim;
      std::vector<double> scores(n_kv);
      double max_score = -INFINITY;
      for (std::uint32_t j = 0; j < n_kv; ++j) {
        const float* kj =
            k_cache.data() +
            (static_cast<std::size_t>(j) * kKvHeads + kvh) * kHeadDim;
        double dot = 0.0;
        for (std::uint32_t i = 0; i < kHeadDim; ++i) {
          dot += static_cast<double>(qh[i]) * kj[i];
        }
        scores[j] = dot * scale;
        max_score = std::max(max_score, scores[j]);
      }
      double denom = 0.0;
      for (double& sc : scores) {
        sc = std::exp(sc - max_score);
        denom += sc;
      }
      float* oh = ref.data() +
                  (static_cast<std::size_t>(token) * kHeads + h) * kHeadDim;
      for (std::uint32_t j = 0; j < n_kv; ++j) {
        const float p = static_cast<float>(scores[j] / denom);
        const float* vj =
            v_cache.data() +
            (static_cast<std::size_t>(j) * kKvHeads + kvh) * kHeadDim;
        for (std::uint32_t i = 0; i < kHeadDim; ++i) {
          oh[i] += p * vj[i];
        }
      }
      for (std::uint32_t i = 0; i < kHeadDim; ++i) {
        const std::size_t g =
            (static_cast<std::size_t>(token) * kHeads + h) * kHeadDim + i;
        oh[i] *= static_cast<float>(t::SigmoidD(gate[g]));
      }
    }
  }
  return ref;
}

// Prefill over a chunk starting at absolute position `start` must reproduce,
// for every token, the causal softmax the decode kernel produces when it is
// fed the same keys/values up to that token's position.
bool RunPrefillCase(std::uint32_t start, std::uint32_t tokens) {
  const std::uint32_t causal_max = start + tokens;
  const std::size_t q_count =
      static_cast<std::size_t>(tokens) * kHeads * kHeadDim;
  const std::size_t cache_count =
      static_cast<std::size_t>(causal_max) * kKvHeads * kHeadDim;
  const auto query = t::MakeValues(q_count, 0x5150A5A5U, 1.0F);
  const auto gate = t::MakeValues(q_count, 0x2468ACE1U, 3.0F);
  const auto k_cache = t::MakeValues(cache_count, 0xDEADBEEFU, 1.0F);
  const auto v_cache = t::MakeValues(cache_count, 0x0BADF00DU, 1.0F);

  t::HipBuffer<float> d_q(q_count);
  t::HipBuffer<float> d_gate(q_count);
  t::HipBuffer<float> d_k(cache_count);
  t::HipBuffer<float> d_v(cache_count);
  t::HipBuffer<float> d_out(q_count);
  t::Upload(&d_q, query);
  t::Upload(&d_gate, gate);
  t::Upload(&d_k, k_cache);
  t::Upload(&d_v, v_cache);
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  q::AttentionPrefill(d_q.get(), d_k.get(), d_v.get(), nullptr, nullptr,
                      d_gate.get(), d_out.get(), start, tokens, kHeads, kKvHeads,
                      kHeadDim, scale, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "AttentionPrefill synchronization");
  const auto got = t::Download(&d_out, q_count);

  return Check("AttentionPrefill start=" + std::to_string(start) +
                   " tokens=" + std::to_string(tokens),
               t::WorstRelative(Reference(query, gate, k_cache, v_cache, start,
                                          tokens),
                                got),
               1e-4);
}

// The matrix-core prefill path (FP16 KV mirror) must reproduce the scalar
// oracle within FP16 rounding. The caller publishes the FP32 cache into the
// FP16 planes with KvCacheWriteF16, then AttentionPrefill dispatches to WMMA.
bool RunWmmaCase(std::uint32_t start, std::uint32_t tokens) {
  const std::uint32_t causal_max = start + tokens;
  const std::size_t q_count =
      static_cast<std::size_t>(tokens) * kHeads * kHeadDim;
  const std::size_t cache_count =
      static_cast<std::size_t>(causal_max) * kKvHeads * kHeadDim;
  const auto query = t::MakeValues(q_count, 0x5150A5A5U, 1.0F);
  const auto gate = t::MakeValues(q_count, 0x2468ACE1U, 3.0F);
  const auto k_cache = t::MakeValues(cache_count, 0xDEADBEEFU, 1.0F);
  const auto v_cache = t::MakeValues(cache_count, 0x0BADF00DU, 1.0F);

  t::HipBuffer<float> d_q(q_count);
  t::HipBuffer<float> d_gate(q_count);
  t::HipBuffer<float> d_k(cache_count);
  t::HipBuffer<float> d_v(cache_count);
  t::HipBuffer<float> d_out(q_count);
  t::HipBuffer<std::uint16_t> d_kf16(cache_count);
  t::HipBuffer<std::uint16_t> d_vf16(cache_count);
  t::Upload(&d_q, query);
  t::Upload(&d_gate, gate);
  t::Upload(&d_k, k_cache);
  t::Upload(&d_v, v_cache);
  q::KvCacheWriteF16(d_k.get(), d_v.get(), d_kf16.get(), d_vf16.get(), 0U,
                     cache_count, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "KvCacheWriteF16 synchronization");
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  q::AttentionPrefill(d_q.get(), d_k.get(), d_v.get(), d_kf16.get(),
                      d_vf16.get(), d_gate.get(), d_out.get(), start, tokens,
                      kHeads, kKvHeads, kHeadDim, scale, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "AttentionPrefill WMMA synchronization");
  const auto got = t::Download(&d_out, q_count);

  return Check("AttentionPrefillWmma start=" + std::to_string(start) +
                   " tokens=" + std::to_string(tokens),
               t::WorstRelativeToScale(Reference(query, gate, k_cache, v_cache,
                                                start, tokens),
                                       got),
               2e-3);
}

// Sparse causal reference: token at absolute position p attends the causal
// prefix [0, p] restricted to the first `sink` keys and the last `window` keys,
// i.e. key j is kept when j < sink or p - j < window.
std::vector<float> ReferenceSparse(const std::vector<float>& query,
                                   const std::vector<float>& gate,
                                   const std::vector<float>& k_cache,
                                   const std::vector<float>& v_cache,
                                   std::uint32_t start, std::uint32_t tokens,
                                   std::uint32_t window, std::uint32_t sink) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  std::vector<float> ref(static_cast<std::size_t>(tokens) * kHeads * kHeadDim,
                         0.0F);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    const std::uint32_t p = start + token;
    const std::uint32_t n_kv = p + 1;
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::uint32_t kvh = h / kGroup;
      const float* qh = query.data() +
                        (static_cast<std::size_t>(token) * kHeads + h) *
                            kHeadDim;
      std::vector<double> scores(n_kv, -INFINITY);
      double max_score = -INFINITY;
      for (std::uint32_t j = 0; j < n_kv; ++j) {
        if (!(j < sink || p - j < window)) {
          continue;
        }
        const float* kj =
            k_cache.data() +
            (static_cast<std::size_t>(j) * kKvHeads + kvh) * kHeadDim;
        double dot = 0.0;
        for (std::uint32_t i = 0; i < kHeadDim; ++i) {
          dot += static_cast<double>(qh[i]) * kj[i];
        }
        scores[j] = dot * scale;
        max_score = std::max(max_score, scores[j]);
      }
      double denom = 0.0;
      for (double& sc : scores) {
        sc = std::exp(sc - max_score);
        denom += sc;
      }
      float* oh = ref.data() +
                  (static_cast<std::size_t>(token) * kHeads + h) * kHeadDim;
      for (std::uint32_t j = 0; j < n_kv; ++j) {
        const float pr = static_cast<float>(scores[j] / denom);
        const float* vj =
            v_cache.data() +
            (static_cast<std::size_t>(j) * kKvHeads + kvh) * kHeadDim;
        for (std::uint32_t i = 0; i < kHeadDim; ++i) {
          oh[i] += pr * vj[i];
        }
      }
      for (std::uint32_t i = 0; i < kHeadDim; ++i) {
        const std::size_t g =
            (static_cast<std::size_t>(token) * kHeads + h) * kHeadDim + i;
        oh[i] *= static_cast<float>(t::SigmoidD(gate[g]));
      }
    }
  }
  return ref;
}

// The sparse WMMA prefill must reproduce the masked reference within FP16
// rounding for a range of window/sink settings that straddle the 16-key tile.
bool RunWmmaSparseCase(std::uint32_t start, std::uint32_t tokens,
                       std::uint32_t window, std::uint32_t sink) {
  const std::uint32_t causal_max = start + tokens;
  const std::size_t q_count =
      static_cast<std::size_t>(tokens) * kHeads * kHeadDim;
  const std::size_t cache_count =
      static_cast<std::size_t>(causal_max) * kKvHeads * kHeadDim;
  const auto query = t::MakeValues(q_count, 0x5150A5A5U, 1.0F);
  const auto gate = t::MakeValues(q_count, 0x2468ACE1U, 3.0F);
  const auto k_cache = t::MakeValues(cache_count, 0xDEADBEEFU, 1.0F);
  const auto v_cache = t::MakeValues(cache_count, 0x0BADF00DU, 1.0F);

  t::HipBuffer<float> d_q(q_count);
  t::HipBuffer<float> d_gate(q_count);
  t::HipBuffer<float> d_k(cache_count);
  t::HipBuffer<float> d_v(cache_count);
  t::HipBuffer<float> d_out(q_count);
  t::HipBuffer<std::uint16_t> d_kf16(cache_count);
  t::HipBuffer<std::uint16_t> d_vf16(cache_count);
  t::Upload(&d_q, query);
  t::Upload(&d_gate, gate);
  t::Upload(&d_k, k_cache);
  t::Upload(&d_v, v_cache);
  q::KvCacheWriteF16(d_k.get(), d_v.get(), d_kf16.get(), d_vf16.get(), 0U,
                     cache_count, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "KvCacheWriteF16 synchronization");
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  q::AttentionPrefill(d_q.get(), d_k.get(), d_v.get(), d_kf16.get(),
                      d_vf16.get(), d_gate.get(), d_out.get(), start, tokens,
                      kHeads, kKvHeads, kHeadDim, scale, nullptr, window, sink);
  t::CheckHip(hipDeviceSynchronize(), "AttentionPrefill WMMA sparse sync");
  const auto got = t::Download(&d_out, q_count);

  return Check("AttentionPrefillWmmaSparse start=" + std::to_string(start) +
                   " tokens=" + std::to_string(tokens) +
                   " window=" + std::to_string(window) +
                   " sink=" + std::to_string(sink),
               t::WorstRelativeToScale(
                   ReferenceSparse(query, gate, k_cache, v_cache, start, tokens,
                                   window, sink),
                   got),
               2e-3);
}

// The multi-row verify kernel must reproduce the same causal rows as the
// reference: row o attends [0, start + 1 + o].
bool RunDecodeRowsCase(std::uint32_t start, std::uint32_t tokens) {
  const std::uint32_t causal_max = start + tokens;
  const std::size_t q_count =
      static_cast<std::size_t>(tokens) * kHeads * kHeadDim;
  const std::size_t cache_count =
      static_cast<std::size_t>(causal_max) * kKvHeads * kHeadDim;
  const auto query = t::MakeValues(q_count, 0x5150A5A5U, 1.0F);
  const auto gate = t::MakeValues(q_count, 0x2468ACE1U, 3.0F);
  const auto k_cache = t::MakeValues(cache_count, 0xDEADBEEFU, 1.0F);
  const auto v_cache = t::MakeValues(cache_count, 0x0BADF00DU, 1.0F);

  t::HipBuffer<float> d_q(q_count);
  t::HipBuffer<float> d_gate(q_count);
  t::HipBuffer<float> d_k(cache_count);
  t::HipBuffer<float> d_v(cache_count);
  t::HipBuffer<float> d_out(q_count);
  t::HipBuffer<float> d_part(static_cast<std::size_t>(tokens) * kHeads * 32U *
                             (kHeadDim + 2U));
  t::Upload(&d_q, query);
  t::Upload(&d_gate, gate);
  t::Upload(&d_k, k_cache);
  t::Upload(&d_v, v_cache);
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  q::AttentionDecodeRows(d_q.get(), d_k.get(), d_v.get(), d_gate.get(),
                         d_out.get(), d_part.get(), start + 1U, tokens, kHeads,
                         kKvHeads, kHeadDim, scale, nullptr);
  t::CheckHip(hipDeviceSynchronize(), "AttentionDecodeRows synchronization");
  const auto got = t::Download(&d_out, q_count);

  return Check("AttentionDecodeRows start=" + std::to_string(start) +
                   " tokens=" + std::to_string(tokens),
               t::WorstRelative(Reference(query, gate, k_cache, v_cache, start,
                                          tokens),
                                got),
               1e-4);
}

}  // namespace

int main() {
  try {
    bool ok = true;
    for (std::uint32_t n_kv : {1U, 5U, 33U, 128U, 1000U, 2400U}) {
      ok = RunCase(n_kv) && ok;
    }
    // Chunk starts and lengths that straddle the kernel's 128-wide softmax tile.
    for (const auto& cs : std::vector<std::pair<std::uint32_t, std::uint32_t>>{
             {0U, 1U}, {0U, 8U}, {5U, 7U}, {100U, 40U}, {200U, 64U}}) {
      ok = RunPrefillCase(cs.first, cs.second) && ok;
    }
    // Matrix-core (WMMA) prefill over the FP16 KV mirror, straddling the
    // 32-row query tile and the 16-key inner tile at several depths.
    for (const auto& cs : std::vector<std::pair<std::uint32_t, std::uint32_t>>{
             {0U, 1U}, {0U, 32U}, {0U, 33U}, {5U, 7U}, {100U, 40U},
             {200U, 64U}, {0U, 128U}, {1024U, 96U}}) {
      ok = RunWmmaCase(cs.first, cs.second) && ok;
    }
    // Sparse (sliding-window + sink) WMMA prefill: windows/sinks that straddle
    // the 16-key tile and leave a non-empty dropped middle range.
    for (const auto& c : std::vector<std::tuple<std::uint32_t, std::uint32_t,
                                                std::uint32_t, std::uint32_t>>{
             {0U, 128U, 32U, 0U}, {0U, 128U, 32U, 4U}, {0U, 128U, 16U, 16U},
             {100U, 40U, 32U, 4U}, {200U, 64U, 48U, 8U}, {1024U, 96U, 64U, 16U},
             {0U, 33U, 8U, 0U}, {5U, 7U, 4U, 2U}}) {
      ok = RunWmmaSparseCase(std::get<0>(c), std::get<1>(c), std::get<2>(c),
                             std::get<3>(c)) &&
           ok;
    }
    // Multi-row verify kernel: split boundaries at 64 and the 32-split cap.
    for (std::uint32_t start : {0U, 5U, 63U, 100U, 2047U, 2300U}) {
      for (std::uint32_t tokens = 2U; tokens <= 5U; ++tokens) {
        ok = RunDecodeRowsCase(start, tokens) && ok;
      }
    }
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}