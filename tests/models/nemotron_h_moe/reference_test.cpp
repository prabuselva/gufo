// Contracts for the Nemotron 3.5 Bolt scalar oracle. The synthetic section
// builds a miniature trunk in memory (one Mamba-2, two NoPE attention and one
// latent-MoE layer, all F32) and pins the end-to-end step semantics: residual
// flow, conv/ssm state growth, Reset, order sensitivity and every error path.
// The real-artifact section streams the 24GB GGUF through the oracle; it
// skips (77) unless GUFO_NEMOTRON_H_MOE_GGUF points at the artifact.
#include "src/models/nemotron_h_moe/reference.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace nm = gufo::models::nemotron_h_moe;
namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

std::uint32_t ArgMax(std::span<const float> v) {
  std::uint32_t best = 0;
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[best]) {
      best = static_cast<std::uint32_t>(i);
    }
  }
  return best;
}

double LogitSum(std::span<const float> v) {
  double s = 0.0;
  for (const float x : v) {
    s += static_cast<double>(x);
  }
  return s;
}

void CheckLogits(std::span<const float> logits, std::uint32_t vocab,
                 const std::string& tag) {
  bool finite = true;
  float lo = std::numeric_limits<float>::infinity();
  float hi = -std::numeric_limits<float>::infinity();
  for (const float v : logits) {
    finite = finite && std::isfinite(v);
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  Expect(finite, tag + " logits are finite");
  Expect(hi > lo, tag + " logits are non-degenerate");
  Expect(ArgMax(logits) < vocab, tag + " argmax in range");
}

// Deterministic LCG in [-1, 1); the top 53 bits feed the mantissa so the
// draw really spans the interval. The oracle's reductions are stable, so
// snapshots of a synthetic trunk stay comparable across runs and thread
// counts as long as this helper does not change.
double NextRnd(std::uint64_t& s) {
  s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  constexpr double kUnit = 1.0 / static_cast<double>(1ULL << 53);
  return static_cast<double>(s >> 11) * (2.0 * kUnit) - 1.0;
}

struct TestTensor {
  std::vector<float> mem;
  nm::TensorRef Ref(std::uint64_t cols, std::uint64_t experts = 1) {
    nm::TensorRef t;
    t.data = mem.data();
    t.type = gufo::core::GgmlType::kF32;
    t.cols = cols;
    t.rows = mem.size() / cols / experts;
    t.experts = experts;
    t.name = "synthetic";
    return t;
  }
};

}  // namespace
namespace {

// Miniature but structurally faithful: every mamba/attention/MoE branch of
// the oracle runs with real state semantics at dimensions a unit test can
// finish instantly.
nm::ModelWeights BuildSynthetic(
    std::vector<std::unique_ptr<TestTensor>>& arena) {
  std::uint64_t rnd = 0x9E3779B97F4A7C15ULL;
  auto mk = [&](std::size_t n, float scale) {
    auto t = std::make_unique<TestTensor>();
    t->mem.resize(n);
    for (float& v : t->mem) {
      v = static_cast<float>(NextRnd(rnd) * scale);
    }
    arena.push_back(std::move(t));
    return arena.back().get();
  };

  nm::Config c;
  c.profile_name = "synthetic";
  c.num_layers = 4;
  c.hidden_size = 8;
  c.vocab_size = 11;
  c.context_length = 32;
  c.rms_eps = 1e-5F;
  c.num_heads = 4;
  c.head_dim = 2;
  c.head_dim_v = 2;
  c.head_count_kv = {0, 2, 0, 2};
  c.ssm_conv_kernel = 3;
  c.ssm_state_size = 3;
  c.ssm_group_count = 1;
  c.ssm_inner_size = 4;
  c.ssm_time_step_rank = 2;
  c.max_experts = 5;
  c.shared_expert_ff = 6;
  c.shared_experts = 1;
  c.moe_latent_size = 4;
  c.expert_weights_scale = 5.0F;
  c.expert_weights_norm = true;
  c.layer_feed_forward = {0, 0, 6, 0};
  c.layer_expert_ff = {0, 0, 3, 0};
  c.layer_expert_used = {0, 0, 2, 0};

  nm::ModelWeights w;
  w.config = c;
  TestTensor* embd = mk(c.vocab_size * c.hidden_size, 0.5F);
  TestTensor* onorm = mk(c.hidden_size, 0.0F);
  for (float& v : onorm->mem) {
    v = 1.0F;
  }
  w.token_embd = embd->Ref(c.hidden_size);
  w.output = w.token_embd;
  w.output_norm = onorm->Ref(c.hidden_size);

  w.layers.resize(4);
  for (std::uint32_t l = 0; l < 4; ++l) {
    nm::LayerWeights& lw = w.layers[l];
    lw.kind = c.Kind(l);
    TestTensor* an = mk(c.hidden_size, 0.0F);
    for (float& v : an->mem) {
      v = 1.0F;
    }
    lw.attn_norm = an->Ref(c.hidden_size);
    if (lw.kind == nm::Config::LayerKind::kMamba) {
      lw.ssm_in = mk(c.SsmInProj(), 1.0F)->Ref(c.hidden_size);
      TestTensor* in_s = mk(1, 0.0F);
      in_s->mem[0] = 0.25F;
      lw.ssm_in_s = in_s->Ref(1);
      lw.ssm_conv = mk(c.SsmConvChannels() * c.ssm_conv_kernel, 0.7F)
                        ->Ref(c.ssm_conv_kernel);
      lw.ssm_conv_b = mk(c.SsmConvChannels(), 0.3F)->Ref(c.SsmConvChannels());
      TestTensor* a = mk(2, 0.0F);
      for (float& v : a->mem) {
        v = -0.05F - static_cast<float>(NextRnd(rnd) * 0.5 + 0.5);
      }
      lw.ssm_a = a->Ref(1);
      lw.ssm_d = mk(2, 0.5F)->Ref(1);
      lw.ssm_dt_b = mk(2, 0.4F)->Ref(c.ssm_time_step_rank);
      TestTensor* sn = mk(c.ssm_inner_size, 0.0F);
      for (float& v : sn->mem) {
        v = 1.0F;
      }
      lw.ssm_norm = sn->Ref(c.ssm_inner_size);  // groups = 1
      lw.ssm_out =
          mk(c.ssm_inner_size * c.hidden_size, 0.6F)->Ref(c.ssm_inner_size);
      TestTensor* out_s = mk(1, 0.0F);
      out_s->mem[0] = 0.5F;
      lw.ssm_out_s = out_s->Ref(1);
      continue;
    }
    if (lw.kind == nm::Config::LayerKind::kAttention) {
      lw.attn_q =
          mk(c.hidden_size * c.AttentionQDim(), 0.5F)->Ref(c.hidden_size);
      lw.attn_k = mk(c.hidden_size * c.head_count_kv[l] * c.head_dim, 0.5F)
                      ->Ref(c.hidden_size);
      lw.attn_v = mk(c.hidden_size * c.head_count_kv[l] * c.head_dim_v, 0.5F)
                      ->Ref(c.hidden_size);
      lw.attn_out =
          mk(c.AttentionQDim() * c.hidden_size, 0.5F)->Ref(c.AttentionQDim());
      continue;
    }
    // Latent MoE: 5 experts, top-2, one shared expert.
    lw.n_experts = 5;
    lw.router = mk(c.hidden_size * lw.n_experts, 0.7F)->Ref(c.hidden_size);
    lw.exp_probs_b = mk(lw.n_experts, 0.5F)->Ref(lw.n_experts);
    lw.ffn_up_exps = mk(4 * 3 * lw.n_experts, 0.6F)->Ref(4, lw.n_experts);
    lw.ffn_down_exps = mk(3 * 4 * lw.n_experts, 0.6F)->Ref(3, lw.n_experts);
    lw.ffn_up_shexp =
        mk(c.hidden_size * c.shared_expert_ff, 0.5F)->Ref(c.hidden_size);
    lw.ffn_down_shexp =
        mk(c.shared_expert_ff * c.hidden_size, 0.5F)->Ref(c.shared_expert_ff);
    lw.ffn_latent_up = mk(4 * c.hidden_size, 0.5F)->Ref(4);
    lw.ffn_latent_down = mk(c.hidden_size * 4, 0.5F)->Ref(c.hidden_size);
  }
  return w;
}

}  // namespace
int main() {
  // Snapshots of the reduction-stable synthetic trunk. They guard the oracle
  // itself against silent numeric drift; agreement with llama.cpp over the
  // real artifact is the independent contract.
  constexpr std::uint32_t kGoldTok1Argmax = 3;
  constexpr double kGoldTok1Sum = 0.793073;
  constexpr std::uint32_t kGoldSeqArgmax = 2;
  constexpr double kGoldSeqSum = 5.106880;

  std::vector<std::unique_ptr<TestTensor>> arena;
  nm::ModelWeights w = BuildSynthetic(arena);
  Expect(w.layers[0].kind == nm::Config::LayerKind::kMamba, "layer 0 mamba");
  Expect(w.layers[1].kind == nm::Config::LayerKind::kAttention,
         "layer 1 attention");
  Expect(w.layers[2].kind == nm::Config::LayerKind::kMoe, "layer 2 moe");

  std::string error;
  nm::ReferenceModel model(w, 64);
  std::vector<float> logits(w.config.vocab_size);

  // Error paths must not advance the stream.
  Expect(!model.Step(static_cast<std::int32_t>(w.config.vocab_size), logits,
                     &error),
         "token at vocabulary size rejected");
  Expect(!model.Step(-1, logits, &error), "negative token rejected");
  Expect(!model.Step(3, std::span<float>(logits.data(), 4), &error),
         "undersized logits span rejected");
  Expect(model.Position() == 0, "failed steps did not advance");

  std::vector<float> first_logits;
  Expect(model.Step(3, logits, &error), "step token 3");
  CheckLogits(logits, w.config.vocab_size, "tok 3");
  first_logits = logits;
  Expect(ArgMax(first_logits) == kGoldTok1Argmax,
         "tok 3 argmax snapshot, got " + std::to_string(ArgMax(first_logits)));
  Expect(std::abs(LogitSum(first_logits) - kGoldTok1Sum) < 1e-4,
         "tok 3 logit-sum snapshot, got " +
             std::to_string(LogitSum(first_logits)));
  Expect(model.Step(7, logits, &error), "step token 7");
  CheckLogits(logits, w.config.vocab_size, "tok 7");
  Expect(model.Step(1, logits, &error), "step token 1");
  CheckLogits(logits, w.config.vocab_size, "tok 1");
  Expect(model.Position() == 3, "position advanced to 3");
  const double seq_sum = LogitSum(logits);
  Expect(ArgMax(logits) == kGoldSeqArgmax,
         "seq argmax snapshot, got " + std::to_string(ArgMax(logits)));
  Expect(std::abs(seq_sum - kGoldSeqSum) < 1e-4,
         "seq logit-sum snapshot, got " + std::to_string(seq_sum));

  // Reset must replay the first token bit-for-bit from the initial state.
  model.Reset();
  Expect(model.Position() == 0, "reset clears position");
  Expect(model.Step(3, logits, &error), "replay token 3");
  bool identical = true;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    identical = identical && logits[i] == first_logits[i];
  }
  Expect(identical, "reset replay is bit-identical");

  // Recurrent state must make the order observable in the logits.
  nm::ReferenceModel swapped(w, 64);
  Expect(swapped.Step(7, logits, &error), "swap step token 7");
  Expect(swapped.Step(3, logits, &error), "swap step token 3");
  Expect(std::abs(LogitSum(logits) - seq_sum) > 1e-6,
         "order changes the output distribution");

  // Context capacity is enforced on the step that would cross it.
  nm::ReferenceModel tiny(w, 1);
  Expect(tiny.Step(3, logits, &error), "first step within capacity");
  Expect(!tiny.Step(3, logits, &error), "second step exceeds capacity");

  if (failures != 0) {
    std::cerr << failures << " synthetic oracle checks failed\n";
    return 1;
  }
  std::cout << "synthetic oracle checks passed.\n";

  const char* path = std::getenv("GUFO_NEMOTRON_H_MOE_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: set GUFO_NEMOTRON_H_MOE_GGUF for the real artifact\n";
    return 77;
  }
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::cerr << "cannot open " << path << ": " << error << "\n";
    return 1;
  }
  const auto real = nm::ModelWeights::Bind(*reader, &error);
  if (!real.has_value()) {
    std::cerr << "trunk bind failed: " << error << "\n";
    return 1;
  }
  const auto& c = real->config;
  logits.assign(c.vocab_size, 0.0F);
  const std::vector<std::int32_t> prompt = {100, 200, 300};
  // Optional parity contract: dump raw f32 logits, one vocab block per
  // prompt step, for cross-checking against the reference server.
  const char* dump_path = std::getenv("GUFO_NEMOTRON_LOGITS_DUMP");
  std::ofstream dump;
  if (dump_path != nullptr && dump_path[0] != '\0') {
    dump.open(dump_path, std::ios::binary | std::ios::trunc);
    if (!dump) {
      std::cerr << "cannot open logits dump target " << dump_path << "\n";
      return 1;
    }
  }
  nm::ReferenceModel engine(*real, 64);
  const char* layer_dump_path = std::getenv("GUFO_NEMOTRON_LAYERDUMP");
  std::ofstream layer_dump;
  std::vector<float> trace;
  if (layer_dump_path != nullptr && layer_dump_path[0] != '\0') {
    layer_dump.open(layer_dump_path, std::ios::binary | std::ios::trunc);
    if (!layer_dump) {
      std::cerr << "cannot open layer dump target " << layer_dump_path << "\n";
      return 1;
    }
    engine.SetLayerDump(&trace);
  }
  std::vector<float> first_real;
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    if (!engine.Step(prompt[i], logits, &error)) {
      std::cerr << "step " << i << " failed: " << error << "\n";
      return 1;
    }
    if (layer_dump) {
      layer_dump.write(
          reinterpret_cast<const char*>(trace.data()),
          static_cast<std::streamsize>(trace.size() * sizeof(float)));
    }
    CheckLogits(logits, c.vocab_size, "real step " + std::to_string(i));
    std::cout << "real step " << i << " argmax " << ArgMax(logits) << " sum "
              << LogitSum(logits) << "\n";
    if (dump) {
      dump.write(reinterpret_cast<const char*>(logits.data()),
                 static_cast<std::streamsize>(logits.size() * sizeof(float)));
    }
    if (i + 1 == prompt.size()) {
      first_real = logits;
    }
  }
  nm::ReferenceModel replay(*real, 64);
  for (const std::int32_t token : prompt) {
    if (!replay.Step(token, logits, &error)) {
      std::cerr << "replay step failed: " << error << "\n";
      return 1;
    }
  }
  double max_abs = 0.0;
  double ref = 1.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    max_abs = std::max(
        max_abs, static_cast<double>(std::abs(logits[i] - first_real[i])));
    ref = std::max(ref, static_cast<double>(std::abs(first_real[i])));
  }
  Expect(max_abs <= 1e-2 * ref, "real replay is stable");

  if (failures != 0) {
    std::cerr << failures << " real-artifact oracle checks failed\n";
    return 1;
  }
  std::cout << "Nemotron Bolt scalar reference passed.\n";
  return 0;
}
