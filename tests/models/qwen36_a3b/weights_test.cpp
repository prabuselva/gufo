// Binds the real Qwen3.6-35B-A3B artifact and checks that every tensor the
// runtime reads resolves with the expected shape and a decodable format. Only
// GGUF headers are touched (the payload stays mapped), so this is CPU-only and
// cheap even for a 39 GB file. It skips when the artifact is not provided.
#include "src/models/qwen36_a3b/weights.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace q36 = gufo::models::qwen36_a3b;
namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

void ExpectBound(const q36::TensorRef& t, const std::string& what) {
  Expect(!t.empty(), what + " bound");
  Expect(t.RowBytes() != 0, what + " has a decodable format");
  Expect(t.SizeBytes() != 0, what + " has a non-zero payload");
}

void CheckLayer(const q36::LayerWeights& l, std::uint32_t il, bool linear) {
  const std::string tag = "layer " + std::to_string(il);
  Expect(l.linear == linear, tag + " classification");
  ExpectBound(l.attn_norm, tag + " attn_norm");
  ExpectBound(l.post_attention_norm, tag + " post_attention_norm");
  ExpectBound(l.router, tag + " router");
  ExpectBound(l.ffn_gate_exps, tag + " ffn_gate_exps");
  ExpectBound(l.ffn_up_exps, tag + " ffn_up_exps");
  ExpectBound(l.ffn_down_exps, tag + " ffn_down_exps");
  ExpectBound(l.shexp_gate_inp, tag + " shexp_gate_inp");
  ExpectBound(l.shexp_gate, tag + " shexp_gate");
  ExpectBound(l.shexp_up, tag + " shexp_up");
  ExpectBound(l.shexp_down, tag + " shexp_down");
  if (linear) {
    ExpectBound(l.ssm_qkv, tag + " ssm_qkv");
    ExpectBound(l.ssm_gate, tag + " ssm_gate");
    ExpectBound(l.ssm_conv1d, tag + " ssm_conv1d");
    ExpectBound(l.ssm_alpha, tag + " ssm_alpha");
    ExpectBound(l.ssm_beta, tag + " ssm_beta");
    ExpectBound(l.ssm_dt, tag + " ssm_dt");
    ExpectBound(l.ssm_a, tag + " ssm_a");
    ExpectBound(l.ssm_norm, tag + " ssm_norm");
    ExpectBound(l.ssm_out, tag + " ssm_out");
    Expect(l.attn_q.empty(), tag + " has no full-attention Q");
  } else {
    ExpectBound(l.attn_q, tag + " attn_q");
    ExpectBound(l.attn_k, tag + " attn_k");
    ExpectBound(l.attn_v, tag + " attn_v");
    ExpectBound(l.attn_out, tag + " attn_out");
    ExpectBound(l.attn_q_norm, tag + " attn_q_norm");
    ExpectBound(l.attn_k_norm, tag + " attn_k_norm");
    Expect(l.ssm_qkv.empty(), tag + " has no linear qkv");
  }
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_QWEN36_A3B_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: set GUFO_QWEN36_A3B_GGUF\n";
    return 77;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::cerr << "cannot open " << path << ": " << error << "\n";
    return 1;
  }

  const auto weights = q36::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    std::cerr << "trunk bind failed: " << error << "\n";
    return 1;
  }
  const auto& c = weights->config;
  Expect(c.num_layers == 40, "trunk layer count");
  Expect(c.num_layers_all == c.num_layers + c.nextn_layers, "layer split");
  Expect(weights->layers.size() == c.num_layers, "layer vector size");
  ExpectBound(weights->token_embd, "token_embd");
  ExpectBound(weights->output, "output");
  ExpectBound(weights->output_norm, "output_norm");
  Expect(weights->token_embd.rows == c.vocab_size, "vocab matches embedding");

  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    CheckLayer(weights->layers[il], il, c.IsLinearLayer(il));
  }

  const auto mtp = q36::MtpWeights::Bind(*reader, c, &error);
  if (mtp.has_value()) {
    CheckLayer(mtp->block, c.num_layers, /*linear=*/false);
    ExpectBound(mtp->block.nextn_enorm, "nextn_enorm");
    ExpectBound(mtp->block.nextn_hnorm, "nextn_hnorm");
    ExpectBound(mtp->block.nextn_eh_proj, "nextn_eh_proj");
    ExpectBound(mtp->block.nextn_shared_head_norm, "nextn_shared_head_norm");
  } else {
    std::cout << "MTP block not present; skipped binding checks\n";
  }

  if (failures != 0) {
    std::cerr << failures << " weight-binding checks failed\n";
    return 1;
  }
  std::cout << "Qwen3.6-35B-A3B weight binding passed.\n";
  return 0;
}