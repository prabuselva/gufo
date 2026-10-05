// Binds the real Gemma-4-26B-A4B trunk and MTP artifacts and checks that
// every tensor the runtime reads resolves with the expected shape and a
// decodable format. Only GGUF headers are touched (the payload stays mapped),
// so this is CPU-only and cheap even for a 27 GB file. It skips when the
// artifacts are not provided.
#include "src/models/gemma4/weights.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace g4 = gufo::models::gemma4;
namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

void ExpectBound(const g4::TensorRef& t, const std::string& what) {
  Expect(!t.empty(), what + " bound");
  Expect(t.RowBytes() != 0, what + " has a decodable format");
  Expect(t.SizeBytes() != 0, what + " has a non-zero payload");
}

void CheckTrunkLayer(const g4::LayerWeights& l, const g4::Config& c,
                     std::uint32_t il) {
  const std::string tag = "trunk layer " + std::to_string(il);
  ExpectBound(l.attn_norm, tag + " attn_norm");
  ExpectBound(l.attn_q, tag + " attn_q");
  ExpectBound(l.attn_k, tag + " attn_k");
  ExpectBound(l.attn_q_norm, tag + " attn_q_norm");
  ExpectBound(l.attn_k_norm, tag + " attn_k_norm");
  ExpectBound(l.attn_output, tag + " attn_output");
  ExpectBound(l.post_attention_norm, tag + " post_attention_norm");
  ExpectBound(l.ffn_norm, tag + " ffn_norm");
  ExpectBound(l.ffn_gate, tag + " ffn_gate");
  ExpectBound(l.ffn_up, tag + " ffn_up");
  ExpectBound(l.ffn_down, tag + " ffn_down");
  ExpectBound(l.post_ffw_norm, tag + " post_ffw_norm");
  ExpectBound(l.router, tag + " router");
  ExpectBound(l.router_scale, tag + " router_scale");
  ExpectBound(l.pre_ffw_norm_2, tag + " pre_ffw_norm_2");
  ExpectBound(l.post_ffw_norm_1, tag + " post_ffw_norm_1");
  ExpectBound(l.post_ffw_norm_2, tag + " post_ffw_norm_2");
  ExpectBound(l.ffn_gate_up_exps, tag + " ffn_gate_up_exps");
  ExpectBound(l.ffn_down_exps, tag + " ffn_down_exps");
  ExpectBound(l.ffn_down_exps_scale, tag + " ffn_down_exps_scale");
  ExpectBound(l.layer_output_scale, tag + " layer_output_scale");
  if (c.HasVProjection(il)) {
    ExpectBound(l.attn_v, tag + " attn_v");
  } else {
    Expect(l.attn_v.empty(), tag + " has no V projection");
  }
}

void CheckDraftLayer(const g4::LayerWeights& l, std::uint32_t il) {
  const std::string tag = "draft layer " + std::to_string(il);
  ExpectBound(l.attn_norm, tag + " attn_norm");
  ExpectBound(l.attn_q, tag + " attn_q");
  ExpectBound(l.attn_q_norm, tag + " attn_q_norm");
  ExpectBound(l.attn_output, tag + " attn_output");
  ExpectBound(l.post_attention_norm, tag + " post_attention_norm");
  ExpectBound(l.ffn_norm, tag + " ffn_norm");
  ExpectBound(l.ffn_gate, tag + " ffn_gate");
  ExpectBound(l.ffn_up, tag + " ffn_up");
  ExpectBound(l.ffn_down, tag + " ffn_down");
  ExpectBound(l.post_ffw_norm, tag + " post_ffw_norm");
  ExpectBound(l.layer_output_scale, tag + " layer_output_scale");
  Expect(l.attn_k.empty(), tag + " reads trunk K");
  Expect(l.attn_v.empty(), tag + " reads trunk V");
  Expect(l.attn_k_norm.empty(), tag + " reads trunk K norm");
  Expect(l.router.empty(), tag + " is dense");
  Expect(l.ffn_gate_up_exps.empty(), tag + " has no routed experts");
}

}  // namespace

int main() {
  const char* trunk_path = std::getenv("GUFO_GEMMA4_GGUF");
  const char* draft_path = std::getenv("GUFO_GEMMA4_MTP_GGUF");
  if (trunk_path == nullptr || trunk_path[0] == '\0') {
    std::cout << "SKIP: set GUFO_GEMMA4_GGUF\n";
    return 77;
  }

  std::string error;
  const auto trunk_reader =
      gufo::core::GgufReader::OpenFile(trunk_path, &error);
  if (trunk_reader == nullptr) {
    std::cerr << "cannot open " << trunk_path << ": " << error << "\n";
    return 1;
  }
  const auto trunk = g4::ModelWeights::Bind(*trunk_reader, &error);
  if (!trunk.has_value()) {
    std::cerr << "trunk bind failed: " << error << "\n";
    return 1;
  }
  const auto& c = trunk->config;
  Expect(c.num_layers == 30, "trunk layer count");
  Expect(trunk->layers.size() == c.num_layers, "layer vector size");
  ExpectBound(trunk->token_embd, "token_embd");
  ExpectBound(trunk->output, "output");
  ExpectBound(trunk->output_norm, "output_norm");
  ExpectBound(trunk->rope_freqs, "rope_freqs");
  Expect(trunk->token_embd.rows == c.vocab_size, "vocab matches embedding");
  Expect(trunk->output.data == trunk->token_embd.data, "LM head is tied");
  Expect(trunk->rope_freqs.cols == c.head_dim_full / 2, "rope_freqs width");

  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    CheckTrunkLayer(trunk->layers[il], c, il);
  }

  if (draft_path == nullptr || draft_path[0] == '\0') {
    std::cout << "SKIP: set GUFO_GEMMA4_MTP_GGUF for draft checks\n";
    if (failures != 0) {
      std::cerr << failures << " weight-binding checks failed\n";
      return 1;
    }
    return 0;
  }
  const auto draft_reader =
      gufo::core::GgufReader::OpenFile(draft_path, &error);
  if (draft_reader == nullptr) {
    std::cerr << "cannot open " << draft_path << ": " << error << "\n";
    return 1;
  }
  const auto draft = g4::DraftWeights::Bind(*draft_reader, c, &error);
  if (!draft.has_value()) {
    std::cerr << "draft bind failed: " << error << "\n";
    return 1;
  }
  const auto& d = draft->config;
  Expect(d.num_layers == 4, "draft layer count");
  Expect(draft->layers.size() == d.num_layers, "draft layer vector size");
  ExpectBound(draft->pre_projection, "pre_projection");
  ExpectBound(draft->post_projection, "post_projection");
  ExpectBound(draft->token_embd, "draft token_embd");
  ExpectBound(draft->output_norm, "draft output_norm");
  ExpectBound(draft->rope_freqs, "draft rope_freqs");
  Expect(draft->pre_projection.cols == 2 * c.hidden_size,
         "pre_projection consumes hidden + embedding");
  Expect(draft->post_projection.rows == c.hidden_size,
         "post_projection returns trunk width");
  for (std::uint32_t il = 0; il < d.num_layers; ++il) {
    CheckDraftLayer(draft->layers[il], il);
  }

  if (failures != 0) {
    std::cerr << failures << " weight-binding checks failed\n";
    return 1;
  }
  std::cout << "Gemma-4-26B-A4B weight binding passed.\n";
  return 0;
}