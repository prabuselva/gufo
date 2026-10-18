// Binds the real Ling 3.0 artifact (Flash or Tiny) and checks that every
// tensor the runtime reads resolves with the expected shape and a decodable
// format, following the matched geometry profile (fused vs low-rank query,
// MTP block only when the artifact has one). Only GGUF headers are touched
// (the payload stays mapped), so this is CPU-only and cheap even for a 40 GB
// split file. It skips when the artifact is not provided.
#include "src/models/ling3_0/weights.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace l3 = gufo::models::ling3_0;
namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

void ExpectBound(const l3::TensorRef& t, const std::string& what) {
  Expect(!t.empty(), what + " bound");
  Expect(t.RowBytes() != 0, what + " has a decodable format");
  Expect(t.SizeBytes() != 0, what + " has a non-zero payload");
}

void ExpectShape(const l3::TensorRef& t, std::uint64_t cols, std::uint64_t rows,
                 std::uint64_t experts, const std::string& what) {
  Expect(t.cols == cols && t.rows == rows && t.experts == experts,
         what + " shape " + std::to_string(t.cols) + "x" +
             std::to_string(t.rows) + "x" + std::to_string(t.experts));
}

void CheckSharedExpert(const l3::LayerWeights& l, const std::string& tag) {
  ExpectBound(l.router, tag + " router");
  ExpectBound(l.exp_probs_b, tag + " exp_probs_b");
  ExpectBound(l.ffn_gate_exps, tag + " ffn_gate_exps");
  ExpectBound(l.ffn_up_exps, tag + " ffn_up_exps");
  ExpectBound(l.ffn_down_exps, tag + " ffn_down_exps");
  ExpectBound(l.shexp_gate, tag + " shexp_gate");
  ExpectBound(l.shexp_up, tag + " shexp_up");
  ExpectBound(l.shexp_down, tag + " shexp_down");
}

void CheckLayer(const l3::LayerWeights& l, const l3::Config& c,
                std::uint32_t il) {
  const std::string tag = "layer " + std::to_string(il);
  ExpectBound(l.attn_norm, tag + " attn_norm");
  ExpectBound(l.ffn_norm, tag + " ffn_norm");
  const std::uint64_t hidden = c.hidden_size;
  if (l.mla) {
    Expect(c.IsMlaLayer(il), tag + " MLA classification");
    if (c.mla_q_split) {
      ExpectShape(l.attn_q_a, hidden, c.mla_q_lora, 1, tag + " attn_q_a");
      ExpectShape(l.attn_q_a_norm, c.mla_q_lora, 1, 1, tag + " attn_q_a_norm");
      ExpectShape(l.attn_q_b, c.mla_q_lora, c.MlaQDim(), 1, tag + " attn_q_b");
      Expect(l.attn_q.empty(), tag + " has no fused q");
    } else {
      ExpectShape(l.attn_q, hidden, c.MlaQDim(), 1, tag + " attn_q");
    }
    ExpectShape(l.attn_kv_a, hidden, c.MlaKeyDim(), 1, tag + " attn_kv_a");
    ExpectShape(l.attn_kv_a_norm, c.kv_lora_rank, 1, 1,
                tag + " attn_kv_a_norm");
    ExpectShape(l.attn_k_b, c.MlaNopeDim(), c.kv_lora_rank, c.num_heads,
                tag + " attn_k_b");
    ExpectShape(l.attn_v_b, c.kv_lora_rank, c.mla_value_dim, c.num_heads,
                tag + " attn_v_b");
    ExpectShape(l.attn_gate, hidden, c.num_heads, 1, tag + " attn_gate");
    ExpectShape(l.attn_out, c.MlaVDim(), hidden, 1, tag + " attn_out");
    Expect(l.ssm_q.empty(), tag + " has no KDA q");
  } else {
    Expect(!c.IsMlaLayer(il), tag + " KDA classification");
    const std::uint64_t inner = c.SsmInnerSize();
    ExpectShape(l.ssm_q, hidden, inner, 1, tag + " ssm_q");
    ExpectShape(l.ssm_k, hidden, inner, 1, tag + " ssm_k");
    ExpectShape(l.ssm_v, hidden, inner, 1, tag + " ssm_v");
    // [kern, 1, inner]: tap index is the contiguous dimension (ne[0]).
    ExpectShape(l.ssm_conv_q, c.ssm_conv_kernel, 1, inner, tag + " ssm_conv_q");
    ExpectShape(l.ssm_conv_k, c.ssm_conv_kernel, 1, inner, tag + " ssm_conv_k");
    ExpectShape(l.ssm_conv_v, c.ssm_conv_kernel, 1, inner, tag + " ssm_conv_v");
    ExpectShape(l.ssm_f, hidden, inner, 1, tag + " ssm_f");
    ExpectShape(l.ssm_g, hidden, inner, 1, tag + " ssm_g");
    ExpectShape(l.ssm_beta, hidden, c.num_heads, 1, tag + " ssm_beta");
    ExpectShape(l.ssm_a, 1, c.num_heads, 1, tag + " ssm_a");
    ExpectShape(l.ssm_dt, inner, 1, 1, tag + " ssm_dt");
    ExpectShape(l.ssm_norm, c.ssm_head_dim, 1, 1, tag + " ssm_norm");
    ExpectShape(l.ssm_out, inner, hidden, 1, tag + " ssm_out");
    Expect(l.attn_q.empty(), tag + " has no MLA q");
    Expect(l.attn_q_a.empty() && l.attn_q_b.empty(),
           tag + " has no low-rank q");
  }
  if (c.IsDenseFfnLayer(il)) {
    ExpectShape(l.ffn_gate, hidden, c.dense_ff, 1, tag + " ffn_gate");
    ExpectShape(l.ffn_up, hidden, c.dense_ff, 1, tag + " ffn_up");
    ExpectShape(l.ffn_down, c.dense_ff, hidden, 1, tag + " ffn_down");
    Expect(l.router.empty(), tag + " has no router");
  } else {
    ExpectShape(l.ffn_gate_exps, hidden, c.expert_ff, c.num_experts,
                tag + " ffn_gate_exps");
    ExpectShape(l.ffn_up_exps, hidden, c.expert_ff, c.num_experts,
                tag + " ffn_up_exps");
    ExpectShape(l.ffn_down_exps, c.expert_ff, hidden, c.num_experts,
                tag + " ffn_down_exps");
    ExpectShape(l.shexp_gate, hidden, c.shared_expert_ff, 1,
                tag + " shexp_gate");
    ExpectShape(l.shexp_down, c.shared_expert_ff, hidden, 1,
                tag + " shexp_down");
    Expect(l.ffn_gate.empty(), tag + " has no dense gate");
    CheckSharedExpert(l, tag);
  }
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_LING3_FLASH_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: set GUFO_LING3_FLASH_GGUF\n";
    return 77;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::cerr << "cannot open " << path << ": " << error << "\n";
    return 1;
  }

  const auto weights = l3::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    std::cerr << "trunk bind failed: " << error << "\n";
    return 1;
  }
  const auto& c = weights->config;
  const bool tiny = c.profile_name == "Ling 3.0 Tiny";
  Expect(c.num_layers == (tiny ? 24U : 42U), "trunk layer count");
  Expect(c.num_layers_all == c.num_layers + c.nextn_layers, "layer split");
  Expect(weights->layers.size() == c.num_layers, "layer vector size");
  ExpectBound(weights->token_embd, "token_embd");
  ExpectBound(weights->output, "output");
  ExpectBound(weights->output_norm, "output_norm");
  Expect(weights->token_embd.rows == c.vocab_size, "vocab matches embedding");

  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    CheckLayer(weights->layers[il], c, il);
  }

  const auto mtp = l3::MtpWeights::Bind(*reader, c, &error);
  if (c.nextn_layers == 0) {
    Expect(!mtp.has_value(), "artifact without nextn layers has no MTP");
  } else if (mtp.has_value()) {
    Expect(mtp->block.mla, "MTP block is MLA");
    CheckSharedExpert(mtp->block, "mtp");
    ExpectBound(mtp->block.nextn_enorm, "nextn_enorm");
    ExpectBound(mtp->block.nextn_hnorm, "nextn_hnorm");
    ExpectShape(mtp->block.nextn_eh_proj, 2 * c.hidden_size, c.hidden_size, 1,
                "nextn_eh_proj");
    ExpectBound(mtp->block.nextn_shared_head_norm, "nextn_shared_head_norm");
  } else {
    std::cerr << "FAIL: MTP block did not bind: " << error << "\n";
    ++failures;
  }

  if (failures != 0) {
    std::cerr << failures << " weight-binding checks failed\n";
    return 1;
  }
  std::cout << "Ling 3.0 weight binding passed.\n";
  return 0;
}