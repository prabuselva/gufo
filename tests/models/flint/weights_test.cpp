#include "src/models/flint/weights.hpp"

#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fl = gufo::models::flint;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_FLINT_GGUF");
  if (path == nullptr || *path == '\0') {
    std::cout << "SKIP: set GUFO_FLINT_GGUF\n";
    return 77;
  }
  try {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
    Require(reader != nullptr, "open: " + error);
    const auto weights = fl::ModelWeights::Bind(*reader, &error);
    Require(weights.has_value(), "bind: " + error);
    const auto& c = weights->config;
    Require(c.num_layers == 49, "measured artifact is the 49-layer trunk");

    Require(!weights->token_embd.empty() && !weights->output.empty() &&
                !weights->output_norm.empty(),
            "embedding, output and final norm must bind");
    Require(weights->token_embd.cols == c.hidden_size &&
                weights->token_embd.rows == c.vocab_size,
            "token_embd is [hidden -> vocab]");

    // The expected name set must cover the file exactly once.
    std::set<std::string> expected;
    expected.insert(
        {"token_embd.weight", "output.weight", "output_norm.weight"});
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      const std::string p = "blk." + std::to_string(il) + ".";
      for (const char* name :
           {"attn_norm", "ffn_norm", "attn_q", "attn_k", "attn_v",
            "attn_output", "attn_q_norm", "attn_k_norm", "attn_sinks"}) {
        expected.insert(p + name + ".weight");
      }
      if (c.IsMoeLayer(il)) {
        for (const char* name :
             {"ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
              "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"}) {
          expected.insert(p + name + ".weight");
        }
      } else {
        for (const char* name : {"ffn_gate", "ffn_up", "ffn_down"}) {
          expected.insert(p + name + ".weight");
        }
      }
    }
    Require(expected.size() == reader->GetTensorCount(),
            "every file tensor must belong to the expected table (got " +
                std::to_string(reader->GetTensorCount()) + " of " +
                std::to_string(expected.size()) + ")");
    for (const auto& name : expected) {
      Require(reader->FindTensor(name) != nullptr, "missing " + name);
    }

    Require(weights->layers.size() == c.num_layers, "one entry per block");
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      const auto& l = weights->layers[il];
      Require(l.moe == c.IsMoeLayer(il), "layer kind mismatch");
      Require(!l.attn_q.empty() && !l.attn_k.empty() && !l.attn_v.empty() &&
                  !l.attn_out.empty() && !l.attn_q_norm.empty() &&
                  !l.attn_k_norm.empty() && !l.attn_sinks.empty(),
              "attention tensors are required in every layer");
      Require(
          l.attn_q.cols == c.hidden_size && l.attn_q.rows == c.AttentionQDim(),
          "q expands past the hidden size");
      Require(l.attn_k.rows == c.AttentionKvDim() &&
                  l.attn_v.rows == c.AttentionKvDim(),
              "kv rows follow the kv head count");
      Require(
          l.attn_sinks.cols == c.num_heads && l.attn_q_norm.cols == c.head_dim,
          "sink and per-head norm widths");
      if (l.moe) {
        Require(!l.router.empty() && !l.ffn_gate_exps.empty() &&
                    !l.ffn_up_exps.empty() && !l.ffn_down_exps.empty() &&
                    !l.shexp_gate.empty() && !l.shexp_up.empty() &&
                    !l.shexp_down.empty(),
                "MoE layer " + std::to_string(il) + " must bind completely");
        Require(
            l.router.cols == c.hidden_size && l.router.rows == c.num_experts,
            "router maps hidden -> experts");
        Require(l.ffn_gate_exps.experts == c.num_experts &&
                    l.ffn_gate_exps.rows == c.expert_ff &&
                    l.ffn_down_exps.cols == c.expert_ff,
                "stacked experts follow the config");
        Require(l.shexp_gate.rows == c.shared_expert_ff &&
                    l.shexp_down.cols == c.shared_expert_ff,
                "one shared expert stream of the configured width");
        Require(l.ffn_gate_exps.Expert(c.num_experts - 1) >
                    l.ffn_gate_exps.Expert(0),
                "expert stacking addresses stay inside the matrix");
        Require(l.ffn_gate.empty(), "MoE layers carry no dense ffn_gate");
      } else {
        Require(!l.ffn_gate.empty() && !l.ffn_up.empty() && !l.ffn_down.empty(),
                "dense FFN must bind");
        Require(
            l.ffn_gate.rows == c.ffn_length && l.ffn_down.cols == c.ffn_length,
            "dense FFN width");
        Require(l.router.empty() && l.ffn_gate_exps.empty(),
                "leading layers carry no experts");
      }
    }
    std::cout << "flint.weights ok (" << weights->layers.size() << " layers, "
              << expected.size() << " tensors)\n";
  } catch (const std::exception& e) {
    std::cerr << "flint.weights FAILED: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
