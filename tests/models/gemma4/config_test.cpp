#include "src/models/gemma4/config.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace g4 = gufo::models::gemma4;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

using Value =
    std::variant<std::uint32_t, float, std::string, std::vector<std::uint64_t>,
                 std::vector<std::string>, std::vector<bool>>;
using Metadata = std::map<std::string, Value>;

// Minimal metadata-only GGUF: no model or GPU is needed to test the loader.
std::optional<g4::Config> Parse(const Metadata& fields) {
  std::vector<std::uint8_t> bytes;
  const auto pod = [&]<class T>(T value) {
    const auto begin = bytes.size();
    bytes.resize(begin + sizeof(value));
    std::memcpy(bytes.data() + begin, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(std::uint64_t{value.size()});
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{0});
  pod(std::uint64_t{fields.size()});
  for (const auto& [key, value] : fields) {
    text(key);
    std::visit(
        [&](const auto& item) {
          using T = std::decay_t<decltype(item)>;
          if constexpr (std::is_same_v<T, std::uint32_t>) {
            pod(std::uint32_t{4});
            pod(item);
          } else if constexpr (std::is_same_v<T, float>) {
            pod(std::uint32_t{6});
            pod(item);
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t{8});
            text(item);
          } else if constexpr (std::is_same_v<T, std::vector<std::uint64_t>>) {
            pod(std::uint32_t{9});
            pod(std::uint32_t{10});
            pod(std::uint64_t{item.size()});
            for (const auto entry : item)
              pod(entry);
          } else if constexpr (std::is_same_v<T, std::vector<bool>>) {
            pod(std::uint32_t{9});
            pod(std::uint32_t{7});
            pod(std::uint64_t{item.size()});
            for (const bool entry : item)
              pod(static_cast<std::uint8_t>(entry ? 1U : 0U));
          } else {  // std::vector<std::string>
            pod(std::uint32_t{9});
            pod(std::uint32_t{8});
            pod(std::uint64_t{item.size()});
            for (const auto& entry : item)
              text(entry);
          }
        },
        value);
  }
  bytes.resize((bytes.size() + 31) / 32 * 32);
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  assert(reader && error.empty());
  const auto result = g4::Config::FromGguf(*reader, &error);
  assert(result.has_value() == error.empty());
  return result;
}

Metadata ValidTrunkMetadata() {
  Metadata fields{
      {"general.architecture", std::string{"gemma4"}},
      {"tokenizer.ggml.tokens", std::vector<std::string>{"a", "b", "c"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 30},
           {"embedding_length", 2816},
           {"context_length", 262144},
           {"attention.head_count", 16},
           {"attention.key_length", 512},
           {"attention.value_length", 512},
           {"attention.key_length_swa", 256},
           {"attention.value_length_swa", 256},
           {"rope.dimension_count", 512},
           {"rope.dimension_count_swa", 256},
           {"attention.sliding_window", 1024},
           {"attention.shared_kv_layers", 0},
           {"feed_forward_length", 2112},
           {"expert_count", 128},
           {"expert_used_count", 8},
           {"expert_feed_forward_length", 704},
           {"embedding_length_per_layer_input", 0}}) {
    fields[std::string{"gemma4."} + key] = value;
  }
  fields["gemma4.attention.layer_norm_rms_epsilon"] = 1e-6F;
  fields["gemma4.rope.freq_base"] = 1e6F;
  fields["gemma4.rope.freq_base_swa"] = 1e4F;
  fields["gemma4.final_logit_softcapping"] = 30.0F;
  std::vector<std::uint64_t> kv_heads;
  std::vector<bool> pattern;
  for (std::uint32_t layer = 0; layer < 30; ++layer) {
    kv_heads.push_back((layer % 6) == 5 ? 2U : 8U);
    pattern.push_back((layer % 6) != 5);
  }
  fields["gemma4.attention.head_count_kv"] = kv_heads;
  fields["gemma4.attention.sliding_window_pattern"] = pattern;
  return fields;
}

Metadata ValidDraftMetadata() {
  Metadata fields{
      {"general.architecture", std::string{"gemma4-assistant"}},
      {"tokenizer.ggml.tokens", std::vector<std::string>{"a", "b", "c"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 4},
           {"embedding_length", 1024},
           {"embedding_length_out", 2816},
           {"context_length", 262144},
           {"attention.head_count", 16},
           {"attention.key_length", 512},
           {"attention.value_length", 512},
           {"attention.key_length_swa", 256},
           {"attention.value_length_swa", 256},
           {"rope.dimension_count", 512},
           {"rope.dimension_count_swa", 256},
           {"attention.sliding_window", 1024},
           {"attention.shared_kv_layers", 4},
           {"feed_forward_length", 8192},
           {"nextn_predict_layers", 4},
           {"embedding_length_per_layer_input", 0}}) {
    fields[std::string{"gemma4-assistant."} + key] = value;
  }
  fields["gemma4-assistant.attention.layer_norm_rms_epsilon"] = 1e-6F;
  fields["gemma4-assistant.rope.freq_base"] = 1e6F;
  fields["gemma4-assistant.rope.freq_base_swa"] = 1e4F;
  fields["gemma4-assistant.attention.head_count_kv"] =
      std::vector<std::uint64_t>{8, 8, 8, 2};
  fields["gemma4-assistant.attention.sliding_window_pattern"] =
      std::vector<bool>{true, true, true, false};
  return fields;
}

void CheckValidTrunk() {
  const auto config = Parse(ValidTrunkMetadata());
  Require(config.has_value(), "valid trunk metadata rejected");
  Require(!config->is_draft && config->num_layers == 30, "trunk misclassified");
  Require(config->vocab_size == 3, "vocab not read from the token list");
  Require(config->hidden_size == 2816 && config->context_length == 262144,
          "core dimensions incorrect");
  Require(config->logit_softcap == 30.0F, "softcap incorrect");
  // SWA on five of six layers; the sixth is full attention.
  Require(config->IsSwa(0) && config->IsSwa(4) && !config->IsSwa(5) &&
              !config->IsSwa(29),
          "sliding-window pattern misclassified");
  Require(config->HeadDim(0) == 256 && config->HeadDim(5) == 512,
          "per-class head dimensions wrong");
  Require(config->RopeTheta(0) == 1e4F && config->RopeTheta(5) == 1e6F,
          "per-class rope bases wrong");
  Require(config->AttentionQDim(0) == 4096 && config->AttentionQDim(5) == 8192,
          "attention Q dimensions wrong");
  Require(
      config->AttentionKvDim(0) == 2048 && config->AttentionKvDim(5) == 1024,
      "attention KV dimensions wrong");
  Require(config->HasVProjection(0) && !config->HasVProjection(5),
          "V projection presence wrong");
  Require(config->HasKv(0) && config->HasKv(29),
          "trunk layers must all own KV");
}

void CheckValidDraft() {
  const auto trunk = Parse(ValidTrunkMetadata());
  const auto draft = Parse(ValidDraftMetadata());
  Require(draft.has_value(), "valid draft metadata rejected");
  Require(draft->is_draft && draft->num_layers == 4, "draft misclassified");
  Require(draft->hidden_size == 1024 && draft->hidden_size_out == 2816,
          "draft dimensions incorrect");
  Require(draft->logit_softcap == 0.0F, "draft must not softcap logits");
  Require(!draft->HasKv(0) && !draft->HasKv(3),
          "draft layers must read trunk KV, not own any");
  Require(trunk && draft->DraftMatches(*trunk), "matching draft rejected");
}

void CheckDraftCompatibility() {
  const auto trunk = Parse(ValidTrunkMetadata());
  const auto draft = Parse(ValidDraftMetadata());
  assert(trunk && draft);
  Require(draft->DraftTargetLayer(0, *trunk) == 28 &&
              draft->DraftTargetLayer(3, *trunk) == 29,
          "draft KV share map wrong");
  for (auto member :
       {&g4::Config::vocab_size, &g4::Config::hidden_size_out,
        &g4::Config::sliding_window, &g4::Config::shared_kv_layers}) {
    auto wrong = *draft;
    ++(wrong.*member);
    Require(!wrong.DraftMatches(*trunk),
            "dimension-compatible draft with different semantics accepted");
  }
  for (auto member : {&g4::Config::rms_eps, &g4::Config::rope_theta,
                      &g4::Config::rope_theta_swa}) {
    auto wrong = *draft;
    wrong.*member *= 2.0F;
    Require(!wrong.DraftMatches(*trunk), "draft numerical mismatch accepted");
  }
  auto wrong_kv = *draft;
  wrong_kv.num_kv_heads[0] = 2;
  Require(!wrong_kv.DraftMatches(*trunk), "draft KV-head mismatch accepted");
  Require(!trunk->DraftMatches(*trunk), "trunk accepted as its own draft");
}

void CheckMalformedMetadata() {
  const auto valid = ValidTrunkMetadata();
  assert(Parse(valid));

  auto arch = valid;
  arch["general.architecture"] = std::string{"gemma3n"};
  assert(!Parse(arch));

  for (const Value& value :
       {Value{std::vector<std::string>{}}, Value{std::uint32_t{262144}},
        Value{std::string{"abc"}}}) {
    auto wrong = valid;
    wrong["tokenizer.ggml.tokens"] = value;
    assert(!Parse(wrong));
  }

  // Per-layer arrays must cover exactly block_count entries, booleans only.
  for (const Value& value :
       {Value{std::vector<std::uint64_t>{8, 8, 8, 8, 8, 2}},
        Value{std::vector<bool>(31, true)},
        Value{std::vector<std::uint64_t>(30, 3U)}}) {
    auto wrong = valid;
    wrong["gemma4.attention.head_count_kv"] = value;
    assert(!Parse(wrong));
  }
  auto short_pattern = valid;
  short_pattern["gemma4.attention.sliding_window_pattern"] =
      std::vector<bool>(29, true);
  assert(!Parse(short_pattern));
  auto bad_pattern = valid;
  bad_pattern["gemma4.attention.sliding_window_pattern"] =
      std::vector<std::uint64_t>(30, 2U);
  assert(!Parse(bad_pattern));

  // Any deviation from the validated kernel geometry is rejected.
  for (const auto& [key, value] :
       {std::pair{"block_count", 29U}, std::pair{"embedding_length", 2560U},
        std::pair{"attention.head_count", 24U},
        std::pair{"attention.key_length", 256U},
        std::pair{"attention.key_length_swa", 128U},
        std::pair{"rope.dimension_count", 256U},
        std::pair{"rope.dimension_count_swa", 512U},
        std::pair{"attention.sliding_window", 2048U},
        std::pair{"attention.shared_kv_layers", 1U},
        std::pair{"feed_forward_length", 4096U}, std::pair{"expert_count", 64U},
        std::pair{"expert_used_count", 10U},
        std::pair{"expert_feed_forward_length", 1024U},
        std::pair{"embedding_length_per_layer_input", 2816U}}) {
    auto wrong = valid;
    wrong[std::string{"gemma4."} + key] = value;
    assert(!Parse(wrong));
  }
  for (const auto& [key, value] :
       {std::pair{"rope.freq_base", 1e7F},
        std::pair{"rope.freq_base_swa", 1e6F},
        std::pair{"final_logit_softcapping", 50.0F}}) {
    auto wrong = valid;
    wrong[std::string{"gemma4."} + key] = value;
    assert(!Parse(wrong));
  }

  // The trunk requires its MoE and softcap keys; the draft must not claim
  // experts, and its share contract keys are mandatory.
  auto no_experts = valid;
  no_experts.erase("gemma4.expert_count");
  assert(!Parse(no_experts));
  auto no_softcap = valid;
  no_softcap.erase("gemma4.final_logit_softcapping");
  assert(!Parse(no_softcap));

  const auto draft_valid = ValidDraftMetadata();
  assert(Parse(draft_valid));
  for (const auto& [key, value] :
       {std::pair{"block_count", 3U}, std::pair{"embedding_length", 2816U},
        std::pair{"embedding_length_out", 1024U},
        std::pair{"attention.shared_kv_layers", 0U},
        std::pair{"nextn_predict_layers", 3U},
        std::pair{"feed_forward_length", 2112U}}) {
    auto wrong = draft_valid;
    wrong[std::string{"gemma4-assistant."} + key] = value;
    assert(!Parse(wrong));
  }
  auto draft_experts = draft_valid;
  draft_experts["gemma4-assistant.expert_count"] = std::uint32_t{128};
  assert(!Parse(draft_experts));
  auto draft_softcap = draft_valid;
  draft_softcap["gemma4-assistant.final_logit_softcapping"] = 30.0F;
  assert(!Parse(draft_softcap));
}

}  // namespace

int main() {
  CheckValidTrunk();
  CheckValidDraft();
  CheckDraftCompatibility();
  CheckMalformedMetadata();
  std::cout << "Gemma-4-26B-A4B metadata checks passed.\n";
}