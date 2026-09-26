#include "src/models/qwen36_a3b/config.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace q36 = gufo::models::qwen36_a3b;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Builds a trunk Config directly (no GGUF) to exercise MTP compatibility.
q36::Config TrunkConfig() {
  q36::Config trunk;
  trunk.num_layers = 40;
  trunk.num_layers_all = 41;
  trunk.context_length = 262144;
  trunk.hidden_size = 2048;
  trunk.vocab_size = 248320;
  trunk.num_experts = 256;
  trunk.num_experts_used = 8;
  trunk.expert_ff = trunk.shared_expert_ff = 512;
  trunk.num_heads = 16;
  trunk.num_kv_heads = 2;
  trunk.head_dim = 256;
  trunk.rotary_dim = 64;
  trunk.rope_theta = 1e7F;
  trunk.rope_sections = {11, 11, 10, 0};
  return trunk;
}

void CheckSidecarCompatibility() {
  const auto trunk = TrunkConfig();
  auto sidecar = trunk;
  sidecar.nextn_layers = 1;
  Require(sidecar.MtpMatches(trunk), "matching MTP constants rejected");
  for (auto member :
       {&q36::Config::context_length, &q36::Config::rotary_dim,
        &q36::Config::num_experts_used, &q36::Config::shared_expert_ff}) {
    auto wrong = sidecar;
    ++(wrong.*member);
    Require(!wrong.MtpMatches(trunk),
            "dimension-compatible MTP with different semantics accepted");
  }
  for (auto member : {&q36::Config::rms_eps, &q36::Config::rope_theta}) {
    auto wrong = sidecar;
    wrong.*member *= 2.0F;
    Require(!wrong.MtpMatches(trunk),
            "MTP numerical constant mismatch accepted");
  }
  auto wrong = sidecar;
  std::swap(wrong.rope_sections[0], wrong.rope_sections[2]);
  Require(!wrong.MtpMatches(trunk), "MTP RoPE partition mismatch accepted");
  auto no_nextn = trunk;
  Require(!no_nextn.MtpMatches(trunk), "trunk accepted as its own MTP draft");
}

using Value =
    std::variant<std::uint32_t, float, std::string, std::vector<std::uint64_t>,
                 std::vector<std::string>>;
using Metadata = std::map<std::string, Value>;

// Minimal metadata-only GGUF: no model or GPU is needed to test the loader.
std::optional<q36::Config> Parse(const Metadata& fields, bool trunk = true) {
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
  const auto result = q36::Config::FromGguf(*reader, trunk, &error);
  assert(result.has_value() == error.empty());
  return result;
}

Metadata ValidMetadata() {
  Metadata fields{
      {"general.architecture", std::string{"qwen35moe"}},
      {"tokenizer.ggml.tokens", std::vector<std::string>{"a", "b", "c"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 41},
           {"nextn_predict_layers", 1},
           {"embedding_length", 2048},
           {"context_length", 262144},
           {"full_attention_interval", 4},
           {"attention.head_count", 16},
           {"attention.head_count_kv", 2},
           {"attention.key_length", 256},
           {"attention.value_length", 256},
           {"rope.dimension_count", 64},
           {"ssm.conv_kernel", 4},
           {"ssm.state_size", 128},
           {"ssm.group_count", 16},
           {"ssm.time_step_rank", 32},
           {"ssm.inner_size", 4096},
           {"expert_count", 256},
           {"expert_used_count", 8},
           {"expert_feed_forward_length", 512},
           {"expert_shared_feed_forward_length", 512}}) {
    fields[std::string{"qwen35moe."} + key] = value;
  }
  fields["qwen35moe.attention.layer_norm_rms_epsilon"] = 1e-6F;
  fields["qwen35moe.rope.freq_base"] = 1e7F;
  fields["qwen35moe.rope.dimension_sections"] =
      std::vector<std::uint64_t>{11, 11, 10, 0};
  return fields;
}

void CheckValidMetadata() {
  const auto config = Parse(ValidMetadata());
  Require(config.has_value(), "valid metadata rejected");
  Require(config->num_layers == 40 && config->num_layers_all == 41,
          "trunk/MTP layer split incorrect");
  Require(config->vocab_size == 3, "vocab not read from the token list");
  Require(config->hidden_size == 2048 && config->context_length == 262144,
          "core dimensions incorrect");
  Require(config->SsmConvChannels() == 8192, "GDN q|k|v channel count wrong");
  Require(config->AttentionQDim() == 4096 && config->AttentionKvDim() == 512,
          "attention dimensions wrong");
  // Full attention sits on every fourth layer (indices 3, 7, ...).
  Require(!config->IsLinearLayer(3) && !config->IsLinearLayer(39),
          "full-attention layer misclassified");
  Require(config->IsLinearLayer(0) && config->IsLinearLayer(2),
          "linear-attention layer misclassified");
}

void CheckMalformedMetadata() {
  const auto valid = ValidMetadata();
  assert(Parse(valid));

  // Wrong architecture is rejected outright.
  auto arch = valid;
  arch["general.architecture"] = std::string{"qwen3next"};
  assert(!Parse(arch));

  // The vocabulary must come from a non-empty token list.
  for (const Value& value :
       {Value{std::vector<std::string>{}}, Value{std::uint32_t{248320}},
        Value{std::string{"abc"}}}) {
    auto wrong = valid;
    wrong["tokenizer.ggml.tokens"] = value;
    assert(!Parse(wrong));
  }

  // mRoPE sections must be three axes plus a zero pad summing to rotary/2.
  for (const auto& sections :
       {std::vector<std::uint64_t>{11, 11, 10},
        std::vector<std::uint64_t>{11, 11, 10, 0, 0},
        std::vector<std::uint64_t>{11, 11, 10, 1},
        std::vector<std::uint64_t>{11, 11, 11, 0},
        std::vector<std::uint64_t>{0, 16, 16, 0},
        std::vector<std::uint64_t>{UINT64_MAX, 11, 10, 0}}) {
    auto wrong = valid;
    wrong["qwen35moe.rope.dimension_sections"] = sections;
    assert(!Parse(wrong));
  }

  // Any deviation from the validated kernel geometry is rejected.
  for (const auto& [key, value] :
       {std::pair{"block_count", 40U}, std::pair{"nextn_predict_layers", 41U},
        std::pair{"embedding_length", 2560U},
        std::pair{"full_attention_interval", 3U},
        std::pair{"attention.head_count", 24U},
        std::pair{"attention.head_count_kv", 4U},
        std::pair{"attention.key_length", 128U},
        std::pair{"attention.value_length", 128U},
        std::pair{"rope.dimension_count", 128U},
        std::pair{"ssm.state_size", 64U}, std::pair{"ssm.group_count", 8U},
        std::pair{"ssm.time_step_rank", 48U},
        std::pair{"ssm.inner_size", 6144U}, std::pair{"expert_count", 512U},
        std::pair{"expert_used_count", 10U},
        std::pair{"expert_feed_forward_length", 640U},
        std::pair{"expert_shared_feed_forward_length", 640U}}) {
    auto wrong = valid;
    wrong[std::string{"qwen35moe."} + key] = value;
    assert(!Parse(wrong));
  }

  // The draft sidecar (block_count 41, nextn 1) matches the trunk.
  const auto draft = Parse(valid, false);
  const auto trunk = Parse(valid);
  assert(draft && trunk && draft->MtpMatches(*trunk));
}

}  // namespace

int main() {
  CheckSidecarCompatibility();
  CheckValidMetadata();
  CheckMalformedMetadata();
  std::cout << "Qwen3.6-35B-A3B metadata checks passed.\n";
}