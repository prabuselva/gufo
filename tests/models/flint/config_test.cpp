#include "src/models/flint/config.hpp"

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

namespace fl = gufo::models::flint;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

using Value = std::variant<std::uint32_t, float, bool, std::string,
                           std::vector<std::uint64_t>, std::vector<std::string>,
                           std::vector<bool>>;
using Metadata = std::map<std::string, Value>;

// Minimal metadata-only GGUF: no model or GPU is needed to test the loader.
std::optional<fl::Config> Parse(const Metadata& fields) {
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
          } else if constexpr (std::is_same_v<T, bool>) {
            pod(std::uint32_t{7});
            pod(static_cast<std::uint8_t>(item ? 1U : 0U));
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
  const auto result = fl::Config::FromGguf(*reader, &error);
  assert(result.has_value() == error.empty());
  return result;
}

// The measured artifact pattern: NoPE and global at layers 0, 1 and every
// layer == 1 (mod 4) from 5 through 45, plus 48; windowed rotary elsewhere.
std::vector<bool> MeasuredPattern() {
  std::vector<bool> pattern;
  for (std::uint32_t layer = 0; layer < 49; ++layer) {
    const bool global_nope = layer == 0 || layer == 1 || layer == 48 ||
                             (layer >= 5 && layer % 4 == 1);
    pattern.push_back(!global_nope);
  }
  return pattern;
}

Metadata ValidMetadata() {
  Metadata fields{
      {"general.architecture", std::string{"flint"}},
      {"tokenizer.ggml.tokens", std::vector<std::string>(150144, "a")}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 49},
           {"embedding_length", 2048},
           {"context_length", 262144},
           {"attention.head_count", 32},
           {"attention.head_count_kv", 4},
           {"attention.key_length", 128},
           {"attention.value_length", 128},
           {"attention.sliding_window", 513},
           {"leading_dense_block_count", 2},
           {"feed_forward_length", 6144},
           {"expert_count", 256},
           {"expert_used_count", 6},
           {"expert_feed_forward_length", 512},
           {"expert_shared_count", 2},
           {"expert_shared_feed_forward_length", 1024},
           {"expert_gating_func", 2}}) {
    fields[std::string{"flint."} + key] = value;
  }
  fields["flint.attention.layer_norm_rms_epsilon"] = 1e-6F;
  fields["flint.rope.freq_base"] = 10000.0F;
  fields["flint.expert_weights_scale"] = 1.0F;
  fields["flint.expert_weights_norm"] = true;
  const auto pattern = MeasuredPattern();
  fields["flint.attention.rope_pattern"] = pattern;
  fields["flint.attention.sliding_window_pattern"] = pattern;
  return fields;
}

void CheckValid() {
  const auto config = Parse(ValidMetadata());
  Require(config.has_value(), "valid flint metadata must parse");
  Require(config->profile_name == std::string_view{"Flint Preview 40B-A2B"},
          "measured artifact must match its kernel geometry profile");
  Require(config->num_layers == 49 && config->hidden_size == 2048,
          "trunk dimensions");
  Require(config->vocab_size == 150144 && config->context_length == 262144,
          "vocab comes from the tokenizer token list");
  Require(config->num_heads == 32 && config->num_kv_heads == 4 &&
              config->head_dim == 128,
          "grouped-query attention dimensions");
  Require(config->AttentionQDim() == 4096 && config->AttentionKvDim() == 512,
          "q expands past the hidden size, kv is narrower");
  Require(config->IsMoeLayer(2) && config->IsMoeLayer(48) &&
              !config->IsMoeLayer(0) && !config->IsMoeLayer(1),
          "2 leading dense layers, the rest MoE");
  const auto pattern = MeasuredPattern();
  for (std::uint32_t layer = 0; layer < 49; ++layer) {
    Require(config->HasRope(layer) == pattern[layer],
            "rope_pattern must reach the config layer by layer");
    Require(config->IsWindowedLayer(layer) == pattern[layer],
            "sliding_window_pattern must reach the config layer by layer");
  }
  Require(!config->IsWindowedLayer(0) && !config->IsWindowedLayer(48),
          "first and last layers stay global/NoPE");
}

void CheckRejections() {
  {
    Metadata fields = ValidMetadata();
    fields["general.architecture"] = std::string{"llama"};
    Require(!Parse(fields).has_value(), "other architectures must be rejected");
  }
  {
    Metadata fields = ValidMetadata();
    fields.erase("flint.expert_gating_func");
    Require(!Parse(fields).has_value(), "gating function is required");
  }
  {
    Metadata fields = ValidMetadata();
    fields["flint.expert_gating_func"] = std::uint32_t{0};
    Require(!Parse(fields).has_value(), "only sigmoid gating is supported");
  }
  {
    Metadata fields = ValidMetadata();
    fields["flint.attention.rope_pattern"] = std::vector<bool>(48, true);
    Require(!Parse(fields).has_value(), "rope_pattern must cover every block");
  }
  {
    Metadata fields = ValidMetadata();
    std::vector<std::uint64_t> not_bool;
    for (std::uint32_t layer = 0; layer < 49; ++layer) {
      not_bool.push_back(2);
    }
    fields["flint.attention.sliding_window_pattern"] = not_bool;
    Require(!Parse(fields).has_value(),
            "sliding_window_pattern must contain booleans");
  }
  {
    Metadata fields = ValidMetadata();
    fields["flint.attention.head_count_kv"] = std::uint32_t{3};
    Require(!Parse(fields).has_value(),
            "kv heads must divide the query head count");
  }
  {
    Metadata fields = ValidMetadata();
    fields["flint.leading_dense_block_count"] = std::uint32_t{49};
    Require(!Parse(fields).has_value(), "at least one MoE layer must exist");
  }
  {
    Metadata fields = ValidMetadata();
    fields.erase("tokenizer.ggml.tokens");
    Require(!Parse(fields).has_value(),
            "the vocabulary comes from the tokenizer");
  }
  {
    Metadata fields = ValidMetadata();
    fields["flint.expert_used_count"] = std::uint32_t{7};
    Require(!Parse(fields).has_value(),
            "top-k outside the measured profile must be rejected");
  }
}

}  // namespace

int main() {
  try {
    CheckValid();
    CheckRejections();
  } catch (const std::exception& error) {
    std::cerr << "flint.config FAILED: " << error.what() << '\n';
    return 1;
  }
  std::cout << "flint.config ok\n";
  return 0;
}
