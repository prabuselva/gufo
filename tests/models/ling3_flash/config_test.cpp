#include "src/models/ling3_flash/config.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace l3 = gufo::models::ling3_flash;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// MLA blocks sit at (il + 1) % stride == 0 plus the trailing MTP block.
std::vector<std::uint64_t> HeadCountKv(std::uint32_t num_layers_all,
                                       std::uint32_t num_layers,
                                       std::uint32_t stride = 6) {
  std::vector<std::uint64_t> kv(num_layers_all, 0);
  for (std::uint32_t il = 0; il < num_layers_all; ++il) {
    if ((il + 1) % stride == 0 || il >= num_layers)
      kv[il] = 1;
  }
  return kv;
}

// Builds a trunk Config directly (no GGUF) to exercise MTP compatibility.
l3::Config TrunkConfig() {
  l3::Config trunk;
  trunk.num_layers = 42;
  trunk.num_layers_all = 43;
  trunk.nextn_layers = 1;
  trunk.hidden_size = 2560;
  trunk.vocab_size = 157184;
  trunk.context_length = 262144;
  trunk.rms_eps = 1e-6F;
  trunk.num_heads = 32;
  trunk.mla_head_dim = 192;
  trunk.mla_value_dim = 128;
  trunk.kv_lora_rank = 512;
  trunk.rotary_dim = 64;
  trunk.rope_theta = 6e6F;
  for (const auto v : HeadCountKv(43, 42)) {
    trunk.head_count_kv.push_back(static_cast<std::uint32_t>(v));
  }
  trunk.num_experts = 512;
  trunk.num_experts_used = 8;
  trunk.expert_groups = 8;
  trunk.expert_groups_used = 4;
  trunk.expert_ff = trunk.shared_expert_ff = 768;
  return trunk;
}

void CheckSidecarCompatibility() {
  const auto trunk = TrunkConfig();
  auto sidecar = trunk;
  Require(sidecar.MtpMatches(trunk), "matching MTP constants rejected");
  for (auto member :
       {&l3::Config::context_length, &l3::Config::rotary_dim,
        &l3::Config::num_experts_used, &l3::Config::shared_expert_ff,
        &l3::Config::expert_groups_used, &l3::Config::kv_lora_rank,
        &l3::Config::mla_value_dim}) {
    auto wrong = sidecar;
    ++(wrong.*member);
    Require(!wrong.MtpMatches(trunk),
            "dimension-compatible MTP with different semantics accepted");
  }
  for (auto member : {&l3::Config::rms_eps, &l3::Config::rope_theta}) {
    auto wrong = sidecar;
    wrong.*member *= 2.0F;
    Require(!wrong.MtpMatches(trunk),
            "MTP numerical constant mismatch accepted");
  }
  auto no_nextn = trunk;
  no_nextn.nextn_layers = 0;
  Require(!no_nextn.MtpMatches(trunk), "trunk accepted as its own MTP draft");
}

using Value = std::variant<std::uint32_t, float, bool, std::string,
                           std::vector<std::uint64_t>, std::vector<double>,
                           std::vector<std::string>>;
using Metadata = std::map<std::string, Value>;

// Minimal metadata-only GGUF: no model or GPU is needed to test the loader.
std::optional<l3::Config> Parse(const Metadata& fields, bool trunk = true) {
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
            pod(static_cast<std::uint8_t>(item));
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t{8});
            text(item);
          } else if constexpr (std::is_same_v<T, std::vector<std::uint64_t>>) {
            pod(std::uint32_t{9});
            pod(std::uint32_t{10});
            pod(std::uint64_t{item.size()});
            for (const auto entry : item)
              pod(entry);
          } else if constexpr (std::is_same_v<T, std::vector<double>>) {
            pod(std::uint32_t{9});
            pod(std::uint32_t{6});
            pod(std::uint64_t{item.size()});
            for (const auto entry : item)
              pod(static_cast<float>(entry));
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
  const auto result = l3::Config::FromGguf(*reader, trunk, &error);
  assert(result.has_value() == error.empty());
  return result;
}

// SwiGLU clamp arrays of the shipped artifact: routed experts clamp from
// layer 35, the shared expert from layer 34, and the MTP block never clamps.
std::vector<double> ClampArray(
    std::initializer_list<std::tuple<int, int, float>> ranges,
    std::size_t size) {
  std::vector<double> values(size, 0.0);
  for (const auto& [start, end, limit] : ranges)
    for (int il = start; il < end; ++il)
      values[static_cast<std::size_t>(il)] = limit;
  return values;
}

Metadata ValidMetadata() {
  Metadata fields{
      {"general.architecture", std::string{"bailingmoe3"}},
      {"tokenizer.ggml.tokens", std::vector<std::string>{"a", "b", "c"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 43},
           {"nextn_predict_layers", 1},
           {"embedding_length", 2560},
           {"vocab_size", 157184},
           {"context_length", 262144},
           {"attention.head_count", 32},
           {"attention.key_length_mla", 192},
           {"attention.value_length_mla", 128},
           {"attention.kv_lora_rank", 512},
           {"rope.dimension_count", 64},
           {"ssm.conv_kernel", 4},
           {"kda.head_dim", 128},
           {"expert_count", 512},
           {"expert_used_count", 8},
           {"expert_group_count", 8},
           {"expert_group_used_count", 4},
           {"expert_feed_forward_length", 768},
           {"expert_shared_feed_forward_length", 768},
           {"expert_gating_func", 2},
           {"leading_dense_block_count", 2},
           {"feed_forward_length", 6144}}) {
    fields[std::string{"bailingmoe3."} + key] = value;
  }
  fields["bailingmoe3.attention.layer_norm_rms_epsilon"] = 1e-6F;
  fields["bailingmoe3.rope.freq_base"] = 6e6F;
  fields["bailingmoe3.attention.head_count_kv"] = HeadCountKv(43, 42);
  fields["bailingmoe3.kda.safe_gate"] = true;
  fields["bailingmoe3.kda.gate_lower_bound"] = -5.0F;
  fields["bailingmoe3.expert_weights_scale"] = 2.5F;
  fields["bailingmoe3.expert_weights_norm"] = true;
  fields["bailingmoe3.swiglu_clamp_exp"] = ClampArray({{35, 42, 4.0F}}, 43);
  fields["bailingmoe3.swiglu_clamp_shexp"] =
      ClampArray({{34, 40, 5.0F}, {40, 42, 7.0F}}, 43);
  return fields;
}

// Tiny artifact metadata: 24 blocks, no MTP key, MLA every fourth block, one
// leading dense FFN block and no SwiGLU clamp arrays.
Metadata TinyMetadata() {
  Metadata fields{
      {"general.architecture", std::string{"bailingmoe3"}},
      {"tokenizer.ggml.tokens", std::vector<std::string>{"a", "b", "c"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 24},
           {"embedding_length", 1536},
           {"vocab_size", 157184},
           {"context_length", 131072},
           {"attention.head_count", 16},
           {"attention.key_length_mla", 192},
           {"attention.value_length_mla", 128},
           {"attention.kv_lora_rank", 512},
           {"attention.q_lora_rank", 256},
           {"rope.dimension_count", 64},
           {"ssm.conv_kernel", 4},
           {"kda.head_dim", 128},
           {"expert_count", 128},
           {"expert_used_count", 8},
           {"expert_group_count", 8},
           {"expert_group_used_count", 4},
           {"expert_feed_forward_length", 512},
           {"expert_shared_feed_forward_length", 512},
           {"expert_gating_func", 2},
           {"leading_dense_block_count", 1},
           {"feed_forward_length", 4608}}) {
    fields[std::string{"bailingmoe3."} + key] = value;
  }
  fields["bailingmoe3.attention.layer_norm_rms_epsilon"] = 1e-6F;
  fields["bailingmoe3.rope.freq_base"] = 6e6F;
  fields["bailingmoe3.attention.head_count_kv"] = HeadCountKv(24, 24, 4);
  fields["bailingmoe3.kda.safe_gate"] = true;
  fields["bailingmoe3.kda.gate_lower_bound"] = -5.0F;
  fields["bailingmoe3.expert_weights_scale"] = 2.5F;
  fields["bailingmoe3.expert_weights_norm"] = true;
  return fields;
}

void CheckValidMetadata() {
  const auto config = Parse(ValidMetadata());
  Require(config.has_value(), "valid metadata rejected");
  Require(config->profile_name == "Ling 3.0 Flash", "wrong geometry profile");
  Require(config->mla_q_lora == 0 && !config->mla_q_split,
          "Flash must use the fused query projection");
  Require(config->num_layers == 42 && config->num_layers_all == 43 &&
              config->nextn_layers == 1,
          "trunk/MTP layer split incorrect");
  Require(config->vocab_size == 157184 && config->hidden_size == 2560 &&
              config->context_length == 262144,
          "core dimensions incorrect");
  Require(config->SsmInnerSize() == 4096, "KDA q/k/v channel count wrong");
  Require(config->MlaQDim() == 6144 && config->MlaNopeDim() == 128 &&
              config->MlaKeyDim() == 576 && config->MlaVDim() == 4096,
          "absorbed-MLA dimensions wrong");
  // MLA sits on every sixth layer (indices 5, 11, ...) and on the MTP block.
  Require(
      config->IsMlaLayer(5) && config->IsMlaLayer(41) && config->IsMlaLayer(42),
      "MLA layer misclassified");
  Require(!config->IsMlaLayer(0) && !config->IsMlaLayer(4),
          "KDA layer misclassified");
  Require(config->IsDenseFfnLayer(0) && config->IsDenseFfnLayer(1) &&
              !config->IsDenseFfnLayer(2),
          "dense FFN prefix misclassified");
  Require(config->kda_gate_lower_bound == -5.0F &&
              config->expert_weights_scale == 2.5F &&
              config->expert_weights_norm && config->kda_safe_gate,
          "routing/gate constants incorrect");
  Require(config->swiglu_clamp_exp.size() == 43 &&
              config->swiglu_clamp_exp[34] == 0.0F &&
              config->swiglu_clamp_exp[35] == 4.0F &&
              config->swiglu_clamp_exp[41] == 4.0F &&
              config->swiglu_clamp_exp[42] == 0.0F,
          "routed-expert clamps incorrect");
  Require(config->swiglu_clamp_shexp[33] == 0.0F &&
              config->swiglu_clamp_shexp[34] == 5.0F &&
              config->swiglu_clamp_shexp[39] == 5.0F &&
              config->swiglu_clamp_shexp[40] == 7.0F,
          "shared-expert clamps incorrect");
}

void CheckTinyMetadata() {
  const auto config = Parse(TinyMetadata());
  Require(config.has_value(), "valid Tiny metadata rejected");
  Require(config->profile_name == "Ling 3.0 Tiny", "wrong geometry profile");
  Require(config->mla_q_lora == 256 && config->mla_q_split,
          "Tiny must use the low-rank query chain");
  Require(config->num_layers == 24 && config->num_layers_all == 24 &&
              config->nextn_layers == 0,
          "Tiny has no MTP block");
  Require(config->hidden_size == 1536 && config->vocab_size == 157184 &&
              config->context_length == 131072,
          "Tiny core dimensions incorrect");
  Require(config->SsmInnerSize() == 2048, "Tiny KDA channel count wrong");
  Require(config->MlaQDim() == 3072 && config->MlaNopeDim() == 128 &&
              config->MlaKeyDim() == 576 && config->MlaVDim() == 2048,
          "Tiny absorbed-MLA dimensions wrong");
  // MLA sits every fourth block (indices 3, 7, ..., 23); the rest are KDA.
  Require(
      config->IsMlaLayer(3) && config->IsMlaLayer(11) && config->IsMlaLayer(23),
      "Tiny MLA layer misclassified");
  Require(!config->IsMlaLayer(0) && !config->IsMlaLayer(2) &&
              !config->IsMlaLayer(22),
          "Tiny KDA layer misclassified");
  Require(config->IsDenseFfnLayer(0) && !config->IsDenseFfnLayer(1),
          "Tiny dense FFN prefix misclassified");
  Require(
      config->swiglu_clamp_exp.empty() && config->swiglu_clamp_shexp.empty(),
      "Tiny must not carry clamp arrays");

  // Mixing one Flash dimension into Tiny (or vice versa) is rejected.
  for (const auto& [key, value] :
       {std::pair{"expert_count", 512U}, std::pair{"embedding_length", 2560U},
        std::pair{"leading_dense_block_count", 2U},
        std::pair{"feed_forward_length", 6144U}}) {
    auto wrong = TinyMetadata();
    wrong[std::string{"bailingmoe3."} + key] = value;
    Require(!Parse(wrong), "Flash/Tiny geometry mix accepted");
  }
  auto with_mtp = TinyMetadata();
  with_mtp["bailingmoe3.nextn_predict_layers"] = std::uint32_t{1};
  Require(!Parse(with_mtp), "Tiny + MTP is not a measured profile");

  // Tiny MLA placement is part of its pattern.
  auto misplaced = HeadCountKv(24, 24, 4);
  misplaced[4] = 1;
  auto wrong_placement = TinyMetadata();
  wrong_placement["bailingmoe3.attention.head_count_kv"] = misplaced;
  Require(!Parse(wrong_placement), "Tiny MLA misplacement accepted");
}

void CheckMalformedMetadata() {
  const auto valid = ValidMetadata();
  assert(Parse(valid));

  // Wrong architecture is rejected outright.
  auto arch = valid;
  arch["general.architecture"] = std::string{"qwen35moe"};
  assert(!Parse(arch));

  // The MTP block count must stay below the total.
  auto nextn = valid;
  nextn["bailingmoe3.nextn_predict_layers"] = std::uint32_t{43};
  assert(!Parse(nextn));

  // Any deviation from the validated kernel geometry is rejected.
  for (const auto& [key, value] :
       {std::pair{"block_count", 42U},
        std::pair{"embedding_length", 2048U},
        std::pair{"vocab_size", 151936U},
        std::pair{"attention.head_count", 16U},
        std::pair{"attention.key_length_mla", 128U},
        std::pair{"attention.value_length_mla", 64U},
        std::pair{"attention.kv_lora_rank", 256U},
        std::pair{"attention.q_lora_rank", 128U},
        std::pair{"attention.q_lora_rank", 256U},
        std::pair{"rope.dimension_count", 128U},
        std::pair{"kda.head_dim", 64U},
        std::pair{"ssm.conv_kernel", 3U},
        std::pair{"expert_count", 256U},
        std::pair{"expert_used_count", 10U},
        std::pair{"expert_group_count", 4U},
        std::pair{"expert_group_used_count", 2U},
        std::pair{"expert_feed_forward_length", 512U},
        std::pair{"expert_gating_func", 0U},
        std::pair{"leading_dense_block_count", 0U},
        std::pair{"feed_forward_length", 5120U}}) {
    auto wrong = valid;
    wrong[std::string{"bailingmoe3."} + key] = value;
    assert(!Parse(wrong));
  }
  // The low-rank query rank is metadata only for the split (Tiny) profile;
  // the fused Flash artifact above must parse without it, and Tiny must not
  // parse when it goes missing.
  auto tiny_no_rank = TinyMetadata();
  tiny_no_rank.erase("bailingmoe3.attention.q_lora_rank");
  assert(!Parse(tiny_no_rank));

  // Gate constants must keep the decay bounded away from zero.
  for (const float limit : {0.0F, 1.0F}) {
    auto wrong = valid;
    wrong["bailingmoe3.kda.gate_lower_bound"] = limit;
    assert(!Parse(wrong));
  }

  // head_count_kv must cover every block with sane per-layer head counts.
  auto short_kv = valid;
  short_kv["bailingmoe3.attention.head_count_kv"] = HeadCountKv(42, 42);
  assert(!Parse(short_kv));
  auto huge_kv = HeadCountKv(43, 42);
  huge_kv[7] = 33;
  auto oversized = valid;
  oversized["bailingmoe3.attention.head_count_kv"] = huge_kv;
  assert(!Parse(oversized));

  // MLA placement is part of the supported pattern.
  auto misplaced = HeadCountKv(43, 42);
  misplaced[0] = 1;
  auto wrong_placement = valid;
  wrong_placement["bailingmoe3.attention.head_count_kv"] = misplaced;
  assert(!Parse(wrong_placement));

  // Clamp arrays must cover every block.
  auto short_clamp = valid;
  short_clamp["bailingmoe3.swiglu_clamp_exp"] =
      ClampArray({{35, 42, 4.0F}}, 42);
  assert(!Parse(short_clamp));

  // The draft sidecar parses with the same metadata and matches the trunk.
  const auto draft = Parse(valid, false);
  const auto trunk = Parse(valid);
  assert(draft && trunk && draft->MtpMatches(*trunk));
}

}  // namespace

int main() {
  CheckSidecarCompatibility();
  CheckValidMetadata();
  CheckTinyMetadata();
  CheckMalformedMetadata();
  std::cout << "Ling 3.0 metadata checks passed.\n";
}