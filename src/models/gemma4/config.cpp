#include "src/models/gemma4/config.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>
#include <variant>

namespace gufo::models::gemma4 {
namespace {

constexpr std::string_view kTrunkArchitecture = "gemma4";
constexpr std::string_view kDraftArchitecture = "gemma4-assistant";

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  std::uint32_t U32(std::string_view key, bool required = true,
                    std::uint32_t fallback = 0) {
    const auto value = gguf.GetMetadataUint32(key);
    if (value.has_value()) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF integer " + std::string(key));
    }
    return fallback;
  }

  float F32(std::string_view key, bool required = true, float fallback = 0.0F) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (value.has_value() && std::isfinite(*value)) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF float " + std::string(key));
    }
    return fallback;
  }

  /// Integer or boolean arrays; GGUF writers pick the narrowest element type
  /// and the reader normalizes booleans to 0/1 integers.
  std::vector<std::uint64_t> U64Array(std::string_view key,
                                      bool required = true) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      if (required)
        Fail("missing GGUF array " + std::string(key));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF array value " + std::string(key));
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF array must contain integers: " + std::string(key));
    }
    return out;
  }
};

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  const auto architecture = gguf.GetMetadataString("general.architecture");
  Config c;
  if (architecture == kTrunkArchitecture) {
    c.is_draft = false;
  } else if (architecture == kDraftArchitecture) {
    c.is_draft = true;
  } else {
    r.Fail("GGUF architecture is neither gemma4 nor gemma4-assistant");
    return std::nullopt;
  }
  const std::string prefix = std::string(*architecture) + ".";
  auto key = [&](std::string_view suffix) {
    return prefix + std::string(suffix);
  };

  c.num_layers = r.U32(key("block_count"));
  c.hidden_size = r.U32(key("embedding_length"));
  // The artifact records no vocab key; the vocabulary is the tokenizer token
  // list, exactly as llama.cpp derives it.
  {
    const auto* meta = gguf.FindMetadata("tokenizer.ggml.tokens");
    const auto* tokens =
        meta == nullptr
            ? nullptr
            : std::get_if<std::vector<std::string_view>>(&meta->value);
    if (tokens == nullptr || tokens->empty() ||
        tokens->size() > std::numeric_limits<std::uint32_t>::max()) {
      r.Fail("tokenizer.ggml.tokens must supply the vocabulary size");
    } else {
      c.vocab_size = static_cast<std::uint32_t>(tokens->size());
    }
  }
  c.context_length = r.U32(key("context_length"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), true, 1e-6F);
  c.logit_softcap = r.F32(key("final_logit_softcapping"), false, 0.0F);

  c.num_heads = r.U32(key("attention.head_count"));
  for (const auto value : r.U64Array(key("attention.head_count_kv"))) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
      r.Fail("attention.head_count_kv entry exceeds uint32");
      break;
    }
    c.num_kv_heads.push_back(static_cast<std::uint32_t>(value));
  }
  {
    const auto pattern = r.U64Array(key("attention.sliding_window_pattern"));
    for (const auto value : pattern) {
      if (value > 1) {
        r.Fail("attention.sliding_window_pattern must contain booleans");
        break;
      }
      c.is_swa.push_back(value != 0);
    }
  }

  c.head_dim_full = r.U32(key("attention.key_length"));
  if (r.U32(key("attention.value_length"), false, c.head_dim_full) !=
      c.head_dim_full)
    r.Fail("attention value length must match the key length");
  c.head_dim_swa = r.U32(key("attention.key_length_swa"));
  if (r.U32(key("attention.value_length_swa"), false, c.head_dim_swa) !=
      c.head_dim_swa)
    r.Fail(
        "sliding-window value length must match the sliding-window key length");
  c.rope_dim_full = r.U32(key("rope.dimension_count"));
  c.rope_dim_swa = r.U32(key("rope.dimension_count_swa"));
  c.rope_theta = r.F32(key("rope.freq_base"), true, 1000000.0F);
  c.rope_theta_swa = r.F32(key("rope.freq_base_swa"), true, 10000.0F);
  c.sliding_window = r.U32(key("attention.sliding_window"));
  c.shared_kv_layers = r.U32(key("attention.shared_kv_layers"), false, 0);

  c.ffn_length = r.U32(key("feed_forward_length"));
  c.num_experts = r.U32(key("expert_count"), !c.is_draft, 0);
  c.num_experts_used = r.U32(key("expert_used_count"), !c.is_draft, 0);
  c.expert_ff = r.U32(key("expert_feed_forward_length"), !c.is_draft, 0);
  if (c.is_draft) {
    c.nextn_predict_layers = r.U32(key("nextn_predict_layers"));
    c.hidden_size_out = r.U32(key("embedding_length_out"));
  }
  if (r.U32(key("embedding_length_per_layer_input"), false, 0) != 0)
    r.Fail("per-layer input embeddings are not supported");

  if (!r.ok) {
    return std::nullopt;
  }

  // Structural limits of this runtime, not of the format.
  if (c.num_layers == 0 || c.vocab_size == 0 || c.context_length == 0 ||
      c.context_length > INT32_MAX || !std::isfinite(c.rms_eps) ||
      c.rms_eps <= 0 || !std::isfinite(c.rope_theta) || c.rope_theta <= 0 ||
      !std::isfinite(c.rope_theta_swa) || c.rope_theta_swa <= 0 ||
      c.hidden_size == 0 || c.num_heads == 0 || c.sliding_window == 0 ||
      c.shared_kv_layers > c.num_layers || c.ffn_length == 0 ||
      c.num_kv_heads.size() != c.num_layers ||
      c.is_swa.size() != c.num_layers) {
    r.Fail("gemma4 metadata describes an unsupported shape");
    return std::nullopt;
  }
  for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
    if (c.num_kv_heads[layer] == 0 ||
        c.num_heads % c.num_kv_heads[layer] != 0) {
      r.Fail("gemma4 metadata describes an unsupported shape");
      return std::nullopt;
    }
  }
  if (!c.is_draft && (c.num_experts == 0 || c.num_experts_used == 0 ||
                      c.num_experts_used > c.num_experts || c.expert_ff == 0 ||
                      c.logit_softcap <= 0)) {
    r.Fail("gemma4 trunk metadata is incomplete");
    return std::nullopt;
  }

  // These dimensions select the specialized attention, MoE and projection
  // kernels. Reject other profiles before allocating or launching;
  // satisfying divisibility alone does not make a shape executable.
  if (!c.is_draft) {
    if (c.num_layers != 30 || c.hidden_size != 2816 || c.num_heads != 16 ||
        c.head_dim_full != 512 || c.head_dim_swa != 256 ||
        c.rope_dim_full != 512 || c.rope_dim_swa != 256 ||
        c.rope_theta != 1000000.0F || c.rope_theta_swa != 10000.0F ||
        c.sliding_window != 1024 || c.shared_kv_layers != 0 ||
        c.ffn_length != 2112 || c.num_experts != 128 ||
        c.num_experts_used != 8 || c.expert_ff != 704 ||
        c.logit_softcap != 30.0F) {
      r.Fail("unsupported Gemma-4-26B-A4B kernel geometry");
      return std::nullopt;
    }
    for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
      const bool swa = (layer % 6) != 5;
      const std::uint32_t kv = swa ? 8 : 2;
      if (c.is_swa[layer] != swa || c.num_kv_heads[layer] != kv) {
        r.Fail("unsupported Gemma-4-26B-A4B kernel geometry");
        return std::nullopt;
      }
    }
  } else {
    if (c.num_layers != 4 || c.hidden_size != 1024 ||
        c.hidden_size_out != 2816 || c.num_heads != 16 ||
        c.head_dim_full != 512 || c.head_dim_swa != 256 ||
        c.rope_dim_full != 512 || c.rope_dim_swa != 256 ||
        c.rope_theta != 1000000.0F || c.rope_theta_swa != 10000.0F ||
        c.sliding_window != 1024 || c.shared_kv_layers != 4 ||
        c.nextn_predict_layers != 4 || c.ffn_length != 8192 ||
        c.num_experts != 0 || c.logit_softcap != 0.0F) {
      r.Fail("unsupported Gemma-4-26B-A4B draft kernel geometry");
      return std::nullopt;
    }
    constexpr bool kDraftPattern[4] = {true, true, true, false};
    constexpr std::uint32_t kDraftKv[4] = {8, 8, 8, 2};
    for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
      if (c.is_swa[layer] != kDraftPattern[layer] ||
          c.num_kv_heads[layer] != kDraftKv[layer]) {
        r.Fail("unsupported Gemma-4-26B-A4B draft kernel geometry");
        return std::nullopt;
      }
    }
  }
  return c;
}

}  // namespace gufo::models::gemma4