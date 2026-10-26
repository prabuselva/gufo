#include "src/models/flint/config.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <variant>

namespace gufo::models::flint {
namespace {

constexpr std::string_view kArchitecture = "flint";

// The single artifact geometry whose kernels are built and measured end to
// end. Meeting the structural checks below is not enough to run; the shape
// must match this measured profile.
struct GeometryProfile {
  std::string_view name;
  std::uint32_t num_layers;
  std::uint32_t hidden_size;
  std::uint32_t vocab_size;
  std::uint32_t num_heads;
  std::uint32_t num_kv_heads;
  std::uint32_t head_dim;
  float rope_theta;
  std::uint32_t sliding_window;
  std::uint32_t leading_dense_blocks;
  std::uint32_t ffn_length;
  std::uint32_t num_experts;
  std::uint32_t num_experts_used;
  std::uint32_t expert_ff;
  std::uint32_t shared_experts;
  std::uint32_t shared_expert_ff;
};

constexpr GeometryProfile kProfiles[] = {
    {
        .name = "Flint Preview 40B-A2B",
        .num_layers = 49,
        .hidden_size = 2048,
        .vocab_size = 150144,
        .num_heads = 32,
        .num_kv_heads = 4,
        .head_dim = 128,
        .rope_theta = 10000.0F,
        .sliding_window = 513,
        .leading_dense_blocks = 2,
        .ffn_length = 6144,
        .num_experts = 256,
        .num_experts_used = 6,
        .expert_ff = 512,
        .shared_experts = 2,
        .shared_expert_ff = 1024,
    },
};

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

  float F32(std::string_view key, float fallback, bool required = true) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value || !std::isfinite(*value)) {
      if (required || gguf.FindMetadata(key) != nullptr) {
        Fail("missing or invalid GGUF float " + std::string(key));
      }
      return fallback;
    }
    return *value;
  }

  bool Boolean(std::string_view key, bool required, bool fallback) {
    const auto value = gguf.GetMetadataBool(key);
    if (value.has_value()) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF bool " + std::string(key));
    }
    return fallback;
  }

  std::vector<std::uint64_t> U64Array(std::string_view key,
                                      bool required = true) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      if (required) {
        Fail("missing GGUF array " + std::string(key));
      }
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (const auto v : *s) {
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

/// Copies a boolean array (stored as widened integers) covering every layer.
bool CopyLayerFlags(std::string_view key, std::uint32_t num_layers,
                    std::vector<bool>* dst, Reader* r) {
  const auto src = r->U64Array(key);
  if (!r->ok) {
    return false;
  }
  if (src.size() != num_layers) {
    r->Fail(std::string(key) + " must cover every block");
    return false;
  }
  for (const auto v : src) {
    if (v > 1) {
      r->Fail(std::string(key) + " must contain booleans");
      return false;
    }
    dst->push_back(v != 0);
  }
  return true;
}

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not flint");
    return std::nullopt;
  }

  Config c;
  const std::string_view p = "flint.";
  auto key = [&](std::string_view suffix) {
    return std::string(p) + std::string(suffix);
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
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), 1e-6F);

  c.num_heads = r.U32(key("attention.head_count"));
  c.num_kv_heads = r.U32(key("attention.head_count_kv"));
  c.head_dim = r.U32(key("attention.key_length"));
  if (r.U32(key("attention.value_length"), false, c.head_dim) != c.head_dim) {
    r.Fail("attention value length must match the key length");
  }
  c.rope_theta = r.F32(key("rope.freq_base"), 10000.0F);
  const bool copy_rope = CopyLayerFlags(key("attention.rope_pattern"),
                                        c.num_layers, &c.rope_layer, &r);
  const bool copy_swa = CopyLayerFlags(key("attention.sliding_window_pattern"),
                                       c.num_layers, &c.swa_layer, &r);
  c.sliding_window = r.U32(key("attention.sliding_window"));
  if (!copy_rope || !copy_swa) {
    return std::nullopt;
  }

  c.leading_dense_blocks = r.U32(key("leading_dense_block_count"));
  c.ffn_length = r.U32(key("feed_forward_length"));

  c.num_experts = r.U32(key("expert_count"));
  c.num_experts_used = r.U32(key("expert_used_count"));
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.shared_experts = r.U32(key("expert_shared_count"), false, 0);
  c.shared_expert_ff =
      r.U32(key("expert_shared_feed_forward_length"), false, 0);
  c.expert_gating_func = r.U32(key("expert_gating_func"));
  c.expert_weights_scale = r.F32(key("expert_weights_scale"), 1.0F, false);
  c.expert_weights_norm = r.Boolean(key("expert_weights_norm"), false, false);
  if (!r.ok) {
    return std::nullopt;
  }

  // Structural limits of this runtime, not of the format.
  if (c.num_layers == 0 || c.context_length == 0 || c.vocab_size == 0 ||
      c.num_layers > 512 || c.sliding_window == 0 ||
      !std::isfinite(c.rms_eps) || c.rms_eps <= 0 ||
      !std::isfinite(c.rope_theta) || c.rope_theta <= 0 || c.hidden_size == 0 ||
      c.num_heads == 0 || c.num_kv_heads == 0 ||
      c.num_heads % c.num_kv_heads != 0 || c.head_dim == 0 ||
      c.leading_dense_blocks >= c.num_layers || c.ffn_length == 0 ||
      c.num_experts == 0 || c.num_experts_used == 0 ||
      c.num_experts_used > c.num_experts || c.expert_ff == 0 ||
      c.shared_experts == 0 || c.shared_expert_ff == 0 ||
      c.expert_gating_func != 2 || !std::isfinite(c.expert_weights_scale)) {
    r.Fail("flint metadata describes an unsupported shape");
    return std::nullopt;
  }

  // These dimensions select the specialized attention, MoE and projection
  // kernels. Reject other artifacts before allocating or launching;
  // satisfying divisibility alone does not make a shape executable.
  const GeometryProfile* profile = nullptr;
  for (const GeometryProfile& g : kProfiles) {
    if (c.num_layers == g.num_layers && c.hidden_size == g.hidden_size &&
        c.vocab_size == g.vocab_size && c.num_heads == g.num_heads &&
        c.num_kv_heads == g.num_kv_heads && c.head_dim == g.head_dim &&
        c.rope_theta == g.rope_theta && c.sliding_window == g.sliding_window &&
        c.leading_dense_blocks == g.leading_dense_blocks &&
        c.ffn_length == g.ffn_length && c.num_experts == g.num_experts &&
        c.num_experts_used == g.num_experts_used &&
        c.expert_ff == g.expert_ff && c.shared_experts == g.shared_experts &&
        c.shared_expert_ff == g.shared_expert_ff) {
      profile = &g;
      break;
    }
  }
  if (profile == nullptr) {
    r.Fail("unsupported flint kernel geometry");
    return std::nullopt;
  }
  c.profile_name = profile->name;
  return c;
}

}  // namespace gufo::models::flint
