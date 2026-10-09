#include "src/models/ling3_flash/config.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

namespace gufo::models::ling3_flash {
namespace {

constexpr std::string_view kArchitecture = "bailingmoe3";

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

  /// Float arrays of any element width; GGUF writers pick the narrowest type.
  std::vector<float> F32Array(std::string_view key, bool required = true) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<float> out;
    if (meta == nullptr) {
      if (required) {
        Fail("missing GGUF array " + std::string(key));
      }
      return out;
    }
    if (const auto* d = std::get_if<std::vector<double>>(&meta->value)) {
      for (const double v : *d) {
        if (!std::isfinite(v)) {
          Fail("non-finite GGUF array value " + std::string(key));
          return {};
        }
        out.push_back(static_cast<float>(v));
      }
    } else {
      Fail("GGUF array must contain floats: " + std::string(key));
    }
    return out;
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

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       bool require_trunk,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not bailingmoe3");
    return std::nullopt;
  }

  Config c;
  const std::string_view p = "bailingmoe3.";
  auto key = [&](std::string_view suffix) {
    return std::string(p) + std::string(suffix);
  };

  c.num_layers_all = r.U32(key("block_count"));
  c.nextn_layers = r.U32(key("nextn_predict_layers"), false, 0);
  if (c.nextn_layers >= c.num_layers_all) {
    r.Fail("nextn_predict_layers must be below block_count");
    return std::nullopt;
  }
  c.num_layers = c.num_layers_all - c.nextn_layers;
  c.hidden_size = r.U32(key("embedding_length"));
  c.vocab_size = r.U32(key("vocab_size"));
  c.context_length = r.U32(key("context_length"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), 1e-6F);

  c.num_heads = r.U32(key("attention.head_count"));
  c.mla_head_dim = r.U32(key("attention.key_length_mla"));
  c.mla_value_dim = r.U32(key("attention.value_length_mla"));
  c.kv_lora_rank = r.U32(key("attention.kv_lora_rank"));
  c.rotary_dim = r.U32(key("rope.dimension_count"));
  c.rope_theta = r.F32(key("rope.freq_base"), 6000000.0F);
  {
    const auto kv = r.U64Array(key("attention.head_count_kv"));
    if (kv.size() != c.num_layers_all) {
      r.Fail("attention.head_count_kv must cover every block");
    } else {
      for (const auto v : kv) {
        if (v > c.num_heads) {
          r.Fail("attention.head_count_kv exceeds the head count");
          break;
        }
        c.head_count_kv.push_back(static_cast<std::uint32_t>(v));
      }
    }
  }

  c.ssm_conv_kernel = r.U32(key("ssm.conv_kernel"));
  c.ssm_head_dim = r.U32(key("kda.head_dim"));
  c.kda_safe_gate = r.Boolean(key("kda.safe_gate"), false, true);
  c.kda_gate_lower_bound = r.F32(key("kda.gate_lower_bound"), 0.0F);

  c.num_experts = r.U32(key("expert_count"));
  c.num_experts_used = r.U32(key("expert_used_count"));
  c.expert_groups = r.U32(key("expert_group_count"), false, 1);
  c.expert_groups_used =
      r.U32(key("expert_group_used_count"), false, c.expert_groups);
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.shared_expert_ff =
      r.U32(key("expert_shared_feed_forward_length"), false, c.expert_ff);
  c.expert_weights_scale = r.F32(key("expert_weights_scale"), 1.0F, false);
  c.expert_weights_norm = r.Boolean(key("expert_weights_norm"), false, false);
  c.expert_gating_func = r.U32(key("expert_gating_func"));

  c.leading_dense_blocks = r.U32(key("leading_dense_block_count"), false, 0);
  c.dense_ff = r.U32(key("feed_forward_length"));
  c.swiglu_clamp_exp = r.F32Array(key("swiglu_clamp_exp"), false);
  c.swiglu_clamp_shexp = r.F32Array(key("swiglu_clamp_shexp"), false);
  if (!c.swiglu_clamp_exp.empty() &&
      c.swiglu_clamp_exp.size() != c.num_layers_all) {
    r.Fail("swiglu_clamp_exp must cover every block");
  }
  if (!c.swiglu_clamp_shexp.empty() &&
      c.swiglu_clamp_shexp.size() != c.num_layers_all) {
    r.Fail("swiglu_clamp_shexp must cover every block");
  }

  if (!r.ok) {
    return std::nullopt;
  }

  // Structural limits of this runtime, not of the format.
  if (c.context_length == 0 || c.vocab_size == 0 || !std::isfinite(c.rms_eps) ||
      c.rms_eps <= 0 || !std::isfinite(c.rope_theta) || c.rope_theta <= 0 ||
      c.hidden_size == 0 || c.num_heads == 0 || c.mla_head_dim == 0 ||
      c.mla_value_dim == 0 || c.kv_lora_rank == 0 || c.rotary_dim == 0 ||
      c.rotary_dim % 2 != 0 || c.rotary_dim >= c.mla_head_dim ||
      c.ssm_conv_kernel == 0 || c.ssm_head_dim == 0 ||
      !std::isfinite(c.kda_gate_lower_bound) ||
      c.kda_gate_lower_bound >= 0.0F || c.num_experts == 0 ||
      c.num_experts_used == 0 || c.num_experts_used > c.num_experts ||
      c.expert_groups == 0 || c.num_experts % c.expert_groups != 0 ||
      c.expert_groups_used == 0 || c.expert_groups_used > c.expert_groups ||
      c.expert_ff == 0 || c.shared_expert_ff == 0 ||
      !std::isfinite(c.expert_weights_scale) || c.expert_gating_func != 2 ||
      c.dense_ff == 0 || c.leading_dense_blocks >= c.num_layers) {
    r.Fail("bailingmoe3 metadata describes an unsupported shape");
    return std::nullopt;
  }
  // These dimensions select the specialized KDA, absorbed-MLA and grouped
  // expert kernels. Reject other profiles before allocating or launching;
  // satisfying divisibility alone does not make a shape executable.
  if (c.num_layers != 42 || c.num_layers_all != 43 || c.nextn_layers != 1 ||
      c.hidden_size != 2560 || c.vocab_size != 157184 || c.num_heads != 32 ||
      c.mla_head_dim != 192 || c.mla_value_dim != 128 ||
      c.kv_lora_rank != 512 || c.rotary_dim != 64 || c.ssm_conv_kernel != 4 ||
      c.ssm_head_dim != 128 || c.num_experts != 512 ||
      c.num_experts_used != 8 || c.expert_groups != 8 ||
      c.expert_groups_used != 4 || c.expert_ff != 768 ||
      c.shared_expert_ff != 768 || c.leading_dense_blocks != 2 ||
      c.dense_ff != 6144) {
    r.Fail("unsupported Ling 3.0 Flash kernel geometry");
    return std::nullopt;
  }
  if (require_trunk) {
    // MLA layers sit at (layer + 1) % 6 == 0 in the trunk and on the MTP
    // block; every other block is KDA.
    for (std::uint32_t il = 0; il < c.num_layers_all; ++il) {
      const bool expected = ((il + 1) % 6 == 0) || il >= c.num_layers;
      if (c.IsMlaLayer(il) != expected) {
        r.Fail(
            "bailingmoe3 MLA layer placement differs from the supported "
            "pattern");
        return std::nullopt;
      }
    }
  }
  return c;
}

}  // namespace gufo::models::ling3_flash