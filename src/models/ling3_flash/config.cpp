#include "src/models/ling3_flash/config.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

namespace gufo::models::ling3_flash {
namespace {

constexpr std::string_view kArchitecture = "bailingmoe3";

// One artifact size whose specialized KDA, absorbed-MLA and grouped-expert
// kernels were built and measured end to end. Membership means the kernel
// tiles, cache strides and expert buckets fit the shape, so meeting the
// structural limits above is not enough to run.
struct GeometryProfile {
  std::string_view name;
  std::uint32_t num_layers;    // trunk layers, MTP block excluded
  std::uint32_t nextn_layers;  // self-speculative MTP blocks
  std::uint32_t hidden_size;
  std::uint32_t vocab_size;
  std::uint32_t num_heads;
  std::uint32_t mla_head_dim;
  std::uint32_t mla_value_dim;
  std::uint32_t kv_lora_rank;
  std::uint32_t rotary_dim;
  // q_lora_rank metadata value: absent (0) with the fused q export, 256 with
  // the split a/norm/b export.
  std::uint32_t mla_q_lora;
  bool mla_q_split;  // q exported as attn_q_a, attn_q_a_norm and attn_q_b
  std::uint32_t ssm_conv_kernel;
  std::uint32_t ssm_head_dim;
  std::uint32_t num_experts;
  std::uint32_t num_experts_used;
  std::uint32_t expert_groups;
  std::uint32_t expert_groups_used;
  std::uint32_t expert_ff;
  std::uint32_t shared_expert_ff;
  std::uint32_t leading_dense_blocks;
  std::uint32_t dense_ff;
  // MLA blocks sit at (layer + 1) % mla_stride == 0 in the trunk and on the
  // MTP block; every other block is KDA.
  std::uint32_t mla_stride;
};

constexpr GeometryProfile kProfiles[] = {
    {
        .name = "Ling 3.0 Flash",
        .num_layers = 42,
        .nextn_layers = 1,
        .hidden_size = 2560,
        .vocab_size = 157184,
        .num_heads = 32,
        .mla_head_dim = 192,
        .mla_value_dim = 128,
        .kv_lora_rank = 512,
        .rotary_dim = 64,
        .mla_q_lora = 0,
        .mla_q_split = false,
        .ssm_conv_kernel = 4,
        .ssm_head_dim = 128,
        .num_experts = 512,
        .num_experts_used = 8,
        .expert_groups = 8,
        .expert_groups_used = 4,
        .expert_ff = 768,
        .shared_expert_ff = 768,
        .leading_dense_blocks = 2,
        .dense_ff = 6144,
        .mla_stride = 6,
    },
    {
        .name = "Ling 3.0 Tiny",
        .num_layers = 24,
        .nextn_layers = 0,
        .hidden_size = 1536,
        .vocab_size = 157184,
        .num_heads = 16,
        .mla_head_dim = 192,
        .mla_value_dim = 128,
        .kv_lora_rank = 512,
        .rotary_dim = 64,
        .mla_q_lora = 256,
        .mla_q_split = true,
        .ssm_conv_kernel = 4,
        .ssm_head_dim = 128,
        .num_experts = 128,
        .num_experts_used = 8,
        .expert_groups = 8,
        .expert_groups_used = 4,
        .expert_ff = 512,
        .shared_expert_ff = 512,
        .leading_dense_blocks = 1,
        .dense_ff = 4608,
        .mla_stride = 4,
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
  c.mla_q_lora = r.U32(key("attention.q_lora_rank"), false, 0);
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
  // expert kernels. Reject artifacts outside the measured profiles before
  // allocating or launching; satisfying divisibility alone does not make a
  // shape executable.
  const GeometryProfile* profile = nullptr;
  for (const GeometryProfile& p : kProfiles) {
    if (c.num_layers == p.num_layers && c.nextn_layers == p.nextn_layers &&
        c.num_layers_all == p.num_layers + p.nextn_layers &&
        c.hidden_size == p.hidden_size && c.vocab_size == p.vocab_size &&
        c.num_heads == p.num_heads && c.mla_head_dim == p.mla_head_dim &&
        c.mla_value_dim == p.mla_value_dim &&
        c.kv_lora_rank == p.kv_lora_rank && c.rotary_dim == p.rotary_dim &&
        c.mla_q_lora == p.mla_q_lora &&
        c.ssm_conv_kernel == p.ssm_conv_kernel &&
        c.ssm_head_dim == p.ssm_head_dim && c.num_experts == p.num_experts &&
        c.num_experts_used == p.num_experts_used &&
        c.expert_groups == p.expert_groups &&
        c.expert_groups_used == p.expert_groups_used &&
        c.expert_ff == p.expert_ff &&
        c.shared_expert_ff == p.shared_expert_ff &&
        c.leading_dense_blocks == p.leading_dense_blocks &&
        c.dense_ff == p.dense_ff) {
      profile = &p;
      break;
    }
  }
  if (profile == nullptr) {
    r.Fail("unsupported Ling 3.0 kernel geometry");
    return std::nullopt;
  }
  c.profile_name = profile->name;
  c.mla_q_split = profile->mla_q_split;
  if (require_trunk) {
    // MLA layers sit at (layer + 1) % mla_stride == 0 in the trunk and on
    // the MTP block; every other block is KDA.
    for (std::uint32_t il = 0; il < c.num_layers_all; ++il) {
      const bool expected =
          ((il + 1) % profile->mla_stride == 0) || il >= c.num_layers;
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