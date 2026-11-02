#include "src/models/nemotron_h_moe/config.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

namespace gufo::models::nemotron_h_moe {
namespace {

constexpr std::string_view kArchitecture = "nemotron_h_moe";

// The single artifact geometry whose kernels are built and measured end to
// end. Meeting the structural checks below is not enough to run; the shape
// must match this measured profile.
struct GeometryProfile {
  std::string_view name;
  std::uint32_t num_layers;
  std::uint32_t hidden_size;
  std::uint32_t vocab_size;
  std::uint32_t num_heads;
  std::uint32_t head_dim;
  std::uint32_t head_dim_v;
  std::uint32_t kv_heads;  // per attention layer
  std::uint32_t ssm_conv_kernel;
  std::uint32_t ssm_state_size;
  std::uint32_t ssm_group_count;
  std::uint32_t ssm_inner_size;
  std::uint32_t ssm_heads;  // dt channels == per-head dt
  std::uint32_t ssm_head_dim;
  std::uint32_t max_experts;
  std::uint32_t max_experts_used;
  std::uint32_t shared_expert_ff;
  std::uint32_t moe_latent_size;
};

constexpr GeometryProfile kProfiles[] = {
    {
        .name = "NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB",
        .num_layers = 88,
        .hidden_size = 4096,
        .vocab_size = 131072,
        .num_heads = 32,
        .head_dim = 128,
        .head_dim_v = 128,
        .kv_heads = 2,
        .ssm_conv_kernel = 4,
        .ssm_state_size = 128,
        .ssm_group_count = 8,
        .ssm_inner_size = 8192,
        .ssm_heads = 128,
        .ssm_head_dim = 64,
        .max_experts = 496,
        .max_experts_used = 9,
        .shared_expert_ff = 5376,
        .moe_latent_size = 1024,
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

/// Copies a validated per-layer integer array; every value must fit u32.
bool CopyLayerArray(const std::vector<std::uint64_t>& src,
                    std::uint32_t num_layers, std::vector<std::uint32_t>* dst,
                    Reader* r, std::string_view key) {
  if (src.size() != num_layers) {
    r->Fail(std::string(key) + " must cover every block");
    return false;
  }
  for (const auto v : src) {
    if (v > 0xFFFFFFFFULL) {
      r->Fail(std::string(key) + " value exceeds u32");
      return false;
    }
    dst->push_back(static_cast<std::uint32_t>(v));
  }
  return true;
}

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not nemotron_h_moe");
    return std::nullopt;
  }

  Config c;
  const std::string_view p = "nemotron_h_moe.";
  auto key = [&](std::string_view suffix) {
    return std::string(p) + std::string(suffix);
  };

  c.num_layers = r.U32(key("block_count"));
  c.hidden_size = r.U32(key("embedding_length"));
  c.vocab_size = r.U32(key("vocab_size"));
  c.context_length = r.U32(key("context_length"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), 1e-5F);

  c.num_heads = r.U32(key("attention.head_count"));
  c.head_dim = r.U32(key("attention.key_length"));
  c.head_dim_v = r.U32(key("attention.value_length"));
  if (!CopyLayerArray(r.U64Array(key("attention.head_count_kv")), c.num_layers,
                      &c.head_count_kv, &r, key("attention.head_count_kv"))) {
    return std::nullopt;
  }

  c.ssm_conv_kernel = r.U32(key("ssm.conv_kernel"));
  c.ssm_state_size = r.U32(key("ssm.state_size"));
  c.ssm_group_count = r.U32(key("ssm.group_count"));
  c.ssm_inner_size = r.U32(key("ssm.inner_size"));
  c.ssm_time_step_rank = r.U32(key("ssm.time_step_rank"));

  c.max_experts = r.U32(key("expert_count"));
  c.expert_groups = r.U32(key("expert_group_count"), false, 1);
  c.expert_groups_used =
      r.U32(key("expert_group_used_count"), false, c.expert_groups);
  c.expert_gating_func = r.U32(key("expert_gating_func"));
  c.shared_expert_ff = r.U32(key("expert_shared_feed_forward_length"));
  c.shared_experts = r.U32(key("expert_shared_count"), false, 1);
  c.moe_latent_size = r.U32(key("moe_latent_size"));
  c.expert_weights_scale = r.F32(key("expert_weights_scale"), 1.0F, false);
  c.expert_weights_norm = r.Boolean(key("expert_weights_norm"), false, false);
  const bool copy_ff =
      CopyLayerArray(r.U64Array(key("feed_forward_length")), c.num_layers,
                     &c.layer_feed_forward, &r, key("feed_forward_length"));
  const bool copy_ee = CopyLayerArray(
      r.U64Array(key("expert_feed_forward_length")), c.num_layers,
      &c.layer_expert_ff, &r, key("expert_feed_forward_length"));
  const bool copy_eseq =
      CopyLayerArray(r.U64Array(key("expert_used_count")), c.num_layers,
                     &c.layer_expert_used, &r, key("expert_used_count"));
  if (!copy_ff || !copy_ee || !copy_eseq) {
    return std::nullopt;
  }
  if (!r.ok) {
    return std::nullopt;
  }

  // Structural limits of this runtime, not of the format.
  if (c.num_layers == 0 || c.context_length == 0 || c.vocab_size == 0 ||
      !std::isfinite(c.rms_eps) || c.rms_eps <= 0 || c.hidden_size == 0 ||
      c.num_heads == 0 || c.head_dim == 0 || c.head_dim_v == 0 ||
      c.ssm_conv_kernel == 0 || c.ssm_state_size == 0 ||
      c.ssm_group_count == 0 || c.ssm_inner_size == 0 ||
      c.ssm_time_step_rank == 0 ||
      c.ssm_inner_size % c.ssm_time_step_rank != 0 || c.max_experts == 0 ||
      c.expert_groups == 0 || c.expert_groups_used == 0 ||
      c.expert_groups_used > c.expert_groups || c.expert_gating_func != 2 ||
      c.shared_expert_ff == 0 || c.shared_experts != 1 ||
      c.moe_latent_size == 0 || !std::isfinite(c.expert_weights_scale)) {
    r.Fail("nemotron_h_moe metadata describes an unsupported shape");
    return std::nullopt;
  }

  // Exactly one job per layer: attention iff head_count_kv is set, MoE iff
  // the layer carry a feed-forward width, Mamba otherwise. The MoE width,
  // expert subset size and routed width arrays must agree layer by layer, and
  // at least one layer of every kind must exist (heterogeneous per-layer
  // expert counts are bounded by max_experts).
  std::uint32_t attn_layers = 0;
  std::uint32_t mamba_layers = 0;
  std::uint32_t moe_layers = 0;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::uint32_t kv = c.head_count_kv[l];
    if (kv != 0 && kv > c.num_heads) {
      r.Fail("attention.head_count_kv exceeds the head count");
      return std::nullopt;
    }
    if (c.layer_feed_forward[l] != c.layer_expert_ff[l]) {
      r.Fail("feed_forward_length and expert_feed_forward_length must match");
      return std::nullopt;
    }
    const std::uint32_t used = c.layer_expert_used[l];
    const bool attention = kv != 0;
    const bool moe = used != 0 || c.layer_feed_forward[l] != 0;
    if (attention && moe) {
      r.Fail("layer mix must be attention, mamba or MoE (exactly one)");
      return std::nullopt;
    }
    if (moe) {
      if (c.layer_feed_forward[l] == 0 || used == 0 || used > c.max_experts) {
        r.Fail("expert_used_count requires a MoE layer within expert_count");
        return std::nullopt;
      }
      ++moe_layers;
    } else if (attention) {
      ++attn_layers;
    } else {
      ++mamba_layers;
    }
  }
  if (attn_layers == 0 || mamba_layers == 0 || moe_layers == 0) {
    r.Fail("nemotron_h_moe trunk must mix attention, mamba and MoE layers");
    return std::nullopt;
  }

  // These dimensions select the specialized kernels, cache strides and
  // expert buckets. Reject artifacts outside the measured profile before
  // allocating or launching.
  const GeometryProfile* profile = nullptr;
  for (const GeometryProfile& g : kProfiles) {
    if (c.num_layers == g.num_layers && c.hidden_size == g.hidden_size &&
        c.vocab_size == g.vocab_size && c.num_heads == g.num_heads &&
        c.head_dim == g.head_dim && c.head_dim_v == g.head_dim_v &&
        c.ssm_conv_kernel == g.ssm_conv_kernel &&
        c.ssm_state_size == g.ssm_state_size &&
        c.ssm_group_count == g.ssm_group_count &&
        c.ssm_inner_size == g.ssm_inner_size &&
        c.SsmHeadCount() == g.ssm_heads && c.SsmHeadDim() == g.ssm_head_dim &&
        c.max_experts == g.max_experts &&
        c.shared_expert_ff == g.shared_expert_ff &&
        c.moe_latent_size == g.moe_latent_size) {
      profile = &g;
      break;
    }
  }
  if (profile == nullptr) {
    r.Fail("unsupported nemotron_h_moe kernel geometry");
    return std::nullopt;
  }
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::uint32_t kv = c.head_count_kv[l];
    if (kv != 0 && kv != profile->kv_heads) {
      r.Fail("attention head_count_kv differs from the supported profile");
      return std::nullopt;
    }
    if (c.layer_expert_used[l] > profile->max_experts_used) {
      r.Fail("expert_used_count exceeds the supported top-k");
      return std::nullopt;
    }
  }
  c.profile_name = profile->name;
  return c;
}

}  // namespace gufo::models::nemotron_h_moe
