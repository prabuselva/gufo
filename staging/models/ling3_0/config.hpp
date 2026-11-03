#ifndef GUFO_MODELS_LING3_FLASH_CONFIG_HPP_
#define GUFO_MODELS_LING3_FLASH_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::ling3_0 {

/// Architecture parameters of a Ling 3.0 (`bailingmoe3`) GGUF artifact:
/// a hybrid trunk of Kimi Delta Attention (KDA) linear layers and absorbed
/// Multi-head Latent Attention (MLA) layers, a sigmoid-gated Mixture-of-Experts
/// FFN with group-limited top-k routing and one always-on shared expert, and
/// the optional self-speculative MTP block. Every value is read from the file;
/// the geometry check in `FromGguf` accepts only the measured profiles (Ling
/// 3.0 Flash and Tiny) and records the matched one in `profile_name`. The
/// documented dimensions below are the Flash profile's values.
struct Config {
  std::string_view profile_name{""};  ///< Matched kernel geometry profile.
  std::uint32_t num_layers{0};        ///< Trunk layers (MTP block excluded).
  std::uint32_t num_layers_all{0};    ///< block_count, includes the MTP block.
  std::uint32_t hidden_size{0};       ///< n_embd, 2560.
  std::uint32_t vocab_size{0};        ///< 157184.
  std::uint32_t context_length{0};    ///< 262144.
  float rms_eps{1e-6F};

  // Absorbed MLA on every layer with a non-zero head_count_kv entry. The Q
  // projection emits per-head nope|rope halves; the KV projection emits the
  // latent plus a shared rope key.
  std::uint32_t num_heads{0};      ///< 32
  std::uint32_t mla_head_dim{0};   ///< key_length_mla, 192 (nope+rope)
  std::uint32_t mla_value_dim{0};  ///< value_length_mla, 128
  std::uint32_t kv_lora_rank{0};   ///< 512
  std::uint32_t rotary_dim{0};     ///< 64
  std::uint32_t mla_q_lora{0};     ///< q_lora_rank metadata; 0 when fused q.
  bool mla_q_split{false};  ///< True when q is attn_q_a, norm and attn_q_b.
  float rope_theta{0.0F};   ///< 6e6
  std::vector<std::uint32_t> head_count_kv;  ///< [num_layers_all], 0 on KDA.

  // Kimi Delta Attention (linear layers). q/k/v are separate projections.
  std::uint32_t ssm_conv_kernel{0};  ///< 4
  std::uint32_t ssm_head_dim{0};     ///< kda.head_dim, 128
  float kda_gate_lower_bound{0.0F};  ///< -5.0
  bool kda_safe_gate{true};

  // Mixture of experts with group-limited top-k and one shared expert.
  std::uint32_t num_experts{0};         ///< 512
  std::uint32_t num_experts_used{0};    ///< 8
  std::uint32_t expert_groups{0};       ///< 8
  std::uint32_t expert_groups_used{0};  ///< 4
  std::uint32_t expert_ff{0};           ///< 768
  std::uint32_t shared_expert_ff{0};    ///< 768
  float expert_weights_scale{2.5F};
  bool expert_weights_norm{true};
  std::uint32_t expert_gating_func{2};  ///< 2 = sigmoid

  // Dense FFN prefix and per-layer SwiGLU clamps (0 disables the clamp; the
  // arrays themselves are absent on artifacts that never clamp, like Tiny).
  std::uint32_t leading_dense_blocks{0};  ///< 2
  std::uint32_t dense_ff{0};              ///< 6144
  std::vector<float> swiglu_clamp_exp;    ///< [num_layers_all] or empty
  std::vector<float> swiglu_clamp_shexp;  ///< [num_layers_all] or empty

  // Speculative MTP blocks (`nextn`), the trailing blocks of the artifact;
  // zero when the artifact ships without a draft (Tiny).
  std::uint32_t nextn_layers{0};

  [[nodiscard]] std::uint32_t SsmInnerSize() const noexcept {
    return ssm_head_dim * num_heads;
  }
  [[nodiscard]] std::uint32_t MlaQDim() const noexcept {
    return num_heads * mla_head_dim;
  }
  /// Per-head nope width of the MLA query, the rank absorbed by attn_k_b.
  [[nodiscard]] std::uint32_t MlaNopeDim() const noexcept {
    return mla_head_dim - rotary_dim;
  }
  [[nodiscard]] std::uint32_t MlaVDim() const noexcept {
    return num_heads * mla_value_dim;
  }
  /// MLA key width: absorbed latent plus the shared rope key.
  [[nodiscard]] std::uint32_t MlaKeyDim() const noexcept {
    return kv_lora_rank + rotary_dim;
  }

  [[nodiscard]] bool IsMlaLayer(std::uint32_t layer) const noexcept {
    return layer < head_count_kv.size() && head_count_kv[layer] != 0;
  }
  [[nodiscard]] bool IsDenseFfnLayer(std::uint32_t layer) const noexcept {
    return layer < leading_dense_blocks;
  }
  /// SwiGLU clamp limits of a block; 0 when the artifact has no clamp arrays.
  [[nodiscard]] float ClampExp(std::uint32_t layer) const noexcept {
    return layer < swiglu_clamp_exp.size() ? swiglu_clamp_exp[layer] : 0.0F;
  }
  [[nodiscard]] float ClampShexp(std::uint32_t layer) const noexcept {
    return layer < swiglu_clamp_shexp.size() ? swiglu_clamp_shexp[layer] : 0.0F;
  }

  /// The draft executes with trunk constants and shared vocabulary weights.
  [[nodiscard]] bool MtpMatches(const Config& trunk) const noexcept {
    return nextn_layers == 1 && num_layers == trunk.num_layers &&
           hidden_size == trunk.hidden_size &&
           context_length == trunk.context_length && rms_eps == trunk.rms_eps &&
           num_heads == trunk.num_heads && mla_head_dim == trunk.mla_head_dim &&
           mla_value_dim == trunk.mla_value_dim &&
           kv_lora_rank == trunk.kv_lora_rank &&
           rotary_dim == trunk.rotary_dim && rope_theta == trunk.rope_theta &&
           num_experts == trunk.num_experts &&
           num_experts_used == trunk.num_experts_used &&
           expert_groups == trunk.expert_groups &&
           expert_groups_used == trunk.expert_groups_used &&
           expert_ff == trunk.expert_ff &&
           shared_expert_ff == trunk.shared_expert_ff;
  }

  /// Reads and validates the `bailingmoe3.*` metadata. `require_trunk` rejects
  /// draft-only sidecars; the MTP sidecar is read with it false.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, bool require_trunk,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::ling3_0

#endif  // GUFO_MODELS_LING3_FLASH_CONFIG_HPP_