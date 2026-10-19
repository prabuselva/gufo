#ifndef GUFO_MODELS_NEMOTRON_H_MOE_CONFIG_HPP_
#define GUFO_MODELS_NEMOTRON_H_MOE_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::nemotron_h_moe {

/// Architecture parameters of a Nemotron 3.5 Bolt (`nemotron_h_moe`) GGUF
/// artifact: a hybrid trunk whose 88 layers each carry exactly one job —
/// Mamba-2 recurrent blocks, NoPE grouped-query attention blocks (no rotary;
/// `rope.dimension_count` is present but unused) and latent-MoE FFN blocks
/// with heterogeneous per-layer expert counts. Every value is read from the
/// file; `FromGguf` accepts only the measured Bolt geometry and records it in
/// `profile_name`. The documented values below are from the 74B-A9B 24GB
/// artifact.
struct Config {
  enum class LayerKind : std::uint8_t { kAttention, kMamba, kMoe };

  std::string_view profile_name{""};  ///< Matched kernel geometry profile.
  std::uint32_t num_layers{0};        ///< 88.
  std::uint32_t hidden_size{0};       ///< n_embd, 4096.
  std::uint32_t vocab_size{0};        ///< 131072.
  std::uint32_t context_length{0};    ///< 262144.
  float rms_eps{1e-5F};

  // NoPE attention (8 layers). Q is num_heads * head_dim with no rotary
  // split; KV heads are 2 per attention layer.
  std::uint32_t num_heads{0};   ///< 32
  std::uint32_t head_dim{0};    ///< attention.key_length == value_length,
                                ///< 128
  std::uint32_t head_dim_v{0};  ///< attention.value_length, 128
  std::vector<std::uint32_t> head_count_kv;  ///< [num_layers], 0 off attn.

  // Mamba-2 (40 layers). d_in_proj = 2*inner (x,z) + 2*groups*state (B,C)
  // + heads (dt, one per head); conv1d runs over the xBC channels only.
  // Proven by the 24GB weights: ssm_a/ssm_d/ssm_dt.bias hold one value per
  // head with 128 entries, ssm_conv1d spans 10240 = inner + 2*groups*state.
  std::uint32_t ssm_conv_kernel{0};     ///< 4
  std::uint32_t ssm_state_size{0};      ///< 128 (dstate of the B/C matrices)
  std::uint32_t ssm_group_count{0};     ///< 8 (n_groups for the B/C matrices)
  std::uint32_t ssm_inner_size{0};      ///< 8192 (x half before the z gate)
  std::uint32_t ssm_time_step_rank{0};  ///< 128 = head count (dt per head)

  // Latent MoE (40 layers). Each MoE layer routes over its own subset of
  // `max_experts` with sigmoid gating plus per-expert bias and top-k from a
  // per-layer array; expert up/down run in `expert_ff[i]`, the routed sum is
  // compressed through the latent bottleneck, and one shared expert of
  // `shared_expert_ff` is added outside the latent path.
  std::uint32_t max_experts{0};         ///< expert_count bound, 496
  std::uint32_t expert_groups{0};       ///< expert_group_count, 1
  std::uint32_t expert_groups_used{0};  ///< 1
  std::uint32_t expert_gating_func{0};  ///< 2 = sigmoid
  std::uint32_t shared_expert_ff{0};    ///< 5376
  std::uint32_t shared_experts{0};      ///< expert_shared_count, 1
  std::uint32_t moe_latent_size{0};     ///< 1024
  float expert_weights_scale{5.0F};
  bool expert_weights_norm{true};
  std::vector<std::uint32_t> layer_feed_forward;  ///< [num_layers], 0 off MoE
  std::vector<std::uint32_t> layer_expert_ff;     ///< [num_layers], 0 off MoE
  std::vector<std::uint32_t> layer_expert_used;   ///< [num_layers], 0 off MoE

  [[nodiscard]] LayerKind Kind(std::uint32_t layer) const noexcept {
    if (layer >= num_layers) {
      return LayerKind::kMamba;
    }
    if (head_count_kv[layer] != 0) {
      return LayerKind::kAttention;
    }
    return layer_feed_forward[layer] != 0 ? LayerKind::kMoe : LayerKind::kMamba;
  }
  [[nodiscard]] bool IsAttentionLayer(std::uint32_t layer) const noexcept {
    return Kind(layer) == LayerKind::kAttention;
  }
  [[nodiscard]] bool IsMambaLayer(std::uint32_t layer) const noexcept {
    return Kind(layer) == LayerKind::kMamba;
  }
  [[nodiscard]] bool IsMoeLayer(std::uint32_t layer) const noexcept {
    return Kind(layer) == LayerKind::kMoe;
  }

  [[nodiscard]] std::uint32_t ExpertFf(std::uint32_t layer) const noexcept {
    return layer_feed_forward[layer];
  }
  [[nodiscard]] std::uint32_t ExpertsUsed(std::uint32_t layer) const noexcept {
    return layer_expert_used[layer];
  }

  /// Mamba-2 head count: dt carries one channel per head, so the dt width of
  /// `ssm_in` is the head count (128).
  std::uint32_t SsmHeadCount() const noexcept { return ssm_time_step_rank; }
  /// Per-head state width the SSM einsums over (inner / heads = 64).
  std::uint32_t SsmHeadDim() const noexcept {
    return ssm_inner_size / SsmHeadCount();
  }
  /// Columns of `ssm_in`: x,z halves plus B,C and dt.
  std::uint32_t SsmInProj() const noexcept {
    return 2 * ssm_inner_size + 2 * ssm_group_count * ssm_state_size +
           ssm_time_step_rank;
  }
  /// Conv1d channel count over the x‖B‖C split.
  std::uint32_t SsmConvChannels() const noexcept {
    return ssm_inner_size + 2 * ssm_group_count * ssm_state_size;
  }
  [[nodiscard]] std::uint32_t AttentionQDim() const noexcept {
    return num_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t AttentionKvDim(std::uint32_t layer) const {
    return head_count_kv[layer] * head_dim_v;
  }

  /// Reads and validates the `nemotron_h_moe.*` metadata. Only the measured
  /// Bolt geometry is accepted; shape-divisibility alone is not executable.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::nemotron_h_moe

#endif  // GUFO_MODELS_NEMOTRON_H_MOE_CONFIG_HPP_
