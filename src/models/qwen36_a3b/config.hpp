#ifndef GUFO_MODELS_QWEN36_A3B_CONFIG_HPP_
#define GUFO_MODELS_QWEN36_A3B_CONFIG_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen36_a3b {

/// Architecture parameters of a Qwen3.6-35B-A3B (`qwen35moe`) GGUF artifact:
/// a hybrid trunk of Gated DeltaNet linear attention and gated grouped-query
/// full attention, a sigmoid-gated Mixture-of-Experts FFN with one always-on
/// shared expert, and a self-speculative MTP block. Every value is read from
/// the file; the validated-geometry check in `FromGguf` locks the shape this
/// runtime was written and tested for.
///
/// This model is intentionally independent of the Qwen3.8-27B (`qwen35`) and
/// Qwen3.8-Flash-Next (`qwen4exp`) engines: it carries no hyper-connections,
/// no QSA indexer and no PLE n-gram embedding.
struct Config {
  std::uint32_t num_layers{0};      ///< Trunk layers (MTP block excluded).
  std::uint32_t num_layers_all{0};  ///< block_count, includes the MTP block.
  std::uint32_t hidden_size{0};     ///< n_embd, 2048.
  std::uint32_t vocab_size{0};      ///< 248320.
  std::uint32_t context_length{0};  ///< 262144.
  float rms_eps{1e-6F};

  // Gated grouped-query full attention on every full_attention_interval-th
  // layer. The Q projection emits query and gate channels; the attention
  // output is multiplied by sigmoid(gate).
  std::uint32_t full_attention_interval{0};  ///< 4
  std::uint32_t num_heads{0};                ///< 16
  std::uint32_t num_kv_heads{0};             ///< 2
  std::uint32_t head_dim{0};                 ///< 256
  std::uint32_t rotary_dim{0};               ///< 64
  float rope_theta{0.0F};                    ///< 1e7
  /// mRoPE axis dimensions (three axes plus a zero pad). Text-only decoding
  /// uses equal positions on every axis, which reduces to plain partial
  /// rotary; the sections are retained for the deferred vision path.
  std::array<std::uint32_t, 4> rope_sections{};

  // Gated DeltaNet linear attention.
  std::uint32_t ssm_conv_kernel{0};  ///< 4
  std::uint32_t ssm_head_dim{0};     ///< 128 (state_size; key and value dim)
  std::uint32_t ssm_num_k_heads{0};  ///< 16 (group_count)
  std::uint32_t ssm_num_v_heads{0};  ///< 32 (time_step_rank)
  std::uint32_t ssm_inner_size{0};   ///< 4096

  // Mixture of experts with one always-on shared expert.
  std::uint32_t num_experts{0};       ///< 256
  std::uint32_t num_experts_used{0};  ///< 8
  std::uint32_t expert_ff{0};         ///< 512
  std::uint32_t shared_expert_ff{0};  ///< 512

  // Speculative MTP block (`nextn`), present in the draft sidecar only.
  std::uint32_t nextn_layers{0};

  [[nodiscard]] std::uint32_t SsmKeyDim() const noexcept {
    return ssm_num_k_heads * ssm_head_dim;
  }
  [[nodiscard]] std::uint32_t SsmValueDim() const noexcept {
    return ssm_num_v_heads * ssm_head_dim;
  }
  /// Channels of the fused q|k|v projection and its causal convolution.
  [[nodiscard]] std::uint32_t SsmConvChannels() const noexcept {
    return 2 * SsmKeyDim() + SsmValueDim();
  }
  [[nodiscard]] std::uint32_t AttentionQDim() const noexcept {
    return num_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t AttentionKvDim() const noexcept {
    return num_kv_heads * head_dim;
  }

  [[nodiscard]] bool IsLinearLayer(std::uint32_t layer) const noexcept {
    return layer < num_layers && ((layer + 1) % full_attention_interval) != 0;
  }

  /// The draft executes with trunk constants and shared vocabulary weights.
  /// SSM parameters are intentionally excluded: the draft is full attention.
  [[nodiscard]] bool MtpMatches(const Config& trunk) const noexcept {
    return nextn_layers == 1 && num_layers == trunk.num_layers &&
           hidden_size == trunk.hidden_size &&
           context_length == trunk.context_length && rms_eps == trunk.rms_eps &&
           num_heads == trunk.num_heads && num_kv_heads == trunk.num_kv_heads &&
           head_dim == trunk.head_dim && rotary_dim == trunk.rotary_dim &&
           rope_theta == trunk.rope_theta &&
           rope_sections == trunk.rope_sections &&
           num_experts == trunk.num_experts &&
           num_experts_used == trunk.num_experts_used &&
           expert_ff == trunk.expert_ff &&
           shared_expert_ff == trunk.shared_expert_ff;
  }

  /// Reads and validates the `qwen35moe.*` metadata. `require_trunk` rejects
  /// draft-only sidecars; the MTP sidecar is read with it false.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, bool require_trunk,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_CONFIG_HPP_