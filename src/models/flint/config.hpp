#ifndef GUFO_MODELS_FLINT_CONFIG_HPP_
#define GUFO_MODELS_FLINT_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::flint {

/// Architecture parameters of a Flint Preview (`flint`) GGUF artifact: a
/// 49-layer trunk where layers [`leading_dense_block_count`, num_layers)
/// route over 256 small sigmoid-gated experts plus 2 shared experts, and the
/// first layers use a plain dense FFN. Attention is grouped-query with full
/// rotary in the metadata sense but per-layer gated: `rope_pattern` and
/// `sliding_window_pattern` (measured identical) mark the layers that rotary
/// and the 513-token sliding window apply to; the other layers are NoPE and
/// read the whole cache. Every per-head QK norm and every attention-sink
/// vector is present in all layers. Values below are from the 40B-A2B Q8_0
/// artifact; `FromGguf` accepts only that measured geometry.
struct Config {
  std::string_view profile_name{""};  ///< Matched kernel geometry profile.
  std::uint32_t num_layers{0};        ///< 49.
  std::uint32_t hidden_size{0};       ///< n_embd, 2048.
  std::uint32_t vocab_size{0};  ///< tokenizer token list, 150144 (not stored).
  std::uint32_t context_length{0};  ///< 262144.
  float rms_eps{1e-6F};

  // Grouped-query attention. Q is num_heads * head_dim = 4096, wider than the
  // hidden size (the projection expands); K/V are num_kv_heads * head_dim.
  std::uint32_t num_heads{0};     ///< 32
  std::uint32_t num_kv_heads{0};  ///< 4
  std::uint32_t head_dim{0};      ///< attention.key_length == value_length,
                                  ///< 128

  // Rotary is present with theta 10000 and no scaling, but each layer is
  // gated by rope_pattern; false layers bypass rotary entirely. swa_layer
  // marks the layers restricted to the trailing `sliding_window` tokens.
  float rope_theta{10000.0F};
  std::vector<bool> rope_layer;     ///< [num_layers] attention.rope_pattern
  std::vector<bool> swa_layer;      ///< [num_layers] sliding_window_pattern
  std::uint32_t sliding_window{0};  ///< 513

  // Dense FFN width for the leading layers; the remaining layers are MoE.
  std::uint32_t leading_dense_blocks{0};  ///< 2
  std::uint32_t ffn_length{0};            ///< 6144

  // MoE: 256 routed experts with sigmoid gating (expert_gating_func 2),
  // top-6, weights optionally L2-normalized and scaled, plus 2 shared
  // experts of `shared_expert_ff` added outside the routed sum.
  std::uint32_t num_experts{0};         ///< 256
  std::uint32_t num_experts_used{0};    ///< 6
  std::uint32_t expert_ff{0};           ///< 512
  std::uint32_t shared_experts{0};      ///< expert_shared_count, 2
  std::uint32_t shared_expert_ff{0};    ///< 1024
  std::uint32_t expert_gating_func{0};  ///< 2 = sigmoid
  float expert_weights_scale{1.0F};
  bool expert_weights_norm{true};

  /// MoE iff the layer index reaches past the leading dense blocks.
  [[nodiscard]] bool IsMoeLayer(std::uint32_t layer) const noexcept {
    return layer >= leading_dense_blocks && layer < num_layers;
  }
  /// Rotary applies iff rope_pattern is set for the layer.
  [[nodiscard]] bool HasRope(std::uint32_t layer) const noexcept {
    return layer < rope_layer.size() && rope_layer[layer];
  }
  /// The layer attends to the trailing `sliding_window` tokens iff its
  /// sliding_window_pattern entry is set; otherwise the cache is global.
  [[nodiscard]] bool IsWindowedLayer(std::uint32_t layer) const noexcept {
    return layer < swa_layer.size() && swa_layer[layer];
  }

  [[nodiscard]] std::uint32_t AttentionQDim() const noexcept {
    return num_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t AttentionKvDim() const noexcept {
    return num_kv_heads * head_dim;
  }

  /// Reads and validates the `flint.*` metadata. Only the measured 40B-A2B
  /// geometry is accepted; shape-divisibility alone is not executable.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_CONFIG_HPP_
