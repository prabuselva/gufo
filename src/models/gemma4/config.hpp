#ifndef GUFO_MODELS_GEMMA4_CONFIG_HPP_
#define GUFO_MODELS_GEMMA4_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gemma4 {

/// Architecture parameters of a Gemma-4-26B-A4B GGUF artifact. The trunk
/// (`gemma4`) is a 30-layer hybrid of sliding-window attention (five of six
/// layers, head_dim 256, own V projection) and full attention (head_dim 512,
/// V derived from the K projection), each followed by a shared dense FFN and
/// a 128-expert top-8 softmax MoE. The draft (`gemma4-assistant`) is a
/// four-layer MTP model reading the trunk KV cache read-only. Every value is
/// read from the file; the validated-geometry check in `FromGguf` locks the
/// shape this runtime was written and tested for.
struct Config {
  bool is_draft{false};          ///< True for `gemma4-assistant`.
  std::uint32_t num_layers{0};   ///< Trunk 30, draft 4.
  std::uint32_t hidden_size{0};  ///< Trunk 2816, draft 1024.
  std::uint32_t vocab_size{0};   ///< 262144, from tokenizer.ggml.tokens.
  std::uint32_t context_length{0};
  float rms_eps{1e-6F};
  float logit_softcap{0.0F};  ///< Trunk 30; draft logits are uncapped (0).

  std::uint32_t num_heads{0};  ///< 16 on both classes and both models.
  std::vector<std::uint32_t> num_kv_heads;  ///< Per layer.
  std::vector<bool> is_swa;                 ///< sliding_window_pattern.

  std::uint32_t head_dim_full{0};   ///< attention.key_length, 512.
  std::uint32_t head_dim_swa{0};    ///< attention.key_length_swa, 256.
  std::uint32_t rope_dim_full{0};   ///< rope.dimension_count, 512.
  std::uint32_t rope_dim_swa{0};    ///< rope.dimension_count_swa, 256.
  float rope_theta{0.0F};           ///< rope.freq_base, 1e6 (full layers).
  float rope_theta_swa{0.0F};       ///< rope.freq_base_swa, 1e4 (SWA layers).
  std::uint32_t sliding_window{0};  ///< 1024.

  /// Trailing layers whose KV the draft reads instead of writing its own.
  /// Trunk 0; draft 4 (all of its layers).
  std::uint32_t shared_kv_layers{0};

  std::uint32_t ffn_length{0};        ///< Trunk shared FFN 2112, draft 8192.
  std::uint32_t num_experts{0};       ///< Trunk 128; draft is dense (0).
  std::uint32_t num_experts_used{0};  ///< 8.
  std::uint32_t expert_ff{0};         ///< 704.

  std::uint32_t nextn_predict_layers{0};  ///< Draft 4.
  std::uint32_t hidden_size_out{0};       ///< Draft: trunk hidden, 2816.

  [[nodiscard]] bool IsSwa(std::uint32_t layer) const noexcept {
    return layer < num_layers && is_swa[layer];
  }
  [[nodiscard]] std::uint32_t HeadDim(std::uint32_t layer) const noexcept {
    return IsSwa(layer) ? head_dim_swa : head_dim_full;
  }
  [[nodiscard]] std::uint32_t RopeDim(std::uint32_t layer) const noexcept {
    return IsSwa(layer) ? rope_dim_swa : rope_dim_full;
  }
  [[nodiscard]] float RopeTheta(std::uint32_t layer) const noexcept {
    return IsSwa(layer) ? rope_theta_swa : rope_theta;
  }
  [[nodiscard]] std::uint32_t NumKvHeads(std::uint32_t layer) const noexcept {
    return layer < num_kv_heads.size() ? num_kv_heads[layer] : 0;
  }
  [[nodiscard]] std::uint32_t AttentionQDim(std::uint32_t layer) const {
    return num_heads * HeadDim(layer);
  }
  [[nodiscard]] std::uint32_t AttentionKvDim(std::uint32_t layer) const {
    return NumKvHeads(layer) * HeadDim(layer);
  }
  /// Only SWA layers carry `attn_v`; full layers derive V from `attn_k`.
  [[nodiscard]] bool HasVProjection(std::uint32_t layer) const noexcept {
    return IsSwa(layer);
  }
  /// False on the trailing layers of the draft, which read trunk KV.
  [[nodiscard]] bool HasKv(std::uint32_t layer) const noexcept {
    return layer < num_layers - shared_kv_layers;
  }
  /// Trunk layer whose KV cache the draft layer reads (draft only).
  [[nodiscard]] std::uint32_t DraftTargetLayer(
      std::uint32_t layer, const Config& trunk) const noexcept {
    return IsSwa(layer) ? trunk.num_layers - 2 : trunk.num_layers - 1;
  }

  /// Locks the KV-share contract between a parsed draft and trunk: equal
  /// vocabulary, epsilon and window, `embedding_length_out` matching the
  /// trunk hidden size, and per-layer head dimensions, KV-head counts and
  /// rope bases matching the trunk layer whose cache is shared.
  [[nodiscard]] bool DraftMatches(const Config& trunk) const noexcept {
    if (!is_draft || trunk.is_draft || vocab_size != trunk.vocab_size ||
        rms_eps != trunk.rms_eps || sliding_window != trunk.sliding_window ||
        hidden_size_out != trunk.hidden_size ||
        shared_kv_layers != num_layers) {
      return false;
    }
    for (std::uint32_t layer = 0; layer < num_layers; ++layer) {
      const std::uint32_t target = DraftTargetLayer(layer, trunk);
      if (target >= trunk.num_layers ||
          HeadDim(layer) != trunk.HeadDim(target) ||
          NumKvHeads(layer) != trunk.NumKvHeads(target) ||
          RopeTheta(layer) != trunk.RopeTheta(target) ||
          num_heads != trunk.num_heads) {
        return false;
      }
    }
    return true;
  }

  /// Reads and validates `gemma4.*` or `gemma4-assistant.*` metadata,
  /// selecting the profile from `general.architecture`.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_CONFIG_HPP_