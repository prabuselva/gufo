#ifndef GUFO_MODELS_FLINT_WEIGHTS_HPP_
#define GUFO_MODELS_FLINT_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/flint/config.hpp"

namespace gufo::models::flint {

/// Non-owning view of one GGUF tensor. `cols` x `rows` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole matrices.
/// The payload stays exactly as stored in the file (Q8_0 blocks for the ROCm
/// kernels, F32 for the CPU oracle); nothing is dequantized at load.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::uint64_t file_offset{0};  ///< Byte offset inside the owning shard.
  std::uint32_t shard{0};        ///< Mapped region index in the reader.
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * experts;
  }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  /// Base address of one stacked expert matrix.
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
};

struct LayerWeights {
  bool moe{false};      ///< False on the leading dense FFN layers.
  TensorRef attn_norm;  ///< [hidden] pre-attention RMSNorm, F32.
  TensorRef ffn_norm;   ///< [hidden] pre-FFN RMSNorm, F32.

  // Grouped-query attention with per-head RMS QK norms and additive
  // attention sinks; every layer carries all of them regardless of the
  // per-layer rotary/window gating.
  TensorRef attn_q;       ///< [hidden -> heads * head_dim].
  TensorRef attn_k;       ///< [hidden -> kv_heads * head_dim].
  TensorRef attn_v;       ///< [hidden -> kv_heads * head_dim].
  TensorRef attn_out;     ///< [heads * head_dim -> hidden].
  TensorRef attn_q_norm;  ///< [head_dim], F32.
  TensorRef attn_k_norm;  ///< [head_dim], F32.
  TensorRef attn_sinks;   ///< [heads], F32, extra softmax entry.

  // Dense FFN (leading layers only): SiLu-gated up-projection.
  TensorRef ffn_gate;  ///< [hidden -> ffn_length].
  TensorRef ffn_up;    ///< [hidden -> ffn_length].
  TensorRef ffn_down;  ///< [ffn_length -> hidden].

  // MoE (layers >= leading_dense_block_count). The router scores all experts
  // with a sigmoid; the top-6 weights are normalized and scaled. One shared
  // expert stream (`expert_shared_feed_forward_length`) is added once,
  // outside the routed sum.
  TensorRef router;         ///< [hidden -> num_experts], BF16 or F32.
  TensorRef ffn_gate_exps;  ///< [hidden -> expert_ff] x experts.
  TensorRef ffn_up_exps;    ///< [hidden -> expert_ff] x experts.
  TensorRef ffn_down_exps;  ///< [expert_ff -> hidden] x experts.
  TensorRef shexp_gate;     ///< [hidden -> shared_expert_ff].
  TensorRef shexp_up;       ///< [hidden -> shared_expert_ff].
  TensorRef shexp_down;     ///< [shared_expert_ff -> hidden].
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;   ///< [hidden -> vocab].
  TensorRef output;       ///< [hidden -> vocab]; falls back to token_embd.
  TensorRef output_norm;  ///< [hidden], F32, before the LM head.
  std::vector<LayerWeights> layers;

  /// Binds and validates the trunk artifact. Tensor payloads stay mapped and
  /// untouched; only headers are read.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_WEIGHTS_HPP_
