#ifndef GUFO_MODELS_GEMMA4_WEIGHTS_HPP_
#define GUFO_MODELS_GEMMA4_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/config.hpp"

namespace gufo::models::gemma4 {

/// Non-owning view of one GGUF tensor. `cols` x `rows` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole matrices.
/// The payload stays exactly as stored in the file (quantized blocks for the
/// ROCm kernels, F32 for the CPU oracle); nothing is dequantized at load.
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

/// One transformer layer. Trunk layers fill every field; draft layers fill
/// only the Q projection, the dense FFN and their norms, because they read
/// the trunk KV cache instead of carrying K/V weights.
struct LayerWeights {
  TensorRef attn_norm;  ///< [hidden] pre-attention RMSNorm, F32.
  TensorRef attn_q;     ///< [hidden -> heads * HeadDim(layer)].
  TensorRef attn_k;     ///< [hidden -> kv_heads * HeadDim(layer)].
  TensorRef attn_v;     ///< SWA layers only; full layers derive V from attn_k.
  TensorRef attn_q_norm;          ///< [HeadDim(layer)], F32.
  TensorRef attn_k_norm;          ///< [HeadDim(layer)], trunk only, F32.
  TensorRef attn_output;          ///< [heads * HeadDim(layer) -> hidden].
  TensorRef post_attention_norm;  ///< [hidden], F32, on the attention output.

  TensorRef ffn_norm;       ///< [hidden], F32, shared FFN input.
  TensorRef ffn_gate;       ///< [hidden -> ffn_length].
  TensorRef ffn_up;         ///< [hidden -> ffn_length].
  TensorRef ffn_down;       ///< [ffn_length -> hidden].
  TensorRef post_ffw_norm;  ///< [hidden], F32, before the FFN residual add.

  // Top-8 softmax MoE (trunk layers only). The router scores the RMS-normal
  /// attention output scaled by `router_scale`, elementwise.
  TensorRef router;               ///< [hidden -> experts], F32.
  TensorRef router_scale;         ///< [hidden], F32.
  TensorRef pre_ffw_norm_2;       ///< [hidden], F32, routed branch input.
  TensorRef post_ffw_norm_1;      ///< [hidden], F32, after the shared branch.
  TensorRef post_ffw_norm_2;      ///< [hidden], F32, after the routed branch.
  TensorRef ffn_gate_up_exps;     ///< [hidden -> 2*expert_ff] x experts.
  TensorRef ffn_down_exps;        ///< [expert_ff -> hidden] x experts.
  TensorRef ffn_down_exps_scale;  ///< [experts], F32.

  TensorRef layer_output_scale;  ///< [1], F32, final layer scalar.
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;   ///< [hidden -> vocab].
  TensorRef output;       ///< [hidden -> vocab]; falls back to token_embd.
  TensorRef output_norm;  ///< [hidden], F32, before the LM head.
  TensorRef rope_freqs;   ///< [head_dim_full / 2], F32, full-attention layers.
  std::vector<LayerWeights> layers;

  /// Binds and validates the trunk artifact. Tensor payloads stay mapped and
  /// untouched; only headers are read.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

/// The MTP draft from the sidecar artifact: four layers reading the trunk KV
/// cache, with `pre_projection` consuming the concatenation of the scaled
/// target token embedding and the trunk post-norm hidden state (both trunk
/// width, embedding first) and `post_projection` mapping back to trunk width
/// for the next step.
struct DraftWeights {
  Config config;
  TensorRef pre_projection;   ///< [2*hidden_out -> hidden].
  TensorRef post_projection;  ///< [hidden -> hidden_out].
  TensorRef token_embd;       ///< Draft LM head [hidden -> vocab].
  TensorRef output_norm;      ///< [hidden], F32.
  TensorRef rope_freqs;       ///< [head_dim_full / 2], F32.
  std::vector<LayerWeights> layers;

  /// Binds the draft and checks the KV-share contract against `trunk`.
  [[nodiscard]] static std::optional<DraftWeights> Bind(
      const core::GgufReader& reader, const Config& trunk,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_WEIGHTS_HPP_