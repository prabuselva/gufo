#ifndef GUFO_MODELS_QWEN36_A3B_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN36_A3B_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen36_a3b/config.hpp"

namespace gufo::models::qwen36_a3b {

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

struct LayerWeights {
  bool linear{false};   ///< True on Gated DeltaNet layers, false on full GQA.
  TensorRef attn_norm;  ///< [hidden] pre-attention RMSNorm, F32.
  TensorRef post_attention_norm;  ///< [hidden] pre-FFN RMSNorm, F32.

  // Gated DeltaNet (linear layers).
  TensorRef ssm_qkv;     ///< [hidden -> 2*key_dim + value_dim]
  TensorRef ssm_gate;    ///< [hidden -> value_dim], the z output gate.
  TensorRef ssm_conv1d;  ///< [conv_kernel, 2*key_dim + value_dim], F32.
  TensorRef ssm_alpha;   ///< [hidden -> v_heads], F32 or Q8_0.
  TensorRef ssm_beta;    ///< [hidden -> v_heads], F32 or Q8_0.
  TensorRef ssm_dt;      ///< [v_heads] softplus bias, F32.
  TensorRef ssm_a;       ///< [v_heads] = -exp(A_log), F32.
  TensorRef ssm_norm;    ///< [ssm_head_dim], F32.
  TensorRef ssm_out;     ///< [value_dim -> hidden].

  // Gated grouped-query full attention (full-attention layers). The Q
  // projection interleaves query and gate channels per head.
  TensorRef attn_q;       ///< [hidden -> heads * 2 * head_dim].
  TensorRef attn_k;       ///< [hidden -> kv_heads * head_dim].
  TensorRef attn_v;       ///< [hidden -> kv_heads * head_dim].
  TensorRef attn_out;     ///< [heads * head_dim -> hidden].
  TensorRef attn_q_norm;  ///< [head_dim], F32.
  TensorRef attn_k_norm;  ///< [head_dim], F32.

  // Mixture of experts with one always-on shared expert.
  TensorRef router;          ///< [hidden -> num_experts], F32 or BF16.
  TensorRef ffn_gate_exps;   ///< [hidden -> expert_ff] x experts.
  TensorRef ffn_up_exps;     ///< [hidden -> expert_ff] x experts.
  TensorRef ffn_down_exps;   ///< [expert_ff -> hidden] x experts.
  TensorRef shexp_gate_inp;  ///< [hidden] -> scalar sigmoid gate, F32/BF16.
  TensorRef shexp_gate;      ///< [hidden -> shared_ff].
  TensorRef shexp_up;        ///< [hidden -> shared_ff].
  TensorRef shexp_down;      ///< [shared_ff -> hidden].

  // Speculative `nextn` block only.
  TensorRef nextn_enorm;             ///< [hidden], F32.
  TensorRef nextn_hnorm;             ///< [hidden], F32.
  TensorRef nextn_eh_proj;           ///< [2*hidden -> hidden].
  TensorRef nextn_shared_head_norm;  ///< [hidden], F32, before the shared head.
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

/// The MTP draft block from the same artifact: one full-attention layer with
/// its own norms and experts; token embedding and LM head are borrowed from
/// the trunk through `nextn_shared_head_norm`.
struct MtpWeights {
  Config config;
  LayerWeights block;

  [[nodiscard]] static std::optional<MtpWeights> Bind(
      const core::GgufReader& reader, const Config& trunk,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_WEIGHTS_HPP_