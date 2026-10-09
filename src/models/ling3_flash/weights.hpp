#ifndef GUFO_MODELS_LING3_FLASH_WEIGHTS_HPP_
#define GUFO_MODELS_LING3_FLASH_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling3_flash/config.hpp"

namespace gufo::models::ling3_flash {

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
  /// Base address of one stacked matrix (expert or attention head).
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
};

struct LayerWeights {
  bool mla{false};      ///< True on absorbed-MLA layers, false on KDA layers.
  TensorRef attn_norm;  ///< [hidden] pre-attention RMSNorm, F32.
  TensorRef ffn_norm;   ///< [hidden] pre-FFN RMSNorm, F32.

  // Kimi Delta Attention (KDA layers). q/k/v are separate projections.
  TensorRef ssm_q;       ///< [hidden -> inner], inner = heads * head_dim.
  TensorRef ssm_k;       ///< [hidden -> inner].
  TensorRef ssm_v;       ///< [hidden -> inner].
  TensorRef ssm_conv_q;  ///< [conv_kernel, 1, inner] F32, taps contiguous.
  TensorRef ssm_conv_k;  ///< [conv_kernel, 1, inner] F32.
  TensorRef ssm_conv_v;  ///< [conv_kernel, 1, inner] F32.
  TensorRef ssm_f;       ///< [hidden -> inner], the decay input f.
  TensorRef ssm_g;       ///< [hidden -> inner], the output gate g_proj.
  TensorRef ssm_beta;    ///< [hidden -> heads], F32.
  TensorRef ssm_a;       ///< [1, heads] = exp(A_log), F32.
  TensorRef ssm_dt;      ///< [inner] dt bias, F32.
  TensorRef ssm_norm;    ///< [head_dim], F32, the per-head gated RMSNorm.
  TensorRef ssm_out;     ///< [inner -> hidden] (attn_output on KDA layers).

  // Absorbed MLA (MLA layers).
  TensorRef attn_q;          ///< [hidden -> heads * (nope + rope)].
  TensorRef attn_kv_a;       ///< [hidden -> kv_lora_rank + rope].
  TensorRef attn_kv_a_norm;  ///< [kv_lora_rank], F32.
  TensorRef attn_k_b;        ///< [nope, kv_lora_rank] x heads (wk_b).
  TensorRef attn_v_b;        ///< [kv_lora_rank, value_dim] x heads (wv_b).
  TensorRef attn_gate;       ///< [hidden -> heads], per-head sigmoid gate.
  TensorRef attn_out;        ///< [heads * value_dim -> hidden].

  // Dense FFN (leading layers only).
  TensorRef ffn_gate;  ///< [hidden -> dense_ff].
  TensorRef ffn_up;    ///< [hidden -> dense_ff].
  TensorRef ffn_down;  ///< [dense_ff -> hidden].

  // Mixture of experts with one always-on shared expert.
  TensorRef router;         ///< [hidden -> num_experts], F32.
  TensorRef exp_probs_b;    ///< [num_experts] selection bias, F32.
  TensorRef ffn_gate_exps;  ///< [hidden -> expert_ff] x experts.
  TensorRef ffn_up_exps;    ///< [hidden -> expert_ff] x experts.
  TensorRef ffn_down_exps;  ///< [expert_ff -> hidden] x experts.
  TensorRef shexp_gate;     ///< [hidden -> shared_ff].
  TensorRef shexp_up;       ///< [hidden -> shared_ff].
  TensorRef shexp_down;     ///< [shared_ff -> hidden].

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

/// The MTP draft block from the same artifact: one absorbed-MLA layer with its
/// own norms and experts; token embedding and LM head are borrowed from the
/// trunk through `nextn_shared_head_norm`.
struct MtpWeights {
  Config config;
  LayerWeights block;

  [[nodiscard]] static std::optional<MtpWeights> Bind(
      const core::GgufReader& reader, const Config& trunk,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::ling3_flash

#endif  // GUFO_MODELS_LING3_FLASH_WEIGHTS_HPP_