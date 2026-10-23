#ifndef GUFO_MODELS_NEMOTRON_H_MOE_WEIGHTS_HPP_
#define GUFO_MODELS_NEMOTRON_H_MOE_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/nemotron_h_moe/config.hpp"

namespace gufo::models::nemotron_h_moe {

/// Non-owning view of one GGUF tensor. `cols` x `rows` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole
/// matrices. The payload stays exactly as stored in the file (quantized
/// blocks for the ROCm kernels, F32 for the CPU oracle); nothing is
/// dequantized at load.
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
  /// Base address of one stacked matrix (routed expert).
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
};

/// Tensor set of one trunk block. Exactly one kind is bound, matching
/// `Config::Kind`: the eight attention blocks, the forty Mamba-2 blocks or
/// the forty latent-MoE blocks. `attn_norm` is the single pre-block RMSNorm
/// every layer carries (the name comes from the GGUF).
struct LayerWeights {
  Config::LayerKind kind{Config::LayerKind::kMamba};
  TensorRef attn_norm;  ///< [hidden] F32.

  // NoPE attention layers.
  TensorRef attn_q;    ///< [hidden -> num_heads * head_dim].
  TensorRef attn_k;    ///< [hidden -> kv_heads * head_dim].
  TensorRef attn_v;    ///< [hidden -> kv_heads * head_dim_v].
  TensorRef attn_out;  ///< [num_heads * head_dim -> hidden].

  // Mamba-2 layers. NVFP4 projections carry a per-tensor F32 `.scale`.
  TensorRef ssm_in;      ///< [hidden -> 2*inner + 2*groups*state + heads].
  TensorRef ssm_in_s;    ///< [1] F32, dequant scale of ssm_in.
  TensorRef ssm_conv;    ///< [conv_kernel, x + 2*groups*state] F32 (taps fast).
  TensorRef ssm_conv_b;  ///< [x + 2*groups*state] F32.
  TensorRef ssm_a;       ///< [1, heads] F32 (A_log based, pre-discretized).
  TensorRef ssm_d;       ///< [1, heads] F32 skip term.
  TensorRef ssm_dt_b;    ///< [heads] F32 dt bias.
  TensorRef ssm_norm;    ///< [heads_per_group * state, groups] F32.
  TensorRef ssm_out;     ///< [heads * head_dim -> hidden].
  TensorRef ssm_out_s;   ///< [1] F32, dequant scale of ssm_out.

  // Latent-MoE layers. Router width is the per-layer expert count.
  std::uint64_t n_experts{0};  ///< 240..496 in the Bolt artifact.
  TensorRef router;            ///< [hidden -> n_experts] F32.
  TensorRef exp_probs_b;       ///< [n_experts] F32 selection bias.
  TensorRef ffn_up_exps;       ///< [latent -> expert_ff] x n_experts.
  TensorRef ffn_down_exps;     ///< [expert_ff -> latent] x n_experts.
  TensorRef ffn_up_shexp;      ///< [hidden -> shared_ff], the 1 shared expert.
  TensorRef ffn_down_shexp;    ///< [shared_ff -> hidden].
  /// Latent bottleneck, stored output-first (unlike every other
  /// projection): up is [latent -> hidden] as stored, down the reverse.
  TensorRef ffn_latent_up;    ///< [latent -> hidden] as stored, Q8_0.
  TensorRef ffn_latent_down;  ///< [hidden -> latent] as stored, Q8_0.
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;   ///< [hidden -> vocab].
  TensorRef output;       ///< [hidden -> vocab]; falls back to token_embd.
  TensorRef output_norm;  ///< [hidden] F32, before the LM head.
  std::vector<LayerWeights> layers;
  /// Mapped GGUF shards backing the tensor offsets; descriptors are borrowed
  /// from the binding reader and stay valid for its lifetime.
  std::vector<core::GgufMappedRegion> regions;

  /// Binds and validates the trunk artifact against `Config::Kind`. Tensor
  /// payloads stay mapped and untouched; only headers are read.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::nemotron_h_moe

#endif  // GUFO_MODELS_NEMOTRON_H_MOE_WEIGHTS_HPP_
