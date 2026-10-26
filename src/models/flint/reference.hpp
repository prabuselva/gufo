#ifndef GUFO_MODELS_FLINT_REFERENCE_HPP_
#define GUFO_MODELS_FLINT_REFERENCE_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/flint/weights.hpp"

namespace gufo::models::flint {

/// Single-token, float32, scalar reference of the Flint Preview graph. It
/// pins the semantics every GPU runtime must reproduce: grouped-query
/// attention with per-head RMS QK norms (LLM_NORM_RMS, applied before the
/// per-layer gated NEOX rotary), softmax over the causal keys plus the head's
/// attention sink, the trailing `sliding_window` read mask on windowed
/// layers, the SiLU-gated dense FFN in the leading layers, and the sigmoid
/// top-k MoE with its summed-and-clamped weight normalisation plus the single
/// shared expert added once. Speed is irrelevant here; this is the numerical
/// oracle the ROCm kernels are validated against, and it is itself validated
/// against llama.cpp logits.
class ReferenceModel {
public:
  ReferenceModel(const ModelWeights& weights, std::uint32_t max_context);

  /// Runs one token at the next position; `logits` (vocab floats) receives
  /// the output distribution. Fails when the cache is exhausted.
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits,
                          std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  void Reset();

private:
  struct AttentionState {
    std::vector<float> k;  ///< [pos][kv_heads][head_dim], post-rope
    std::vector<float> v;  ///< [pos][kv_heads][head_dim]
  };

  void Attention(const LayerWeights& l, AttentionState& s, std::uint32_t layer,
                 std::uint32_t pos, std::span<const float> normed,
                 std::span<float> out);
  void Ffn(const LayerWeights& l, std::span<const float> normed,
           std::span<float> out);

  const ModelWeights& w_;
  const Config& c_;
  std::uint32_t max_context_;
  std::uint32_t position_{0};
  std::vector<AttentionState> layers_;
};

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_REFERENCE_HPP_
