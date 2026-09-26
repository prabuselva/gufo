#ifndef GUFO_MODELS_QWEN36_A3B_REFERENCE_HPP_
#define GUFO_MODELS_QWEN36_A3B_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen36_a3b/weights.hpp"

namespace gufo::models::qwen36_a3b {

/// Single-token, float32, scalar reference of the Qwen3.6-35B-A3B graph. It
/// pins the semantics every GPU runtime must reproduce: Gated DeltaNet linear
/// attention (causal depthwise convolution, per-head L2-normalised delta rule
/// and the SiLU-gated output norm), gated grouped-query full attention with
/// per-head Q/K RMSNorm and partial rotary, the softmax top-k
/// Mixture-of-Experts FFN with its sigmoid-gated shared expert, and the
/// self-speculative MTP block. Speed is irrelevant here; this is the numerical
/// oracle the ROCm kernels are validated against, and it is itself validated
/// against llama.cpp logits.
class ReferenceModel {
public:
  ReferenceModel(const ModelWeights& weights, std::uint32_t max_context);

  /// Runs one token at the next position. `logits` (vocab floats) receives the
  /// output distribution when non-empty. `h_out` (hidden floats) receives the
  /// post-`output_norm` trunk stream, the MTP block's hidden input.
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits,
                          std::span<float> h_out = {},
                          std::string* error_msg = nullptr);

  /// Independent scalar MTP draft. `hidden` is the trunk stream from `Step`'s
  /// `h_out`; `token` is the shifted token embedding. The draft shares the
  /// trunk vocabulary and LM head.
  [[nodiscard]] bool MtpStep(const MtpWeights& mtp, std::int32_t token,
                             std::span<const float> hidden,
                             std::span<float> logits,
                             std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  void Reset();

private:
  struct LinearState {
    std::vector<float> conv;   ///< [kernel-1][channels], oldest first
    std::vector<float> state;  ///< [v_heads][head_dim(v)][head_dim(k)]
  };
  struct AttentionState {
    std::vector<float> k;  ///< [pos][kv_heads][head_dim], rotated
    std::vector<float> v;  ///< [pos][kv_heads][head_dim]
  };

  void LinearAttention(const LayerWeights& l, LinearState& s,
                       std::span<const float> x, std::span<float> out);
  void Attention(const LayerWeights& l, AttentionState& s,
                 std::span<const float> x, std::uint32_t pos,
                 std::span<float> out);
  void Moe(const LayerWeights& l, std::span<const float> x,
           std::span<float> out);

  const ModelWeights& w_;
  const Config& c_;
  std::uint32_t max_context_;
  std::uint32_t position_{0};
  std::vector<LinearState> linear_;
  std::vector<AttentionState> attention_;
  AttentionState mtp_attention_;
  std::uint32_t mtp_position_{0};
};

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_REFERENCE_HPP_