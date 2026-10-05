#ifndef GUFO_MODELS_GEMMA4_REFERENCE_HPP_
#define GUFO_MODELS_GEMMA4_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/gemma4/weights.hpp"

namespace gufo::models::gemma4 {

/// Single-token, float32, scalar reference of the Gemma-4-26B-A4B trunk. It
/// pins the semantics every GPU runtime must reproduce: embedding scaled by
/// sqrt(hidden), sliding-window/full hybrid attention with per-head Q/K
/// RMSNorm and NEOX rope (proportional factors on full layers, weightless V
/// normalization), the shared dense FFN plus the 128-expert top-8 softmax MoE
/// with renormalized weights and per-expert down scales, the per-layer output
/// scalars, and the final softcapped LM head. Speed is irrelevant here; this
/// is the numerical oracle the ROCm kernels are validated against, and it is
/// itself validated against llama.cpp logits.
class ReferenceModel {
public:
  ReferenceModel(const ModelWeights& weights, std::uint32_t max_context);

  /// Runs one token at the next position. `logits` (vocab floats) receives the
  /// softcapped output distribution when non-empty. `h_out` (hidden floats)
  /// receives the post-`output_norm` trunk stream, the MTP draft's hidden
  /// input.
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits,
                          std::span<float> h_out = {},
                          std::string* error_msg = nullptr);

  /// Independent scalar MTP draft step. `h` is the trunk stream from `Step`'s
  /// `h_out` and `token` the target token embedding input; the draft reads the
  /// trunk KV cache written so far and produces uncapped `logits` plus the
  /// `h_next` stream (trunk width) for the next draft step.
  [[nodiscard]] bool DraftStep(const DraftWeights& draft, std::int32_t token,
                               std::span<const float> h,
                               std::span<float> logits, std::span<float> h_next,
                               std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  void Reset();

private:
  struct AttentionState {
    std::vector<float> k;  ///< [pos][kv_heads][head_dim], rotated
    std::vector<float> v;  ///< [pos][kv_heads][head_dim]
  };

  /// Causal GQA over one layer's cache. `window` > 0 restricts attention to
  /// the last `window` positions (sliding-window layers).
  void Attention(const AttentionState& s, std::span<const float> q,
                 std::uint32_t num_heads, std::uint32_t num_kv_heads,
                 std::uint32_t head_dim, std::uint32_t window,
                 std::span<float> out);

  const ModelWeights& w_;
  const Config& c_;
  std::uint32_t max_context_;
  std::uint32_t position_{0};
  std::vector<AttentionState> attention_;
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_REFERENCE_HPP_