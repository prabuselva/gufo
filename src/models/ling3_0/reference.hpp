#ifndef GUFO_MODELS_LING3_FLASH_REFERENCE_HPP_
#define GUFO_MODELS_LING3_FLASH_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/ling3_0/weights.hpp"

namespace gufo::models::ling3_0 {

/// Single-token, float32, scalar reference of the Ling 3.0 Flash graph. It
/// pins the semantics every GPU runtime must reproduce: Kimi Delta Attention
/// (causal depthwise convolution with SiLU, per-head L2-normalised delta rule
/// with a per-channel decay gate and the SiLU-gated output norm), absorbed
/// Multi-head Latent Attention with the shared rope key and per-head sigmoid
/// gate, the sigmoid-gated Mixture-ofExperts FFN with group-limited top-k
/// routing, SwiGLU clamps and one always-on shared expert, and the
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
  struct KdaState {
    std::vector<float> conv;   ///< [3][kernel-1][inner] q,k,v; oldest first
    std::vector<float> state;  ///< [heads][head_dim][head_dim]
  };
  struct MlaState {
    std::vector<float> kv;   ///< [pos][kv_lora_rank]
    std::vector<float> kpe;  ///< [pos][rotary_dim], rotated
  };

  void KdaAttention(const LayerWeights& l, KdaState& s,
                    std::span<const float> x, std::span<float> out);
  void MlaAttention(const LayerWeights& l, MlaState& s,
                    std::span<const float> x, std::uint32_t pos,
                    std::span<float> out);
  void DenseFfn(const LayerWeights& l, float limit, std::span<const float> x,
                std::span<float> out);
  void MoeFfn(const LayerWeights& l, float limit_exp, float limit_shexp,
              std::span<const float> x, std::span<float> out);

  const ModelWeights& w_;
  const Config& c_;
  std::uint32_t max_context_;
  std::uint32_t position_{0};
  std::vector<KdaState> kda_;
  std::vector<MlaState> mla_;
  MlaState mtp_state_;
  std::uint32_t mtp_position_{0};
};

}  // namespace gufo::models::ling3_0

#endif  // GUFO_MODELS_LING3_FLASH_REFERENCE_HPP_