#ifndef GUFO_MODELS_NEMOTRON_H_MOE_REFERENCE_HPP_
#define GUFO_MODELS_NEMOTRON_H_MOE_REFERENCE_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/nemotron_h_moe/weights.hpp"

namespace gufo::models::nemotron_h_moe {

/// Single-token, float32, scalar reference of the Nemotron 3.5 Bolt graph.
/// It pins the semantics every GPU runtime must reproduce: per-layer
/// pre-RMSNorm and residual, the Mamba-2 blocks (z|x|B|C|dt projection with
/// the NVFP4 `.scale`, causal depthwise conv over xBC with SiLU, the
/// scalar-decay recurrence dt = softplus(W x + b), D skip, y * silu(z),
/// grouped RMSNorm), the NoPE grouped-query attention blocks with causal
/// softmax and 1/sqrt(head_dim), and the latent-MoE blocks (sigmoid router
/// with selection bias, top-k weights normalized and scaled, ReLU^2 experts
/// with no gate in the latent space, shared expert outside the bottleneck).
/// Speed is irrelevant here; this is the numerical oracle the ROCm kernels
/// are validated against, and it is itself validated against llama.cpp.
class ReferenceModel {
public:
  /// `max_context` bounds the attention KV cache capacity; it may be below
  /// `Config::context_length` for short oracle runs and must never be 0.
  ReferenceModel(const ModelWeights& weights, std::uint32_t max_context);

  /// Advances one token. `logits` (vocab floats) receives the output
  /// distribution when non-empty; pass an empty span for cache-only warmup.
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits,
                          std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  void Reset();

  /// Parity hook: when set, every `Step` overwrites `dump` with the residual
  /// stream at each layer input (`num_layers` rows of `hidden_size` floats).
  void SetLayerDump(std::vector<float>* dump) noexcept { layer_dump_ = dump; }

private:
  struct MambaState {
    std::vector<float> conv;  ///< [kernel-1][xBC channels], oldest first.
    std::vector<float> ssm;   ///< [heads][head_dim][state_size].
  };
  struct AttnState {
    std::vector<float> k;  ///< [context][kv_heads * head_dim].
    std::vector<float> v;  ///< [context][kv_heads * head_dim_v].
  };

  void MambaLayer(const LayerWeights& l, MambaState& s,
                  std::span<const float> x, std::span<float> out);
  void AttnLayer(const LayerWeights& l, AttnState& s, std::uint32_t layer,
                 std::span<const float> x, std::span<float> out);
  void MoeLayer(const LayerWeights& l, std::uint32_t layer,
                std::span<const float> x, std::span<float> out);

  const ModelWeights& w_;
  const Config& c_;
  std::uint32_t max_context_;
  std::uint32_t position_{0};
  std::vector<float>* layer_dump_{nullptr};
  std::vector<MambaState> mamba_;  ///< Indexed by layer id.
  std::vector<AttnState> attn_;    ///< Indexed by layer id.
};

}  // namespace gufo::models::nemotron_h_moe

#endif  // GUFO_MODELS_NEMOTRON_H_MOE_REFERENCE_HPP_
