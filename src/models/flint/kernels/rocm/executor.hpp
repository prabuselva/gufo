#ifndef GUFO_MODELS_FLINT_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_FLINT_KERNELS_ROCM_EXECUTOR_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/models/flint/weights.hpp"

namespace gufo::core {
class GgufReader;
}  // namespace gufo::core

namespace gufo::models::flint {

/// Single-resident-model ROCm driver for Flint-40B-A2B. Uploads every
/// bound weight once (native encoding, one device buffer per tensor), keeps
/// an f16 K/V cache per layer and runs the trunk; positions advance with
/// each forwarded block, mirroring the incremental scalar reference.
class FlintExecutor {
public:
  ~FlintExecutor();
  FlintExecutor(const FlintExecutor&) = delete;
  FlintExecutor& operator=(const FlintExecutor&) = delete;

  /// `batch_capacity` sizes the activation buffers and `kv_capacity` the
  /// attention cache (both must cover the largest planned forward).
  [[nodiscard]] static std::unique_ptr<FlintExecutor> Create(
      const core::GgufReader& reader, const ModelWeights& weights,
      std::uint32_t batch_capacity, std::uint32_t kv_capacity,
      std::string* error);

  /// Runs `count` tokens (<= batch capacity) from the current position and
  /// writes the next-token logits for the final token into `logits`
  /// (sized to the vocabulary, resized if needed). Blocking.
  [[nodiscard]] bool Forward(const std::int32_t* tokens, std::uint32_t count,
                             std::vector<float>* logits, std::string* error);

  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  void Reset() noexcept { position_ = 0; }

  /// Device bytes held by this executor (weights, activations, KV caches),
  /// accumulated while `Create` allocates.
  [[nodiscard]] std::size_t device_bytes() const noexcept;

  /// Debug aid: record the MoE intermediates of one chosen layer during the
  /// next Forward (router input, sigmoid scores, chosen ids/weights, routed
  /// outputs and the shared-expert stream) for host-side comparison.
  struct MoETrace {
    std::vector<float> normed;  ///< Router input row
    std::vector<float> probs;   ///< Sigmoid scores [num_experts]
    std::vector<std::int32_t> ids;
    std::vector<float> wts;
    std::vector<float> routed;  ///< down outputs per slot [used*hidden]
    std::vector<float> shared;  ///< shared-expert down output [hidden]
  };
  void EnableMoeTrace(int layer) noexcept;
  [[nodiscard]] const MoETrace& LastMoeTrace() const noexcept;

  /// Debug aid: after a single-token Forward, capture the hidden row per
  /// layer (and the post-embedding row first) so a run can be diffed layer
  /// by layer against the scalar oracle.
  void EnableLayerCapture(bool enable) noexcept;
  [[nodiscard]] const std::vector<std::vector<float>>& CapturedHidden() const;

private:
  FlintExecutor() = default;
  struct State;
  State* state_{nullptr};
  std::uint32_t position_{0};
};

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_KERNELS_ROCM_EXECUTOR_HPP_
