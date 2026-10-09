#ifndef GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/models/ling3_flash/config.hpp"
#include "src/models/ling3_flash/kernels/rocm/device_model.hpp"

namespace gufo::models::ling3_flash::rocm {

class Executor;

/// Per-sequence state on the device: the KDA recurrent state and conv history,
/// the absorbed-MLA latent/rope caches, the MTP draft's own MLA cache, and the
/// trunk/draft positions. The forward scratch is shared across sessions in the
/// Executor (the scheduler serializes compute through one worker), so a
/// session owns only the buffers that must persist between forward calls.
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  [[nodiscard]] std::uint32_t max_context() const noexcept {
    return max_context_;
  }
  /// Drops every token and zeroes the recurrent state and caches; the next
  /// forward starts at position 0.
  void Reset();

private:
  friend class Executor;
  Session(const Config& c, bool has_mtp) : c_(c), has_mtp_(has_mtp) {}

  float* AllocFloats(std::size_t n);

  const Config& c_;
  const bool has_mtp_;
  std::uint32_t max_context_{0};
  std::uint32_t position_{0};
  std::uint32_t mtp_position_{0};

  // Recurrent state, indexed by layer (null where the layer does not use it).
  // KDA: conv history [3][kern-1][inner] oldest-first plus state
  // [heads][head_dim][head_dim]. MLA: latent cache [position][kv_lora_rank]
  // and the rotated rope key cache [position][rotary_dim].
  std::vector<float*> kda_conv_;
  std::vector<float*> kda_state_;
  std::vector<float*> mla_kv_;
  std::vector<float*> mla_kpe_;
  float* mtp_kv_{nullptr};
  float* mtp_kpe_{nullptr};

  std::vector<void*> allocations_;
  std::size_t allocated_bytes_{0};
  bool alloc_failed_{false};
};

/// Runs the trunk (and the MTP draft block) of the Ling 3.0 Flash model on the
/// GPU for one session at a time. Weights stay in their GGUF encoding and are
/// decoded on the fly by the GEMV tier; the fused operators handle everything
/// else and the split-KV kernels run the absorbed-MLA decode. The forward pass
/// mirrors the scalar oracle in reference.cpp exactly, so a GPU step and a CPU
/// step agree to rounding.
///
/// The scratch buffers are sized once for the model's maximum context and
/// shared by every session (compute is serialized), while each session owns
/// its recurrent state and caches. `Create` allocates the shared scratch;
/// `CreateSession` allocates one session's state at its own context length.
/// Prefill currently drives the same single-token path as decode; a batched
/// prefill tier is a later optimization, not a correctness requirement.
class Executor {
public:
  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  /// Allocates the shared forward scratch for up to `max_context` tokens and
  /// binds the uploaded weights. Fails if any tensor is in a format the GEMV
  /// tier cannot decode. Per-session state is allocated separately by
  /// `CreateSession`.
  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const DeviceModel& model, std::uint32_t max_context,
      std::string* error_msg = nullptr);

  /// Allocates one session's recurrent state and caches for up to
  /// `max_context` tokens (bounded by the scratch `Create` was given). The
  /// session is returned zeroed and ready for its first forward.
  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr) const;
  /// Device bytes one session allocates at `max_context` (KDA state and conv
  /// history, MLA caches and the MTP caches). Matches the allocations
  /// `CreateSession` makes, so callers can bound the resident session count
  /// against free VRAM.
  [[nodiscard]] std::size_t SessionBytes(
      std::uint32_t max_context) const noexcept;
  /// Scratch the Executor grows lazily after `Create`. All Ling 3.0 Flash
  /// scratch is allocated eagerly, so this is zero; it exists so the serve
  /// resource claim can reserve the same shape as other models.
  [[nodiscard]] std::size_t DeferredScratchBytes() const noexcept { return 0; }

  /// Advances the trunk by one token; the result is in logits() and h_out().
  bool Step(Session& session, std::int32_t token,
            std::string* error_msg = nullptr);
  /// Advances the trunk by `count` prompt tokens. Mirrors `count` Step()
  /// calls; `logits()` and `h_out()` hold the final token's result.
  bool Prefill(Session& session, const std::int32_t* tokens,
               std::uint32_t count, std::string* error_msg = nullptr);
  /// Runs the MTP draft block for `token` using the trunk hidden state from
  /// the most recent Step; the result is in mtp_logits().
  bool MtpStep(Session& session, std::int32_t token,
               std::string* error_msg = nullptr);

  [[nodiscard]] const float* logits() const noexcept { return logits_; }
  [[nodiscard]] const float* mtp_logits() const noexcept { return mtp_logits_; }
  [[nodiscard]] const float* h_out() const noexcept { return h_out_; }

private:
  Executor(const Config& c, const DeviceModel& model) : c_(c), model_(model) {}

  void KdaLayer(Session& session, std::uint32_t il, const DeviceLayer& l,
                const float* x, float* out);
  /// One absorbed-MLA decode step against the given caches at `pos`.
  void MlaLayer(const DeviceLayer& l, float* kv_cache, float* kpe_cache,
                std::uint32_t pos, const float* x, float* out);
  /// Dense FFN or group-routed MoE + shared expert for one row, following the
  /// layer's clamps.
  void Ffn(const DeviceLayer& l, std::uint32_t il, const float* x, float* out);

  float* AllocFloats(std::size_t n, std::string* error);
  std::int32_t* AllocInts(std::size_t n, std::string* error);

  const Config& c_;
  const DeviceModel& model_;

  // Residual stream and outputs.
  float* x_{nullptr};
  float* normed_{nullptr};
  float* attn_{nullptr};
  float* ffn_{nullptr};
  float* ffn_inp_{nullptr};
  float* h_out_{nullptr};
  float* logits_{nullptr};
  float* mtp_logits_{nullptr};

  // KDA scratch (reused across KDA layers).
  float* kda_q_{nullptr};
  float* kda_k_{nullptr};
  float* kda_v_{nullptr};
  float* kda_f_{nullptr};
  float* kda_g_{nullptr};
  float* kda_beta_{nullptr};
  float* kda_conv_q_{nullptr};
  float* kda_conv_k_{nullptr};
  float* kda_conv_v_{nullptr};
  float* kda_attn_{nullptr};

  // MLA scratch (reused across MLA layers).
  float* mla_q_{nullptr};       ///< [heads * (nope + rope)] raw query.
  float* mla_kv_a_{nullptr};    ///< [kv_rank + rope] latent + rope key row.
  float* mla_q_abs_{nullptr};   ///< [heads * (kv_rank + rope)] absorbed query.
  float* mla_latent_{nullptr};  ///< [heads * kv_rank] attention output latent.
  float* mla_head_out_{nullptr};  ///< [heads * value_dim] gated head outputs.
  float* mla_gate_{nullptr};      ///< [heads] sigmoid gate.
  float* mla_ws_{nullptr};        ///< split-KV partial workspace.

  // Mixture-of-experts / dense FFN scratch (reused across layers).
  float* moe_logits_{nullptr};
  std::int32_t* moe_ids_{nullptr};
  float* moe_weights_{nullptr};
  float* moe_gate_{nullptr};
  float* moe_up_{nullptr};
  float* moe_inter_{nullptr};
  float* moe_down_{nullptr};
  float* sh_gate_{nullptr};
  float* sh_up_{nullptr};
  float* sh_inter_{nullptr};
  float* sh_down_{nullptr};

  // MTP scratch.
  float* mtp_e_{nullptr};
  float* mtp_h_{nullptr};
  float* mtp_cat_{nullptr};
  float* mtp_x_{nullptr};

  std::uint32_t scratch_context_{0};
  std::vector<void*> allocations_;
};

}  // namespace gufo::models::ling3_flash::rocm

#endif  // GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_