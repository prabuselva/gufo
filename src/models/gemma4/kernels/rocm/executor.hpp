#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/models/gemma4/config.hpp"
#include "src/models/gemma4/kernels/rocm/device_model.hpp"

namespace gufo::models::gemma4::rocm {

class Executor;

/// Per-sequence state: the F16 KV caches of every trunk layer (position-major
/// [position][kv_head][head_dim], sized to the session's own context) and the
/// trunk position. The forward scratch is shared across sessions in the
/// Executor (compute is serialized), so a session owns only the caches.
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  [[nodiscard]] std::uint32_t max_context() const noexcept {
    return max_context_;
  }
  /// Drops every token; the next forward starts at position 0. Stale cache
  /// rows are never read before being rewritten.
  void Reset() noexcept { position_ = 0; }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept {
    return allocated_bytes_;
  }

private:
  friend class Executor;
  Session(const Config& c, std::uint32_t max_context)
      : c_(c), max_context_(max_context) {}

  const Config& c_;
  std::uint32_t max_context_{0};
  std::uint32_t position_{0};
  // Position the last Verify() started from, so RollbackVerify() can rewind
  // the trunk to any committed prefix of the verified block.
  std::uint32_t verify_base_pos_{0};
  std::vector<__half*> k_cache_;
  std::vector<__half*> v_cache_;
  std::vector<void*> allocations_;
  std::size_t allocated_bytes_{0};
};

/// Runs the Gemma-4-26B-A4B trunk on the GPU for one session at a time.
/// Weights stay in their GGUF encoding and are decoded on the fly by the
/// GEMV/GEMM tiers; the fused operators handle everything else and the routed
/// F16 WMMA tier runs the prefill experts on the matrix cores. The forward
/// mirrors the scalar oracle in reference.cpp exactly, so a GPU step and a
/// CPU step agree to rounding.
///
/// The scratch buffers are sized once at `Create` (decode buffers plus
/// prefill_chunk_-row batch buffers); each session owns its KV caches.
class Executor {
public:
  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  /// Allocates the shared forward scratch and binds the uploaded weights.
  /// Fails if any tensor is in a format the GEMV tier cannot decode.
  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const DeviceModel& model, std::uint32_t prefill_chunk,
      std::string* error_msg = nullptr);

  /// Allocates one session's KV caches for up to `max_context` tokens. The
  /// session is returned ready for its first forward.
  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr) const;
  /// Device bytes one session allocates at `max_context` (both KV caches of
  /// every layer), so callers can bound the resident session count.
  [[nodiscard]] std::size_t SessionBytes(
      std::uint32_t max_context) const noexcept;

  /// Advances the trunk by one token; the result is in logits() and h_out().
  bool Step(Session& session, std::int32_t token,
            std::string* error_msg = nullptr);
  /// Advances the trunk by `count` prompt tokens in prefill-sized chunks,
  /// driving the batched GEMM tier, the fused prefill attention and the
  /// routed F16 experts. The KV caches and the session position advance
  /// exactly as under `count` Step() calls; logits() and h_out() hold the
  /// final token's result.
  bool Prefill(Session& session, const std::int32_t* tokens,
               std::uint32_t count, std::string* error_msg = nullptr);

  /// One image's trunk-width embeddings, already projected by the vision
  /// tower and left unscaled (Gemma replaces the sqrt(hidden)-scaled token
  /// embedding at these positions, it does not scale the image rows). `rows`
  /// is a device buffer of `row_count` rows of `hidden_size` floats; `offset`
  /// is the absolute session position of the first image row.
  struct VisionImage {
    const float* rows{nullptr};
    std::uint32_t offset{0};
    std::uint32_t row_count{0};
  };

  /// Stages the image embeddings the next `Prefill` splices into the residual
  /// stream. The rows stay on the device (uploaded by the caller); the
  /// executor only records the pointers and their absolute positions. Cleared
  /// at the end of the next `Prefill`, so it never leaks into `Step`/`Verify`.
  void SetVision(std::vector<VisionImage> images) {
    vision_ = std::move(images);
  }

[[nodiscard]] const float* logits() const noexcept { return logits_; }
  [[nodiscard]] const float* h_out() const noexcept { return h_out_; }

  /// Attaches the MTP draft and allocates its forward scratch. The draft reads
  /// the trunk KV cache read-only, so it shares this executor's sessions.
  /// Fails if any draft tensor is in a format the GEMV tier cannot decode.
  [[nodiscard]] bool AttachDraft(const DeviceDraft& draft,
                                 std::string* error_msg = nullptr);
  [[nodiscard]] bool has_draft() const noexcept { return draft_ != nullptr; }

  /// Runs one MTP draft step: consumes `token` (embedded through the trunk
  /// embedding) and the trunk-width hidden `h` (device), reads the trunk KV
  /// cache at positions [0, session.position()), and leaves the uncapped draft
  /// logits in draft_logits() and the next trunk-width hidden in
  /// draft_h_next(). Mirrors the scalar oracle `DraftStep` exactly.
  bool DraftStep(Session& session, std::int32_t token, const float* h,
                 std::string* error_msg = nullptr);

  [[nodiscard]] const float* draft_logits() const noexcept {
    return draft_logits_;
  }
  [[nodiscard]] const float* draft_h_next() const noexcept {
    return draft_h_next_;
  }

  /// Upper bound on the verified block (the trunk token plus its drafts). The
  /// batched output GEMV (`GemvRows`) covers 2..5 rows, so k + 1 <= 5.
  static constexpr std::uint32_t kMaxVerifyRows = 5U;

  /// Runs the trunk over the speculative block `[t0, drafts[0..k-1]]` (k + 1
  /// rows) at positions [position, position + k + 1) through the batched
  /// prefill tier, writing the trunk KV cache exactly as the matching Step()
  /// sequence would. Row `r` of verify_logits() ([k + 1][vocab], softcapped)
  /// is the trunk's prediction after consuming row `r`; row `r` of
  /// verify_hidden() ([k + 1][hidden]) is the post-output-norm hidden the next
  /// draft step conditions on. h_out() is left at row k so a full accept needs
  /// no extra work. The session position advances by k + 1; RollbackVerify()
  /// rewinds it to any committed prefix.
  bool Verify(Session& session, std::int32_t t0, const std::int32_t* drafts,
              std::uint32_t k, std::string* error_msg = nullptr);
  /// Rewinds the trunk to the first `keep` verified rows (1 <= keep <= k + 1):
  /// the position returns to the verify base plus `keep` and h_out() is set to
  /// row keep - 1, the hidden of the last committed token. Stale KV rows beyond
  /// are never read before being rewritten.
  void RollbackVerify(Session& session, std::uint32_t keep);

  [[nodiscard]] const float* verify_logits() const noexcept {
    return verify_logits_;
  }
  [[nodiscard]] const float* verify_hidden() const noexcept {
    return verify_hidden_;
  }

 private:
  Executor(const Config& c, const DeviceModel& model) : c_(c), model_(model) {}

  /// One decode layer; swaps cur_/other_ so cur_ holds the new residual.
  bool StepLayer(Session& session, std::uint32_t il, std::string* error_msg);
  /// One prefill layer over `tokens` rows starting at absolute position
  /// `start`; swaps pf_cur_/pf_next_ likewise.
  bool PrefillLayer(Session& session, std::uint32_t il, std::uint32_t start,
                    std::uint32_t tokens, std::string* error_msg);
  /// Routed F16 WMMA MoE over `tokens` rows of `x` ([tokens][hidden]) into
  /// `out`. Downloads the expert histogram once per layer to build the tile
  /// map on the host (the qwen36 route).
  bool MoeBatch(const DeviceLayer& l, const float* x, const std::int32_t* ids,
                const float* weights, float* out, std::uint32_t tokens,
                std::string* error_msg);

  float* AllocFloats(std::size_t n, std::string* error);
  void* AllocBytes(std::size_t bytes, std::string* error);

  const Config& c_;
  const DeviceModel& model_;

  // Decode (single-token) buffers. cur_/other_ ping-pong the residual stream.
  float* cur_{nullptr};
  float* other_{nullptr};
  float* normed_{nullptr};
  float* q_{nullptr};
  float* qn_{nullptr};
  float* k_raw_{nullptr};
  float* v_raw_{nullptr};
  float* attn_{nullptr};
  float* proj_{nullptr};
  float* attn_out_{nullptr};
  float* mlp_{nullptr};
  float* mlp1_{nullptr};
  float* moe_{nullptr};
  float* moe2_{nullptr};
  float* gate_{nullptr};
  float* up_{nullptr};
  float* act_{nullptr};
  float* gu_{nullptr};
  float* dn_{nullptr};
  float* router_in_{nullptr};
  float* router_logits_{nullptr};
  std::int32_t* ids_{nullptr};
  float* weights_{nullptr};
  float* attn_scratch_{nullptr};
  float* logits_{nullptr};
  float* h_out_{nullptr};

  // Prefill buffers, sized to prefill_chunk_ rows.
  std::uint32_t prefill_chunk_{0};
  float* pf_next_{nullptr};
  float* pf_normed_{nullptr};
  float* pf_q_{nullptr};
  float* pf_qn_{nullptr};
  float* pf_k_{nullptr};
  float* pf_v_{nullptr};
  float* pf_attn_{nullptr};
  float* pf_proj_{nullptr};
  float* pf_attn_out_{nullptr};
  float* pf_mlp_{nullptr};
  float* pf_mlp1_{nullptr};
  float* pf_moe_{nullptr};
  float* pf_moe2_{nullptr};
  float* pf_gate_{nullptr};
  float* pf_up_{nullptr};
  float* pf_act_{nullptr};
  float* pf_router_in_{nullptr};
  float* pf_router_logits_{nullptr};
  std::int32_t* pf_ids_{nullptr};
  float* pf_weights_{nullptr};
  std::uint32_t* pf_counts_{nullptr};
  std::int32_t* pf_pad_bounds_{nullptr};
  std::int32_t* pf_cursors_{nullptr};
  std::int32_t* pf_rows_token_{nullptr};
  std::int32_t* pf_rows_slot_{nullptr};
  std::int32_t* pf_tiles_dev_{nullptr};
  __half* pf_x_half_{nullptr};
  __half* pf_gu_half_{nullptr};
  __half* pf_act_half_{nullptr};
  float* pf_expert_out_{nullptr};
  /// Prefill residual ping-pong: pf_cur_ alternates with pf_next_.
  float* pf_cur_{nullptr};

  // MTP draft scratch (attached on demand). cur_/other_ ping-pong the draft
  // residual; the draft attention reuses attn_scratch_ (same head geometry).
  const DeviceDraft* draft_{nullptr};
  float* d_xh_{nullptr};
  float* d_cur_{nullptr};
  float* d_other_{nullptr};
  float* d_normed_{nullptr};
  float* d_q_{nullptr};
  float* d_qn_{nullptr};
  float* d_attn_{nullptr};
  float* d_proj_{nullptr};
  float* d_attn_out_{nullptr};
  float* d_gate_{nullptr};
  float* d_up_{nullptr};
  float* d_act_{nullptr};
  float* d_mlp1_{nullptr};
  float* draft_logits_{nullptr};
  float* draft_h_next_{nullptr};

  // Speculative verify scratch: per-row logits ([kMaxVerifyRows][vocab]) and
  // post-output-norm hidden ([kMaxVerifyRows][hidden]).
  float* verify_logits_{nullptr};
  float* verify_hidden_{nullptr};

  // Image embeddings staged for the next Prefill (see SetVision). Cleared at
  // the end of Prefill so they never affect Step or Verify.
  std::vector<VisionImage> vision_;

  std::vector<void*> allocations_;
};

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_EXECUTOR_HPP_