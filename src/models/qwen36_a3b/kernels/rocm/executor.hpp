#ifndef GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/models/qwen/vision/device_input.hpp"
#include "src/models/qwen36_a3b/config.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/device_model.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/profile.hpp"

namespace gufo::models::qwen36_a3b::rocm {

class Executor;

/// Per-sequence state on the device: the Gated DeltaNet recurrent state and
/// conv history, the full-attention KV caches (and their FP16 mirrors), the
/// MTP draft KV caches, and the trunk/draft positions. The forward scratch is
/// shared across sessions in the Executor (the scheduler serializes compute
/// through one worker), so a session owns only the buffers that must persist
/// between forward calls. A speculative session additionally keeps per-row
/// snapshots of the recurrent state and conv history so a rejected draft can
/// rewind to any committed prefix.
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  [[nodiscard]] std::uint32_t max_context() const noexcept {
    return max_context_;
  }
  [[nodiscard]] bool mtp_enabled() const noexcept { return mtp_enabled_; }
  /// Enables building the MTP draft cache during prefill and using speculative
  /// decoding. Off by default: a plain prefill never touches the draft block,
  /// so it skips the draft's attention entirely. Callers that will decode with
  /// MTP (serve, speculative bench) enable this before the first Prefill; the
  /// flag must not change mid-session because the draft cache is filled during
  /// prefill and cannot be rebuilt lazily afterwards.
  void SetMtpEnabled(bool enabled) noexcept { mtp_enabled_ = enabled; }
  /// Drops every token and zeroes the recurrent state and caches; the next
  /// forward starts at position 0.
  void Reset();
  /// Installs the vision prompt (image embeddings + MRoPE layout) so the next
  /// Prefill injects the projected image rows and every attention uses the
  /// vision rope positions. A null prompt clears the vision state (text-only).
  void ConfigureVision(std::shared_ptr<const qwen::vision::Prompt> prompt,
                       std::shared_ptr<qwen::vision::Encoder> encoder,
                       hipStream_t stream);
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept {
    return allocated_bytes_ + vision_input_.Bytes();
  }

private:
  friend class Executor;
  Session(const Config& c, bool has_mtp) : c_(c), has_mtp_(has_mtp) {}

  float* AllocFloats(std::size_t n);
  void* AllocBytes(std::size_t bytes);

  const Config& c_;
  const bool has_mtp_;
  std::uint32_t max_context_{0};
  std::uint32_t position_{0};
  std::uint32_t mtp_position_{0};
  /// Trunk position before the most recent Verify, the rewind base.
  std::uint32_t verify_base_pos_{0};
  bool mtp_enabled_{false};

  // Recurrent state, indexed by layer (null where the layer does not use it).
  std::vector<float*> gdn_state_;
  std::vector<float*> gdn_history_;
  std::vector<float*> k_cache_;
  std::vector<float*> v_cache_;
  // FP16 mirror of the full-attention KV cache, [position][kv_head][head_dim],
  // consumed by the WMMA prefill kernel. The FP32 planes above stay the source
  // of truth for decode and the scalar oracle; these are written alongside them
  // for every prefill chunk.
  std::vector<void*> k_cache_f16_;
  std::vector<void*> v_cache_f16_;
  // Per-linear-layer verify snapshots: the recurrent state and conv history
  // after each of the first rows of the most recent Verify (null without an
  // MTP block).
  std::vector<float*> gdn_state_snap_;
  std::vector<float*> gdn_hist_snap_;
  float* mtp_k_cache_{nullptr};
  float* mtp_v_cache_{nullptr};
  // Half-precision mirrors of the draft KV caches, consumed by the WMMA
  // prefill attention kernel (same role as k_cache_f16_/v_cache_f16_ for the
  // trunk). Without them the draft falls back to the scalar oracle.
  void* mtp_k_cache_f16_{nullptr};
  void* mtp_v_cache_f16_{nullptr};
  // The last prefill chunk's hidden row, the draft block's hidden input for
  // the next chunk's first row.
  float* mtp_prev_hidden_{nullptr};

  // Per-session image inputs: the projected vision embeddings and the MRoPE
  // position descriptor shared by every attention in the forward. Empty for
  // text-only prompts (rope() is null, attention falls back to physical).
  qwen::vision::DeviceInput vision_input_;

  std::vector<void*> allocations_;
  std::size_t allocated_bytes_{0};
  bool alloc_failed_{false};
};

/// Runs the trunk (and the MTP draft block) of the Qwen3.6-35B-A3B model on
/// the GPU for one session at a time. Weights stay in their GGUF encoding and
/// are decoded on the fly by the GEMV tier; the fused operators handle
/// everything else. The forward pass mirrors the scalar oracle in
/// reference.cpp exactly, so a GPU step and a CPU step agree to rounding.
///
/// The scratch buffers are sized once for the model's maximum context and
/// shared by every session (compute is serialized), while each session owns
/// its recurrent state and caches. `Create` allocates the shared scratch;
/// `CreateSession` allocates one session's state at its own context length.
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
      std::string* error_msg = nullptr, std::uint32_t attn_window = 0,
      std::uint32_t attn_sink = 0);

  /// Allocates one session's recurrent state and caches for up to
  /// `max_context` tokens (bounded by the scratch `Create` was given). The
  /// session is returned zeroed and ready for its first forward.
  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr) const;
  /// Device bytes one session allocates at `max_context` (recurrent state, KV
  /// caches and their FP16 mirrors, MTP caches and verify snapshots). Matches
  /// the allocations `CreateSession` makes, so callers can bound the resident
  /// session count against free VRAM.
  [[nodiscard]] std::size_t SessionBytes(
      std::uint32_t max_context) const noexcept;
  /// Scratch the Executor grows lazily after `Create`. All Qwen3.6-35B-A3B
  /// scratch is allocated eagerly, so this is zero; it exists so the serve
  /// resource claim can reserve the same shape as other models.
  [[nodiscard]] std::size_t DeferredScratchBytes() const noexcept { return 0; }

  /// Advances the trunk by one token; the result is in logits() and h_out().
  bool Step(Session& session, std::int32_t token,
            std::string* error_msg = nullptr);
  /// Advances the trunk by `count` prompt tokens in prefill-sized chunks,
  /// driving the batched GEMM tier and the fused prefill operators so the whole
  /// prompt is consumed without a per-token host round-trip. Recurrent state
  /// (Gated DeltaNet state/history and the KV caches) and the session position
  /// advance exactly as they would have under `count` Step() calls, so decoding
  /// resumes from the next position. `logits()` and `h_out()` hold the final
  /// token's result.
  bool Prefill(Session& session, const std::int32_t* tokens,
               std::uint32_t count, std::string* error_msg = nullptr);
  /// Runs the MTP draft block for `token` using the trunk hidden state from
  /// the most recent Step; the result is in mtp_logits().
  bool MtpStep(Session& session, std::int32_t token,
               std::string* error_msg = nullptr);
  /// Advances the MTP cache by `token` against the given device-side trunk
  /// hidden state, skipping the shared head and LM head. Used to keep the
  /// draft cache aligned with the trunk after an accepted draft.
  bool MtpAdvance(Session& session, std::int32_t token, const float* hidden,
                  std::string* error_msg = nullptr);
  /// Chains one more MTP draft: feeds `token` back through the draft block
  /// against the previous draft's own hidden state, producing new
  /// mtp_logits(). The draft KV cache advances with each call.
  bool MtpDraft(Session& session, std::int32_t token,
                std::string* error_msg = nullptr);
  /// Upper bound on the rows a Verify pass may run (kMaxDraftTokens + 1).
  static constexpr std::uint32_t kMaxVerifyRows = 5U;
  /// Speculative verification: advances the trunk by `k` + 1 tokens (the
  /// accepted token `t0` followed by the `k` drafts) in one batched pass
  /// (the GEMM tier, like Prefill). Row `r` of verify_logits()
  /// ([k+1][vocab]) holds the logits after consuming row token `r`, and row
  /// `r` of verify_hidden() ([k+1][hidden]) the post-output-norm hidden
  /// state. The Gated DeltaNet state and conv history after each of the
  /// first `k` rows are snapshotted so RollbackVerify() can commit any
  /// prefix of the block.
  bool Verify(Session& session, std::int32_t t0, const std::int32_t* drafts,
              std::uint32_t k, std::string* error_msg = nullptr);
  /// Commits the first `keep` rows of the most recent Verify (1 <= keep <= k)
  /// and discards the rest: restores the recurrent state from the snapshot
  /// after row `keep` - 1, rewinds the session position and the MTP cache to
  /// just past the committed rows and leaves their hidden state in h_out().
  void RollbackVerify(Session& session, std::uint32_t keep);

  [[nodiscard]] const float* logits() const noexcept { return logits_; }
  [[nodiscard]] const float* mtp_logits() const noexcept { return mtp_logits_; }
  [[nodiscard]] const float* verify_logits() const noexcept {
    return verify_logits_;
  }
  [[nodiscard]] const float* verify_hidden() const noexcept {
    return verify_h_;
  }
  [[nodiscard]] const float* h_out() const noexcept { return h_out_; }

private:
  Executor(const Config& c, const DeviceModel& model) : c_(c), model_(model) {}

  void LinearAttention(Session& session, const DeviceLayer& l, std::uint32_t il,
                       const float* x, float* out);
  void Attention(const DeviceLayer& l, const float* x, std::uint32_t pos,
                 float* out, float* k_cache, float* v_cache, void* k_cache_f16,
                 void* v_cache_f16, const std::uint32_t* pos_dev,
                 const qwen::vision::DeviceRope* rope = nullptr);
  void Moe(const DeviceLayer& l, const float* x, float* out);
  /// Batched (prefill) Mixture-of-Experts over `tokens` rows of `x`
  /// ([tokens][hidden]) into `out` ([tokens][hidden]). Mirrors Moe() but drives
  /// the GEMM tier and the batched router/epilogue kernels, so the routed
  /// expert ids never round-trip through the host.
  void MoeBatch(const DeviceLayer& l, const float* x, float* out,
                std::uint32_t tokens);

  /// Batched (prefill) Gated DeltaNet over `tokens` rows of `x`
  /// ([tokens][hidden]) into `out` ([tokens][hidden]), advancing the session's
  /// recurrent state and conv history past the chunk. Mirrors
  /// LinearAttention(). When `state_snap` is non-null it receives the recurrent
  /// state after each of the first `tokens - 1` rows (see GdnDeltaLoop);
  /// `hist_snap` receives the conv history after the first row.
  void LinearAttentionBatch(Session& session, const DeviceLayer& l,
                            std::uint32_t il, const float* x, float* out,
                            std::uint32_t tokens, float* state_snap = nullptr,
                            float* hist_snap = nullptr);
  /// Batched MTP cache fill over one prefill chunk. `hidden_rows`
  /// ([rows][hidden]) is the trunk's post-output-norm hidden state for the
  /// chunk; the block's own hidden input for row 0 comes from
  /// `session.mtp_prev_hidden_`. Advances the MTP KV caches and the session's
  /// MTP position.
  void MtpPrefillChunk(Session& session, const std::int32_t* tokens,
                       std::uint32_t rows, std::uint32_t start,
                       const float* hidden_rows);
  /// The MTP block forward. `hidden` is the device-side trunk (or previous
  /// draft) hidden state; with `with_logits` the shared head and LM head run
  /// into mtp_logits_.
  bool MtpForward(Session& session, std::int32_t token, const float* hidden,
                  bool with_logits, std::string* error_msg);
  /// Batched (prefill) gated grouped-query attention over `tokens` rows of `x`
  /// starting at absolute position `start`, writing the chunk's keys/values
  /// into the caches and reading them back causally. Mirrors Attention().
  void AttentionBatch(const DeviceLayer& l, const float* x, std::uint32_t start,
                      float* out, float* k_cache, float* v_cache,
                      void* k_cache_f16, void* v_cache_f16,
                      const std::uint32_t* pos_dev, std::uint32_t tokens,
                      const qwen::vision::DeviceRope* rope = nullptr);

  float* AllocFloats(std::size_t n, std::string* error);
  std::int32_t* AllocInts(std::size_t n, std::string* error);
  std::uint32_t* AllocUints(std::size_t n, std::string* error);
  void* AllocBytes(std::size_t bytes, std::string* error);

  const Config& c_;
  const DeviceModel& model_;

  // Residual stream and outputs.
  float* x_{nullptr};
  float* normed_{nullptr};
  float* attn_{nullptr};
  float* ffn_{nullptr};
  float* h_out_{nullptr};
  float* logits_{nullptr};
  float* mtp_logits_{nullptr};
  std::uint32_t* pos_dev_{nullptr};
  std::uint32_t* mtp_pos_dev_{nullptr};
  std::uint32_t pos_host_{0};
  std::uint32_t mtp_pos_host_{0};

  // Gated DeltaNet scratch (reused across linear layers).
  float* gdn_qkv_{nullptr};
  float* gdn_z_{nullptr};
  float* gdn_alpha_{nullptr};
  float* gdn_beta_{nullptr};
  float* gdn_convolved_{nullptr};
  float* gdn_qn_{nullptr};
  float* gdn_kn_{nullptr};
  float* gdn_attn_{nullptr};

  // Gated grouped-query attention scratch (reused across full layers).
  float* gqa_qg_{nullptr};
  float* gqa_k_{nullptr};
  float* gqa_v_{nullptr};
  float* gqa_q_{nullptr};
  float* gqa_gate_{nullptr};
  float* gqa_ctx_{nullptr};
  float* gqa_scratch_{nullptr};
  float* gqa_part_{nullptr};

  // Mixture-of-experts scratch (reused across layers).
  float* moe_logits_{nullptr};
  std::int32_t* moe_ids_{nullptr};
  float* moe_weights_{nullptr};
  float* moe_expert_out_{nullptr};
  float* moe_gate_{nullptr};
  float* moe_up_{nullptr};
  float* moe_shared_down_{nullptr};
  float* moe_shared_gate_{nullptr};

  // Upper bound on any session's context this scratch supports: the context
  // `Create` was given. `gqa_scratch_` is sized to num_heads * scratch_context_
  // so a decode at any position below it fits, and `CreateSession` rejects a
  // session context above it. Per-session state is sized to each session's own
  // (possibly divided) context, never to this bound.
  std::uint32_t scratch_context_{0};
  // Batched prefill scratch, sized to prefill_chunk_ tokens. The decode path
  // reuses the single-token buffers above; prefill drives these and never
  // syncs to the host. The MoE buffers hold one row per (token, slot) pair, so
  // they scale with prefill_chunk_ * num_experts_used.
  std::uint32_t prefill_chunk_{0};
  // Opt-in sliding-window + attention-sink prefill sparsity. Both default to 0,
  // which selects the exact dense WMMA kernel; a positive window restricts each
  // prefill query to the last `attn_window_` keys plus the first `attn_sink_`.
  std::uint32_t attn_window_{0};
  std::uint32_t attn_sink_{0};
  float* pf_router_logits_{nullptr};
  std::int32_t* pf_ids_{nullptr};
  std::uint32_t* pf_expert_counts_{nullptr};
  std::vector<std::uint32_t> pf_counts_host_;
  float* pf_weights_{nullptr};
  float* pf_gate_{nullptr};
  float* pf_up_{nullptr};
  float* pf_expert_out_{nullptr};
  float* pf_shared_gate_{nullptr};
  float* pf_shared_up_{nullptr};
  float* pf_shared_down_{nullptr};
  float* pf_shared_gate_inp_{nullptr};
  // [kMaxVerifyRows*used][hidden] copy of the verify normed rows (row t
  // repeated `used` times) so the grouped expert GEMVs can index x by slot.
  float* pf_x_dup_{nullptr};
  // [kMaxVerifyRows][heads][32][head_dim + 2] flash-decoding partials for
  // the verify rows (AttentionDecodeRows).
  float* pf_part2_{nullptr};

  // Routed F16 WMMA MoE scratch (prefill). The compacted bucket layout, the
  // (expert, row-tile) map and the F16 activation/intermediate rows used by
  // the matrix-core expert GEMMs. Only touched on the WMMA route.
  std::int32_t* pf_pad_bounds_{nullptr};
  std::int32_t* pf_cursors_{nullptr};
  std::int32_t* pf_rows_token_{nullptr};
  std::int32_t* pf_rows_slot_{nullptr};
  std::int32_t* pf_tiles_dev_{nullptr};
  std::vector<std::int32_t> pf_tiles_host_;
  void* pf_x_half_{nullptr};
  void* pf_up_half_{nullptr};

  // Prefill residual stream and per-layer intermediates, [chunk][hidden].
  float* pf_x_{nullptr};
  float* pf_normed_{nullptr};
  float* pf_attn_{nullptr};
  float* pf_ffn_{nullptr};

  // Prefill Gated DeltaNet scratch (reused across linear layers), row-major
  // [chunk][...]. `pf_hist_new_` is the disjoint conv-history output the chunk
  // writes back over the session's rolling history.
  float* pf_qkv_{nullptr};
  float* pf_z_{nullptr};
  float* pf_alpha_{nullptr};
  float* pf_beta_{nullptr};
  float* pf_alpha_pre_{nullptr};
  float* pf_beta_pre_{nullptr};
  float* pf_kq_pre_{nullptr};
  float* pf_convolved_{nullptr};
  float* pf_qn_{nullptr};
  float* pf_kn_{nullptr};
  float* pf_gdn_attn_{nullptr};
  float* pf_hist_new_{nullptr};

  // Prefill gated grouped-query attention scratch (reused across full layers),
  // row-major [chunk][...]. `pf_pos_` holds the absolute position of each row.
  float* pf_qg_{nullptr};
  float* pf_k_{nullptr};
  float* pf_v_{nullptr};
  float* pf_q_{nullptr};
  float* pf_qgate_{nullptr};
  float* pf_ctx_{nullptr};
  std::uint32_t* pf_pos_{nullptr};

  // MTP scratch. `mtp_chain_` keeps the draft block's unnormalized output
  // hidden before the shared head norm, the hidden input of the next draft.
  float* mtp_e_{nullptr};
  float* mtp_h_{nullptr};
  float* mtp_concat_{nullptr};
  float* mtp_cur_{nullptr};
  float* mtp_chain_{nullptr};

  // Speculative verify scratch: per-row logits ([kMaxVerifyRows][vocab]) and
  // post-output-norm hidden ([kMaxVerifyRows][hidden]).
  float* verify_logits_{nullptr};
  float* verify_h_{nullptr};

  // Batched MTP cache fill scratch ([chunk][2 * hidden] and [chunk][hidden]).
  float* pf_mtp_concat_{nullptr};
  float* pf_mtp_cur_{nullptr};

  std::vector<void*> allocations_;

  // Env-gated (GUFO_QWEN36_PROFILE=1) per-stage GPU timer for the forward
  // path. No-op unless the variable is set.
  StageProfiler prof_;
  std::uint32_t decode_steps_{0};
  std::uint32_t spec_rounds_{0};
};

}  // namespace gufo::models::qwen36_a3b::rocm

#endif  // GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_EXECUTOR_HPP_