#ifndef GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/models/qwen36_a3b/kernels/rocm/device_model.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/profile.hpp"

namespace gufo::models::qwen36_a3b::rocm {

/// Runs the trunk (and the MTP draft block) of the Qwen3.6-35B-A3B model on
/// the GPU, one token at a time. Weights stay in their GGUF encoding and are
/// decoded on the fly by the GEMV tier; the fused operators handle everything
/// else. The forward pass mirrors the scalar oracle in reference.cpp exactly,
/// so a GPU step and a CPU step agree to rounding.
class Executor {
public:
  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  /// Allocates the recurrent state and scratch for up to `max_context` tokens
  /// and binds the uploaded weights. Fails if any tensor is in a format the
  /// GEMV tier cannot decode.
  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const DeviceModel& model, std::uint32_t max_context,
      std::string* error_msg = nullptr);

  /// Advances the trunk by one token; the result is in logits() and h_out().
  bool Step(std::int32_t token, std::string* error_msg = nullptr);
  /// Advances the trunk by `count` prompt tokens in prefill-sized chunks,
  /// driving the batched GEMM tier and the fused prefill operators so the whole
  /// prompt is consumed without a per-token host round-trip. Recurrent state
  /// (Gated DeltaNet state/history and the KV caches) and `position_` advance
  /// exactly as they would have under `count` Step() calls, so decoding resumes
  /// from the next position. `logits()` and `h_out()` hold the final token's
  /// result.
  bool Prefill(const std::int32_t* tokens, std::uint32_t count,
               std::string* error_msg = nullptr);
  /// Runs the MTP draft block for `token` using the trunk hidden state from
  /// the most recent Step; the result is in mtp_logits().
  bool MtpStep(std::int32_t token, std::string* error_msg = nullptr);
  /// Advances the MTP cache by `token` against the given device-side trunk
  /// hidden state, skipping the shared head and LM head. Used to keep the
  /// draft cache aligned with the trunk after an accepted draft.
  bool MtpAdvance(std::int32_t token, const float* hidden,
                  std::string* error_msg = nullptr);
  /// Speculative verification: advances the trunk by two tokens in one
  /// batched pass (the GEMM tier, like Prefill). Row `r` of verify_logits()
  /// ([2][vocab]) holds the logits after consuming token `r`, and row `r` of
  /// verify_hidden() ([2][hidden]) the post-output-norm hidden state. The
  /// Gated DeltaNet state and conv history after the first row are
  /// snapshotted so RollbackVerify() can undo the second token.
  bool Verify(std::int32_t t0, std::int32_t t1,
              std::string* error_msg = nullptr);
  /// Undoes the second token of the most recent Verify: restores the
  /// recurrent state from the snapshot, rewinds `position_` by one and leaves
  /// the first row's hidden state in h_out().
  void RollbackVerify();
  /// Clears the recurrent state and both positions.
  void Reset();

  [[nodiscard]] const float* logits() const noexcept { return logits_; }
  [[nodiscard]] const float* mtp_logits() const noexcept { return mtp_logits_; }
  [[nodiscard]] const float* verify_logits() const noexcept {
    return verify_logits_;
  }
  [[nodiscard]] const float* verify_hidden() const noexcept {
    return verify_h_;
  }
  [[nodiscard]] const float* h_out() const noexcept { return h_out_; }
  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  [[nodiscard]] std::uint32_t max_context() const noexcept {
    return max_context_;
  }

private:
  Executor(const Config& c, const DeviceModel& model) : c_(c), model_(model) {}

  void LinearAttention(const DeviceLayer& l, std::uint32_t il, const float* x,
                       float* out);
  void Attention(const DeviceLayer& l, const float* x, std::uint32_t pos,
                 float* out, float* k_cache, float* v_cache,
                 const std::uint32_t* pos_dev);
  void Moe(const DeviceLayer& l, const float* x, float* out);
  /// Batched (prefill) Mixture-of-Experts over `tokens` rows of `x`
  /// ([tokens][hidden]) into `out` ([tokens][hidden]). Mirrors Moe() but drives
  /// the GEMM tier and the batched router/epilogue kernels, so the routed
  /// expert ids never round-trip through the host.
  void MoeBatch(const DeviceLayer& l, const float* x, float* out,
                std::uint32_t tokens);

  /// Batched (prefill) Gated DeltaNet over `tokens` rows of `x`
  /// ([tokens][hidden]) into `out` ([tokens][hidden]), advancing the layer's
  /// recurrent state and conv history past the chunk. Mirrors LinearAttention().
  /// When `state_snap` is non-null it receives the recurrent state after each
  /// of the first `tokens - 1` rows (see GdnDeltaLoop); `hist_snap` receives
  /// the conv history after the first row.
  void LinearAttentionBatch(const DeviceLayer& l, std::uint32_t il,
                            const float* x, float* out, std::uint32_t tokens,
                            float* state_snap = nullptr,
                            float* hist_snap = nullptr);
  /// Batched MTP cache fill over one prefill chunk. `hidden_rows`
  /// ([rows][hidden]) is the trunk's post-output-norm hidden state for the
  /// chunk; the block's own hidden input for row 0 comes from
  /// `mtp_prev_hidden_`. Advances the MTP KV caches and `mtp_position_`.
  void MtpPrefillChunk(const std::int32_t* tokens, std::uint32_t rows,
                       std::uint32_t start, const float* hidden_rows);
  /// The MTP block forward. `hidden` is the device-side trunk (or previous
  /// draft) hidden state; with `with_logits` the shared head and LM head run
  /// into mtp_logits_.
  bool MtpForward(std::int32_t token, const float* hidden, bool with_logits,
                  std::string* error_msg);
  /// Batched (prefill) gated grouped-query attention over `tokens` rows of `x`
  /// starting at absolute position `start`, writing the chunk's keys/values into
  /// the caches and reading them back causally. Mirrors Attention().
  void AttentionBatch(const DeviceLayer& l, const float* x, std::uint32_t start,
                      float* out, float* k_cache, float* v_cache,
                      const std::uint32_t* pos_dev, std::uint32_t tokens);

  float* AllocFloats(std::size_t n, std::string* error);
  std::int32_t* AllocInts(std::size_t n, std::string* error);
  std::uint32_t* AllocUints(std::size_t n, std::string* error);
  void* AllocBytes(std::size_t bytes, std::string* error);

  const Config& c_;
  const DeviceModel& model_;
  std::uint32_t max_context_{0};
  std::uint32_t position_{0};
  std::uint32_t mtp_position_{0};

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

  // Batched prefill scratch, sized to prefill_chunk_ tokens. The decode path
  // reuses the single-token buffers above; prefill drives these and never
  // syncs to the host. The MoE buffers hold one row per (token, slot) pair, so
  // they scale with prefill_chunk_ * num_experts_used.
  std::uint32_t prefill_chunk_{0};
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
  // [2*used][hidden] copy of the verify normed rows (row t repeated `used`
  // times) so the grouped expert GEMVs can index x by slot.
  float* pf_x_dup_{nullptr};
  // [2][heads][32][head_dim + 2] flash-decoding partials for the two verify
  // rows (AttentionDecode2).
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
  // writes back over the layer's rolling history.
  float* pf_qkv_{nullptr};
  float* pf_z_{nullptr};
  float* pf_alpha_{nullptr};
  float* pf_beta_{nullptr};
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

  // MTP scratch.
  float* mtp_e_{nullptr};
  float* mtp_h_{nullptr};
  float* mtp_concat_{nullptr};
  float* mtp_cur_{nullptr};

  // Speculative verify scratch: per-row logits ([2][vocab]) and post-output-
  // norm hidden ([2][hidden]) for the two-token verify pass.
  float* verify_logits_{nullptr};
  float* verify_h_{nullptr};

  // Batched MTP cache fill scratch ([chunk][2 * hidden] and [chunk][hidden])
  // and the last prefill chunk's hidden row, the draft block's hidden input
  // for the next chunk's first row.
  float* pf_mtp_concat_{nullptr};
  float* pf_mtp_cur_{nullptr};
  float* mtp_prev_hidden_{nullptr};

  // Recurrent state, indexed by layer (null where the layer does not use it).
  std::vector<float*> gdn_state_;
  std::vector<float*> gdn_history_;
  std::vector<float*> k_cache_;
  std::vector<float*> v_cache_;
  // Per-linear-layer verify snapshots: the recurrent state and conv history
  // after the first row of the most recent Verify (null without an MTP block).
  std::vector<float*> gdn_state_snap_;
  std::vector<float*> gdn_hist_snap_;
  float* mtp_k_cache_{nullptr};
  float* mtp_v_cache_{nullptr};

  std::vector<void*> allocations_;

  // Env-gated (GUFO_QWEN36_PROFILE=1) per-stage GPU timer for the forward
  // path. No-op unless the variable is set.
  StageProfiler prof_;
  std::uint32_t decode_steps_{0};
  std::uint32_t spec_rounds_{0};
};

}  // namespace gufo::models::qwen36_a3b::rocm

#endif  // GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_EXECUTOR_HPP_