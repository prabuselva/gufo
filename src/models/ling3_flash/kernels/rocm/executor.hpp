#ifndef GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_fp16.h>
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

  // Verify rollback snapshots for the KDA layers (null on MLA layers): the
  // conv history and recurrent state right after each of the first
  // kMaxVerifyRows - 1 rows of the latest Verify, so a rejected draft can
  // rewind to any committed prefix.
  std::vector<float*> kda_conv_snap_;
  std::vector<float*> kda_state_snap_;
  std::uint32_t verify_base_pos_{0};

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
/// Prefill advances recurrent state with the same semantics as Step and
/// produces bit-for-bit interchangeable session state; the chunked batch
/// tier runs it whenever the dense weights support it.
class Executor {
public:
  /// Rows a speculative verify pass can hold (token + drafts); matches the
  /// per-token tier of the batched GEMV kernels.
  static constexpr std::uint32_t kMaxVerifyRows = 5U;

  /// Tokens one chunked-prefill chunk holds; the batch tier scratch is sized
  /// for exactly this many rows and a longer prompt is split into chunks of
  /// it, so prefill cost is linear in the prompt with bounded scratch.
  static constexpr std::uint32_t kPrefillChunkRows = 1024U;

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
  /// Advances the trunk by `count` prompt tokens in chunks of at most
  /// kPrefillChunkRows: the batch tier (chunked GEMMs, the recurrent KDA
  /// recurrence, the tiled MLA prefill attention and the routed expert GEMM)
  /// runs the whole chunk between the projections. `logits()` and `h_out()`
  /// hold the final token's result, exactly as `count` Step() calls would.
  bool Prefill(Session& session, const std::int32_t* tokens,
               std::uint32_t count, std::string* error_msg = nullptr);
  /// Runs the MTP draft block for `token` using the trunk hidden state from
  /// the most recent Step; the result is in mtp_logits().
  bool MtpStep(Session& session, std::int32_t token,
               std::string* error_msg = nullptr);
  /// Runs the MTP draft block for `token` from the previous draft's own
  /// output hidden; the result is in mtp_logits(). Call after MtpStep or a
  /// previous MtpDraft to chain speculative drafts.
  bool MtpDraft(Session& session, std::int32_t token,
                std::string* error_msg = nullptr);
  /// Advances the draft cache over `token` with the trunk hidden row from
  /// `verify_hidden()` after a fully accepted verify; runs no head.
  bool MtpAdvance(Session& session, std::int32_t token, const float* hidden,
                  std::string* error_msg = nullptr);
  /// Batched speculative check: runs the trunk over `t0` followed by the `k`
  /// draft tokens at consecutive positions, reading the trunk weights once
  /// for the whole block. Row logits land in verify_logits() and the
  /// output-normalized rows in verify_hidden(); h_out() then equals row k
  /// and the trunk advances by k + 1 rows. RollbackVerify() rewinds the
  /// recurrent state to any committed prefix afterwards.
  bool Verify(Session& session, std::int32_t t0, const std::int32_t* drafts,
              std::uint32_t k, std::string* error_msg = nullptr);
  /// Commits the first `keep` verify rows: restores the KDA state and conv
  /// history from the snapshot after row keep - 1 and rewinds the trunk and
  /// draft positions (and h_out()) to just past that row.
  void RollbackVerify(Session& session, std::uint32_t keep);

  [[nodiscard]] const float* logits() const noexcept { return logits_; }
  [[nodiscard]] const float* mtp_logits() const noexcept { return mtp_logits_; }
  [[nodiscard]] const float* h_out() const noexcept { return h_out_; }
  [[nodiscard]] const float* verify_logits() const noexcept {
    return pf_logits_;
  }
  [[nodiscard]] const float* verify_hidden() const noexcept { return pf_h_; }

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

  /// Draft-block forward fed by `hidden`; `with_logits` runs the shared head
  /// (and keeps the unnormalized output in mtp_chain_ for the next draft).
  bool MtpForward(Session& session, std::int32_t token, const float* hidden,
                  bool with_logits, std::string* error_msg);
  /// Multi-row variants driving the verify block over pf_normed_ -> pf_attn_
  /// / pf_ffn_; the KDA batch also writes the per-row rollback snapshots.
  void KdaBatch(Session& session, std::uint32_t il, const DeviceLayer& l,
                std::uint32_t rows);
  void MlaBatch(const DeviceLayer& l, float* kv_cache, float* kpe_cache,
                std::uint32_t base_pos, std::uint32_t rows);
  void FfnBatch(const DeviceLayer& l, std::uint32_t il, std::uint32_t rows);

  /// One chunked-prefill chunk over pf_x_[0:tokens]: the batch-tier layer
  /// loop, updating the session caches and recurrent state for positions
  /// [base_pos, base_pos + tokens).
  void PrefillChunk(Session& session, std::uint32_t tokens,
                    std::uint32_t base_pos);
  void PrefillKda(Session& session, std::uint32_t il, const DeviceLayer& l,
                  std::uint32_t tokens);
  void PrefillMla(const DeviceLayer& l, float* kv_cache, float* kpe_cache,
                  std::uint32_t base_pos, std::uint32_t tokens);
  void PrefillFfn(const DeviceLayer& l, std::uint32_t il, std::uint32_t tokens);

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
  float* mtp_chain_{nullptr};  ///< Unnormalized draft output for MtpDraft.

  // Multi-row block scratch, shared by the speculative verify block (up to
  // kMaxVerifyRows rows, allocated when the model has an MTP block) and the
  // chunked prefill (up to kPrefillChunkRows rows, always allocated). Rows
  // are dense with the strides given below.
  float* pf_x_{nullptr};       ///< [rows][hidden] residual stream.
  float* pf_normed_{nullptr};  ///< [rows][hidden] normalized input.
  float* pf_attn_{nullptr};    ///< [rows][hidden] attention results.
  float* pf_ffn_{nullptr};     ///< [rows][hidden] FFN results.
  float* pf_h_{nullptr};       ///< [rows][hidden] output-normalized rows.
  float* pf_logits_{nullptr};  ///< [rows][vocab] row logits.
  float* pf_kda_q_{
      nullptr};  ///< [rows][inner] projections (beta [rows][heads]).
  float* pf_kda_k_{nullptr};
  float* pf_kda_v_{nullptr};
  float* pf_kda_f_{nullptr};
  float* pf_kda_g_{nullptr};
  float* pf_kda_beta_{nullptr};
  float* pf_mla_q_{nullptr};    ///< [rows][heads * mla_head_dim] raw queries.
  float* pf_mla_a_{nullptr};    ///< [rows][MlaKeyDim] latent + rope rows.
  float* pf_mla_abs_{nullptr};  ///< [rows][heads * MlaKeyDim] absorbed queries.
  float* pf_mla_head_{nullptr};  ///< [rows][heads * value_dim] gated heads.
  std::int32_t* pf_moe_ids_{nullptr};  ///< [rows][used] routed slot ids.
  float* pf_moe_weights_{nullptr};     ///< [rows][used] routing weights.
  float* pf_moe_log_{nullptr};         ///< [rows][num_experts] router rows.
  float* pf_moe_gate_{nullptr};        ///< [rows][used * expert_ff]
  float* pf_moe_up_{nullptr};          ///< [rows][used * expert_ff]
  float* pf_moe_inter_{nullptr};       ///< [rows][used * expert_ff]
  float* pf_moe_down_{nullptr};        ///< [rows][used * hidden]
  float* pf_sh_gate_{nullptr};         ///< [rows][shared_expert_ff]
  float* pf_sh_up_{nullptr};           ///< [rows][shared_expert_ff]
  float* pf_sh_inter_{nullptr};        ///< [rows][shared_expert_ff]
  float* pf_sh_down_{nullptr};         ///< [rows][hidden]

  // Chunked-prefill-only scratch, sized for kPrefillChunkRows rows. The raw
  // projection planes above (pf_kda_*, pf_mla_q_/_a_/_head_, pf_moe_*,
  // pf_sh_*) and the residual stream planes are shared with the verify pass.
  __half* pf_xh_{nullptr};  ///< [rows][k] F16 activations for the F16 GEMMs
                            ///< (row stride is the launch's k).
  float* pf_cq_{nullptr};   ///< [rows][inner] convolved q/k/v streams.
  float* pf_ck_{nullptr};
  float* pf_cv_{nullptr};
  float* pf_o_{nullptr};      ///< [rows][inner] raw pre-out-norm KDA attention.
  float* pf_decay_{nullptr};  ///< [rows][inner] KdaPrep decay planes.
  float* pf_bpre_{nullptr};   ///< [rows][heads] sigmoid(beta).
  float* pf_kq_{nullptr};     ///< [rows][heads] dot(kn, qn).
  float* pf_kda_hist_{nullptr};  ///< [3][kern-1][inner] next-history staging.
  float* pf_mla_pabs_{nullptr};  ///< [rows][heads][kv_lora_rank] absorbed q.
  float* pf_mla_qrp_{nullptr};   ///< [rows][heads][rotary] rotated rope halves.
  float* pf_mla_qlatent_{nullptr};  ///< [rows][heads][kv_lora_rank] attn out.
  float* pf_mla_pgate_{nullptr};    ///< [rows][heads] attention output gate.
  float* pf_dn_gate_{nullptr};  ///< [rows][dense_ff] dense-FFN gate (up, inter)
  float* pf_dn_up_{nullptr};
  float* pf_dn_inter_{nullptr};
  __half* pf_interh_{nullptr};  ///< [rows * used][expert_ff] routed inter.
  float* pf_pdown_{nullptr};    ///< [rows * used][hidden] routed down.
  std::int32_t* pf_counts_{nullptr};    ///< [num_experts] expert histogram.
  std::int32_t* pf_bounds_{nullptr};    ///< [num_experts + 1] bucket starts.
  std::int32_t* pf_cursors_{nullptr};   ///< [num_experts] scatter scratch.
  std::int32_t* pf_rows_tok_{nullptr};  ///< compact row -> token / slot.
  std::int32_t* pf_rows_slot_{nullptr};
  std::int32_t* pf_tiles_{nullptr};  ///< compacted (expert, tile) list.
  __half* pf_qnc_{nullptr};  ///< [rows * heads][nope] slot-major F16 queries.
  std::int32_t* pf_hbounds_{nullptr};  ///< head-bucket starts [heads + 1].
  std::int32_t* pf_hcursors_{
      nullptr};                       ///< [heads] head-bucket scatter scratch.
  std::int32_t* pf_htiles_{nullptr};  ///< head-bucket (head, tile) list.
  std::int32_t* pf_zero_ids_{nullptr};  ///< [rows] zeros (F32 router grouping).
  std::int32_t* pf_head_ids_{nullptr};  ///< [rows * heads] slot -> head id.
  std::int32_t* pf_drows_{nullptr};     ///< dense identity row map for a chunk.
  std::int32_t* pf_dbounds_{nullptr};
  std::int32_t* pf_dtiles_{nullptr};
  bool prefill_ready_{false};  ///< The dense tier can run every trunk layer.
  bool tier_failure_{false};   ///< A chunked-tier launch was rejected.

  std::uint32_t scratch_context_{0};
  std::vector<void*> allocations_;
  std::vector<std::int32_t> tiles_host_;    ///< tile build scratch (host).
  std::vector<std::uint32_t> counts_host_;  ///< expert counts download (host).
};

}  // namespace gufo::models::ling3_flash::rocm

#endif  // GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_EXECUTOR_HPP_