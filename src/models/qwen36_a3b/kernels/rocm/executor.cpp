#include "src/models/qwen36_a3b/kernels/rocm/executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "qfn_mmq.h"
#include "src/models/qwen36_a3b/kernels/rocm/gemm.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/gemv.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/kernels.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/routed_f16.hpp"

namespace gufo::models::qwen36_a3b::rocm {
namespace {

/// Maps the artifact's GGUF type onto the GEMV tier's row format. The dense
/// tensors store Q8_0/F32/BF16; the routed experts additionally store the Q4_K
/// and Q5_K super-block encodings of a Q4_K_XL artifact. Anything else is a
/// load error (ValidateTypes).
[[nodiscard]] GemvType ToGemvType(core::GgmlType t) noexcept {
  switch (t) {
    case core::GgmlType::kQ8_0:
      return GemvType::kQ8_0;
    case core::GgmlType::kF32:
      return GemvType::kF32;
    case core::GgmlType::kBF16:
      return GemvType::kBF16;
    case core::GgmlType::kQ4_K:
      return GemvType::kQ4_K;
    case core::GgmlType::kQ5_K:
      return GemvType::kQ5_K;
    default:
      return GemvType::kF32;  // unreachable: Create rejects other types
  }
}

[[nodiscard]] bool SupportedType(core::GgmlType t) noexcept {
  return t == core::GgmlType::kQ8_0 || t == core::GgmlType::kF32 ||
         t == core::GgmlType::kBF16;
}

// The routed-expert stacks (ffn_gate/up/down_exps) additionally accept the
// Q4_K/Q5_K super-block encodings, decoded natively by the grouped GEMV and the
// WMMA routed GEMM. Q6_K experts are upcast to Q8_0 during upload, so they
// never reach here.
[[nodiscard]] bool ExpertType(core::GgmlType t) noexcept {
  return SupportedType(t) || t == core::GgmlType::kQ4_K ||
         t == core::GgmlType::kQ5_K;
}

/// Rejects any resident tensor the GEMV tier cannot decode, so a mis-quantized
/// artifact fails at Create rather than as a wrong number later.
bool ValidateTypes(const DeviceModel& m, std::string* error) {
  const auto check = [&](const DeviceTensor& t, const char* name) {
    if (t.empty() || SupportedType(t.type)) {
      return true;
    }
    if (error != nullptr) {
      *error = std::string("tensor ") + name +
               " has a format the GEMV tier cannot decode";
    }
    return false;
  };
  // The routed expert stacks carry the wider Q4_K/Q5_K encodings; every other
  // tensor is restricted to the dense Q8_0/F32/BF16 set.
  const auto check_expert = [&](const DeviceTensor& t, const char* name) {
    if (t.empty() || ExpertType(t.type)) {
      return true;
    }
    if (error != nullptr) {
      *error = std::string("tensor ") + name +
               " has a format the GEMV tier cannot decode";
    }
    return false;
  };
  if (!check(m.token_embd(), "token_embd") || !check(m.output(), "output")) {
    return false;
  }
  const auto check_layer = [&](const DeviceLayer& l, const char* prefix) {
    const char* base = prefix;
    if (!check(l.attn_norm, base) || !check(l.post_attention_norm, base) ||
        !check(l.ssm_qkv, base) || !check(l.ssm_gate, base) ||
        !check(l.ssm_conv1d, base) || !check(l.ssm_alpha, base) ||
        !check(l.ssm_beta, base) || !check(l.ssm_dt, base) ||
        !check(l.ssm_a, base) || !check(l.ssm_norm, base) ||
        !check(l.ssm_out, base) || !check(l.attn_q, base) ||
        !check(l.attn_k, base) || !check(l.attn_v, base) ||
        !check(l.attn_out, base) || !check(l.attn_q_norm, base) ||
        !check(l.attn_k_norm, base) || !check(l.router, base) ||
        !check_expert(l.ffn_gate_exps, base) ||
        !check_expert(l.ffn_up_exps, base) ||
        !check_expert(l.ffn_down_exps, base) ||
        !check(l.shexp_gate_inp, base) || !check(l.shexp_gate, base) ||
        !check(l.shexp_up, base) || !check(l.shexp_down, base) ||
        !check(l.nextn_enorm, base) || !check(l.nextn_hnorm, base) ||
        !check(l.nextn_eh_proj, base) ||
        !check(l.nextn_shared_head_norm, base)) {
      return false;
    }
    return true;
  };
  for (std::size_t il = 0; il < m.layers().size(); ++il) {
    if (!check_layer(m.layers()[il], "layer")) {
      return false;
    }
  }
  if (m.has_mtp() && !check_layer(m.mtp(), "mtp")) {
    return false;
  }
  return true;
}

}  // namespace

Executor::~Executor() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

float* Executor::AllocFloats(std::size_t n, std::string* error) {
  float* p = nullptr;
  if (n > 0 && hipMalloc(&p, n * sizeof(float)) != hipSuccess) {
    if (error != nullptr) {
      *error = "hipMalloc failed for executor scratch";
    }
    return nullptr;
  }
  if (p != nullptr) {
    allocations_.push_back(p);
  }
  return p;
}

std::int32_t* Executor::AllocInts(std::size_t n, std::string* error) {
  std::int32_t* p = nullptr;
  if (n > 0 && hipMalloc(&p, n * sizeof(std::int32_t)) != hipSuccess) {
    if (error != nullptr) {
      *error = "hipMalloc failed for executor scratch";
    }
    return nullptr;
  }
  if (p != nullptr) {
    allocations_.push_back(p);
  }
  return p;
}

std::uint32_t* Executor::AllocUints(std::size_t n, std::string* error) {
  std::uint32_t* p = nullptr;
  if (n > 0 && hipMalloc(&p, n * sizeof(std::uint32_t)) != hipSuccess) {
    if (error != nullptr) {
      *error = "hipMalloc failed for executor scratch";
    }
    return nullptr;
  }
  if (p != nullptr) {
    allocations_.push_back(p);
  }
  return p;
}

void* Executor::AllocBytes(std::size_t bytes, std::string* error) {
  void* p = nullptr;
  if (bytes > 0 && hipMalloc(&p, bytes) != hipSuccess) {
    if (error != nullptr) {
      *error = "hipMalloc failed for executor scratch";
    }
    return nullptr;
  }
  if (p != nullptr) {
    allocations_.push_back(p);
  }
  return p;
}

Session::~Session() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

float* Session::AllocFloats(std::size_t n) {
  float* p = nullptr;
  if (n > 0 && hipMalloc(&p, n * sizeof(float)) != hipSuccess) {
    alloc_failed_ = true;
    return nullptr;
  }
  if (p != nullptr) {
    allocations_.push_back(p);
    allocated_bytes_ += n * sizeof(float);
  }
  return p;
}

void* Session::AllocBytes(std::size_t bytes) {
  void* p = nullptr;
  if (bytes > 0 && hipMalloc(&p, bytes) != hipSuccess) {
    alloc_failed_ = true;
    return nullptr;
  }
  if (p != nullptr) {
    allocations_.push_back(p);
    allocated_bytes_ += bytes;
  }
  return p;
}

void Session::Reset() {
  position_ = 0;
  mtp_position_ = 0;
  const std::size_t state_elems = static_cast<std::size_t>(c_.ssm_num_v_heads) *
                                  c_.ssm_head_dim * c_.ssm_head_dim;
  const std::size_t history_elems =
      static_cast<std::size_t>(c_.ssm_conv_kernel - 1) * c_.SsmConvChannels();
  const std::size_t kv_elems =
      static_cast<std::size_t>(c_.num_kv_heads) * c_.head_dim * max_context_;
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    if (c_.IsLinearLayer(il)) {
      (void)hipMemsetAsync(gdn_state_[il], 0, state_elems * sizeof(float),
                           nullptr);
      (void)hipMemsetAsync(gdn_history_[il], 0, history_elems * sizeof(float),
                           nullptr);
    } else {
      (void)hipMemsetAsync(k_cache_[il], 0, kv_elems * sizeof(float), nullptr);
      (void)hipMemsetAsync(v_cache_[il], 0, kv_elems * sizeof(float), nullptr);
      (void)hipMemsetAsync(k_cache_f16_[il], 0, kv_elems * sizeof(__half),
                           nullptr);
      (void)hipMemsetAsync(v_cache_f16_[il], 0, kv_elems * sizeof(__half),
                           nullptr);
    }
  }
  if (has_mtp_) {
    (void)hipMemsetAsync(mtp_k_cache_, 0, kv_elems * sizeof(float), nullptr);
    (void)hipMemsetAsync(mtp_v_cache_, 0, kv_elems * sizeof(float), nullptr);
    (void)hipMemsetAsync(mtp_k_cache_f16_, 0, kv_elems * sizeof(__half),
                         nullptr);
    (void)hipMemsetAsync(mtp_v_cache_f16_, 0, kv_elems * sizeof(__half),
                         nullptr);
    (void)hipMemsetAsync(mtp_prev_hidden_, 0, c_.hidden_size * sizeof(float),
                         nullptr);
  }
}

std::unique_ptr<Session> Executor::CreateSession(
    std::uint32_t max_context, std::string* error_msg) const {
  if (max_context == 0) {
    if (error_msg != nullptr) {
      *error_msg = "max_context must be positive";
    }
    return nullptr;
  }
  if (max_context > scratch_context_) {
    if (error_msg != nullptr) {
      *error_msg = "session context exceeds the executor scratch bound";
    }
    return nullptr;
  }
  const Config& c = model_.config();
  std::unique_ptr<Session> s(new Session(c, model_.has_mtp()));
  s->max_context_ = max_context;

  // Recurrent state, per layer, sized to this session's context.
  const std::size_t state_elems = static_cast<std::size_t>(c.ssm_num_v_heads) *
                                  c.ssm_head_dim * c.ssm_head_dim;
  const std::size_t history_elems =
      static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
  const std::size_t kv_elems =
      static_cast<std::size_t>(c.num_kv_heads) * c.head_dim * max_context;
  s->gdn_state_.assign(c.num_layers, nullptr);
  s->gdn_history_.assign(c.num_layers, nullptr);
  s->k_cache_.assign(c.num_layers, nullptr);
  s->v_cache_.assign(c.num_layers, nullptr);
  s->k_cache_f16_.assign(c.num_layers, nullptr);
  s->v_cache_f16_.assign(c.num_layers, nullptr);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (c.IsLinearLayer(il)) {
      s->gdn_state_[il] = s->AllocFloats(state_elems);
      s->gdn_history_[il] = s->AllocFloats(history_elems);
    } else {
      s->k_cache_[il] = s->AllocFloats(kv_elems);
      s->v_cache_[il] = s->AllocFloats(kv_elems);
      s->k_cache_f16_[il] = s->AllocBytes(kv_elems * sizeof(__half));
      s->v_cache_f16_[il] = s->AllocBytes(kv_elems * sizeof(__half));
    }
  }
  s->gdn_state_snap_.assign(c.num_layers, nullptr);
  s->gdn_hist_snap_.assign(c.num_layers, nullptr);
  if (s->has_mtp_) {
    s->mtp_k_cache_ = s->AllocFloats(kv_elems);
    s->mtp_v_cache_ = s->AllocFloats(kv_elems);
    s->mtp_k_cache_f16_ = s->AllocBytes(kv_elems * sizeof(__half));
    s->mtp_v_cache_f16_ = s->AllocBytes(kv_elems * sizeof(__half));
    s->mtp_prev_hidden_ = s->AllocFloats(c.hidden_size);
    // One recurrent-state snapshot row per droppable verify row (all but the
    // last), so a rejected draft can rewind to any committed prefix.
    constexpr std::size_t kRows = Executor::kMaxVerifyRows;
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      if (c.IsLinearLayer(il)) {
        s->gdn_state_snap_[il] = s->AllocFloats((kRows - 1U) * state_elems);
        s->gdn_hist_snap_[il] = s->AllocFloats((kRows - 1U) * history_elems);
      }
    }
  }

  // Any allocation failure leaves a null pointer; free the partial session and
  // report once.
  if (s->alloc_failed_) {
    if (error_msg != nullptr) {
      *error_msg = "hipMalloc failed for session state";
    }
    return nullptr;
  }
  s->Reset();
  return s;
}

std::size_t Executor::SessionBytes(std::uint32_t max_context) const noexcept {
  if (max_context == 0) {
    return 0;
  }
  const Config& c = model_.config();
  const std::size_t state_bytes =
      static_cast<std::size_t>(c.ssm_num_v_heads) * c.ssm_head_dim *
      c.ssm_head_dim * sizeof(float);
  const std::size_t history_bytes =
      static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels() *
      sizeof(float);
  const std::size_t kv_f32 =
      static_cast<std::size_t>(c.num_kv_heads) * c.head_dim * max_context;
  const std::size_t kv_bytes = kv_f32 * sizeof(float);
  const std::size_t kv_f16_bytes = kv_f32 * sizeof(__half);
  std::size_t bytes = 0;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (c.IsLinearLayer(il)) {
      bytes += state_bytes + history_bytes;
    } else {
      bytes += 2 * kv_bytes + 2 * kv_f16_bytes;
    }
  }
  if (model_.has_mtp()) {
    bytes += 2 * kv_bytes + 2 * kv_f16_bytes;  // draft KV + FP16 mirrors
    bytes += static_cast<std::size_t>(c.hidden_size) * sizeof(float);
    constexpr std::size_t kRows = Executor::kMaxVerifyRows;
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      if (c.IsLinearLayer(il)) {
        bytes += (kRows - 1U) * (state_bytes + history_bytes);
      }
    }
  }
  return bytes;
}

std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           std::uint32_t max_context,
                                           std::string* error_msg,
                                           std::uint32_t attn_window,
                                           std::uint32_t attn_sink) {
  if (max_context == 0) {
    if (error_msg != nullptr) {
      *error_msg = "max_context must be positive";
    }
    return nullptr;
  }
  if (!ValidateTypes(model, error_msg)) {
    return nullptr;
  }
  // The shared mmq tensor-core path needs a one-time device-context setup
  // before its first launch. Guarded so repeated Executor::Create is safe.
  static std::once_flag mmq_once;
  std::call_once(mmq_once, [] { (void)qfn_mmq_init(0); });
  std::unique_ptr<Executor> e(new Executor(model.config(), model));
  e->scratch_context_ = max_context;
  const Config& c = e->c_;

  // Residual stream and outputs.
  e->x_ = e->AllocFloats(c.hidden_size, error_msg);
  e->normed_ = e->AllocFloats(c.hidden_size, error_msg);
  e->attn_ = e->AllocFloats(c.hidden_size, error_msg);
  e->ffn_ = e->AllocFloats(c.hidden_size, error_msg);
  e->h_out_ = e->AllocFloats(c.hidden_size, error_msg);
  e->logits_ = e->AllocFloats(c.vocab_size, error_msg);
  e->mtp_logits_ = e->AllocFloats(c.vocab_size, error_msg);
  e->pos_dev_ = e->AllocUints(1, error_msg);
  e->mtp_pos_dev_ = e->AllocUints(1, error_msg);

  // Gated DeltaNet scratch.
  e->gdn_qkv_ = e->AllocFloats(c.SsmConvChannels(), error_msg);
  e->gdn_z_ = e->AllocFloats(c.SsmValueDim(), error_msg);
  e->gdn_alpha_ = e->AllocFloats(c.ssm_num_v_heads, error_msg);
  e->gdn_beta_ = e->AllocFloats(c.ssm_num_v_heads, error_msg);
  e->gdn_convolved_ = e->AllocFloats(c.SsmConvChannels(), error_msg);
  e->gdn_qn_ = e->AllocFloats(c.SsmKeyDim(), error_msg);
  e->gdn_kn_ = e->AllocFloats(c.SsmKeyDim(), error_msg);
  e->gdn_attn_ = e->AllocFloats(c.SsmValueDim(), error_msg);

  // Gated grouped-query attention scratch.
  e->gqa_qg_ = e->AllocFloats(2 * c.AttentionQDim(), error_msg);
  e->gqa_k_ = e->AllocFloats(c.AttentionKvDim(), error_msg);
  e->gqa_v_ = e->AllocFloats(c.AttentionKvDim(), error_msg);
  e->gqa_q_ = e->AllocFloats(c.AttentionQDim(), error_msg);
  e->gqa_gate_ = e->AllocFloats(c.AttentionQDim(), error_msg);
  e->gqa_ctx_ = e->AllocFloats(c.AttentionQDim(), error_msg);
  e->gqa_scratch_ = e->AllocFloats(
      static_cast<std::size_t>(c.num_heads) * max_context, error_msg);
  e->gqa_part_ = e->AllocFloats(static_cast<std::size_t>(c.num_heads) * 32U *
                                    (c.AttentionQDim() / c.num_heads + 2U),
                                error_msg);

  // Mixture-of-experts scratch.
  e->moe_logits_ = e->AllocFloats(c.num_experts, error_msg);
  e->moe_ids_ = e->AllocInts(c.num_experts_used, error_msg);
  e->moe_weights_ = e->AllocFloats(c.num_experts_used, error_msg);
  e->moe_expert_out_ = e->AllocFloats(
      static_cast<std::size_t>(c.num_experts_used) * c.hidden_size, error_msg);
  e->moe_gate_ = e->AllocFloats(
      static_cast<std::size_t>(c.num_experts_used) * c.expert_ff, error_msg);
  e->moe_up_ = e->AllocFloats(
      static_cast<std::size_t>(c.num_experts_used) * c.expert_ff, error_msg);
  e->moe_shared_down_ = e->AllocFloats(c.hidden_size, error_msg);
  e->moe_shared_gate_ = e->AllocFloats(1, error_msg);

  // Batched prefill scratch. The chunk is capped at 2048 tokens (and at the
  // context length) to bound the (token, slot) MoE buffers near ~230 MB.
  const std::uint32_t chunk = max_context < 2048u ? max_context : 2048u;
  e->prefill_chunk_ = chunk;
  // Opt-in prefill attention sparsity (off unless a window is set). A positive
  // `attn_window`/`attn_sink` from the caller (CLI) wins; otherwise fall back
  // to GUFO_QWEN36_ATTN_WINDOW / GUFO_QWEN36_ATTN_SINK. The sink is the number
  // of always-attended initial tokens.
  if (attn_window > 0) {
    e->attn_window_ = attn_window;
  } else if (const char* w = std::getenv("GUFO_QWEN36_ATTN_WINDOW")) {
    if (w[0] != '\0') {
      e->attn_window_ =
          static_cast<std::uint32_t>(std::strtoul(w, nullptr, 10));
    }
  }
  if (attn_sink > 0) {
    e->attn_sink_ = attn_sink;
  } else if (const char* s = std::getenv("GUFO_QWEN36_ATTN_SINK")) {
    if (s[0] != '\0') {
      e->attn_sink_ = static_cast<std::uint32_t>(std::strtoul(s, nullptr, 10));
    }
  }
  const std::size_t pairs =
      static_cast<std::size_t>(chunk) * c.num_experts_used;
  e->pf_router_logits_ = e->AllocFloats(
      static_cast<std::size_t>(chunk) * c.num_experts, error_msg);
  e->pf_ids_ = e->AllocInts(pairs, error_msg);
  e->pf_expert_counts_ = e->AllocUints(c.num_experts, error_msg);
  e->pf_counts_host_.assign(c.num_experts, 0u);
  e->pf_weights_ = e->AllocFloats(pairs, error_msg);
  e->pf_gate_ = e->AllocFloats(pairs * c.expert_ff, error_msg);
  e->pf_up_ = e->AllocFloats(pairs * c.expert_ff, error_msg);
  e->pf_expert_out_ = e->AllocFloats(pairs * c.hidden_size, error_msg);
  e->pf_shared_gate_ = e->AllocFloats(
      static_cast<std::size_t>(chunk) * c.shared_expert_ff, error_msg);
  e->pf_shared_up_ = e->AllocFloats(
      static_cast<std::size_t>(chunk) * c.shared_expert_ff, error_msg);
  e->pf_shared_down_ = e->AllocFloats(
      static_cast<std::size_t>(chunk) * c.hidden_size, error_msg);
  e->pf_shared_gate_inp_ = e->AllocFloats(chunk, error_msg);
  e->pf_x_dup_ =
      e->AllocFloats(static_cast<std::size_t>(Executor::kMaxVerifyRows) *
                         c.num_experts_used * c.hidden_size,
                     error_msg);
  e->pf_part2_ = e->AllocFloats(
      static_cast<std::size_t>(Executor::kMaxVerifyRows) * c.num_heads * 32U *
          (c.AttentionQDim() / c.num_heads + 2U),
      error_msg);

  // Routed F16 WMMA MoE scratch. The compacted bucket layout needs
  // RoutedCompactRows rows; the (expert, row-tile) map is bounded by one tile
  // per 16 padded rows plus one per expert. The F16 activation and up rows
  // mirror the F32 gate/expert buffers at half the bytes.
  const std::size_t compact_rows = RoutedCompactRows(pairs, c.num_experts);
  e->pf_pad_bounds_ = e->AllocInts(c.num_experts + 1, error_msg);
  e->pf_cursors_ = e->AllocInts(c.num_experts, error_msg);
  e->pf_rows_token_ = e->AllocInts(compact_rows, error_msg);
  e->pf_rows_slot_ = e->AllocInts(compact_rows, error_msg);
  // The 64-row map plus an optional paired (gate+up) map of the same bound.
  const std::size_t max_tiles = pairs / 16 + 2 * (c.num_experts + 16);
  e->pf_tiles_dev_ = e->AllocInts(max_tiles, error_msg);
  e->pf_tiles_host_.assign(max_tiles, 0);
  e->pf_x_half_ = e->AllocBytes(
      static_cast<std::size_t>(chunk) * c.hidden_size * sizeof(__half),
      error_msg);
  e->pf_up_half_ =
      e->AllocBytes(pairs * c.expert_ff * sizeof(__half), error_msg);

  // Prefill residual stream and per-layer intermediates.
  e->pf_x_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c.hidden_size,
                            error_msg);
  e->pf_normed_ = e->AllocFloats(
      static_cast<std::size_t>(chunk) * c.hidden_size, error_msg);
  e->pf_attn_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c.hidden_size,
                               error_msg);
  e->pf_ffn_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c.hidden_size,
                              error_msg);

  // Prefill Gated DeltaNet scratch.
  const std::size_t c_key = c.SsmKeyDim();
  const std::size_t c_val = c.SsmValueDim();
  const std::size_t c_chan = c.SsmConvChannels();
  e->pf_qkv_ =
      e->AllocFloats(static_cast<std::size_t>(chunk) * c_chan, error_msg);
  e->pf_z_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_val, error_msg);
  e->pf_alpha_ = e->AllocFloats(
      static_cast<std::size_t>(chunk) * c.ssm_num_v_heads, error_msg);
  e->pf_beta_ = e->AllocFloats(
      static_cast<std::size_t>(chunk) * c.ssm_num_v_heads, error_msg);
  e->pf_alpha_pre_ = e->AllocFloats(
      static_cast<std::size_t>(chunk) * c.ssm_num_v_heads, error_msg);
  e->pf_beta_pre_ = e->AllocFloats(
      static_cast<std::size_t>(chunk) * c.ssm_num_v_heads, error_msg);
  e->pf_kq_pre_ = e->AllocFloats(
      static_cast<std::size_t>(chunk) * c.ssm_num_k_heads, error_msg);
  e->pf_convolved_ =
      e->AllocFloats(static_cast<std::size_t>(chunk) * c_chan, error_msg);
  e->pf_qn_ =
      e->AllocFloats(static_cast<std::size_t>(chunk) * c_key, error_msg);
  e->pf_kn_ =
      e->AllocFloats(static_cast<std::size_t>(chunk) * c_key, error_msg);
  e->pf_gdn_attn_ =
      e->AllocFloats(static_cast<std::size_t>(chunk) * c_val, error_msg);
  e->pf_hist_new_ = e->AllocFloats(
      static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c_chan, error_msg);

  // Prefill gated grouped-query attention scratch.
  const std::size_t c_q = c.AttentionQDim();
  const std::size_t c_kv = c.AttentionKvDim();
  e->pf_qg_ =
      e->AllocFloats(static_cast<std::size_t>(chunk) * 2 * c_q, error_msg);
  e->pf_k_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_kv, error_msg);
  e->pf_v_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_kv, error_msg);
  e->pf_q_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_q, error_msg);
  e->pf_qgate_ =
      e->AllocFloats(static_cast<std::size_t>(chunk) * c_q, error_msg);
  e->pf_ctx_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_q, error_msg);
  e->pf_pos_ = e->AllocUints(chunk, error_msg);

  // MTP scratch.
  e->mtp_e_ = e->AllocFloats(c.hidden_size, error_msg);
  e->mtp_h_ = e->AllocFloats(c.hidden_size, error_msg);
  e->mtp_concat_ = e->AllocFloats(2 * c.hidden_size, error_msg);
  e->mtp_cur_ = e->AllocFloats(c.hidden_size, error_msg);
  e->mtp_chain_ = e->AllocFloats(c.hidden_size, error_msg);

  // Speculative verify scratch: kMaxVerifyRows rows of logits and hidden
  // states and the batched MTP fill buffers. These are shared across sessions
  // (compute is serialized); the per-session recurrent state, KV caches and
  // verify snapshots are allocated by CreateSession.
  if (model.has_mtp()) {
    constexpr std::size_t kRows = Executor::kMaxVerifyRows;
    e->verify_logits_ = e->AllocFloats(kRows * c.vocab_size, error_msg);
    e->verify_h_ = e->AllocFloats(kRows * c.hidden_size, error_msg);
    e->pf_mtp_concat_ = e->AllocFloats(
        static_cast<std::size_t>(chunk) * 2 * c.hidden_size, error_msg);
    e->pf_mtp_cur_ = e->AllocFloats(
        static_cast<std::size_t>(chunk) * c.hidden_size, error_msg);
  }

  if (error_msg != nullptr && !error_msg->empty()) {
    return nullptr;
  }
  return e;
}

void Executor::LinearAttention(Session& session, const DeviceLayer& l,
                               std::uint32_t il, const float* x, float* out) {
  const std::uint32_t channels = c_.SsmConvChannels();
  const std::uint32_t key_dim = c_.SsmKeyDim();
  const std::uint32_t d = c_.ssm_head_dim;
  const std::uint32_t kern = c_.ssm_conv_kernel;

  prof_.Mark("lin_gemm_in");
  const GemvMultiProj ssm_in[4] = {
      {l.ssm_qkv.data, ToGemvType(l.ssm_qkv.type), l.ssm_qkv.rows,
       l.ssm_qkv.cols, gdn_qkv_},
      {l.ssm_gate.data, ToGemvType(l.ssm_gate.type), l.ssm_gate.rows,
       l.ssm_gate.cols, gdn_z_},
      {l.ssm_alpha.data, ToGemvType(l.ssm_alpha.type), l.ssm_alpha.rows,
       l.ssm_alpha.cols, gdn_alpha_},
      {l.ssm_beta.data, ToGemvType(l.ssm_beta.type), l.ssm_beta.rows,
       l.ssm_beta.cols, gdn_beta_},
  };
  if (!GemvMulti(ssm_in, 4U, x, nullptr)) {
    Gemv(l.ssm_qkv.data, ToGemvType(l.ssm_qkv.type), l.ssm_qkv.rows,
         l.ssm_qkv.cols, l.ssm_qkv.row_bytes, x, gdn_qkv_, nullptr);
    Gemv(l.ssm_gate.data, ToGemvType(l.ssm_gate.type), l.ssm_gate.rows,
         l.ssm_gate.cols, l.ssm_gate.row_bytes, x, gdn_z_, nullptr);
    Gemv(l.ssm_alpha.data, ToGemvType(l.ssm_alpha.type), l.ssm_alpha.rows,
         l.ssm_alpha.cols, l.ssm_alpha.row_bytes, x, gdn_alpha_, nullptr);
    Gemv(l.ssm_beta.data, ToGemvType(l.ssm_beta.type), l.ssm_beta.rows,
         l.ssm_beta.cols, l.ssm_beta.row_bytes, x, gdn_beta_, nullptr);
  }

  prof_.Mark("lin_conv");
  const std::uint32_t v_heads = c_.ssm_num_v_heads;
  if (d <= 1024U && channels == (2U * c_.ssm_num_k_heads + v_heads) * d) {
    GdnConvNormQk(gdn_qkv_, l.ssm_conv1d.f32(), session.gdn_history_[il],
                  gdn_convolved_, gdn_qn_, gdn_kn_, channels, kern,
                  c_.ssm_num_k_heads, v_heads, d, c_.rms_eps, nullptr);
  } else {
    GdnConv(gdn_qkv_, l.ssm_conv1d.f32(), session.gdn_history_[il],
            gdn_convolved_, channels, kern, nullptr);
    prof_.Mark("lin_normqk");
    GdnNormQk(gdn_convolved_, gdn_qn_, gdn_kn_, c_.ssm_num_k_heads, d,
              c_.rms_eps, nullptr);
  }
  prof_.Mark("lin_delta");
  GdnDelta(gdn_qn_, gdn_kn_, gdn_convolved_ + 2 * key_dim, gdn_alpha_,
           gdn_beta_, l.ssm_a.f32(), l.ssm_dt.f32(), session.gdn_state_[il],
           gdn_attn_, c_.ssm_num_k_heads, c_.ssm_num_v_heads, d, nullptr);
  prof_.Mark("lin_outnorm");
  GdnOutNorm(gdn_attn_, gdn_z_, l.ssm_norm.f32(), c_.ssm_num_v_heads, d,
             c_.rms_eps, nullptr);
  prof_.Mark("lin_gemm_out");
  Gemv(l.ssm_out.data, ToGemvType(l.ssm_out.type), l.ssm_out.rows,
       l.ssm_out.cols, l.ssm_out.row_bytes, gdn_attn_, out, nullptr);
}

void Executor::Attention(const DeviceLayer& l, const float* x,
                         std::uint32_t pos, float* out, float* k_cache,
                         float* v_cache, void* k_cache_f16, void* v_cache_f16,
                         const std::uint32_t* pos_dev) {
  const std::uint32_t hd = c_.head_dim;
  const std::uint32_t nh = c_.num_heads;
  const std::uint32_t nkv = c_.num_kv_heads;

  prof_.Mark("attn_gemm_qkv");
  const GemvMultiProj qkv[3] = {
      {l.attn_q.data, ToGemvType(l.attn_q.type), l.attn_q.rows, l.attn_q.cols,
       gqa_qg_},
      {l.attn_k.data, ToGemvType(l.attn_k.type), l.attn_k.rows, l.attn_k.cols,
       gqa_k_},
      {l.attn_v.data, ToGemvType(l.attn_v.type), l.attn_v.rows, l.attn_v.cols,
       gqa_v_},
  };
  if (!GemvMulti(qkv, 3U, x, nullptr)) {
    Gemv(l.attn_q.data, ToGemvType(l.attn_q.type), l.attn_q.rows, l.attn_q.cols,
         l.attn_q.row_bytes, x, gqa_qg_, nullptr);
    Gemv(l.attn_k.data, ToGemvType(l.attn_k.type), l.attn_k.rows, l.attn_k.cols,
         l.attn_k.row_bytes, x, gqa_k_, nullptr);
    Gemv(l.attn_v.data, ToGemvType(l.attn_v.type), l.attn_v.rows, l.attn_v.cols,
         l.attn_v.row_bytes, x, gqa_v_, nullptr);
  }

  prof_.Mark("attn_rope_norm");
  const std::size_t kv_row = static_cast<std::size_t>(nkv) * hd;
  // Fast path: deinterleave + per-head RMSNorm + partial RoPE + KV-cache write
  // (FP32 planes and FP16 mirror) in one launch, bit-identical to the unfused
  // chain below. Falls back when the head shape is unsupported.
  if (!FusedQKNormRoPEKvWrite(gqa_qg_, gqa_k_, gqa_v_, l.attn_q_norm.f32(),
                              l.attn_k_norm.f32(), gqa_q_, gqa_gate_, gqa_k_,
                              k_cache, v_cache, k_cache_f16, v_cache_f16, pos,
                              nh, nkv, hd, c_.rotary_dim, c_.rope_theta,
                              c_.rms_eps, nullptr)) {
    SplitQGate(gqa_qg_, gqa_q_, gqa_gate_, nh, hd, 1, nullptr);
    RmsNormRows(gqa_q_, l.attn_q_norm.f32(), gqa_q_, nh, hd, c_.rms_eps,
                nullptr);
    RmsNormRows(gqa_k_, l.attn_k_norm.f32(), gqa_k_, nkv, hd, c_.rms_eps,
                nullptr);
    Rope(gqa_q_, pos_dev, 1, nh, hd, c_.rotary_dim, c_.rope_theta, nullptr);
    Rope(gqa_k_, pos_dev, 1, nkv, hd, c_.rotary_dim, c_.rope_theta, nullptr);
    (void)hipMemcpyAsync(k_cache + static_cast<std::size_t>(pos) * kv_row,
                         gqa_k_, kv_row * sizeof(float), hipMemcpyDeviceToDevice,
                         nullptr);
    (void)hipMemcpyAsync(v_cache + static_cast<std::size_t>(pos) * kv_row,
                         gqa_v_, kv_row * sizeof(float), hipMemcpyDeviceToDevice,
                         nullptr);
    KvCacheWriteF16(k_cache, v_cache, k_cache_f16, v_cache_f16, pos * kv_row,
                    kv_row, nullptr);
  }

  prof_.Mark("attn_core");

  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  AttentionDecode(gqa_q_, k_cache, v_cache, gqa_gate_, gqa_ctx_, gqa_scratch_,
                  gqa_part_, pos + 1, nh, nkv, hd, scale, nullptr);
  prof_.Mark("attn_gemm_out");
  Gemv(l.attn_out.data, ToGemvType(l.attn_out.type), l.attn_out.rows,
       l.attn_out.cols, l.attn_out.row_bytes, gqa_ctx_, out, nullptr);
}

void Executor::Moe(const DeviceLayer& l, const float* x, float* out) {
  prof_.Mark("moe_router");
  Gemv(l.router.data, ToGemvType(l.router.type), l.router.rows, l.router.cols,
       l.router.row_bytes, x, moe_logits_, nullptr);
  RouterTopK(moe_logits_, c_.num_experts, moe_ids_, moe_weights_, 1,
             c_.num_experts, c_.num_experts_used, nullptr);

  // One launch per stage over all selected experts; the ids never leave the
  // device, so the decode pipeline never stalls on a host round-trip. The
  // gate and up projections share one launch (identical per-row dots).
  prof_.Mark("moe_gateup");
  const bool pair_routed = l.ffn_up_exps.rows == l.ffn_gate_exps.rows &&
                           l.ffn_up_exps.cols == l.ffn_gate_exps.cols &&
                           l.ffn_up_exps.row_bytes == l.ffn_gate_exps.row_bytes;
  // Fast path: fold gate+up+SwiGLU into one launch (bit-identical to the pair
  // GEMV followed by Swiglu), so the routed activation stays in registers and
  // the separate Swiglu pass and its two global buffers disappear.
  if (!(pair_routed && ToGemvType(l.ffn_gate_exps.type) == GemvType::kQ8_0) ||
      !GemvGroupedSwiglu(l.ffn_gate_exps.data, l.ffn_up_exps.data,
                         ToGemvType(l.ffn_gate_exps.type),
                         l.ffn_gate_exps.row_bytes * l.ffn_gate_exps.rows,
                         moe_ids_, c_.num_experts_used, l.ffn_gate_exps.rows,
                         l.ffn_gate_exps.cols, x, 0U, moe_gate_, nullptr)) {
    if (!pair_routed ||
        !GemvGroupedPair(l.ffn_gate_exps.data, l.ffn_up_exps.data,
                         ToGemvType(l.ffn_gate_exps.type),
                         l.ffn_gate_exps.row_bytes * l.ffn_gate_exps.rows,
                         moe_ids_, c_.num_experts_used, l.ffn_gate_exps.rows,
                         l.ffn_gate_exps.cols, x, 0U, moe_gate_, moe_up_,
                         nullptr)) {
      GemvGrouped(l.ffn_gate_exps.data, ToGemvType(l.ffn_gate_exps.type),
                  l.ffn_gate_exps.row_bytes * l.ffn_gate_exps.rows, moe_ids_,
                  c_.num_experts_used, l.ffn_gate_exps.rows,
                  l.ffn_gate_exps.cols, x, 0U, moe_gate_, nullptr);
      GemvGrouped(l.ffn_up_exps.data, ToGemvType(l.ffn_up_exps.type),
                  l.ffn_up_exps.row_bytes * l.ffn_up_exps.rows, moe_ids_,
                  c_.num_experts_used, l.ffn_up_exps.rows, l.ffn_up_exps.cols, x,
                  0U, moe_up_, nullptr);
    }
    Swiglu(moe_gate_, moe_up_, c_.num_experts_used * c_.expert_ff, nullptr);
  }
  prof_.Mark("moe_routed_down");
  GemvGrouped(l.ffn_down_exps.data, ToGemvType(l.ffn_down_exps.type),
              l.ffn_down_exps.row_bytes * l.ffn_down_exps.rows, moe_ids_,
              c_.num_experts_used, l.ffn_down_exps.rows, l.ffn_down_exps.cols,
              moe_gate_, c_.expert_ff, moe_expert_out_, nullptr);

  prof_.Mark("moe_shared");
  const GemvMultiProj shared_in[3] = {
      {l.shexp_gate.data, ToGemvType(l.shexp_gate.type), l.shexp_gate.rows,
       l.shexp_gate.cols, moe_gate_},
      {l.shexp_up.data, ToGemvType(l.shexp_up.type), l.shexp_up.rows,
       l.shexp_up.cols, moe_up_},
      {l.shexp_gate_inp.data, ToGemvType(l.shexp_gate_inp.type),
       l.shexp_gate_inp.rows, l.shexp_gate_inp.cols, moe_shared_gate_},
  };
  if (!GemvMulti(shared_in, 3U, x, nullptr)) {
    Gemv(l.shexp_gate.data, ToGemvType(l.shexp_gate.type), l.shexp_gate.rows,
         l.shexp_gate.cols, l.shexp_gate.row_bytes, x, moe_gate_, nullptr);
    Gemv(l.shexp_up.data, ToGemvType(l.shexp_up.type), l.shexp_up.rows,
         l.shexp_up.cols, l.shexp_up.row_bytes, x, moe_up_, nullptr);
    Gemv(l.shexp_gate_inp.data, ToGemvType(l.shexp_gate_inp.type),
         l.shexp_gate_inp.rows, l.shexp_gate_inp.cols,
         l.shexp_gate_inp.row_bytes, x, moe_shared_gate_, nullptr);
  }
  Swiglu(moe_gate_, moe_up_, c_.shared_expert_ff, nullptr);
  Gemv(l.shexp_down.data, ToGemvType(l.shexp_down.type), l.shexp_down.rows,
       l.shexp_down.cols, l.shexp_down.row_bytes, moe_gate_, moe_shared_down_,
       nullptr);
  prof_.Mark("moe_epilogue");
  MoeEpilogue(moe_expert_out_, moe_weights_, moe_shared_down_, moe_shared_gate_,
              1, out, 1, c_.num_experts_used, c_.hidden_size, nullptr);
}

void Executor::MoeBatch(const DeviceLayer& l, const float* x, float* out,
                        std::uint32_t tokens) {
  const std::uint32_t experts = c_.num_experts;
  const std::uint32_t used = c_.num_experts_used;
  const std::uint32_t hidden = c_.hidden_size;
  const std::uint32_t expert_ff = c_.expert_ff;
  const std::uint32_t shared_ff = c_.shared_expert_ff;
  const std::uint32_t pairs = tokens * used;

  // Verify fast path: the grouped decode GEMVs over the tokens*used slots
  // read each selected expert's weights once and keep the ids on the device,
  // so the WMMA bucket route (16-row tiles) and its per-layer host
  // round-trip are both strictly worse at a handful of tokens.
  if (tokens <= kMaxVerifyRows) {
    prof_.Mark("moe_router");
    GemvRows(l.router.data, ToGemvType(l.router.type), tokens, l.router.rows,
             l.router.cols, l.router.row_bytes, x, hidden, pf_router_logits_,
             experts, nullptr);
    RouterTopK(pf_router_logits_, experts, pf_ids_, pf_weights_, tokens,
               experts, used, nullptr);
    DupRows(x, tokens, used, hidden, pf_x_dup_, nullptr);
    prof_.Mark("moe_gateup");
    const bool pair_routed =
        l.ffn_up_exps.rows == l.ffn_gate_exps.rows &&
        l.ffn_up_exps.cols == l.ffn_gate_exps.cols &&
        l.ffn_up_exps.row_bytes == l.ffn_gate_exps.row_bytes;
    if (!pair_routed ||
        !GemvGroupedPair(l.ffn_gate_exps.data, l.ffn_up_exps.data,
                         ToGemvType(l.ffn_gate_exps.type),
                         l.ffn_gate_exps.row_bytes * l.ffn_gate_exps.rows,
                         pf_ids_, pairs, l.ffn_gate_exps.rows,
                         l.ffn_gate_exps.cols, pf_x_dup_, hidden, pf_gate_,
                         pf_up_, nullptr)) {
      GemvGrouped(l.ffn_gate_exps.data, ToGemvType(l.ffn_gate_exps.type),
                  l.ffn_gate_exps.row_bytes * l.ffn_gate_exps.rows, pf_ids_,
                  pairs, l.ffn_gate_exps.rows, l.ffn_gate_exps.cols, pf_x_dup_,
                  hidden, pf_gate_, nullptr);
      GemvGrouped(l.ffn_up_exps.data, ToGemvType(l.ffn_up_exps.type),
                  l.ffn_up_exps.row_bytes * l.ffn_up_exps.rows, pf_ids_, pairs,
                  l.ffn_up_exps.rows, l.ffn_up_exps.cols, pf_x_dup_, hidden,
                  pf_up_, nullptr);
    }
    Swiglu(pf_gate_, pf_up_, static_cast<std::size_t>(pairs) * expert_ff,
           nullptr);
    prof_.Mark("moe_routed_down");
    GemvGrouped(l.ffn_down_exps.data, ToGemvType(l.ffn_down_exps.type),
                l.ffn_down_exps.row_bytes * l.ffn_down_exps.rows, pf_ids_,
                pairs, l.ffn_down_exps.rows, l.ffn_down_exps.cols, pf_gate_,
                expert_ff, pf_expert_out_, nullptr);
    prof_.Mark("moe_shared");
    GemvRows(l.shexp_gate.data, ToGemvType(l.shexp_gate.type), tokens,
             l.shexp_gate.rows, l.shexp_gate.cols, l.shexp_gate.row_bytes, x,
             hidden, pf_shared_gate_, l.shexp_gate.rows, nullptr);
    GemvRows(l.shexp_up.data, ToGemvType(l.shexp_up.type), tokens,
             l.shexp_up.rows, l.shexp_up.cols, l.shexp_up.row_bytes, x, hidden,
             pf_shared_up_, l.shexp_up.rows, nullptr);
    Swiglu(pf_shared_gate_, pf_shared_up_,
           static_cast<std::size_t>(tokens) * shared_ff, nullptr);
    GemvRows(l.shexp_down.data, ToGemvType(l.shexp_down.type), tokens,
             l.shexp_down.rows, l.shexp_down.cols, l.shexp_down.row_bytes,
             pf_shared_gate_, shared_ff, pf_shared_down_, hidden, nullptr);
    GemvRows(l.shexp_gate_inp.data, ToGemvType(l.shexp_gate_inp.type), tokens,
             l.shexp_gate_inp.rows, l.shexp_gate_inp.cols,
             l.shexp_gate_inp.row_bytes, x, hidden, pf_shared_gate_inp_, 1U,
             nullptr);
    prof_.Mark("moe_epilogue");
    MoeEpilogue(pf_expert_out_, pf_weights_, pf_shared_down_,
                pf_shared_gate_inp_, 1, out, tokens, used, hidden, nullptr);
    return;
  }

  // Router: [tokens][experts], then the batched top-k. The ids stay on device.
  prof_.Mark("moe_router");
  Gemm(l.router.data, ToGemvType(l.router.type), l.router.rows, l.router.cols,
       l.router.row_bytes, x, pf_router_logits_, tokens, nullptr);
  RouterTopK(pf_router_logits_, experts, pf_ids_, pf_weights_, tokens, experts,
             used, nullptr);

  // Routed experts: gate/up -> SwiGLU -> down, one row per (token, slot) pair.
  // The matrix-core F16 route (ported from Qwen3.8-Flash-Next) is ~55x faster
  // per MAC than the dp4a MMQ grouped path on gfx1151: the assignments are
  // compacted by expert into 16-row padded buckets, the token rows narrowed to
  // F16 once, then one WMMA GEMM per (expert, row tile). Expert encodings the
  // WMMA kernel does not decode (Q6_K, F16, BF16, F32) fall back to the dp4a
  // MMQ grouped GEMM below.
  ExpertCounts(pf_ids_, pf_expert_counts_, tokens, experts, used, nullptr);
  (void)hipMemcpy(pf_counts_host_.data(), pf_expert_counts_,
                  experts * sizeof(std::uint32_t), hipMemcpyDeviceToHost);
  // Per-tensor route: the WMMA matrix-core path decodes Q8_0, BF16 and the
  // Q4_K/Q5_K super-block expert weights (a Q4_K_XL artifact stores its routed
  // experts in the latter two). Each of gate/up/down is dispatched with its own
  // encoding; a layer with any other encoding (F16, F32) falls back to the dp4a
  // MMQ grouped GEMM below.
  const auto supported = [](core::GgmlType t) {
    return t == core::GgmlType::kQ8_0 || t == core::GgmlType::kBF16 ||
           t == core::GgmlType::kQ4_K || t == core::GgmlType::kQ5_K;
  };
  // The fused gate+up kernel only decodes Q8_0 and BF16, so a Q4_K/Q5_K layer
  // runs gate and up as two separate WMMA launches instead.
  const auto gated_supported = [](core::GgmlType t) {
    return t == core::GgmlType::kQ8_0 || t == core::GgmlType::kBF16;
  };
  const auto wtype = [](core::GgmlType t) {
    switch (t) {
      case core::GgmlType::kBF16:
        return WeightType::kBF16;
      case core::GgmlType::kQ4_K:
        return WeightType::kQ4_K;
      case core::GgmlType::kQ5_K:
        return WeightType::kQ5_K;
      default:
        return WeightType::kQ8_0;
    }
  };
  const bool wmma = supported(l.ffn_gate_exps.type) &&
                    supported(l.ffn_up_exps.type) &&
                    supported(l.ffn_down_exps.type) && hidden % 64 == 0 &&
                    expert_ff % 64 == 0;

  // The fused gate+up route pairs the two projections over a second tile map
  // once the chunk is large enough to amortise the wider tiles and gate/up
  // share a fused-decodable encoding. It writes F16 activation rows, so the
  // down projection (which reads F16) stays valid.
  const bool pair = wmma && tokens >= 1024 &&
                    gated_supported(l.ffn_gate_exps.type) &&
                    l.ffn_gate_exps.type == l.ffn_up_exps.type;
  std::uint32_t n_tiles = 0;
  std::uint32_t n_pair = 0;
  std::uint32_t pair_rows = 64;
  if (wmma) {
    // Build the (expert, row-tile) map: one entry per 64-row tile of each
    // expert's 16-padded bucket, packed as expert | (tile << 16).
    for (std::uint32_t e = 0; e < experts; ++e) {
      const std::uint32_t padded = (pf_counts_host_[e] + 15u) / 16u * 16u;
      for (std::uint32_t j = 0; j < (padded + 63u) / 64u; ++j) {
        pf_tiles_host_[n_tiles++] = static_cast<std::int32_t>(e | (j << 16));
      }
    }
    if (pair) {
      // Prefer 128-row paired tiles when they cut the launch to at most three
      // quarters of the 64-row tile count.
      std::uint32_t tiles_64 = 0;
      std::uint32_t tiles_128 = 0;
      for (std::uint32_t e = 0; e < experts; ++e) {
        const std::uint32_t padded = (pf_counts_host_[e] + 15u) / 16u * 16u;
        tiles_64 += (padded + 63u) / 64u;
        tiles_128 += (padded + 127u) / 128u;
      }
      pair_rows = tiles_128 * 4u <= tiles_64 * 3u ? 128u : 64u;
      for (std::uint32_t e = 0; e < experts; ++e) {
        const std::uint32_t padded = (pf_counts_host_[e] + 15u) / 16u * 16u;
        for (std::uint32_t j = 0; j * pair_rows < padded; ++j) {
          pf_tiles_host_[n_tiles + n_pair++] =
              static_cast<std::int32_t>(e | (j << 16));
        }
      }
    }
    if (n_tiles + n_pair > 0) {
      (void)hipMemcpy(pf_tiles_dev_, pf_tiles_host_.data(),
                      (n_tiles + n_pair) * sizeof(std::int32_t),
                      hipMemcpyHostToDevice);
    }
  } else {
    // Bound the routed MMQ column grid to the real largest expert bucket.
    // Without this the grid is sized by the full pair count (tokens*used), so
    // every expert launches ~pair/mmq_x column tiles that reserve shared memory
    // and exit, collapsing occupancy. The down projection routes the same pairs
    // with n_expert_used=1, so one set of hints covers gate, up and down.
    std::uint32_t max_rows = 0;
    for (std::uint32_t e = 0; e < experts; ++e) {
      max_rows = std::max(max_rows, pf_counts_host_[e]);
    }
    qfn_mmq_set_routed_max_expert_rows(
        static_cast<int>(std::max(1u, max_rows)));
    qfn_mmq_set_routed_tile_cols(qfn_mmq_routed_tile_cols_for_counts(
        pf_counts_host_.data(), static_cast<int>(experts)));
  }

  prof_.Mark("moe_compact");
  if (wmma) {
    RoutedCompact(pf_ids_, pf_expert_counts_, pf_pad_bounds_, pf_cursors_,
                  pf_rows_token_, pf_rows_slot_, tokens, used, experts,
                  nullptr);
    NarrowActivations(x, pf_x_half_, false,
                      static_cast<std::size_t>(tokens) * hidden, nullptr);
    const auto* x_half = static_cast<const __half*>(pf_x_half_);
    auto* up_half = static_cast<__half*>(pf_up_half_);
    prof_.Mark("moe_gate_up");
    if (pair) {
      // Fused gate+up -> F16 up_half in one launch over the paired tile map
      // (stored after the 64-row map). Bit-identical to the two launches below.
      (void)RoutedGatedF16Gemm(
          l.ffn_gate_exps.data, l.ffn_up_exps.data, wtype(l.ffn_gate_exps.type),
          x_half, pf_tiles_dev_ + n_tiles, n_pair, pair_rows, pf_pad_bounds_,
          pf_rows_token_, pf_rows_slot_, up_half, expert_ff, hidden, nullptr);
    } else if (n_tiles > 0) {
      // gate -> F32 pf_gate_ (indexed by slot).
      (void)RoutedF16Gemm(l.ffn_gate_exps.data, wtype(l.ffn_gate_exps.type),
                          x_half, pf_tiles_dev_, n_tiles, 64, pf_pad_bounds_,
                          pf_rows_token_, pf_rows_slot_, nullptr, pf_gate_,
                          nullptr, expert_ff, hidden, nullptr);
      // up -> F16 up_half with the SwiGLU gate folded in from pf_gate_.
      (void)RoutedF16Gemm(l.ffn_up_exps.data, wtype(l.ffn_up_exps.type), x_half,
                          pf_tiles_dev_, n_tiles, 64, pf_pad_bounds_,
                          pf_rows_token_, pf_rows_slot_, pf_gate_, nullptr,
                          up_half, expert_ff, hidden, nullptr);
    }
    prof_.Mark("moe_routed_down");
    if (n_tiles > 0) {
      // down reads the F16 up rows (indexed by slot) and scatters F32 to
      // pf_expert_out_ (indexed by slot), matching MoeEpilogue's layout.
      (void)RoutedF16Gemm(l.ffn_down_exps.data, wtype(l.ffn_down_exps.type),
                          up_half, pf_tiles_dev_, n_tiles, 64, pf_pad_bounds_,
                          pf_rows_slot_, pf_rows_slot_, nullptr, pf_expert_out_,
                          nullptr, hidden, expert_ff, nullptr);
    }
  } else {
    prof_.Mark("dp4a_gateup");
    GemmMoe(l.ffn_gate_exps.data, ToGemvType(l.ffn_gate_exps.type),
            l.ffn_gate_exps.rows, l.ffn_gate_exps.cols,
            l.ffn_gate_exps.row_bytes, x, pf_ids_, pf_gate_, tokens, experts,
            used, nullptr);
    GemmMoe(l.ffn_up_exps.data, ToGemvType(l.ffn_up_exps.type),
            l.ffn_up_exps.rows, l.ffn_up_exps.cols, l.ffn_up_exps.row_bytes, x,
            pf_ids_, pf_up_, tokens, experts, used, nullptr);
    Swiglu(pf_gate_, pf_up_, static_cast<std::size_t>(pairs) * expert_ff,
           nullptr);
    prof_.Mark("dp4a_down");
    GemmMoe(l.ffn_down_exps.data, ToGemvType(l.ffn_down_exps.type),
            l.ffn_down_exps.rows, l.ffn_down_exps.cols,
            l.ffn_down_exps.row_bytes, pf_gate_, pf_ids_, pf_expert_out_, pairs,
            experts, 1, nullptr);
  }

  // Shared expert: a single dense projection, batched over the tokens.
  prof_.Mark("moe_shared");
  Gemm(l.shexp_gate.data, ToGemvType(l.shexp_gate.type), l.shexp_gate.rows,
       l.shexp_gate.cols, l.shexp_gate.row_bytes, x, pf_shared_gate_, tokens,
       nullptr);
  Gemm(l.shexp_up.data, ToGemvType(l.shexp_up.type), l.shexp_up.rows,
       l.shexp_up.cols, l.shexp_up.row_bytes, x, pf_shared_up_, tokens,
       nullptr);
  Swiglu(pf_shared_gate_, pf_shared_up_,
         static_cast<std::size_t>(tokens) * shared_ff, nullptr);
  Gemm(l.shexp_down.data, ToGemvType(l.shexp_down.type), l.shexp_down.rows,
       l.shexp_down.cols, l.shexp_down.row_bytes, pf_shared_gate_,
       pf_shared_down_, tokens, nullptr);
  Gemm(l.shexp_gate_inp.data, ToGemvType(l.shexp_gate_inp.type),
       l.shexp_gate_inp.rows, l.shexp_gate_inp.cols, l.shexp_gate_inp.row_bytes,
       x, pf_shared_gate_inp_, tokens, nullptr);

  prof_.Mark("moe_epilogue");
  MoeEpilogue(pf_expert_out_, pf_weights_, pf_shared_down_, pf_shared_gate_inp_,
              1, out, tokens, used, hidden, nullptr);
}

void Executor::LinearAttentionBatch(Session& session, const DeviceLayer& l,
                                    std::uint32_t il, const float* x,
                                    float* out, std::uint32_t tokens,
                                    float* state_snap, float* hist_snap) {
  const std::uint32_t channels = c_.SsmConvChannels();
  const std::uint32_t d = c_.ssm_head_dim;
  const std::uint32_t kern = c_.ssm_conv_kernel;

  // Small verify passes (up to kMaxVerifyRows rows) use the shared-weight
  // multi-row GEMV; larger chunks use the tiled GEMM tier. Both read `in`
  // rows at stride `cols` and write contiguous [tokens][rows] outputs.
  const auto proj = [&](const DeviceTensor& t, const float* in, float* out) {
    if (tokens <= kMaxVerifyRows) {
      GemvRows(t.data, ToGemvType(t.type), tokens, t.rows, t.cols, t.row_bytes,
               in, t.cols, out, t.rows, nullptr);
    } else {
      Gemm(t.data, ToGemvType(t.type), t.rows, t.cols, t.row_bytes, in, out,
           tokens, nullptr);
    }
  };

  prof_.Mark("lin_gemm_in");
  proj(l.ssm_qkv, x, pf_qkv_);
  proj(l.ssm_gate, x, pf_z_);
  proj(l.ssm_alpha, x, pf_alpha_);
  proj(l.ssm_beta, x, pf_beta_);

  prof_.Mark("lin_conv_normqk");
  GdnConvNormQkPrefill(pf_qkv_, l.ssm_conv1d.f32(), session.gdn_history_[il],
                       pf_convolved_, pf_qn_, pf_kn_, tokens, channels, kern,
                       c_.ssm_num_k_heads, c_.ssm_num_v_heads, d, c_.rms_eps,
                       nullptr);
  prof_.Mark("lin_delta");
  GdnPrep(pf_alpha_, pf_beta_, l.ssm_a.f32(), l.ssm_dt.f32(), pf_qn_, pf_kn_,
          pf_alpha_pre_, pf_beta_pre_, pf_kq_pre_, tokens, c_.ssm_num_k_heads,
          c_.ssm_num_v_heads, d, nullptr);
  GdnDeltaLoop(pf_qn_, pf_kn_, pf_convolved_, pf_alpha_, pf_beta_,
               l.ssm_a.f32(), l.ssm_dt.f32(), pf_alpha_pre_, pf_beta_pre_,
               pf_kq_pre_, session.gdn_state_[il], pf_gdn_attn_, tokens,
               c_.ssm_num_k_heads, c_.ssm_num_v_heads, d, channels, state_snap,
               state_snap != nullptr ? tokens - 1 : 0, nullptr);
  prof_.Mark("lin_outnorm");
  GdnOutNormPrefill(pf_gdn_attn_, pf_z_, l.ssm_norm.f32(), tokens,
                    c_.ssm_num_v_heads, d, c_.rms_eps, nullptr);
  prof_.Mark("lin_gemm_out");
  proj(l.ssm_out, pf_gdn_attn_, out);

  // Advance the rolling conv history past the chunk (reading the pre-chunk
  // history, writing a disjoint buffer) and publish it for the next chunk.
  // Snapshot the history after each of the first tokens-1 rows when a verify
  // rollback may need to rewind to any prefix of the block.
  prof_.Mark("lin_hist");
  if (hist_snap != nullptr) {
    const std::size_t hist_row = static_cast<std::size_t>(kern - 1U) * channels;
    for (std::uint32_t t = 1; t < tokens; ++t) {
      GdnHistoryUpdate(pf_qkv_, session.gdn_history_[il],
                       hist_snap + (t - 1U) * hist_row, t, channels, kern,
                       nullptr);
    }
  }
  GdnHistoryUpdate(pf_qkv_, session.gdn_history_[il], pf_hist_new_, tokens,
                   channels, kern, nullptr);
  (void)hipMemcpyAsync(
      session.gdn_history_[il], pf_hist_new_,
      static_cast<std::size_t>(kern - 1) * channels * sizeof(float),
      hipMemcpyDeviceToDevice, nullptr);
}

void Executor::AttentionBatch(const DeviceLayer& l, const float* x,
                              std::uint32_t start, float* out, float* k_cache,
                              float* v_cache, void* k_cache_f16,
                              void* v_cache_f16, const std::uint32_t* pos_dev,
                              std::uint32_t tokens) {
  const std::uint32_t hd = c_.head_dim;
  const std::uint32_t nh = c_.num_heads;
  const std::uint32_t nkv = c_.num_kv_heads;
  const std::size_t kv_row = static_cast<std::size_t>(nkv) * hd;

  // Small verify passes (up to kMaxVerifyRows rows) use the shared-weight
  // multi-row GEMV; larger chunks use the tiled GEMM tier.
  const auto proj = [&](const DeviceTensor& t, const float* in, float* out) {
    if (tokens <= kMaxVerifyRows) {
      GemvRows(t.data, ToGemvType(t.type), tokens, t.rows, t.cols, t.row_bytes,
               in, t.cols, out, t.rows, nullptr);
    } else {
      Gemm(t.data, ToGemvType(t.type), t.rows, t.cols, t.row_bytes, in, out,
           tokens, nullptr);
    }
  };

  prof_.Mark("attn_gemm_qkv");
  proj(l.attn_q, x, pf_qg_);
  proj(l.attn_k, x, pf_k_);
  proj(l.attn_v, x, pf_v_);

  prof_.Mark("attn_splitqgate");
  SplitQGate(pf_qg_, pf_q_, pf_qgate_, nh, hd, tokens, nullptr);
  prof_.Mark("attn_rope_norm");
  RmsNormRows(pf_q_, l.attn_q_norm.f32(), pf_q_, tokens * nh, hd, c_.rms_eps,
              nullptr);
  RmsNormRows(pf_k_, l.attn_k_norm.f32(), pf_k_, tokens * nkv, hd, c_.rms_eps,
              nullptr);
  Rope(pf_q_, pos_dev, tokens, nh, hd, c_.rotary_dim, c_.rope_theta, nullptr);
  Rope(pf_k_, pos_dev, tokens, nkv, hd, c_.rotary_dim, c_.rope_theta, nullptr);

  // Publish the chunk's rotated keys/values at their absolute positions, then
  // read the whole prefix back causally.
  prof_.Mark("attn_kv_write");
  (void)hipMemcpyAsync(k_cache + static_cast<std::size_t>(start) * kv_row,
                       pf_k_, tokens * kv_row * sizeof(float),
                       hipMemcpyDeviceToDevice, nullptr);
  (void)hipMemcpyAsync(v_cache + static_cast<std::size_t>(start) * kv_row,
                       pf_v_, tokens * kv_row * sizeof(float),
                       hipMemcpyDeviceToDevice, nullptr);
  KvCacheWriteF16(k_cache, v_cache, k_cache_f16, v_cache_f16, start * kv_row,
                  tokens * kv_row, nullptr);

  prof_.Mark("attn_core");
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  if (tokens <= kMaxVerifyRows) {
    AttentionDecodeRows(pf_q_, k_cache, v_cache, pf_qgate_, pf_ctx_, pf_part2_,
                        start + 1U, tokens, nh, nkv, hd, scale, nullptr);
  } else {
    AttentionPrefill(pf_q_, k_cache, v_cache, k_cache_f16, v_cache_f16,
                     pf_qgate_, pf_ctx_, start, tokens, nh, nkv, hd, scale,
                     nullptr, attn_window_, attn_sink_);
  }
  prof_.Mark("attn_gemm_out");
  proj(l.attn_out, pf_ctx_, out);
}

bool Executor::Prefill(Session& session, const std::int32_t* tokens,
                       std::uint32_t count, std::string* error_msg) {
  if (tokens == nullptr || count == 0) {
    if (error_msg != nullptr) {
      *error_msg = "no tokens to prefill";
    }
    return false;
  }
  for (std::uint32_t i = 0; i < count; ++i) {
    if (tokens[i] < 0 ||
        static_cast<std::uint32_t>(tokens[i]) >= c_.vocab_size) {
      if (error_msg != nullptr) {
        *error_msg = "token id out of range";
      }
      return false;
    }
  }
  if (count > session.max_context_ - session.position_) {
    if (error_msg != nullptr) {
      *error_msg = "context length exceeded";
    }
    return false;
  }

  const std::uint32_t hidden = c_.hidden_size;
  prof_.Reset();
  std::uint32_t done = 0;
  std::uint32_t last_rows = 0;
  while (done < count) {
    const std::uint32_t rows =
        (count - done) < prefill_chunk_ ? (count - done) : prefill_chunk_;
    const std::uint32_t start = session.position_ + done;

    std::vector<std::uint32_t> host_pos(rows);
    for (std::uint32_t t = 0; t < rows; ++t) {
      host_pos[t] = start + t;
    }
    prof_.Mark("embed");
    (void)hipMemcpyAsync(pf_pos_, host_pos.data(), rows * sizeof(std::uint32_t),
                         hipMemcpyHostToDevice, nullptr);

    for (std::uint32_t t = 0; t < rows; ++t) {
      EmbedRow(model_.token_embd().data, ToGemvType(model_.token_embd().type),
               static_cast<std::uint32_t>(tokens[done + t]), hidden,
               pf_x_ + static_cast<std::size_t>(t) * hidden, nullptr);
    }

    for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
      const DeviceLayer& l = model_.layers()[il];
      prof_.Mark("rmsnorm");
      RmsNormRows(pf_x_, l.attn_norm.f32(), pf_normed_, rows, hidden,
                  c_.rms_eps, nullptr);
      if (c_.IsLinearLayer(il)) {
        LinearAttentionBatch(session, l, il, pf_normed_, pf_attn_, rows);
      } else {
        AttentionBatch(l, pf_normed_, start, pf_attn_, session.k_cache_[il],
                       session.v_cache_[il], session.k_cache_f16_[il],
                       session.v_cache_f16_[il], pf_pos_, rows);
      }
      prof_.Mark("add");
      Add(pf_x_, pf_attn_, static_cast<std::size_t>(rows) * hidden, nullptr);
      prof_.Mark("rmsnorm");
      RmsNormRows(pf_x_, l.post_attention_norm.f32(), pf_normed_, rows, hidden,
                  c_.rms_eps, nullptr);
      MoeBatch(l, pf_normed_, pf_ffn_, rows);
      prof_.Mark("add");
      Add(pf_x_, pf_ffn_, static_cast<std::size_t>(rows) * hidden, nullptr);
    }

    // Fill the MTP block's caches over this chunk so the first draft after
    // prefill reads a prefix aligned with the trunk. Norming the residual
    // yields the block's hidden inputs; the last row also drives the logits
    // stage below (MtpPrefillChunk reuses pf_x_ for its own residual).
    if (model_.has_mtp()) {
      RmsNormRows(pf_x_, model_.output_norm().f32(), pf_normed_, rows, hidden,
                  c_.rms_eps, nullptr);
      (void)hipMemcpyAsync(x_, pf_normed_ + (rows - 1) * hidden,
                           hidden * sizeof(float), hipMemcpyDeviceToDevice,
                           nullptr);
      // The draft cache is only consumed when this session decodes with MTP.
      // A plain prefill leaves mtp_enabled_ false so the draft's attention
      // never runs; the last-row hidden above still feeds the logits stage.
      if (session.mtp_enabled_) {
        MtpPrefillChunk(session, tokens + done, rows, start, pf_normed_);
      }
    }
    done += rows;
    last_rows = rows;
  }

  // The final token's hidden state drives the output norm, h_out and logits.
  prof_.Mark("output");
  if (!model_.has_mtp()) {
    const float* last =
        pf_x_ + static_cast<std::size_t>(last_rows - 1) * hidden;
    RmsNormRows(last, model_.output_norm().f32(), x_, 1, hidden, c_.rms_eps,
                nullptr);
  }
  (void)hipMemcpy(h_out_, x_, hidden * sizeof(float), hipMemcpyDeviceToDevice);
  Gemv(model_.output().data, ToGemvType(model_.output().type),
       model_.output().rows, model_.output().cols, model_.output().row_bytes,
       x_, logits_, nullptr);
  session.position_ += count;
  if (prof_.enabled()) {
    char header[128];
    std::snprintf(header, sizeof(header), "prefill %u tokens, chunk %u", count,
                  prefill_chunk_);
    prof_.Report(header);
  }
  return true;
}

bool Executor::Step(Session& session, std::int32_t token,
                    std::string* error_msg) {
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "token id out of range";
    }
    return false;
  }
  if (session.position_ >= session.max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "context length exceeded";
    }
    return false;
  }

  pos_host_ = session.position_;
  if (prof_.enabled() && decode_steps_ % 32 == 0) {
    prof_.Reset();
  }
  (void)hipMemcpy(pos_dev_, &pos_host_, sizeof(std::uint32_t),
                  hipMemcpyHostToDevice);

  prof_.Mark("embed");
  EmbedRow(model_.token_embd().data, ToGemvType(model_.token_embd().type),
           static_cast<std::uint32_t>(token), c_.hidden_size, x_, nullptr);

  if (c_.num_layers > 0) {
    RmsNormRows(x_, model_.layers()[0].attn_norm.f32(), normed_, 1,
                c_.hidden_size, c_.rms_eps, nullptr);
  }
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const DeviceLayer& l = model_.layers()[il];
    prof_.Mark("attn");
    if (c_.IsLinearLayer(il)) {
      LinearAttention(session, l, il, normed_, attn_);
    } else {
      Attention(l, normed_, session.position_, attn_, session.k_cache_[il],
                session.v_cache_[il], session.k_cache_f16_[il],
                session.v_cache_f16_[il], pos_dev_);
    }
    prof_.Mark("add");
    FusedAddRmsNorm(x_, attn_, l.post_attention_norm.f32(), normed_,
                    c_.hidden_size, c_.rms_eps, nullptr);
    prof_.Mark("ffn");
    Moe(l, normed_, ffn_);
    prof_.Mark("add");
    if (il + 1 < c_.num_layers) {
      FusedAddRmsNorm(x_, ffn_, model_.layers()[il + 1].attn_norm.f32(),
                      normed_, c_.hidden_size, c_.rms_eps, nullptr);
    } else {
      FusedAddRmsNorm(x_, ffn_, model_.output_norm().f32(), x_, c_.hidden_size,
                      c_.rms_eps, nullptr);
    }
  }

  prof_.Mark("output");
  (void)hipMemcpy(h_out_, x_, c_.hidden_size * sizeof(float),
                  hipMemcpyDeviceToDevice);
  Gemv(model_.output().data, ToGemvType(model_.output().type),
       model_.output().rows, model_.output().cols, model_.output().row_bytes,
       x_, logits_, nullptr);
  // Close the GPU-side output stage here; the logits readback and host
  // sampling between steps get their own row instead of inflating lm_head.
  prof_.Mark("sample");
  ++session.position_;
  if (prof_.enabled()) {
    ++decode_steps_;
    if (decode_steps_ % 32 == 0) {
      prof_.Report("decode (32 steps)");
    }
  }
  return true;
}

void Executor::MtpPrefillChunk(Session& session, const std::int32_t* tokens,
                               std::uint32_t rows, std::uint32_t start,
                               const float* hidden_rows) {
  const std::uint32_t hidden = c_.hidden_size;
  const DeviceLayer& l = model_.mtp();

  // Row t of the block's input is [norm(e(token_t))][hidden of row t - 1],
  // with the previous chunk's last row (or zeros before the first chunk)
  // standing in for row -1.
  for (std::uint32_t t = 0; t < rows; ++t) {
    EmbedRow(model_.token_embd().data, ToGemvType(model_.token_embd().type),
             static_cast<std::uint32_t>(tokens[t]), hidden,
             pf_x_ + static_cast<std::size_t>(t) * hidden, nullptr);
  }
  RmsNormRows(pf_x_, l.nextn_enorm.f32(), pf_x_, rows, hidden, c_.rms_eps,
              nullptr);
  MtpConcat(pf_x_, hidden_rows, session.mtp_prev_hidden_, pf_mtp_concat_, rows,
            hidden, nullptr);
  Gemm(l.nextn_eh_proj.data, ToGemvType(l.nextn_eh_proj.type),
       l.nextn_eh_proj.rows, l.nextn_eh_proj.cols, l.nextn_eh_proj.row_bytes,
       pf_mtp_concat_, pf_mtp_cur_, rows, nullptr);

  RmsNormRows(pf_mtp_cur_, l.attn_norm.f32(), pf_normed_, rows, hidden,
              c_.rms_eps, nullptr);
  AttentionBatch(l, pf_normed_, start, pf_attn_, session.mtp_k_cache_,
                 session.mtp_v_cache_, session.mtp_k_cache_f16_,
                 session.mtp_v_cache_f16_, pf_pos_, rows);
  Add(pf_mtp_cur_, pf_attn_, static_cast<std::size_t>(rows) * hidden, nullptr);
  RmsNormRows(pf_mtp_cur_, l.post_attention_norm.f32(), pf_normed_, rows,
              hidden, c_.rms_eps, nullptr);
  MoeBatch(l, pf_normed_, pf_ffn_, rows);
  Add(pf_mtp_cur_, pf_ffn_, static_cast<std::size_t>(rows) * hidden, nullptr);

  (void)hipMemcpyAsync(
      session.mtp_prev_hidden_, hidden_rows + (rows - 1) * hidden,
      hidden * sizeof(float), hipMemcpyDeviceToDevice, nullptr);
  session.mtp_position_ += rows;
}

bool Executor::MtpForward(Session& session, std::int32_t token,
                          const float* hidden, bool with_logits,
                          std::string* error_msg) {
  if (!model_.has_mtp()) {
    if (error_msg != nullptr) {
      *error_msg = "model has no MTP block";
    }
    return false;
  }
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "token id out of range";
    }
    return false;
  }
  if (session.mtp_position_ >= session.max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "context length exceeded";
    }
    return false;
  }

  const DeviceLayer& l = model_.mtp();
  prof_.Mark("mtp");
  mtp_pos_host_ = session.mtp_position_;
  (void)hipMemcpy(mtp_pos_dev_, &mtp_pos_host_, sizeof(std::uint32_t),
                  hipMemcpyHostToDevice);

  EmbedRow(model_.token_embd().data, ToGemvType(model_.token_embd().type),
           static_cast<std::uint32_t>(token), c_.hidden_size, mtp_e_, nullptr);
  RmsNormRows(mtp_e_, l.nextn_enorm.f32(), mtp_e_, 1, c_.hidden_size,
              c_.rms_eps, nullptr);
  RmsNormRows(hidden, l.nextn_hnorm.f32(), mtp_h_, 1, c_.hidden_size,
              c_.rms_eps, nullptr);
  (void)hipMemcpy(mtp_concat_, mtp_e_, c_.hidden_size * sizeof(float),
                  hipMemcpyDeviceToDevice);
  (void)hipMemcpy(mtp_concat_ + c_.hidden_size, mtp_h_,
                  c_.hidden_size * sizeof(float), hipMemcpyDeviceToDevice);
  Gemv(l.nextn_eh_proj.data, ToGemvType(l.nextn_eh_proj.type),
       l.nextn_eh_proj.rows, l.nextn_eh_proj.cols, l.nextn_eh_proj.row_bytes,
       mtp_concat_, mtp_cur_, nullptr);

  RmsNormRows(mtp_cur_, l.attn_norm.f32(), normed_, 1, c_.hidden_size,
              c_.rms_eps, nullptr);
  Attention(l, normed_, session.mtp_position_, attn_, session.mtp_k_cache_,
            session.mtp_v_cache_, session.mtp_k_cache_f16_,
            session.mtp_v_cache_f16_, mtp_pos_dev_);
  Add(mtp_cur_, attn_, c_.hidden_size, nullptr);
  RmsNormRows(mtp_cur_, l.post_attention_norm.f32(), normed_, 1, c_.hidden_size,
              c_.rms_eps, nullptr);
  Moe(l, normed_, ffn_);
  Add(mtp_cur_, ffn_, c_.hidden_size, nullptr);
  if (with_logits) {
    // Keep the unnormalized output hidden for the next draft in the chain;
    // the shared head norm below normalizes mtp_cur_ in place.
    (void)hipMemcpyAsync(mtp_chain_, mtp_cur_, c_.hidden_size * sizeof(float),
                         hipMemcpyDeviceToDevice, nullptr);
    RmsNormRows(mtp_cur_, l.nextn_shared_head_norm.f32(), mtp_cur_, 1,
                c_.hidden_size, c_.rms_eps, nullptr);
    Gemv(model_.output().data, ToGemvType(model_.output().type),
         model_.output().rows, model_.output().cols, model_.output().row_bytes,
         mtp_cur_, mtp_logits_, nullptr);
  }
  ++session.mtp_position_;
  return true;
}

bool Executor::MtpStep(Session& session, std::int32_t token,
                       std::string* error_msg) {
  return MtpForward(session, token, h_out_, true, error_msg);
}

bool Executor::MtpDraft(Session& session, std::int32_t token,
                        std::string* error_msg) {
  return MtpForward(session, token, mtp_chain_, true, error_msg);
}

bool Executor::MtpAdvance(Session& session, std::int32_t token,
                          const float* hidden, std::string* error_msg) {
  return MtpForward(session, token, hidden, false, error_msg);
}

bool Executor::Verify(Session& session, std::int32_t t0,
                      const std::int32_t* drafts, std::uint32_t k,
                      std::string* error_msg) {
  if (!model_.has_mtp()) {
    if (error_msg != nullptr) {
      *error_msg = "model has no MTP block";
    }
    return false;
  }
  if (drafts == nullptr || k == 0U || k + 1U > kMaxVerifyRows) {
    if (error_msg != nullptr) {
      *error_msg = "invalid speculative draft count";
    }
    return false;
  }
  if (t0 < 0 || static_cast<std::uint32_t>(t0) >= c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "token id out of range";
    }
    return false;
  }
  for (std::uint32_t i = 0; i < k; ++i) {
    if (drafts[i] < 0 ||
        static_cast<std::uint32_t>(drafts[i]) >= c_.vocab_size) {
      if (error_msg != nullptr) {
        *error_msg = "token id out of range";
      }
      return false;
    }
  }
  const std::uint32_t rows = k + 1U;
  if (session.position_ + rows > session.max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "context length exceeded";
    }
    return false;
  }

  const std::uint32_t hidden = c_.hidden_size;
  if (prof_.enabled() && spec_rounds_ % 16 == 0) {
    prof_.Reset();
  }
  prof_.Mark("verify");
  session.verify_base_pos_ = session.position_;
  std::uint32_t host_pos[kMaxVerifyRows];
  for (std::uint32_t r = 0; r < rows; ++r) {
    host_pos[r] = session.position_ + r;
  }
  (void)hipMemcpyAsync(pf_pos_, host_pos, rows * sizeof(std::uint32_t),
                       hipMemcpyHostToDevice, nullptr);
  EmbedRow(model_.token_embd().data, ToGemvType(model_.token_embd().type),
           static_cast<std::uint32_t>(t0), hidden, pf_x_, nullptr);
  for (std::uint32_t i = 0; i < k; ++i) {
    EmbedRow(model_.token_embd().data, ToGemvType(model_.token_embd().type),
             static_cast<std::uint32_t>(drafts[i]), hidden,
             pf_x_ + static_cast<std::size_t>(i + 1U) * hidden, nullptr);
  }

  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const DeviceLayer& l = model_.layers()[il];
    RmsNormRows(pf_x_, l.attn_norm.f32(), pf_normed_, rows, hidden, c_.rms_eps,
                nullptr);
    if (c_.IsLinearLayer(il)) {
      // Snapshot the recurrent state and conv history after each of the
      // first rows so a rejected draft can rewind to any committed prefix.
      LinearAttentionBatch(session, l, il, pf_normed_, pf_attn_, rows,
                           session.gdn_state_snap_[il],
                           session.gdn_hist_snap_[il]);
    } else {
      AttentionBatch(l, pf_normed_, session.position_, pf_attn_,
                     session.k_cache_[il], session.v_cache_[il],
                     session.k_cache_f16_[il], session.v_cache_f16_[il],
                     pf_pos_, rows);
    }
    Add(pf_x_, pf_attn_, static_cast<std::size_t>(rows) * hidden, nullptr);
    RmsNormRows(pf_x_, l.post_attention_norm.f32(), pf_normed_, rows, hidden,
                c_.rms_eps, nullptr);
    MoeBatch(l, pf_normed_, pf_ffn_, rows);
    Add(pf_x_, pf_ffn_, static_cast<std::size_t>(rows) * hidden, nullptr);
  }

  prof_.Mark("output");
  RmsNormRows(pf_x_, model_.output_norm().f32(), verify_h_, rows, hidden,
              c_.rms_eps, nullptr);
  (void)hipMemcpyAsync(h_out_, verify_h_ + static_cast<std::size_t>(k) * hidden,
                       hidden * sizeof(float), hipMemcpyDeviceToDevice,
                       nullptr);
  GemvRows(model_.output().data, ToGemvType(model_.output().type), rows,
           model_.output().rows, model_.output().cols,
           model_.output().row_bytes, verify_h_, hidden, verify_logits_,
           model_.output().rows, nullptr);
  prof_.Mark("sample");
  session.position_ += rows;
  if (prof_.enabled()) {
    ++spec_rounds_;
    if (spec_rounds_ % 16 == 0) {
      prof_.Report("spec decode (16 rounds)");
    }
  }
  return true;
}

void Executor::RollbackVerify(Session& session, std::uint32_t keep) {
  const std::size_t hidden = c_.hidden_size;
  const std::size_t state_elems = static_cast<std::size_t>(c_.ssm_num_v_heads) *
                                  c_.ssm_head_dim * c_.ssm_head_dim;
  const std::size_t history_elems =
      static_cast<std::size_t>(c_.ssm_conv_kernel - 1) * c_.SsmConvChannels();
  // Commit the first `keep` rows: restore the recurrent state and conv
  // history from the snapshot after row keep - 1. Their KV entries and
  // hidden state were already published by Verify.
  const std::size_t snap = static_cast<std::size_t>(keep - 1U);
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    if (!c_.IsLinearLayer(il)) {
      continue;
    }
    (void)hipMemcpyAsync(session.gdn_state_[il],
                         session.gdn_state_snap_[il] + snap * state_elems,
                         state_elems * sizeof(float), hipMemcpyDeviceToDevice,
                         nullptr);
    (void)hipMemcpyAsync(session.gdn_history_[il],
                         session.gdn_hist_snap_[il] + snap * history_elems,
                         history_elems * sizeof(float), hipMemcpyDeviceToDevice,
                         nullptr);
  }
  session.position_ = session.verify_base_pos_ + keep;
  // The draft cache already holds an entry per chained draft, so it is
  // aligned with the committed prefix once the trunk rewinds to it.
  session.mtp_position_ = session.verify_base_pos_ + keep;
  (void)hipMemcpyAsync(h_out_, verify_h_ + snap * hidden,
                       hidden * sizeof(float), hipMemcpyDeviceToDevice,
                       nullptr);
}

}  // namespace gufo::models::qwen36_a3b::rocm