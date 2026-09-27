#include "src/models/qwen36_a3b/kernels/rocm/executor.hpp"

#include <cmath>
#include <cstring>
#include <mutex>

#include "src/models/qwen36_a3b/kernels/rocm/gemv.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/gemm.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/kernels.hpp"
#include "qfn_mmq.h"

namespace gufo::models::qwen36_a3b::rocm {
namespace {

/// Maps the artifact's GGUF type onto the GEMV tier's row format. Only the
/// three formats this model stores appear; anything else is a load error.
[[nodiscard]] GemvType ToGemvType(core::GgmlType t) noexcept {
  switch (t) {
    case core::GgmlType::kQ8_0:
      return GemvType::kQ8_0;
    case core::GgmlType::kF32:
      return GemvType::kF32;
    case core::GgmlType::kBF16:
      return GemvType::kBF16;
    default:
      return GemvType::kF32;  // unreachable: Create rejects other types
  }
}

[[nodiscard]] bool SupportedType(core::GgmlType t) noexcept {
  return t == core::GgmlType::kQ8_0 || t == core::GgmlType::kF32 ||
         t == core::GgmlType::kBF16;
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
        !check(l.ffn_gate_exps, base) || !check(l.ffn_up_exps, base) ||
        !check(l.ffn_down_exps, base) || !check(l.shexp_gate_inp, base) ||
        !check(l.shexp_gate, base) || !check(l.shexp_up, base) ||
        !check(l.shexp_down, base) || !check(l.nextn_enorm, base) ||
        !check(l.nextn_hnorm, base) || !check(l.nextn_eh_proj, base) ||
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

std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           std::uint32_t max_context,
                                           std::string* error_msg) {
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
  e->max_context_ = max_context;
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

  // Mixture-of-experts scratch.
  e->moe_logits_ = e->AllocFloats(c.num_experts, error_msg);
  e->moe_ids_ = e->AllocInts(c.num_experts_used, error_msg);
  e->moe_weights_ = e->AllocFloats(c.num_experts_used, error_msg);
  e->moe_expert_out_ = e->AllocFloats(
      static_cast<std::size_t>(c.num_experts_used) * c.hidden_size, error_msg);
  e->moe_gate_ = e->AllocFloats(c.expert_ff, error_msg);
  e->moe_up_ = e->AllocFloats(c.expert_ff, error_msg);
  e->moe_shared_down_ = e->AllocFloats(c.hidden_size, error_msg);
  e->moe_shared_gate_ = e->AllocFloats(1, error_msg);
  e->moe_ids_host_.assign(c.num_experts_used, 0);

  // Batched prefill scratch. The chunk is capped at 2048 tokens (and at the
  // context length) to bound the (token, slot) MoE buffers near ~230 MB.
  const std::uint32_t chunk =
      max_context < 2048u ? max_context : 2048u;
  e->prefill_chunk_ = chunk;
  const std::size_t pairs =
      static_cast<std::size_t>(chunk) * c.num_experts_used;
  e->pf_router_logits_ = e->AllocFloats(
      static_cast<std::size_t>(chunk) * c.num_experts, error_msg);
  e->pf_ids_ = e->AllocInts(pairs, error_msg);
  e->pf_weights_ = e->AllocFloats(pairs, error_msg);
  e->pf_gate_ = e->AllocFloats(pairs * c.expert_ff, error_msg);
  e->pf_up_ = e->AllocFloats(pairs * c.expert_ff, error_msg);
  e->pf_expert_out_ = e->AllocFloats(pairs * c.hidden_size, error_msg);
  e->pf_shared_gate_ =
      e->AllocFloats(static_cast<std::size_t>(chunk) * c.shared_expert_ff,
                     error_msg);
  e->pf_shared_up_ =
      e->AllocFloats(static_cast<std::size_t>(chunk) * c.shared_expert_ff,
                     error_msg);
  e->pf_shared_down_ =
      e->AllocFloats(static_cast<std::size_t>(chunk) * c.hidden_size, error_msg);
  e->pf_shared_gate_inp_ = e->AllocFloats(chunk, error_msg);

  // Prefill residual stream and per-layer intermediates.
  e->pf_x_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c.hidden_size,
                            error_msg);
  e->pf_normed_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c.hidden_size,
                                 error_msg);
  e->pf_attn_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c.hidden_size,
                               error_msg);
  e->pf_ffn_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c.hidden_size,
                              error_msg);

  // Prefill Gated DeltaNet scratch.
  const std::size_t c_key = c.SsmKeyDim();
  const std::size_t c_val = c.SsmValueDim();
  const std::size_t c_chan = c.SsmConvChannels();
  e->pf_qkv_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_chan,
                              error_msg);
  e->pf_z_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_val, error_msg);
  e->pf_alpha_ =
      e->AllocFloats(static_cast<std::size_t>(chunk) * c.ssm_num_v_heads,
                     error_msg);
  e->pf_beta_ =
      e->AllocFloats(static_cast<std::size_t>(chunk) * c.ssm_num_v_heads,
                     error_msg);
  e->pf_convolved_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_chan,
                                    error_msg);
  e->pf_qn_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_key, error_msg);
  e->pf_kn_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_key, error_msg);
  e->pf_gdn_attn_ =
      e->AllocFloats(static_cast<std::size_t>(chunk) * c_val, error_msg);
  e->pf_hist_new_ = e->AllocFloats(
      static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c_chan, error_msg);

  // Prefill gated grouped-query attention scratch.
  const std::size_t c_q = c.AttentionQDim();
  const std::size_t c_kv = c.AttentionKvDim();
  e->pf_qg_ = e->AllocFloats(static_cast<std::size_t>(chunk) * 2 * c_q,
                             error_msg);
  e->pf_k_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_kv, error_msg);
  e->pf_v_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_kv, error_msg);
  e->pf_q_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_q, error_msg);
  e->pf_qgate_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_q, error_msg);
  e->pf_ctx_ = e->AllocFloats(static_cast<std::size_t>(chunk) * c_q, error_msg);
  e->pf_pos_ = e->AllocUints(chunk, error_msg);

  // MTP scratch.
  e->mtp_e_ = e->AllocFloats(c.hidden_size, error_msg);
  e->mtp_h_ = e->AllocFloats(c.hidden_size, error_msg);
  e->mtp_concat_ = e->AllocFloats(2 * c.hidden_size, error_msg);
  e->mtp_cur_ = e->AllocFloats(c.hidden_size, error_msg);

  // Recurrent state, per layer.
  const std::size_t state_elems = static_cast<std::size_t>(c.ssm_num_v_heads) *
                                  c.ssm_head_dim * c.ssm_head_dim;
  const std::size_t history_elems =
      static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
  const std::size_t kv_elems =
      static_cast<std::size_t>(c.num_kv_heads) * c.head_dim * max_context;
  e->gdn_state_.assign(c.num_layers, nullptr);
  e->gdn_history_.assign(c.num_layers, nullptr);
  e->k_cache_.assign(c.num_layers, nullptr);
  e->v_cache_.assign(c.num_layers, nullptr);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (c.IsLinearLayer(il)) {
      e->gdn_state_[il] = e->AllocFloats(state_elems, error_msg);
      e->gdn_history_[il] = e->AllocFloats(history_elems, error_msg);
    } else {
      e->k_cache_[il] = e->AllocFloats(kv_elems, error_msg);
      e->v_cache_[il] = e->AllocFloats(kv_elems, error_msg);
    }
  }
  if (model.has_mtp()) {
    e->mtp_k_cache_ = e->AllocFloats(kv_elems, error_msg);
    e->mtp_v_cache_ = e->AllocFloats(kv_elems, error_msg);
  }

  if (error_msg != nullptr && !error_msg->empty()) {
    return nullptr;
  }
  e->Reset();
  return e;
}

void Executor::Reset() {
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
    }
  }
  if (model_.has_mtp()) {
    (void)hipMemsetAsync(mtp_k_cache_, 0, kv_elems * sizeof(float), nullptr);
    (void)hipMemsetAsync(mtp_v_cache_, 0, kv_elems * sizeof(float), nullptr);
  }
}

void Executor::LinearAttention(const DeviceLayer& l, std::uint32_t il,
                               const float* x, float* out) {
  const std::uint32_t channels = c_.SsmConvChannels();
  const std::uint32_t key_dim = c_.SsmKeyDim();
  const std::uint32_t d = c_.ssm_head_dim;
  const std::uint32_t kern = c_.ssm_conv_kernel;

  Gemv(l.ssm_qkv.data, ToGemvType(l.ssm_qkv.type), l.ssm_qkv.rows,
       l.ssm_qkv.cols, l.ssm_qkv.row_bytes, x, gdn_qkv_, nullptr);
  Gemv(l.ssm_gate.data, ToGemvType(l.ssm_gate.type), l.ssm_gate.rows,
       l.ssm_gate.cols, l.ssm_gate.row_bytes, x, gdn_z_, nullptr);
  Gemv(l.ssm_alpha.data, ToGemvType(l.ssm_alpha.type), l.ssm_alpha.rows,
       l.ssm_alpha.cols, l.ssm_alpha.row_bytes, x, gdn_alpha_, nullptr);
  Gemv(l.ssm_beta.data, ToGemvType(l.ssm_beta.type), l.ssm_beta.rows,
       l.ssm_beta.cols, l.ssm_beta.row_bytes, x, gdn_beta_, nullptr);

  GdnConv(gdn_qkv_, l.ssm_conv1d.f32(), gdn_history_[il], gdn_convolved_,
          channels, kern, nullptr);
  GdnNormQk(gdn_convolved_, gdn_qn_, gdn_kn_, c_.ssm_num_k_heads, d, c_.rms_eps,
            nullptr);
  GdnDelta(gdn_qn_, gdn_kn_, gdn_convolved_ + 2 * key_dim, gdn_alpha_,
           gdn_beta_, l.ssm_a.f32(), l.ssm_dt.f32(), gdn_state_[il], gdn_attn_,
           c_.ssm_num_k_heads, c_.ssm_num_v_heads, d, nullptr);
  GdnOutNorm(gdn_attn_, gdn_z_, l.ssm_norm.f32(), c_.ssm_num_v_heads, d,
             c_.rms_eps, nullptr);
  Gemv(l.ssm_out.data, ToGemvType(l.ssm_out.type), l.ssm_out.rows,
       l.ssm_out.cols, l.ssm_out.row_bytes, gdn_attn_, out, nullptr);
}

void Executor::Attention(const DeviceLayer& l, const float* x,
                         std::uint32_t pos, float* out, float* k_cache,
                         float* v_cache, const std::uint32_t* pos_dev) {
  const std::uint32_t hd = c_.head_dim;
  const std::uint32_t nh = c_.num_heads;
  const std::uint32_t nkv = c_.num_kv_heads;

  Gemv(l.attn_q.data, ToGemvType(l.attn_q.type), l.attn_q.rows, l.attn_q.cols,
       l.attn_q.row_bytes, x, gqa_qg_, nullptr);
  Gemv(l.attn_k.data, ToGemvType(l.attn_k.type), l.attn_k.rows, l.attn_k.cols,
       l.attn_k.row_bytes, x, gqa_k_, nullptr);
  Gemv(l.attn_v.data, ToGemvType(l.attn_v.type), l.attn_v.rows, l.attn_v.cols,
       l.attn_v.row_bytes, x, gqa_v_, nullptr);

  SplitQGate(gqa_qg_, gqa_q_, gqa_gate_, nh, hd, nullptr);
  RmsNormRows(gqa_q_, l.attn_q_norm.f32(), gqa_q_, nh, hd, c_.rms_eps, nullptr);
  RmsNormRows(gqa_k_, l.attn_k_norm.f32(), gqa_k_, nkv, hd, c_.rms_eps,
              nullptr);
  Rope(gqa_q_, pos_dev, 1, nh, hd, c_.rotary_dim, c_.rope_theta, nullptr);
  Rope(gqa_k_, pos_dev, 1, nkv, hd, c_.rotary_dim, c_.rope_theta, nullptr);

  const std::size_t kv_row = static_cast<std::size_t>(nkv) * hd;
  (void)hipMemcpyAsync(k_cache + static_cast<std::size_t>(pos) * kv_row, gqa_k_,
                       kv_row * sizeof(float), hipMemcpyDeviceToDevice,
                       nullptr);
  (void)hipMemcpyAsync(v_cache + static_cast<std::size_t>(pos) * kv_row, gqa_v_,
                       kv_row * sizeof(float), hipMemcpyDeviceToDevice,
                       nullptr);

  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  AttentionDecode(gqa_q_, k_cache, v_cache, gqa_gate_, gqa_ctx_, gqa_scratch_,
                  pos + 1, nh, nkv, hd, scale, nullptr);
  Gemv(l.attn_out.data, ToGemvType(l.attn_out.type), l.attn_out.rows,
       l.attn_out.cols, l.attn_out.row_bytes, gqa_ctx_, out, nullptr);
}

void Executor::Moe(const DeviceLayer& l, const float* x, float* out) {
  Gemv(l.router.data, ToGemvType(l.router.type), l.router.rows, l.router.cols,
       l.router.row_bytes, x, moe_logits_, nullptr);
  RouterTopK(moe_logits_, c_.num_experts, moe_ids_, moe_weights_, 1,
             c_.num_experts, c_.num_experts_used, nullptr);
  (void)hipMemcpy(moe_ids_host_.data(), moe_ids_,
                  c_.num_experts_used * sizeof(std::int32_t),
                  hipMemcpyDeviceToHost);

  for (std::uint32_t s = 0; s < c_.num_experts_used; ++s) {
    const std::uint32_t e = static_cast<std::uint32_t>(moe_ids_host_[s]);
    Gemv(l.ffn_gate_exps.Expert(e), ToGemvType(l.ffn_gate_exps.type),
         l.ffn_gate_exps.rows, l.ffn_gate_exps.cols, l.ffn_gate_exps.row_bytes,
         x, moe_gate_, nullptr);
    Gemv(l.ffn_up_exps.Expert(e), ToGemvType(l.ffn_up_exps.type),
         l.ffn_up_exps.rows, l.ffn_up_exps.cols, l.ffn_up_exps.row_bytes, x,
         moe_up_, nullptr);
    Swiglu(moe_gate_, moe_up_, c_.expert_ff, nullptr);
    Gemv(l.ffn_down_exps.Expert(e), ToGemvType(l.ffn_down_exps.type),
         l.ffn_down_exps.rows, l.ffn_down_exps.cols, l.ffn_down_exps.row_bytes,
         moe_gate_,
         moe_expert_out_ + static_cast<std::size_t>(s) * c_.hidden_size,
         nullptr);
  }

  Gemv(l.shexp_gate.data, ToGemvType(l.shexp_gate.type), l.shexp_gate.rows,
       l.shexp_gate.cols, l.shexp_gate.row_bytes, x, moe_gate_, nullptr);
  Gemv(l.shexp_up.data, ToGemvType(l.shexp_up.type), l.shexp_up.rows,
       l.shexp_up.cols, l.shexp_up.row_bytes, x, moe_up_, nullptr);
  Swiglu(moe_gate_, moe_up_, c_.shared_expert_ff, nullptr);
  Gemv(l.shexp_down.data, ToGemvType(l.shexp_down.type), l.shexp_down.rows,
       l.shexp_down.cols, l.shexp_down.row_bytes, moe_gate_, moe_shared_down_,
       nullptr);
  Gemv(l.shexp_gate_inp.data, ToGemvType(l.shexp_gate_inp.type),
       l.shexp_gate_inp.rows, l.shexp_gate_inp.cols, l.shexp_gate_inp.row_bytes,
       x, moe_shared_gate_, nullptr);
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

  // Router: [tokens][experts], then the batched top-k. The ids stay on device.
  Gemm(l.router.data, ToGemvType(l.router.type), l.router.rows, l.router.cols,
       l.router.row_bytes, x, pf_router_logits_, tokens, nullptr);
  RouterTopK(pf_router_logits_, experts, pf_ids_, pf_weights_, tokens, experts,
             used, nullptr);

  // Routed experts: gate/up -> SwiGLU -> down, one row per (token, slot) pair.
  // The down projection treats each pair as its own row routed by that pair's
  // expert id, so it runs with n_expert_used = 1 over `pairs` rows.
  GemmMoe(l.ffn_gate_exps.data, ToGemvType(l.ffn_gate_exps.type),
          l.ffn_gate_exps.rows, l.ffn_gate_exps.cols,
          l.ffn_gate_exps.row_bytes, x, pf_ids_, pf_gate_, tokens, experts, used,
          nullptr);
  GemmMoe(l.ffn_up_exps.data, ToGemvType(l.ffn_up_exps.type),
          l.ffn_up_exps.rows, l.ffn_up_exps.cols, l.ffn_up_exps.row_bytes, x,
          pf_ids_, pf_up_, tokens, experts, used, nullptr);
  Swiglu(pf_gate_, pf_up_, static_cast<std::size_t>(pairs) * expert_ff,
         nullptr);
  GemmMoe(l.ffn_down_exps.data, ToGemvType(l.ffn_down_exps.type),
          l.ffn_down_exps.rows, l.ffn_down_exps.cols,
          l.ffn_down_exps.row_bytes, pf_gate_, pf_ids_, pf_expert_out_, pairs,
          experts, 1, nullptr);

  // Shared expert: a single dense projection, batched over the tokens.
  Gemm(l.shexp_gate.data, ToGemvType(l.shexp_gate.type), l.shexp_gate.rows,
       l.shexp_gate.cols, l.shexp_gate.row_bytes, x, pf_shared_gate_, tokens,
       nullptr);
  Gemm(l.shexp_up.data, ToGemvType(l.shexp_up.type), l.shexp_up.rows,
       l.shexp_up.cols, l.shexp_up.row_bytes, x, pf_shared_up_, tokens, nullptr);
  Swiglu(pf_shared_gate_, pf_shared_up_,
         static_cast<std::size_t>(tokens) * shared_ff, nullptr);
  Gemm(l.shexp_down.data, ToGemvType(l.shexp_down.type), l.shexp_down.rows,
       l.shexp_down.cols, l.shexp_down.row_bytes, pf_shared_gate_,
       pf_shared_down_, tokens, nullptr);
  Gemm(l.shexp_gate_inp.data, ToGemvType(l.shexp_gate_inp.type),
       l.shexp_gate_inp.rows, l.shexp_gate_inp.cols,
       l.shexp_gate_inp.row_bytes, x, pf_shared_gate_inp_, tokens, nullptr);

  MoeEpilogue(pf_expert_out_, pf_weights_, pf_shared_down_, pf_shared_gate_inp_,
              1, out, tokens, used, hidden, nullptr);
}

void Executor::LinearAttentionBatch(const DeviceLayer& l, std::uint32_t il,
                                    const float* x, float* out,
                                    std::uint32_t tokens) {
  const std::uint32_t channels = c_.SsmConvChannels();
  const std::uint32_t d = c_.ssm_head_dim;
  const std::uint32_t kern = c_.ssm_conv_kernel;

  Gemm(l.ssm_qkv.data, ToGemvType(l.ssm_qkv.type), l.ssm_qkv.rows,
       l.ssm_qkv.cols, l.ssm_qkv.row_bytes, x, pf_qkv_, tokens, nullptr);
  Gemm(l.ssm_gate.data, ToGemvType(l.ssm_gate.type), l.ssm_gate.rows,
       l.ssm_gate.cols, l.ssm_gate.row_bytes, x, pf_z_, tokens, nullptr);
  Gemm(l.ssm_alpha.data, ToGemvType(l.ssm_alpha.type), l.ssm_alpha.rows,
       l.ssm_alpha.cols, l.ssm_alpha.row_bytes, x, pf_alpha_, tokens, nullptr);
  Gemm(l.ssm_beta.data, ToGemvType(l.ssm_beta.type), l.ssm_beta.rows,
       l.ssm_beta.cols, l.ssm_beta.row_bytes, x, pf_beta_, tokens, nullptr);

  GdnConvPrefill(pf_qkv_, l.ssm_conv1d.f32(), gdn_history_[il], pf_convolved_,
                 tokens, channels, kern, nullptr);
  GdnNormQkPrefill(pf_convolved_, pf_qn_, pf_kn_, tokens, c_.ssm_num_k_heads,
                   channels, d, c_.rms_eps, nullptr);
  GdnDeltaLoop(pf_qn_, pf_kn_, pf_convolved_, pf_alpha_, pf_beta_,
               l.ssm_a.f32(), l.ssm_dt.f32(), gdn_state_[il], pf_gdn_attn_,
               tokens, c_.ssm_num_k_heads, c_.ssm_num_v_heads, d, channels,
               nullptr);
  GdnOutNormPrefill(pf_gdn_attn_, pf_z_, l.ssm_norm.f32(), tokens,
                    c_.ssm_num_v_heads, d, c_.rms_eps, nullptr);
  Gemm(l.ssm_out.data, ToGemvType(l.ssm_out.type), l.ssm_out.rows,
       l.ssm_out.cols, l.ssm_out.row_bytes, pf_gdn_attn_, out, tokens, nullptr);

  // Advance the rolling conv history past the chunk (reading the pre-chunk
  // history, writing a disjoint buffer) and publish it for the next chunk.
  GdnHistoryUpdate(pf_qkv_, gdn_history_[il], pf_hist_new_, tokens, channels,
                   kern, nullptr);
  (void)hipMemcpyAsync(gdn_history_[il], pf_hist_new_,
                       static_cast<std::size_t>(kern - 1) * channels *
                           sizeof(float),
                       hipMemcpyDeviceToDevice, nullptr);
}

void Executor::AttentionBatch(const DeviceLayer& l, const float* x,
                              std::uint32_t start, float* out, float* k_cache,
                              float* v_cache, const std::uint32_t* pos_dev,
                              std::uint32_t tokens) {
  const std::uint32_t hd = c_.head_dim;
  const std::uint32_t nh = c_.num_heads;
  const std::uint32_t nkv = c_.num_kv_heads;
  const std::size_t q_dim = c_.AttentionQDim();
  const std::size_t kv_row = static_cast<std::size_t>(nkv) * hd;

  Gemm(l.attn_q.data, ToGemvType(l.attn_q.type), l.attn_q.rows, l.attn_q.cols,
       l.attn_q.row_bytes, x, pf_qg_, tokens, nullptr);
  Gemm(l.attn_k.data, ToGemvType(l.attn_k.type), l.attn_k.rows, l.attn_k.cols,
       l.attn_k.row_bytes, x, pf_k_, tokens, nullptr);
  Gemm(l.attn_v.data, ToGemvType(l.attn_v.type), l.attn_v.rows, l.attn_v.cols,
       l.attn_v.row_bytes, x, pf_v_, tokens, nullptr);

  for (std::uint32_t t = 0; t < tokens; ++t) {
    SplitQGate(pf_qg_ + static_cast<std::size_t>(t) * 2 * q_dim,
               pf_q_ + static_cast<std::size_t>(t) * q_dim,
               pf_qgate_ + static_cast<std::size_t>(t) * q_dim, nh, hd, nullptr);
  }
  RmsNormRows(pf_q_, l.attn_q_norm.f32(), pf_q_, tokens * nh, hd, c_.rms_eps,
              nullptr);
  RmsNormRows(pf_k_, l.attn_k_norm.f32(), pf_k_, tokens * nkv, hd, c_.rms_eps,
              nullptr);
  Rope(pf_q_, pos_dev, tokens, nh, hd, c_.rotary_dim, c_.rope_theta, nullptr);
  Rope(pf_k_, pos_dev, tokens, nkv, hd, c_.rotary_dim, c_.rope_theta, nullptr);

  // Publish the chunk's rotated keys/values at their absolute positions, then
  // read the whole prefix back causally.
  (void)hipMemcpyAsync(k_cache + static_cast<std::size_t>(start) * kv_row,
                       pf_k_, tokens * kv_row * sizeof(float),
                       hipMemcpyDeviceToDevice, nullptr);
  (void)hipMemcpyAsync(v_cache + static_cast<std::size_t>(start) * kv_row,
                       pf_v_, tokens * kv_row * sizeof(float),
                       hipMemcpyDeviceToDevice, nullptr);

  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  AttentionPrefill(pf_q_, k_cache, v_cache, pf_qgate_, pf_ctx_, start, tokens,
                   nh, nkv, hd, scale, nullptr);
  Gemm(l.attn_out.data, ToGemvType(l.attn_out.type), l.attn_out.rows,
       l.attn_out.cols, l.attn_out.row_bytes, pf_ctx_, out, tokens, nullptr);
}

bool Executor::Prefill(const std::int32_t* tokens, std::uint32_t count,
                       std::string* error_msg) {
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
  if (count > max_context_ - position_) {
    if (error_msg != nullptr) {
      *error_msg = "context length exceeded";
    }
    return false;
  }

  const std::uint32_t hidden = c_.hidden_size;
  std::uint32_t done = 0;
  std::uint32_t last_rows = 0;
  while (done < count) {
    const std::uint32_t rows =
        (count - done) < prefill_chunk_ ? (count - done) : prefill_chunk_;
    const std::uint32_t start = position_ + done;

    std::vector<std::uint32_t> host_pos(rows);
    for (std::uint32_t t = 0; t < rows; ++t) {
      host_pos[t] = start + t;
    }
    (void)hipMemcpyAsync(pf_pos_, host_pos.data(),
                         rows * sizeof(std::uint32_t), hipMemcpyHostToDevice,
                         nullptr);

    for (std::uint32_t t = 0; t < rows; ++t) {
      EmbedRow(model_.token_embd().data, ToGemvType(model_.token_embd().type),
               static_cast<std::uint32_t>(tokens[done + t]), hidden,
               pf_x_ + static_cast<std::size_t>(t) * hidden, nullptr);
    }

    for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
      const DeviceLayer& l = model_.layers()[il];
      RmsNormRows(pf_x_, l.attn_norm.f32(), pf_normed_, rows, hidden,
                  c_.rms_eps, nullptr);
      if (c_.IsLinearLayer(il)) {
        LinearAttentionBatch(l, il, pf_normed_, pf_attn_, rows);
      } else {
        AttentionBatch(l, pf_normed_, start, pf_attn_, k_cache_[il],
                       v_cache_[il], pf_pos_, rows);
      }
      Add(pf_x_, pf_attn_, static_cast<std::size_t>(rows) * hidden, nullptr);
      RmsNormRows(pf_x_, l.post_attention_norm.f32(), pf_normed_, rows, hidden,
                  c_.rms_eps, nullptr);
      MoeBatch(l, pf_normed_, pf_ffn_, rows);
      Add(pf_x_, pf_ffn_, static_cast<std::size_t>(rows) * hidden, nullptr);
    }
    done += rows;
    last_rows = rows;
  }

  // The final token's hidden state drives the output norm, h_out and logits.
  const float* last = pf_x_ + static_cast<std::size_t>(last_rows - 1) * hidden;
  RmsNormRows(last, model_.output_norm().f32(), x_, 1, hidden, c_.rms_eps,
              nullptr);
  (void)hipMemcpy(h_out_, x_, hidden * sizeof(float), hipMemcpyDeviceToDevice);
  Gemv(model_.output().data, ToGemvType(model_.output().type),
       model_.output().rows, model_.output().cols, model_.output().row_bytes, x_,
       logits_, nullptr);
  position_ += count;
  return true;
}

bool Executor::Step(std::int32_t token, std::string* error_msg) {
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "token id out of range";
    }
    return false;
  }
  if (position_ >= max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "context length exceeded";
    }
    return false;
  }

  pos_host_ = position_;
  (void)hipMemcpy(pos_dev_, &pos_host_, sizeof(std::uint32_t),
                  hipMemcpyHostToDevice);

  EmbedRow(model_.token_embd().data, ToGemvType(model_.token_embd().type),
           static_cast<std::uint32_t>(token), c_.hidden_size, x_, nullptr);

  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const DeviceLayer& l = model_.layers()[il];
    RmsNormRows(x_, l.attn_norm.f32(), normed_, 1, c_.hidden_size, c_.rms_eps,
                nullptr);
    if (c_.IsLinearLayer(il)) {
      LinearAttention(l, il, normed_, attn_);
    } else {
      Attention(l, normed_, position_, attn_, k_cache_[il], v_cache_[il],
                pos_dev_);
    }
    Add(x_, attn_, c_.hidden_size, nullptr);
    RmsNormRows(x_, l.post_attention_norm.f32(), normed_, 1, c_.hidden_size,
                c_.rms_eps, nullptr);
    Moe(l, normed_, ffn_);
    Add(x_, ffn_, c_.hidden_size, nullptr);
  }

  RmsNormRows(x_, model_.output_norm().f32(), x_, 1, c_.hidden_size, c_.rms_eps,
              nullptr);
  (void)hipMemcpy(h_out_, x_, c_.hidden_size * sizeof(float),
                  hipMemcpyDeviceToDevice);
  Gemv(model_.output().data, ToGemvType(model_.output().type),
       model_.output().rows, model_.output().cols, model_.output().row_bytes,
       x_, logits_, nullptr);
  ++position_;
  return true;
}

bool Executor::MtpStep(std::int32_t token, std::string* error_msg) {
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
  if (mtp_position_ >= max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "context length exceeded";
    }
    return false;
  }

  const DeviceLayer& l = model_.mtp();
  mtp_pos_host_ = mtp_position_;
  (void)hipMemcpy(mtp_pos_dev_, &mtp_pos_host_, sizeof(std::uint32_t),
                  hipMemcpyHostToDevice);

  EmbedRow(model_.token_embd().data, ToGemvType(model_.token_embd().type),
           static_cast<std::uint32_t>(token), c_.hidden_size, mtp_e_, nullptr);
  RmsNormRows(mtp_e_, l.nextn_enorm.f32(), mtp_e_, 1, c_.hidden_size,
              c_.rms_eps, nullptr);
  RmsNormRows(h_out_, l.nextn_hnorm.f32(), mtp_h_, 1, c_.hidden_size,
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
  Attention(l, normed_, mtp_position_, attn_, mtp_k_cache_, mtp_v_cache_,
            mtp_pos_dev_);
  Add(mtp_cur_, attn_, c_.hidden_size, nullptr);
  RmsNormRows(mtp_cur_, l.post_attention_norm.f32(), normed_, 1, c_.hidden_size,
              c_.rms_eps, nullptr);
  Moe(l, normed_, ffn_);
  Add(mtp_cur_, ffn_, c_.hidden_size, nullptr);
  RmsNormRows(mtp_cur_, l.nextn_shared_head_norm.f32(), mtp_cur_, 1,
              c_.hidden_size, c_.rms_eps, nullptr);
  Gemv(model_.output().data, ToGemvType(model_.output().type),
       model_.output().rows, model_.output().cols, model_.output().row_bytes,
       mtp_cur_, mtp_logits_, nullptr);
  ++mtp_position_;
  return true;
}

}  // namespace gufo::models::qwen36_a3b::rocm