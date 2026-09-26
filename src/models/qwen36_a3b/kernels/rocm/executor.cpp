#include "src/models/qwen36_a3b/kernels/rocm/executor.hpp"

#include <cmath>
#include <cstring>

#include "src/models/qwen36_a3b/kernels/rocm/gemv.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/kernels.hpp"

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