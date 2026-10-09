#include "src/models/ling3_flash/kernels/rocm/executor.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>
#include <string>

#include "src/models/ling3_flash/kernels/rocm/attention.hpp"
#include "src/models/ling3_flash/kernels/rocm/gemv.hpp"
#include "src/models/ling3_flash/kernels/rocm/kernels.hpp"

namespace gufo::models::ling3_flash::rocm {
namespace {

constexpr std::uint32_t kMlaTile = 1024U;

/// Maps a stored GGUF type onto the GEMV tier. Returns false for any format
/// the tier cannot decode so `Create` can reject the artifact up front.
bool GemvTypeOf(core::GgmlType t, GemvType* out) {
  switch (t) {
    case core::GgmlType::kQ4_0:
      *out = GemvType::kQ4_0;
      return true;
    case core::GgmlType::kQ8_0:
      *out = GemvType::kQ8_0;
      return true;
    case core::GgmlType::kQ4_K:
      *out = GemvType::kQ4_K;
      return true;
    case core::GgmlType::kQ5_K:
      *out = GemvType::kQ5_K;
      return true;
    case core::GgmlType::kQ6_K:
      *out = GemvType::kQ6_K;
      return true;
    case core::GgmlType::kF32:
      *out = GemvType::kF32;
      return true;
    default:
      return false;
  }
}

bool CheckTensor(const DeviceTensor& t, const char* what,
                 std::string* error_msg) {
  GemvType type;
  if (t.empty() || GemvTypeOf(t.type, &type)) {
    return true;
  }
  if (error_msg != nullptr) {
    *error_msg = std::string(what) + " has unsupported type " +
                 std::to_string(static_cast<int>(t.type));
  }
  return false;
}

bool CheckLayer(const DeviceLayer& l, std::uint32_t il,
                std::string* error_msg) {
  const std::string tag = "layer " + std::to_string(il);
  const DeviceTensor* tensors[] = {&l.attn_norm,
                                   &l.ffn_norm,
                                   &l.ssm_q,
                                   &l.ssm_k,
                                   &l.ssm_v,
                                   &l.ssm_conv_q,
                                   &l.ssm_conv_k,
                                   &l.ssm_conv_v,
                                   &l.ssm_f,
                                   &l.ssm_g,
                                   &l.ssm_beta,
                                   &l.ssm_a,
                                   &l.ssm_dt,
                                   &l.ssm_norm,
                                   &l.ssm_out,
                                   &l.attn_q,
                                   &l.attn_kv_a,
                                   &l.attn_kv_a_norm,
                                   &l.attn_k_b,
                                   &l.attn_v_b,
                                   &l.attn_gate,
                                   &l.attn_out,
                                   &l.ffn_gate,
                                   &l.ffn_up,
                                   &l.ffn_down,
                                   &l.router,
                                   &l.exp_probs_b,
                                   &l.ffn_gate_exps,
                                   &l.ffn_up_exps,
                                   &l.ffn_down_exps,
                                   &l.shexp_gate,
                                   &l.shexp_up,
                                   &l.shexp_down,
                                   &l.nextn_enorm,
                                   &l.nextn_hnorm,
                                   &l.nextn_eh_proj,
                                   &l.nextn_shared_head_norm};
  for (const DeviceTensor* t : tensors) {
    if (!CheckTensor(*t, tag.c_str(), error_msg)) {
      return false;
    }
  }
  return true;
}

}  // namespace

Session::~Session() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

float* Session::AllocFloats(std::size_t n) {
  void* p = nullptr;
  if (hipMalloc(&p, n * sizeof(float)) != hipSuccess) {
    alloc_failed_ = true;
    return nullptr;
  }
  allocations_.push_back(p);
  allocated_bytes_ += n * sizeof(float);
  (void)hipMemset(p, 0, n * sizeof(float));
  return static_cast<float*>(p);
}

void Session::Reset() {
  for (float* p : kda_conv_) {
    if (p != nullptr) {
      (void)hipMemset(
          p, 0,
          3U * (c_.ssm_conv_kernel - 1U) * c_.SsmInnerSize() * sizeof(float));
    }
  }
  for (float* p : kda_state_) {
    if (p != nullptr) {
      (void)hipMemset(p, 0,
                      static_cast<std::size_t>(c_.num_heads) * c_.ssm_head_dim *
                          c_.ssm_head_dim * sizeof(float));
    }
  }
  position_ = 0;
  mtp_position_ = 0;
}

Executor::~Executor() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

float* Executor::AllocFloats(std::size_t n, std::string* error) {
  void* p = nullptr;
  if (hipMalloc(&p, n * sizeof(float)) != hipSuccess) {
    if (error != nullptr) {
      *error = "hipMalloc failed for " + std::to_string(n) + " floats";
    }
    return nullptr;
  }
  allocations_.push_back(p);
  return static_cast<float*>(p);
}

std::int32_t* Executor::AllocInts(std::size_t n, std::string* error) {
  return reinterpret_cast<std::int32_t*>(AllocFloats(n, error));
}

std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           std::uint32_t max_context,
                                           std::string* error_msg) {
  const Config& c = model.config();
  if (!CheckTensor(model.token_embd(), "token_embd", error_msg) ||
      !CheckTensor(model.output(), "output", error_msg) ||
      !CheckTensor(model.output_norm(), "output_norm", error_msg)) {
    return nullptr;
  }
  for (std::uint32_t il = 0; il < model.layers().size(); ++il) {
    if (!CheckLayer(model.layers()[il], il, error_msg)) {
      return nullptr;
    }
  }
  if (model.has_mtp() && !CheckLayer(model.mtp(), c.num_layers, error_msg)) {
    return nullptr;
  }

  std::unique_ptr<Executor> e(new Executor(c, model));
  e->scratch_context_ = max_context;
  std::string* err = error_msg;
  const std::size_t hid = c.hidden_size;
  const std::size_t inner = c.SsmInnerSize();
  const std::size_t heads = c.num_heads;
  const std::size_t used = c.num_experts_used;
  const std::size_t ff = c.expert_ff;
  const std::size_t wide_ff = used * ff > c.dense_ff ? used * ff : c.dense_ff;
  const std::size_t tiles = MlaDecodeTiles(max_context, kMlaTile);
  float** f[] = {&e->x_,       &e->normed_, &e->attn_,   &e->ffn_,
                 &e->ffn_inp_, &e->h_out_,  &e->sh_down_};
  for (float** p : f) {
    *p = e->AllocFloats(hid, err);
    if (*p == nullptr) {
      return nullptr;
    }
  }
  e->logits_ = e->AllocFloats(c.vocab_size, err);
  if (e->logits_ == nullptr) {
    return nullptr;
  }
  e->mtp_logits_ = e->AllocFloats(c.vocab_size, err);
  if (e->mtp_logits_ == nullptr) {
    return nullptr;
  }
  for (float** p :
       {&e->kda_q_, &e->kda_k_, &e->kda_v_, &e->kda_f_, &e->kda_g_,
        &e->kda_conv_q_, &e->kda_conv_k_, &e->kda_conv_v_, &e->kda_attn_}) {
    *p = e->AllocFloats(inner, err);
    if (*p == nullptr) {
      return nullptr;
    }
  }
  e->kda_beta_ = e->AllocFloats(heads, err);
  if (e->kda_beta_ == nullptr) {
    return nullptr;
  }
  e->mla_q_ = e->AllocFloats(heads * c.mla_head_dim, err);
  e->mla_kv_a_ = e->AllocFloats(c.MlaKeyDim(), err);
  e->mla_q_abs_ = e->AllocFloats(heads * c.MlaKeyDim(), err);
  e->mla_latent_ = e->AllocFloats(heads * c.kv_lora_rank, err);
  e->mla_head_out_ = e->AllocFloats(heads * c.mla_value_dim, err);
  e->mla_gate_ = e->AllocFloats(heads, err);
  e->mla_ws_ = e->AllocFloats(tiles * heads * (c.kv_lora_rank + 2U), err);
  if (e->mla_q_ == nullptr || e->mla_kv_a_ == nullptr ||
      e->mla_q_abs_ == nullptr || e->mla_latent_ == nullptr ||
      e->mla_head_out_ == nullptr || e->mla_gate_ == nullptr ||
      e->mla_ws_ == nullptr) {
    return nullptr;
  }
  e->moe_logits_ = e->AllocFloats(c.num_experts, err);
  e->moe_ids_ = e->AllocInts(used, err);
  e->moe_weights_ = e->AllocFloats(used, err);
  e->moe_gate_ = e->AllocFloats(used * ff, err);
  e->moe_up_ = e->AllocFloats(used * ff, err);
  e->moe_inter_ = e->AllocFloats(used * ff, err);
  e->moe_down_ = e->AllocFloats(used * hid, err);
  e->sh_gate_ = e->AllocFloats(wide_ff, err);
  e->sh_up_ = e->AllocFloats(wide_ff, err);
  e->sh_inter_ = e->AllocFloats(wide_ff, err);
  if (e->moe_logits_ == nullptr || e->moe_ids_ == nullptr ||
      e->moe_weights_ == nullptr || e->moe_gate_ == nullptr ||
      e->moe_up_ == nullptr || e->moe_inter_ == nullptr ||
      e->moe_down_ == nullptr || e->sh_gate_ == nullptr ||
      e->sh_up_ == nullptr || e->sh_inter_ == nullptr) {
    return nullptr;
  }
  if (model.has_mtp()) {
    e->mtp_e_ = e->AllocFloats(hid, err);
    e->mtp_h_ = e->AllocFloats(hid, err);
    e->mtp_cat_ = e->AllocFloats(2 * hid, err);
    e->mtp_x_ = e->AllocFloats(hid, err);
    if (e->mtp_e_ == nullptr || e->mtp_h_ == nullptr ||
        e->mtp_cat_ == nullptr || e->mtp_x_ == nullptr) {
      return nullptr;
    }
  }
  return e;
}

std::size_t Executor::SessionBytes(std::uint32_t max_context) const noexcept {
  std::size_t bytes = 0;
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    if (c_.IsMlaLayer(il)) {
      bytes += static_cast<std::size_t>(max_context) * c_.MlaKeyDim() *
               sizeof(float);
    } else {
      bytes += (3U * (c_.ssm_conv_kernel - 1U) * c_.SsmInnerSize() +
                static_cast<std::size_t>(c_.num_heads) * c_.ssm_head_dim *
                    c_.ssm_head_dim) *
               sizeof(float);
    }
  }
  if (model_.has_mtp()) {
    bytes +=
        static_cast<std::size_t>(max_context) * c_.MlaKeyDim() * sizeof(float);
  }
  return bytes;
}

std::unique_ptr<Session> Executor::CreateSession(std::uint32_t max_context,
                                                 std::string* error_msg) const {
  if (max_context == 0 || max_context > scratch_context_) {
    if (error_msg != nullptr) {
      *error_msg = "session context exceeds executor scratch";
    }
    return nullptr;
  }
  std::unique_ptr<Session> s(new Session(c_, model_.has_mtp()));
  s->max_context_ = max_context;
  s->kda_conv_.assign(c_.num_layers, nullptr);
  s->kda_state_.assign(c_.num_layers, nullptr);
  s->mla_kv_.assign(c_.num_layers, nullptr);
  s->mla_kpe_.assign(c_.num_layers, nullptr);
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    if (c_.IsMlaLayer(il)) {
      s->mla_kv_[il] = s->AllocFloats(static_cast<std::size_t>(max_context) *
                                      c_.kv_lora_rank);
      s->mla_kpe_[il] =
          s->AllocFloats(static_cast<std::size_t>(max_context) * c_.rotary_dim);
    } else {
      s->kda_conv_[il] =
          s->AllocFloats(3U * (c_.ssm_conv_kernel - 1U) * c_.SsmInnerSize());
      s->kda_state_[il] =
          s->AllocFloats(static_cast<std::size_t>(c_.num_heads) *
                         c_.ssm_head_dim * c_.ssm_head_dim);
    }
    if (s->alloc_failed_) {
      if (error_msg != nullptr) {
        *error_msg = "session allocation failed";
      }
      return nullptr;
    }
  }
  if (model_.has_mtp()) {
    s->mtp_kv_ =
        s->AllocFloats(static_cast<std::size_t>(max_context) * c_.kv_lora_rank);
    s->mtp_kpe_ =
        s->AllocFloats(static_cast<std::size_t>(max_context) * c_.rotary_dim);
    if (s->alloc_failed_) {
      if (error_msg != nullptr) {
        *error_msg = "session allocation failed";
      }
      return nullptr;
    }
  }
  return s;
}

void Executor::KdaLayer(Session& session, std::uint32_t il,
                        const DeviceLayer& l, const float* x, float* out) {
  const std::uint32_t inner = c_.SsmInnerSize();
  const std::uint32_t heads = c_.num_heads;
  const std::uint32_t d = c_.ssm_head_dim;
  const std::uint32_t kern = c_.ssm_conv_kernel;
  GemvType type;
  (void)GemvTypeOf(l.ssm_q.type, &type);
  Gemv(l.ssm_q.data, type, l.ssm_q.rows, l.ssm_q.cols, l.ssm_q.row_bytes, x,
       kda_q_, nullptr);
  (void)GemvTypeOf(l.ssm_k.type, &type);
  Gemv(l.ssm_k.data, type, l.ssm_k.rows, l.ssm_k.cols, l.ssm_k.row_bytes, x,
       kda_k_, nullptr);
  (void)GemvTypeOf(l.ssm_v.type, &type);
  Gemv(l.ssm_v.data, type, l.ssm_v.rows, l.ssm_v.cols, l.ssm_v.row_bytes, x,
       kda_v_, nullptr);
  (void)GemvTypeOf(l.ssm_f.type, &type);
  Gemv(l.ssm_f.data, type, l.ssm_f.rows, l.ssm_f.cols, l.ssm_f.row_bytes, x,
       kda_f_, nullptr);
  (void)GemvTypeOf(l.ssm_g.type, &type);
  Gemv(l.ssm_g.data, type, l.ssm_g.rows, l.ssm_g.cols, l.ssm_g.row_bytes, x,
       kda_g_, nullptr);
  (void)GemvTypeOf(l.ssm_beta.type, &type);
  Gemv(l.ssm_beta.data, type, l.ssm_beta.rows, l.ssm_beta.cols,
       l.ssm_beta.row_bytes, x, kda_beta_, nullptr);
  // The decay input carries the per-channel dt bias before the gate.
  Add(kda_f_, l.ssm_dt.f32(), kda_f_, inner, nullptr);

  // The conv history holds the last kern-1 projected rows for q, k and v in
  // one per-layer buffer; ConvSilu shifts it and appends the new row.
  const std::size_t hist_stride = static_cast<std::size_t>(kern - 1U) * inner;
  float* hist = session.kda_conv_[il];
  ConvSilu(l.ssm_conv_q.f32(), kda_q_, hist, kda_conv_q_, inner, kern, nullptr);
  ConvSilu(l.ssm_conv_k.f32(), kda_k_, hist + hist_stride, kda_conv_k_, inner,
           kern, nullptr);
  ConvSilu(l.ssm_conv_v.f32(), kda_v_, hist + 2U * hist_stride, kda_conv_v_,
           inner, kern, nullptr);

  KdaNormHeads(kda_conv_q_, kda_conv_k_, heads, d, c_.rms_eps,
               1.0F / std::sqrt(static_cast<float>(d)), nullptr);
  KdaDelta(kda_conv_q_, kda_conv_k_, kda_conv_v_, kda_f_, kda_beta_,
           l.ssm_a.f32(), l.ssm_norm.f32(), kda_g_, session.kda_state_[il],
           kda_attn_, heads, d, c_.rms_eps, c_.kda_gate_lower_bound, nullptr);
  (void)GemvTypeOf(l.ssm_out.type, &type);
  Gemv(l.ssm_out.data, type, l.ssm_out.rows, l.ssm_out.cols,
       l.ssm_out.row_bytes, kda_attn_, out, nullptr);
}

void Executor::MlaLayer(const DeviceLayer& l, float* kv_cache, float* kpe_cache,
                        std::uint32_t pos, const float* x, float* out) {
  const std::uint32_t heads = c_.num_heads;
  const std::uint32_t nope = c_.MlaNopeDim();
  const std::uint32_t rot = c_.rotary_dim;
  const std::uint32_t kv_rank = c_.kv_lora_rank;
  const std::uint32_t v_dim = c_.mla_value_dim;
  const float scale = 1.0F / std::sqrt(static_cast<float>(c_.mla_head_dim));
  GemvType type;

  (void)GemvTypeOf(l.attn_q.type, &type);
  Gemv(l.attn_q.data, type, l.attn_q.rows, l.attn_q.cols, l.attn_q.row_bytes, x,
       mla_q_, nullptr);
  (void)GemvTypeOf(l.attn_kv_a.type, &type);
  Gemv(l.attn_kv_a.data, type, l.attn_kv_a.rows, l.attn_kv_a.cols,
       l.attn_kv_a.row_bytes, x, mla_kv_a_, nullptr);

  // The latent half is RMS-normalized and appended to the cache; the rope half
  // is rotated at this position and appended beside it. The q rope tail is
  // rotated in place inside each 192-wide head row.
  RmsNorm(mla_kv_a_, l.attn_kv_a_norm.f32(), c_.rms_eps, mla_kv_a_, kv_rank,
          nullptr);
  Rope(mla_kv_a_ + kv_rank, 1U, rot, rot, pos, c_.rope_theta, nullptr);
  RopeStrided(mla_q_ + nope, heads, c_.mla_head_dim, rot, pos, c_.rope_theta,
              nullptr);
  (void)hipMemcpyAsync(kv_cache + static_cast<std::size_t>(pos) * kv_rank,
                       mla_kv_a_, kv_rank * sizeof(float),
                       hipMemcpyDeviceToDevice, nullptr);
  (void)hipMemcpyAsync(kpe_cache + static_cast<std::size_t>(pos) * rot,
                       mla_kv_a_ + kv_rank, rot * sizeof(float),
                       hipMemcpyDeviceToDevice, nullptr);

  // Absorbed query: wk_b maps each head's nope half to the latent rank; the
  // rotated rope tail is copied behind it.
  (void)GemvTypeOf(l.attn_k_b.type, &type);
  for (std::uint32_t h = 0; h < heads; ++h) {
    Gemv(l.attn_k_b.Expert(h), type, kv_rank, nope, l.attn_k_b.row_bytes,
         mla_q_ + static_cast<std::size_t>(h) * c_.mla_head_dim,
         mla_q_abs_ + static_cast<std::size_t>(h) * (kv_rank + rot), nullptr);
    (void)hipMemcpyAsync(
        mla_q_abs_ + static_cast<std::size_t>(h) * (kv_rank + rot) + kv_rank,
        mla_q_ + static_cast<std::size_t>(h) * c_.mla_head_dim + nope,
        rot * sizeof(float), hipMemcpyDeviceToDevice, nullptr);
  }

  const std::uint32_t n_pos = pos + 1U;
  const std::uint32_t tiles = MlaDecodeTiles(n_pos, kMlaTile);
  MlaDecodePartial(mla_q_abs_, kv_cache, kpe_cache, n_pos, kMlaTile, scale,
                   mla_ws_, heads, kv_rank, rot, nullptr);
  MlaDecodeCombine(mla_ws_, tiles, heads, kv_rank, mla_latent_, nullptr);

  (void)GemvTypeOf(l.attn_v_b.type, &type);
  for (std::uint32_t h = 0; h < heads; ++h) {
    Gemv(l.attn_v_b.Expert(h), type, v_dim, kv_rank, l.attn_v_b.row_bytes,
         mla_latent_ + static_cast<std::size_t>(h) * kv_rank,
         mla_head_out_ + static_cast<std::size_t>(h) * v_dim, nullptr);
  }
  (void)GemvTypeOf(l.attn_gate.type, &type);
  Gemv(l.attn_gate.data, type, l.attn_gate.rows, l.attn_gate.cols,
       l.attn_gate.row_bytes, x, mla_gate_, nullptr);
  SigmoidInPlace(mla_gate_, heads, nullptr);
  RowScale(mla_head_out_, mla_gate_, heads, v_dim, nullptr);
  (void)GemvTypeOf(l.attn_out.type, &type);
  Gemv(l.attn_out.data, type, l.attn_out.rows, l.attn_out.cols,
       l.attn_out.row_bytes, mla_head_out_, out, nullptr);
}

void Executor::Ffn(const DeviceLayer& l, std::uint32_t il, const float* x,
                   float* out) {
  GemvType type;
  if (c_.IsDenseFfnLayer(il)) {
    const float limit = c_.swiglu_clamp_shexp[il];
    (void)GemvTypeOf(l.ffn_gate.type, &type);
    Gemv(l.ffn_gate.data, type, l.ffn_gate.rows, l.ffn_gate.cols,
         l.ffn_gate.row_bytes, x, sh_gate_, nullptr);
    (void)GemvTypeOf(l.ffn_up.type, &type);
    Gemv(l.ffn_up.data, type, l.ffn_up.rows, l.ffn_up.cols, l.ffn_up.row_bytes,
         x, sh_up_, nullptr);
    SwigluClamped(sh_gate_, sh_up_, sh_inter_, c_.dense_ff, limit, nullptr);
    (void)GemvTypeOf(l.ffn_down.type, &type);
    Gemv(l.ffn_down.data, type, l.ffn_down.rows, l.ffn_down.cols,
         l.ffn_down.row_bytes, sh_inter_, out, nullptr);
    return;
  }

  const std::uint32_t used = c_.num_experts_used;
  const std::uint32_t ff = c_.expert_ff;
  const std::uint32_t hid = c_.hidden_size;
  (void)GemvTypeOf(l.router.type, &type);
  Gemv(l.router.data, type, l.router.rows, l.router.cols, l.router.row_bytes, x,
       moe_logits_, nullptr);
  RouterTopK(moe_logits_, l.exp_probs_b.f32(), c_.num_experts, c_.expert_groups,
             c_.expert_groups_used, used, c_.expert_weights_scale, moe_ids_,
             moe_weights_, nullptr);
  (void)GemvTypeOf(l.ffn_gate_exps.type, &type);
  GemvGrouped(l.ffn_gate_exps.data, type,
              l.ffn_gate_exps.row_bytes * l.ffn_gate_exps.rows, moe_ids_, used,
              ff, c_.hidden_size, x, 0U, moe_gate_, nullptr);
  (void)GemvTypeOf(l.ffn_up_exps.type, &type);
  GemvGrouped(l.ffn_up_exps.data, type,
              l.ffn_up_exps.row_bytes * l.ffn_up_exps.rows, moe_ids_, used, ff,
              c_.hidden_size, x, 0U, moe_up_, nullptr);
  SwigluClamped(moe_gate_, moe_up_, moe_inter_, used * ff,
                c_.swiglu_clamp_exp[il], nullptr);
  (void)GemvTypeOf(l.ffn_down_exps.type, &type);
  GemvGrouped(l.ffn_down_exps.data, type,
              l.ffn_down_exps.row_bytes * l.ffn_down_exps.rows, moe_ids_, used,
              hid, ff, moe_inter_, ff, moe_down_, nullptr);
  RowWeightedSum(moe_down_, moe_weights_, out, used, hid, nullptr);

  // Always-on shared expert.
  (void)GemvTypeOf(l.shexp_gate.type, &type);
  Gemv(l.shexp_gate.data, type, l.shexp_gate.rows, l.shexp_gate.cols,
       l.shexp_gate.row_bytes, x, sh_gate_, nullptr);
  (void)GemvTypeOf(l.shexp_up.type, &type);
  Gemv(l.shexp_up.data, type, l.shexp_up.rows, l.shexp_up.cols,
       l.shexp_up.row_bytes, x, sh_up_, nullptr);
  SwigluClamped(sh_gate_, sh_up_, sh_inter_, c_.shared_expert_ff,
                c_.swiglu_clamp_shexp[il], nullptr);
  (void)GemvTypeOf(l.shexp_down.type, &type);
  Gemv(l.shexp_down.data, type, l.shexp_down.rows, l.shexp_down.cols,
       l.shexp_down.row_bytes, sh_inter_, sh_down_, nullptr);
  Add(out, sh_down_, out, hid, nullptr);
}

bool Executor::Step(Session& session, std::int32_t token,
                    std::string* error_msg) {
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "token id out of range";
    }
    return false;
  }
  if (session.position() >= session.max_context()) {
    if (error_msg != nullptr) {
      *error_msg = "context exhausted";
    }
    return false;
  }
  const std::uint32_t hid = c_.hidden_size;
  GemvType type;
  (void)GemvTypeOf(model_.token_embd().type, &type);
  EmbedRow(model_.token_embd().data, type, static_cast<std::uint32_t>(token),
           static_cast<std::uint32_t>(hid), x_, nullptr);

  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const DeviceLayer& l = model_.layers()[il];
    RmsNorm(x_, l.attn_norm.f32(), c_.rms_eps, normed_, hid, nullptr);
    if (c_.IsMlaLayer(il)) {
      MlaLayer(l, session.mla_kv_[il], session.mla_kpe_[il], session.position_,
               normed_, attn_);
    } else {
      KdaLayer(session, il, l, normed_, attn_);
    }
    Add(x_, attn_, ffn_inp_, hid, nullptr);
    RmsNorm(ffn_inp_, l.ffn_norm.f32(), c_.rms_eps, normed_, hid, nullptr);
    Ffn(l, il, normed_, ffn_);
    Add(ffn_inp_, ffn_, x_, hid, nullptr);
  }

  RmsNorm(x_, model_.output_norm().f32(), c_.rms_eps, h_out_, hid, nullptr);
  (void)GemvTypeOf(model_.output().type, &type);
  Gemv(model_.output().data, type, model_.output().rows, model_.output().cols,
       model_.output().row_bytes, h_out_, logits_, nullptr);
  ++session.position_;
  return true;
}

bool Executor::Prefill(Session& session, const std::int32_t* tokens,
                       std::uint32_t count, std::string* error_msg) {
  // The decode path is the reference-shaped implementation; a batched prefill
  // tier is a pure optimization on top of it.
  std::vector<std::int32_t> host(tokens, tokens + count);
  for (const std::int32_t token : host) {
    if (!Step(session, token, error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::MtpStep(Session& session, std::int32_t token,
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
  if (session.mtp_position_ >= session.max_context()) {
    if (error_msg != nullptr) {
      *error_msg = "context exhausted";
    }
    return false;
  }
  const std::uint32_t hid = c_.hidden_size;
  const DeviceLayer& l = model_.mtp();
  GemvType type;
  (void)GemvTypeOf(model_.token_embd().type, &type);
  EmbedRow(model_.token_embd().data, type, static_cast<std::uint32_t>(token),
           static_cast<std::uint32_t>(hid), mtp_e_, nullptr);
  RmsNorm(mtp_e_, l.nextn_enorm.f32(), c_.rms_eps, mtp_e_, hid, nullptr);
  RmsNorm(h_out_, l.nextn_hnorm.f32(), c_.rms_eps, mtp_h_, hid, nullptr);
  Concat2(mtp_e_, mtp_h_, mtp_cat_, hid, nullptr);
  (void)GemvTypeOf(l.nextn_eh_proj.type, &type);
  Gemv(l.nextn_eh_proj.data, type, l.nextn_eh_proj.rows, l.nextn_eh_proj.cols,
       l.nextn_eh_proj.row_bytes, mtp_cat_, mtp_x_, nullptr);

  RmsNorm(mtp_x_, l.attn_norm.f32(), c_.rms_eps, normed_, hid, nullptr);
  MlaLayer(l, session.mtp_kv_, session.mtp_kpe_, session.mtp_position_, normed_,
           attn_);
  Add(mtp_x_, attn_, ffn_inp_, hid, nullptr);
  RmsNorm(ffn_inp_, l.ffn_norm.f32(), c_.rms_eps, normed_, hid, nullptr);
  Ffn(l, c_.num_layers, normed_, ffn_);
  Add(ffn_inp_, ffn_, mtp_x_, hid, nullptr);

  RmsNorm(mtp_x_, l.nextn_shared_head_norm.f32(), c_.rms_eps, normed_, hid,
          nullptr);
  (void)GemvTypeOf(model_.output().type, &type);
  Gemv(model_.output().data, type, model_.output().rows, model_.output().cols,
       model_.output().row_bytes, normed_, mtp_logits_, nullptr);
  ++session.mtp_position_;
  return true;
}

}  // namespace gufo::models::ling3_flash::rocm