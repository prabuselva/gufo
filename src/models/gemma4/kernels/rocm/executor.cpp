#include "src/models/gemma4/kernels/rocm/executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "src/models/gemma4/kernels/rocm/attention.hpp"
#include "src/models/gemma4/kernels/rocm/fused.hpp"
#include "src/models/gemma4/kernels/rocm/gemm.hpp"
#include "src/models/gemma4/kernels/rocm/gemv.hpp"
#include "src/models/gemma4/kernels/rocm/routed_f16.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

bool GemvTypeOf(core::GgmlType t, GemvType* out) {
  switch (t) {
    case core::GgmlType::kF32:
      *out = GemvType::kF32;
      return true;
    case core::GgmlType::kQ8_0:
      *out = GemvType::kQ8_0;
      return true;
    case core::GgmlType::kBF16:
      *out = GemvType::kBF16;
      return true;
    case core::GgmlType::kQ4_K:
      *out = GemvType::kQ4_K;
      return true;
    case core::GgmlType::kQ5_K:
      *out = GemvType::kQ5_K;
      return true;
    default:
      return false;
  }
}

WeightType RoutedTypeOf(core::GgmlType t) {
  return static_cast<WeightType>(static_cast<std::uint32_t>(t));
}

bool Fail(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
  return false;
}

}  // namespace

Session::~Session() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

Executor::~Executor() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

float* Executor::AllocFloats(std::size_t n, std::string* error) {
  void* p = nullptr;
  if (hipMalloc(&p, n * sizeof(float)) != hipSuccess) {
    Fail(error, "hipMalloc failed for " + std::to_string(n) + " floats");
    return nullptr;
  }
  allocations_.push_back(p);
  return static_cast<float*>(p);
}

void* Executor::AllocBytes(std::size_t bytes, std::string* error) {
  void* p = nullptr;
  if (hipMalloc(&p, bytes) != hipSuccess) {
    Fail(error, "hipMalloc failed for " + std::to_string(bytes) + " bytes");
    return nullptr;
  }
  allocations_.push_back(p);
  return p;
}

std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           std::uint32_t prefill_chunk,
                                           std::string* error_msg) {
  std::unique_ptr<Executor> e(new Executor(model.config(), model));
  const Config& c = e->c_;
  if (prefill_chunk == 0) {
    prefill_chunk = 512;
  }
  if (model.token_embd().empty() || model.output().empty()) {
    Fail(error_msg, "model has no token embedding");
    return nullptr;
  }
  GemvType emb_type{};
  if (!GemvTypeOf(model.token_embd().type, &emb_type)) {
    Fail(error_msg, "unsupported token embedding type");
    return nullptr;
  }
  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t max_hd = std::max(c.head_dim_full, c.head_dim_swa);
  const std::uint32_t q_dim = c.num_heads * max_hd;
  const std::uint32_t kv_dim = 8 * max_hd;
  const std::uint32_t ffn = c.ffn_length;
  const std::uint32_t expert_ff = c.expert_ff;
  const std::uint32_t used = c.num_experts_used;
  const std::uint32_t C = prefill_chunk;
  const std::size_t slots = static_cast<std::size_t>(C) * used;

#define GUFO_ALLOC(member, n)                   \
  do {                                          \
    e->member = e->AllocFloats((n), error_msg); \
    if (e->member == nullptr)                   \
      return nullptr;                           \
  } while (0)

  GUFO_ALLOC(cur_, hidden);
  GUFO_ALLOC(other_, hidden);
  GUFO_ALLOC(normed_, hidden);
  GUFO_ALLOC(q_, q_dim);
  GUFO_ALLOC(qn_, q_dim);
  GUFO_ALLOC(k_raw_, kv_dim);
  GUFO_ALLOC(v_raw_, kv_dim);
  GUFO_ALLOC(attn_, q_dim);
  GUFO_ALLOC(proj_, hidden);
  GUFO_ALLOC(attn_out_, hidden);
  GUFO_ALLOC(mlp_, hidden);
  GUFO_ALLOC(mlp1_, hidden);
  GUFO_ALLOC(moe_, hidden);
  GUFO_ALLOC(moe2_, hidden);
  GUFO_ALLOC(gate_, ffn);
  GUFO_ALLOC(up_, ffn);
  GUFO_ALLOC(act_, std::max<std::size_t>(
                       ffn, static_cast<std::size_t>(used) * expert_ff));
  GUFO_ALLOC(gu_, static_cast<std::size_t>(used) * 2 * expert_ff);
  GUFO_ALLOC(dn_, static_cast<std::size_t>(used) * hidden);
  GUFO_ALLOC(router_in_, hidden);
  GUFO_ALLOC(router_logits_, c.num_experts);
  e->ids_ = reinterpret_cast<std::int32_t*>(
      e->AllocBytes(used * sizeof(std::int32_t), error_msg));
  if (e->ids_ == nullptr)
    return nullptr;
  GUFO_ALLOC(weights_, used);
  GUFO_ALLOC(attn_scratch_,
             static_cast<std::size_t>(c.num_heads) * 32U * (max_hd + 2U));
  GUFO_ALLOC(logits_, c.vocab_size);
  GUFO_ALLOC(h_out_, hidden);

  GUFO_ALLOC(pf_cur_, static_cast<std::size_t>(C) * hidden);
  GUFO_ALLOC(pf_next_, static_cast<std::size_t>(C) * hidden);
  GUFO_ALLOC(pf_normed_, static_cast<std::size_t>(C) * hidden);
  GUFO_ALLOC(pf_q_, static_cast<std::size_t>(C) * q_dim);
  GUFO_ALLOC(pf_qn_, static_cast<std::size_t>(C) * q_dim);
  GUFO_ALLOC(pf_k_, static_cast<std::size_t>(C) * kv_dim);
  GUFO_ALLOC(pf_v_, static_cast<std::size_t>(C) * kv_dim);
  GUFO_ALLOC(pf_attn_, static_cast<std::size_t>(C) * q_dim);
  GUFO_ALLOC(pf_proj_, static_cast<std::size_t>(C) * hidden);
  GUFO_ALLOC(pf_attn_out_, static_cast<std::size_t>(C) * hidden);
  GUFO_ALLOC(pf_mlp_, static_cast<std::size_t>(C) * hidden);
  GUFO_ALLOC(pf_mlp1_, static_cast<std::size_t>(C) * hidden);
  GUFO_ALLOC(pf_moe_, static_cast<std::size_t>(C) * hidden);
  GUFO_ALLOC(pf_moe2_, static_cast<std::size_t>(C) * hidden);
  GUFO_ALLOC(pf_gate_, static_cast<std::size_t>(C) * ffn);
  GUFO_ALLOC(pf_up_, static_cast<std::size_t>(C) * ffn);
  GUFO_ALLOC(pf_act_, static_cast<std::size_t>(C) * ffn);
  GUFO_ALLOC(pf_router_in_, static_cast<std::size_t>(C) * hidden);
  GUFO_ALLOC(pf_router_logits_, static_cast<std::size_t>(C) * c.num_experts);
  GUFO_ALLOC(pf_expert_out_, slots * hidden);
  e->pf_ids_ = reinterpret_cast<std::int32_t*>(
      e->AllocBytes(slots * sizeof(std::int32_t), error_msg));
  if (e->pf_ids_ == nullptr)
    return nullptr;
  GUFO_ALLOC(pf_weights_, slots);
  e->pf_counts_ = reinterpret_cast<std::uint32_t*>(
      e->AllocBytes(c.num_experts * sizeof(std::uint32_t), error_msg));
  if (e->pf_counts_ == nullptr)
    return nullptr;
  e->pf_pad_bounds_ = reinterpret_cast<std::int32_t*>(
      e->AllocBytes((c.num_experts + 1) * sizeof(std::int32_t), error_msg));
  if (e->pf_pad_bounds_ == nullptr)
    return nullptr;
  e->pf_cursors_ = reinterpret_cast<std::int32_t*>(
      e->AllocBytes(c.num_experts * sizeof(std::int32_t), error_msg));
  if (e->pf_cursors_ == nullptr)
    return nullptr;
  const std::size_t compact_rows = RoutedCompactRows(slots, c.num_experts);
  e->pf_rows_token_ = reinterpret_cast<std::int32_t*>(
      e->AllocBytes(compact_rows * sizeof(std::int32_t), error_msg));
  if (e->pf_rows_token_ == nullptr)
    return nullptr;
  e->pf_rows_slot_ = reinterpret_cast<std::int32_t*>(
      e->AllocBytes(compact_rows * sizeof(std::int32_t), error_msg));
  if (e->pf_rows_slot_ == nullptr)
    return nullptr;
  const std::size_t max_tiles = slots / 64 + c.num_experts + 1;
  e->pf_tiles_dev_ = reinterpret_cast<std::int32_t*>(
      e->AllocBytes(max_tiles * sizeof(std::int32_t), error_msg));
  if (e->pf_tiles_dev_ == nullptr)
    return nullptr;
  e->pf_x_half_ = reinterpret_cast<__half*>(e->AllocBytes(
      static_cast<std::size_t>(C) * hidden * sizeof(__half), error_msg));
  if (e->pf_x_half_ == nullptr)
    return nullptr;
  e->pf_gu_half_ = reinterpret_cast<__half*>(
      e->AllocBytes(slots * 2 * expert_ff * sizeof(__half), error_msg));
  if (e->pf_gu_half_ == nullptr)
    return nullptr;
  e->pf_act_half_ = reinterpret_cast<__half*>(
      e->AllocBytes(slots * expert_ff * sizeof(__half), error_msg));
  if (e->pf_act_half_ == nullptr)
    return nullptr;
#undef GUFO_ALLOC

  e->counts_host_.resize(c.num_experts);
  e->prefill_chunk_ = prefill_chunk;
  return e;
}

std::size_t Executor::SessionBytes(std::uint32_t max_context) const noexcept {
  std::size_t bytes = 0;
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const std::size_t row =
        static_cast<std::size_t>(c_.NumKvHeads(il)) * c_.HeadDim(il);
    bytes += 2 * static_cast<std::size_t>(max_context) * row * sizeof(__half);
  }
  return bytes;
}

std::unique_ptr<Session> Executor::CreateSession(std::uint32_t max_context,
                                                 std::string* error_msg) const {
  if (max_context == 0) {
    Fail(error_msg, "max_context must be positive");
    return nullptr;
  }
  std::unique_ptr<Session> s(new Session(c_, max_context));
  s->k_cache_.resize(c_.num_layers);
  s->v_cache_.resize(c_.num_layers);
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const std::size_t n = static_cast<std::size_t>(max_context) *
                          c_.NumKvHeads(il) * c_.HeadDim(il);
    if (hipMalloc(&s->k_cache_[il], n * sizeof(__half)) != hipSuccess) {
      Fail(error_msg, "hipMalloc failed for k_cache");
      return nullptr;
    }
    s->allocations_.push_back(s->k_cache_[il]);
    s->allocated_bytes_ += n * sizeof(__half);
    if (hipMalloc(&s->v_cache_[il], n * sizeof(__half)) != hipSuccess) {
      Fail(error_msg, "hipMalloc failed for v_cache");
      return nullptr;
    }
    s->allocations_.push_back(s->v_cache_[il]);
    s->allocated_bytes_ += n * sizeof(__half);
  }
  return s;
}

bool Executor::Step(Session& session, std::int32_t token,
                    std::string* error_msg) {
  const Config& c = c_;
  if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) {
    return Fail(error_msg, "token id out of range");
  }
  if (session.position_ >= session.max_context()) {
    return Fail(error_msg, "session context exhausted");
  }
  GemvType emb_type{};
  if (!GemvTypeOf(model_.token_embd().type, &emb_type)) {
    return Fail(error_msg, "unsupported token embedding type");
  }
  const std::uint32_t hidden = c.hidden_size;
  EmbedRow(model_.token_embd().data, emb_type,
           static_cast<std::uint32_t>(token), hidden, cur_, nullptr);
  ScaleInPlace(cur_, std::sqrt(static_cast<float>(hidden)), hidden, nullptr);

  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (!StepLayer(session, il, error_msg)) {
      return false;
    }
  }

  RmsNormRows(cur_, model_.output_norm().f32(), normed_, 1, hidden, c.rms_eps,
              nullptr);
  if (hipMemcpyAsync(h_out_, normed_, hidden * sizeof(float),
                     hipMemcpyDeviceToDevice) != hipSuccess) {
    return Fail(error_msg, "hidden state copy failed");
  }
  GemvType out_type{};
  if (!GemvTypeOf(model_.output().type, &out_type)) {
    return Fail(error_msg, "unsupported output type");
  }
  Gemv(model_.output().data, out_type, c.vocab_size, hidden,
       model_.output().row_bytes, normed_, logits_, nullptr);
  if (c.logit_softcap > 0.0F) {
    SoftcapInPlace(logits_, c.logit_softcap, c.vocab_size, nullptr);
  }
  ++session.position_;
  return true;
}

bool Executor::StepLayer(Session& session, std::uint32_t il,
                         std::string* error_msg) {
  const Config& c = c_;
  const DeviceLayer& l = model_.layers()[il];
  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t head_dim = c.HeadDim(il);
  const std::uint32_t heads = c.num_heads;
  const std::uint32_t kvh = c.NumKvHeads(il);
  const std::uint32_t pos = session.position_;
  const float* inv_freq = model_.inv_freq(il);

  RmsNormRows(cur_, l.attn_norm.f32(), normed_, 1, hidden, c.rms_eps, nullptr);

  GemvType qt{};
  GemvType kt{};
  GemvType vt{};
  if (!GemvTypeOf(l.attn_q.type, &qt) || !GemvTypeOf(l.attn_k.type, &kt)) {
    return Fail(error_msg,
                "unsupported attention type at layer " + std::to_string(il));
  }
  GemvMultiProj projs[3];
  std::size_t proj_row_bytes[3] = {l.attn_q.row_bytes, l.attn_k.row_bytes,
                                   l.attn_v.row_bytes};
  std::uint32_t n_projs = 0;
  projs[n_projs++] = {l.attn_q.data, qt, heads * head_dim, hidden, q_};
  projs[n_projs++] = {l.attn_k.data, kt, kvh * head_dim, hidden, k_raw_};
  if (!l.attn_v.empty()) {
    if (!GemvTypeOf(l.attn_v.type, &vt)) {
      return Fail(error_msg,
                  "unsupported attn_v type at layer " + std::to_string(il));
    }
    projs[n_projs++] = {l.attn_v.data, vt, kvh * head_dim, hidden, v_raw_};
  }
  if (!GemvMulti(projs, n_projs, normed_, nullptr)) {
    for (std::uint32_t p = 0; p < n_projs; ++p) {
      Gemv(projs[p].base, projs[p].type, projs[p].rows, projs[p].cols,
           proj_row_bytes[p], normed_, projs[p].out, nullptr);
    }
  }

  QknormRope(q_, l.attn_q_norm.f32(), qn_, 1, heads, head_dim, inv_freq, pos,
             nullptr);
  KvNormRopeWrite(k_raw_, l.attn_v.empty() ? k_raw_ : v_raw_,
                  l.attn_k_norm.f32(), session.k_cache_[il],
                  session.v_cache_[il], pos, 1, kvh, head_dim, inv_freq,
                  nullptr);

  const std::uint32_t window = l.swa ? c.sliding_window : 0;
  const std::uint32_t n_keys = pos + 1;
  const std::uint32_t lo =
      (window > 0 && n_keys > window) ? n_keys - window : 0;
  AttentionDecode(qn_, session.k_cache_[il], session.v_cache_[il], attn_,
                  attn_scratch_, n_keys, lo, heads, kvh, head_dim, nullptr);

  GemvType ot{};
  if (!GemvTypeOf(l.attn_output.type, &ot)) {
    return Fail(error_msg, "unsupported attn_output type");
  }
  Gemv(l.attn_output.data, ot, hidden, heads * head_dim,
       l.attn_output.row_bytes, attn_, proj_, nullptr);
  RmsNormRows(proj_, l.post_attention_norm.f32(), attn_out_, 1, hidden,
              c.rms_eps, nullptr);
  Add(attn_out_, cur_, hidden, nullptr);

  // Shared dense FFN: down( gelu(gate(x)) * up(x) ), then its post-norm.
  RmsNormRows(attn_out_, l.ffn_norm.f32(), normed_, 1, hidden, c.rms_eps,
              nullptr);
  GemvType ft{};
  if (!GemvTypeOf(l.ffn_gate.type, &ft) || !GemvTypeOf(l.ffn_up.type, &ft) ||
      !GemvTypeOf(l.ffn_down.type, &ft)) {
    return Fail(error_msg, "unsupported shared FFN type");
  }
  Gemv(l.ffn_up.data, ft, c.ffn_length, hidden, l.ffn_up.row_bytes, normed_,
       up_, nullptr);
  Gemv(l.ffn_gate.data, ft, c.ffn_length, hidden, l.ffn_gate.row_bytes, normed_,
       gate_, nullptr);
  GegluF32Separate(gate_, up_, act_, c.ffn_length, nullptr);
  Gemv(l.ffn_down.data, ft, hidden, c.ffn_length, l.ffn_down.row_bytes, act_,
       mlp_, nullptr);
  RmsNormRows(mlp_, l.post_ffw_norm_1.f32(), mlp1_, 1, hidden, c.rms_eps,
              nullptr);

  // Router: softmax over the scaled RMS-normal attention output.
  RmsNormRows(attn_out_, nullptr, router_in_, 1, hidden, c.rms_eps, nullptr);
  MulInPlace(router_in_, l.router_scale.f32(), hidden, hidden, nullptr);
  ScaleInPlace(router_in_, 1.0F / std::sqrt(static_cast<float>(hidden)), hidden,
               nullptr);
  GemvType rt{};
  if (!GemvTypeOf(l.router.type, &rt)) {
    return Fail(error_msg, "unsupported router type");
  }
  Gemv(l.router.data, rt, c.num_experts, hidden, l.router.row_bytes, router_in_,
       router_logits_, nullptr);
  RouterTopK(router_logits_, c.num_experts, ids_, weights_, 1, c.num_experts,
             c.num_experts_used, nullptr);

  // Top-8 experts through the grouped decode GEMV.
  RmsNormRows(attn_out_, l.pre_ffw_norm_2.f32(), normed_, 1, hidden, c.rms_eps,
              nullptr);
  GemvType gt{};
  GemvType dt{};
  if (!GemvTypeOf(l.ffn_gate_up_exps.type, &gt) ||
      !GemvTypeOf(l.ffn_down_exps.type, &dt)) {
    return Fail(error_msg, "unsupported expert type");
  }
  const std::size_t gu_stride =
      l.ffn_gate_up_exps.row_bytes * l.ffn_gate_up_exps.rows;
  GemvGrouped(l.ffn_gate_up_exps.data, gt, gu_stride, ids_, c.num_experts_used,
              2 * c.expert_ff, hidden, normed_, 0, gu_, nullptr);
  GegluF32(gu_, act_,
           static_cast<std::size_t>(c.num_experts_used) * c.expert_ff,
           c.expert_ff, nullptr);
  const std::size_t dn_stride =
      l.ffn_down_exps.row_bytes * l.ffn_down_exps.rows;
  GemvGrouped(l.ffn_down_exps.data, dt, dn_stride, ids_, c.num_experts_used,
              hidden, c.expert_ff, act_, c.expert_ff, dn_, nullptr);
  MoeEpilogue(dn_, weights_, ids_, l.ffn_down_exps_scale.f32(), moe_, 1,
              c.num_experts_used, hidden, nullptr);
  RmsNormRows(moe_, l.post_ffw_norm_2.f32(), moe2_, 1, hidden, c.rms_eps,
              nullptr);

  // cur = rmsnorm(mlp + moe) + attn_out, scaled by the layer output scale.
  if (hipMemcpyAsync(other_, mlp1_, hidden * sizeof(float),
                     hipMemcpyDeviceToDevice) != hipSuccess) {
    return Fail(error_msg, "mlp copy failed");
  }
  Add(other_, moe2_, hidden, nullptr);
  RmsNormRows(other_, l.post_ffw_norm.f32(), normed_, 1, hidden, c.rms_eps,
              nullptr);
  if (hipMemcpyAsync(other_, normed_, hidden * sizeof(float),
                     hipMemcpyDeviceToDevice) != hipSuccess) {
    return Fail(error_msg, "residual copy failed");
  }
  Add(other_, attn_out_, hidden, nullptr);
  ScaleInPlace(other_, l.layer_output_scale, hidden, nullptr);
  std::swap(cur_, other_);
  return true;
}

bool Executor::Prefill(Session& session, const std::int32_t* tokens,
                       std::uint32_t count, std::string* error_msg) {
  const Config& c = c_;
  if (count == 0) {
    return Fail(error_msg, "prefill count must be positive");
  }
  if (session.position_ + count > session.max_context()) {
    return Fail(error_msg, "prefill exceeds session context");
  }
  GemvType emb_type{};
  if (!GemvTypeOf(model_.token_embd().type, &emb_type)) {
    return Fail(error_msg, "unsupported token embedding type");
  }
  const std::uint32_t hidden = c.hidden_size;
  std::vector<std::int32_t> host_tokens(tokens, tokens + count);

  for (std::uint32_t start = 0; start < count; start += prefill_chunk_) {
    const std::uint32_t T = std::min(prefill_chunk_, count - start);
    for (std::uint32_t t = 0; t < T; ++t) {
      const std::int32_t tok = host_tokens[start + t];
      if (tok < 0 || static_cast<std::uint32_t>(tok) >= c.vocab_size) {
        return Fail(error_msg, "token id out of range");
      }
      EmbedRow(model_.token_embd().data, emb_type,
               static_cast<std::uint32_t>(tok), hidden,
               pf_cur_ + static_cast<std::size_t>(t) * hidden, nullptr);
    }
    ScaleInPlace(pf_cur_, std::sqrt(static_cast<float>(hidden)),
                 static_cast<std::size_t>(T) * hidden, nullptr);
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      if (!PrefillLayer(session, il, session.position_ + start, T, error_msg)) {
        return false;
      }
    }
    // The final token's row feeds the shared head after the last chunk.
    if (start + T == count) {
      const float* last = pf_cur_ + static_cast<std::size_t>(T - 1) * hidden;
      RmsNormRows(last, model_.output_norm().f32(), normed_, 1, hidden,
                  c.rms_eps, nullptr);
      if (hipMemcpyAsync(h_out_, normed_, hidden * sizeof(float),
                         hipMemcpyDeviceToDevice) != hipSuccess) {
        return Fail(error_msg, "hidden state copy failed");
      }
      GemvType out_type{};
      if (!GemvTypeOf(model_.output().type, &out_type)) {
        return Fail(error_msg, "unsupported output type");
      }
      Gemv(model_.output().data, out_type, c.vocab_size, hidden,
           model_.output().row_bytes, normed_, logits_, nullptr);
      if (c.logit_softcap > 0.0F) {
        SoftcapInPlace(logits_, c.logit_softcap, c.vocab_size, nullptr);
      }
    }
  }
  session.position_ += count;
  return true;
}

bool Executor::PrefillLayer(Session& session, std::uint32_t il,
                            std::uint32_t start, std::uint32_t tokens,
                            std::string* error_msg) {
  const Config& c = c_;
  const DeviceLayer& l = model_.layers()[il];
  const std::uint32_t hidden = c.hidden_size;
  const std::uint32_t head_dim = c.HeadDim(il);
  const std::uint32_t heads = c.num_heads;
  const std::uint32_t kvh = c.NumKvHeads(il);
  const float* inv_freq = model_.inv_freq(il);
  const std::size_t rows = tokens;

  RmsNormRows(pf_cur_, l.attn_norm.f32(), pf_normed_, tokens, hidden, c.rms_eps,
              nullptr);

  GemvType qt{};
  GemvType kt{};
  GemvType vt{};
  if (!GemvTypeOf(l.attn_q.type, &qt) || !GemvTypeOf(l.attn_k.type, &kt)) {
    return Fail(error_msg,
                "unsupported attention type at layer " + std::to_string(il));
  }
  Gemm(l.attn_q.data, qt, heads * head_dim, hidden, l.attn_q.row_bytes,
       pf_normed_, pf_q_, tokens, nullptr);
  Gemm(l.attn_k.data, kt, kvh * head_dim, hidden, l.attn_k.row_bytes,
       pf_normed_, pf_k_, tokens, nullptr);
  if (!l.attn_v.empty()) {
    if (!GemvTypeOf(l.attn_v.type, &vt)) {
      return Fail(error_msg,
                  "unsupported attn_v type at layer " + std::to_string(il));
    }
    Gemm(l.attn_v.data, vt, kvh * head_dim, hidden, l.attn_v.row_bytes,
         pf_normed_, pf_v_, tokens, nullptr);
  }

  QknormRope(pf_q_, l.attn_q_norm.f32(), pf_qn_, tokens, heads, head_dim,
             inv_freq, start, nullptr);
  KvNormRopeWrite(pf_k_, l.attn_v.empty() ? pf_k_ : pf_v_, l.attn_k_norm.f32(),
                  session.k_cache_[il], session.v_cache_[il], start, tokens,
                  kvh, head_dim, inv_freq, nullptr);

  const std::uint32_t window = l.swa ? c.sliding_window : 0;
  AttentionPrefill(pf_qn_, session.k_cache_[il], session.v_cache_[il], pf_attn_,
                   start, tokens, heads, kvh, head_dim, window, nullptr);

  GemvType ot{};
  if (!GemvTypeOf(l.attn_output.type, &ot)) {
    return Fail(error_msg, "unsupported attn_output type");
  }
  Gemm(l.attn_output.data, ot, hidden, heads * head_dim,
       l.attn_output.row_bytes, pf_attn_, pf_proj_, tokens, nullptr);
  RmsNormRows(pf_proj_, l.post_attention_norm.f32(), pf_attn_out_, tokens,
              hidden, c.rms_eps, nullptr);
  Add(pf_attn_out_, pf_cur_, rows * hidden, nullptr);

  RmsNormRows(pf_attn_out_, l.ffn_norm.f32(), pf_normed_, tokens, hidden,
              c.rms_eps, nullptr);
  GemvType ft{};
  if (!GemvTypeOf(l.ffn_gate.type, &ft) || !GemvTypeOf(l.ffn_up.type, &ft) ||
      !GemvTypeOf(l.ffn_down.type, &ft)) {
    return Fail(error_msg, "unsupported shared FFN type");
  }
  Gemm(l.ffn_up.data, ft, c.ffn_length, hidden, l.ffn_up.row_bytes, pf_normed_,
       pf_up_, tokens, nullptr);
  Gemm(l.ffn_gate.data, ft, c.ffn_length, hidden, l.ffn_gate.row_bytes,
       pf_normed_, pf_gate_, tokens, nullptr);
  GegluF32Separate(pf_gate_, pf_up_, pf_act_, rows * c.ffn_length, nullptr);
  Gemm(l.ffn_down.data, ft, hidden, c.ffn_length, l.ffn_down.row_bytes, pf_act_,
       pf_mlp_, tokens, nullptr);
  RmsNormRows(pf_mlp_, l.post_ffw_norm_1.f32(), pf_mlp1_, tokens, hidden,
              c.rms_eps, nullptr);

  RmsNormRows(pf_attn_out_, nullptr, pf_router_in_, tokens, hidden, c.rms_eps,
              nullptr);
  MulInPlace(pf_router_in_, l.router_scale.f32(), rows * hidden, hidden,
             nullptr);
  ScaleInPlace(pf_router_in_, 1.0F / std::sqrt(static_cast<float>(hidden)),
               rows * hidden, nullptr);
  GemvType rt{};
  if (!GemvTypeOf(l.router.type, &rt)) {
    return Fail(error_msg, "unsupported router type");
  }
  Gemm(l.router.data, rt, c.num_experts, hidden, l.router.row_bytes,
       pf_router_in_, pf_router_logits_, tokens, nullptr);
  RouterTopK(pf_router_logits_, c.num_experts, pf_ids_, pf_weights_, tokens,
             c.num_experts, c.num_experts_used, nullptr);

  RmsNormRows(pf_attn_out_, l.pre_ffw_norm_2.f32(), pf_normed_, tokens, hidden,
              c.rms_eps, nullptr);
  if (!MoeBatch(l, pf_normed_, pf_ids_, pf_weights_, pf_moe_, tokens,
                error_msg)) {
    return false;
  }
  RmsNormRows(pf_moe_, l.post_ffw_norm_2.f32(), pf_moe2_, tokens, hidden,
              c.rms_eps, nullptr);

  if (hipMemcpyAsync(pf_next_, pf_mlp1_, rows * hidden * sizeof(float),
                     hipMemcpyDeviceToDevice) != hipSuccess) {
    return Fail(error_msg, "mlp copy failed");
  }
  Add(pf_next_, pf_moe2_, rows * hidden, nullptr);
  RmsNormRows(pf_next_, l.post_ffw_norm.f32(), pf_normed_, tokens, hidden,
              c.rms_eps, nullptr);
  if (hipMemcpyAsync(pf_next_, pf_normed_, rows * hidden * sizeof(float),
                     hipMemcpyDeviceToDevice) != hipSuccess) {
    return Fail(error_msg, "residual copy failed");
  }
  Add(pf_next_, pf_attn_out_, rows * hidden, nullptr);
  ScaleInPlace(pf_next_, l.layer_output_scale, rows * hidden, nullptr);
  std::swap(pf_cur_, pf_next_);
  return true;
}

bool Executor::MoeBatch(const DeviceLayer& l, const float* x,
                        const std::int32_t* ids, const float* weights,
                        float* out, std::uint32_t tokens,
                        std::string* error_msg) {
  const Config& c = c_;
  const std::uint32_t used = c.num_experts_used;
  const std::size_t slots = static_cast<std::size_t>(tokens) * used;

  NarrowActivations(x, pf_x_half_, false,
                    static_cast<std::size_t>(tokens) * c.hidden_size, nullptr);
  ExpertCounts(ids, pf_counts_, tokens, c.num_experts, used, nullptr);
  if (hipMemcpy(counts_host_.data(), pf_counts_,
                c.num_experts * sizeof(std::uint32_t),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return Fail(error_msg, "expert histogram download failed");
  }
  tiles_host_.clear();
  for (std::uint32_t e = 0; e < c.num_experts; ++e) {
    const std::uint32_t padded = (counts_host_[e] + 15U) / 16U * 16U;
    for (std::uint32_t j = 0; j < (padded + 63U) / 64U; ++j) {
      tiles_host_.push_back(static_cast<std::int32_t>(e | (j << 16)));
    }
  }
  if (tiles_host_.empty()) {
    return Fail(error_msg, "routed MoE produced no tiles");
  }
  if (hipMemcpyAsync(pf_tiles_dev_, tiles_host_.data(),
                     tiles_host_.size() * sizeof(std::int32_t),
                     hipMemcpyHostToDevice) != hipSuccess) {
    return Fail(error_msg, "tile map upload failed");
  }
  RoutedCompact(ids, pf_counts_, pf_pad_bounds_, pf_cursors_, pf_rows_token_,
                pf_rows_slot_, tokens, used, c.num_experts, nullptr);

  if (l.ffn_gate_up_exps.type != l.ffn_down_exps.type) {
    return Fail(error_msg, "expert gate/up and down types differ");
  }
  const WeightType wt = RoutedTypeOf(l.ffn_gate_up_exps.type);
  const std::uint32_t n_tiles = static_cast<std::uint32_t>(tiles_host_.size());
  if (!RoutedF16Gemm(l.ffn_gate_up_exps.data, wt, pf_x_half_, pf_tiles_dev_,
                     n_tiles, 64, pf_pad_bounds_, pf_rows_token_, pf_rows_slot_,
                     nullptr, pf_gu_half_, 2 * c.expert_ff, c.hidden_size,
                     nullptr)) {
    return Fail(error_msg, "RoutedF16Gemm gate_up unsupported");
  }
  GegluF16(pf_gu_half_, pf_act_half_, slots * c.expert_ff, c.expert_ff,
           nullptr);
  if (!RoutedF16Gemm(l.ffn_down_exps.data, wt, pf_act_half_, pf_tiles_dev_,
                     n_tiles, 64, pf_pad_bounds_, pf_rows_slot_, pf_rows_slot_,
                     pf_expert_out_, nullptr, c.hidden_size, c.expert_ff,
                     nullptr)) {
    return Fail(error_msg, "RoutedF16Gemm down unsupported");
  }
  MoeEpilogue(pf_expert_out_, weights, ids, l.ffn_down_exps_scale.f32(), out,
              tokens, used, c.hidden_size, nullptr);
  return true;
}

}  // namespace gufo::models::gemma4::rocm