#include "src/models/ling3_flash/kernels/rocm/executor.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "src/models/ling3_flash/kernels/rocm/attention.hpp"
#include "src/models/ling3_flash/kernels/rocm/gemv.hpp"
#include "src/models/ling3_flash/kernels/rocm/kernels.hpp"
#include "src/models/ling3_flash/kernels/rocm/routed_f16.hpp"

namespace gufo::models::ling3_flash::rocm {
namespace {

constexpr std::uint32_t kMlaTile = 256U;

/// Maps a stored GGUF type onto the F16 WMMA tier (dense projections and
/// routed experts). The block formats are decoded in-kernel; F32/F16 tensors
/// (e.g. the KDA beta projection) and Q4_0 are not, and stay on the GEMV
/// tier or the per-token path.
bool WeightTypeOf(core::GgmlType t, WeightType* out) {
  switch (t) {
    case core::GgmlType::kQ4_0:
      *out = WeightType::kQ4_0;
      return true;
    case core::GgmlType::kQ5_1:
      *out = WeightType::kQ5_1;
      return true;
    case core::GgmlType::kQ8_0:
      *out = WeightType::kQ8_0;
      return true;
    case core::GgmlType::kQ4_K:
      *out = WeightType::kQ4_K;
      return true;
    case core::GgmlType::kQ5_K:
      *out = WeightType::kQ5_K;
      return true;
    case core::GgmlType::kQ6_K:
      *out = WeightType::kQ6_K;
      return true;
    case core::GgmlType::kBF16:
      *out = WeightType::kBF16;
      return true;
    case core::GgmlType::kF16:
      *out = WeightType::kF16;
      return true;
    default:
      return false;
  }
}

/// True when every matrix the chunked-prefill tier projects for one layer
/// can be decoded by the F16 WMMA tier; a trunk format it cannot decode
/// falls back to the per-token path in Prefill instead of failing Create.
/// Only the projections this layer kind actually uses are checked: KDA
/// layers hold no MLA tensors and vice versa, and ssm_beta (F32) runs on
/// the GEMV tier while the router (converted to binary16 at upload) projects
/// through the WMMA tier. The swiglu gate variant only instantiates three
/// decoders.
bool DenseTierCovers(const Config& c, std::uint32_t il, const DeviceLayer& l) {
  WeightType w;
  std::vector<core::GgmlType> types;
  if (c.IsMlaLayer(il)) {
    types = {l.attn_q.type,   l.attn_kv_a.type, l.attn_gate.type,
             l.attn_out.type, l.attn_k_b.type,  l.attn_v_b.type};
  } else {
    types = {l.ssm_q.type, l.ssm_k.type, l.ssm_v.type,
             l.ssm_f.type, l.ssm_g.type, l.ssm_out.type};
  }
  if (c.IsDenseFfnLayer(il)) {
    types.insert(types.end(),
                 {l.ffn_gate.type, l.ffn_up.type, l.ffn_down.type});
  } else {
    types.insert(types.end(), {l.ffn_down_exps.type, l.shexp_gate.type,
                               l.shexp_up.type, l.shexp_down.type});
  }
  for (const core::GgmlType t : types) {
    if (!WeightTypeOf(t, &w)) {
      return false;
    }
  }
  if (!c.IsDenseFfnLayer(il)) {
    const core::GgmlType gated[] = {l.ffn_gate_exps.type, l.ffn_up_exps.type};
    for (const core::GgmlType t : gated) {
      if (!WeightTypeOf(t, &w) ||
          (t != core::GgmlType::kQ4_K && t != core::GgmlType::kQ8_0 &&
           t != core::GgmlType::kBF16)) {
        return false;
      }
    }
  }
  return true;
}

/// True when the F16 WMMA tier can run the draft layer's projections. MTP
/// checkpoints quantize their matrices as Q4_0, so the gate/up expert pair
/// additionally accepts that format; a draft layer the tier cannot decode
/// keeps MtpPrefill on the sequential per-token path.
bool MtpDraftTierCovers(const DeviceLayer& l) {
  WeightType w;
  const core::GgmlType dense[] = {
      l.attn_q.type,        l.attn_kv_a.type,     l.attn_gate.type,
      l.attn_out.type,      l.attn_k_b.type,      l.attn_v_b.type,
      l.nextn_eh_proj.type, l.ffn_down_exps.type, l.shexp_gate.type,
      l.shexp_up.type,      l.shexp_down.type};
  for (const core::GgmlType t : dense) {
    if (!WeightTypeOf(t, &w)) {
      return false;
    }
  }
  const core::GgmlType gated[] = {l.ffn_gate_exps.type, l.ffn_up_exps.type};
  for (const core::GgmlType t : gated) {
    if (!WeightTypeOf(t, &w) ||
        (t != core::GgmlType::kQ4_0 && t != core::GgmlType::kQ4_K &&
         t != core::GgmlType::kQ8_0 && t != core::GgmlType::kBF16)) {
      return false;
    }
  }
  return true;
}

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
    case core::GgmlType::kF16:
      *out = GemvType::kF16;
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
  for (hipEvent_t e : {ev_up_fork_, ev_shared_fork_, ev_up_, ev_shared_}) {
    if (e != nullptr) {
      (void)hipEventDestroy(e);
    }
  }
  for (hipStream_t s : {moe_up_stream_, moe_shared_stream_}) {
    if (s != nullptr) {
      (void)hipStreamDestroy(s);
    }
  }
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
  // Chunked-prefill scratch: sized for one max-size chunk, always allocated.
  // The planes the verify pass shares (residual stream, raw KDA/MLA/MoE
  // projections) grow from kMaxVerifyRows to this row count.
  const std::size_t cap = Executor::kPrefillChunkRows;
  const std::size_t slots = cap * used;
  const std::size_t compact_rows =
      std::max(RoutedCompactRows(slots, c.num_experts),
               RoutedCompactRows(cap * heads, heads));
  const std::size_t shared_ff = c.shared_expert_ff;
  float** cf[] = {
      &e->pf_x_,           &e->pf_normed_,   &e->pf_attn_,     &e->pf_ffn_,
      &e->pf_kda_q_,       &e->pf_kda_k_,    &e->pf_kda_v_,    &e->pf_kda_f_,
      &e->pf_kda_g_,       &e->pf_cq_,       &e->pf_ck_,       &e->pf_cv_,
      &e->pf_o_,           &e->pf_decay_,    &e->pf_kda_hist_, &e->pf_dn_gate_,
      &e->pf_dn_up_,       &e->pf_dn_inter_, &e->pf_pdown_,    &e->pf_sh_gate_,
      &e->pf_sh_up_,       &e->pf_sh_inter_, &e->pf_sh_down_,  &e->pf_mla_qrp_,
      &e->pf_mla_qlatent_, &e->pf_mla_pabs_,
  };
  const std::size_t c_n[] = {cap * hid,
                             cap * hid,
                             cap * hid,
                             cap * hid,
                             cap * inner,
                             cap * inner,
                             cap * inner,
                             cap * inner,
                             cap * inner,
                             cap * inner,
                             cap * inner,
                             cap * inner,
                             cap * inner,
                             cap * inner,
                             3U * (c.ssm_conv_kernel - 1U) * inner,
                             cap * c.dense_ff,
                             cap * c.dense_ff,
                             cap * c.dense_ff,
                             slots * hid,
                             cap * shared_ff,
                             cap * shared_ff,
                             cap * shared_ff,
                             cap * hid,
                             cap * heads * c.rotary_dim,
                             cap * heads * c.kv_lora_rank,
                             cap * heads * c.kv_lora_rank};
  for (std::size_t i = 0; i < sizeof(cf) / sizeof(cf[0]); ++i) {
    *cf[i] = e->AllocFloats(c_n[i], err);
    if (*cf[i] == nullptr) {
      return nullptr;
    }
  }
  // x rows are narrowed once per launch with the launch's k as stride; the
  // plane is wide enough for the widest input (the dense-FFN inter).
  e->pf_xh_ =
      reinterpret_cast<__half*>(e->AllocFloats((cap * wide_ff + 1U) / 2U, err));
  e->pf_interh_ =
      reinterpret_cast<__half*>(e->AllocFloats((slots * ff + 1U) / 2U, err));
  e->pf_kda_beta_ = e->AllocFloats(cap * heads, err);
  e->pf_bpre_ = e->AllocFloats(cap * heads, err);
  e->pf_kq_ = e->AllocFloats(cap * heads, err);
  e->pf_mla_pgate_ = e->AllocFloats(cap * heads, err);
  e->pf_mla_q_ = e->AllocFloats(cap * heads * c.mla_head_dim, err);
  e->pf_mla_a_ = e->AllocFloats(cap * c.MlaKeyDim(), err);
  e->pf_mla_head_ = e->AllocFloats(cap * heads * c.mla_value_dim, err);
  e->pf_moe_weights_ = e->AllocFloats(cap * used, err);
  e->pf_moe_log_ = e->AllocFloats(cap * c.num_experts, err);
  e->pf_moe_ids_ = e->AllocInts(cap * used, err);
  e->pf_counts_ = e->AllocInts(c.num_experts, err);
  e->pf_bounds_ = e->AllocInts(c.num_experts + 1U, err);
  e->pf_cursors_ = e->AllocInts(c.num_experts, err);
  e->pf_rows_tok_ = e->AllocInts(compact_rows, err);
  e->pf_rows_slot_ = e->AllocInts(compact_rows, err);
  e->pf_tiles_ = e->AllocInts(compact_rows / 64U + c.num_experts + 8U, err);
  e->pf_zero_ids_ = e->AllocInts(cap, err);
  e->pf_head_ids_ = e->AllocInts(cap * heads, err);
  e->pf_drows_ = e->AllocInts(cap, err);
  e->pf_dbounds_ = e->AllocInts(2U, err);
  e->pf_dtiles_ = e->AllocInts(cap / 16U, err);
  // Head-grouped (uniform-bucket) scratch for the absorbed MLA projections.
  e->pf_qnc_ = reinterpret_cast<__half*>(
      e->AllocFloats((cap * heads * c.MlaNopeDim() + 1U) / 2U, err));
  e->pf_hbounds_ = e->AllocInts(heads + 1U, err);
  e->pf_hcursors_ = e->AllocInts(heads, err);
  e->pf_htiles_ = e->AllocInts(heads * (cap / 64U + 1U) + 8U, err);
  if (e->pf_xh_ == nullptr || e->pf_interh_ == nullptr ||
      e->pf_kda_beta_ == nullptr || e->pf_bpre_ == nullptr ||
      e->pf_kq_ == nullptr || e->pf_mla_pgate_ == nullptr ||
      e->pf_mla_q_ == nullptr || e->pf_mla_a_ == nullptr ||
      e->pf_mla_head_ == nullptr || e->pf_moe_weights_ == nullptr ||
      e->pf_moe_log_ == nullptr || e->pf_moe_ids_ == nullptr ||
      e->pf_counts_ == nullptr || e->pf_bounds_ == nullptr ||
      e->pf_cursors_ == nullptr || e->pf_rows_tok_ == nullptr ||
      e->pf_rows_slot_ == nullptr || e->pf_tiles_ == nullptr ||
      e->pf_zero_ids_ == nullptr || e->pf_head_ids_ == nullptr ||
      e->pf_drows_ == nullptr || e->pf_dbounds_ == nullptr ||
      e->pf_dtiles_ == nullptr || e->pf_qnc_ == nullptr ||
      e->pf_hbounds_ == nullptr || e->pf_hcursors_ == nullptr ||
      e->pf_htiles_ == nullptr) {
    return nullptr;
  }
  // Constant index vectors for the grouped per-head GEMVs and the F32 router.
  FillModuloIds(e->pf_head_ids_, static_cast<std::uint32_t>(cap * heads),
                c.num_heads, nullptr);
  FillModuloIds(e->pf_zero_ids_, static_cast<std::uint32_t>(cap), 1U, nullptr);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (!DenseTierCovers(c, il, model.layers()[il])) {
      break;
    }
    if (il + 1U == c.num_layers) {
      e->prefill_ready_ = true;
    }
  }
  if (model.has_mtp()) {
    e->mtp_draft_tier_ready_ =
        e->prefill_ready_ && MtpDraftTierCovers(model.mtp());
    e->mtp_e_ = e->AllocFloats(hid, err);
    e->mtp_h_ = e->AllocFloats(hid, err);
    e->mtp_cat_ = e->AllocFloats(2 * hid, err);
    e->mtp_x_ = e->AllocFloats(hid, err);
    e->mtp_chain_ = e->AllocFloats(hid, err);
    const std::size_t k = Executor::kMaxVerifyRows;
    e->pf_h_ = e->AllocFloats(k * hid, err);
    e->pf_logits_ = e->AllocFloats(k * c.vocab_size, err);
    e->pf_mla_abs_ = e->AllocFloats(k * heads * c.MlaKeyDim(), err);
    e->pf_moe_gate_ = e->AllocFloats(k * used * ff, err);
    e->pf_moe_up_ = e->AllocFloats(k * used * ff, err);
    e->pf_moe_inter_ = e->AllocFloats(k * used * ff, err);
    e->pf_moe_down_ = e->AllocFloats(k * used * hid, err);
    e->mtp_pf_e_ = e->AllocFloats(cap * hid, err);
    e->mtp_pf_h_ = e->AllocFloats(cap * hid, err);
    e->mtp_pf_x_ = e->AllocFloats(cap * hid, err);
    if (e->mtp_e_ == nullptr || e->mtp_h_ == nullptr ||
        e->mtp_cat_ == nullptr || e->mtp_x_ == nullptr ||
        e->mtp_chain_ == nullptr || e->pf_h_ == nullptr ||
        e->pf_logits_ == nullptr || e->pf_mla_abs_ == nullptr ||
        e->pf_moe_gate_ == nullptr || e->pf_moe_up_ == nullptr ||
        e->pf_moe_inter_ == nullptr || e->pf_moe_down_ == nullptr ||
        e->mtp_pf_e_ == nullptr || e->mtp_pf_h_ == nullptr ||
        e->mtp_pf_x_ == nullptr) {
      return nullptr;
    }
  }
  if (hipStreamCreateWithFlags(&e->moe_up_stream_, hipStreamNonBlocking) !=
          hipSuccess ||
      hipStreamCreateWithFlags(&e->moe_shared_stream_, hipStreamNonBlocking) !=
          hipSuccess ||
      hipEventCreateWithFlags(&e->ev_shared_fork_, hipEventDisableTiming) !=
          hipSuccess ||
      hipEventCreateWithFlags(&e->ev_up_fork_, hipEventDisableTiming) !=
          hipSuccess ||
      hipEventCreateWithFlags(&e->ev_up_, hipEventDisableTiming) !=
          hipSuccess ||
      hipEventCreateWithFlags(&e->ev_shared_, hipEventDisableTiming) !=
          hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = "failed to create MoE FFN side streams/events";
    }
    return nullptr;
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
    const std::size_t slots = kMaxVerifyRows - 1U;
    for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
      if (!c_.IsMlaLayer(il)) {
        bytes += slots *
                 (3U * (c_.ssm_conv_kernel - 1U) * c_.SsmInnerSize() +
                  static_cast<std::size_t>(c_.num_heads) * c_.ssm_head_dim *
                      c_.ssm_head_dim) *
                 sizeof(float);
      }
    }
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
    s->kda_conv_snap_.assign(c_.num_layers, nullptr);
    s->kda_state_snap_.assign(c_.num_layers, nullptr);
    const std::size_t slots = kMaxVerifyRows - 1U;
    for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
      if (c_.IsMlaLayer(il)) {
        continue;
      }
      s->kda_conv_snap_[il] = s->AllocFloats(
          slots * 3U * (c_.ssm_conv_kernel - 1U) * c_.SsmInnerSize());
      s->kda_state_snap_[il] =
          s->AllocFloats(slots * static_cast<std::size_t>(c_.num_heads) *
                         c_.ssm_head_dim * c_.ssm_head_dim);
    }
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

  // The latent half is RMS-normalized straight into the cache row; the rope
  // half is rotated into the kpe row beside it. Each q rope tail is rotated
  // out of its 192-wide head row directly into the absorbed-query row.
  RmsNorm(mla_kv_a_, l.attn_kv_a_norm.f32(), c_.rms_eps,
          kv_cache + static_cast<std::size_t>(pos) * kv_rank, kv_rank, nullptr);
  RopeTo(mla_kv_a_ + kv_rank, kpe_cache + static_cast<std::size_t>(pos) * rot,
         1U, rot, rot, rot, pos, c_.rope_theta, nullptr);
  RopeTo(mla_q_ + nope, mla_q_abs_ + kv_rank, heads, c_.mla_head_dim,
         kv_rank + rot, rot, pos, c_.rope_theta, nullptr);

  // Absorbed query: wk_b maps each head's nope half to the latent rank; the
  // rope tail already sits behind it.
  (void)GemvTypeOf(l.attn_k_b.type, &type);
  for (std::uint32_t h = 0; h < heads; ++h) {
    Gemv(l.attn_k_b.Expert(h), type, kv_rank, nope, l.attn_k_b.row_bytes,
         mla_q_ + static_cast<std::size_t>(h) * c_.mla_head_dim,
         mla_q_abs_ + static_cast<std::size_t>(h) * (kv_rank + rot), nullptr);
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

  // Overlap the MoE branches with the serial router -> gate -> down chain on
  // non-blocking side streams. The always-on shared expert only reads x, so
  // it starts immediately; routed `up` waits for the expert ids. Main joins
  // ev_up before the routed Swiglu and ev_shared before the final Add, so
  // the dependency graph - and the arithmetic - match the serial order.
  (void)hipEventRecord(ev_shared_fork_, nullptr);
  (void)hipStreamWaitEvent(moe_shared_stream_, ev_shared_fork_, 0);
  (void)GemvTypeOf(l.shexp_gate.type, &type);
  Gemv(l.shexp_gate.data, type, l.shexp_gate.rows, l.shexp_gate.cols,
       l.shexp_gate.row_bytes, x, sh_gate_, moe_shared_stream_);
  (void)GemvTypeOf(l.shexp_up.type, &type);
  Gemv(l.shexp_up.data, type, l.shexp_up.rows, l.shexp_up.cols,
       l.shexp_up.row_bytes, x, sh_up_, moe_shared_stream_);
  SwigluClamped(sh_gate_, sh_up_, sh_inter_, c_.shared_expert_ff,
                c_.swiglu_clamp_shexp[il], moe_shared_stream_);
  (void)GemvTypeOf(l.shexp_down.type, &type);
  Gemv(l.shexp_down.data, type, l.shexp_down.rows, l.shexp_down.cols,
       l.shexp_down.row_bytes, sh_inter_, sh_down_, moe_shared_stream_);
  (void)hipEventRecord(ev_shared_, moe_shared_stream_);

  (void)GemvTypeOf(l.router.type, &type);
  Gemv(l.router.data, type, l.router.rows, l.router.cols, l.router.row_bytes, x,
       moe_logits_, nullptr);
  RouterTopK(moe_logits_, l.exp_probs_b.f32(), c_.num_experts, c_.expert_groups,
             c_.expert_groups_used, used, c_.expert_weights_scale, moe_ids_,
             moe_weights_, nullptr);
  (void)hipEventRecord(ev_up_fork_, nullptr);
  (void)hipStreamWaitEvent(moe_up_stream_, ev_up_fork_, 0);
  (void)GemvTypeOf(l.ffn_up_exps.type, &type);
  GemvGrouped(l.ffn_up_exps.data, type,
              l.ffn_up_exps.row_bytes * l.ffn_up_exps.rows, moe_ids_, used,
              used, ff, c_.hidden_size, x, 0U, moe_up_, moe_up_stream_);
  (void)hipEventRecord(ev_up_, moe_up_stream_);

  (void)GemvTypeOf(l.ffn_gate_exps.type, &type);
  GemvGrouped(l.ffn_gate_exps.data, type,
              l.ffn_gate_exps.row_bytes * l.ffn_gate_exps.rows, moe_ids_, used,
              used, ff, c_.hidden_size, x, 0U, moe_gate_, nullptr);
  (void)hipStreamWaitEvent(nullptr, ev_up_, 0);
  SwigluClamped(moe_gate_, moe_up_, moe_inter_, used * ff,
                c_.swiglu_clamp_exp[il], nullptr);
  (void)GemvTypeOf(l.ffn_down_exps.type, &type);
  GemvGrouped(l.ffn_down_exps.data, type,
              l.ffn_down_exps.row_bytes * l.ffn_down_exps.rows, moe_ids_, used,
              1U, hid, ff, moe_inter_, ff, moe_down_, nullptr);
  RowWeightedSum(moe_down_, moe_weights_, out, used, hid, nullptr);
  (void)hipStreamWaitEvent(nullptr, ev_shared_, 0);
  Add(out, sh_down_, out, hid, nullptr);
}

void Executor::KdaBatch(Session& session, std::uint32_t il,
                        const DeviceLayer& l, std::uint32_t rows) {
  const std::uint32_t inner = c_.SsmInnerSize();
  const std::uint32_t heads = c_.num_heads;
  const std::uint32_t d = c_.ssm_head_dim;
  const std::uint32_t kern = c_.ssm_conv_kernel;
  const std::size_t hid = c_.hidden_size;
  GemvType type;
  const float* in = pf_normed_;
  GemvTypeOf(l.ssm_q.type, &type);
  GemvRows(l.ssm_q.data, type, rows, l.ssm_q.rows, l.ssm_q.cols,
           l.ssm_q.row_bytes, in, static_cast<std::uint32_t>(hid), pf_kda_q_,
           inner, nullptr);
  GemvTypeOf(l.ssm_k.type, &type);
  GemvRows(l.ssm_k.data, type, rows, l.ssm_k.rows, l.ssm_k.cols,
           l.ssm_k.row_bytes, in, static_cast<std::uint32_t>(hid), pf_kda_k_,
           inner, nullptr);
  GemvTypeOf(l.ssm_v.type, &type);
  GemvRows(l.ssm_v.data, type, rows, l.ssm_v.rows, l.ssm_v.cols,
           l.ssm_v.row_bytes, in, static_cast<std::uint32_t>(hid), pf_kda_v_,
           inner, nullptr);
  GemvTypeOf(l.ssm_f.type, &type);
  GemvRows(l.ssm_f.data, type, rows, l.ssm_f.rows, l.ssm_f.cols,
           l.ssm_f.row_bytes, in, static_cast<std::uint32_t>(hid), pf_kda_f_,
           inner, nullptr);
  GemvTypeOf(l.ssm_g.type, &type);
  GemvRows(l.ssm_g.data, type, rows, l.ssm_g.rows, l.ssm_g.cols,
           l.ssm_g.row_bytes, in, static_cast<std::uint32_t>(hid), pf_kda_g_,
           inner, nullptr);
  GemvTypeOf(l.ssm_beta.type, &type);
  GemvRows(l.ssm_beta.data, type, rows, l.ssm_beta.rows, l.ssm_beta.cols,
           l.ssm_beta.row_bytes, in, static_cast<std::uint32_t>(hid),
           pf_kda_beta_, heads, nullptr);

  // The recurrent parts stay sequential: each row advances the conv history
  // and the delta state, and the first rows-1 states are snapshotted so
  // RollbackVerify can commit any prefix.
  const std::size_t hist_stride = static_cast<std::size_t>(kern - 1U) * inner;
  const std::size_t conv_slot = 3U * hist_stride;
  const std::size_t state_elems = static_cast<std::size_t>(heads) * d * d;
  float* hist = session.kda_conv_[il];
  (void)GemvTypeOf(l.ssm_out.type, &type);
  for (std::uint32_t r = 0; r < rows; ++r) {
    Add(pf_kda_f_ + static_cast<std::size_t>(r) * inner, l.ssm_dt.f32(),
        pf_kda_f_ + static_cast<std::size_t>(r) * inner, inner, nullptr);
    ConvSilu(l.ssm_conv_q.f32(),
             pf_kda_q_ + static_cast<std::size_t>(r) * inner, hist, kda_conv_q_,
             inner, kern, nullptr);
    ConvSilu(l.ssm_conv_k.f32(),
             pf_kda_k_ + static_cast<std::size_t>(r) * inner,
             hist + hist_stride, kda_conv_k_, inner, kern, nullptr);
    ConvSilu(l.ssm_conv_v.f32(),
             pf_kda_v_ + static_cast<std::size_t>(r) * inner,
             hist + 2U * hist_stride, kda_conv_v_, inner, kern, nullptr);
    KdaNormHeads(kda_conv_q_, kda_conv_k_, heads, d, c_.rms_eps,
                 1.0F / std::sqrt(static_cast<float>(d)), nullptr);
    KdaDelta(kda_conv_q_, kda_conv_k_, kda_conv_v_,
             pf_kda_f_ + static_cast<std::size_t>(r) * inner,
             pf_kda_beta_ + static_cast<std::size_t>(r) * heads, l.ssm_a.f32(),
             l.ssm_norm.f32(), pf_kda_g_ + static_cast<std::size_t>(r) * inner,
             session.kda_state_[il], kda_attn_, heads, d, c_.rms_eps,
             c_.kda_gate_lower_bound, nullptr);
    if (r + 1U < rows) {
      (void)hipMemcpyAsync(session.kda_conv_snap_[il] + r * conv_slot, hist,
                           conv_slot * sizeof(float), hipMemcpyDeviceToDevice,
                           nullptr);
      (void)hipMemcpyAsync(session.kda_state_snap_[il] + r * state_elems,
                           session.kda_state_[il], state_elems * sizeof(float),
                           hipMemcpyDeviceToDevice, nullptr);
    }
    Gemv(l.ssm_out.data, type, l.ssm_out.rows, l.ssm_out.cols,
         l.ssm_out.row_bytes, kda_attn_,
         pf_attn_ + static_cast<std::size_t>(r) * hid, nullptr);
  }
}

void Executor::MlaBatch(const DeviceLayer& l, float* kv_cache, float* kpe_cache,
                        std::uint32_t base_pos, std::uint32_t rows) {
  const std::uint32_t heads = c_.num_heads;
  const std::uint32_t nope = c_.MlaNopeDim();
  const std::uint32_t rot = c_.rotary_dim;
  const std::uint32_t kv_rank = c_.kv_lora_rank;
  const std::uint32_t v_dim = c_.mla_value_dim;
  const std::size_t hid = c_.hidden_size;
  const std::size_t qrow = static_cast<std::size_t>(heads) * c_.mla_head_dim;
  const std::size_t arow = c_.MlaKeyDim();
  const std::size_t absrow = static_cast<std::size_t>(heads) * c_.MlaKeyDim();
  const std::size_t hrow = static_cast<std::size_t>(heads) * v_dim;
  const float scale = 1.0F / std::sqrt(static_cast<float>(c_.mla_head_dim));
  GemvType type;
  GemvTypeOf(l.attn_q.type, &type);
  GemvRows(l.attn_q.data, type, rows, l.attn_q.rows, l.attn_q.cols,
           l.attn_q.row_bytes, pf_normed_, static_cast<std::uint32_t>(hid),
           pf_mla_q_, static_cast<std::uint32_t>(qrow), nullptr);
  GemvTypeOf(l.attn_kv_a.type, &type);
  GemvRows(l.attn_kv_a.data, type, rows, l.attn_kv_a.rows, l.attn_kv_a.cols,
           l.attn_kv_a.row_bytes, pf_normed_, static_cast<std::uint32_t>(hid),
           pf_mla_a_, static_cast<std::uint32_t>(arow), nullptr);
  for (std::uint32_t r = 0; r < rows; ++r) {
    const std::uint32_t pos = base_pos + r;
    float* a = pf_mla_a_ + static_cast<std::size_t>(r) * arow;
    float* q = pf_mla_q_ + static_cast<std::size_t>(r) * qrow;
    float* abs = pf_mla_abs_ + static_cast<std::size_t>(r) * absrow;
    RmsNorm(a, l.attn_kv_a_norm.f32(), c_.rms_eps,
            kv_cache + static_cast<std::size_t>(pos) * kv_rank, kv_rank,
            nullptr);
    RopeTo(a + kv_rank, kpe_cache + static_cast<std::size_t>(pos) * rot, 1U,
           rot, rot, rot, pos, c_.rope_theta, nullptr);
    RopeTo(q + nope, abs + kv_rank, heads, c_.mla_head_dim, c_.MlaKeyDim(), rot,
           pos, c_.rope_theta, nullptr);
    (void)GemvTypeOf(l.attn_k_b.type, &type);
    for (std::uint32_t h = 0; h < heads; ++h) {
      Gemv(l.attn_k_b.Expert(h), type, kv_rank, nope, l.attn_k_b.row_bytes,
           q + static_cast<std::size_t>(h) * c_.mla_head_dim,
           abs + static_cast<std::size_t>(h) * c_.MlaKeyDim(), nullptr);
    }
    const std::uint32_t n_pos = pos + 1U;
    const std::uint32_t tiles = MlaDecodeTiles(n_pos, kMlaTile);
    MlaDecodePartial(abs, kv_cache, kpe_cache, n_pos, kMlaTile, scale, mla_ws_,
                     heads, kv_rank, rot, nullptr);
    MlaDecodeCombine(mla_ws_, tiles, heads, kv_rank, mla_latent_, nullptr);
    GemvTypeOf(l.attn_v_b.type, &type);
    float* head = pf_mla_head_ + static_cast<std::size_t>(r) * hrow;
    for (std::uint32_t h = 0; h < heads; ++h) {
      Gemv(l.attn_v_b.Expert(h), type, v_dim, kv_rank, l.attn_v_b.row_bytes,
           mla_latent_ + static_cast<std::size_t>(h) * kv_rank,
           head + static_cast<std::size_t>(h) * v_dim, nullptr);
    }
    GemvTypeOf(l.attn_gate.type, &type);
    Gemv(l.attn_gate.data, type, l.attn_gate.rows, l.attn_gate.cols,
         l.attn_gate.row_bytes, pf_normed_ + static_cast<std::size_t>(r) * hid,
         mla_gate_, nullptr);
    SigmoidInPlace(mla_gate_, heads, nullptr);
    RowScale(head, mla_gate_, heads, v_dim, nullptr);
  }
  GemvTypeOf(l.attn_out.type, &type);
  GemvRows(l.attn_out.data, type, rows, l.attn_out.rows, l.attn_out.cols,
           l.attn_out.row_bytes, pf_mla_head_, static_cast<std::uint32_t>(hrow),
           pf_attn_, static_cast<std::uint32_t>(hid), nullptr);
}

void Executor::FfnBatch(const DeviceLayer& l, std::uint32_t il,
                        std::uint32_t rows) {
  const std::size_t hid = c_.hidden_size;
  if (c_.IsDenseFfnLayer(il)) {
    // Dense layers are a small minority; the single-row path keeps their
    // wide scratch reusable and the weight traffic is identical.
    for (std::uint32_t r = 0; r < rows; ++r) {
      Ffn(l, il, pf_normed_ + static_cast<std::size_t>(r) * hid,
          pf_ffn_ + static_cast<std::size_t>(r) * hid);
    }
    return;
  }
  const std::uint32_t used = c_.num_experts_used;
  const std::uint32_t ff = c_.expert_ff;
  const std::size_t slots = rows * used;
  const std::size_t slot_gate = static_cast<std::size_t>(used) * ff;
  const std::size_t slot_down = static_cast<std::size_t>(used) * hid;
  const std::size_t shared = c_.shared_expert_ff;
  GemvType type;
  GemvTypeOf(l.router.type, &type);
  GemvRows(l.router.data, type, rows, l.router.rows, l.router.cols,
           l.router.row_bytes, pf_normed_, static_cast<std::uint32_t>(hid),
           pf_moe_log_, c_.num_experts, nullptr);
  for (std::uint32_t r = 0; r < rows; ++r) {
    RouterTopK(pf_moe_log_ + static_cast<std::size_t>(r) * c_.num_experts,
               l.exp_probs_b.f32(), c_.num_experts, c_.expert_groups,
               c_.expert_groups_used, used, c_.expert_weights_scale,
               pf_moe_ids_ + static_cast<std::size_t>(r) * used,
               pf_moe_weights_ + static_cast<std::size_t>(r) * used, nullptr);
  }
  GemvTypeOf(l.ffn_gate_exps.type, &type);
  GemvGrouped(l.ffn_gate_exps.data, type,
              l.ffn_gate_exps.row_bytes * l.ffn_gate_exps.rows, pf_moe_ids_,
              slots, used, ff, c_.hidden_size, pf_normed_,
              static_cast<std::uint32_t>(hid), pf_moe_gate_, nullptr);
  GemvTypeOf(l.ffn_up_exps.type, &type);
  GemvGrouped(l.ffn_up_exps.data, type,
              l.ffn_up_exps.row_bytes * l.ffn_up_exps.rows, pf_moe_ids_, slots,
              used, ff, c_.hidden_size, pf_normed_,
              static_cast<std::uint32_t>(hid), pf_moe_up_, nullptr);
  SwigluClamped(pf_moe_gate_, pf_moe_up_, pf_moe_inter_, slots * ff,
                c_.swiglu_clamp_exp[il], nullptr);
  GemvTypeOf(l.ffn_down_exps.type, &type);
  GemvGrouped(l.ffn_down_exps.data, type,
              l.ffn_down_exps.row_bytes * l.ffn_down_exps.rows, pf_moe_ids_,
              slots, 1U, static_cast<std::uint32_t>(hid), ff, pf_moe_inter_, ff,
              pf_moe_down_, nullptr);
  for (std::uint32_t r = 0; r < rows; ++r) {
    RowWeightedSum(pf_moe_down_ + static_cast<std::size_t>(r) * slot_down,
                   pf_moe_weights_ + static_cast<std::size_t>(r) * used,
                   pf_ffn_ + static_cast<std::size_t>(r) * hid, used,
                   static_cast<std::uint32_t>(hid), nullptr);
  }

  // Always-on shared expert for every row.
  GemvTypeOf(l.shexp_gate.type, &type);
  GemvRows(l.shexp_gate.data, type, rows, l.shexp_gate.rows, l.shexp_gate.cols,
           l.shexp_gate.row_bytes, pf_normed_, static_cast<std::uint32_t>(hid),
           pf_sh_gate_, static_cast<std::uint32_t>(shared), nullptr);
  GemvTypeOf(l.shexp_up.type, &type);
  GemvRows(l.shexp_up.data, type, rows, l.shexp_up.rows, l.shexp_up.cols,
           l.shexp_up.row_bytes, pf_normed_, static_cast<std::uint32_t>(hid),
           pf_sh_up_, static_cast<std::uint32_t>(shared), nullptr);
  SwigluClamped(pf_sh_gate_, pf_sh_up_, pf_sh_inter_, rows * shared,
                c_.swiglu_clamp_shexp[il], nullptr);
  GemvTypeOf(l.shexp_down.type, &type);
  GemvRows(l.shexp_down.data, type, rows, l.shexp_down.rows, l.shexp_down.cols,
           l.shexp_down.row_bytes, pf_sh_inter_,
           static_cast<std::uint32_t>(shared), pf_sh_down_,
           static_cast<std::uint32_t>(hid), nullptr);
  for (std::uint32_t r = 0; r < rows; ++r) {
    Add(pf_ffn_ + static_cast<std::size_t>(r) * hid,
        pf_sh_down_ + static_cast<std::size_t>(r) * hid,
        pf_ffn_ + static_cast<std::size_t>(r) * hid,
        static_cast<std::uint32_t>(hid), nullptr);
  }
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

void Executor::PrefillKda(Session& session, std::uint32_t il,
                          const DeviceLayer& l, std::uint32_t tokens) {
  const std::uint32_t inner = c_.SsmInnerSize();
  const std::uint32_t heads = c_.num_heads;
  const std::uint32_t d = c_.ssm_head_dim;
  const std::uint32_t kern = c_.ssm_conv_kernel;
  const std::size_t hid = c_.hidden_size;
  const std::size_t hist_stride = static_cast<std::size_t>(kern - 1U) * inner;
  WeightType w;
  float* hist = session.kda_conv_[il];

  NarrowActivations(pf_normed_, pf_xh_, false, tokens * hid, nullptr);
  (void)WeightTypeOf(l.ssm_q.type, &w);
  tier_failure_ |=
      !DenseF16Gemm(l.ssm_q.data, w, pf_xh_, pf_drows_, pf_dbounds_, pf_dtiles_,
                    tokens, pf_kda_q_, nullptr, inner, hid, 0U, nullptr);
  (void)WeightTypeOf(l.ssm_k.type, &w);
  tier_failure_ |=
      !DenseF16Gemm(l.ssm_k.data, w, pf_xh_, pf_drows_, pf_dbounds_, pf_dtiles_,
                    tokens, pf_kda_k_, nullptr, inner, hid, 0U, nullptr);
  (void)WeightTypeOf(l.ssm_v.type, &w);
  tier_failure_ |=
      !DenseF16Gemm(l.ssm_v.data, w, pf_xh_, pf_drows_, pf_dbounds_, pf_dtiles_,
                    tokens, pf_kda_v_, nullptr, inner, hid, 0U, nullptr);
  (void)WeightTypeOf(l.ssm_f.type, &w);
  tier_failure_ |=
      !DenseF16Gemm(l.ssm_f.data, w, pf_xh_, pf_drows_, pf_dbounds_, pf_dtiles_,
                    tokens, pf_kda_f_, nullptr, inner, hid, 0U, nullptr);
  (void)WeightTypeOf(l.ssm_g.type, &w);
  tier_failure_ |=
      !DenseF16Gemm(l.ssm_g.data, w, pf_xh_, pf_drows_, pf_dbounds_, pf_dtiles_,
                    tokens, pf_kda_g_, nullptr, inner, hid, 0U, nullptr);
  // beta is stored in F32, which the WMMA tier cannot decode; one grouped
  // GEMV over zero-filled expert ids walks every token row at once.
  GemvType beta;
  (void)GemvTypeOf(l.ssm_beta.type, &beta);
  GemvGrouped(l.ssm_beta.data, beta, 0U, pf_zero_ids_, tokens, 1U, heads,
              static_cast<std::uint32_t>(hid), pf_normed_,
              static_cast<std::uint32_t>(hid), pf_kda_beta_, nullptr);
  AddBiasRows(pf_kda_f_, l.ssm_dt.f32(), tokens, inner, nullptr);

  // The convolved streams land on separate planes (reading the raw projection
  // rows of earlier tokens makes an in-place run wrong). The next conv
  // history is staged outside the session buffer, then swapped in; the
  // staging must be written before the swap overwrites the old rows.
  KdaConvSiluPrefill(l.ssm_conv_q.f32(), pf_kda_q_, hist, pf_cq_, tokens, inner,
                     kern, nullptr);
  KdaConvSiluPrefill(l.ssm_conv_k.f32(), pf_kda_k_, hist + hist_stride, pf_ck_,
                     tokens, inner, kern, nullptr);
  KdaConvSiluPrefill(l.ssm_conv_v.f32(), pf_kda_v_, hist + 2U * hist_stride,
                     pf_cv_, tokens, inner, kern, nullptr);
  const float* proj[3] = {pf_kda_q_, pf_kda_k_, pf_kda_v_};
  for (std::uint32_t s = 0; s < 3U; ++s) {
    KdaHistoryUpdate(proj[s], hist + s * hist_stride,
                     pf_kda_hist_ + s * hist_stride, tokens, inner, kern,
                     nullptr);
    (void)hipMemcpyAsync(hist + s * hist_stride, pf_kda_hist_ + s * hist_stride,
                         hist_stride * sizeof(float), hipMemcpyDeviceToDevice,
                         nullptr);
  }

  KdaNormHeadsPrefill(pf_cq_, pf_ck_, tokens, heads, d, c_.rms_eps,
                      1.0F / std::sqrt(static_cast<float>(d)), nullptr);
  KdaPrep(pf_kda_f_, l.ssm_a.f32(), pf_kda_beta_, pf_cq_, pf_ck_,
          c_.kda_gate_lower_bound, pf_decay_, pf_bpre_, pf_kq_, tokens, heads,
          d, nullptr);
  KdaDeltaLoop(pf_cq_, pf_ck_, pf_cv_, pf_decay_, pf_bpre_, pf_kq_,
               session.kda_state_[il], pf_o_, tokens, heads, d, nullptr, 0U,
               nullptr);
  KdaOutNormPrefill(pf_o_, pf_kda_g_, l.ssm_norm.f32(), tokens, heads, d,
                    c_.rms_eps, nullptr);
  NarrowActivations(pf_o_, pf_xh_, false, tokens * inner, nullptr);
  (void)WeightTypeOf(l.ssm_out.type, &w);
  tier_failure_ |= !DenseF16Gemm(l.ssm_out.data, w, pf_xh_, pf_drows_,
                                 pf_dbounds_, pf_dtiles_, tokens, pf_attn_,
                                 nullptr, hid, inner, 0U, nullptr);
}

void Executor::PrefillMla(const DeviceLayer& l, float* kv_cache,
                          float* kpe_cache, std::uint32_t base_pos,
                          std::uint32_t tokens) {
  const std::uint32_t heads = c_.num_heads;
  const std::uint32_t nope = c_.MlaNopeDim();
  const std::uint32_t rot = c_.rotary_dim;
  const std::uint32_t kv_rank = c_.kv_lora_rank;
  const std::uint32_t v_dim = c_.mla_value_dim;
  const std::size_t hid = c_.hidden_size;
  const float scale = 1.0F / std::sqrt(static_cast<float>(c_.mla_head_dim));
  WeightType w;

  NarrowActivations(pf_normed_, pf_xh_, false, tokens * hid, nullptr);
  (void)WeightTypeOf(l.attn_q.type, &w);
  tier_failure_ |= !DenseF16Gemm(
      l.attn_q.data, w, pf_xh_, pf_drows_, pf_dbounds_, pf_dtiles_, tokens,
      pf_mla_q_, nullptr, static_cast<std::size_t>(heads) * c_.mla_head_dim,
      hid, 0U, nullptr);
  (void)WeightTypeOf(l.attn_kv_a.type, &w);
  tier_failure_ |= !DenseF16Gemm(l.attn_kv_a.data, w, pf_xh_, pf_drows_,
                                 pf_dbounds_, pf_dtiles_, tokens, pf_mla_a_,
                                 nullptr, c_.MlaKeyDim(), hid, 0U, nullptr);
  (void)WeightTypeOf(l.attn_gate.type, &w);
  tier_failure_ |= !DenseF16Gemm(l.attn_gate.data, w, pf_xh_, pf_drows_,
                                 pf_dbounds_, pf_dtiles_, tokens, pf_mla_pgate_,
                                 nullptr, heads, hid, 0U, nullptr);

  // RoPE, the latent cache write and the absorbed query over the whole chunk.
  MlaQRopeRows(pf_mla_q_, pf_mla_qrp_, base_pos, tokens, heads, c_.mla_head_dim,
               nope, rot, c_.rope_theta, nullptr);
  MlaKvANormRopeRows(pf_mla_a_, l.attn_kv_a_norm.f32(), c_.rms_eps, kv_cache,
                     kpe_cache, base_pos, tokens, kv_rank, rot, c_.rope_theta,
                     nullptr);
  // The absorbed per-head projections run on the F16 WMMA tier with the
  // heads as virtual experts: every token of the chunk visits every head,
  // so the buckets are uniform, one compact map serves both GEMMs and the
  // tile list builds on the device (no host fetch of the histogram).
  const std::uint32_t head_tile_rows = 64U;
  const std::uint32_t n_head_tiles =
      heads * ((tokens + head_tile_rows - 1U) / head_tile_rows);
  ExpertCounts(pf_head_ids_, reinterpret_cast<std::uint32_t*>(pf_counts_),
               tokens * heads, heads, nullptr);
  RoutedCompact(pf_head_ids_, reinterpret_cast<std::uint32_t*>(pf_counts_),
                pf_hbounds_, pf_hcursors_, pf_rows_tok_, pf_rows_slot_, tokens,
                heads, heads, nullptr);
  HeadTiles(pf_htiles_, heads, tokens, head_tile_rows, nullptr);
  // k_b contracts over the nope part of each raw query row; the gathered
  // and scattered rows are indexed by slot, matching both the [slot][nope]
  // narrowed query plane and the [slot][kv_rank] absorbed query plane.
  NarrowRowsStripped(pf_mla_q_, pf_qnc_, tokens * heads, c_.mla_head_dim, nope,
                     nullptr);
  (void)WeightTypeOf(l.attn_k_b.type, &w);
  tier_failure_ |= !RoutedF16Gemm(
      l.attn_k_b.data, w, pf_qnc_, pf_htiles_, n_head_tiles, head_tile_rows,
      pf_hbounds_, pf_rows_slot_, pf_rows_slot_, nullptr, 0.0F, pf_mla_pabs_,
      nullptr, kv_rank, nope, 0U, nullptr);
  MlaPrefill(pf_mla_pabs_, pf_mla_qrp_, kv_cache, kpe_cache, pf_mla_qlatent_,
             base_pos, tokens, heads, kv_rank, rot, scale, nullptr);
  // v_b contracts over the [token][kv_rank] attention latent through the
  // token gather rows of the same head-bucket map.
  NarrowActivations(pf_mla_qlatent_, pf_xh_, false, tokens * kv_rank, nullptr);
  (void)WeightTypeOf(l.attn_v_b.type, &w);
  tier_failure_ |= !RoutedF16Gemm(
      l.attn_v_b.data, w, pf_xh_, pf_htiles_, n_head_tiles, head_tile_rows,
      pf_hbounds_, pf_rows_tok_, pf_rows_slot_, nullptr, 0.0F, pf_mla_head_,
      nullptr, v_dim, kv_rank, 0U, nullptr);
  SigmoidInPlace(pf_mla_pgate_, tokens * heads, nullptr);
  RowScale(pf_mla_head_, pf_mla_pgate_, tokens * heads, v_dim, nullptr);

  NarrowActivations(pf_mla_head_, pf_xh_, false, tokens * heads * v_dim,
                    nullptr);
  (void)WeightTypeOf(l.attn_out.type, &w);
  tier_failure_ |=
      !DenseF16Gemm(l.attn_out.data, w, pf_xh_, pf_drows_, pf_dbounds_,
                    pf_dtiles_, tokens, pf_attn_, nullptr, hid,
                    static_cast<std::size_t>(heads) * v_dim, 0U, nullptr);
}

void Executor::PrefillFfn(const DeviceLayer& l, std::uint32_t il,
                          std::uint32_t tokens) {
  const std::uint32_t hid = c_.hidden_size;
  const std::uint32_t used = c_.num_experts_used;
  const std::uint32_t ff = c_.expert_ff;
  WeightType w;

  NarrowActivations(pf_normed_, pf_xh_, false, tokens * hid, nullptr);
  if (c_.IsDenseFfnLayer(il)) {
    (void)WeightTypeOf(l.ffn_gate.type, &w);
    tier_failure_ |= !DenseF16Gemm(l.ffn_gate.data, w, pf_xh_, pf_drows_,
                                   pf_dbounds_, pf_dtiles_, tokens, pf_dn_gate_,
                                   nullptr, c_.dense_ff, hid, 0U, nullptr);
    (void)WeightTypeOf(l.ffn_up.type, &w);
    tier_failure_ |= !DenseF16Gemm(l.ffn_up.data, w, pf_xh_, pf_drows_,
                                   pf_dbounds_, pf_dtiles_, tokens, pf_dn_up_,
                                   nullptr, c_.dense_ff, hid, 0U, nullptr);
    SwigluClamped(pf_dn_gate_, pf_dn_up_, pf_dn_inter_,
                  tokens * static_cast<std::uint32_t>(c_.dense_ff),
                  c_.swiglu_clamp_shexp[il], nullptr);
    NarrowActivations(pf_dn_inter_, pf_xh_, false,
                      tokens * static_cast<std::uint32_t>(c_.dense_ff),
                      nullptr);
    (void)WeightTypeOf(l.ffn_down.type, &w);
    tier_failure_ |= !DenseF16Gemm(l.ffn_down.data, w, pf_xh_, pf_drows_,
                                   pf_dbounds_, pf_dtiles_, tokens, pf_ffn_,
                                   nullptr, hid, c_.dense_ff, 0U, nullptr);
    return;
  }

  // Router over the chunk: the [num_experts x hid] matrix was retyped to
  // binary16 at upload, so the dense WMMA tier reads the chunk's F16 rows
  // (same row/tile maps as the dense FFN path) and writes F32 logits.
  (void)WeightTypeOf(l.router.type, &w);
  tier_failure_ |= !DenseF16Gemm(l.router.data, w, pf_xh_, pf_drows_,
                                 pf_dbounds_, pf_dtiles_, tokens, pf_moe_log_,
                                 nullptr, c_.num_experts, hid, 0U, nullptr);
  RouterTopKRows(pf_moe_log_, l.exp_probs_b.f32(), c_.num_experts,
                 c_.expert_groups, c_.expert_groups_used, used,
                 c_.expert_weights_scale, pf_moe_ids_, pf_moe_weights_, tokens,
                 nullptr);
  ExpertCounts(pf_moe_ids_, reinterpret_cast<std::uint32_t*>(pf_counts_),
               tokens * used, c_.num_experts, nullptr);
  RoutedCompact(pf_moe_ids_, reinterpret_cast<std::uint32_t*>(pf_counts_),
                pf_bounds_, pf_cursors_, pf_rows_tok_, pf_rows_slot_, tokens,
                used, c_.num_experts, nullptr);

  // The tile list depends on the per-expert row counts; one device sync per
  // MoE layer fetches the (tiny) histogram to build it on the host.
  const std::uint32_t tile_rows = 64U;
  (void)hipDeviceSynchronize();
  counts_host_.resize(c_.num_experts);
  (void)hipMemcpy(counts_host_.data(), pf_counts_,
                  c_.num_experts * sizeof(std::uint32_t),
                  hipMemcpyDeviceToHost);
  tiles_host_.clear();
  for (std::uint32_t e = 0; e < c_.num_experts; ++e) {
    const std::uint32_t n_e = (counts_host_[e] + tile_rows - 1U) / tile_rows;
    for (std::uint32_t j = 0; j < n_e; ++j) {
      tiles_host_.push_back(static_cast<std::int32_t>(e | (j << 16U)));
    }
  }
  const auto n_tiles = static_cast<std::uint32_t>(tiles_host_.size());
  (void)hipMemcpyAsync(pf_tiles_, tiles_host_.data(),
                       tiles_host_.size() * sizeof(std::int32_t),
                       hipMemcpyHostToDevice, nullptr);

  (void)WeightTypeOf(l.ffn_gate_exps.type, &w);
  tier_failure_ |= !RoutedGatedF16Gemm(
      l.ffn_gate_exps.data, l.ffn_up_exps.data, w, pf_xh_, pf_tiles_, n_tiles,
      tile_rows, pf_bounds_, pf_rows_tok_, pf_rows_slot_,
      c_.swiglu_clamp_exp[il], pf_interh_, ff, hid, nullptr);
  (void)WeightTypeOf(l.ffn_down_exps.type, &w);
  tier_failure_ |=
      !RoutedF16Gemm(l.ffn_down_exps.data, w, pf_interh_, pf_tiles_, n_tiles,
                     tile_rows, pf_bounds_, pf_rows_slot_, pf_rows_slot_,
                     nullptr, 0.0F, pf_pdown_, nullptr, hid, ff, hid, nullptr);
  RowWeightedSumTokens(pf_pdown_, pf_moe_weights_, pf_ffn_, tokens, used, hid,
                       nullptr);

  // Always-on shared expert for every row.
  const std::size_t shared = c_.shared_expert_ff;
  (void)WeightTypeOf(l.shexp_gate.type, &w);
  tier_failure_ |= !DenseF16Gemm(l.shexp_gate.data, w, pf_xh_, pf_drows_,
                                 pf_dbounds_, pf_dtiles_, tokens, pf_sh_gate_,
                                 nullptr, shared, hid, 0U, nullptr);
  (void)WeightTypeOf(l.shexp_up.type, &w);
  tier_failure_ |= !DenseF16Gemm(l.shexp_up.data, w, pf_xh_, pf_drows_,
                                 pf_dbounds_, pf_dtiles_, tokens, pf_sh_up_,
                                 nullptr, shared, hid, 0U, nullptr);
  SwigluClamped(pf_sh_gate_, pf_sh_up_, pf_sh_inter_,
                tokens * static_cast<std::uint32_t>(shared),
                c_.swiglu_clamp_shexp[il], nullptr);
  NarrowActivations(pf_sh_inter_, pf_xh_, false, tokens * shared, nullptr);
  (void)WeightTypeOf(l.shexp_down.type, &w);
  tier_failure_ |= !DenseF16Gemm(l.shexp_down.data, w, pf_xh_, pf_drows_,
                                 pf_dbounds_, pf_dtiles_, tokens, pf_sh_down_,
                                 nullptr, hid, shared, 0U, nullptr);
  Add(pf_ffn_, pf_sh_down_, pf_ffn_, tokens * hid, nullptr);
}

void Executor::PrefillChunk(Session& session, std::uint32_t tokens,
                            std::uint32_t base_pos) {
  const std::uint32_t hid = c_.hidden_size;
  tier_failure_ = false;
  DenseRowPrepare(pf_drows_, pf_dbounds_, pf_dtiles_, tokens, nullptr);
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const DeviceLayer& l = model_.layers()[il];
    RmsNormRows(pf_x_, l.attn_norm.f32(), c_.rms_eps, pf_normed_, tokens, hid,
                nullptr);
    if (c_.IsMlaLayer(il)) {
      PrefillMla(l, session.mla_kv_[il], session.mla_kpe_[il], base_pos,
                 tokens);
    } else {
      PrefillKda(session, il, l, tokens);
    }
    Add(pf_x_, pf_attn_, pf_x_, tokens * hid, nullptr);
    RmsNormRows(pf_x_, l.ffn_norm.f32(), c_.rms_eps, pf_normed_, tokens, hid,
                nullptr);
    PrefillFfn(l, il, tokens);
    Add(pf_x_, pf_ffn_, pf_x_, tokens * hid, nullptr);
  }
}

void Executor::PrefillMtpDraft(Session& session,
                               const std::int32_t* next_tokens,
                               std::uint32_t rows) {
  const std::uint32_t hid = c_.hidden_size;
  const DeviceLayer& l = model_.mtp();
  GemvType type;
  WeightType w;
  DenseRowPrepare(pf_drows_, pf_dbounds_, pf_dtiles_, rows, nullptr);
  (void)GemvTypeOf(model_.token_embd().type, &type);
  for (std::uint32_t r = 0U; r < rows; ++r) {
    EmbedRow(model_.token_embd().data, type,
             static_cast<std::uint32_t>(next_tokens[r]),
             static_cast<std::uint32_t>(hid),
             mtp_pf_e_ + static_cast<std::size_t>(r) * hid, nullptr);
  }
  RmsNormRows(mtp_pf_e_, l.nextn_enorm.f32(), c_.rms_eps, mtp_pf_e_, rows, hid,
              nullptr);
  // The draft consumes the trunk hidden the per-token loop exposes as
  // h_out_: output_norm first, then the draft's own hnorm.
  RmsNormRows(pf_x_, model_.output_norm().f32(), c_.rms_eps, mtp_pf_h_, rows,
              hid, nullptr);
  RmsNormRows(mtp_pf_h_, l.nextn_hnorm.f32(), c_.rms_eps, mtp_pf_h_, rows, hid,
              nullptr);
  Concat2RowsHalf(mtp_pf_e_, mtp_pf_h_, pf_xh_, rows, hid, nullptr);
  (void)WeightTypeOf(l.nextn_eh_proj.type, &w);
  tier_failure_ |= !DenseF16Gemm(l.nextn_eh_proj.data, w, pf_xh_, pf_drows_,
                                 pf_dbounds_, pf_dtiles_, rows, mtp_pf_x_,
                                 nullptr, hid, 2U * hid, 0U, nullptr);
  RmsNormRows(mtp_pf_x_, l.attn_norm.f32(), c_.rms_eps, pf_normed_, rows, hid,
              nullptr);
  PrefillMla(l, session.mtp_kv_, session.mtp_kpe_, session.mtp_position_, rows);
  Add(mtp_pf_x_, pf_attn_, mtp_pf_x_, rows * hid, nullptr);
  RmsNormRows(mtp_pf_x_, l.ffn_norm.f32(), c_.rms_eps, pf_normed_, rows, hid,
              nullptr);
  PrefillFfn(l, c_.num_layers, rows);
  Add(mtp_pf_x_, pf_ffn_, mtp_pf_x_, rows * hid, nullptr);
  session.mtp_position_ += rows;
}

bool Executor::Prefill(Session& session, const std::int32_t* tokens,
                       std::uint32_t count, std::string* error_msg) {
  for (std::uint32_t i = 0; i < count; ++i) {
    if (tokens[i] < 0 ||
        static_cast<std::uint32_t>(tokens[i]) >= c_.vocab_size) {
      if (error_msg != nullptr) {
        *error_msg = "token id out of range";
      }
      return false;
    }
  }
  if (count > 0U && session.position() + count > session.max_context()) {
    if (error_msg != nullptr) {
      *error_msg = "context exhausted";
    }
    return false;
  }
  if (count == 0U) {
    return true;
  }
  if (!prefill_ready_) {
    // A trunk format the F16 WMMA tier cannot decode; the per-token path is
    // the reference-shaped implementation and always correct.
    for (std::uint32_t i = 0; i < count; ++i) {
      if (!Step(session, tokens[i], error_msg)) {
        return false;
      }
    }
    return true;
  }

  const std::uint32_t hid = c_.hidden_size;
  GemvType type;
  (void)GemvTypeOf(model_.token_embd().type, &type);
  std::uint32_t done = 0U;
  while (done < count) {
    const std::uint32_t rows =
        count - done > kPrefillChunkRows ? kPrefillChunkRows : count - done;
    for (std::uint32_t r = 0; r < rows; ++r) {
      EmbedRow(model_.token_embd().data, type,
               static_cast<std::uint32_t>(tokens[done + r]),
               static_cast<std::uint32_t>(hid),
               pf_x_ + static_cast<std::size_t>(r) * hid, nullptr);
    }
    PrefillChunk(session, rows, session.position_);
    session.position_ += rows;
    done += rows;
    if (tier_failure_) {
      // A kernel rejected a launch; the scratch beyond it is stale, so fail
      // loudly instead of reporting garbage as a forward pass.
      if (error_msg != nullptr) {
        *error_msg = "chunked prefill tier rejected a projection";
      }
      return false;
    }
  }

  // The final token's normalized hidden and logits, identical in shape to
  // the tail of Step.
  const std::uint32_t last = (count - 1U) % kPrefillChunkRows;
  RmsNorm(pf_x_ + static_cast<std::size_t>(last) * hid,
          model_.output_norm().f32(), c_.rms_eps, h_out_, hid, nullptr);
  (void)GemvTypeOf(model_.output().type, &type);
  Gemv(model_.output().data, type, model_.output().rows, model_.output().cols,
       model_.output().row_bytes, h_out_, logits_, nullptr);
  return true;
}

bool Executor::MtpPrefill(Session& session, const std::int32_t* tokens,
                          std::uint32_t count, std::string* error_msg) {
  const auto fail = [&](const char* msg) {
    if (error_msg != nullptr) {
      *error_msg = msg;
    }
    return false;
  };
  for (std::uint32_t i = 0; i < count; ++i) {
    if (tokens[i] < 0 ||
        static_cast<std::uint32_t>(tokens[i]) >= c_.vocab_size) {
      return fail("token id out of range");
    }
  }
  if (count > 0U && session.position() + count > session.max_context()) {
    return fail("context exhausted");
  }
  if (count == 0U) {
    return true;
  }
  if (!prefill_ready_) {
    // Same reference-shaped fallback as Prefill, interleaved with the
    // per-token draft pairs the spec session needs. Draft logits are not
    // needed mid-prefill, so the pairs run the head-free MtpAdvance.
    if (!Step(session, tokens[0], error_msg)) {
      return false;
    }
    for (std::uint32_t i = 0U; i + 1U < count; ++i) {
      if (!MtpAdvance(session, tokens[i + 1U], h_out_, error_msg) ||
          !Step(session, tokens[i + 1U], error_msg)) {
        return false;
      }
    }
    return true;
  }

  const std::uint32_t hid = c_.hidden_size;
  GemvType type;
  (void)GemvTypeOf(model_.token_embd().type, &type);
  std::uint32_t done = 0U;
  while (done < count) {
    const std::uint32_t rows =
        count - done > kPrefillChunkRows ? kPrefillChunkRows : count - done;
    for (std::uint32_t r = 0; r < rows; ++r) {
      EmbedRow(model_.token_embd().data, type,
               static_cast<std::uint32_t>(tokens[done + r]),
               static_cast<std::uint32_t>(hid),
               pf_x_ + static_cast<std::size_t>(r) * hid, nullptr);
    }
    PrefillChunk(session, rows, session.position_);
    session.position_ += rows;
    if (tier_failure_) {
      return fail("chunked prefill tier rejected a projection");
    }
    // Pair the draft block with every trunk position except the final
    // prompt token: row r of pf_x_ is the hidden of tokens[done + r],
    // which the per-token loop exposes as h_out_ to MtpStep(tokens[i + 1]).
    // The batch tier runs the one draft layer over these rows with the
    // same WMMA kernels the 46 trunk layers use, instead of one sequential
    // GEMV graph per row.
    const std::uint32_t draft_rows =
        done + rows + 1U <= count
            ? rows
            : (done < count - 1U ? count - 1U - done : 0U);
    if (draft_rows > 0U) {
      if (mtp_draft_tier_ready_) {
        PrefillMtpDraft(session, tokens + done + 1U, draft_rows);
        if (tier_failure_) {
          return fail("chunked prefill tier rejected a draft projection");
        }
      } else {
        // A draft format the WMMA tier cannot decode: the per-token draft
        // step is the reference-shaped path and always correct.
        for (std::uint32_t r = 0U; r < draft_rows; ++r) {
          RmsNorm(pf_x_ + static_cast<std::size_t>(r) * hid,
                  model_.output_norm().f32(), c_.rms_eps, h_out_, hid, nullptr);
          if (!MtpAdvance(session, tokens[done + r + 1U], h_out_, error_msg)) {
            return false;
          }
        }
      }
    }
    done += rows;
  }

  // Final token hidden and logits, identical to Prefill's tail.
  const std::uint32_t last = (count - 1U) % kPrefillChunkRows;
  RmsNorm(pf_x_ + static_cast<std::size_t>(last) * hid,
          model_.output_norm().f32(), c_.rms_eps, h_out_, hid, nullptr);
  (void)GemvTypeOf(model_.output().type, &type);
  Gemv(model_.output().data, type, model_.output().rows, model_.output().cols,
       model_.output().row_bytes, h_out_, logits_, nullptr);
  return true;
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
  RmsNorm(hidden, l.nextn_hnorm.f32(), c_.rms_eps, mtp_h_, hid, nullptr);
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

  if (with_logits) {
    // Keep the unnormalized output as the next draft's hidden input.
    (void)hipMemcpyAsync(mtp_chain_, mtp_x_, hid * sizeof(*mtp_chain_),
                         hipMemcpyDeviceToDevice, nullptr);
    RmsNorm(mtp_x_, l.nextn_shared_head_norm.f32(), c_.rms_eps, normed_, hid,
            nullptr);
    (void)GemvTypeOf(model_.output().type, &type);
    Gemv(model_.output().data, type, model_.output().rows, model_.output().cols,
         model_.output().row_bytes, normed_, mtp_logits_, nullptr);
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
  const auto fail = [&](const char* msg) {
    if (error_msg != nullptr) {
      *error_msg = msg;
    }
    return false;
  };
  if (!model_.has_mtp()) {
    return fail("model has no MTP block");
  }
  if (drafts == nullptr || k == 0U || k + 1U > kMaxVerifyRows) {
    return fail("invalid speculative draft count");
  }
  const std::uint32_t rows = k + 1U;
  std::int32_t seq[kMaxVerifyRows];
  seq[0] = t0;
  for (std::uint32_t i = 0; i < k; ++i) {
    seq[i + 1U] = drafts[i];
  }
  for (std::uint32_t r = 0; r < rows; ++r) {
    const std::int32_t token = seq[r];
    if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
      return fail("token id out of range");
    }
  }
  if (session.position_ + rows > session.max_context()) {
    return fail("context exhausted");
  }

  const std::size_t hid = c_.hidden_size;
  session.verify_base_pos_ = session.position_;
  GemvType type;
  (void)GemvTypeOf(model_.token_embd().type, &type);
  for (std::uint32_t r = 0; r < rows; ++r) {
    EmbedRow(model_.token_embd().data, type, static_cast<std::uint32_t>(seq[r]),
             static_cast<std::uint32_t>(hid),
             pf_x_ + static_cast<std::size_t>(r) * hid, nullptr);
  }

  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const DeviceLayer& l = model_.layers()[il];
    for (std::uint32_t r = 0; r < rows; ++r) {
      RmsNorm(pf_x_ + static_cast<std::size_t>(r) * hid, l.attn_norm.f32(),
              c_.rms_eps, pf_normed_ + static_cast<std::size_t>(r) * hid, hid,
              nullptr);
    }
    if (c_.IsMlaLayer(il)) {
      MlaBatch(l, session.mla_kv_[il], session.mla_kpe_[il], session.position_,
               rows);
    } else {
      KdaBatch(session, il, l, rows);
    }
    Add(pf_x_, pf_attn_, pf_x_, rows * hid, nullptr);
    for (std::uint32_t r = 0; r < rows; ++r) {
      RmsNorm(pf_x_ + static_cast<std::size_t>(r) * hid, l.ffn_norm.f32(),
              c_.rms_eps, pf_normed_ + static_cast<std::size_t>(r) * hid, hid,
              nullptr);
    }
    FfnBatch(l, il, rows);
    Add(pf_x_, pf_ffn_, pf_x_, rows * hid, nullptr);
  }

  for (std::uint32_t r = 0; r < rows; ++r) {
    RmsNorm(pf_x_ + static_cast<std::size_t>(r) * hid,
            model_.output_norm().f32(), c_.rms_eps,
            pf_h_ + static_cast<std::size_t>(r) * hid, hid, nullptr);
  }
  (void)hipMemcpyAsync(h_out_, pf_h_ + static_cast<std::size_t>(k) * hid,
                       hid * sizeof(*h_out_), hipMemcpyDeviceToDevice, nullptr);
  (void)GemvTypeOf(model_.output().type, &type);
  GemvRows(model_.output().data, type, rows, model_.output().rows,
           model_.output().cols, model_.output().row_bytes, pf_h_,
           static_cast<std::uint32_t>(hid), pf_logits_, c_.vocab_size, nullptr);
  session.position_ += rows;
  return true;
}

void Executor::RollbackVerify(Session& session, std::uint32_t keep) {
  const std::uint32_t snap = keep - 1U;
  const std::size_t conv_slot = 3U * (c_.ssm_conv_kernel - 1U) *
                                static_cast<std::size_t>(c_.SsmInnerSize());
  const std::size_t state_elems = static_cast<std::size_t>(c_.num_heads) *
                                  c_.ssm_head_dim * c_.ssm_head_dim;
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    if (session.kda_state_snap_[il] == nullptr) {
      continue;
    }
    (void)hipMemcpyAsync(session.kda_state_[il],
                         session.kda_state_snap_[il] + snap * state_elems,
                         state_elems * sizeof(float), hipMemcpyDeviceToDevice,
                         nullptr);
    (void)hipMemcpyAsync(
        session.kda_conv_[il], session.kda_conv_snap_[il] + snap * conv_slot,
        conv_slot * sizeof(float), hipMemcpyDeviceToDevice, nullptr);
  }
  session.position_ = session.verify_base_pos_ + keep;
  session.mtp_position_ = session.verify_base_pos_ + keep;
  (void)hipMemcpyAsync(
      h_out_, pf_h_ + static_cast<std::size_t>(snap) * c_.hidden_size,
      c_.hidden_size * sizeof(*h_out_), hipMemcpyDeviceToDevice, nullptr);
}

}  // namespace gufo::models::ling3_flash::rocm