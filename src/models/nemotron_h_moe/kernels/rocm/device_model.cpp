#include "src/models/nemotron_h_moe/kernels/rocm/device_model.hpp"

#include <cmath>
#include <cstring>

#include "src/models/nemotron_h_moe/kernels/rocm/mmq/nmb_mmq.h"
#include "src/models/nemotron_h_moe/kernels/rocm/nb_attention.h"
#include "src/models/nemotron_h_moe/kernels/rocm/nb_mamba2.h"
#include "src/models/nemotron_h_moe/kernels/rocm/nb_ops.h"

namespace gufo::models::nemotron_h_moe {
namespace {

bool Fail(std::string* error_msg, const std::string& msg) {
  if (error_msg != nullptr) {
    *error_msg = msg;
  }
  return false;
}

}  // namespace

DeviceModel::DeviceModel(const ModelWeights& weights, std::uint32_t max_context)
    : w_(weights), c_(weights.config), max_context_(max_context) {}

DeviceModel::~DeviceModel() {
  Free();
}

bool DeviceModel::Upload(void** dst, const TensorRef& src,
                         std::string* error_msg) {
  const std::size_t bytes = src.SizeBytes();
  if (src.empty() || bytes == 0) {
    return Fail(error_msg, "empty tensor uploaded");
  }
  if (hipMalloc(dst, bytes) != hipSuccess) {
    return Fail(error_msg, "hipMalloc failed for " + std::string(src.name));
  }
  if (hipMemcpy(*dst, src.data, bytes, hipMemcpyHostToDevice) != hipSuccess) {
    return Fail(error_msg, "hipMemcpy failed for " + std::string(src.name));
  }
  weight_bytes_ += bytes;
  return true;
}

bool DeviceModel::UploadScalar(float* dst, const TensorRef& src,
                               float* host_value, std::string* error_msg) {
  if (src.empty()) {
    return Fail(error_msg, "missing scalar tensor");
  }
  std::memcpy(host_value, src.data, sizeof(float));
  if (hipMemcpy(dst, host_value, sizeof(float), hipMemcpyHostToDevice) !=
      hipSuccess) {
    return Fail(error_msg, "scalar upload failed");
  }
  return true;
}

bool DeviceModel::Init(std::string* error_msg) {
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess || nmb_mmq_init(device) != 0) {
    return Fail(error_msg, "mmq init failed");
  }
  layers_.assign(c_.num_layers, LayerDev{});
  if (hipMalloc(reinterpret_cast<void**>(&token_embd_),
                w_.token_embd.SizeBytes()) != hipSuccess) {
    return Fail(error_msg, "token_embd alloc");
  }
  if (hipMemcpy(token_embd_, w_.token_embd.data, w_.token_embd.SizeBytes(),
                hipMemcpyHostToDevice) != hipSuccess) {
    return Fail(error_msg, "token_embd upload");
  }
  const TensorRef out_t = w_.output.empty() ? w_.token_embd : w_.output;
  if (hipMalloc(reinterpret_cast<void**>(&output_), out_t.SizeBytes()) !=
      hipSuccess) {
    return Fail(error_msg, "output alloc");
  }
  weight_bytes_ += out_t.SizeBytes();
  if (hipMemcpy(output_, out_t.data, out_t.SizeBytes(),
                hipMemcpyHostToDevice) != hipSuccess) {
    return Fail(error_msg, "output upload");
  }
  if (hipMalloc(reinterpret_cast<void**>(&output_norm_),
                w_.output_norm.SizeBytes()) != hipSuccess) {
    return Fail(error_msg, "output_norm alloc");
  }
  if (hipMemcpy(output_norm_, w_.output_norm.data, w_.output_norm.SizeBytes(),
                hipMemcpyHostToDevice) != hipSuccess) {
    return Fail(error_msg, "output_norm upload");
  }
  const std::uint32_t hid = c_.hidden_size;
  const std::uint32_t xbc = c_.SsmConvChannels();
  const std::uint32_t inner = c_.ssm_inner_size;
  const std::uint32_t qdim = c_.AttentionQDim();
  for (std::uint32_t l = 0; l < c_.num_layers; ++l) {
    const LayerWeights& lw = w_.layers[l];
    LayerDev& ld = layers_[l];
    ld.kind = c_.Kind(l);
    if (!Upload(&ld.norm, lw.attn_norm, error_msg)) {
      return false;
    }
    switch (ld.kind) {
      case Config::LayerKind::kAttention: {
        ld.kvh = c_.head_count_kv[l];
        ld.kvdim = c_.AttentionKvDim(l);
        if (!Upload(&ld.q, lw.attn_q, error_msg) ||
            !Upload(&ld.k, lw.attn_k, error_msg) ||
            !Upload(&ld.v, lw.attn_v, error_msg) ||
            !Upload(&ld.attn_out, lw.attn_out, error_msg)) {
          return false;
        }
        const std::size_t kv_bytes = static_cast<std::size_t>(max_context_) *
                                     2 * ld.kvdim * sizeof(float);
        if (hipMalloc(reinterpret_cast<void**>(&ld.kv), kv_bytes) !=
            hipSuccess) {
          return Fail(error_msg, "kv alloc");
        }
        if (hipMemsetAsync(ld.kv, 0, kv_bytes, nullptr) != hipSuccess) {
          return Fail(error_msg, "kv memset");
        }
        kv_bytes_ += kv_bytes;
        break;
      }
      case Config::LayerKind::kMamba: {
        if (!Upload(&ld.ssm_in, lw.ssm_in, error_msg) ||
            !Upload(&ld.ssm_out, lw.ssm_out, error_msg) ||
            !Upload(&ld.conv_w, lw.ssm_conv, error_msg) ||
            !Upload(&ld.ssm_a, lw.ssm_a, error_msg) ||
            !Upload(&ld.ssm_d, lw.ssm_d, error_msg) ||
            !Upload(&ld.ssm_dt_b, lw.ssm_dt_b, error_msg) ||
            !Upload(&ld.ssm_norm, lw.ssm_norm, error_msg)) {
          return false;
        }
        if (!lw.ssm_conv_b.empty() &&
            !Upload(&ld.conv_b, lw.ssm_conv_b, error_msg)) {
          return false;
        }
        if (!lw.ssm_in_s.empty()) {
          ld.ssm_in_scale = static_cast<const float*>(lw.ssm_in_s.data)[0];
        }
        if (!lw.ssm_out_s.empty()) {
          ld.ssm_out_scale = static_cast<const float*>(lw.ssm_out_s.data)[0];
        }
        const std::size_t conv_bytes =
            static_cast<std::size_t>(c_.ssm_conv_kernel - 1) * xbc *
            sizeof(float);
        const std::size_t ssm_bytes =
            static_cast<std::size_t>(c_.SsmHeadCount()) * c_.SsmHeadDim() *
            c_.ssm_state_size * sizeof(float);
        if (hipMalloc(reinterpret_cast<void**>(&ld.conv_state), conv_bytes) !=
                hipSuccess ||
            hipMalloc(reinterpret_cast<void**>(&ld.ssm_state), ssm_bytes) !=
                hipSuccess) {
          return Fail(error_msg, "mamba state alloc");
        }
        if (hipMemsetAsync(ld.conv_state, 0, conv_bytes, nullptr) !=
                hipSuccess ||
            hipMemsetAsync(ld.ssm_state, 0, ssm_bytes, nullptr) != hipSuccess) {
          return Fail(error_msg, "mamba state memset");
        }
        state_bytes_ += conv_bytes + ssm_bytes;
        break;
      }
      case Config::LayerKind::kMoe: {
        ld.n_experts = lw.n_experts;
        ld.expert_ff = c_.ExpertFf(l);
        ld.top_k = c_.ExpertsUsed(l);
        max_experts_ = static_cast<int>(
            std::max(static_cast<std::uint64_t>(max_experts_), lw.n_experts));
        max_expert_ff_ = static_cast<int>(std::max(
            static_cast<std::uint32_t>(max_expert_ff_), c_.ExpertFf(l)));
        if (!Upload(&ld.router, lw.router, error_msg) ||
            !Upload(&ld.exp_b, lw.exp_probs_b, error_msg) ||
            !Upload(&ld.up_exps, lw.ffn_up_exps, error_msg) ||
            !Upload(&ld.down_exps, lw.ffn_down_exps, error_msg) ||
            !Upload(&ld.up_shexp, lw.ffn_up_shexp, error_msg) ||
            !Upload(&ld.down_shexp, lw.ffn_down_shexp, error_msg) ||
            !Upload(&ld.latent_up, lw.ffn_latent_up, error_msg) ||
            !Upload(&ld.latent_down, lw.ffn_latent_down, error_msg)) {
          return false;
        }
        break;
      }
    }
  }
  const std::uint32_t latent = c_.moe_latent_size;
  auto alloc = [&](float** p, std::size_t n, const char* tag) {
    if (hipMalloc(reinterpret_cast<void**>(p), n * sizeof(float)) !=
        hipSuccess) {
      Fail(error_msg, std::string("scratch alloc ") + tag);
      return false;
    }
    scratch_bytes_ += n * sizeof(float);
    return true;
  };
  const std::uint32_t t = kPrefillChunk;
  const std::size_t top_k = 16;
  auto nrow = [t](std::size_t w) { return t * w; };
  bool ok = alloc(&x_, nrow(hid), "x") &&
            alloc(&normed_, nrow(hid), "normed") &&
            alloc(&out_, nrow(hid), "out") &&
            alloc(&logits_, c_.vocab_size, "logits") &&
            alloc(&zxbc_, nrow(c_.SsmInProj()), "zxbc") &&
            alloc(&y_, nrow(inner), "y") && alloc(&q_, nrow(qdim), "q") &&
            alloc(&o_, nrow(qdim), "o");
  if (ok) {
    ok = alloc(&k_, nrow(hid), "k") && alloc(&v_, nrow(hid), "v");
  }
  if (ok) {
    ok = alloc(&router_logits_, nrow(static_cast<std::size_t>(max_experts_)),
               "router") &&
         alloc(&moe_w_, nrow(top_k), "moe_w") &&
         alloc(&lat_, nrow(latent), "lat") &&
         alloc(&up_out_, nrow(top_k * static_cast<std::size_t>(max_expert_ff_)),
               "up_out") &&
         alloc(&act_, nrow(top_k * static_cast<std::size_t>(max_expert_ff_)),
               "act") &&
         alloc(&down_out_, nrow(top_k * static_cast<std::size_t>(latent)),
               "dn") &&
         alloc(&acc_, nrow(latent), "acc") && alloc(&moe_, nrow(hid), "moe") &&
         alloc(&sh_, nrow(c_.shared_expert_ff), "sh") &&
         hipMalloc(reinterpret_cast<void**>(&ids_),
                   nrow(top_k) * sizeof(std::int32_t)) == hipSuccess &&
         hipMalloc(reinterpret_cast<void**>(&tok_dev_),
                   nrow(1) * sizeof(std::int32_t)) == hipSuccess;
    scratch_bytes_ += (nrow(top_k) + nrow(1)) * sizeof(std::int32_t);
  }
  if (ok) {
    // list: one slot rank per (token, slot); desc: tile table with at most
    // ceil(slots/NBL_ROW_NT) + 512 tiles of 3 ints; n_desc: device tile count.
    const std::size_t slots = nrow(top_k);
    const std::size_t tiles = slots / NBL_ROW_NT + 512U;
    ok = hipMalloc(reinterpret_cast<void**>(&exp_list_),
                   slots * sizeof(std::int32_t)) == hipSuccess &&
         hipMalloc(reinterpret_cast<void**>(&exp_desc_),
                   3 * tiles * sizeof(std::int32_t)) == hipSuccess &&
         hipMalloc(reinterpret_cast<void**>(&exp_ndesc_),
                   sizeof(std::int32_t)) == hipSuccess;
    scratch_bytes_ += (slots + 3 * tiles + 1) * sizeof(std::int32_t);
  }
  if (!ok) {
    if (error_msg != nullptr && error_msg->empty()) {
      *error_msg = "scratch alloc failed";
    }
    return false;
  }
  return true;
}

void DeviceModel::Free() {
  auto freelist = [&](void* p) {
    if (p != nullptr) {
      (void)hipFree(p);
    }
  };
  for (LayerDev& ld : layers_) {
    freelist(ld.norm);
    freelist(ld.q);
    freelist(ld.k);
    freelist(ld.v);
    freelist(ld.attn_out);
    freelist(ld.kv);
    freelist(ld.ssm_in);
    freelist(ld.ssm_out);
    freelist(ld.conv_w);
    freelist(ld.conv_b);
    freelist(ld.ssm_a);
    freelist(ld.ssm_d);
    freelist(ld.ssm_dt_b);
    freelist(ld.ssm_norm);
    freelist(ld.conv_state);
    freelist(ld.ssm_state);
    freelist(ld.router);
    freelist(ld.exp_b);
    freelist(ld.up_exps);
    freelist(ld.down_exps);
    freelist(ld.up_shexp);
    freelist(ld.down_shexp);
    freelist(ld.latent_up);
    freelist(ld.latent_down);
  }
  layers_.clear();
  freelist(token_embd_);
  freelist(output_);
  freelist(output_norm_);
  freelist(x_);
  freelist(normed_);
  freelist(out_);
  freelist(logits_);
  freelist(tok_dev_);
  tok_dev_ = nullptr;
  freelist(zxbc_);
  freelist(y_);
  freelist(q_);
  freelist(k_);
  freelist(v_);
  freelist(o_);
  freelist(router_logits_);
  freelist(ids_);
  freelist(moe_w_);
  freelist(lat_);
  freelist(up_out_);
  freelist(act_);
  freelist(down_out_);
  freelist(acc_);
  freelist(moe_);
  freelist(sh_);
  freelist(exp_list_);
  freelist(exp_desc_);
  freelist(exp_ndesc_);
  exp_list_ = nullptr;
  exp_desc_ = nullptr;
  exp_ndesc_ = nullptr;
  token_embd_ = nullptr;
  output_ = nullptr;
  weight_bytes_ = 0;
  kv_bytes_ = 0;
  state_bytes_ = 0;
  scratch_bytes_ = 0;
}

void DeviceModel::Reset() {
  const std::size_t conv_bytes =
      static_cast<std::size_t>(c_.ssm_conv_kernel - 1) * c_.SsmConvChannels() *
      sizeof(float);
  const std::size_t ssm_bytes = static_cast<std::size_t>(c_.SsmHeadCount()) *
                                c_.SsmHeadDim() * c_.ssm_state_size *
                                sizeof(float);
  for (LayerDev& ld : layers_) {
    if (ld.kind == Config::LayerKind::kMamba) {
      (void)hipMemsetAsync(ld.conv_state, 0, conv_bytes, nullptr);
      (void)hipMemsetAsync(ld.ssm_state, 0, ssm_bytes, nullptr);
    }
  }
  position_ = 0;
}

void DeviceModel::PassMamba(const LayerDev& l, int t, hipStream_t stream) {
  const std::uint32_t inner = c_.ssm_inner_size;
  const std::uint32_t proj = c_.SsmInProj();
  const bool wide = t >= static_cast<int>(kRowsBatchMin);
  if (wide) {
    (void)nmb_mmq_nvfp4_dense(l.ssm_in, normed_, zxbc_, static_cast<int>(proj),
                              t, static_cast<int>(c_.hidden_size), stream);
  } else {
    nbl_gemv_nvfp4_f32(normed_, l.ssm_in, zxbc_, static_cast<int>(proj),
                       static_cast<int>(c_.hidden_size), t, stream);
  }
  nbl_scale(zxbc_, l.ssm_in_scale, static_cast<int64_t>(t) * proj, stream);
  nbl_mamba_scan(
      zxbc_, static_cast<const float*>(l.conv_w),
      static_cast<const float*>(l.conv_b), static_cast<const float*>(l.ssm_a),
      static_cast<const float*>(l.ssm_d), static_cast<const float*>(l.ssm_dt_b),
      l.conv_state, l.ssm_state, y_, static_cast<int>(inner),
      static_cast<int>(c_.SsmHeadCount()), static_cast<int>(c_.SsmHeadDim()),
      static_cast<int>(c_.ssm_state_size), static_cast<int>(c_.ssm_group_count),
      static_cast<int>(c_.ssm_conv_kernel),
      static_cast<int>(c_.SsmConvChannels()), static_cast<int>(proj), t,
      stream);
  nbl_rms_norm_grouped(y_, static_cast<const float*>(l.ssm_norm), y_,
                       t * static_cast<int>(c_.ssm_group_count),
                       static_cast<int>(inner / c_.ssm_group_count),
                       static_cast<int>(c_.ssm_group_count), c_.rms_eps,
                       stream);
  if (wide) {
    (void)nmb_mmq_nvfp4_dense(l.ssm_out, y_, out_,
                              static_cast<int>(c_.hidden_size), t,
                              static_cast<int>(inner), stream);
  } else {
    nbl_gemv_f32(kNblGemvNvfp4, l.ssm_out, y_, out_,
                 static_cast<int>(c_.hidden_size), static_cast<int>(inner), t,
                 stream);
  }
  nbl_scale(out_, l.ssm_out_scale, static_cast<int64_t>(t) * c_.hidden_size,
            stream);
}

void DeviceModel::PassAttn(const LayerDev& l, int t, hipStream_t stream) {
  const int hid = static_cast<int>(c_.hidden_size);
  const int kvh = static_cast<int>(l.kvh);
  const int kvdim = static_cast<int>(l.kvdim);
  const int hd = static_cast<int>(c_.head_dim);
  const int hdv = static_cast<int>(c_.head_dim_v);
  const bool wide = t >= static_cast<int>(kRowsBatchMin);
  if (wide) {
    (void)nmb_mmq_q8_0_dense(
        l.q, normed_, q_, static_cast<int>(c_.AttentionQDim()), t, hid, stream);
    (void)nmb_mmq_q8_0_dense(l.k, normed_, k_, kvdim, t, hid, stream);
    (void)nmb_mmq_q8_0_dense(l.v, normed_, v_, kvdim, t, hid, stream);
  } else {
    nbl_gemv_f32(kNblGemvQ8_0, l.q, normed_, q_,
                 static_cast<int>(c_.AttentionQDim()), hid, t, stream);
    nbl_gemv_f32(kNblGemvQ8_0, l.k, normed_, k_, kvdim, hid, t, stream);
    nbl_gemv_f32(kNblGemvQ8_0, l.v, normed_, v_, kvdim, hid, t, stream);
  }
  nbl_attn_append(k_, v_, l.kv, static_cast<int>(position_), t, kvdim, stream);
  nbl_attn(q_, l.kv, static_cast<int>(position_), t, o_,
           static_cast<int>(c_.num_heads), hd, hdv, kvh, kvdim,
           1.0F / std::sqrt(static_cast<float>(hd)), stream);
  if (wide) {
    (void)nmb_mmq_q8_0_dense(l.attn_out, o_, out_, hid, t,
                             static_cast<int>(c_.AttentionQDim()), stream);
  } else {
    nbl_gemv_f32(kNblGemvQ8_0, l.attn_out, o_, out_, hid,
                 static_cast<int>(c_.AttentionQDim()), t, stream);
  }
}

void DeviceModel::PassMoe(const LayerDev& l, int t, hipStream_t stream) {
  const int hid = static_cast<int>(c_.hidden_size);
  const int latent = static_cast<int>(c_.moe_latent_size);
  const int top_k = static_cast<int>(l.top_k);
  const int n_exp = static_cast<int>(l.n_experts);
  const int expert_ff = static_cast<int>(l.expert_ff);
  const bool wide = t >= static_cast<int>(kRowsBatchMin);
  const int grid_tiles = (t * top_k + NBL_ROW_NT - 1) / NBL_ROW_NT + n_exp;
  nbl_gemm_f32(normed_, static_cast<const float*>(l.router), router_logits_, t,
               n_exp, hid, stream);
  nbl_router(router_logits_, static_cast<const float*>(l.exp_b), n_exp, top_k,
             c_.expert_weights_norm ? 1 : 0, c_.expert_weights_scale, t, ids_,
             moe_w_, stream);
  if (wide) {
    nbl_expert_index(ids_, n_exp, t * top_k, NBL_ROW_NT, exp_list_, exp_desc_,
                     exp_ndesc_, stream);
  }
  if (wide) {
    (void)nmb_mmq_q8_0_dense(l.latent_down, normed_, lat_, latent, t, hid,
                             stream);
  } else {
    nbl_gemv_f32(kNblGemvQ8_0, l.latent_down, normed_, lat_, latent, hid, t,
                 stream);
  }
  if (wide) {
    (void)nmb_mmq_iq2_xxs_moe_raw(l.up_exps, lat_, ids_, up_out_, expert_ff,
                                  latent, t, n_exp, top_k, stream);
  } else {
    nbl_gemv_moe_f32(kNblGemvIq2Xxs, l.up_exps, lat_, ids_, up_out_, expert_ff,
                     latent, top_k, 0, t, stream);
  }
  nbl_relu_sqr(up_out_, act_, static_cast<int64_t>(t) * top_k * expert_ff,
               stream);
  if (wide) {
    nbl_gemv_rows_moe(kNblGemvIq2Xxs, l.down_exps, act_, exp_list_, exp_desc_,
                      exp_ndesc_, grid_tiles, 0, down_out_, latent, expert_ff,
                      stream);
  } else {
    nbl_gemv_moe_f32(kNblGemvIq2Xxs, l.down_exps, act_, ids_, down_out_, latent,
                     expert_ff, top_k, expert_ff, t, stream);
  }
  nbl_moe_combine(acc_, down_out_, moe_w_, latent, top_k, t, stream);
  if (wide) {
    (void)nmb_mmq_q8_0_dense(l.latent_up, acc_, moe_, hid, t, latent, stream);
  } else {
    nbl_gemv_f32(kNblGemvQ8_0, l.latent_up, acc_, moe_, hid, latent, t, stream);
  }
  if (wide) {
    (void)nmb_mmq_iq2_xxs_dense(l.up_shexp, normed_, sh_,
                                static_cast<int>(c_.shared_expert_ff), t, hid,
                                stream);
  } else {
    nbl_gemv_f32(kNblGemvIq2Xxs, l.up_shexp, normed_, sh_,
                 static_cast<int>(c_.shared_expert_ff), hid, t, stream);
  }
  nbl_relu_sqr(sh_, sh_, static_cast<int64_t>(t) * c_.shared_expert_ff, stream);
  if (wide) {
    (void)nmb_mmq_iq2_xxs_dense(l.down_shexp, sh_, out_, hid, t,
                                static_cast<int>(c_.shared_expert_ff), stream);
  } else {
    nbl_gemv_f32(kNblGemvIq2Xxs, l.down_shexp, sh_, out_, hid,
                 static_cast<int>(c_.shared_expert_ff), t, stream);
  }
  nbl_add(out_, moe_, static_cast<int64_t>(t) * hid, stream);
}

bool DeviceModel::ForwardLayers(int t, bool with_dump, std::string* error_msg) {
  const std::uint32_t hid = c_.hidden_size;
  for (std::uint32_t l = 0; l < c_.num_layers; ++l) {
    const LayerDev& ld = layers_[l];
    if (with_dump) {
      const std::size_t base = static_cast<std::size_t>(l) * hid;
      layer_dump_->resize(base + hid);
      if (hipMemcpy(layer_dump_->data() + base, x_, hid * sizeof(float),
                    hipMemcpyDeviceToHost) != hipSuccess) {
        return Fail(error_msg, "layer dump failed");
      }
    }
    nbl_rms_norm(x_, static_cast<const float*>(ld.norm), normed_, t,
                 static_cast<int>(hid), c_.rms_eps, nullptr);
    switch (ld.kind) {
      case Config::LayerKind::kMamba:
        PassMamba(ld, t, nullptr);
        break;
      case Config::LayerKind::kAttention:
        PassAttn(ld, t, nullptr);
        break;
      case Config::LayerKind::kMoe:
        PassMoe(ld, t, nullptr);
        break;
    }
    nbl_add(x_, out_, static_cast<int64_t>(t) * hid, nullptr);
  }
  nbl_rms_norm(x_, output_norm_, normed_, t, static_cast<int>(hid), c_.rms_eps,
               nullptr);
  position_ += static_cast<std::uint32_t>(t);
  if (hipGetLastError() != hipSuccess) {
    return Fail(error_msg, "kernel launch failed during forward");
  }
  return true;
}

bool DeviceModel::Step(std::int32_t token, std::span<float> logits,
                       std::string* error_msg) {
  auto check = [&](hipError_t err, const char* tag) {
    if (err != hipSuccess) {
      Fail(error_msg, std::string(tag) + " failed");
      return false;
    }
    return true;
  };
  if (!check(hipGetLastError(), "pre-step")) {
    return false;
  }
  if (position_ >= max_context_) {
    return Fail(error_msg, "prompt exceeds KV capacity");
  }
  const std::uint32_t hid = c_.hidden_size;
  nbl_dequant_row_q8_0(token_embd_, token, static_cast<int>(hid), x_, nullptr);
  if (!ForwardLayers(1, layer_dump_ != nullptr, error_msg)) {
    return false;
  }
  if (!logits.empty()) {
    nbl_gemv_f32(kNblGemvQ8_0, output_, normed_, logits_,
                 static_cast<int>(c_.vocab_size), static_cast<int>(hid), 1,
                 nullptr);
    if (hipMemcpy(logits.data(), logits_, c_.vocab_size * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      return Fail(error_msg, "logits download failed");
    }
  }
  if (hipGetLastError() != hipSuccess) {
    return Fail(error_msg, "kernel launch failed during step");
  }
  return true;
}

bool DeviceModel::Process(std::span<const std::int32_t> tokens,
                          std::span<float> logits, std::string* error_msg) {
  if (tokens.empty()) {
    return true;
  }
  if (hipGetLastError() != hipSuccess) {
    return Fail(error_msg, "pre-process");
  }
  if (static_cast<std::uint64_t>(position_) + tokens.size() > max_context_) {
    return Fail(error_msg, "prompt exceeds KV capacity");
  }
  const std::uint32_t hid = c_.hidden_size;
  std::size_t done = 0;
  while (done < tokens.size()) {
    const int n = static_cast<int>(
        std::min<std::size_t>(kPrefillChunk, tokens.size() - done));
    if (hipMemcpyAsync(tok_dev_, tokens.data() + done, n * sizeof(std::int32_t),
                       hipMemcpyHostToDevice, nullptr) != hipSuccess) {
      return Fail(error_msg, "token upload failed");
    }
    nbl_dequant_rows_q8_0(token_embd_, tok_dev_, n, static_cast<int>(hid), x_,
                          nullptr);
    if (!ForwardLayers(n, false, error_msg)) {
      return false;
    }
    done += static_cast<std::size_t>(n);
  }
  if (!logits.empty()) {
    const float* last =
        normed_ + static_cast<std::size_t>(tokens.size() - 1) * hid;
    nbl_gemv_f32(kNblGemvQ8_0, output_, last, logits_,
                 static_cast<int>(c_.vocab_size), static_cast<int>(hid), 1,
                 nullptr);
    if (hipMemcpy(logits.data(), logits_, c_.vocab_size * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      return Fail(error_msg, "logits download failed");
    }
  }
  return true;
}

}  // namespace gufo::models::nemotron_h_moe
