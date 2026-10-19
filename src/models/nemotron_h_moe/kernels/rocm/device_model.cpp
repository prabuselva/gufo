#include "src/models/nemotron_h_moe/kernels/rocm/device_model.hpp"

#include <cmath>
#include <cstring>

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
  const std::uint32_t top_k = 16;
  auto alloc = [&](float** p, std::size_t n, const char* tag) {
    if (hipMalloc(reinterpret_cast<void**>(p), n * sizeof(float)) !=
        hipSuccess) {
      Fail(error_msg, std::string("scratch alloc ") + tag);
      return false;
    }
    return true;
  };
  const std::uint32_t latent = c_.moe_latent_size;
  bool ok = alloc(&x_, hid, "x") && alloc(&normed_, hid, "normed") &&
            alloc(&out_, hid, "out") &&
            alloc(&logits_, c_.vocab_size, "logits") &&
            alloc(&zxbc_, c_.SsmInProj(), "zxbc") && alloc(&y_, inner, "y") &&
            alloc(&q_, qdim, "q") && alloc(&o_, qdim, "o");
  if (ok) {
    ok = alloc(&k_, hid, "k") && alloc(&v_, hid, "v");
  }
  if (ok) {
    ok =
        alloc(&router_logits_, static_cast<std::size_t>(max_experts_),
              "router") &&
        alloc(&moe_w_, top_k, "moe_w") && alloc(&lat_, latent, "lat") &&
        alloc(&up_out_, static_cast<std::size_t>(top_k) * max_expert_ff_,
              "up_out") &&
        alloc(&act_, static_cast<std::size_t>(top_k) * max_expert_ff_, "act") &&
        alloc(&down_out_, static_cast<std::size_t>(top_k) * latent, "dn") &&
        alloc(&acc_, latent, "acc") && alloc(&moe_, hid, "moe") &&
        alloc(&sh_, c_.shared_expert_ff, "sh") &&
        hipMalloc(reinterpret_cast<void**>(&ids_),
                  top_k * sizeof(std::int32_t)) == hipSuccess;
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
  token_embd_ = nullptr;
  output_ = nullptr;
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

#include <cstdio>

void DeviceModel::StepMamba(const LayerDev& l, hipStream_t stream) {
  const std::uint32_t inner = c_.ssm_inner_size;
  const std::uint32_t proj = c_.SsmInProj();
  nbl_gemv_nvfp4_f32(normed_, l.ssm_in, zxbc_, static_cast<int>(proj),
                     static_cast<int>(c_.hidden_size), stream);
  nbl_scale(zxbc_, l.ssm_in_scale, static_cast<int64_t>(proj), stream);
  nbl_mamba_scan(
      zxbc_, static_cast<const float*>(l.conv_w),
      static_cast<const float*>(l.conv_b), static_cast<const float*>(l.ssm_a),
      static_cast<const float*>(l.ssm_d), static_cast<const float*>(l.ssm_dt_b),
      l.conv_state, l.ssm_state, y_, static_cast<int>(inner),
      static_cast<int>(c_.SsmHeadCount()), static_cast<int>(c_.SsmHeadDim()),
      static_cast<int>(c_.ssm_state_size), static_cast<int>(c_.ssm_group_count),
      static_cast<int>(c_.ssm_conv_kernel),
      static_cast<int>(c_.SsmConvChannels()), static_cast<int>(proj), 1,
      stream);
  nbl_rms_norm_grouped(y_, static_cast<const float*>(l.ssm_norm), y_,
                       static_cast<int>(c_.ssm_group_count),
                       static_cast<int>(inner / c_.ssm_group_count),
                       static_cast<int>(c_.ssm_group_count), c_.rms_eps,
                       stream);
  nbl_gemv_f32(kNblGemvNvfp4, l.ssm_out, y_, out_,
               static_cast<int>(c_.hidden_size), static_cast<int>(inner),
               stream);
  nbl_scale(out_, l.ssm_out_scale, static_cast<int64_t>(c_.hidden_size),
            stream);
}

void DeviceModel::StepAttn(const LayerDev& l, hipStream_t stream) {
  const int hid = static_cast<int>(c_.hidden_size);
  const int kvh = static_cast<int>(l.kvh);
  const int kvdim = static_cast<int>(l.kvdim);
  const int hd = static_cast<int>(c_.head_dim);
  const int hdv = static_cast<int>(c_.head_dim_v);
  nbl_gemv_f32(kNblGemvQ8_0, l.q, normed_, q_,
               static_cast<int>(c_.AttentionQDim()), hid, stream);
  nbl_gemv_f32(kNblGemvQ8_0, l.k, normed_, k_, kvdim, hid, stream);
  nbl_gemv_f32(kNblGemvQ8_0, l.v, normed_, v_, kvdim, hid, stream);
  nbl_attn_append(k_, v_, l.kv, static_cast<int>(position_), 1, kvdim, stream);
  nbl_attn(q_, l.kv, static_cast<int>(position_), 1, o_,
           static_cast<int>(c_.num_heads), hd, hdv, kvh, kvdim,
           1.0F / std::sqrt(static_cast<float>(hd)), stream);
  nbl_gemv_f32(kNblGemvQ8_0, l.attn_out, o_, out_, hid,
               static_cast<int>(c_.AttentionQDim()), stream);
}

void DeviceModel::StepMoe(const LayerDev& l, hipStream_t stream) {
  const int hid = static_cast<int>(c_.hidden_size);
  const int latent = static_cast<int>(c_.moe_latent_size);
  const int top_k = static_cast<int>(l.top_k);
  const int n_exp = static_cast<int>(l.n_experts);
  const int expert_ff = static_cast<int>(l.expert_ff);
  nbl_gemm_f32(normed_, static_cast<const float*>(l.router), router_logits_, 1,
               n_exp, hid, stream);
  nbl_router(router_logits_, static_cast<const float*>(l.exp_b), n_exp, top_k,
             c_.expert_weights_norm ? 1 : 0, c_.expert_weights_scale, 1, ids_,
             moe_w_, stream);
  nbl_gemv_f32(kNblGemvQ8_0, l.latent_down, normed_, lat_, latent, hid, stream);
  nbl_gemv_moe_f32(kNblGemvIq2Xxs, l.up_exps, lat_, ids_, up_out_, expert_ff,
                   latent, top_k, 0, stream);
  nbl_relu_sqr(up_out_, act_, static_cast<int64_t>(top_k) * expert_ff, stream);
  nbl_gemv_moe_f32(kNblGemvIq2Xxs, l.down_exps, act_, ids_, down_out_, latent,
                   expert_ff, top_k, expert_ff, stream);
  nbl_moe_combine(acc_, down_out_, moe_w_, latent, top_k, 1, stream);
  nbl_gemv_f32(kNblGemvQ8_0, l.latent_up, acc_, moe_, hid, latent, stream);
  nbl_gemv_f32(kNblGemvIq2Xxs, l.up_shexp, normed_, sh_,
               static_cast<int>(c_.shared_expert_ff), hid, stream);
  nbl_relu_sqr(sh_, sh_, static_cast<int64_t>(c_.shared_expert_ff), stream);
  nbl_gemv_f32(kNblGemvIq2Xxs, l.down_shexp, sh_, out_, hid,
               static_cast<int>(c_.shared_expert_ff), stream);
  nbl_add(out_, moe_, static_cast<int64_t>(hid), stream);
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
  const std::uint32_t hid = c_.hidden_size;
  nbl_dequant_row_q8_0(token_embd_, token, static_cast<int>(hid), x_, nullptr);
  for (std::uint32_t l = 0; l < c_.num_layers; ++l) {
    const LayerDev& ld = layers_[l];
    if (layer_dump_ != nullptr) {
      const std::size_t base = static_cast<std::size_t>(l) * hid;
      layer_dump_->resize(base + hid);
      if (hipMemcpy(layer_dump_->data() + base, x_, hid * sizeof(float),
                    hipMemcpyDeviceToHost) != hipSuccess) {
        return Fail(error_msg, "layer dump failed");
      }
    }
    nbl_rms_norm(x_, static_cast<const float*>(ld.norm), normed_, 1,
                 static_cast<int>(hid), c_.rms_eps, nullptr);
    switch (ld.kind) {
      case Config::LayerKind::kMamba:
        StepMamba(ld, nullptr);
        break;
      case Config::LayerKind::kAttention:
        StepAttn(ld, nullptr);
        break;
      case Config::LayerKind::kMoe:
        StepMoe(ld, nullptr);
        break;
    }
    nbl_add(x_, out_, static_cast<int64_t>(hid), nullptr);
  }
  nbl_rms_norm(x_, output_norm_, normed_, 1, static_cast<int>(hid), c_.rms_eps,
               nullptr);
  if (!logits.empty()) {
    nbl_gemv_f32(kNblGemvQ8_0, output_, normed_, logits_,
                 static_cast<int>(c_.vocab_size), static_cast<int>(hid),
                 nullptr);
    if (hipMemcpy(logits.data(), logits_, c_.vocab_size * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      return Fail(error_msg, "logits download failed");
    }
  }
  ++position_;
  if (hipGetLastError() != hipSuccess) {
    return Fail(error_msg, "kernel launch failed during step");
  }
  return true;
}

}  // namespace gufo::models::nemotron_h_moe
