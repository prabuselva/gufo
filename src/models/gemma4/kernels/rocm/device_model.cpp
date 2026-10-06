#include "src/models/gemma4/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "src/core/hip/weight_upload.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

struct Uploader {
  hip::WeightUpload& stager;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size;
    if (!stager.Copy(t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    d.row_bytes = t.RowBytes();
    return d;
  }

  float* CopyF32(const float* host, std::size_t count,
                 const std::string& name) {
    float* ptr = nullptr;
    if (hipMalloc(&ptr, count * sizeof(float)) != hipSuccess) {
      Fail("hipMalloc failed for " + name);
      return nullptr;
    }
    allocations.push_back(ptr);
    bytes += count * sizeof(float);
    if (hipMemcpy(ptr, host, count * sizeof(float), hipMemcpyHostToDevice) !=
        hipSuccess) {
      Fail("upload failed for " + name);
      return nullptr;
    }
    return ptr;
  }

  DeviceLayer Layer(const LayerWeights& l, bool swa) {
    DeviceLayer d;
    d.swa = swa;
    d.attn_norm = Copy(l.attn_norm);
    d.attn_q = Copy(l.attn_q);
    d.attn_k = Copy(l.attn_k);
    d.attn_v = Copy(l.attn_v);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.attn_output = Copy(l.attn_output);
    d.post_attention_norm = Copy(l.post_attention_norm);
    d.ffn_norm = Copy(l.ffn_norm);
    d.ffn_gate = Copy(l.ffn_gate);
    d.ffn_up = Copy(l.ffn_up);
    d.ffn_down = Copy(l.ffn_down);
    d.post_ffw_norm = Copy(l.post_ffw_norm);
    d.router = Copy(l.router);
    d.router_scale = Copy(l.router_scale);
    d.pre_ffw_norm_2 = Copy(l.pre_ffw_norm_2);
    d.post_ffw_norm_1 = Copy(l.post_ffw_norm_1);
    d.post_ffw_norm_2 = Copy(l.post_ffw_norm_2);
    d.ffn_gate_up_exps = Copy(l.ffn_gate_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.ffn_down_exps_scale = Copy(l.ffn_down_exps_scale);
    if (!l.layer_output_scale.empty()) {
      d.layer_output_scale =
          static_cast<const float*>(l.layer_output_scale.data)[0];
    }
    return d;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(const ModelWeights& w,
                                                 const core::GgufReader& reader,
                                                 std::string* error_msg) {
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  Uploader up{*stager, m->allocations_, m->bytes_, error_msg};
  m->token_embd_ = up.Copy(w.token_embd);
  m->output_ =
      w.output.data == w.token_embd.data ? m->token_embd_ : up.Copy(w.output);
  m->output_norm_ = up.Copy(w.output_norm);

  // The per-class inverse-frequency tables, computed in double exactly as the
  // oracle's RopeNeox does per pair. The full-class table folds the
  // proportional rope_freqs division; near-zero entries make the kernel's
  // rotation an exact pass-through for those pairs.
  const Config& c = m->config_;
  std::vector<float> inv_swa(c.rope_dim_swa / 2);
  for (std::size_t i = 0; i < inv_swa.size(); ++i) {
    inv_swa[i] = static_cast<float>(std::pow(
        static_cast<double>(c.rope_theta_swa),
        -2.0 * static_cast<double>(i) / static_cast<double>(c.rope_dim_swa)));
  }
  std::vector<float> inv_full(c.rope_dim_full / 2);
  const auto* factors = static_cast<const float*>(w.rope_freqs.data);
  for (std::size_t i = 0; i < inv_full.size(); ++i) {
    const double base = std::pow(
        static_cast<double>(c.rope_theta),
        -2.0 * static_cast<double>(i) / static_cast<double>(c.rope_dim_full));
    inv_full[i] = factors != nullptr ? static_cast<float>(base / factors[i])
                                     : static_cast<float>(base);
  }
  m->inv_freq_swa_ = up.CopyF32(inv_swa.data(), inv_swa.size(), "inv_freq_swa");
  m->inv_freq_full_ =
      up.CopyF32(inv_full.data(), inv_full.size(), "inv_freq_full");

  m->layers_.reserve(w.layers.size());
  for (std::uint32_t l = 0; l < w.layers.size(); ++l) {
    m->layers_.push_back(up.Layer(w.layers[l], c.IsSwa(l)));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  return m;
}

DeviceDraft::~DeviceDraft() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceDraft> DeviceDraft::Upload(const DraftWeights& w,
                                                 const core::GgufReader& reader,
                                                 std::string* error_msg) {
  std::unique_ptr<DeviceDraft> m(new DeviceDraft());
  m->config_ = w.config;
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  Uploader up{*stager, m->allocations_, m->bytes_, error_msg};
  m->pre_projection_ = up.Copy(w.pre_projection);
  m->post_projection_ = up.Copy(w.post_projection);
  m->token_embd_ = up.Copy(w.token_embd);
  m->output_norm_ = up.Copy(w.output_norm);

  // The draft's own inverse-frequency tables, computed exactly as the trunk's
  // but from the draft `rope_freqs`, so a draft layer's rope matches the trunk
  // layer whose KV cache it reads.
  const Config& c = m->config_;
  std::vector<float> inv_swa(c.rope_dim_swa / 2);
  for (std::size_t i = 0; i < inv_swa.size(); ++i) {
    inv_swa[i] = static_cast<float>(std::pow(
        static_cast<double>(c.rope_theta_swa),
        -2.0 * static_cast<double>(i) / static_cast<double>(c.rope_dim_swa)));
  }
  std::vector<float> inv_full(c.rope_dim_full / 2);
  const auto* factors = static_cast<const float*>(w.rope_freqs.data);
  for (std::size_t i = 0; i < inv_full.size(); ++i) {
    const double base = std::pow(
        static_cast<double>(c.rope_theta),
        -2.0 * static_cast<double>(i) / static_cast<double>(c.rope_dim_full));
    inv_full[i] = factors != nullptr ? static_cast<float>(base / factors[i])
                                     : static_cast<float>(base);
  }
  m->inv_freq_swa_ = up.CopyF32(inv_swa.data(), inv_swa.size(), "draft_inv_swa");
  m->inv_freq_full_ =
      up.CopyF32(inv_full.data(), inv_full.size(), "draft_inv_full");

  m->layers_.reserve(w.layers.size());
  for (std::uint32_t l = 0; l < w.layers.size(); ++l) {
    m->layers_.push_back(up.Layer(w.layers[l], c.IsSwa(l)));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  return m;
}

}  // namespace gufo::models::gemma4::rocm