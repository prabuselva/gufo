#include "src/models/qwen36_a3b/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <string>
#include <vector>

#include "src/core/hip/weight_upload.hpp"

namespace gufo::models::qwen36_a3b::rocm {
namespace {

/// Streams one tensor's payload from disk into a fresh device allocation. The
/// GEMV tier reads exactly `row_bytes` per row, so no tail margin is needed.
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

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.attn_norm = Copy(l.attn_norm);
    d.post_attention_norm = Copy(l.post_attention_norm);
    if (l.linear) {
      d.ssm_qkv = Copy(l.ssm_qkv);
      d.ssm_gate = Copy(l.ssm_gate);
      d.ssm_conv1d = Copy(l.ssm_conv1d);
      d.ssm_alpha = Copy(l.ssm_alpha);
      d.ssm_beta = Copy(l.ssm_beta);
      d.ssm_dt = Copy(l.ssm_dt);
      d.ssm_a = Copy(l.ssm_a);
      d.ssm_norm = Copy(l.ssm_norm);
      d.ssm_out = Copy(l.ssm_out);
    } else {
      d.attn_q = Copy(l.attn_q);
      d.attn_k = Copy(l.attn_k);
      d.attn_v = Copy(l.attn_v);
      d.attn_out = Copy(l.attn_out);
      d.attn_q_norm = Copy(l.attn_q_norm);
      d.attn_k_norm = Copy(l.attn_k_norm);
    }
    d.router = Copy(l.router);
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.shexp_gate_inp = Copy(l.shexp_gate_inp);
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    d.nextn_eh_proj = Copy(l.nextn_eh_proj);
    d.nextn_shared_head_norm = Copy(l.nextn_shared_head_norm);
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
                                                 const MtpWeights* mtp,
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
  m->layers_.reserve(w.layers.size());
  for (const auto& l : w.layers) {
    m->layers_.push_back(up.Layer(l));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (mtp != nullptr) {
    m->mtp_ = up.Layer(mtp->block);
    m->has_mtp_ = true;
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  return m;
}

}  // namespace gufo::models::qwen36_a3b::rocm