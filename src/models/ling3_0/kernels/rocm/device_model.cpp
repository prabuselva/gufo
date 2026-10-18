#include "src/models/ling3_0/kernels/rocm/device_model.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "src/core/hip/weight_upload.hpp"
#include "src/models/ling3_0/kernels/rocm/gemv.hpp"

namespace gufo::models::ling3_0::rocm {
namespace {

/// Streams one tensor's payload from disk into a fresh device allocation. The
/// GEMV tier reads exactly `row_bytes` per row, so no tail margin is needed.
/// Every stored format (F32, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0) decodes on the fly,
/// so nothing is transformed during upload.
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
    d.mla = l.mla;
    d.attn_norm = Copy(l.attn_norm);
    d.ffn_norm = Copy(l.ffn_norm);
    if (l.mla) {
      d.attn_q = Copy(l.attn_q);
      d.attn_q_a = Copy(l.attn_q_a);
      d.attn_q_a_norm = Copy(l.attn_q_a_norm);
      d.attn_q_b = Copy(l.attn_q_b);
      d.attn_kv_a = Copy(l.attn_kv_a);
      d.attn_kv_a_norm = Copy(l.attn_kv_a_norm);
      d.attn_k_b = Copy(l.attn_k_b);
      d.attn_v_b = Copy(l.attn_v_b);
      d.attn_gate = Copy(l.attn_gate);
      d.attn_out = Copy(l.attn_out);
    } else {
      d.ssm_q = Copy(l.ssm_q);
      d.ssm_k = Copy(l.ssm_k);
      d.ssm_v = Copy(l.ssm_v);
      d.ssm_conv_q = Copy(l.ssm_conv_q);
      d.ssm_conv_k = Copy(l.ssm_conv_k);
      d.ssm_conv_v = Copy(l.ssm_conv_v);
      d.ssm_f = Copy(l.ssm_f);
      d.ssm_g = Copy(l.ssm_g);
      d.ssm_beta = Copy(l.ssm_beta);
      d.ssm_a = Copy(l.ssm_a);
      d.ssm_dt = Copy(l.ssm_dt);
      d.ssm_norm = Copy(l.ssm_norm);
      d.ssm_out = Copy(l.ssm_out);
    }
    d.ffn_gate = Copy(l.ffn_gate);
    d.ffn_up = Copy(l.ffn_up);
    d.ffn_down = Copy(l.ffn_down);
    d.router = Copy(l.router);
    d.exp_probs_b = Copy(l.exp_probs_b);
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    d.nextn_eh_proj = Copy(l.nextn_eh_proj);
    d.nextn_shared_head_norm = Copy(l.nextn_shared_head_norm);
    return d;
  }

  // Retypes the resident F32 router matrices to binary16 out of place (the
  // port of qwen36 lever 4.1b): 4 bytes shrink to 2 so each router is
  // converted into a fresh buffer, swapped over and freed, halving both the
  // prefill router GEMM's weight traffic and the per-token decode read.
  // Runs after the staging flush, when layers_ / mtp_ are final and the
  // DeviceTensor addresses are stable; any weight magnitude outside the
  // binary16 range fails the load.
  bool ConvertRouters(std::vector<DeviceLayer>& layers, DeviceLayer* mtp) {
    struct Job {
      DeviceTensor* tensor;
      void* src;
      std::size_t src_size;
      void* dst;
      std::uint32_t* counter;
      std::string name;
    };
    std::vector<Job> jobs;
    auto want = [&](DeviceTensor& t, const std::string& name) {
      if (t.empty() || t.type != core::GgmlType::kF32 || !ok) {
        return;
      }
      Job job{
          &t,
          t.data,
          static_cast<std::size_t>(t.rows) * t.cols * t.experts * sizeof(float),
          nullptr,
          nullptr,
          name};
      const std::size_t count =
          static_cast<std::size_t>(t.rows) * t.cols * t.experts;
      if (hipMalloc(&job.dst, count * sizeof(__half)) != hipSuccess) {
        Fail("hipMalloc failed for F32 retype of " + name);
        return;
      }
      allocations.push_back(job.dst);
      bytes += count * sizeof(__half);
      if (hipMalloc(&job.counter, sizeof(std::uint32_t)) != hipSuccess ||
          hipMemset(job.counter, 0, sizeof(std::uint32_t)) != hipSuccess) {
        Fail("hipMalloc failed for F32 conversion check of " + name);
        (void)hipFree(job.dst);
        allocations.erase(
            std::find(allocations.begin(), allocations.end(), job.dst));
        return;
      }
      ConvertF32ToHalf(static_cast<const float*>(t.data),
                       static_cast<__half*>(job.dst), count, job.counter,
                       nullptr);
      jobs.push_back(job);
    };
    for (std::size_t i = 0; i < layers.size() && ok; ++i) {
      want(layers[i].router, "layer " + std::to_string(i) + " router");
    }
    if (mtp != nullptr && ok) {
      want(mtp->router, "mtp router");
    }
    if (ok && hipDeviceSynchronize() != hipSuccess) {
      Fail("device sync failed during F32 -> binary16 router retype");
    }
    for (Job& job : jobs) {
      std::uint32_t overflow = 0;
      if (hipMemcpy(&overflow, job.counter, sizeof(overflow),
                    hipMemcpyDeviceToHost) != hipSuccess) {
        Fail("readback failed for F32 conversion check of " + job.name);
        break;
      }
      if (overflow > 0) {
        Fail("tensor " + job.name + " has " + std::to_string(overflow) +
             " F32 values outside the binary16 range");
        break;
      }
      DeviceTensor* t = job.tensor;
      bytes -= job.src_size;
      auto slot = std::find(allocations.begin(), allocations.end(), job.src);
      if (slot != allocations.end()) {
        allocations.erase(slot);
      }
      t->data = job.dst;
      t->type = core::GgmlType::kF16;
      t->row_bytes = static_cast<std::size_t>(t->cols) * sizeof(__half);
      (void)hipFree(job.src);
    }
    for (Job& job : jobs) {
      (void)hipFree(job.counter);
    }
    return ok;
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
  if (!up.ConvertRouters(m->layers_, m->has_mtp_ ? &m->mtp_ : nullptr)) {
    return nullptr;
  }
  return m;
}

}  // namespace gufo::models::ling3_0::rocm