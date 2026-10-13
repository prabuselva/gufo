#include "src/models/gemma4/kernels/rocm/device_model.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/hip/weight_upload.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/gemma4/kernels/rocm/gemv.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

/// Requantizes one k-quant row (Q4_K/Q5_K/Q6_K, `cols` elements, `cols % 256
/// == 0`) to Q8_0 at `q8` (`cols / 32` blocks of 34 bytes). The row is
/// dequantized to float through the canonical oracle routine, then each
/// 32-element group is encoded as {half d; int8 qs[32]} with d = amax / 127
/// rounded to half and qs the nearest integer of v / d. Q8_0 is finer than any
/// k-quant, so the requant error is below the source quant's own step;
/// `scratch` holds `cols` floats.
void RequantKQuantRowToQ8_0(const void* src, core::GgmlType type,
                            std::uint8_t* q8, std::size_t cols,
                            float* scratch) {
  switch (type) {
    case core::GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(src, scratch, cols);
      break;
    case core::GgmlType::kQ4_0:
      gufo::quant::DequantizeQ4_0(src, scratch, cols);
      break;
    case core::GgmlType::kQ5_K:
      gufo::quant::DequantizeQ5_K(src, scratch, cols);
      break;
    default:
      gufo::quant::DequantizeQ6_K(src, scratch, cols);
      break;
  }
  const std::size_t nblocks = cols / 32;
  for (std::size_t b = 0; b < nblocks; ++b) {
    const float* v = scratch + b * 32;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
      const float a = std::fabs(v[i]);
      if (a > amax) {
        amax = a;
      }
    }
    const __half dh = __float2half_rn(amax * (1.0f / 127.0f));
    const float d = __half2float(dh);
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    gufo::quant::block_q8_0 blk;
    std::memcpy(&blk.d, &dh, sizeof(blk.d));
    for (int i = 0; i < 32; ++i) {
      long q = std::lrintf(v[i] * id);
      if (q > 127) {
        q = 127;
      } else if (q < -127) {
        q = -127;
      }
      blk.qs[i] = static_cast<std::int8_t>(q);
    }
    std::memcpy(q8 + b * sizeof(gufo::quant::block_q8_0), &blk, sizeof(blk));
  }
}

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

  // Dense projections uploaded as BF16 are retyped to binary16 and converted
  // in place after the staged uploads finish. F16 gives the dense tier a
  // native tensor-core route (no BF16 staging/conversion per call), and any
  // weight magnitude outside the binary16 range fails the load rather than
  // saturating silently.
  struct HalfJob {
    void* data;
    std::size_t count;
    std::string name;
  };
  std::vector<HalfJob> halves;

  void Half(DeviceTensor& d, const std::string& name) {
    if (!ok || d.empty() || d.type != core::GgmlType::kBF16) {
      return;
    }
    halves.push_back(
        {d.data, static_cast<std::size_t>(d.rows) * d.cols * d.experts, name});
    d.type = core::GgmlType::kF16;
  }

  // Runs the queued in-place BF16 -> binary16 conversions. Must be called
  // after the staging pipeline has flushed, so every source buffer holds its
  // final bytes.
  bool ConvertHalves() {
    if (halves.empty() || !ok) {
      return ok;
    }
    std::vector<std::uint32_t*> counters(halves.size(), nullptr);
    for (std::size_t i = 0; i < halves.size(); ++i) {
      if (hipMalloc(&counters[i], sizeof(std::uint32_t)) != hipSuccess ||
          hipMemset(counters[i], 0, sizeof(std::uint32_t)) != hipSuccess) {
        Fail("hipMalloc failed for BF16 conversion check of " + halves[i].name);
        break;
      }
      ConvertBf16ToHalf(halves[i].data, halves[i].count, counters[i], nullptr);
    }
    if (ok && hipDeviceSynchronize() != hipSuccess) {
      Fail("device sync failed during BF16 -> binary16 conversion");
    }
    for (std::size_t i = 0; ok && i < halves.size(); ++i) {
      std::uint32_t overflow = 0;
      if (counters[i] != nullptr) {
        if (hipMemcpy(&overflow, counters[i], sizeof(overflow),
                      hipMemcpyDeviceToHost) != hipSuccess) {
          Fail("readback failed for BF16 conversion check of " +
               halves[i].name);
          break;
        }
      }
      if (overflow > 0) {
        Fail("tensor " + halves[i].name + " has " + std::to_string(overflow) +
             " BF16 values outside the binary16 range");
      }
    }
    for (std::uint32_t* c : counters) {
      if (c != nullptr) {
        (void)hipFree(c);
      }
    }
    return ok;
  }

  DeviceTensor Copy(const TensorRef& t, bool expert = false) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    // The dense GEMM tier decodes only Q8_0/F32/BF16 and the routed WMMA tier
    // decodes Q8_0/Q4_K/Q5_K/BF16. Any k-quant the target tier cannot decode is
    // requantized to Q8_0 on the host during upload: Q6_K everywhere (neither
    // tier decodes it), Q4_0 everywhere (a QAT artifact stores every matmul as
    // Q4_0 and neither tier decodes it) and Q4_K/Q5_K on dense tensors (only
    // the routed tier does). A *_K_M artifact stores the dense
    // attention/FFN projections as Q4_K, so those upcast while the routed
    // experts stay native.
    const bool upcast =
        t.type == core::GgmlType::kQ6_K || t.type == core::GgmlType::kQ4_0 ||
        (!expert &&
         (t.type == core::GgmlType::kQ4_K || t.type == core::GgmlType::kQ5_K));
    if (upcast) {
      return CopyUpcast(t);
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

  // The GEMV and WMMA tiers decode Q8_0/Q4_K/Q5_K but not Q6_K, and the dense
  // GEMM tier decodes neither Q4_K nor Q5_K, so the k-quant tensors a *_K_XL or
  // *_K_M artifact stores in a tier that cannot decode them are requantized to
  // Q8_0 on the host during upload (see RequantKQuantRowToQ8_0). The rows are
  // read straight from the mapped file and copied up in one transfer; the
  // resident tensor is then a plain Q8_0 stack the rest of the pipeline already
  // handles.
  DeviceTensor CopyUpcast(const TensorRef& t) {
    DeviceTensor d;
    const std::size_t cols = static_cast<std::size_t>(t.cols);
    const std::size_t rows = static_cast<std::size_t>(t.rows);
    const std::size_t experts = static_cast<std::size_t>(t.experts);
    const std::size_t src_row = t.RowBytes();
    const std::size_t q8_row = (cols / 32) * sizeof(gufo::quant::block_q8_0);
    const std::size_t total = q8_row * rows * experts;
    std::vector<std::uint8_t> host(total);
    std::vector<float> scratch(cols);
    for (std::size_t e = 0; e < experts; ++e) {
      const std::uint8_t* src = t.Expert(e);
      for (std::size_t r = 0; r < rows; ++r) {
        RequantKQuantRowToQ8_0(src + src_row * r, t.type,
                               host.data() + q8_row * (e * rows + r), cols,
                               scratch.data());
      }
    }
    void* ptr = nullptr;
    if (hipMalloc(&ptr, total) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(total) + " bytes)");
      return d;
    }
    allocations.push_back(ptr);
    bytes += total;
    if (hipMemcpy(ptr, host.data(), total, hipMemcpyHostToDevice) !=
        hipSuccess) {
      Fail("upload failed for " + std::string(t.name));
      return d;
    }
    d.data = ptr;
    d.type = core::GgmlType::kQ8_0;
    d.cols = static_cast<std::uint32_t>(cols);
    d.rows = static_cast<std::uint32_t>(rows);
    d.experts = static_cast<std::uint32_t>(experts);
    d.row_bytes = q8_row;
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

  // Same retyping as Half() for a fused buffer: one BF16 -> binary16 job
  // over the whole buffer, retyping the per-tensor views too. Binary16
  // elements are the same size as BF16, so every view offset stays valid.
  void HalfFused(DeviceTensor& fused, const std::string& name,
                 std::initializer_list<DeviceTensor*> views) {
    if (!ok || fused.empty() || fused.type != core::GgmlType::kBF16) {
      return;
    }
    halves.push_back(
        {fused.data, static_cast<std::size_t>(fused.rows) * fused.cols, name});
    fused.type = core::GgmlType::kF16;
    for (DeviceTensor* v : views) {
      if (!v->empty()) {
        v->type = core::GgmlType::kF16;
      }
    }
  }

  struct FusePart {
    const TensorRef* t;
    bool required;
  };

  // Uploads projections that share type and input columns as ONE row-major
  // concatenation (q, k, v or gate, up in order) and points the per-tensor
  // views at their column slices. GGUF stores every part as a stack of
  // fixed-length rows over the same columns, so the concatenation is exactly
  // the fused [sum(rows)][row_bytes] matrix every decoder already reads:
  // a view at its start offset is indistinguishable from a standalone
  // tensor, while prefill can run one wide GEMM instead of one per part.
  // Returns an empty tensor (and leaves every view untouched) when the
  // parts cannot share a buffer, so the caller stays on the per-tensor
  // path. Optional empty parts just drop out of the fusion.
  DeviceTensor Fuse(const std::vector<FusePart>& parts,
                    const std::vector<DeviceTensor*>& views,
                    const std::string& name) {
    DeviceTensor f;
    if (!ok || parts.size() != views.size() || parts.size() < 2) {
      return f;
    }
    std::size_t total = 0;
    core::GgmlType type{core::GgmlType::kF32};
    std::size_t cols = 0;
    std::size_t row_bytes = 0;
    bool have = false;
    for (const FusePart& p : parts) {
      const TensorRef& t = *p.t;
      if (t.empty()) {
        if (p.required) {
          return f;
        }
        continue;
      }
      // Requantized parts would change row length and per-row layout; the
      // per-tensor path already handles those encodings.
      const bool native =
          t.type != core::GgmlType::kQ6_K && t.type != core::GgmlType::kQ4_0 &&
          t.type != core::GgmlType::kQ4_K && t.type != core::GgmlType::kQ5_K;
      if (!native || t.experts != 1) {
        return f;
      }
      if (!have) {
        type = t.type;
        cols = t.cols;
        row_bytes = t.RowBytes();
        have = true;
      } else if (t.type != type || t.cols != cols ||
                 t.RowBytes() != row_bytes) {
        return f;
      }
      total += t.SizeBytes();
    }
    if (!have) {
      return f;
    }
    void* ptr = nullptr;
    if (hipMalloc(&ptr, total) != hipSuccess) {
      Fail("hipMalloc failed for " + name + " (" + std::to_string(total) +
           " bytes)");
      return f;
    }
    allocations.push_back(ptr);
    bytes += total;
    std::size_t off = 0;
    std::uint32_t rows = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
      const TensorRef& t = *parts[i].t;
      if (t.empty()) {
        continue;
      }
      const std::size_t size = t.SizeBytes();
      auto* dst = static_cast<std::uint8_t*>(ptr) + off;
      if (!stager.Copy(t.shard, t.file_offset, size, dst, error)) {
        Fail("upload failed for " + name + " part " + std::string(t.name) +
             (error != nullptr ? ": " + *error : std::string()));
        return f;
      }
      views[i]->data = dst;
      views[i]->type = t.type;
      views[i]->cols = static_cast<std::uint32_t>(t.cols);
      views[i]->rows = static_cast<std::uint32_t>(t.rows);
      views[i]->experts = 1;
      views[i]->row_bytes = t.RowBytes();
      rows += static_cast<std::uint32_t>(t.rows);
      off += size;
    }
    f.data = ptr;
    f.type = type;
    f.cols = static_cast<std::uint32_t>(cols);
    f.rows = rows;
    f.row_bytes = row_bytes;
    return f;
  }

  DeviceLayer Layer(const LayerWeights& l, bool swa) {
    DeviceLayer d;
    d.swa = swa;
    d.attn_norm = Copy(l.attn_norm);
    d.attn_qkv =
        Fuse({{&l.attn_q, true}, {&l.attn_k, true}, {&l.attn_v, false}},
             {&d.attn_q, &d.attn_k, &d.attn_v}, "attn_qkv");
    if (d.attn_qkv.empty()) {
      d.attn_q = Copy(l.attn_q);
      Half(d.attn_q, "attn_q");
      d.attn_k = Copy(l.attn_k);
      Half(d.attn_k, "attn_k");
      d.attn_v = Copy(l.attn_v);
      Half(d.attn_v, "attn_v");
    } else {
      HalfFused(d.attn_qkv, "attn_qkv", {&d.attn_q, &d.attn_k, &d.attn_v});
    }
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.attn_output = Copy(l.attn_output);
    Half(d.attn_output, "attn_output");
    d.post_attention_norm = Copy(l.post_attention_norm);
    d.ffn_norm = Copy(l.ffn_norm);
    d.ffn_gate_up = Fuse({{&l.ffn_gate, true}, {&l.ffn_up, true}},
                         {&d.ffn_gate, &d.ffn_up}, "ffn_gate_up");
    if (d.ffn_gate_up.empty()) {
      d.ffn_gate = Copy(l.ffn_gate);
      Half(d.ffn_gate, "ffn_gate");
      d.ffn_up = Copy(l.ffn_up);
      Half(d.ffn_up, "ffn_up");
    } else {
      HalfFused(d.ffn_gate_up, "ffn_gate_up", {&d.ffn_gate, &d.ffn_up});
    }
    d.ffn_down = Copy(l.ffn_down);
    Half(d.ffn_down, "ffn_down");
    d.post_ffw_norm = Copy(l.post_ffw_norm);
    d.router = Copy(l.router);
    d.router_scale = Copy(l.router_scale);
    d.pre_ffw_norm_2 = Copy(l.pre_ffw_norm_2);
    d.post_ffw_norm_1 = Copy(l.post_ffw_norm_1);
    d.post_ffw_norm_2 = Copy(l.post_ffw_norm_2);
    d.ffn_gate_up_exps = Copy(l.ffn_gate_up_exps, true);
    d.ffn_down_exps = Copy(l.ffn_down_exps, true);
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
  up.Half(m->token_embd_, "token_embd");
  // The tied head shares the embedding buffer; retyping (and converting) it
  // once covers both, so only an untied head gets its own job.
  if (w.output.data == w.token_embd.data) {
    m->output_ = m->token_embd_;
  } else {
    m->output_ = up.Copy(w.output);
    up.Half(m->output_, "output");
  }
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
  if (!up.ok || !stager->Finish(error_msg) || !up.ConvertHalves()) {
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
  up.Half(m->pre_projection_, "draft_pre_projection");
  m->post_projection_ = up.Copy(w.post_projection);
  up.Half(m->post_projection_, "draft_post_projection");
  m->token_embd_ = up.Copy(w.token_embd);
  up.Half(m->token_embd_, "draft_token_embd");
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
  m->inv_freq_swa_ =
      up.CopyF32(inv_swa.data(), inv_swa.size(), "draft_inv_swa");
  m->inv_freq_full_ =
      up.CopyF32(inv_full.data(), inv_full.size(), "draft_inv_full");

  m->layers_.reserve(w.layers.size());
  for (std::uint32_t l = 0; l < w.layers.size(); ++l) {
    m->layers_.push_back(up.Layer(w.layers[l], c.IsSwa(l)));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (!up.ok || !stager->Finish(error_msg) || !up.ConvertHalves()) {
    return nullptr;
  }
  return m;
}

}  // namespace gufo::models::gemma4::rocm