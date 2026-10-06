#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/config.hpp"
#include "src/models/gemma4/weights.hpp"

namespace gufo::models::gemma4::rocm {

/// One weight resident in device memory, still in its GGUF encoding. The
/// GEMV/GEMM tiers decode Q8_0/F32/BF16 rows on the fly; the fused operators
/// read the F32 norms and scales directly. Nothing is dequantized at load, so
/// the resident footprint is the artifact's own.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};
  std::size_t row_bytes{0};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] const float* f32() const noexcept {
    return static_cast<const float*>(data);
  }
  /// Base address of one stacked expert matrix.
  [[nodiscard]] const void* Expert(std::uint32_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) +
           row_bytes * static_cast<std::size_t>(rows) * e;
  }
};

/// The device-side mirror of one trunk layer. SWA layers carry `attn_v`;
/// full-attention layers derive V from `attn_k` and leave it empty.
struct DeviceLayer {
  bool swa{true};
  DeviceTensor attn_norm, attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm,
      attn_output, post_attention_norm;
  DeviceTensor ffn_norm, ffn_gate, ffn_up, ffn_down, post_ffw_norm;
  DeviceTensor router, router_scale, pre_ffw_norm_2, post_ffw_norm_1,
      post_ffw_norm_2;
  DeviceTensor ffn_gate_up_exps, ffn_down_exps, ffn_down_exps_scale;
  float layer_output_scale{1.0F};
};

/// The trunk uploaded to the GPU in its native encodings, plus the two
/// per-class inverse-frequency tables the fused rope kernels consume:
/// `inv_freq_swa[i] = theta_swa^(-2i/rope_dim_swa)` and
/// `inv_freq_full[i] = theta_full^(-2i/rope_dim_full) / rope_freqs[i]`,
/// matching the scalar oracle's `RopeNeox` exactly.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Streams every tensor from the same open file used to bind its metadata.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept { return config_; }
  [[nodiscard]] const DeviceTensor& token_embd() const noexcept {
    return token_embd_;
  }
  [[nodiscard]] const DeviceTensor& output() const noexcept { return output_; }
  [[nodiscard]] const DeviceTensor& output_norm() const noexcept {
    return output_norm_;
  }
  [[nodiscard]] const std::vector<DeviceLayer>& layers() const noexcept {
    return layers_;
  }
  /// [head_dim_swa / 2] and [head_dim_full / 2] F32 tables; the full table
  /// already carries the proportional `rope_freqs` division.
  [[nodiscard]] const float* inv_freq(std::uint32_t layer) const noexcept {
    return config_.IsSwa(layer) ? inv_freq_swa_ : inv_freq_full_;
  }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }

private:
  DeviceModel() = default;

  Config config_;
  DeviceTensor token_embd_;
  DeviceTensor output_;
  DeviceTensor output_norm_;
  float* inv_freq_swa_{nullptr};
  float* inv_freq_full_{nullptr};
  std::vector<DeviceLayer> layers_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

/// The MTP draft uploaded to the GPU. Its four layers carry only the Q
/// projection, the dense FFN and their norms (they read the trunk KV cache
/// read-only), plus the pre/post projections that map between the doubled
/// trunk-width input and the draft hidden width. The inverse-frequency tables
/// are built from the draft's own `rope_freqs`, which the KV-share contract
/// keeps consistent with the trunk layer whose cache each draft layer reads.
class DeviceDraft {
 public:
  ~DeviceDraft();
  DeviceDraft(const DeviceDraft&) = delete;
  DeviceDraft& operator=(const DeviceDraft&) = delete;

  /// Streams every draft tensor from the sidecar file used to bind it.
  [[nodiscard]] static std::unique_ptr<DeviceDraft> Upload(
      const DraftWeights& weights, const core::GgufReader& reader,
      std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept { return config_; }
  [[nodiscard]] const DeviceTensor& pre_projection() const noexcept {
    return pre_projection_;
  }
  [[nodiscard]] const DeviceTensor& post_projection() const noexcept {
    return post_projection_;
  }
  [[nodiscard]] const DeviceTensor& token_embd() const noexcept {
    return token_embd_;
  }
  [[nodiscard]] const DeviceTensor& output_norm() const noexcept {
    return output_norm_;
  }
  [[nodiscard]] const std::vector<DeviceLayer>& layers() const noexcept {
    return layers_;
  }
  [[nodiscard]] const float* inv_freq(std::uint32_t layer) const noexcept {
    return config_.IsSwa(layer) ? inv_freq_swa_ : inv_freq_full_;
  }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }

 private:
  DeviceDraft() = default;

  Config config_;
  DeviceTensor pre_projection_;
  DeviceTensor post_projection_;
  DeviceTensor token_embd_;
  DeviceTensor output_norm_;
  float* inv_freq_swa_{nullptr};
  float* inv_freq_full_{nullptr};
  std::vector<DeviceLayer> layers_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_DEVICE_MODEL_HPP_