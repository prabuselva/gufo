#ifndef GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen36_a3b/weights.hpp"

namespace gufo::models::qwen36_a3b::rocm {

/// One weight resident in device memory, still in its GGUF encoding. The GEMV
/// tier decodes Q8_0/F32/BF16 rows on the fly; the fused operators read the
/// F32 norms, convolution and per-head scalars directly. Nothing is
/// dequantized at load, so the resident footprint is the artifact's own.
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

/// The device-side mirror of one trunk (or MTP) layer. Linear layers fill the
/// Gated DeltaNet fields; full-attention layers fill the GQA fields. The
/// mixture-of-experts fields are present on every layer.
struct DeviceLayer {
  bool linear{false};
  DeviceTensor attn_norm, post_attention_norm;
  // Gated DeltaNet (linear layers).
  DeviceTensor ssm_qkv, ssm_gate, ssm_conv1d, ssm_alpha, ssm_beta, ssm_dt,
      ssm_a, ssm_norm, ssm_out;
  // Gated grouped-query full attention (full-attention layers).
  DeviceTensor attn_q, attn_k, attn_v, attn_out, attn_q_norm, attn_k_norm;
  // Mixture of experts with one always-on shared expert.
  DeviceTensor router, ffn_gate_exps, ffn_up_exps, ffn_down_exps,
      shexp_gate_inp, shexp_gate, shexp_up, shexp_down;
  // Speculative `nextn` block only.
  DeviceTensor nextn_enorm, nextn_hnorm, nextn_eh_proj, nextn_shared_head_norm;
};

/// The trunk (and, from the same artifact, the MTP draft block) uploaded to
/// the GPU. Weights stay in their GGUF encoding; the executor decodes them on
/// the fly. The trunk and the draft share one file, so one reader suffices.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Streams every tensor from the same open file used to bind its metadata.
  /// `mtp` (and its bound weights) is optional; passing it also uploads the
  /// draft block.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      const MtpWeights* mtp, std::string* error_msg = nullptr);

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
  [[nodiscard]] bool has_mtp() const noexcept { return has_mtp_; }
  [[nodiscard]] const DeviceLayer& mtp() const noexcept { return mtp_; }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }

private:
  DeviceModel() = default;

  Config config_;
  DeviceTensor token_embd_;
  DeviceTensor output_;
  DeviceTensor output_norm_;
  std::vector<DeviceLayer> layers_;
  DeviceLayer mtp_;
  bool has_mtp_{false};
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

}  // namespace gufo::models::qwen36_a3b::rocm

#endif  // GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_DEVICE_MODEL_HPP_