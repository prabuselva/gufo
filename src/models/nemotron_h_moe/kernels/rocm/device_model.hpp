// GPU executor for Nemotron 3.5 Bolt: quantized weights uploaded to the
// device once, recurrent state resident, one token per Step mirroring the
// scalar oracle in reference.cpp through the nbl_/nmb_ kernel contracts.
#ifndef GUFO_MODELS_NEMOTRON_H_MOE_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_NEMOTRON_H_MOE_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/nemotron_h_moe/weights.hpp"

namespace gufo::models::nemotron_h_moe {

class DeviceModel {
public:
  /// `max_context` bounds the per-attention-layer KV capacity in rows.
  DeviceModel(const ModelWeights& weights, std::uint32_t max_context);
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Uploads every weight tensor and allocates the resident state.
  [[nodiscard]] bool Init(std::string* error_msg = nullptr);

  /// Advances one token; `logits` (vocab floats) receives the output
  /// distribution when non-empty.
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits,
                          std::string* error_msg = nullptr);

  /// Zeroes conv/SSM state and rewinds the position. Attention caches keep
  /// their bytes; every row is rewritten before it is read again.
  void Reset();

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }

  /// Parity hook: when set, every `Step` overwrites `dump` with the residual
  /// stream before each layer, mirroring `ReferenceModel::SetLayerDump`.
  void SetLayerDump(std::vector<float>* dump) noexcept { layer_dump_ = dump; }

private:
  struct LayerDev {
    Config::LayerKind kind{Config::LayerKind::kMamba};
    void* norm{nullptr};  // attn_norm F32 [hidden]
    // Attention (Q8_0 q/k/v/out) with the resident KV rows [cap][2*kvdim].
    void* q{nullptr};
    void* k{nullptr};
    void* v{nullptr};
    void* attn_out{nullptr};
    float* kv{nullptr};
    std::uint32_t kvdim{0};
    std::uint32_t kvh{0};
    // Mamba (NVFP4 in/out + per-tensor scales on the host, F32 rest).
    void* ssm_in{nullptr};
    void* ssm_out{nullptr};
    float ssm_in_scale{1.0F};
    float ssm_out_scale{1.0F};
    void* conv_w{nullptr};
    void* conv_b{nullptr};
    void* ssm_a{nullptr};
    void* ssm_d{nullptr};
    void* ssm_dt_b{nullptr};
    void* ssm_norm{nullptr};
    float* conv_state{nullptr};
    float* ssm_state{nullptr};
    // MoE: F32 router/bias on device, IQ2_XXS experts and latent Q8_0.
    void* router{nullptr};
    void* exp_b{nullptr};
    void* up_exps{nullptr};
    void* down_exps{nullptr};
    void* up_shexp{nullptr};
    void* down_shexp{nullptr};
    void* latent_up{nullptr};
    void* latent_down{nullptr};
    std::uint64_t n_experts{0};
    std::uint32_t expert_ff{0};
    std::uint32_t top_k{0};
  };

  [[nodiscard]] bool Upload(void** dst, const TensorRef& src,
                            std::string* error_msg);
  [[nodiscard]] bool UploadScalar(float* dst, const TensorRef& src,
                                  float* host_value, std::string* error_msg);
  void StepMamba(const LayerDev& l, hipStream_t stream);
  void StepAttn(const LayerDev& l, hipStream_t stream);
  void StepMoe(const LayerDev& l, hipStream_t stream);
  void Free();

  const ModelWeights& w_;
  const Config& c_;
  std::uint32_t max_context_;
  std::uint32_t position_{0};
  std::vector<LayerDev> layers_;
  // Global weights.
  void* token_embd_{nullptr};
  void* output_{nullptr};
  float* output_norm_{nullptr};
  // Single-token activations.
  float* x_{nullptr};
  float* normed_{nullptr};
  float* out_{nullptr};
  float* logits_{nullptr};
  // Mamba scratch.
  float* zxbc_{nullptr};
  float* y_{nullptr};
  // Attention scratch.
  float* q_{nullptr};
  float* k_{nullptr};
  float* v_{nullptr};
  float* o_{nullptr};
  // MoE scratch.
  float* router_logits_{nullptr};
  std::int32_t* ids_{nullptr};
  float* moe_w_{nullptr};
  float* lat_{nullptr};
  float* up_out_{nullptr};
  float* act_{nullptr};
  float* down_out_{nullptr};
  float* acc_{nullptr};
  float* moe_{nullptr};
  float* sh_{nullptr};
  int max_experts_{0};
  int max_expert_ff_{0};
  std::vector<float>* layer_dump_{nullptr};
};

}  // namespace gufo::models::nemotron_h_moe

#endif  // GUFO_MODELS_NEMOTRON_H_MOE_KERNELS_ROCM_DEVICE_MODEL_HPP_
