#ifndef GUFO_MODELS_QWEN36_A3B_ENGINE_HPP_
#define GUFO_MODELS_QWEN36_A3B_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen36_a3b/config.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::qwen36_a3b {

struct ModelWeights;
namespace rocm {
class DeviceModel;
class Executor;
class Session;
}  // namespace rocm

struct ModelOptions {
  std::uint32_t max_context = 4096;
  // Opt-in prefill attention sparsity: restrict each prefill query to the last
  // `attn_window` keys plus the first `attn_sink`. 0 disables sparsity (dense).
  std::uint32_t attn_window = 0;
  std::uint32_t attn_sink = 0;
};

class Session;

/// Gufo-owned API over the ROCm runtime: one resident model and one session
/// with independent context state. The Qwen3.6-35B-A3B executor is a single
/// shared object, so the model supports exactly one session.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] bool HasMtp() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] const tokenization::QwenTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Worst-case private device state one session of `context` tokens owns.
  [[nodiscard]] std::size_t SessionBytes(std::uint32_t context) const noexcept;
  /// Device scratch the executor keeps shared across every session.
  [[nodiscard]] std::size_t DeferredScratchBytes() const;

private:
  Model() = default;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<rocm::Executor> executor_;

  friend class Session;
};

/// One conversation's context. Sync feeds a prompt (reusing whatever prefix
/// the session already holds), Evaluate appends one token, DecodeStep samples
/// and feeds one token. The logits of the last token are kept.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Makes the session state equal to `prompt`: keeps the longest common
  /// prefix with the current tokens, reprocesses the rest.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  struct DecodeResult {
    std::vector<std::int32_t> tokens;
    bool stop{false};
    std::size_t drafted{0};
    std::size_t accepted{0};
  };
  /// Samples one token from the current logits, checks for stop, then feeds
  /// the token. The logits of the new last token are kept.
  [[nodiscard]] bool DecodeStep(std::size_t max_tokens,
                                sampling::SamplerState& sampler,
                                DecodeResult* result,
                                std::string* error_msg = nullptr,
                                bool stop_at_eos = true);
  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return valid_ ? std::span<const float>(logits_) : std::span<const float>{};
  }
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t ContextSize() const noexcept;
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept {
    return valid_ ? std::span<const std::int32_t>(tokens_)
                  : std::span<const std::int32_t>{};
  }
  void Reset();
  [[nodiscard]] bool IsValid() const noexcept { return valid_; }
  /// Bounds on the per-round MTP draft count. The session starts at `max`
  /// drafts, grows while drafts keep being fully accepted and shrinks on
  /// rejections down to `min`; `min` == `max` pins the count.
  void SetDraftLimits(std::uint32_t min_drafts, std::uint32_t max_drafts);
  /// Enables the MTP draft cache and speculative decoding for this session.
  /// Must be called before the first Sync: the draft cache is filled during
  /// prefill and cannot be rebuilt lazily. A plain (non-speculative) session
  /// leaves it off so prefill skips the draft block entirely.
  void SetMtpEnabled(bool enabled);

private:
  friend class Model;
  Session(std::shared_ptr<Model> model, std::unique_ptr<rocm::Session> session);

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::Session> session_;
  // Shared executor that runs every forward; owned by model_, which outlives
  // this session. Kept as a raw pointer for the scratch/logit accessors.
  rocm::Executor* executor_;
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  // Readback scratch for speculative decoding: the draft block's logits and
  // the verify rows (kMaxVerifyRows * vocab).
  std::vector<float> draft_logits_;
  std::vector<float> verify_rows_;
  std::uint32_t draft_min_{1};
  std::uint32_t draft_max_{4};
  std::uint32_t draft_k_{4};
  bool valid_{true};
};

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_ENGINE_HPP_