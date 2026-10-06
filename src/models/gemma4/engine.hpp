#ifndef GUFO_MODELS_GEMMA4_ENGINE_HPP_
#define GUFO_MODELS_GEMMA4_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/config.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::gemma4 {

struct ModelWeights;
struct DraftWeights;
class Tokenizer;
namespace rocm {
class DeviceModel;
class DeviceDraft;
class Executor;
class Session;
}  // namespace rocm

struct ModelOptions {
  std::uint32_t max_context = 4096;
  // Rows per batched prefill pass. Bounds the shared prefill scratch; the
  // executor loops over chunks of this size for longer prompts.
  std::uint32_t prefill_chunk = 512;
  // Optional MTP draft sidecar artifact. When set, the draft is loaded and
  // speculative decoding is available; sessions opt in with SetMtpEnabled.
  std::string draft_path;
};

class Session;

/// Gufo-owned API over the ROCm runtime: one resident trunk and one session
/// with independent context state. The Gemma-4-26B-A4B executor serializes
/// compute on shared scratch, so the model supports exactly one session.
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
  /// Renders a conversation with the pinned Gemma-4 chat template and encodes
  /// it. The template already emits BOS, so the tokenizer does not add one.
  [[nodiscard]] std::vector<std::int32_t> EncodeChat(
      std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
      const ChatTemplateOptions& options) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] const Tokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// True when an MTP draft sidecar was loaded and speculative decoding is
  /// available to sessions.
  [[nodiscard]] bool HasMtp() const noexcept {
    return draft_device_ != nullptr;
  }
  /// Worst-case private device state one session of `context` tokens owns.
  [[nodiscard]] std::size_t SessionBytes(std::uint32_t context) const noexcept;

private:
  Model() = default;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<rocm::Executor> executor_;
  // Optional MTP draft sidecar: a separate GGUF whose tensors are uploaded to
  // the device and attached to the trunk executor. Null when no draft loaded.
  std::shared_ptr<core::GgufReader> draft_reader_;
  std::unique_ptr<DraftWeights> draft_weights_;
  std::unique_ptr<rocm::DeviceDraft> draft_device_;

  friend class Session;
};

/// One conversation's context. Sync feeds a prompt (reusing whatever prefix the
/// session already holds), Evaluate appends one token, DecodeStep samples and
/// feeds one token. The logits of the last token are kept.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Makes the session state equal to `prompt`: keeps the longest common
  /// prefix with the current tokens, reprocesses the rest through prefill.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  struct DecodeResult {
    std::vector<std::int32_t> tokens;
    bool stop{false};
    // Speculative counters: drafts proposed and accepted across the round.
    // Zero on the non-speculative path.
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
  /// Enables or disables MTP speculation for this session. A no-op when the
  /// model has no draft sidecar; the session then always decodes one token.
  void SetMtpEnabled(bool enabled);
  /// Bounds the speculative draft length to [min_drafts, max_drafts], clamped
  /// to the executor's verified-block limit. The adaptive length starts at the
  /// upper bound and shrinks toward the lower one on repeated rejections.
  void SetDraftLimits(std::uint32_t min_drafts, std::uint32_t max_drafts);

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
  // Speculative scratch: the draft logits (one vocab row), the draft ids, the
  // downloaded verify logits ([k + 1][vocab]) and the adaptive draft length.
  std::vector<float> draft_logits_;
  std::vector<std::int32_t> drafts_;
  std::vector<float> verify_rows_;
  std::uint32_t draft_min_{1U};
  std::uint32_t draft_max_{4U};
  std::uint32_t draft_k_{4U};
  bool mtp_enabled_{false};
  bool valid_{true};
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_ENGINE_HPP_