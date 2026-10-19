#ifndef GUFO_MODELS_NEMOTRON_H_MOE_ENGINE_HPP_
#define GUFO_MODELS_NEMOTRON_H_MOE_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/nemotron_h_moe/config.hpp"
#include "src/models/nemotron_h_moe/tokenizer.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::nemotron_h_moe {

struct ModelWeights;
class DeviceModel;

struct ModelOptions {
  std::uint32_t max_context = 4096;
};

class Session;

/// Gufo-owned API over the ROCm runtime: one resident model with a single
/// session. The device executor keeps its recurrence scratch and KV caches
/// in one fixed allocation, so the model supports exactly one session.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::shared_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(std::string_view text) const;
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
  [[nodiscard]] const NemotronTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Worst-case private device state one session of `context` tokens owns.
  [[nodiscard]] std::size_t SessionBytes(std::uint32_t context) const noexcept;
  /// Device scratch the executor keeps shared across every session.
  [[nodiscard]] std::size_t DeferredScratchBytes() const noexcept;

private:
  Model() = default;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<NemotronTokenizer> tokenizer_;
  std::unique_ptr<DeviceModel> device_;
  std::weak_ptr<Session> live_session_;
  std::size_t resident_bytes_{0};
  std::size_t scratch_bytes_{0};

  friend class Session;
};

/// One conversation's context. Sync feeds a prompt (reusing whatever prefix
/// the session already holds), Evaluate appends one token. The logits of the
/// last token are kept. The mamba state cannot be rewound, so any prompt
/// divergence restarts the whole context.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Makes the session state equal to `prompt`: appends when the current
  /// tokens are a prefix of the prompt, otherwise restarts and replays it.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
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

private:
  friend class Model;
  Session(std::shared_ptr<Model> model, std::uint32_t max_context);

  std::shared_ptr<Model> model_;
  DeviceModel* device_;
  std::uint32_t max_context_;
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  bool valid_{true};
};

}  // namespace gufo::models::nemotron_h_moe

#endif  // GUFO_MODELS_NEMOTRON_H_MOE_ENGINE_HPP_
