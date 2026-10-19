#include "src/models/nemotron_h_moe/serve_runner.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/nemotron_h_moe/chat_template.hpp"

namespace gufo::models::nemotron_h_moe {
namespace {

using server::ChatRequest;
using server::TextModelRunner;
using server::TextPreparedPrompt;
using server::TextPromptContext;
using server::TextRunnerState;
using server::TextRunnerToken;

constexpr std::string_view kNemotronStateAbi = "nemotron-h-moe-rocm-session-v1";

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::vector<std::int32_t> EngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Nemotron Bolt token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

class NemotronTextRunnerState final : public TextRunnerState {
public:
  NemotronTextRunnerState(const std::shared_ptr<Model>& model,
                          std::uint32_t max_context) {
    std::string error;
    session_ = model->CreateSession(max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create Nemotron Bolt session: " +
                               error);
    }
  }

  void Invalidate() noexcept override {
    session_->Reset();
    position_ = 0;
  }

  [[nodiscard]] Session& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  std::shared_ptr<Session> session_;
  std::size_t position_{0};
};

NemotronTextRunnerState& RequireNemotronState(TextRunnerState& state) {
  auto* runner = dynamic_cast<NemotronTextRunnerState*>(&state);
  if (runner == nullptr) {
    throw std::logic_error("text runner state is not Nemotron Bolt");
  }
  return *runner;
}

const NemotronTextRunnerState& RequireNemotronState(
    const TextRunnerState& state) {
  const auto* runner = dynamic_cast<const NemotronTextRunnerState*>(&state);
  if (runner == nullptr) {
    throw std::logic_error("text runner state is not Nemotron Bolt");
  }
  return *runner;
}

/// Serial runner for Nemotron 3.5 Bolt. The model owns one shared executor;
/// its KV and recurrent state live inside it, so the pool creates exactly
/// one session.
class NemotronTextRunner final : public TextModelRunner {
public:
  NemotronTextRunner(std::shared_ptr<Model> model, std::uint32_t max_context)
      : model_(std::move(model)), max_context_(max_context) {}

  // Grammar-constrained decoding needs token-level pieces: a greedy-merge
  // tokenizer otherwise hides the native delimiters behind merges the
  // sampler cannot validate against the grammar.
  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->tokenizer().GetVocabSize(), [this](std::uint32_t id) {
          const auto& tokenizer = model_->tokenizer();
          return sampling::ConstraintVocabulary::Piece{
              tokenizer.DecodeTokenCopy(id), tokenizer.IsStopToken(id)};
        });
  }

  [[nodiscard]] server::TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kNemotronStateAbi),
        .max_context = max_context_,
        .capabilities =
            server::TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = false,
                .fork = false,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = false,
                .batched_multi_token_decode = false,
                .batched_multi_token_decode_max_width = 0u,
                .prefix_reuse = true,
            },
        .persistence = std::nullopt,
    };
  }

  [[nodiscard]] server::TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      const auto deferred = model_->DeferredScratchBytes();
      capacity = free_bytes > deferred ? free_bytes - deferred : 0;
    }
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = model_->SessionBytes(max_context_),
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = 0,
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<server::TextExecutionPlan> SupportedPlans()
      const override {
    return {
        {.kind = server::TextExecutionPlanKind::kSerial, .physical_width = 1}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    const auto prompt = RenderChat(
        request.messages, true,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools});
    if (!prompt.has_value()) {
      return std::nullopt;
    }
    return model_->tokenizer().Encode(*prompt);
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->Decode(EngineTokens(tokens));
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<NemotronTextRunnerState>(model_, max_context_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& runner = RequireNemotronState(state);
    if (runner.position() != prefix.size()) {
      throw std::logic_error(
          "Nemotron Bolt reused prefix does not match checkpoint");
    }
  }

  [[nodiscard]] server::TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& runner = RequireNemotronState(state);
    if (offset != runner.position()) {
      throw std::logic_error(
          "Nemotron Bolt prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("Nemotron Bolt prefill has no remaining input");
    }
    const std::size_t consumed =
        std::min<std::size_t>(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    const auto prefix = EngineTokens(prompt.first(next_position));
    std::string error;
    if (!runner.session().Sync(prefix, &error)) {
      runner.set_position(0);
      throw std::runtime_error("Nemotron Bolt prefill failed: " + error);
    }
    runner.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] server::TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& runner = RequireNemotronState(state);
    if (runner.position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = runner.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error("Nemotron Bolt token selection has no logits");
    }
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    if (model_->IsStopToken(token)) {
      return {.stop = true, .token = 0, .piece = {}};
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->TokenText(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Nemotron Bolt token ID exceeds engine range");
    }
    auto& runner = RequireNemotronState(state);
    std::string error;
    if (!runner.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
      throw std::runtime_error("Nemotron Bolt decode failed: " + error);
    }
    runner.set_position(runner.position() + 1);
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireNemotronState(state).position();
  }

private:
  std::shared_ptr<Model> model_;
  std::uint32_t max_context_;
};

}  // namespace

std::shared_ptr<Model> LoadServeModel(
    const std::string& model_path, const core::GgufReader& reader,
    std::uint32_t max_context,
    const server::TextSpeculativeConfig& speculative_config,
    std::string* error) {
  if (speculative_config.backend != server::TextSpeculativeBackend::kDisabled) {
    SetError(error,
             "Nemotron Bolt HTTP models do not support speculative "
             "decoding");
    return nullptr;
  }
  std::string load_error;
  auto model = Model::Load(model_path, ModelOptions{.max_context = max_context},
                           &load_error);
  if (model == nullptr) {
    SetError(error, "Failed to create Nemotron Bolt model: " + load_error);
    return nullptr;
  }
  return model;
}

std::shared_ptr<TextModelRunner> CreateTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    std::size_t session_count,
    const server::TextSpeculativeConfig& speculative_config,
    const server::TextDiskCacheConfig& disk_cache_config, std::string* error) {
  if (session_count != 1) {
    SetError(error,
             "Nemotron Bolt serves one session: its KV and recurrent state "
             "live in the single shared device executor (--sessions 1)");
    return nullptr;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(error,
             "HTTP context exceeds the loaded Nemotron Bolt model context");
    return nullptr;
  }
  if (speculative_config.backend != server::TextSpeculativeBackend::kDisabled) {
    SetError(error,
             "Nemotron Bolt HTTP models do not support speculative "
             "decoding");
    return nullptr;
  }
  if (!disk_cache_config.directory.empty()) {
    SetError(error, "Nemotron Bolt HTTP models do not support the disk cache");
    return nullptr;
  }
  return std::make_shared<NemotronTextRunner>(std::move(model), max_context);
}

}  // namespace gufo::models::nemotron_h_moe

#endif  // defined(ENGINE_ENABLE_HIP)
