#include "src/models/qwen36_a3b/serve_runner.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <algorithm>
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
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/vision/encoder.hpp"
#include "src/models/qwen/vision/prompt.hpp"

namespace gufo::models::qwen36_a3b {
namespace {

using server::ChatRequest;
using server::TextModelRunner;
using server::TextPreparedPrompt;
using server::TextPromptContext;
using server::TextRunnerState;
using server::TextRunnerToken;

using Qwen36A3BModel = Model;
using Qwen36A3BSession = Session;

constexpr std::string_view kQwen36A3BStateAbi = "qwen36-a3b-rocm-session-v1";

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

tokenization::ChatTemplateOptions QwenChatOptions(const ChatRequest& request,
                                                  std::uint32_t max_context) {
  auto options = tokenization::ResolveQwenChatOptions(request.reasoning,
                                                      request.add_vision_id);
  options.require_tool_call =
      request.tool_choice == ChatRequest::ToolChoice::kRequired;
  // gufo #285: the rendered prompt may be as long as the context can hold.
  options.max_output_bytes =
      tokenization::RenderedPromptBoundBytes(max_context);
  return options;
}

struct QwenImageContext final : TextPromptContext {
  std::shared_ptr<const models::qwen::vision::Prompt> prompt;
};

TextPreparedPrompt PrepareQwenPrompt(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context) {
  const bool has_images = std::ranges::any_of(
      request.messages, [](const auto& m) { return !m.images.empty(); });
  const auto options = QwenChatOptions(request, max_context);
  auto prompt = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          tokenizer, request.messages,
          request.tool_choice == ChatRequest::ToolChoice::kNone
              ? std::span<const tokenization::ChatTool>{}
              : std::span<const tokenization::ChatTool>{request.tools},
          options,
          encoder && has_images ? encoder->identity() : std::string_view{},
          max_context));
  const auto cache_prefix = prompt->stable_prefix_tokens;
  if (prompt->images.empty())
    return {std::move(prompt->tokens), {}, cache_prefix};
  if (!encoder)
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  auto context = std::make_shared<QwenImageContext>();
  context->cache_identity = prompt->cache_identity;
  for (const auto& image : prompt->images) {
    const auto identity = prompt->IdentityForPrefix(image.grid.offset);
    context->cache_prefixes.push_back(
        {image.grid.offset, {identity.begin(), identity.end()}});
  }
  context->prompt = prompt;
  return {prompt->tokens, std::move(context), cache_prefix};
}

std::shared_ptr<const models::qwen::vision::Prompt> QwenPrompt(
    const std::shared_ptr<const TextPromptContext>& context) {
  if (!context)
    return {};
  const auto* image = dynamic_cast<const QwenImageContext*>(context.get());
  if (!image)
    throw std::invalid_argument("invalid Qwen prompt context");
  return image->prompt;
}

std::vector<std::int32_t> Qwen36A3BEngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Qwen3.6-35B-A3B token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

class Qwen36A3BTextRunnerState final : public TextRunnerState {
public:
  Qwen36A3BTextRunnerState(const std::shared_ptr<Qwen36A3BModel>& model,
                           std::uint32_t max_context, std::uint32_t min_drafts,
                           std::uint32_t max_drafts) {
    std::string error;
    session_ = model->CreateSession(max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create Qwen3.6-35B-A3B session: " +
                               error);
    }
    // Serve decodes with MTP speculative decoding, so the draft cache must be
    // filled during prefill. Enable it before the first Sync.
    session_->SetMtpEnabled(true);
    session_->SetDraftLimits(min_drafts, max_drafts);
  }

  void Invalidate() noexcept override {
    session_->Reset();
    position_ = 0;
  }

  [[nodiscard]] Qwen36A3BSession& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  std::unique_ptr<Qwen36A3BSession> session_;
  std::size_t position_{0};
};

Qwen36A3BTextRunnerState& RequireQwen36A3BState(TextRunnerState& state) {
  auto* q36 = dynamic_cast<Qwen36A3BTextRunnerState*>(&state);
  if (q36 == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.6-35B-A3B");
  }
  return *q36;
}

const Qwen36A3BTextRunnerState& RequireQwen36A3BState(
    const TextRunnerState& state) {
  const auto* q36 = dynamic_cast<const Qwen36A3BTextRunnerState*>(&state);
  if (q36 == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.6-35B-A3B");
  }
  return *q36;
}

/// Serial runner for Qwen3.6-35B-A3B. The model owns one shared executor
/// (weights and scratch) plus one private KV/recurrent/snapshot state per
/// session; the pool creates up to the configured session count, each with a
/// divided context window.
class Qwen36A3BTextRunner final : public TextModelRunner {
public:
  Qwen36A3BTextRunner(std::shared_ptr<Qwen36A3BModel> model,
                      std::uint32_t max_context, bool use_mtp,
                      std::uint32_t min_drafts, std::uint32_t max_drafts)
      : model_(std::move(model)),
        max_context_(max_context),
        use_mtp_(use_mtp),
        min_drafts_(min_drafts),
        max_drafts_(max_drafts) {}

  [[nodiscard]] server::TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kQwen36A3BStateAbi),
        .max_context = max_context_,
        .capabilities =
            server::TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = false,
                .fork = false,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = use_mtp_,
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
    // Each session owns its KV/recurrent/snapshot state; the executor scratch
    // is shared and already allocated at model load, so reserve it once.
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
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->tokenizer(), request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        QwenChatOptions(request, max_context_));
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    return PrepareQwenPrompt(request, model_->tokenizer(),
                             model_->VisionEncoder(), max_context_);
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    RequireQwen36A3BState(state).session().ConfigureVision(QwenPrompt(context));
  }

  [[nodiscard]] server::TextGenerationBackend::InitialOutputState
  InitialOutputState(const ChatRequest& request) const override {
    return QwenChatOptions(request, max_context_).enable_thinking
               ? server::TextGenerationBackend::InitialOutputState::kReasoning
               : server::TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<Qwen36A3BTextRunnerState>(model_, max_context_,
                                                      min_drafts_, max_drafts_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& q36 = RequireQwen36A3BState(state);
    if (q36.position() != prefix.size()) {
      throw std::logic_error(
          "Qwen3.6-35B-A3B reused prefix does not match checkpoint");
    }
  }

  [[nodiscard]] server::TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& q36 = RequireQwen36A3BState(state);
    if (offset != q36.position()) {
      throw std::logic_error(
          "Qwen3.6-35B-A3B prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("Qwen3.6-35B-A3B prefill has no remaining input");
    }
    const std::size_t consumed =
        std::min<std::size_t>(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    const auto prefix = Qwen36A3BEngineTokens(prompt.first(next_position));
    std::string error;
    if (!q36.session().Sync(prefix, &error)) {
      q36.set_position(0);
      throw std::runtime_error("Qwen3.6-35B-A3B prefill failed: " + error);
    }
    q36.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] server::TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& q36 = RequireQwen36A3BState(state);
    if (q36.position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = q36.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error("Qwen3.6-35B-A3B token selection has no logits");
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
          "Qwen3.6-35B-A3B token ID exceeds engine range");
    }
    auto& q36 = RequireQwen36A3BState(state);
    std::string error;
    if (!q36.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
      throw std::runtime_error("Qwen3.6-35B-A3B decode failed: " + error);
    }
    q36.set_position(q36.position() + 1);
  }

  [[nodiscard]] std::optional<server::TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] server::TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_mtp_ || max_tokens == 1) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Qwen3.6-35B-A3B MTP decode budget must be at least one token");
    }
    auto& q36 = RequireQwen36A3BState(state);
    if (q36.position() >= max_context_) {
      return {.selections = {}, .stop = true};
    }
    Qwen36A3BSession::DecodeResult decoded;
    std::string error;
    if (!q36.session().DecodeStep(max_tokens, sampler, &decoded, &error)) {
      throw std::runtime_error("Qwen3.6-35B-A3B MTP decode failed: " + error);
    }
    server::TextDecodeStep step;
    step.stop = decoded.stop;
    step.selections.reserve(decoded.tokens.size());
    for (const std::int32_t token : decoded.tokens) {
      step.selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->TokenText(token),
      });
    }
    q36.set_position(q36.session().Position());
    step.draft_tokens = decoded.drafted;
    step.draft_accepted_tokens = decoded.accepted;
    return step;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireQwen36A3BState(state).position();
  }

private:
  std::shared_ptr<Qwen36A3BModel> model_;
  std::uint32_t max_context_;
  bool use_mtp_;
  std::uint32_t min_drafts_;
  std::uint32_t max_drafts_;
};

}  // namespace

std::shared_ptr<Model> LoadServeModel(
    const std::string& model_path, const core::GgufReader& reader,
    std::uint32_t max_context, std::uint32_t attn_window,
    std::uint32_t attn_sink, const std::string& vision_model_path,
    const server::TextSpeculativeConfig& speculative_config,
    std::string* error) {
  if (speculative_config.backend != server::TextSpeculativeBackend::kDisabled &&
      speculative_config.backend != server::TextSpeculativeBackend::kMtp) {
    SetError(error,
             "Qwen3.6-35B-A3B HTTP models support only MTP speculative "
             "decoding (--speculative mtp)");
    return nullptr;
  }
  std::string template_error;
  if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(reader,
                                                            &template_error)) {
    SetError(error,
             "Unsupported Qwen3.6-35B-A3B chat template: " + template_error);
    return nullptr;
  }
  std::string load_error;
  auto model = Model::Load(model_path,
                           ModelOptions{.max_context = max_context,
                                        .attn_window = attn_window,
                                        .attn_sink = attn_sink,
                                        .vision_model_path = vision_model_path},
                           &load_error);
  if (model == nullptr) {
    SetError(error, "Failed to create Qwen3.6-35B-A3B model: " + load_error);
    return nullptr;
  }
  return model;
}

std::shared_ptr<TextModelRunner> CreateTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    std::size_t session_count,
    const server::TextSpeculativeConfig& speculative_config,
    const server::TextDiskCacheConfig& disk_cache_config, std::string* error) {
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return nullptr;
  }
  constexpr std::size_t kMaxQwen36Sessions = 16;
  if (session_count > kMaxQwen36Sessions) {
    SetError(error,
             "Qwen3.6-35B-A3B supports at most 16 concurrent HTTP sessions "
             "(--sessions 16)");
    return nullptr;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(error,
             "HTTP context exceeds the loaded Qwen3.6-35B-A3B model context");
    return nullptr;
  }
  // Divide the configured context across the concurrent sessions so their
  // combined KV/recurrent/snapshot state fits the device budget; a single
  // session keeps the full window.
  const auto per_session_context = static_cast<std::uint32_t>(
      std::max<std::size_t>(1, max_context / session_count));
  if (speculative_config.backend != server::TextSpeculativeBackend::kDisabled &&
      speculative_config.backend != server::TextSpeculativeBackend::kMtp) {
    SetError(error,
             "Qwen3.6-35B-A3B HTTP models support only MTP speculative "
             "decoding (--speculative mtp)");
    return nullptr;
  }
  const bool use_mtp =
      speculative_config.backend == server::TextSpeculativeBackend::kMtp;
  if (use_mtp && !model->HasMtp()) {
    SetError(error,
             "Qwen3.6-35B-A3B MTP speculative decoding requires a GGUF with "
             "an MTP block");
    return nullptr;
  }
  if (!disk_cache_config.directory.empty()) {
    SetError(error,
             "Qwen3.6-35B-A3B HTTP models do not support the disk cache");
    return nullptr;
  }
  return std::make_shared<Qwen36A3BTextRunner>(
      std::move(model), per_session_context, use_mtp,
      speculative_config.min_draft_tokens, speculative_config.max_draft_tokens);
}

}  // namespace gufo::models::qwen36_a3b

#endif  // defined(ENGINE_ENABLE_HIP)