#include "src/models/gemma4/serve_runner.hpp"

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
#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include "src/core/sampling.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/vision/prompt.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::models::gemma4 {
namespace {

using server::ChatRequest;
using server::TextDecodeSelection;
using server::TextDecodeStep;
using server::TextExecutionPlan;
using server::TextExecutionPlanKind;
using server::TextGenerationBackend;
using server::TextModelRunner;
using server::TextPrefillStep;
using server::TextPreparedPrompt;
using server::TextPromptContext;
using server::TextRunnerCapabilities;
using server::TextRunnerDescriptor;
using server::TextRunnerResourceClaim;
using server::TextRunnerState;
using server::TextRunnerToken;

using Gemma4Model = Model;
using Gemma4Session = Session;

constexpr std::string_view kGemma4StateAbi = "gemma4-rocm-session-v1";

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::string_view ChatRoleName(tokenization::ChatRole role) {
  switch (role) {
    case tokenization::ChatRole::kSystem:
      return "system";
    case tokenization::ChatRole::kDeveloper:
      return "developer";
    case tokenization::ChatRole::kAssistant:
      return "assistant";
    case tokenization::ChatRole::kTool:
      return "tool";
    case tokenization::ChatRole::kUser:
      return "user";
  }
  return "user";
}

std::vector<TextRunnerToken> Gemma4RunnerTokens(
    std::span<const std::int32_t> tokens) {
  std::vector<TextRunnerToken> converted;
  converted.reserve(tokens.size());
  for (const std::int32_t token : tokens) {
    if (token < 0) {
      throw std::invalid_argument("Gemma-4 token ID must not be negative");
    }
    converted.push_back(static_cast<TextRunnerToken>(token));
  }
  return converted;
}

std::vector<std::int32_t> Gemma4EngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("Gemma-4 token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

struct Gemma4ImageContext final : TextPromptContext {
  std::vector<vision::VisionSlot> slots;
};

class Gemma4TextRunnerState final : public TextRunnerState {
public:
  Gemma4TextRunnerState(const std::shared_ptr<Gemma4Model>& model,
                        std::uint32_t max_context, bool use_mtp,
                        std::uint32_t min_drafts, std::uint32_t max_drafts) {
    std::string error;
    session_ = model->CreateSession(max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create Gemma-4 session: " + error);
    }
    // Serve decodes with MTP speculative decoding, so the draft cache must be
    // filled during prefill. Enable it before the first Sync.
    if (use_mtp) {
      session_->SetMtpEnabled(true);
      session_->SetDraftLimits(min_drafts, max_drafts);
    }
  }

  void Invalidate() noexcept override {
    session_->Reset();
    vision_context_.reset();
    position_ = 0;
  }

  [[nodiscard]] Gemma4Session& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }
  void set_vision_context(
      std::shared_ptr<const Gemma4ImageContext> context) noexcept {
    vision_context_ = std::move(context);
  }
  [[nodiscard]] const std::shared_ptr<const Gemma4ImageContext>&
  vision_context() const noexcept {
    return vision_context_;
  }

private:
  std::unique_ptr<Gemma4Session> session_;
  std::shared_ptr<const Gemma4ImageContext> vision_context_;
  std::size_t position_{0};
};

Gemma4TextRunnerState& RequireGemma4State(TextRunnerState& state) {
  auto* gemma = dynamic_cast<Gemma4TextRunnerState*>(&state);
  if (gemma == nullptr) {
    throw std::logic_error("text runner state is not Gemma-4");
  }
  return *gemma;
}

const Gemma4TextRunnerState& RequireGemma4State(const TextRunnerState& state) {
  const auto* gemma = dynamic_cast<const Gemma4TextRunnerState*>(&state);
  if (gemma == nullptr) {
    throw std::logic_error("text runner state is not Gemma-4");
  }
  return *gemma;
}

/// Serial runner for Gemma-4-26B-A4B. The model owns one shared executor
/// (weights and scratch) plus one private context state per session; the pool
/// creates up to the configured session count, each with a divided context
/// window. Decode is single-token unless MTP speculative decoding is enabled,
/// in which case each step drafts and verifies a block in one pass.
class Gemma4TextRunner final : public TextModelRunner {
public:
  Gemma4TextRunner(std::shared_ptr<Gemma4Model> model,
                   std::uint32_t max_context, bool use_mtp,
                   std::uint32_t min_drafts, std::uint32_t max_drafts)
      : model_(std::move(model)),
        max_context_(max_context),
        use_mtp_(use_mtp),
        min_drafts_(min_drafts),
        max_drafts_(max_drafts) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kGemma4StateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
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

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
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

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {{.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return Gemma4RunnerTokens(model_->Tokenize(text));
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    const auto inputs = ConvertInputs(request, nullptr);
    auto tokens = Gemma4RunnerTokens(model_->EncodeChat(
        inputs.messages, inputs.tools,
        ChatTemplateOptions{
            .add_generation_prompt = true,
            .enable_thinking = request.reasoning.enabled.value_or(false),
            .preserve_thinking =
                request.reasoning.preserve_thinking.value_or(false),
        }));
    if (tokens.empty()) {
      return std::nullopt;
    }
    return tokens;
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    const bool has_images = std::ranges::any_of(
        request.messages, [](const auto& m) { return !m.images.empty(); });
    if (!has_images) {
      auto tokens = RenderAndTokenize(request);
      if (!tokens) {
        return std::nullopt;
      }
      return TextPreparedPrompt{std::move(*tokens), {}};
    }
    if (!model_->HasVision()) {
      throw std::invalid_argument(
          "image input requires a matching --mmproj sidecar");
    }
    std::vector<core::Image> images;
    const auto inputs = ConvertInputs(request, &images);
    std::string error;
    auto prepared = model_->EncodeChatVision(
        inputs.messages, inputs.tools,
        ChatTemplateOptions{
            .add_generation_prompt = true,
            .enable_thinking = request.reasoning.enabled.value_or(false),
            .preserve_thinking =
                request.reasoning.preserve_thinking.value_or(false),
        },
        images, &error);
    if (prepared.tokens.empty()) {
      throw std::invalid_argument("Gemma-4 image prompt failed: " + error);
    }
    auto context = std::make_shared<Gemma4ImageContext>();
    context->slots = std::move(prepared.images);
    context->cache_identity = ImageCacheIdentity(request);
    return TextPreparedPrompt{Gemma4RunnerTokens(prepared.tokens),
                              std::move(context), 0};
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    auto& gemma = RequireGemma4State(state);
    if (context == nullptr) {
      gemma.set_vision_context(nullptr);
      return;
    }
    if (dynamic_cast<const Gemma4ImageContext*>(context.get()) == nullptr) {
      throw std::invalid_argument("invalid Gemma-4 prompt context");
    }
    gemma.set_vision_context(
        std::static_pointer_cast<const Gemma4ImageContext>(std::move(context)));
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return request.reasoning.enabled.value_or(false)
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->Decode(Gemma4EngineTokens(tokens));
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<Gemma4TextRunnerState>(
        model_, max_context_, use_mtp_, min_drafts_, max_drafts_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& gemma = RequireGemma4State(state);
    if (gemma.position() != prefix.size()) {
      throw std::logic_error("Gemma-4 reused prefix does not match checkpoint");
    }
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& gemma = RequireGemma4State(state);
    if (offset != gemma.position()) {
      throw std::logic_error(
          "Gemma-4 prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("Gemma-4 prefill has no remaining input");
    }
    std::string error;
    if (const auto vision = gemma.vision_context(); vision != nullptr) {
      if (offset != 0) {
        throw std::logic_error(
            "Gemma-4 image prefill cannot resume mid-prompt");
      }
      // Image rows cannot be reused from a text prefix, so the whole prompt is
      // reprocessed with the staged embeddings spliced over the token rows; the
      // executor chunks internally, so one call bounds peak memory.
      if (!gemma.session().Sync(Gemma4EngineTokens(prompt), vision->slots,
                                &error)) {
        gemma.set_position(0);
        throw std::runtime_error("Gemma-4 image prefill failed: " + error);
      }
      gemma.set_position(prompt.size());
      return {
          .consumed_tokens = prompt.size(),
          .decode_ready = true,
      };
    }
    const std::size_t consumed =
        std::min<std::size_t>(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    const auto prefix = Gemma4EngineTokens(prompt.first(next_position));
    if (!gemma.session().Sync(prefix, &error)) {
      gemma.set_position(0);
      throw std::runtime_error("Gemma-4 prefill failed: " + error);
    }
    gemma.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& gemma = RequireGemma4State(state);
    if (gemma.position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = gemma.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error("Gemma-4 token selection has no logits");
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
      throw std::invalid_argument("Gemma-4 token ID exceeds engine range");
    }
    auto& gemma = RequireGemma4State(state);
    std::string error;
    if (!gemma.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
      throw std::runtime_error("Gemma-4 decode failed: " + error);
    }
    gemma.set_position(gemma.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_mtp_ || max_tokens == 1) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Gemma-4 MTP decode budget must be at least one token");
    }
    auto& gemma = RequireGemma4State(state);
    if (gemma.position() >= max_context_) {
      return {.selections = {}, .stop = true};
    }
    Gemma4Session::DecodeResult decoded;
    std::string error;
    if (!gemma.session().DecodeStep(max_tokens, sampler, &decoded, &error)) {
      throw std::runtime_error("Gemma-4 MTP decode failed: " + error);
    }
    TextDecodeStep step;
    step.stop = decoded.stop;
    step.selections.reserve(decoded.tokens.size());
    for (const std::int32_t token : decoded.tokens) {
      step.selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->TokenText(token),
      });
    }
    gemma.set_position(gemma.session().Position());
    step.draft_tokens = decoded.drafted;
    step.draft_accepted_tokens = decoded.accepted;
    return step;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireGemma4State(state).position();
  }

private:
  struct Inputs {
    std::vector<ChatMessage> messages;
    std::vector<ChatTool> tools;
  };

  /// Converts the serve request into Gemma-4 chat messages and tools. When
  /// `images_out` is non-null, each message's image parts are decoded (in
  /// order) and appended, and `image_count` is set so the template emits the
  /// matching `<|image|>` markers.
  [[nodiscard]] Inputs ConvertInputs(
      const ChatRequest& request, std::vector<core::Image>* images_out) const {
    Inputs inputs;
    inputs.messages.reserve(request.messages.size());
    for (const auto& message : request.messages) {
      ChatMessage converted{
          .role = std::string(ChatRoleName(message.role)),
          .content = message.content,
          .reasoning_content = message.thought,
          .tool_calls = {},
          .tool_call_id = message.tool_call_id,
      };
      converted.tool_calls.reserve(message.tool_calls.size());
      for (const auto& call : message.tool_calls) {
        ChatMessage::ToolCall converted_call{
            .name = call.name,
            .arguments = {},
            .id = call.id,
        };
        converted_call.arguments.reserve(call.arguments.size());
        for (const auto& argument : call.arguments) {
          converted_call.arguments.push_back({
              .name = argument.name,
              .value = argument.value,
              .is_string = argument.is_string,
          });
        }
        converted.tool_calls.push_back(std::move(converted_call));
      }
      converted.image_count = message.images.size();
      if (images_out != nullptr) {
        for (const auto& part : message.images) {
          images_out->push_back(
              core::DecodeImage(std::span<const std::uint8_t>(*part.bytes)));
        }
      }
      inputs.messages.push_back(std::move(converted));
    }
    if (request.tool_choice != ChatRequest::ToolChoice::kNone) {
      inputs.tools.reserve(request.tools.size());
      for (const auto& tool : request.tools) {
        inputs.tools.push_back({
            .name = tool.name,
            .description = tool.description,
            .parameters_json = tool.parameters_json,
            .definition_json =
                tool.definition_json.empty()
                    ? std::string{}
                    : json::parse(tool.definition_json)["function"].dump(),
        });
      }
    }
    return inputs;
  }

  /// FNV-1a over every image's raw bytes so distinct image sets never share a
  /// cached KV prefix; the count is folded in first to separate ordering.
  [[nodiscard]] static std::vector<std::uint8_t> ImageCacheIdentity(
      const ChatRequest& request) {
    std::uint64_t hash = 1469598103934665603ULL;
    auto mix = [&hash](std::uint64_t value) {
      for (int byte = 0; byte < 8; ++byte) {
        hash = (hash ^ ((value >> (byte * 8)) & 0xFFULL)) * 1099511628211ULL;
      }
    };
    std::size_t count = 0;
    for (const auto& message : request.messages) {
      count += message.images.size();
    }
    mix(count);
    for (const auto& message : request.messages) {
      for (const auto& part : message.images) {
        if (part.bytes == nullptr) {
          mix(0);
          continue;
        }
        for (const std::uint8_t byte : *part.bytes) {
          hash = (hash ^ byte) * 1099511628211ULL;
        }
        mix(part.bytes->size());
      }
    }
    std::vector<std::uint8_t> identity;
    identity.reserve(8);
    for (int byte = 0; byte < 8; ++byte) {
      identity.push_back(
          static_cast<std::uint8_t>((hash >> (byte * 8)) & 0xFF));
    }
    return identity;
  }

  std::shared_ptr<Gemma4Model> model_;
  std::uint32_t max_context_;
  bool use_mtp_;
  std::uint32_t min_drafts_;
  std::uint32_t max_drafts_;
};

}  // namespace

std::shared_ptr<TextModelRunner> CreateTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    std::size_t session_count,
    const server::TextSpeculativeConfig& speculative_config,
    const server::TextDiskCacheConfig& disk_cache_config, std::string* error) {
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded Gemma-4 model context");
    return nullptr;
  }
  // Compute is serialized on the shared executor; each session owns a private
  // KV cache, so the configured context is divided across the concurrent
  // sessions to bound their combined KV footprint. A single session keeps the
  // full window.
  const auto per_session_context = static_cast<std::uint32_t>(
      std::max<std::size_t>(1, max_context / session_count));
  if (speculative_config.backend != server::TextSpeculativeBackend::kDisabled &&
      speculative_config.backend != server::TextSpeculativeBackend::kMtp) {
    SetError(error,
             "Gemma-4 HTTP models support only MTP speculative decoding "
             "(--speculative mtp --mtp-model)");
    return nullptr;
  }
  const bool use_mtp =
      speculative_config.backend == server::TextSpeculativeBackend::kMtp;
  if (use_mtp && !model->HasMtp()) {
    SetError(error,
             "Gemma-4 MTP speculative decoding requires an --mtp-model draft");
    return nullptr;
  }
  if (!disk_cache_config.directory.empty()) {
    SetError(error, "Gemma-4 HTTP models do not support the disk cache");
    return nullptr;
  }
  return std::make_shared<Gemma4TextRunner>(
      std::move(model), per_session_context, use_mtp,
      speculative_config.min_draft_tokens, speculative_config.max_draft_tokens);
}

}  // namespace gufo::models::gemma4

#endif  // defined(ENGINE_ENABLE_HIP)