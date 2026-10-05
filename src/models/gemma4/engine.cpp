#include "src/models/gemma4/engine.hpp"

#include <hip/hip_runtime.h>

#include <utility>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/kernels/rocm/device_model.hpp"
#include "src/models/gemma4/kernels/rocm/executor.hpp"
#include "src/models/gemma4/tokenizer.hpp"
#include "src/models/gemma4/weights.hpp"

namespace gufo::models::gemma4 {
namespace {

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

}  // namespace

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  if (options.max_context == 0) {
    AssignError(error_msg, "max_context must be positive");
    return nullptr;
  }
  if (options.prefill_chunk == 0) {
    AssignError(error_msg, "prefill_chunk must be positive");
    return nullptr;
  }
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(model_path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  auto weights = ModelWeights::Bind(*m->reader_, error_msg);
  if (!weights) {
    return nullptr;
  }
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  const Config& c = m->weights_->config;
  if (c.is_draft) {
    AssignError(error_msg, "this artifact is a draft, not the trunk");
    return nullptr;
  }
  if (options.max_context > c.context_length) {
    AssignError(error_msg, "context exceeds the model's " +
                               std::to_string(c.context_length) + " tokens");
    return nullptr;
  }
  m->tokenizer_ = Tokenizer::FromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, *m->reader_, error_msg);
  if (!m->device_) {
    return nullptr;
  }
  m->executor_ =
      rocm::Executor::Create(*m->device_, options.prefill_chunk, error_msg);
  if (!m->executor_) {
    return nullptr;
  }
  return m;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  auto native = executor_->CreateSession(max_context, error_msg);
  if (!native) {
    return nullptr;
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(native)));
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  return tokenizer_->Encode(text);
}

std::vector<std::int32_t> Model::EncodeChat(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options) const {
  const std::string rendered = RenderChat(messages, tools, options);
  return tokenizer_->Encode(rendered, /*add_special=*/false,
                            /*parse_special=*/true);
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  return tokenizer_->Decode(tokens);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->TokenToPiece(token);
}

std::int32_t Model::EosToken() const noexcept {
  return tokenizer_->EosId();
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return token == tokenizer_->EosId();
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::string Model::ModelName() const {
  return std::string(
      reader_->GetMetadataString("general.name").value_or("Gemma-4-26B-A4B"));
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes();
}

std::size_t Model::SessionBytes(std::uint32_t context) const noexcept {
  return executor_->SessionBytes(context);
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)),
      session_(std::move(session)),
      executor_(model_->executor_.get()) {
  logits_.resize(model_->VocabSize());
}

Session::~Session() = default;

std::uint32_t Session::Position() const noexcept {
  return session_->position();
}

std::uint32_t Session::ContextSize() const noexcept {
  return session_->max_context();
}

void Session::Reset() {
  valid_ = false;
  session_->Reset();
  tokens_.clear();
  valid_ = true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.empty()) {
    AssignError(error_msg, "prompt is empty");
    return false;
  }
  if (prompt.size() > ContextSize()) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  if (!valid_) {
    Reset();
  }
  // The KV cache is position-major and rewinds by rewriting rows, so a pure
  // extension only feeds the new tail; any divergence restarts the session.
  std::size_t common = 0;
  while (common < tokens_.size() && common < prompt.size() &&
         tokens_[common] == prompt[common]) {
    ++common;
  }
  if (common == prompt.size() && common == tokens_.size()) {
    return true;
  }
  if (common != tokens_.size()) {
    Reset();
    common = 0;
  }
  valid_ = false;
  // Feed the new tail through the batched prefill path: one GEMM-driven pass
  // over the whole span instead of a per-token GEMV step. The executor's
  // position already equals the common prefix length, so Prefill continues
  // from there and leaves the final token's logits.
  if (!executor_->Prefill(*session_, prompt.data() + common,
                          static_cast<std::uint32_t>(prompt.size() - common),
                          error_msg)) {
    return false;
  }
  for (std::size_t i = common; i < prompt.size(); ++i) {
    tokens_.push_back(prompt[i]);
  }
  (void)hipMemcpy(logits_.data(), executor_->logits(),
                  logits_.size() * sizeof(float), hipMemcpyDeviceToHost);
  valid_ = true;
  return true;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  if (tokens_.size() >= ContextSize()) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  valid_ = false;
  if (!executor_->Step(*session_, token, error_msg)) {
    return false;
  }
  tokens_.push_back(token);
  (void)hipMemcpy(logits_.data(), executor_->logits(),
                  logits_.size() * sizeof(float), hipMemcpyDeviceToHost);
  valid_ = true;
  return true;
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos) {
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  if (result == nullptr || max_tokens == 0 || tokens_.empty()) {
    AssignError(
        error_msg,
        "decode needs an output, a positive budget and a synced prompt");
    return false;
  }
  *result = {};
  const auto is_stop = [&](std::int32_t token) {
    return stop_at_eos && model_->IsStopToken(token);
  };
  // Non-speculative single-token decode. MTP speculation arrives with M8.
  const auto token = static_cast<std::int32_t>(sampler.Sample(logits_));
  if (is_stop(token)) {
    result->stop = true;
    return true;
  }
  valid_ = false;
  if (!executor_->Step(*session_, token, error_msg)) {
    return false;
  }
  tokens_.push_back(token);
  (void)hipMemcpy(logits_.data(), executor_->logits(),
                  logits_.size() * sizeof(float), hipMemcpyDeviceToHost);
  sampler.Accept(static_cast<sampling::TokenId>(token));
  result->tokens.push_back(token);
  valid_ = true;
  return true;
}

}  // namespace gufo::models::gemma4