#include "src/models/nemotron_h_moe/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <utility>

#include "src/core/gguf_reader.hpp"
#include "src/models/nemotron_h_moe/kernels/rocm/device_model.hpp"
#include "src/models/nemotron_h_moe/weights.hpp"

namespace gufo::models::nemotron_h_moe {
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
  if (options.max_context > c.context_length) {
    AssignError(error_msg, "context exceeds the model's " +
                               std::to_string(c.context_length) + " tokens");
    return nullptr;
  }
  m->tokenizer_ = NemotronTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  m->device_ = std::make_unique<DeviceModel>(*m->weights_, options.max_context);
  if (!m->device_->Init(error_msg)) {
    m->device_.reset();
    return nullptr;
  }
  m->resident_bytes_ = m->device_->WeightBytes();
  m->scratch_bytes_ = m->device_->ScratchBytes();
  return m;
}

std::shared_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  if (!live_session_.expired()) {
    AssignError(error_msg,
                "the Nemotron Bolt device state is shared by all sessions");
    return nullptr;
  }
  auto session =
      std::shared_ptr<Session>(new Session(shared_from_this(), max_context));
  live_session_ = session;
  return session;
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  const std::vector<TokenId> ids = tokenizer_->Encode(text);
  return std::vector<std::int32_t>(ids.begin(), ids.end());
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<TokenId> ids;
  ids.reserve(tokens.size());
  for (const std::int32_t t : tokens) {
    ids.push_back(static_cast<TokenId>(t));
  }
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return tokenizer_->IsStopToken(static_cast<TokenId>(token));
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::string Model::ModelName() const {
  return std::string(reader_->GetMetadataString("general.name")
                         .value_or("NVIDIA Nemotron 3.5 Bolt"));
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return resident_bytes_;
}

std::size_t Model::SessionBytes(std::uint32_t context) const noexcept {
  return device_->StateBytes(context);
}

std::size_t Model::DeferredScratchBytes() const noexcept {
  return scratch_bytes_;
}

Session::Session(std::shared_ptr<Model> model, std::uint32_t max_context)
    : model_(std::move(model)),
      device_(model_->device_.get()),
      max_context_(max_context) {}

Session::~Session() = default;

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  logits_.assign(model_->VocabSize(), 0.0f);
  if (prompt.empty()) {
    device_->Reset();
    tokens_.clear();
    valid_ = true;
    return true;
  }
  const auto common = std::mismatch(tokens_.begin(), tokens_.end(),
                                    prompt.begin(), prompt.end());
  const std::span<const std::int32_t> tail(
      prompt.begin() +
          static_cast<std::ptrdiff_t>(common.first - tokens_.begin()),
      prompt.end());
  if (common.first != tokens_.end()) {
    device_->Reset();
    tokens_.clear();
    if (!device_->Process(prompt, logits_, error_msg)) {
      valid_ = false;
      return false;
    }
    tokens_.assign(prompt.begin(), prompt.end());
    return true;
  }
  if (!tail.empty()) {
    if (!device_->Process(tail, logits_, error_msg)) {
      valid_ = false;
      return false;
    }
    tokens_.insert(tokens_.end(), tail.begin(), tail.end());
  }
  return true;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  logits_.assign(model_->VocabSize(), 0.0f);
  if (!device_->Step(token, logits_, error_msg)) {
    valid_ = false;
    return false;
  }
  tokens_.push_back(token);
  return true;
}

std::uint32_t Session::Position() const noexcept {
  return device_->Position();
}

std::uint32_t Session::ContextSize() const noexcept {
  return static_cast<std::uint32_t>(tokens_.size());
}

void Session::Reset() {
  device_->Reset();
  tokens_.clear();
  logits_.clear();
  valid_ = true;
}

}  // namespace gufo::models::nemotron_h_moe
