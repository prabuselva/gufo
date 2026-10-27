#include "src/models/flint/engine.hpp"

#include <algorithm>
#include <utility>

#include "src/core/gguf_reader.hpp"
#include "src/models/flint/kernels/rocm/executor.hpp"
#include "src/models/flint/weights.hpp"

namespace gufo::models::flint {
namespace {

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

/// Prefill chunk fed to FlintExecutor::Forward in one launch. The chunk is
/// also the MoE expert id-GEMM batch: at 256 the 256-expert weight planes
/// are re-read once per chunk, which dominated the routed tier
/// (tools/bench/flint_moe_mmq_bench.hip: one 2048-token id launch beats
/// eight 256-token launches ~7x on gate/up and ~8x on down). 4096 further
/// halves the per-chunk fixed cost for prompts past one chunk: exclusive
/// bench vs 2048 gives pp8192 1706.82 vs 1657.07 (+3.0 %) and pp16384
/// 1342.98 vs 1324.80 (+1.4 %), gates green; single-chunk prompts are
/// unaffected.
constexpr std::uint32_t kPrefillChunk = 4096;

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
  m->tokenizer_ = FlintTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  m->executor_ = FlintExecutor::Create(*m->reader_, *m->weights_, kPrefillChunk,
                                       options.max_context, error_msg);
  if (m->executor_ == nullptr) {
    return nullptr;
  }
  return m;
}

std::shared_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  if (!live_session_.expired()) {
    AssignError(error_msg, "the Flint device state is shared by all sessions");
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
                         .value_or("Flint Preview 40B-A2B"));
}

std::size_t Model::ResidentBytes() const noexcept {
  return executor_ == nullptr ? 0 : executor_->device_bytes();
}

std::size_t Model::SessionBytes(std::uint32_t context) const noexcept {
  // K+V fp16 per layer and position, matching the executor's KV planes.
  const Config& c = weights_->config;
  return static_cast<std::size_t>(c.num_layers) * 2u *
         static_cast<std::size_t>(context) * c.AttentionKvDim() * 2u;
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

Session::Session(std::shared_ptr<Model> model, std::uint32_t max_context)
    : model_(std::move(model)), max_context_(max_context) {}

Session::~Session() = default;

bool Session::ForwardChunked(std::span<const std::int32_t> tokens,
                             std::string* error_msg) {
  std::size_t offset = 0;
  while (offset < tokens.size()) {
    const std::size_t count =
        std::min<std::size_t>(kPrefillChunk, tokens.size() - offset);
    if (!model_->executor_->Forward(tokens.data() + offset,
                                    static_cast<std::uint32_t>(count), &logits_,
                                    error_msg)) {
      valid_ = false;
      return false;
    }
    offset += count;
  }
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.size() > max_context_) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  if (prompt.empty()) {
    model_->executor_->Reset();
    tokens_.clear();
    logits_.clear();
    valid_ = true;
    return true;
  }
  logits_.assign(model_->VocabSize(), 0.0f);
  const auto common = std::mismatch(tokens_.begin(), tokens_.end(),
                                    prompt.begin(), prompt.end());
  const std::span<const std::int32_t> tail(
      prompt.begin() +
          static_cast<std::ptrdiff_t>(common.first - tokens_.begin()),
      prompt.end());
  if (common.first != tokens_.end()) {
    model_->executor_->Reset();
    tokens_.clear();
    if (!ForwardChunked(prompt, error_msg)) {
      return false;
    }
    tokens_.assign(prompt.begin(), prompt.end());
    return true;
  }
  if (!tail.empty()) {
    if (!ForwardChunked(tail, error_msg)) {
      return false;
    }
    tokens_.insert(tokens_.end(), tail.begin(), tail.end());
  }
  return true;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  if (tokens_.size() + 1 > max_context_) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  logits_.assign(model_->VocabSize(), 0.0f);
  if (!model_->executor_->Forward(&token, 1, &logits_, error_msg)) {
    valid_ = false;
    return false;
  }
  tokens_.push_back(token);
  return true;
}

std::uint32_t Session::Position() const noexcept {
  return model_->executor_->position();
}

std::uint32_t Session::ContextSize() const noexcept {
  return static_cast<std::uint32_t>(tokens_.size());
}

void Session::Reset() {
  model_->executor_->Reset();
  tokens_.clear();
  logits_.clear();
  valid_ = true;
}

}  // namespace gufo::models::flint
