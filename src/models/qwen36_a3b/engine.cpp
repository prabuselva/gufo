#include "src/models/qwen36_a3b/engine.hpp"

#include <algorithm>
#include <cstring>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/device_model.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/executor.hpp"
#include "src/models/qwen36_a3b/weights.hpp"

namespace gufo::models::qwen36_a3b {
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
  m->tokenizer_ =
      tokenization::QwenTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  // The MTP draft block is optional: artifacts without it load and decode
// single-token, artifacts with it enable draft-verify speculation.
const auto mtp = MtpWeights::Bind(*m->reader_, c);
  m->device_ = rocm::DeviceModel::Upload(
      *m->weights_, *m->reader_, mtp.has_value() ? &*mtp : nullptr, error_msg);
  if (!m->device_) {
    return nullptr;
  }
  m->executor_ =
      rocm::Executor::Create(*m->device_, options.max_context, error_msg);
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
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), executor_.get()));
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  std::vector<std::int32_t> out;
  for (auto id : tokenizer_->Encode(text)) {
    out.push_back(static_cast<std::int32_t>(id));
  }
  return out;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<tokenization::TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<tokenization::TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return token == EosToken() ||
         token == static_cast<std::int32_t>(tokenizer_->GetPadTokenId());
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

bool Model::HasMtp() const noexcept {
  return device_->has_mtp();
}

std::string Model::ModelName() const {
  return std::string(reader_->GetMetadataString("general.name")
                         .value_or("Qwen3.6-35B-A3B"));
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes();
}

Session::Session(std::shared_ptr<Model> model, rocm::Executor* executor)
    : model_(std::move(model)), executor_(executor) {
  logits_.resize(model_->VocabSize());
  draft_logits_.resize(model_->VocabSize());
  verify_rows_.resize(2 * model_->VocabSize());
}

Session::~Session() = default;

std::uint32_t Session::Position() const noexcept {
  return executor_->position();
}

std::uint32_t Session::ContextSize() const noexcept {
  return executor_->max_context();
}

void Session::Reset() {
  valid_ = false;
  executor_->Reset();
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
  if (!valid_)
    Reset();
  // Recurrent state cannot be rewound, so any divergence restarts the
  // session; an extension only feeds the new tail.
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
  // over the whole span instead of a per-token GEMV step, so prompt processing
  // runs at prefill throughput. The executor's position already equals the
  // common prefix length (reset to 0 above, or a pure extension), so Prefill
  // continues from there and leaves the final token's logits.
  if (!executor_->Prefill(prompt.data() + common,
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
  if (!executor_->Step(token, error_msg)) {
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
    AssignError(error_msg,
                "decode needs an output, a positive budget and a synced prompt");
    return false;
  }
  *result = {};
  const auto is_stop = [&](std::int32_t token) {
    return stop_at_eos && model_->IsStopToken(token);
  };

  // Speculative decoding: with an MTP block and a greedy sampler, draft one
  // token with the MTP block, verify both tokens in one batched trunk pass,
  // and accept the draft only when the trunk agrees. The verify pass reads
  // the trunk weights once for two tokens, so a hit costs ~1.2 tokens' time.
  const bool spec = model_->HasMtp() && max_tokens >= 2 &&
                    sampler.config().can_use_unmodified_argmax() &&
                    tokens_.size() + 2 <= ContextSize();
  if (!spec) {
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits_));
    if (is_stop(token)) {
      result->stop = true;
      return true;
    }
    valid_ = false;
    if (!executor_->Step(token, error_msg)) {
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

  const auto token = static_cast<std::int32_t>(sampler.Sample(logits_));
  if (is_stop(token)) {
    result->stop = true;
    return true;
  }
  valid_ = false;
  if (!executor_->MtpStep(token, error_msg)) {
    return false;
  }
  ++result->drafted;
  (void)hipMemcpy(draft_logits_.data(), executor_->mtp_logits(),
                  draft_logits_.size() * sizeof(float), hipMemcpyDeviceToHost);
  const auto draft = static_cast<std::int32_t>(
      sampler.SampleGreedy(std::span<const float>(draft_logits_)));
  if (!executor_->Verify(token, draft, error_msg)) {
    return false;
  }
  (void)hipMemcpy(verify_rows_.data(), executor_->verify_logits(),
                  verify_rows_.size() * sizeof(float), hipMemcpyDeviceToHost);
  const std::span<const float> row0(verify_rows_.data(), draft_logits_.size());
  const std::span<const float> row1(verify_rows_.data() + draft_logits_.size(),
                                    draft_logits_.size());
  if (static_cast<std::int32_t>(sampler.SampleGreedy(row0)) == draft) {
    // The trunk reproduces the draft: keep both tokens and advance the MTP
    // caches over the accepted pair so the next draft stays aligned.
    if (!executor_->MtpAdvance(draft, executor_->verify_hidden(), error_msg)) {
      return false;
    }
    logits_.assign(row1.begin(), row1.end());
    tokens_.push_back(token);
    tokens_.push_back(draft);
    const sampling::TokenId accepted[2] = {
        static_cast<sampling::TokenId>(token),
        static_cast<sampling::TokenId>(draft)};
    sampler.Accept(std::span<const sampling::TokenId>(accepted));
    result->tokens.push_back(token);
    result->tokens.push_back(draft);
    ++result->accepted;
    result->stop = is_stop(draft);
  } else {
    // The trunk disagrees at the draft position: rewind to the state after
    // the accepted token and continue from the trunk's own logits.
    executor_->RollbackVerify();
    logits_.assign(row0.begin(), row0.end());
    tokens_.push_back(token);
    sampler.Accept(static_cast<sampling::TokenId>(token));
    result->tokens.push_back(token);
  }
  valid_ = true;
  return true;
}

}  // namespace gufo::models::qwen36_a3b