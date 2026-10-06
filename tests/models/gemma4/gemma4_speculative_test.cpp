// M8b speculative decode: greedy generation with the MTP draft enabled must be
// token-for-token identical to greedy generation without it. Speculation is
// lossless by construction: the trunk verify pass reproduces the exact greedy
// prefix, so enabling the draft can only change *how many* tokens a round
// commits, never *which* tokens. The test drives a realistic chat prompt
// through the engine twice -- once non-speculative, once with MTP on -- and
// asserts the emitted sequences agree (cut at the stop token, which the two
// paths report at the same logical position). It also asserts the speculative
// run actually accepted drafts, so a silently-degenerate k == 0 path cannot
// pass. Skips (77) unless GUFO_GEMMA4_GGUF and GUFO_GEMMA4_MTP_GGUF point at
// the trunk and draft GGUFs; needs the full GPU (stop the resident server).
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/engine.hpp"

namespace g4 = gufo::models::gemma4;
namespace sampling = gufo::sampling;
namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

// Drops the trailing stop token (and anything after it) so the two paths, which
// report EOS at the same position but append it differently, compare equal.
void CutAtStop(const g4::Model& model, std::vector<std::int32_t>& tokens) {
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (model.IsStopToken(tokens[i])) {
      tokens.resize(i);
      return;
    }
  }
}

// Greedily decodes up to `budget` tokens from `prompt`. When `speculative`, the
// session enables MTP and the drafted/accepted counters are accumulated.
bool Run(const std::shared_ptr<g4::Model>& model,
         const std::vector<std::int32_t>& prompt, std::size_t budget,
         bool speculative, std::vector<std::int32_t>* out, std::size_t* drafted,
         std::size_t* accepted, std::string* error) {
  auto session = model->CreateSession(model->MaxContext(), error);
  if (session == nullptr) {
    return false;
  }
  if (speculative) {
    session->SetMtpEnabled(true);
  }
  if (!session->Sync(prompt, error)) {
    return false;
  }
  sampling::SamplerState sampler;  // default config is greedy argmax
  while (out->size() < budget) {
    g4::Session::DecodeResult decoded;
    if (!session->DecodeStep(budget - out->size(), sampler, &decoded, error)) {
      return false;
    }
    if (drafted != nullptr) {
      *drafted += decoded.drafted;
    }
    if (accepted != nullptr) {
      *accepted += decoded.accepted;
    }
    for (const std::int32_t token : decoded.tokens) {
      out->push_back(token);
    }
    if (decoded.stop) {
      break;
    }
  }
  return true;
}

}  // namespace

int main() {
  const char* trunk_path = std::getenv("GUFO_GEMMA4_GGUF");
  const char* draft_path = std::getenv("GUFO_GEMMA4_MTP_GGUF");
  if (trunk_path == nullptr || trunk_path[0] == '\0' || draft_path == nullptr ||
      draft_path[0] == '\0') {
    std::cout << "SKIP: set GUFO_GEMMA4_GGUF and GUFO_GEMMA4_MTP_GGUF\n";
    return 77;
  }
  std::string error;
  g4::ModelOptions options;
  options.max_context = 512;
  options.prefill_chunk = 512;
  options.draft_path = draft_path;
  const auto model = g4::Model::Load(trunk_path, options, &error);
  if (model == nullptr) {
    std::cerr << "load failed: " << error << "\n";
    return 1;
  }
  Expect(model->HasMtp(), "model reports MTP available");

  const std::vector<g4::ChatMessage> messages = {
      {.role = "user",
       .content = "List three reasons the sky appears blue, in one short "
                  "paragraph."},
  };
  const auto prompt = model->EncodeChat(messages, {}, {});
  if (prompt.empty()) {
    std::cerr << "chat prompt encoded to zero tokens\n";
    return 1;
  }

  const std::size_t budget = 48;
  std::vector<std::int32_t> reference;
  if (!Run(model, prompt, budget, /*speculative=*/false, &reference, nullptr,
           nullptr, &error)) {
    std::cerr << "non-speculative decode failed: " << error << "\n";
    return 1;
  }
  Expect(!reference.empty(), "non-speculative decode produced tokens");

  std::vector<std::int32_t> speculative;
  std::size_t drafted = 0;
  std::size_t accepted = 0;
  if (!Run(model, prompt, budget, /*speculative=*/true, &speculative, &drafted,
           &accepted, &error)) {
    std::cerr << "speculative decode failed: " << error << "\n";
    return 1;
  }
  Expect(!speculative.empty(), "speculative decode produced tokens");

  CutAtStop(*model, reference);
  CutAtStop(*model, speculative);
  std::cout << "reference " << reference.size() << " tokens, speculative "
            << speculative.size() << " tokens, drafted " << drafted
            << ", accepted " << accepted << "\n";
  Expect(reference == speculative,
         "speculative greedy output matches non-speculative");
  Expect(drafted > 0, "speculative path proposed drafts");
  Expect(accepted > 0, "speculative path accepted at least one draft");

  if (failures != 0) {
    std::cerr << failures << " speculative decode checks failed\n";
    return 1;
  }
  std::cout << "Gemma-4-26B-A4B speculative decode parity passed.\n";
  return 0;
}