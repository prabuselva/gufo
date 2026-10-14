// Width-invariance hash gate for MTP speculative decoding: greedy decode
// through the speculative path must emit exactly the token stream greedy
// autoregressive decoding produces, because acceptance uses exact argmax
// agreement. Runs both paths over the real artifact (engine Session level,
// the same path serve and bench use), pins the draft count so the draft
// block provably runs (drafted > 0), and hashes both streams. Skips (77)
// unless GUFO_QWEN36_A3B_GGUF points at a Qwen3.6 GGUF with an MTP block.
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen36_a3b/engine.hpp"

namespace q36 = gufo::models::qwen36_a3b;
namespace sampling = gufo::sampling;
namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

std::string StreamSha(std::span<const std::int32_t> tokens) {
  std::string hex;
  static constexpr char kDigits[] = "0123456789abcdef";
  std::uint64_t h = 1469598103934665603ULL;
  for (const std::int32_t t : tokens) {
    for (int b = 0; b < 4; ++b) {
      h ^= static_cast<std::uint8_t>(static_cast<std::uint32_t>(t) >> (b * 8));
      h *= 1099511628211ULL;
    }
  }
  for (int i = 60; i >= 0; i -= 4) {
    hex.push_back(kDigits[(h >> i) & 0xFU]);
  }
  return hex;
}

struct Decoded {
  std::vector<std::int32_t> tokens;
  std::size_t drafted{0};
  std::size_t accepted{0};
};

Decoded Decode(q36::Session& session, std::size_t steps, std::size_t budget) {
  sampling::SamplerState sampler;
  Decoded out;
  std::string error;
  while (out.tokens.size() < steps) {
    q36::Session::DecodeResult result;
    if (!session.DecodeStep(budget, sampler, &result, &error, false) ||
        result.tokens.empty()) {
      std::cerr << "FAIL: decode stopped: " << error << "\n";
      ++failures;
      break;
    }
    out.tokens.insert(out.tokens.end(), result.tokens.begin(),
                      result.tokens.end());
    out.drafted += result.drafted;
    out.accepted += result.accepted;
  }
  out.tokens.resize(steps);
  return out;
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_QWEN36_A3B_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: set GUFO_QWEN36_A3B_GGUF\n";
    return 77;
  }
  std::string error;
  q36::ModelOptions options;
  options.max_context = 4096;
  const auto model = q36::Model::Load(path, options, &error);
  if (model == nullptr) {
    std::cerr << "load failed: " << error << "\n";
    return 1;
  }
  if (!model->HasMtp()) {
    std::cout << "SKIP: artifact has no MTP block\n";
    return 77;
  }
  const auto prompt = model->Tokenize(
      "Explain in one short paragraph why the sky appears blue during the "
      "day and red at sunset.");
  constexpr std::size_t kSteps = 64;
  constexpr std::uint32_t kDrafts = 4;

  const auto session = model->CreateSession(4096, &error);
  if (session == nullptr) {
    std::cerr << "session failed: " << error << "\n";
    return 1;
  }

  // Reference: greedy autoregressive decode without the draft block.
  if (!session->Sync(prompt, &error)) {
    std::cerr << "AR sync failed: " << error << "\n";
    return 1;
  }
  const Decoded ar = Decode(*session, kSteps, 1);

  // Candidate: the same session re-prefilled with MTP enabled and the per-
  // round draft count pinned, so every round chains exactly `kDrafts`.
  session->Reset();
  session->SetMtpEnabled(true);
  session->SetDraftLimits(kDrafts, kDrafts);
  if (!session->Sync(prompt, &error)) {
    std::cerr << "MTP sync failed: " << error << "\n";
    return 1;
  }
  const Decoded spec = Decode(*session, kSteps, kSteps);

  std::cout << "ar   sha=" << StreamSha(ar.tokens) << "\n"
            << "spec sha=" << StreamSha(spec.tokens) << "\n"
            << "drafted=" << spec.drafted << " accepted=" << spec.accepted
            << " (rate="
            << (spec.drafted == 0 ? 0.0
                                  : static_cast<double>(spec.accepted) /
                                        static_cast<double>(spec.drafted))
            << ")\n";
  Expect(spec.drafted > 0, "speculative path drafted nothing (gate vacuous)");
  Expect(spec.accepted > 0, "speculative path accepted nothing");
  Expect(ar.tokens == spec.tokens,
         "greedy speculative stream differs from autoregressive");
  if (ar.tokens != spec.tokens) {
    for (std::size_t i = 0; i < std::min(ar.tokens.size(), spec.tokens.size());
         ++i) {
      if (ar.tokens[i] == spec.tokens[i]) {
        continue;
      }
      std::cerr << "first divergence at index " << i << ": ar=" << ar.tokens[i]
                << " spec=" << spec.tokens[i] << "\n";
      break;
    }
  }
  if (failures == 0) {
    std::cout << "OK: greedy MTP speculative decoding is hash-equal to AR\n";
  }
  return failures == 0 ? 0 : 1;
}
