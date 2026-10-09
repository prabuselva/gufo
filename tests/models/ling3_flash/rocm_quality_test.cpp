// Greedy quality gate over real text: a raw continuation of "The capital of
// France is" must stay coherent and name Paris, matching the reference
// server's greedy output for the same artifact. The parity tests only check
// that the CPU oracle and GPU executor agree, so they cannot catch a shared
// formula error (a corrupted conv history once decoded pure whitespace while
// both paths matched). Skips (77) unless GUFO_LING3_FLASH_GGUF points at a
// Ling 3.0 Flash GGUF.
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "src/models/ling3_flash/engine.hpp"

namespace l3 = gufo::models::ling3_flash;
namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

std::int32_t ArgMax(std::span<const float> logits) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) {
      best = i;
    }
  }
  return static_cast<std::int32_t>(best);
}

std::int32_t GreedyRun(const std::shared_ptr<l3::Model>& model,
                       const std::string& prompt, int max_tokens,
                       std::string* out_text, std::string* error) {
  auto session = model->CreateSession(model->MaxContext(), error);
  if (session == nullptr) {
    return 1;
  }
  const auto ids = model->Tokenize(prompt);
  if (!session->Sync(ids, error)) {
    return 1;
  }
  std::vector<std::int32_t> generated;
  for (int i = 0; i < max_tokens; ++i) {
    const std::int32_t next = ArgMax(session->Logits());
    if (model->IsStopToken(next)) {
      break;
    }
    generated.push_back(next);
    if (!session->Evaluate(next, error)) {
      return 1;
    }
  }
  *out_text = model->Decode(generated);
  return 0;
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_LING3_FLASH_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: set GUFO_LING3_FLASH_GGUF\n";
    return 77;
  }

  std::string error;
  l3::ModelOptions options;
  options.max_context = 256;
  const auto model = l3::Model::Load(path, options, &error);
  if (model == nullptr) {
    std::cerr << "load failed: " << error << "\n";
    return 1;
  }

  std::string text;
  if (GreedyRun(model, "The capital of France is", 12, &text, &error) != 0) {
    std::cerr << "greedy run failed: " << error << "\n";
    return 1;
  }
  std::cout << "raw greedy: " << text << "\n";
  Expect(text.find("Paris") != std::string::npos,
         "raw greedy continuation names Paris");

  if (failures != 0) {
    std::cerr << failures << " quality checks failed\n";
    return 1;
  }
  std::cout << "Ling 3.0 Flash greedy generation is coherent.\n";
  return 0;
}
