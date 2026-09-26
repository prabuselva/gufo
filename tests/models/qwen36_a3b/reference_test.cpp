// Runs the scalar oracle over the real Qwen3.6-35B-A3B artifact. It is the
// numerical contract every ROCm kernel must reproduce, so the first gate is
// that a short forward pass produces finite, non-degenerate and repeatable
// logits and that the MTP draft does the same. Strict parity against llama.cpp
// logits is captured separately once a matched run is available. Skips (77)
// unless GUFO_QWEN36_A3B_GGUF points at a Qwen3.6 GGUF.
#include "src/models/qwen36_a3b/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace q36 = gufo::models::qwen36_a3b;
namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

std::uint32_t ArgMax(std::span<const float> v) {
  std::uint32_t best = 0;
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[best]) {
      best = static_cast<std::uint32_t>(i);
    }
  }
  return best;
}

// Every value finite, the spread non-zero (the head did not collapse) and the
// winning id inside the vocabulary.
void CheckLogits(std::span<const float> logits, std::uint32_t vocab,
                 const std::string& tag) {
  Expect(logits.size() == vocab, tag + " logit width");
  float lo = std::numeric_limits<float>::infinity();
  float hi = -std::numeric_limits<float>::infinity();
  bool finite = true;
  for (const float v : logits) {
    if (!std::isfinite(v)) {
      finite = false;
      break;
    }
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  Expect(finite, tag + " logits are finite");
  Expect(hi > lo, tag + " logits are non-degenerate");
  const std::uint32_t pick = ArgMax(logits);
  Expect(pick < vocab, tag + " argmax in range");
}

// Two identical runs must agree to float32 rounding; the reduction order is
// fixed for a given thread count, so a loose bound only catches real drift.
void CheckRepeat(const std::vector<float>& a, const std::vector<float>& b,
                 const std::string& tag) {
  Expect(a.size() == b.size(), tag + " repeat width");
  double max_abs = 0.0;
  double ref = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_abs = std::max(max_abs, static_cast<double>(std::abs(a[i] - b[i])));
    ref = std::max(ref, static_cast<double>(std::abs(a[i])));
  }
  Expect(max_abs <= 1e-2 * std::max(1.0, ref), tag + " repeat is stable");
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_QWEN36_A3B_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: set GUFO_QWEN36_A3B_GGUF\n";
    return 77;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::cerr << "cannot open " << path << ": " << error << "\n";
    return 1;
  }
  const auto weights = q36::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    std::cerr << "trunk bind failed: " << error << "\n";
    return 1;
  }
  const auto mtp = q36::MtpWeights::Bind(*reader, weights->config, &error);
  if (!mtp.has_value()) {
    std::cerr << "MTP bind failed: " << error << "\n";
    return 1;
  }

  const auto& c = weights->config;
  const std::vector<std::int32_t> prompt = {100, 200, 300};
  std::vector<float> logits(c.vocab_size);
  std::vector<float> hidden(c.hidden_size);

  q36::ReferenceModel model(*weights, /*max_context=*/64);
  std::vector<float> first_logits;
  std::vector<float> last_hidden;
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    if (!model.Step(prompt[i], logits, hidden, &error)) {
      std::cerr << "step " << i << " failed: " << error << "\n";
      return 1;
    }
    CheckLogits(logits, c.vocab_size, "step " + std::to_string(i));
    Expect(std::all_of(hidden.begin(), hidden.end(),
                       [](float v) { return std::isfinite(v); }),
           "hidden stream is finite");
    if (i + 1 == prompt.size()) {
      first_logits = logits;
      last_hidden = hidden;
    }
  }
  Expect(model.Position() == static_cast<std::uint32_t>(prompt.size()),
         "position advanced");

  // The MTP draft consumes the last trunk hidden and the next token.
  if (!model.MtpStep(*mtp, /*token=*/400, last_hidden, logits, &error)) {
    std::cerr << "mtp step failed: " << error << "\n";
    return 1;
  }
  CheckLogits(logits, c.vocab_size, "mtp draft");

  // A fresh model replaying the same prompt must reproduce the trunk logits.
  q36::ReferenceModel replay(*weights, /*max_context=*/64);
  std::vector<float> replay_hidden(c.hidden_size);
  for (const std::int32_t token : prompt) {
    if (!replay.Step(token, logits, replay_hidden, &error)) {
      std::cerr << "replay step failed: " << error << "\n";
      return 1;
    }
  }
  CheckRepeat(first_logits, logits, "trunk logits");

  if (failures != 0) {
    std::cerr << failures << " reference-model checks failed\n";
    return 1;
  }
  std::cout << "Qwen3.6-35B-A3B scalar reference passed.\n";
  return 0;
}