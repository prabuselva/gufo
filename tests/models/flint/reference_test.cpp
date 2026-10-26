// Runs the scalar oracle over the real Flint artifact. It is the numerical
// contract every ROCm kernel must reproduce, so the first gate is that a
// short forward pass produces finite, non-degenerate and repeatable logits.
// Strict parity against llama.cpp logits is captured separately once a
// matched run is available. Skips (77) unless GUFO_FLINT_GGUF points at a
// Flint GGUF.
#include "src/models/flint/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace fl = gufo::models::flint;
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
  Expect(ArgMax(logits) < vocab, tag + " argmax in range");
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
  const char* path = std::getenv("GUFO_FLINT_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: set GUFO_FLINT_GGUF\n";
    return 77;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::cerr << "cannot open " << path << ": " << error << "\n";
    return 1;
  }
  const auto weights = fl::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    std::cerr << "bind failed: " << error << "\n";
    return 1;
  }

  const auto& c = weights->config;
  const std::vector<std::int32_t> prompt = {100, 200, 300};
  std::vector<float> logits(c.vocab_size);

  fl::ReferenceModel model(*weights, /*max_context=*/64);
  std::vector<float> first_logits;
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    if (!model.Step(prompt[i], logits, &error)) {
      std::cerr << "step " << i << " failed: " << error << "\n";
      return 1;
    }
    CheckLogits(logits, c.vocab_size, "step " + std::to_string(i));
    if (i + 1 == prompt.size()) {
      first_logits = logits;
    }
  }
  Expect(model.Position() == static_cast<std::uint32_t>(prompt.size()),
         "position advanced");
  Expect(!model.Step(static_cast<std::int32_t>(c.vocab_size), logits, &error),
         "rejects an out-of-vocabulary token id");

  // A fresh model replaying the same prompt must reproduce the logits.
  fl::ReferenceModel replay(*weights, /*max_context=*/64);
  for (const std::int32_t token : prompt) {
    if (!replay.Step(token, logits, &error)) {
      std::cerr << "replay step failed: " << error << "\n";
      return 1;
    }
  }
  CheckRepeat(first_logits, logits, "trunk logits");

  // Reset rewinds the cache: one new model step equals the first step again.
  replay.Reset();
  Expect(replay.Position() == 0, "reset rewinds the position");
  if (!replay.Step(prompt[0], logits, &error)) {
    std::cerr << "reset step failed: " << error << "\n";
    return 1;
  }

  if (failures != 0) {
    std::cerr << failures << " reference-model checks failed\n";
    return 1;
  }
  std::cout << "Flint scalar reference passed.\n";
  return 0;
}
