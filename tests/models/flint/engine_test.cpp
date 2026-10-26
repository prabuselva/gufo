#include "src/models/flint/engine.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace fl = gufo::models::flint;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::size_t ArgMax(std::span<const float> row) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < row.size(); ++i) {
    if (row[i] > row[best]) {
      best = i;
    }
  }
  return best;
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_FLINT_GGUF");
  if (path == nullptr || *path == '\0') {
    std::cout << "SKIP: set GUFO_FLINT_GGUF\n";
    return 77;
  }
  try {
    std::string error;
    auto model = fl::Model::Load(path, {.max_context = 512}, &error);
    Require(model != nullptr, "load: " + error);
    Require(model->VocabSize() == 150144, "measured vocabulary size");
    Require(model->tokenizer().GetVocabSize() == model->VocabSize(),
            "tokenizer and model vocabularies must agree");
    Require(model->EosToken() >= 0, "eos token id");

    const std::string sample =
        "The capital of France is Paris.\nCounting: 12 345, rock'n'roll.";
    const auto ids = model->Tokenize(sample);
    Require(!ids.empty(), "tokenize produced no tokens");
    Require(model->Decode(ids) == sample, "decode(encode) mismatch");

    auto session = model->CreateSession(512, &error);
    Require(session != nullptr, "create session: " + error);
    Require(model->CreateSession(512, &error) == nullptr,
            "only one session may be live");

    Require(session->Sync(ids, &error), "sync prompt: " + error);
    Require(session->Logits().size() == model->VocabSize(), "logit width");
    Require(session->Position() == ids.size(), "position after prompt");
    Require(session->ContextSize() == ids.size(), "context after prompt");
    const auto first = static_cast<std::int32_t>(ArgMax(session->Logits()));

    Require(session->Evaluate(first, &error), "evaluate 1: " + error);
    Require(session->Position() == ids.size() + 1, "position after step");
    const std::vector<float> after_step(session->Logits().begin(),
                                        session->Logits().end());

    // Replaying the same history from a fresh state must reproduce the
    // logits bit-for-bit (deterministic kernels, divergence restart).
    session->Reset();
    std::vector<std::int32_t> history = ids;
    history.push_back(first);
    Require(session->Sync(history, &error), "sync history: " + error);
    Require(std::span(session->Logits()).data() != nullptr &&
                session->Logits().size() == after_step.size(),
            "history logit width");
    Require(std::equal(after_step.begin(), after_step.end(),
                       session->Logits().begin()),
            "replayed history must reproduce the last logits");

    // Extending the prompt through the already-forwarded prefix keeps the
    // append path and the same result as the step.
    const auto second = static_cast<std::int32_t>(ArgMax(session->Logits()));
    Require(session->Evaluate(second, &error), "evaluate 2: " + error);
    const std::vector<float> step2(session->Logits().begin(),
                                   session->Logits().end());
    session->Reset();
    history.push_back(second);
    Require(session->Sync(history, &error), "sync history 2: " + error);
    Require(std::equal(step2.begin(), step2.end(), session->Logits().begin()),
            "full replay must reproduce step 2 logits");

    session->Reset();
    Require(session->Position() == 0, "reset position");
    Require(session->Sync({}, &error), "sync empty");

    std::cout << "flint engine_test: OK (first=" << first
              << " second=" << second << ")\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "flint engine_test FAILED: " << e.what() << '\n';
    return 1;
  }
}
