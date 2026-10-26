// Manual C6 parity probe (not a ctest): prints the greedy chain and the
// top-K logits per position so they can be compared against llama.cpp
// `--logprob all` dumps offline.
//
// Usage: GUFO_FLINT_GGUF=path flint_logits_probe "raw prompt text" [steps]
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/flint/engine.hpp"

namespace {

int Die(const std::string& message) {
  std::cerr << "error: " << message << "\n";
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = std::getenv("GUFO_FLINT_GGUF");
  if (path == nullptr || argc < 2) {
    return Die("GUFO_FLINT_GGUF and a prompt argument are required");
  }
  const std::string prompt = argv[1];
  const int steps = argc > 2 ? std::atoi(argv[2]) : 8;
  std::string error;
  auto model = gufo::models::flint::Model::Load(
      path, gufo::models::flint::ModelOptions{.max_context = 4096}, &error);
  if (model == nullptr) {
    return Die("load: " + error);
  }
  const auto ids = model->tokenizer().Encode(prompt);
  std::vector<std::int32_t> tokens;
  tokens.reserve(ids.size() + static_cast<std::size_t>(steps));
  for (const auto id : ids) {
    tokens.push_back(static_cast<std::int32_t>(id));
  }
  auto session = model->CreateSession(4096, &error);
  if (session == nullptr) {
    return Die("session: " + error);
  }
  if (!session->Sync(tokens, &error)) {
    return Die("sync: " + error);
  }
  std::cout << "prompt_tokens " << tokens.size() << "\n";
  for (int step = 0; step < steps; ++step) {
    auto logits = session->Logits();
    if (logits.empty()) {
      return Die("no logits");
    }
    std::vector<std::int32_t> order(logits.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
      order[i] = static_cast<std::int32_t>(i);
    }
    const int top_k = 32;
    std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                      [&logits](std::int32_t a, std::int32_t b) {
                        return logits[a] > logits[b];
                      });
    const float max_logit = logits[static_cast<std::size_t>(order[0])];
    for (int rank = 0; rank < top_k; ++rank) {
      const std::int32_t id = order[static_cast<std::size_t>(rank)];
      std::cout << "s" << step << " " << id << " " << logits[id] - max_logit
                << "\n";
    }
    const std::int32_t next = order[0];
    std::cout << "s" << step << " chosen " << next << "\n";
    tokens.push_back(next);
    if (step + 1 < steps) {
      if (!session->Evaluate(next, &error)) {
        return Die("evaluate: " + error);
      }
    }
  }
  return 0;
}
