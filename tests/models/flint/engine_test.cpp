#include "src/models/flint/engine.hpp"

#include <algorithm>
#include <cmath>
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

// Continuation parity between a replayed prefill and the incremental decode
// step that produced the same history.
//
// Same-sequence determinism stays bit-for-bit in every tier, but a replay
// through the mmq tensor-core prefill tier is no longer bit-identical to the
// batch-1 SIMT decode step: mmq tiles couple a row's accumulation order to
// the batch size, removing exactly the per-token row independence that made
// the two paths bitwise equal. The residual is bounded chaotic drift (the
// MoE trunk amplifies ordering noise into expert-selection flips): measured
// at max|diff|=2.08 with an identical argmax, 3.8 % of the 54.9-spread
// vocab logits of this artifact and prompt. Maintainer decision 2026-10-27
// replaced the bitwise equality with a 5 %-of-spread bound (floor 0.25)
// plus the tie-margin argmax rule of rocm_forward_test.cpp: the replay must
// pick the step's top-1 unless its top1-top2 margin lies inside the budget,
// in which case the replay's chosen candidate must itself sit within the
// budget of the step's top-1.
void ExpectReplay(std::span<const float> got, std::span<const float> ref,
                  const std::string& tag) {
  Require(got.size() == ref.size(), tag + ": logit width");
  bool finite = true;
  double max_abs = 0.0;
  double ref_hi = -1e30;
  double ref_lo = 1e30;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (!std::isfinite(got[i])) {
      finite = false;
    }
    max_abs = std::max(max_abs, std::abs(static_cast<double>(got[i]) -
                                         static_cast<double>(ref[i])));
    ref_hi = std::max(ref_hi, static_cast<double>(ref[i]));
    ref_lo = std::min(ref_lo, static_cast<double>(ref[i]));
  }
  Require(finite, tag + ": replay logits are finite");
  const double spread = ref_hi - ref_lo;
  const double budget = std::max(0.25, 0.05 * spread);
  const std::size_t ref_best = ArgMax(ref);
  double ref_second = -1e30;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (i != ref_best) {
      ref_second = std::max(ref_second, static_cast<double>(ref[i]));
    }
  }
  const double margin = ref_hi - ref_second;
  const std::size_t got_best = ArgMax(got);
  std::cout << tag << ": max|diff|=" << max_abs << " spread=" << spread
            << " budget=" << budget << " top1-top2 margin=" << margin << "\n";
  Require(max_abs <= budget, tag + ": replay logits within the budget");
  if (margin > budget) {
    Require(got_best == ref_best, tag + ": replay argmax matches the step");
  } else {
    Require(static_cast<double>(got[ref_best]) -
                    static_cast<double>(got[got_best]) <=
                budget,
            tag + ": tie argmax within budget of the step top-1");
  }
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

    // Replaying the same history from a fresh state must reproduce the step
    // logits under the continuation-parity contract (bounded drift +
    // tie-margin argmax; see ExpectReplay).
    session->Reset();
    std::vector<std::int32_t> history = ids;
    history.push_back(first);
    Require(session->Sync(history, &error), "sync history: " + error);
    ExpectReplay(session->Logits(), after_step, "replay history 1");

    // Extending the prompt through the already-forwarded prefix keeps the
    // append path and the same result as the step.
    const auto second = static_cast<std::int32_t>(ArgMax(session->Logits()));
    Require(session->Evaluate(second, &error), "evaluate 2: " + error);
    const std::vector<float> step2(session->Logits().begin(),
                                   session->Logits().end());
    session->Reset();
    history.push_back(second);
    Require(session->Sync(history, &error), "sync history 2: " + error);
    ExpectReplay(session->Logits(), step2, "replay history 2");

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
