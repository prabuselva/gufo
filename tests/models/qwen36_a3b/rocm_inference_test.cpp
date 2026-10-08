// Runs the scalar oracle and the GPU executor over the same real text prompt
// and greedy-decodes a short continuation with each. The two token streams
// must agree exactly (the forward test already proves the logits match to
// float32 rounding); printing the decoded text is the quality gate the
// forward test cannot provide, because it only checks GPU-vs-CPU agreement,
// not that the shared forward pass is actually correct. Skips (77) unless
// GUFO_QWEN36_A3B_GGUF points at a Qwen3.6 GGUF.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/device_model.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/executor.hpp"
#include "src/models/qwen36_a3b/reference.hpp"
#include "tests/models/qwen36_a3b/hip_test.hpp"

namespace q36 = gufo::models::qwen36_a3b;
namespace rocm = gufo::models::qwen36_a3b::rocm;
namespace test = gufo::tests::qwen36_a3b;
namespace tok = gufo::tokenization;
namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

std::uint32_t ArgMax(const std::vector<float>& v) {
  std::uint32_t best = 0;
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[best]) {
      best = static_cast<std::uint32_t>(i);
    }
  }
  return best;
}

// Greedy-decodes `steps` tokens from a starting logits vector, feeding each
// sampled token back through `advance`. Returns the sampled token ids.
template<typename Advance>
std::vector<std::int32_t> Greedy(std::vector<float> logits, int steps,
                                 Advance&& advance) {
  std::vector<std::int32_t> tokens;
  for (int s = 0; s < steps; ++s) {
    const std::int32_t token = static_cast<std::int32_t>(ArgMax(logits));
    tokens.push_back(token);
    if (!advance(token, logits)) {
      break;
    }
  }
  return tokens;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = std::getenv("GUFO_QWEN36_A3B_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: set GUFO_QWEN36_A3B_GGUF\n";
    return 77;
  }
  const std::string prompt_text =
      argc > 1 ? argv[1] : "The capital of France is";
  const int steps = argc > 2 ? std::atoi(argv[2]) : 12;

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
  const auto tokenizer = tok::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::cerr << "tokenizer failed: " << error << "\n";
    return 1;
  }

  const auto& c = weights->config;
  std::cerr << "DBG: vocab=" << c.vocab_size << " hidden=" << c.hidden_size
            << " layers=" << c.num_layers << "\n";
  const auto encoded = tokenizer->Encode(prompt_text);
  std::cerr << "DBG: encoded " << encoded.size() << " tokens\n";
  const std::vector<std::int32_t> prompt(encoded.begin(), encoded.end());
  std::cout << "prompt: \"" << prompt_text << "\" (" << prompt.size()
            << " tokens)\n";
  for (const std::int32_t t : prompt) {
    std::cout << "  " << t << " " << tokenizer->DecodeTokenCopy(t) << "\n";
  }

  const std::uint32_t max_context = static_cast<std::uint32_t>(prompt.size()) +
                                    static_cast<std::uint32_t>(steps) + 8;

  // --- CPU oracle + GPU executor, interleaved prefill with per-step compare
  // --- The forward test proves a 3-token prefill matches; this pinpoints which
  // prefill step (if any) diverges on the real prompt, and whether the bug is
  // in prefill or only in the decode loop.
  q36::ReferenceModel reference(*weights, max_context);
  std::vector<float> ref_logits(c.vocab_size);
  std::vector<float> ref_hidden(c.hidden_size);

  const auto device_model =
      rocm::DeviceModel::Upload(*weights, *reader, nullptr, &error);
  if (device_model == nullptr) {
    std::cerr << "upload failed: " << error << "\n";
    return 1;
  }
  const auto executor =
      rocm::Executor::Create(*device_model, max_context, &error);
  if (executor == nullptr) {
    std::cerr << "executor create failed: " << error << "\n";
    return 1;
  }
  const auto session = executor->CreateSession(max_context, &error);
  if (session == nullptr) {
    std::cerr << "session create failed: " << error << "\n";
    return 1;
  }

  std::vector<float> gpu_logits(c.vocab_size);
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    const std::int32_t token = prompt[i];
    if (!reference.Step(token, ref_logits, ref_hidden, &error)) {
      std::cerr << "CPU prefill step " << i << " failed: " << error << "\n";
      return 1;
    }
    if (!executor->Step(*session, token, &error)) {
      std::cerr << "GPU prefill step " << i << " failed: " << error << "\n";
      return 1;
    }
    test::CheckHip(
        hipMemcpy(gpu_logits.data(), executor->logits(),
                  c.vocab_size * sizeof(float), hipMemcpyDeviceToHost),
        "download prefill logits");
    const std::uint32_t cpu_arg = ArgMax(ref_logits);
    const std::uint32_t gpu_arg = ArgMax(gpu_logits);
    const double worst = test::WorstRelative(ref_logits, gpu_logits, 1e-3);
    std::cout << "prefill step " << i << " (token " << token
              << "): cpu_arg=" << cpu_arg << " gpu_arg=" << gpu_arg
              << (cpu_arg == gpu_arg ? "  OK" : "  *** DIVERGED ***")
              << "  worst_rel=" << worst << "\n";
  }

  // --- Generation from the post-prefill logits ---
  const auto cpu_tokens = Greedy(
      ref_logits, steps, [&](std::int32_t token, std::vector<float>& logits) {
        return reference.Step(token, logits, ref_hidden, &error);
      });
  const auto gpu_tokens = Greedy(
      gpu_logits, steps, [&](std::int32_t token, std::vector<float>& logits) {
        if (!executor->Step(*session, token, &error)) {
          return false;
        }
        test::CheckHip(
            hipMemcpy(logits.data(), executor->logits(),
                      c.vocab_size * sizeof(float), hipMemcpyDeviceToHost),
            "download gen logits");
        return true;
      });

  // --- Batched prefill path: one Prefill() over the whole prompt must land on
  // the same post-prompt logits as the per-token Step() loop and the oracle,
  // and decoding must resume correctly from the advanced position. ---
  const auto pf_executor =
      rocm::Executor::Create(*device_model, max_context, &error);
  if (pf_executor == nullptr) {
    std::cerr << "prefill executor create failed: " << error << "\n";
    return 1;
  }
  const auto pf_session = pf_executor->CreateSession(max_context, &error);
  if (pf_session == nullptr) {
    std::cerr << "prefill session create failed: " << error << "\n";
    return 1;
  }
  if (!pf_executor->Prefill(*pf_session, prompt.data(),
                            static_cast<std::uint32_t>(prompt.size()),
                            &error)) {
    std::cerr << "GPU batched prefill failed: " << error << "\n";
    return 1;
  }
  std::vector<float> pf_logits(c.vocab_size);
  test::CheckHip(hipMemcpy(pf_logits.data(), pf_executor->logits(),
                           c.vocab_size * sizeof(float), hipMemcpyDeviceToHost),
                 "download batched prefill logits");
  {
    const std::uint32_t cpu_arg = ArgMax(ref_logits);
    const std::uint32_t pf_arg = ArgMax(pf_logits);
    const double worst = test::WorstRelative(ref_logits, pf_logits, 1e-3);
    std::cout << "\nbatched prefill: cpu_arg=" << cpu_arg
              << " pf_arg=" << pf_arg
              << (cpu_arg == pf_arg ? "  OK" : "  *** DIVERGED ***")
              << "  worst_rel=" << worst << "\n";
    Expect(cpu_arg == pf_arg, "batched prefill argmax matches oracle");
    Expect(pf_session->position() == static_cast<std::uint32_t>(prompt.size()),
           "batched prefill advanced position");
    // Decisive cross-check: compare the batched prefill directly against the
    // per-token GPU path (the original working path), not just the oracle. If
    // these match, the batched path is correct and the oracle is wrong; if
    // they diverge, the batched path is genuinely buggy.
    const std::uint32_t gpu_arg = ArgMax(gpu_logits);
    const double worst_gpu = test::WorstRelative(gpu_logits, pf_logits, 1e-3);
    std::cout << "batched vs per-token GPU: gpu_arg=" << gpu_arg
              << " pf_arg=" << pf_arg
              << (gpu_arg == pf_arg ? "  OK" : "  *** DIVERGED ***")
              << "  worst_rel=" << worst_gpu << "\n";
  }
  const auto pf_tokens = Greedy(
      pf_logits, steps, [&](std::int32_t token, std::vector<float>& logits) {
        if (!pf_executor->Step(*pf_session, token, &error)) {
          return false;
        }
        test::CheckHip(
            hipMemcpy(logits.data(), pf_executor->logits(),
                      c.vocab_size * sizeof(float), hipMemcpyDeviceToHost),
            "download batched-prefill gen logits");
        return true;
      });
  std::cout << "prefill text: "
            << tokenizer->Decode(std::vector<tok::TokenId>(pf_tokens.begin(),
                                                           pf_tokens.end()))
            << "\n";
  Expect(pf_tokens.size() == cpu_tokens.size(),
         "batched-prefill token count matches oracle");
  for (std::size_t i = 0; i < std::min(pf_tokens.size(), cpu_tokens.size());
       ++i) {
    Expect(pf_tokens[i] == cpu_tokens[i],
           "batched-prefill token " + std::to_string(i) +
               " matches oracle (pf=" + std::to_string(pf_tokens[i]) +
               " cpu=" + std::to_string(cpu_tokens[i]) + ")");
  }

  // --- Compare ---
  std::cout << "\nCPU tokens: ";
  for (const std::int32_t t : cpu_tokens)
    std::cout << t << " ";
  std::cout << "\nGPU tokens: ";
  for (const std::int32_t t : gpu_tokens)
    std::cout << t << " ";
  std::cout << "\n\nCPU text: "
            << tokenizer->Decode(std::vector<tok::TokenId>(cpu_tokens.begin(),
                                                           cpu_tokens.end()))
            << "\nGPU text: "
            << tokenizer->Decode(std::vector<tok::TokenId>(gpu_tokens.begin(),
                                                           gpu_tokens.end()))
            << "\n";

  Expect(cpu_tokens.size() == gpu_tokens.size(), "token count matches");
  const std::size_t n = std::min(cpu_tokens.size(), gpu_tokens.size());
  for (std::size_t i = 0; i < n; ++i) {
    Expect(cpu_tokens[i] == gpu_tokens[i],
           "token " + std::to_string(i) +
               " matches (cpu=" + std::to_string(cpu_tokens[i]) +
               " gpu=" + std::to_string(gpu_tokens[i]) + ")");
  }

  if (failures != 0) {
    std::cerr << failures << " inference checks failed\n";
    return 1;
  }
  std::cout << "Qwen3.6-35B-A3B CPU/GPU inference matched.\n";
  return 0;
}