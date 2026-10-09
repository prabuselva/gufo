// End-to-end semantic gate for the MTP draft block. Parity tests pin the
// draft logits against the oracle for one step, but only real text shows the
// draft chain is aligned with the trunk: fed the trunk's hidden state and the
// freshly sampled token, the draft must predict the trunk's next greedy
// token most of the time. A position, cache or gate misalignment leaves the
// formulas intact and collapses agreement to near zero. Skips (77) unless
// GUFO_LING3_FLASH_GGUF points at a Ling 3.0 Flash GGUF.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling3_flash/kernels/rocm/executor.hpp"
#include "src/models/ling3_flash/weights.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "tests/models/ling3_flash/hip_test.hpp"

namespace l3 = gufo::models::ling3_flash;
namespace rocm = l3::rocm;
namespace ht = gufo::tests::ling3_flash;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

std::vector<float> Download(const float* device, std::size_t count) {
  std::vector<float> values(count);
  ht::CheckHip(hipMemcpy(values.data(), device, count * sizeof(float),
                         hipMemcpyDeviceToHost),
               "download");
  return values;
}

std::int32_t ArgMax(const std::vector<float>& v) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[best]) {
      best = i;
    }
  }
  return static_cast<std::int32_t>(best);
}

constexpr int kDecodeTokens = 32;
// Observed agreement on this artifact is well above this floor; a broken
// draft chain sits near the single-token chance level for a 157k vocab.
constexpr double kMinAcceptRate = 0.6;

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_LING3_FLASH_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: set GUFO_LING3_FLASH_GGUF\n";
    return 77;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::cerr << "cannot open " << path << ": " << error << "\n";
    return 1;
  }
  const auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    std::cerr << "tokenizer load failed: " << error << "\n";
    return 1;
  }
  const auto weights = l3::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    std::cerr << "trunk bind failed: " << error << "\n";
    return 1;
  }
  const auto mtp = l3::MtpWeights::Bind(*reader, weights->config, &error);
  if (!mtp.has_value()) {
    std::cerr << "MTP bind failed: " << error << "\n";
    return 1;
  }
  const auto device =
      rocm::DeviceModel::Upload(*weights, *reader, &*mtp, &error);
  if (device == nullptr) {
    std::cerr << "device upload failed: " << error << "\n";
    return 1;
  }
  constexpr std::uint32_t kContext = 128;
  const auto executor = rocm::Executor::Create(*device, kContext, &error);
  if (executor == nullptr) {
    std::cerr << "executor create failed: " << error << "\n";
    return 1;
  }
  const auto session = executor->CreateSession(kContext, &error);
  if (session == nullptr) {
    std::cerr << "session create failed: " << error << "\n";
    return 1;
  }

  std::vector<std::int32_t> prompt;
  for (const auto id : tokenizer->Encode("The capital of France is")) {
    prompt.push_back(static_cast<std::int32_t>(id));
  }
  if (prompt.size() < 4) {
    std::cerr << "prompt tokenized to " << prompt.size() << " tokens\n";
    return 1;
  }

  // The draft cache must track the trunk one entry per token, so every prompt
  // token feeds both blocks; the last prompt token's draft pair is produced
  // by the first sampled token below.
  const std::size_t vocab = weights->config.vocab_size;
  if (!executor->Step(*session, prompt[0], &error)) {
    std::cerr << "prefill step failed: " << error << "\n";
    return 1;
  }
  for (std::size_t i = 1; i < prompt.size(); ++i) {
    if (!executor->MtpStep(*session, prompt[i], &error)) {
      std::cerr << "prefill draft step " << i << " failed: " << error << "\n";
      return 1;
    }
    if (!executor->Step(*session, prompt[i], &error)) {
      std::cerr << "prefill step " << i << " failed: " << error << "\n";
      return 1;
    }
  }

  const std::int32_t eos =
      static_cast<std::int32_t>(tokenizer->GetEosTokenId());
  const std::int32_t pad =
      static_cast<std::int32_t>(tokenizer->GetPadTokenId());
  int agreed = 0;
  int total = 0;
  for (int i = 0; i < kDecodeTokens; ++i) {
    const auto token = ArgMax(Download(executor->logits(), vocab));
    if (token == eos || token == pad) {
      break;
    }
    if (!executor->MtpStep(*session, token, &error)) {
      std::cerr << "draft step failed: " << error << "\n";
      return 1;
    }
    const auto draft = ArgMax(Download(executor->mtp_logits(), vocab));
    if (!executor->Step(*session, token, &error)) {
      std::cerr << "decode step failed: " << error << "\n";
      return 1;
    }
    const auto actual = ArgMax(Download(executor->logits(), vocab));
    ++total;
    if (draft == actual) {
      ++agreed;
    }
  }
  if (total < kDecodeTokens / 2) {
    std::cerr << "only " << total << " decode steps measured\n";
    ++failures;
  }
  const double rate = total == 0 ? 0.0 : static_cast<double>(agreed) / total;
  std::cout << "MTP greedy draft agreement: " << agreed << "/" << total << " ("
            << rate * 100.0 << "%)\n";
  Expect(rate >= kMinAcceptRate,
         "draft agrees with the trunk most of the time");

  if (failures != 0) {
    std::cerr << failures << " MTP acceptance checks failed\n";
    return 1;
  }
  std::cout << "Ling 3.0 Flash MTP drafts track the trunk.\n";
  return 0;
}
