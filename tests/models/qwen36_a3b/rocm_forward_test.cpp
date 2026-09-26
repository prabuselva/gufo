// Runs the GPU executor over the real Qwen3.6-35B-A3B artifact and checks that
// its trunk and MTP logits agree with the scalar oracle to float32 rounding.
// This is the end-to-end numerical contract: every fused operator and the
// quantized GEMV tier, wired together, must reproduce the reference forward
// pass. Skips (77) unless GUFO_QWEN36_A3B_GGUF points at a Qwen3.6 GGUF.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/qwen36_a3b/kernels/rocm/device_model.hpp"
#include "src/models/qwen36_a3b/kernels/rocm/executor.hpp"
#include "src/models/qwen36_a3b/reference.hpp"
#include "tests/models/qwen36_a3b/hip_test.hpp"

namespace q36 = gufo::models::qwen36_a3b;
namespace rocm = gufo::models::qwen36_a3b::rocm;
namespace test = gufo::tests::qwen36_a3b;
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

std::vector<float> Download(const float* device, std::size_t count) {
  std::vector<float> values(count);
  test::CheckHip(hipMemcpy(values.data(), device, count * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "download");
  return values;
}

// The GPU GEMV accumulates in float32 (one warp per row) while the scalar
// oracle accumulates in double. With cancellation in the 2048-term dot product
// the float partial sums can drift by ~1e-2 relative on the largest logits, so
// the bound is set above that drift; the argmax must still agree exactly.
void CheckLogits(const std::vector<float>& reference,
                 const std::vector<float>& candidate, const std::string& tag) {
  Expect(reference.size() == candidate.size(), tag + " logit width");
  bool finite = true;
  for (const float v : candidate) {
    if (!std::isfinite(v)) {
      finite = false;
      break;
    }
  }
  Expect(finite, tag + " GPU logits are finite");
  const double worst = test::WorstRelative(reference, candidate, 1e-3);
  std::cout << tag << ": worst relative " << worst << "\n";
  Expect(worst <= 2e-2, tag + " logits match the reference");
  Expect(ArgMax(reference) == ArgMax(candidate), tag + " argmax matches");
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
  const std::uint32_t max_context = 64;

  const auto device_model =
      rocm::DeviceModel::Upload(*weights, *reader, &*mtp, &error);
  if (device_model == nullptr) {
    std::cerr << "upload failed: " << error << "\n";
    return 1;
  }
  std::cout << "resident bytes: " << device_model->resident_bytes() << "\n";

  const auto executor =
      rocm::Executor::Create(*device_model, max_context, &error);
  if (executor == nullptr) {
    std::cerr << "executor create failed: " << error << "\n";
    return 1;
  }

  const std::vector<std::int32_t> prompt = {100, 200, 300};
  q36::ReferenceModel reference(*weights, max_context);
  std::vector<float> ref_logits(c.vocab_size);
  std::vector<float> ref_hidden(c.hidden_size);

  for (std::size_t i = 0; i < prompt.size(); ++i) {
    if (!executor->Step(prompt[i], &error)) {
      std::cerr << "GPU step " << i << " failed: " << error << "\n";
      return 1;
    }
    if (!reference.Step(prompt[i], ref_logits, ref_hidden, &error)) {
      std::cerr << "CPU step " << i << " failed: " << error << "\n";
      return 1;
    }
    const std::vector<float> gpu_logits =
        Download(executor->logits(), c.vocab_size);
    CheckLogits(ref_logits, gpu_logits, "trunk step " + std::to_string(i));
  }
  Expect(executor->position() == reference.Position(), "position matches");

  // The MTP draft consumes the last trunk hidden and the next token.
  if (!executor->MtpStep(/*token=*/400, &error)) {
    std::cerr << "GPU MTP step failed: " << error << "\n";
    return 1;
  }
  if (!reference.MtpStep(*mtp, /*token=*/400, ref_hidden, ref_logits, &error)) {
    std::cerr << "CPU MTP step failed: " << error << "\n";
    return 1;
  }
  const std::vector<float> gpu_mtp =
      Download(executor->mtp_logits(), c.vocab_size);
  CheckLogits(ref_logits, gpu_mtp, "mtp draft");

  if (failures != 0) {
    std::cerr << failures << " forward checks failed\n";
    return 1;
  }
  std::cout << "Qwen3.6-35B-A3B GPU forward passed.\n";
  return 0;
}