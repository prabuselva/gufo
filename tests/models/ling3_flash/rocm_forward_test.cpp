// Runs the ROCm executor and the scalar oracle over the same tokens on the
// real Ling 3.0 artifact (Flash or Tiny) and requires them to agree: trunk
// logits, the post-output_norm hidden stream, chunked Prefill at block
// boundaries, the MTP draft logits when the artifact ships a draft, and a
// replay after Reset. This is the end-to-end numerical contract for the GPU
// path; the per-operator tests only pin the pieces.
// Skips (77) unless GUFO_LING3_FLASH_GGUF points at a Ling 3.0 GGUF.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/ling3_flash/kernels/rocm/executor.hpp"
#include "src/models/ling3_flash/reference.hpp"
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

std::vector<float> DownloadRaw(const float* device, std::size_t count) {
  std::vector<float> values(count);
  ht::CheckHip(hipMemcpy(values.data(), device, count * sizeof(float),
                         hipMemcpyDeviceToHost),
               "download");
  return values;
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

// A 42-layer float32 forward with different reduction orders drifts more than
// a single operator; the observed worst case is 3e-3 (MTP logits), so 1e-2
// keeps a 3x margin far below anything that could flip a sampled token.
constexpr double kRelTol = 1e-2;

// Chunked Prefill feeds the F16 WMMA tier. The tier kernels themselves are
// gated against the reference at the unit level; end to end the per-layer
// F16 input rounding (~2e-4) is amplified by the KDA state update, so the
// worst elementwise deviation is ~0.16 relative to the real (order-10) logit
// and hidden scale while the argmax still matches at every boundary. Gate
// that path with a scale-referenced metric (floor 1.0); the F32 per-token
// path keeps the small-value metric above.
constexpr double kTierFloor = 1.0;
constexpr double kTierRelTol = 0.25;

void Compare(const std::vector<float>& cpu, const std::vector<float>& gpu,
             const std::string& tag, double floor = 1e-3,
             double tol = kRelTol) {
  const double rel = ht::WorstRelative(cpu, gpu, floor);
  std::cout << tag << ": worst relative " << rel << "\n";
  Expect(rel <= tol, tag + " within tolerance");
  bool finite = true;
  for (const float v : gpu) {
    if (!std::isfinite(v)) {
      finite = false;
      break;
    }
  }
  Expect(finite, tag + " gpu values are finite");
}

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
  const auto weights = l3::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    std::cerr << "trunk bind failed: " << error << "\n";
    return 1;
  }
  const bool has_mtp = weights->config.nextn_layers > 0;
  std::optional<l3::MtpWeights> mtp;
  if (has_mtp) {
    auto bound = l3::MtpWeights::Bind(*reader, weights->config, &error);
    if (!bound.has_value()) {
      std::cerr << "MTP bind failed: " << error << "\n";
      return 1;
    }
    mtp = std::move(bound);
  }

  const auto device = rocm::DeviceModel::Upload(
      *weights, *reader, has_mtp ? &*mtp : nullptr, &error);
  if (device == nullptr) {
    std::cerr << "device upload failed: " << error << "\n";
    return 1;
  }
  const auto executor = rocm::Executor::Create(*device, 64, &error);
  if (executor == nullptr) {
    std::cerr << "executor create failed: " << error << "\n";
    return 1;
  }
  const auto session = executor->CreateSession(64, &error);
  if (session == nullptr) {
    std::cerr << "session create failed: " << error << "\n";
    return 1;
  }

  const auto& c = weights->config;
  l3::ReferenceModel oracle(*weights, 64);
  const std::vector<std::int32_t> prompt = {100, 200, 300};
  std::vector<float> cpu_logits(c.vocab_size);
  std::vector<float> cpu_hidden(c.hidden_size);

  for (std::size_t i = 0; i < prompt.size(); ++i) {
    if (!oracle.Step(prompt[i], cpu_logits, cpu_hidden, &error)) {
      std::cerr << "oracle step " << i << " failed: " << error << "\n";
      return 1;
    }
    if (!executor->Step(*session, prompt[i], &error)) {
      std::cerr << "executor step " << i << " failed: " << error << "\n";
      return 1;
    }
    const auto gpu_logits = DownloadRaw(executor->logits(), c.vocab_size);
    const auto gpu_hidden = DownloadRaw(executor->h_out(), c.hidden_size);
    Compare(cpu_logits, gpu_logits, "step " + std::to_string(i) + " logits");
    Compare(cpu_hidden, gpu_hidden, "step " + std::to_string(i) + " hidden");
    Expect(ArgMax(cpu_logits) == ArgMax(gpu_logits),
           "step " + std::to_string(i) + " argmax matches");
  }
  Expect(session->position() == static_cast<std::uint32_t>(prompt.size()),
         "session position advanced");

  if (has_mtp) {
    std::vector<float> cpu_mtp(c.vocab_size);
    if (!oracle.MtpStep(*mtp, /*token=*/400, cpu_hidden, cpu_mtp, &error)) {
      std::cerr << "oracle mtp failed: " << error << "\n";
      return 1;
    }
    if (!executor->MtpStep(*session, /*token=*/400, &error)) {
      std::cerr << "executor mtp failed: " << error << "\n";
      return 1;
    }
    const auto gpu_mtp = DownloadRaw(executor->mtp_logits(), c.vocab_size);
    Compare(cpu_mtp, gpu_mtp, "mtp logits");
    Expect(ArgMax(cpu_mtp) == ArgMax(gpu_mtp), "mtp argmax matches");
  }

  // After Reset both must replay the first token identically again.
  oracle.Reset();
  session->Reset();
  Expect(session->position() == 0, "reset clears position");
  if (!oracle.Step(prompt[0], cpu_logits, cpu_hidden, &error) ||
      !executor->Step(*session, prompt[0], &error)) {
    std::cerr << "replay step failed: " << error << "\n";
    return 1;
  }
  const auto replay_logits = DownloadRaw(executor->logits(), c.vocab_size);
  Compare(cpu_logits, replay_logits, "replay logits");

  // Chunked Prefill must land where the same prompt driven one Step at a
  // time lands, at every block boundary: the KDA recurrence, the MLA caches
  // and the position have to carry across separate Prefill calls.
  oracle.Reset();
  session->Reset();
  std::vector<std::int32_t> long_prompt;
  long_prompt.reserve(40);
  for (std::uint32_t i = 0; i < 40; ++i) {
    long_prompt.push_back(static_cast<std::int32_t>(1 + (i * 37U) % 1000U));
  }

  const std::uint32_t blocks[] = {1, 16, 22, 1};
  std::size_t done = 0;
  for (const std::uint32_t block : blocks) {
    if (!executor->Prefill(*session, long_prompt.data() + done, block,
                           &error)) {
      std::cerr << "prefill block " << block << " failed: " << error << "\n";
      return 1;
    }
    for (std::uint32_t i = 0; i < block; ++i) {
      if (!oracle.Step(long_prompt[done + i], cpu_logits, cpu_hidden, &error)) {
        std::cerr << "oracle prefill step failed: " << error << "\n";
        return 1;
      }
    }
    done += block;
    const std::string tag = "prefill through " + std::to_string(done);
    const auto gpu_logits = DownloadRaw(executor->logits(), c.vocab_size);
    const auto gpu_hidden = DownloadRaw(executor->h_out(), c.hidden_size);
    Compare(cpu_logits, gpu_logits, tag + " logits", kTierFloor, kTierRelTol);
    Compare(cpu_hidden, gpu_hidden, tag + " hidden", kTierFloor, kTierRelTol);
    Expect(ArgMax(cpu_logits) == ArgMax(gpu_logits), tag + " argmax matches");
    Expect(session->position() == done, tag + " position advanced");
  }

  if (failures != 0) {
    std::cerr << failures << " forward parity checks failed\n";
    return 1;
  }
  std::cout << "Ling 3.0 GPU forward matches the oracle.\n";
  return 0;
}