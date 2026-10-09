// Greedy-decodes a short continuation twice over the real Ling 3.0 Flash
// artifact: once through the scalar oracle and once through the ROCm
// executor, feeding each argmax back as the next token. The token streams
// must agree exactly at every step; a single divergence means the GPU path
// is not yet quality-safe even when the forward test's logit tolerance
// passes. Skips (77) unless GUFO_LING3_FLASH_GGUF points at a Ling 3.0 Flash
// GGUF.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
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

// The winning id plus the gap to the runner-up, so a divergence reports how
// close the contest was.
std::int32_t ArgMax(const std::vector<float>& v, float* gap) {
  std::uint32_t best = 0;
  float second = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[best]) {
      second = v[best];
      best = static_cast<std::uint32_t>(i);
    } else if (v[i] > second) {
      second = v[i];
    }
  }
  *gap = v[best] - second;
  return static_cast<std::int32_t>(best);
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
  constexpr int kGenerated = 16;

  std::vector<float> cpu_logits(c.vocab_size);
  std::vector<float> cpu_hidden(c.hidden_size);
  for (const std::int32_t token : prompt) {
    if (!oracle.Step(token, cpu_logits, cpu_hidden, &error)) {
      std::cerr << "oracle prefill failed: " << error << "\n";
      return 1;
    }
  }
  if (!executor->Prefill(*session, prompt.data(),
                         static_cast<std::uint32_t>(prompt.size()), &error)) {
    std::cerr << "executor prefill failed: " << error << "\n";
    return 1;
  }

  std::vector<std::int32_t> cpu_stream;
  std::vector<std::int32_t> gpu_stream;
  for (int i = 0; i < kGenerated; ++i) {
    float gap = 0.0F;
    const auto gpu_logits = DownloadRaw(executor->logits(), c.vocab_size);
    const std::int32_t next_cpu = ArgMax(cpu_logits, &gap);
    const std::int32_t next_gpu = ArgMax(gpu_logits, &gap);
    cpu_stream.push_back(next_cpu);
    gpu_stream.push_back(next_gpu);
    if (next_cpu != next_gpu) {
      std::cerr << "divergence at step " << i << ": cpu " << next_cpu << " gpu "
                << next_gpu << " (cpu top-2 gap " << gap << ")\n";
      break;
    }
    if (!oracle.Step(next_cpu, cpu_logits, cpu_hidden, &error)) {
      std::cerr << "oracle decode failed: " << error << "\n";
      return 1;
    }
    if (!executor->Step(*session, next_gpu, &error)) {
      std::cerr << "executor decode failed: " << error << "\n";
      return 1;
    }
  }

  Expect(cpu_stream.size() == gpu_stream.size(), "stream lengths match");
  Expect(cpu_stream == gpu_stream, "greedy token streams agree exactly");
  std::cout << "cpu stream:";
  for (const std::int32_t t : cpu_stream) {
    std::cout << " " << t;
  }
  std::cout << "\ngpu stream:";
  for (const std::int32_t t : gpu_stream) {
    std::cout << " " << t;
  }
  std::cout << "\n";

  if (failures != 0) {
    std::cerr << failures << " inference parity checks failed\n";
    return 1;
  }
  std::cout << "Ling 3.0 Flash GPU greedy decode matches the oracle.\n";
  return 0;
}