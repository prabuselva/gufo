// Runs the GPU executor over the real Qwen3.6-35B-A3B artifact and checks that
// its trunk and MTP logits agree with the scalar oracle to float32 rounding.
// This is the end-to-end numerical contract: every fused operator and the
// quantized GEMV tier, wired together, must reproduce the reference forward
// pass. Skips (77) unless GUFO_QWEN36_A3B_GGUF points at a Qwen3.6 GGUF.
#include <algorithm>
#include <chrono>
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

// Prefill-throughput sweep over the real executor. Drives Executor::Prefill at
// increasing prompt lengths (each runs as prefill_chunk_-sized chunks with the
// WMMA attention reading the growing prefix) and reports average tokens/second,
// which is the metric that exposed the O(n^2) prefill collapse. Gated by
// GUFO_QWEN36_A3B_BENCH (comma-separated lengths; a default ladder is used when
// the value is empty).
int RunPrefillBench(const rocm::DeviceModel& device_model,
                    const q36::Config& c) {
  const char* spec = std::getenv("GUFO_QWEN36_A3B_BENCH");
  std::vector<std::uint32_t> lengths;
  if (spec != nullptr) {
    std::string text(spec);
    std::size_t pos = 0;
    while (pos < text.size()) {
      const std::size_t comma = text.find(',', pos);
      const std::string field = text.substr(
          pos, comma == std::string::npos ? std::string::npos : comma - pos);
      if (!field.empty()) {
        lengths.push_back(static_cast<std::uint32_t>(std::stoul(field)));
      }
      if (comma == std::string::npos) {
        break;
      }
      pos = comma + 1;
    }
  }
  if (lengths.empty()) {
    lengths = {512U, 2048U, 8192U, 16384U, 32768U, 65536U, 100000U};
  }
  std::sort(lengths.begin(), lengths.end());
  const std::uint32_t max_len = lengths.back();

  std::string error;
  const auto executor = rocm::Executor::Create(device_model, max_len, &error);
  if (executor == nullptr) {
    std::cerr << "bench executor create failed (max_context=" << max_len
              << "): " << error << "\n";
    return 1;
  }
  const auto session = executor->CreateSession(max_len, &error);
  if (session == nullptr) {
    std::cerr << "bench session create failed (max_context=" << max_len
              << "): " << error << "\n";
    return 1;
  }
  std::vector<std::int32_t> tokens(max_len);
  for (std::uint32_t i = 0; i < max_len; ++i) {
    tokens[i] = static_cast<std::int32_t>(i % 1000U);
  }

  std::cout << "prefill throughput sweep (chunk="
            << (max_len < 2048U ? max_len : 2048U) << "):\n";
  for (const std::uint32_t n : lengths) {
    session->Reset();
    if (!executor->Prefill(*session, tokens.data(), n, &error)) {
      std::cerr << "prefill " << n << " failed: " << error << "\n";
      return 1;
    }
    (void)hipDeviceSynchronize();
    double best = 0.0;
    for (int rep = 0; rep < 3; ++rep) {
      session->Reset();
      const auto start = std::chrono::steady_clock::now();
      if (!executor->Prefill(*session, tokens.data(), n, &error)) {
        std::cerr << "prefill " << n << " failed: " << error << "\n";
        return 1;
      }
      (void)hipDeviceSynchronize();
      const double seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
      best = std::max(best, static_cast<double>(n) / seconds);
    }
    std::cout << "  n=" << n << ": " << best << " tok/s\n";
  }
  (void)c;
  return 0;
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

  if (std::getenv("GUFO_QWEN36_A3B_BENCH") != nullptr) {
    return RunPrefillBench(*device_model, c);
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

  const std::vector<std::int32_t> prompt = {100, 200, 300};
  q36::ReferenceModel reference(*weights, max_context);
  std::vector<float> ref_logits(c.vocab_size);
  std::vector<float> ref_hidden(c.hidden_size);

  for (std::size_t i = 0; i < prompt.size(); ++i) {
    if (!executor->Step(*session, prompt[i], &error)) {
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
  Expect(session->position() == reference.Position(), "position matches");

  // The MTP draft consumes the last trunk hidden and the next token.
  if (!executor->MtpStep(*session, /*token=*/400, &error)) {
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