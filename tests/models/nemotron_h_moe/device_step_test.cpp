// Device-side step parity: the GPU executor versus the scalar oracle on the
// real 24GB artifact. Same prompt through both models, token by token; the
// logits must agree to f32 noise and the argmax token must be identical.
// Skips (77) unless GUFO_NEMOTRON_H_MOE_GGUF points at the artifact.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/nemotron_h_moe/kernels/rocm/device_model.hpp"
#include "src/models/nemotron_h_moe/reference.hpp"

namespace {

using namespace gufo::models;

[[nodiscard]] std::size_t Argmax(const std::vector<float>& v) {
  return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) -
                                  v.begin());
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_NEMOTRON_H_MOE_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: set GUFO_NEMOTRON_H_MOE_GGUF for the real artifact\n";
    return 77;
  }
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::cerr << "cannot open " << path << ": " << error << "\n";
    return 1;
  }
  const auto weights = nemotron_h_moe::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    std::cerr << "trunk bind failed: " << error << "\n";
    return 1;
  }
  const auto& c = weights->config;
  nemotron_h_moe::ReferenceModel reference(*weights, 8);
  nemotron_h_moe::DeviceModel device(*weights, 8);
  if (!device.Init(&error)) {
    std::cerr << "device init failed: " << error << "\n";
    return 1;
  }
  std::vector<float> dump_ref;
  std::vector<float> dump_gpu;
  const char* dump_path = std::getenv("GUFO_NEMOTRON_LAYERDUMP");
  const bool dumping = dump_path != nullptr && dump_path[0] != '\0';
  if (dumping) {
    reference.SetLayerDump(&dump_ref);
    device.SetLayerDump(&dump_gpu);
  }
  std::vector<float> logits_ref(c.vocab_size);
  std::vector<float> logits_gpu(c.vocab_size);
  const std::vector<std::int32_t> prompt = {100, 200, 300};
  bool ok = true;
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    if (!reference.Step(prompt[i], logits_ref, &error)) {
      std::cerr << "reference step " << i << ": " << error << "\n";
      return 1;
    }
    if (!device.Step(prompt[i], logits_gpu, &error)) {
      std::cerr << "device step " << i << ": " << error << "\n";
      return 1;
    }
    const std::size_t arg_ref = Argmax(logits_ref);
    const std::size_t arg_gpu = Argmax(logits_gpu);
    double scale = 0.0;
    double worst = 0.0;
    for (std::size_t j = 0; j < logits_ref.size(); ++j) {
      scale = std::max(scale, std::abs(static_cast<double>(logits_ref[j])));
      worst = std::max(worst, std::abs(static_cast<double>(logits_ref[j]) -
                                       static_cast<double>(logits_gpu[j])));
    }
    const double rel = worst / std::max(scale, 1e-30);
    const bool step_ok = arg_ref == arg_gpu && rel < 0.02;
    std::printf(
        "device step %zu  %s  argmax ref=%zu gpu=%zu  rel_logits=%.3e\n", i,
        step_ok ? "PASS" : "FAIL", arg_ref, arg_gpu, rel);
    ok = ok && step_ok;
    if (dumping && i == 0) {
      const std::string base(dump_path);
      std::ofstream(base + ".ref", std::ios::binary | std::ios::trunc)
          .write(reinterpret_cast<const char*>(dump_ref.data()),
                 static_cast<std::streamsize>(dump_ref.size() * sizeof(float)));
      std::ofstream(base + ".gpu", std::ios::binary | std::ios::trunc)
          .write(reinterpret_cast<const char*>(dump_gpu.data()),
                 static_cast<std::streamsize>(dump_gpu.size() * sizeof(float)));
    }
  }
  // A batched prefill of the same prompt must yield the step-by-step final
  // distribution, and leave the state identical for the next decode step.
  device.Reset();
  reference.Reset();
  std::vector<float> logits_batch(c.vocab_size);
  for (const std::int32_t tok : prompt) {
    if (!reference.Step(tok, logits_ref, &error)) {
      std::cerr << "reference rebuild step: " << error << "\n";
      return 1;
    }
  }
  if (!device.Process(prompt, logits_batch, &error)) {
    std::cerr << "device process: " << error << "\n";
    return 1;
  }
  {
    const std::size_t arg_ref = Argmax(logits_ref);
    const std::size_t arg_gpu = Argmax(logits_batch);
    double scale = 0.0;
    double worst = 0.0;
    for (std::size_t j = 0; j < logits_ref.size(); ++j) {
      scale = std::max(scale, std::abs(static_cast<double>(logits_ref[j])));
      worst = std::max(worst, std::abs(static_cast<double>(logits_ref[j]) -
                                       static_cast<double>(logits_batch[j])));
    }
    const double rel = worst / std::max(scale, 1e-30);
    const bool batch_ok = arg_ref == arg_gpu && rel < 0.02;
    std::printf(
        "batched prefill     %s  argmax ref=%zu gpu=%zu  "
        "rel_logits=%.3e\n",
        batch_ok ? "PASS" : "FAIL", arg_ref, arg_gpu, rel);
    ok = ok && batch_ok;
  }
  // Decode from the prefilled prompt must match a step-by-step continuation.
  if (!reference.Step(prompt[2], logits_ref, &error) ||
      !device.Step(prompt[2], logits_batch, &error)) {
    std::cerr << "post-prefill step failed: " << error << "\n";
    return 1;
  }
  {
    const bool cont_ok = Argmax(logits_ref) == Argmax(logits_batch);
    std::printf("prefill + decode    %s  argmax ref=%zu gpu=%zu\n",
                cont_ok ? "PASS" : "FAIL", Argmax(logits_ref),
                Argmax(logits_batch));
    ok = ok && cont_ok;
  }
  // Reset must replay the identical first-step distribution.
  device.Reset();
  reference.Reset();
  if (!reference.Step(prompt[0], logits_ref, &error) ||
      !device.Step(prompt[0], logits_gpu, &error)) {
    std::cerr << "reset replay step failed: " << error << "\n";
    return 1;
  }
  const bool replay_ok = Argmax(logits_ref) == Argmax(logits_gpu);
  std::printf("reset replay        %s  argmax ref=%zu gpu=%zu\n",
              replay_ok ? "PASS" : "FAIL", Argmax(logits_ref),
              Argmax(logits_gpu));
  ok = ok && replay_ok;
  if (!ok) {
    std::cerr << "device step parity failures\n";
    return 1;
  }
  std::cout << "device step parity passed.\n";
  return 0;
}
