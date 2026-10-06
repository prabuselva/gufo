// Runs the ROCm vision tower and the scalar oracle over the same deterministic
// 96x96 gradient and checks they agree to full-precision tolerance. The device
// path upcasts the BF16 mmproj weights to F32, so both paths compute in F32 and
// agree to rounding. Skips (77) without GUFO_GEMMA4_MMPROJ_GGUF; needs the GPU.
#include "src/models/gemma4/vision/encoder.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/gemma4/vision/reference.hpp"
#include "src/models/gemma4/vision/weights.hpp"

namespace g4v = gufo::models::gemma4::vision;
namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_GEMMA4_MMPROJ_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: set GUFO_GEMMA4_MMPROJ_GGUF\n";
    return 77;
  }
  std::string error;

  // Oracle reference output.
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::cerr << "cannot open " << path << ": " << error << "\n";
    return 1;
  }
  const auto weights = g4v::VisionWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    std::cerr << "vision bind failed: " << error << "\n";
    return 1;
  }

  const std::uint32_t nx = 96;
  const std::uint32_t ny = 96;
  std::vector<float> pixels(static_cast<std::size_t>(3) * nx * ny);
  for (std::uint32_t ch = 0; ch < 3; ++ch) {
    for (std::uint32_t y = 0; y < ny; ++y) {
      for (std::uint32_t x = 0; x < nx; ++x) {
        pixels[(ch * ny + y) * nx + x] =
            static_cast<float>((x + y + ch * 7) % 256) / 255.0F;
      }
    }
  }

  g4v::ReferenceEncoder oracle(*weights);
  std::vector<float> ref;
  if (!oracle.Encode(pixels.data(), nx, ny, ref, &error)) {
    std::cerr << "oracle encode failed: " << error << "\n";
    return 1;
  }

  // Device output.
  auto encoder = g4v::Encoder::Open(path, &error);
  if (encoder == nullptr) {
    std::cerr << "device encoder open failed: " << error << "\n";
    return 1;
  }
  std::vector<float> gpu;
  if (!encoder->Encode(pixels.data(), nx, ny, gpu, &error)) {
    std::cerr << "device encode failed: " << error << "\n";
    return 1;
  }

  Expect(gpu.size() == ref.size(), "output sizes match");
  double ref_max = 0.0;
  double max_abs = 0.0;
  double sum_abs = 0.0;
  double ref_sum = 0.0;
  double gpu_sum = 0.0;
  bool all_finite = true;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (!std::isfinite(gpu[i])) {
      all_finite = false;
    }
    const double d = std::abs(static_cast<double>(gpu[i]) - ref[i]);
    max_abs = std::max(max_abs, d);
    sum_abs += d;
    ref_max = std::max(ref_max, std::abs(static_cast<double>(ref[i])));
    ref_sum += ref[i];
    gpu_sum += gpu[i];
  }
  const double mean_abs = sum_abs / static_cast<double>(ref.size());
  std::cout << "oracle checksum: " << ref_sum << "\n";
  std::cout << "device checksum: " << gpu_sum << "\n";
  std::cout << "max abs diff: " << max_abs << " (ref max " << ref_max
            << "), mean abs diff: " << mean_abs << "\n";

  Expect(all_finite, "device output is finite");
  Expect(ref_max > 1e-3, "oracle output is non-trivial");
  // Both paths compute in F32; agree to floating-point rounding.
  Expect(max_abs <= 1e-3 * ref_max, "max diff within F32 rounding");
  Expect(std::abs(gpu_sum - ref_sum) <= 1e-3 * std::abs(ref_sum) + 1e-6,
         "checksum within F32 rounding");

  if (failures != 0) {
    std::cerr << failures << " vision parity checks failed\n";
    return 1;
  }
  std::cout << "Gemma-4 vision device parity passed.\n";
  return 0;
}