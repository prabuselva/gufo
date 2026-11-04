#ifndef GUFO_TESTS_MODELS_NEMOTRON_H_MOE_HIP_TEST_HPP_
#define GUFO_TESTS_MODELS_NEMOTRON_H_MOE_HIP_TEST_HPP_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Shared scaffolding for the nemotron_h_moe GPU operator checks: a RAII
// device buffer, a deterministic xorshift generator, host/device transfer
// helpers and error reductions. Each check computes its own CPU reference
// from the scalar oracle's formulas (or the oracle's own dequantizers for
// the model-private IQ2_XXS and NVFP4 layouts) and compares it against the
// kernel output.
namespace gufo::tests::nemotron_h_moe {

inline void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t count() const noexcept { return count_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

inline std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

inline std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                                     float scale, float offset = 0.0F) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = offset +
            scale *
                static_cast<float>(
                    static_cast<int>(NextRandom(&seed) & 0xFFFFU) - 32768) /
                32768.0F;
  }
  return values;
}

inline std::vector<std::uint8_t> MakeBytes(std::size_t count,
                                           std::uint32_t seed) {
  std::vector<std::uint8_t> bytes(count);
  for (std::uint8_t& byte : bytes) {
    byte = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFFU);
  }
  return bytes;
}

inline void Upload(void* destination, const void* source, std::size_t bytes) {
  CheckHip(hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice),
           "upload");
}

inline void Upload(HipBuffer<float>* destination,
                   const std::vector<float>& source) {
  Upload(destination->get(), source.data(), destination->bytes());
}

template<typename T>
inline std::vector<T> Download(HipBuffer<T>* source) {
  std::vector<T> values(source->count());
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

// Worst relative error over the elementwise scale floor, so near-zero
// reference values do not inflate the metric.
inline double WorstRelative(const std::vector<float>& reference,
                            const std::vector<float>& candidate,
                            double floor = 1e-3) {
  double worst = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double scale =
        std::max(floor, std::abs(static_cast<double>(reference[i])));
    worst = std::max(
        worst,
        std::abs(static_cast<double>(reference[i]) - candidate[i]) / scale);
  }
  return worst;
}

// Worst absolute difference. Used where the reference is bounded and float
// transcendental rounding, not relative magnitude, is the only source of
// deviation.
inline double WorstAbsolute(const std::vector<float>& reference,
                            const std::vector<float>& candidate) {
  double worst = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    worst = std::max(
        worst, std::abs(static_cast<double>(reference[i]) - candidate[i]));
  }
  return worst;
}

// The scalar oracle's transcendental helpers, evaluated in double so the
// reference is at least as accurate as the kernel's float approximations.
inline double SigmoidD(double x) noexcept {
  return 1.0 / (1.0 + std::exp(-x));
}
inline double ReluSqrD(double x) noexcept {
  return x <= 0.0 ? 0.0 : x * x;
}

}  // namespace gufo::tests::nemotron_h_moe

#endif  // GUFO_TESTS_MODELS_NEMOTRON_H_MOE_HIP_TEST_HPP_
