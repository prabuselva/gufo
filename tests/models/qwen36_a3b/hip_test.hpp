#ifndef GUFO_TESTS_MODELS_QWEN36_A3B_HIP_TEST_HPP_
#define GUFO_TESTS_MODELS_QWEN36_A3B_HIP_TEST_HPP_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Shared scaffolding for the qwen36_a3b GPU operator checks: a RAII device
// buffer, a deterministic xorshift generator, host/device transfer helpers and
// a relative-error reduction. Each check computes its own CPU reference from
// the scalar oracle's formulas and compares it against the kernel output.
namespace gufo::tests::qwen36_a3b {

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

inline void Upload(HipBuffer<float>* destination,
                   const std::vector<float>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

inline std::vector<float> Download(HipBuffer<float>* source,
                                   std::size_t count) {
  std::vector<float> values(count);
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

// Worst absolute difference. Used where the reference is bounded (rotary
// outputs, gated sums) and float transcendental rounding, not relative
// magnitude, is the only source of deviation.
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
inline double SiluD(double x) noexcept {
  return x * SigmoidD(x);
}
inline double SoftplusD(double x) noexcept {
  return x > 20.0 ? x : std::log1p(std::exp(x));
}

}  // namespace gufo::tests::qwen36_a3b

#endif  // GUFO_TESTS_MODELS_QWEN36_A3B_HIP_TEST_HPP_