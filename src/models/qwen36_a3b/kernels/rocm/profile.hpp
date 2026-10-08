#ifndef GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_PROFILE_HPP_
#define GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_PROFILE_HPP_

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace gufo::models::qwen36_a3b::rocm {

/// Env-gated (GUFO_QWEN36_PROFILE=1) wall-clock stage timer for the Qwen3.6
/// forward path. The executor launches every kernel on the default stream, so
/// a device synchronize at each stage boundary measures that stage's GPU time
/// with no overlap to attribute. When the variable is unset every method is a
/// no-op, so production runs are completely unaffected.
///
/// Usage is a single linear chain of Mark() calls: each Mark closes the open
/// segment (attributing its elapsed GPU time to the previous label) and opens a
/// new one. Report() closes the last segment, prints the accumulated breakdown
/// sorted by cost, and clears the counters.
class StageProfiler {
public:
  StageProfiler() {
    const char* e = std::getenv("GUFO_QWEN36_PROFILE");
    enabled_ = (e != nullptr && e[0] != '\0' && e[0] != '0');
  }

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }

  /// Closes the open segment and opens a new one named `label`. A null label
  /// only closes.
  void Mark(const char* label) {
    if (!enabled_) {
      return;
    }
    Close();
    if (label != nullptr) {
      label_ = label;
      t0_ = std::chrono::steady_clock::now();
    }
  }

  /// Drops any accumulated timings without printing (used at the start of a
  /// fresh prefill so a prior error path cannot leak stale rows).
  void Reset() {
    label_ = nullptr;
    ms_.clear();
    count_.clear();
  }

  /// Closes the open segment, prints the breakdown under `header`, and clears.
  void Report(const char* header) {
    if (!enabled_) {
      return;
    }
    Close();
    double total = 0.0;
    for (const auto& kv : ms_) {
      total += kv.second;
    }
    std::vector<std::pair<std::string, double>> rows(ms_.begin(), ms_.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::fprintf(stderr,
                 "\n===== qwen36_a3b profile: %s =====\n"
                 "  %-22s %11s %7s %8s %11s\n",
                 header, "stage", "total_ms", "pct", "calls", "avg_ms");
    for (const auto& row : rows) {
      const std::uint64_t n = count_[row.first];
      std::fprintf(stderr, "  %-22s %11.2f %6.1f%% %8llu %11.3f\n",
                   row.first.c_str(), row.second,
                   total > 0.0 ? 100.0 * row.second / total : 0.0,
                   static_cast<unsigned long long>(n),
                   n != 0 ? row.second / static_cast<double>(n) : 0.0);
    }
    std::fprintf(stderr, "  %-22s %11.2f\n\n", "TOTAL", total);
    std::fflush(stderr);
    ms_.clear();
    count_.clear();
  }

private:
  void Close() {
    if (label_ == nullptr) {
      return;
    }
    (void)hipDeviceSynchronize();
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0_)
                          .count();
    ms_[label_] += ms;
    count_[label_] += 1;
    label_ = nullptr;
  }

  bool enabled_{false};
  const char* label_{nullptr};
  std::chrono::steady_clock::time_point t0_{};
  std::map<std::string, double> ms_;
  std::map<std::string, std::uint64_t> count_;
};

}  // namespace gufo::models::qwen36_a3b::rocm

#endif  // GUFO_MODELS_QWEN36_A3B_KERNELS_ROCM_PROFILE_HPP_