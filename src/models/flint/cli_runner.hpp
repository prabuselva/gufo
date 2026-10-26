#ifndef GUFO_MODELS_FLINT_CLI_RUNNER_HPP_
#define GUFO_MODELS_FLINT_CLI_RUNNER_HPP_

#include <chrono>
#include <memory>

#include "src/models/flint/engine.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::cli {
struct BenchOptions;
}

namespace gufo::models::flint {

#if defined(ENGINE_ENABLE_HIP)

bool IsFlint(const core::GgufReader& reader);

int RunFlintBenchmark(const cli::BenchOptions& options,
                      const std::shared_ptr<const core::GgufReader>& reader,
                      std::chrono::steady_clock::time_point model_load_start);

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_CLI_RUNNER_HPP_
