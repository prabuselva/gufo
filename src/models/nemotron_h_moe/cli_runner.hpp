#ifndef GUFO_MODELS_NEMOTRON_H_MOE_CLI_RUNNER_HPP_
#define GUFO_MODELS_NEMOTRON_H_MOE_CLI_RUNNER_HPP_

#include <chrono>
#include <memory>

#include "src/models/nemotron_h_moe/engine.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::cli {
struct BenchOptions;
}

namespace gufo::models::nemotron_h_moe {

#if defined(ENGINE_ENABLE_HIP)

bool IsNemotronHBolt(const core::GgufReader& reader);

int RunNemotronBenchmark(
    const cli::BenchOptions& options,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start);

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::models::nemotron_h_moe

#endif  // GUFO_MODELS_NEMOTRON_H_MOE_CLI_RUNNER_HPP_
