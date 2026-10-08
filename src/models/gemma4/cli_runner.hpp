#ifndef GUFO_MODELS_GEMMA4_CLI_RUNNER_HPP_
#define GUFO_MODELS_GEMMA4_CLI_RUNNER_HPP_

#include <chrono>
#include <memory>

namespace gufo::core {
class GgufReader;
}

namespace gufo::cli {
struct BenchOptions;
struct PromptOptions;
}  // namespace gufo::cli

namespace gufo::models::gemma4 {

bool IsGemma4(const core::GgufReader& reader);

#if defined(ENGINE_ENABLE_HIP)

int RunGemma4Benchmark(const cli::BenchOptions& options,
                       const std::shared_ptr<const core::GgufReader>& reader,
                       std::chrono::steady_clock::time_point model_load_start);

int RunGemma4Prompt(const cli::PromptOptions& opt,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point load_start);

int RunGemma4Chat(const cli::PromptOptions& opt, const core::GgufReader& reader,
                  std::chrono::steady_clock::time_point load_start);

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_CLI_RUNNER_HPP_