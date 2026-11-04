#ifndef GUFO_MODELS_GEMMA4_CLI_RUNNER_HPP_
#define GUFO_MODELS_GEMMA4_CLI_RUNNER_HPP_

#include <chrono>
#include <memory>

namespace gufo::core {
class GgufReader;
}

namespace gufo::ext {
struct TextBenchOptions;
struct TextPromptOptions;
}  // namespace gufo::ext

namespace gufo::models::gemma4 {

bool IsGemma4(const core::GgufReader& reader);

#if defined(ENGINE_ENABLE_HIP)

int RunGemma4Benchmark(const ext::TextBenchOptions& options,
                       const std::shared_ptr<const core::GgufReader>& reader,
                       std::chrono::steady_clock::time_point model_load_start);

int RunGemma4Prompt(const ext::TextPromptOptions& opt,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point load_start);

int RunGemma4Chat(const ext::TextPromptOptions& opt,
                  const core::GgufReader& reader,
                  std::chrono::steady_clock::time_point load_start);

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_CLI_RUNNER_HPP_