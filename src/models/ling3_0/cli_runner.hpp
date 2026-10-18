#ifndef GUFO_MODELS_LING3_FLASH_CLI_RUNNER_HPP_
#define GUFO_MODELS_LING3_FLASH_CLI_RUNNER_HPP_

#include <chrono>
#include <memory>

#include "src/core/reasoning.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::cli {
struct BenchOptions;
struct PromptOptions;
}  // namespace gufo::cli

namespace gufo::models::ling3_0 {

bool IsLing3Model(const core::GgufReader& reader);

#if defined(ENGINE_ENABLE_HIP)

/// Runs `gufo prompt` for a Ling 3.0 Flash (bailingmoe3) GGUF. The Bailing V3
/// framing belongs to this model, so the caller passes the resolved reasoning
/// options instead of a shared-template rendering.
int RunLing3Prompt(const cli::PromptOptions& opt,
                   const core::GgufReader& reader,
                   std::chrono::steady_clock::time_point load_start,
                   const ReasoningOptions& reasoning);

/// Runs `gufo chat` interactive mode for a Ling 3.0 Flash GGUF, re-rendering
/// the accumulated history through the Bailing V3 template on every turn.
int RunLing3Chat(const cli::PromptOptions& opt, const core::GgufReader& reader,
                 std::chrono::steady_clock::time_point load_start,
                 const ReasoningOptions& reasoning);

/// Runs `gufo bench` for a Ling 3.0 Flash GGUF: prefill and generation
/// sweeps over the requested prompt/generation lengths, with optional MTP
/// speculative decoding.
int RunLing3Benchmark(const cli::BenchOptions& options,
                      const std::shared_ptr<const core::GgufReader>& reader,
                      std::chrono::steady_clock::time_point model_load_start);

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::models::ling3_0

#endif  // GUFO_MODELS_LING3_FLASH_CLI_RUNNER_HPP_
