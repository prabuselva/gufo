#ifndef GUFO_MODELS_QWEN36_A3B_CLI_RUNNER_HPP_
#define GUFO_MODELS_QWEN36_A3B_CLI_RUNNER_HPP_

#include <chrono>
#include <memory>
#include <span>
#include <string>

#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen36_a3b/engine.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::cli {
struct BenchOptions;
struct PromptOptions;
}  // namespace gufo::cli

namespace gufo::models::qwen36_a3b {

#if defined(ENGINE_ENABLE_HIP)

bool IsQwen36A3B(const core::GgufReader& reader);

int RunQwen36A3BBenchmark(
    const cli::BenchOptions& options,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start);

std::shared_ptr<Model> LoadQwen36A3BModel(
    const cli::PromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point load_start);

int GenerateQwen36A3BResponse(const cli::PromptOptions& opt, const Model& model,
                              Session& session,
                              std::span<const tokenization::TokenId> prompt,
                              std::string* reply = nullptr);

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_CLI_RUNNER_HPP_