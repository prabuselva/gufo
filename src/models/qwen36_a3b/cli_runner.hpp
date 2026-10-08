#ifndef GUFO_MODELS_QWEN36_A3B_CLI_RUNNER_HPP_
#define GUFO_MODELS_QWEN36_A3B_CLI_RUNNER_HPP_

#include <chrono>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen36_a3b/engine.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::qwen::vision {
class Encoder;
struct Prompt;
}  // namespace gufo::models::qwen::vision

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

/// Callbacks into the shared prompt.cpp pipeline for image handling.
using AttachImagesFn = void (*)(const cli::PromptOptions&,
                                tokenization::ChatMessage&);
using PrepareVisionFn = std::shared_ptr<const qwen::vision::Prompt> (*)(
    const cli::PromptOptions&, const tokenization::QwenTokenizer&,
    std::span<const tokenization::ChatMessage>,
    const std::shared_ptr<qwen::vision::Encoder>&);

/// Runs `gufo prompt` for a Qwen3.6-35B-A3B GGUF using the shared rendered
/// prompt and chat messages built by the caller.
int RunQwen36A3BPrompt(const cli::PromptOptions& opt,
                       const core::GgufReader& reader,
                       std::chrono::steady_clock::time_point load_start,
                       const std::string& rendered_prompt,
                       std::vector<tokenization::ChatMessage>& messages,
                       AttachImagesFn attach_images,
                       PrepareVisionFn prepare_vision);

/// Loads the model and creates the MTP-enabled session used by `gufo chat`.
/// Returns false after printing the failure reason.
bool LoadQwen36A3BChat(const cli::PromptOptions& opt,
                       const core::GgufReader& reader,
                       std::chrono::steady_clock::time_point load_start,
                       std::shared_ptr<Model>* model,
                       std::unique_ptr<Session>* session);

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::models::qwen36_a3b

#endif  // GUFO_MODELS_QWEN36_A3B_CLI_RUNNER_HPP_