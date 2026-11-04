#ifndef GUFO_STAGING_INTEGRATION_STAGING_CLI_HPP_
#define GUFO_STAGING_INTEGRATION_STAGING_CLI_HPP_

#include <chrono>
#include <memory>
#include <optional>

#include "gufo/ext/text.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::cli {

class ArgParser;

// Staging model packages predate the extension registry; when no
// provider claims the architecture, the `gufo prompt` flow is served here.
// Returns nullopt when no staging model claims the architecture; the value
// is the process exit code otherwise.
std::optional<int> StagingRunPrompt(
    const ext::TextPromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point model_load_start);

// Serves the `gufo chat` flow for staging models; same claiming rule as
// StagingRunPrompt.
std::optional<int> StagingRunChat(
    const ext::TextPromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point model_load_start);

// Serves the `gufo bench` flow for staging models; same claiming rule as
// StagingRunPrompt.
std::optional<int> StagingRunBenchmark(
    const ext::TextBenchOptions& opt,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start);

// Registers the staging-model-only CLI options in the shared option tables.
void RegisterStagingTextOptions(ArgParser& parser,
                                ext::TextPromptOptions& opt);

}  // namespace gufo::cli

#endif  // GUFO_STAGING_INTEGRATION_STAGING_CLI_HPP_
