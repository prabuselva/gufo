#ifndef GUFO_MODELS_GEMMA4_SERVE_RUNNER_HPP_
#define GUFO_MODELS_GEMMA4_SERVE_RUNNER_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "src/cli/serve/inference_backend.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/models/gemma4/engine.hpp"

namespace gufo::models::gemma4 {

#if defined(ENGINE_ENABLE_HIP)

/// Validates the serve configuration for a loaded Gemma-4 model and creates
/// its text runner. Returns nullptr and sets `error` when the configuration
/// is unsupported.
[[nodiscard]] std::shared_ptr<server::TextModelRunner> CreateTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    std::size_t session_count,
    const server::TextSpeculativeConfig& speculative_config,
    const server::TextDiskCacheConfig& disk_cache_config, std::string* error);

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_SERVE_RUNNER_HPP_