#ifndef GUFO_STAGING_INTEGRATION_STAGING_BACKEND_HPP_
#define GUFO_STAGING_INTEGRATION_STAGING_BACKEND_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "gufo/ext/text.hpp"
#include "src/cli/serve/inference_backend.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::server {

// A staging model loaded through its own serve seam, ready for the shared
// InferenceBackend to install.
struct StagingLoadedModel {
  std::shared_ptr<ext::TextModelRunner> runner = nullptr;
  /// True when a loaded `gemma4v` mmproj tower lets the HTTP layer accept
  /// image inputs; read before the model moves into the runner.
  bool supports_images{false};
};

// Staging model packages predate the extension registry; when no
// provider claims the architecture, build its runner here. Returns nullopt
// when no staging model claims the architecture; on a handled load failure
// it returns nullopt with `error` set.
std::optional<StagingLoadedModel> StagingLoadModel(
    const std::string& model_path, const core::GgufReader& reader,
    std::uint32_t max_context, std::size_t session_count,
    const TextSpeculativeConfig& speculative_config,
    const TextDiskCacheConfig& disk_cache_config,
    const std::string& vision_model_path, std::uint32_t attn_window,
    std::uint32_t attn_sink, std::string* error);

}  // namespace gufo::server

#endif  // GUFO_STAGING_INTEGRATION_STAGING_BACKEND_HPP_
