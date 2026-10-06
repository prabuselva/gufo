#ifndef GUFO_MODELS_GEMMA4_VISION_ENCODER_HPP_
#define GUFO_MODELS_GEMMA4_VISION_ENCODER_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/models/gemma4/vision/config.hpp"

namespace gufo::models::gemma4::vision {

/// ROCm vision tower for `gemma4v`. Streams the mmproj weights to the device
/// once and runs the full encoder (patch embed, 27 blocks, merger) to produce
/// trunk-width image embeddings. Mirrors `ReferenceEncoder`'s contract: CHW
/// pixels in [0,1] with nx/ny multiples of patch*merge, output row-major
/// [n_tokens x projection_dim].
class Encoder {
 public:
  ~Encoder();
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;

  /// Opens the mmproj sidecar, binds and uploads every tower tensor.
  [[nodiscard]] static std::unique_ptr<Encoder> Open(
      const std::string& mmproj_path, std::string* error_msg);

  /// Runs the tower over one image. `pixels` is CHW, `3 * ny * nx` floats in
  /// [0,1]. `out` receives `n_tokens * projection_dim` floats, row-major.
  [[nodiscard]] bool Encode(const float* pixels, std::uint32_t nx,
                            std::uint32_t ny, std::vector<float>& out,
                            std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept;

 private:
  struct Impl;
  explicit Encoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::gemma4::vision

#endif  // GUFO_MODELS_GEMMA4_VISION_ENCODER_HPP_