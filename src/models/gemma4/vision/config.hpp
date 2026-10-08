#ifndef GUFO_MODELS_GEMMA4_VISION_CONFIG_HPP_
#define GUFO_MODELS_GEMMA4_VISION_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gemma4::vision {

/// Architecture parameters of the Gemma-4 vision tower (`clip` GGUF with the
/// `gemma4v` projector). A 27-block CLIP-style ViT over 16x16 patches, hidden
/// 1152, 16 heads of width 72, with per-head Q/K RMSNorm, weightless V norm,
/// 2D RoPE (theta 100, x over dims [0,36), y over [36,72)), a 3x3 average-pool
/// merger and a clippable linear projection to the trunk width (2816). Every
/// value is read from the mmproj file; `FromGguf` locks the geometry this
/// runtime was written and tested for.
struct Config {
  std::uint32_t projection_dim{0};       ///< Output width to the trunk, 2816.
  std::uint32_t image_size{0};           ///< Reference square side, 224.
  std::uint32_t patch_size{0};           ///< 16.
  std::uint32_t embedding_length{0};     ///< Vision hidden, 1152.
  std::uint32_t feed_forward_length{0};  ///< 4304.
  std::uint32_t block_count{0};          ///< 27.
  std::uint32_t head_count{0};           ///< 16.
  float layer_norm_epsilon{1e-6F};
  std::vector<float> image_mean;  ///< 3 entries, all 0.
  std::vector<float> image_std;   ///< 3 entries, all 1.

  /// Fixed by the `gemma4v` projector, not stored in the file.
  static constexpr std::uint32_t kMergeSize = 3;
  static constexpr float kRopeTheta = 100.0F;

  [[nodiscard]] std::uint32_t HeadDim() const noexcept {
    return head_count == 0 ? 0 : embedding_length / head_count;
  }
  /// Resize grid is a multiple of `patch_size * kMergeSize` so the merger's
  /// 3x3 stride-3 pool divides the patch grid exactly.
  [[nodiscard]] std::uint32_t AlignSize() const noexcept {
    return patch_size * kMergeSize;
  }
  /// Half of the head width rotated per RoPE pass (x then y).
  [[nodiscard]] std::uint32_t RopeHalfDim() const noexcept {
    return HeadDim() / 2;
  }

  /// Reads and validates `clip.vision.*` metadata for the `gemma4v`
  /// projector.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::gemma4::vision

#endif  // GUFO_MODELS_GEMMA4_VISION_CONFIG_HPP_