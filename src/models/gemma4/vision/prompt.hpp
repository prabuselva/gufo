#ifndef GUFO_MODELS_GEMMA4_VISION_PROMPT_HPP_
#define GUFO_MODELS_GEMMA4_VISION_PROMPT_HPP_

#include <cstdint>
#include <string_view>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/gemma4/tokenizer.hpp"

namespace gufo::models::gemma4::vision {

/// Aspect-preserving resize policy for the `gemma4v` tower, matching the
/// reference `smart_resize` (llama.cpp `calc_size_preserved_ratio`): the
/// resized side lengths are multiples of `align_size` (patch * merge = 48) so
/// the merger's 3x3 stride-3 pool divides the patch grid, and the resized
/// pixel count is clamped into `[min_pixels, max_pixels]`. Gemma-4 locks the
/// token budget to 40..280 tokens, i.e. `min_pixels = 40 * 48^2` and
/// `max_pixels = 280 * 48^2` (patch area 2304).
struct ResizeLimits {
  std::uint32_t align_size{48};
  std::uint32_t min_pixels{40U * 48U * 48U};
  std::uint32_t max_pixels{280U * 48U * 48U};
};

struct Size {
  std::uint32_t width{0};
  std::uint32_t height{0};
};

/// The resized target for an `width` x `height` source under `limits`. Mirrors
/// the reference exactly: round up to the nearest `align_size` multiple, then
/// scale down (floor) if over `max_pixels` or up (ceil) if under `min_pixels`,
/// keeping the aspect ratio and never dropping below one `align_size` block.
[[nodiscard]] Size CalcSizePreservedRatio(std::uint32_t width,
                                          std::uint32_t height,
                                          const ResizeLimits& limits);

/// One image after resize and normalization, ready for the tower. `pixels` is
/// CHW, `3 * ny * nx` floats in `[0, 1]` (raw / 255; the tower applies the
/// `x * 2 - 1` itself). `nx` and `ny` are multiples of `align_size` and
/// `tokens == (nx / align_size) * (ny / align_size)`.
struct PreparedImage {
  std::vector<float> pixels;
  std::uint32_t nx{0};
  std::uint32_t ny{0};
  std::uint32_t tokens{0};
};

/// Bicubic-resizes `image` to `CalcSizePreservedRatio(...)` and converts the
/// RGB8 result to CHW floats in `[0, 1]`. A zero-sized or empty source yields
/// an empty `PreparedImage`.
[[nodiscard]] PreparedImage PrepareImage(const core::Image& image,
                                         const ResizeLimits& limits);

/// One image's placeholder span inside a built prompt plus the pixels the
/// encoder must run over. `offset` is the absolute index of the first filler
/// token and `count` is the number of filler tokens (the encoder's
/// `n_tokens`); the executor overwrites exactly those `count` embedding rows.
struct VisionSlot {
  std::uint32_t offset{0};
  std::uint32_t count{0};
  std::vector<float> pixels;
  std::uint32_t nx{0};
  std::uint32_t ny{0};
};

/// A tokenized prompt with every image marker expanded into its placeholder
/// span, plus the per-image slots needed to fill them.
struct PreparedPrompt {
  std::vector<Tokenizer::TokenId> tokens;
  std::vector<VisionSlot> images;
};

/// Expands a rendered chat prompt into token ids, replacing each `<|image|>`
/// marker with `<|image>` + `n_tokens` filler `<|image|>` tokens + `<image|>`
/// and recording a `VisionSlot` per image. `images` are consumed in marker
/// order. Returns an empty prompt and fills `error_msg` if a marker token is
/// missing from the vocabulary or the marker count differs from `images`.
[[nodiscard]] PreparedPrompt BuildPrompt(const Tokenizer& tokenizer,
                                         std::string_view rendered,
                                         const std::vector<core::Image>& images,
                                         const ResizeLimits& limits,
                                         std::string* error_msg);

}  // namespace gufo::models::gemma4::vision

#endif  // GUFO_MODELS_GEMMA4_VISION_PROMPT_HPP_