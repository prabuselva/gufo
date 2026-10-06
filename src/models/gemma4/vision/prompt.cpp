#include "src/models/gemma4/vision/prompt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gufo::models::gemma4::vision {
namespace {

// Catmull-Rom cubic kernel (a = -0.5), the filter Pillow uses for BICUBIC.
double CubicWeight(double t) {
  constexpr double kA = -0.5;
  const double x = std::fabs(t);
  if (x <= 1.0) {
    return ((kA + 2.0) * x - (kA + 3.0)) * x * x + 1.0;
  }
  if (x < 2.0) {
    return ((kA * x - 5.0 * kA) * x + 8.0 * kA) * x - 4.0 * kA;
  }
  return 0.0;
}

// One output pixel's taps along a single axis: the input indices to read and
// their normalized weights. Antialiasing widens the support by the downscale
// factor so minification does not alias.
struct AxisTaps {
  std::vector<int> index;
  std::vector<float> weight;
  // CSR offsets: pixel i reads [offset[i], offset[i + 1]).
  std::vector<std::size_t> offset;
};

AxisTaps BuildTaps(int in_size, int out_size) {
  AxisTaps taps;
  if (in_size <= 0 || out_size <= 0) {
    return taps;
  }
  const double scale = static_cast<double>(in_size) / out_size;
  const double filter_scale = std::max(1.0, scale);
  const double support = 2.0 * filter_scale;
  taps.index.reserve(static_cast<std::size_t>(2 * support) + 2);
  taps.weight.reserve(taps.index.capacity());
  taps.offset.reserve(static_cast<std::size_t>(out_size) + 1);
  for (int i = 0; i < out_size; ++i) {
    const double center = (i + 0.5) * scale - 0.5;
    const int left = static_cast<int>(std::floor(center - support));
    const int right = static_cast<int>(std::ceil(center + support));
    double sum = 0.0;
    const std::size_t base = taps.index.size();
    taps.offset.push_back(base);
    for (int j = left; j <= right; ++j) {
      const double w = CubicWeight((j - center) / filter_scale);
      if (w == 0.0) {
        continue;
      }
      taps.index.push_back(std::min(std::max(j, 0), in_size - 1));
      taps.weight.push_back(static_cast<float>(w));
      sum += w;
    }
    if (sum != 0.0) {
      const float inv = static_cast<float>(1.0 / sum);
      for (std::size_t k = base; k < taps.weight.size(); ++k) {
        taps.weight[k] *= inv;
      }
    }
  }
  taps.offset.push_back(taps.index.size());
  return taps;
}

}  // namespace

Size CalcSizePreservedRatio(std::uint32_t width, std::uint32_t height,
                            const ResizeLimits& limits) {
  if (width == 0 || height == 0 || limits.align_size == 0) {
    return {0, 0};
  }
  const float f = static_cast<float>(limits.align_size);
  const std::uint32_t align = limits.align_size;
  const auto round_by = [f, align](float x) {
    return static_cast<std::uint32_t>(std::round(x / f)) * align;
  };
  const auto ceil_by = [f, align](float x) {
    return static_cast<std::uint32_t>(std::ceil(x / f)) * align;
  };
  const auto floor_by = [f, align](float x) {
    return static_cast<std::uint32_t>(std::floor(x / f)) * align;
  };
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  // No longest_edge clamp (Gemma-4 uses the dyn-size path): align up first.
  std::uint32_t w_bar = std::max(limits.align_size, round_by(w));
  std::uint32_t h_bar = std::max(limits.align_size, round_by(h));
  if (limits.max_pixels > 0 &&
      static_cast<double>(h_bar) * w_bar > limits.max_pixels) {
    const double beta = std::sqrt(static_cast<double>(height) * width /
                                  limits.max_pixels);
    h_bar = std::max(limits.align_size, floor_by(static_cast<float>(h / beta)));
    w_bar = std::max(limits.align_size, floor_by(static_cast<float>(w / beta)));
  } else if (limits.min_pixels > 0 &&
             static_cast<double>(h_bar) * w_bar < limits.min_pixels) {
    const double beta =
        std::sqrt(static_cast<double>(limits.min_pixels) / (height * width));
    h_bar = ceil_by(static_cast<float>(h * beta));
    w_bar = ceil_by(static_cast<float>(w * beta));
  }
  return {w_bar, h_bar};
}

PreparedImage PrepareImage(const core::Image& image,
                           const ResizeLimits& limits) {
  PreparedImage out;
  const Size target =
      CalcSizePreservedRatio(image.width, image.height, limits);
  if (target.width == 0 || target.height == 0 || image.pixels.empty()) {
    return out;
  }
  const int sw = static_cast<int>(image.width);
  const int sh = static_cast<int>(image.height);
  const int tw = static_cast<int>(target.width);
  const int th = static_cast<int>(target.height);
  const AxisTaps x_taps = BuildTaps(sw, tw);
  const AxisTaps y_taps = BuildTaps(sh, th);
  // Horizontal pass: source [sh][sw][3] -> [sh][tw][3] float.
  std::vector<float> hres(static_cast<std::size_t>(sh) * tw * 3, 0.0F);
  for (int y = 0; y < sh; ++y) {
    const std::uint8_t* src_row = image.pixels.data() +
                                  static_cast<std::size_t>(y) * sw * 3;
    float* dst_row = hres.data() + static_cast<std::size_t>(y) * tw * 3;
    for (int x = 0; x < tw; ++x) {
      const std::size_t base = static_cast<std::size_t>(x) * 3;
      float acc[3] = {0.0F, 0.0F, 0.0F};
      for (std::size_t t = x_taps.offset[x]; t < x_taps.offset[x + 1]; ++t) {
        const float wgt = x_taps.weight[t];
        const std::uint8_t* s = src_row + static_cast<std::size_t>(x_taps.index[t]) * 3;
        acc[0] += wgt * s[0];
        acc[1] += wgt * s[1];
        acc[2] += wgt * s[2];
      }
      dst_row[base] = acc[0];
      dst_row[base + 1] = acc[1];
      dst_row[base + 2] = acc[2];
    }
  }
  // Vertical pass + CHW [0,1] conversion: [sh][tw][3] -> 3 x (th*tw).
  const std::size_t plane = static_cast<std::size_t>(th) * tw;
  out.pixels.assign(3 * plane, 0.0F);
  for (int y = 0; y < th; ++y) {
    for (int x = 0; x < tw; ++x) {
      float acc[3] = {0.0F, 0.0F, 0.0F};
      for (std::size_t t = y_taps.offset[y]; t < y_taps.offset[y + 1]; ++t) {
        const float wgt = y_taps.weight[t];
        const float* s = hres.data() +
                         static_cast<std::size_t>(y_taps.index[t]) * tw * 3 +
                         static_cast<std::size_t>(x) * 3;
        acc[0] += wgt * s[0];
        acc[1] += wgt * s[1];
        acc[2] += wgt * s[2];
      }
      const std::size_t p = static_cast<std::size_t>(y) * tw + x;
      out.pixels[p] = acc[0] / 255.0F;
      out.pixels[plane + p] = acc[1] / 255.0F;
      out.pixels[2 * plane + p] = acc[2] / 255.0F;
    }
  }
  out.nx = target.width;
  out.ny = target.height;
  out.tokens = (target.width / limits.align_size) *
               (target.height / limits.align_size);
  return out;
}

PreparedPrompt BuildPrompt(const Tokenizer& tokenizer,
                           std::string_view rendered,
                           const std::vector<core::Image>& images,
                           const ResizeLimits& limits,
                           std::string* error_msg) {
  const Tokenizer::TokenId begin = tokenizer.TokenToId("<|image>");
  const Tokenizer::TokenId marker = tokenizer.TokenToId("<|image|>");
  const Tokenizer::TokenId end = tokenizer.TokenToId("<image|>");
  auto fail = [&](const std::string& message) {
    if (error_msg != nullptr) {
      *error_msg = message;
    }
    return PreparedPrompt{};
  };
  if (begin == Tokenizer::kNullToken || marker == Tokenizer::kNullToken ||
      end == Tokenizer::kNullToken) {
    return fail("image marker tokens missing from vocabulary");
  }

  const std::vector<Tokenizer::TokenId> base =
      tokenizer.Encode(rendered, /*add_special=*/false, /*parse_special=*/true);
  std::size_t marker_count = 0;
  for (const Tokenizer::TokenId id : base) {
    marker_count += (id == marker) ? 1 : 0;
  }
  if (marker_count != images.size()) {
    return fail("image marker count (" + std::to_string(marker_count) +
                ") does not match image count (" +
                std::to_string(images.size()) + ")");
  }

  PreparedPrompt out;
  out.tokens.reserve(base.size());
  out.images.reserve(images.size());
  std::size_t next_image = 0;
  for (const Tokenizer::TokenId id : base) {
    if (id != marker) {
      out.tokens.push_back(id);
      continue;
    }
    const PreparedImage prepared = PrepareImage(images[next_image++], limits);
    if (prepared.tokens == 0) {
      return fail("image resized to zero tokens");
    }
    VisionSlot slot;
    slot.count = prepared.tokens;
    slot.pixels = std::move(prepared.pixels);
    slot.nx = prepared.nx;
    slot.ny = prepared.ny;
    out.tokens.push_back(begin);
    slot.offset = static_cast<std::uint32_t>(out.tokens.size());
    out.tokens.insert(out.tokens.end(), prepared.tokens, marker);
    out.tokens.push_back(end);
    out.images.push_back(std::move(slot));
  }
  return out;
}

}  // namespace gufo::models::gemma4::vision