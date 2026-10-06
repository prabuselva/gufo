#ifndef GUFO_MODELS_GEMMA4_VISION_REFERENCE_HPP_
#define GUFO_MODELS_GEMMA4_VISION_REFERENCE_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "src/models/gemma4/vision/weights.hpp"

namespace gufo::models::gemma4::vision {

/// Float32 scalar reference of the `gemma4v` vision tower. It pins the exact
/// semantics the ROCm kernels must reproduce: the 16x16 patch convolution over
/// pixels scaled to [-1,1], the two stacked x/y positional tables, the 27
/// blocks with per-head Q/K RMSNorm, weightless V norm, 2D NEOX rope (theta
/// 100, x over dims [0,36), y over [36,72)), non-causal attention with no
/// 1/sqrt(d) scaling, the geglu_quick FFN, the 3x3 average-pool merger scaled
/// by sqrt(1152), the per-channel standardization and weightless RMSNorm, and
/// the clippable projection to the trunk width. Speed is irrelevant; this is
/// the numerical oracle the vision kernels are validated against.
class ReferenceEncoder {
public:
  explicit ReferenceEncoder(const VisionWeights& weights);

  /// Runs the tower over one image. `pixels` is CHW, `3 * ny * nx` floats in
  /// [0,1] (raw /255), with `nx` and `ny` multiples of the patch size. The
  /// grid must be a multiple of `patch_size * kMergeSize` so the merger pool
  /// divides it. `out` receives `n_tokens * projection_dim` floats, row-major,
  /// where `n_tokens = (nx / (patch*kMerge)) * (ny / (patch*kMerge))`.
  [[nodiscard]] bool Encode(const float* pixels, std::uint32_t nx,
                            std::uint32_t ny, std::vector<float>& out,
                            std::string* error_msg = nullptr);

private:
  const VisionWeights& w_;
};

}  // namespace gufo::models::gemma4::vision

#endif  // GUFO_MODELS_GEMMA4_VISION_REFERENCE_HPP_