#ifndef GUFO_MODELS_GEMMA4_VISION_WEIGHTS_HPP_
#define GUFO_MODELS_GEMMA4_VISION_WEIGHTS_HPP_

#include <cstdint>
#include <optional>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/vision/config.hpp"
#include "src/models/gemma4/weights.hpp"

namespace gufo::models::gemma4::vision {

/// One vision transformer block. All projections are BF16 (the mmproj ships
/// BF16/F32 only); norms are F32. Q/K/V are full 1152-wide square projections
/// (no GQA, no bias). `q_norm`/`k_norm` are per-head RMSNorm weights of length
/// head_dim (72); V is normalized weightlessly. `attn_post_norm` and
/// `ffn_post_norm` normalize the branch output before the residual add.
struct BlockWeights {
  TensorRef ln1;            ///< [1152] pre-attention RMSNorm.
  TensorRef attn_q;         ///< [1152 -> 1152].
  TensorRef attn_k;         ///< [1152 -> 1152].
  TensorRef attn_v;         ///< [1152 -> 1152].
  TensorRef q_norm;         ///< [72] per-head RMSNorm.
  TensorRef k_norm;         ///< [72] per-head RMSNorm.
  TensorRef attn_out;       ///< [1152 -> 1152].
  TensorRef attn_post_norm; ///< [1152].
  TensorRef ln2;            ///< [1152] pre-FFN RMSNorm.
  TensorRef ffn_gate;       ///< [1152 -> 4304].
  TensorRef ffn_up;         ///< [1152 -> 4304].
  TensorRef ffn_down;       ///< [4304 -> 1152].
  TensorRef ffn_post_norm;  ///< [1152].
};

/// The `gemma4v` vision tower bound from the mmproj sidecar. `patch_embed` is
/// the conv kernel stored as [cout=1152][cin*ky*kx=768] (kx fastest, then ky,
/// then cin). `pos_x`/`pos_y` are the two stacked [1152][10240] lookup tables.
/// `std_bias`/`std_scale` are the merger's per-channel standardization, and
/// `projection` is the clippable linear to the trunk width.
struct VisionWeights {
  Config config;
  TensorRef patch_embed;  ///< [768 -> 1152], F32.
  TensorRef pos_x;        ///< [1152 -> 10240], F32.
  TensorRef pos_y;        ///< [1152 -> 10240], F32.
  TensorRef std_bias;     ///< [1152], F32.
  TensorRef std_scale;    ///< [1152], F32.
  TensorRef projection;   ///< [1152 -> 2816], BF16.
  std::vector<BlockWeights> blocks;

  [[nodiscard]] static std::optional<VisionWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::gemma4::vision

#endif  // GUFO_MODELS_GEMMA4_VISION_WEIGHTS_HPP_