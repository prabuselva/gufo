// Prefill attention on the gfx1151 WMMA matrix cores.
#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_ATTENTION_WMMA_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_ATTENTION_WMMA_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::gemma4::rocm {

/// One causal, optionally windowed prefill chunk over the F16 caches.
/// Positions are absolute: query row `r` sits at `first_position + r`
/// (or every row shares `first_position` when `shared_position`), and keys
/// exist at positions [0, key_limit).
struct AttentionArgs {
  const float* q;
  const std::uint16_t* k_cache;  ///< binary16; rotated dims when derived
  const std::uint16_t* v_cache;  ///< binary16
  float* out;
  std::uint32_t rows;
  std::uint32_t heads;
  std::uint32_t kv_heads;
  std::uint32_t head_dim;
  std::uint32_t first_position;
  bool shared_position;
  std::uint32_t key_limit;
  std::uint32_t window;
  std::uint32_t ring;
  std::uint32_t rope_pairs;  ///< rotated pairs of derived keys, or 0
  const std::uint32_t* key_ends;
  /// Optional binary16 copy of `out` (the WMMA prefill path writes it; there
  /// `out` may be null).
  void* out_half;
};

/// Prefill attention on the WMMA matrix cores (binary16 operands, FP32
/// accumulation) for head_dim 256 and 512 with an even number of query heads
/// per KV head. Returns false without launching for other geometries.
[[nodiscard]] bool LaunchWmmaPrefillAttention(const AttentionArgs& args,
                                              hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_ATTENTION_WMMA_HPP_
