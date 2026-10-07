#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_ROUTED_F16_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_ROUTED_F16_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Routed F16 WMMA expert GEMM for the Gemma-4-26B-A4B MoE prefill, ported
/// from the Qwen3.8-Flash-Next route (src/models/qwen38_flash_next/kernels/
/// rocm/kernels.hip.cpp). The routed assignments are compacted by expert into
/// 16-row padded buckets; activations are narrowed to F16; one WMMA GEMM per
/// (expert, row tile) runs on the matrix cores with the weights dequantized to
/// F16 after the LDS read. Nothing here is shared with another model.
namespace gufo::models::gemma4::rocm {

/// GGUF type ids the routed F16 GEMM accepts for the expert weights.
enum class WeightType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ4_K = 12,
  kQ5_K = 13,
  kBF16 = 30,
};

/// Narrows `count` F32 activations to F16 (or BF16 when `bf16`) in `out`.
void NarrowActivations(const float* x, void* out, bool bf16, std::size_t count,
                       hipStream_t stream);

/// Rows a compacted bucket layout needs for `slots` routed assignments over
/// `n_experts` experts with 16-row padded buckets.
std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts);

/// Compacts the routed assignments by expert into 16-row padded buckets.
/// `ids` [n_tokens * k] holds the chosen expert per (token, slot) or -1;
/// `counts` [n_experts] is the per-expert histogram. `pad_bounds`
/// [n_experts + 1] receives the exclusive scan of the padded bucket starts,
/// `cursors` [n_experts] is scratch, and `rows_token` / `rows_slot`
/// (RoutedCompactRows long) map each compacted row back to its token and slot.
void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream);

/// Entries of a routed tile map of `rows`-row tiles that any routing of
/// `slots` assignments over `experts` 16-padded buckets fits in.
[[nodiscard]] constexpr std::uint32_t RoutedTileCapacity(
    std::uint32_t slots, std::uint32_t experts, std::uint32_t rows) noexcept {
  return (slots + 15 * experts) / rows + experts;
}

/// Builds the routed tile map on the device from the per-expert assignment
/// counts: `expert | tile << 16` in expert order, entries past the last tile
/// holding a tile index beyond every bucket, which the routed GEMM skips. The
/// GEMM launches `capacity` tiles, so the host never reads the counts back.
/// `experts` must be at most 256.
void BuildRoutedTiles(const std::uint32_t* counts, std::uint32_t experts,
                      std::uint32_t rows, std::uint32_t capacity,
                      std::int32_t* tiles, hipStream_t stream);

/// Routed F16 WMMA expert GEMM. `w` is the [n_experts][m][k] weight tensor in
/// `type`; `x` is the F16 activation rows indexed by `rows_in`; `tiles`
/// [n_tiles] packs the expert in the low 16 bits and the token macro-tile
/// index in the high 16. `pad_bounds` [n_experts + 1] gives the bucket starts.
/// Each compacted output row `rows_out[c]` receives the result as F32 `out`
/// (m-wide) or narrowed to F16 `out_half`. Exactly one of `out` / `out_half`
/// is non-null. Gemma-4's gelu-tanh activation is applied separately by
/// GegluF16/GegluF32 between the gate_up and down GEMMs. `tile_rows` is 16,
/// 48 or 64. Returns false for an unsupported shape.
bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   float* out, __half* out_half, std::size_t m, std::size_t k,
                   hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_ROUTED_F16_HPP_