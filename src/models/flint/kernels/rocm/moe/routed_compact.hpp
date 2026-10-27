#ifndef GUFO_MODELS_FLINT_KERNELS_ROCM_MOE_ROUTED_COMPACT_HPP_
#define GUFO_MODELS_FLINT_KERNELS_ROCM_MOE_ROUTED_COMPACT_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::flint::rocm {

/// Narrows `count` F32 activations to F16 in `out`.
void NarrowActivations(const float* x, __half* out, std::size_t count,
                       hipStream_t stream);

/// Rows a compacted bucket layout needs for `slots` routed assignments over
/// `n_experts` experts with 16-row padded buckets.
std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts);

/// Compacts the routed assignments by expert into 16-row padded buckets.
/// `ids` [n_tokens * k] holds the chosen expert per (token, slot) or -1;
/// `counts` [n_experts] is the per-expert histogram. `pad_bounds`
/// [n_experts + 1] receives the exclusive scan of the padded bucket starts,
/// `cursors` [n_experts] is scratch, and `rows_token` / `rows_slot`
/// (RoutedCompactRows long) map each compacted row back to its token and
/// routed slot (slot = token * k + j).
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

/// Builds the routed tile maps of two tile heights on the device from the
/// per-expert assignment counts: `expert | tile << 16` in expert order,
/// entries past the last tile holding a tile index beyond every bucket,
/// which the routed GEMMs skip. `experts` must be at most 256.
void BuildRoutedTiles(const std::uint32_t* counts, std::uint32_t experts,
                      std::uint32_t rows_a, std::uint32_t capacity_a,
                      std::int32_t* tiles_a, std::uint32_t rows_b,
                      std::uint32_t capacity_b, std::int32_t* tiles_b,
                      hipStream_t stream);

}  // namespace gufo::models::flint::rocm

#endif  // GUFO_MODELS_FLINT_KERNELS_ROCM_MOE_ROUTED_COMPACT_HPP_
