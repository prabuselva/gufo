// RoutedCompact builders forked from the Gemma-4 route (see VENDOR.md):
// 16-row padded expert buckets, the rows_token/rows_slot maps and the
// `expert | tile << 16` tile maps consumed by moe_prefill.hip.cpp, plus
// the F32->F16 activation narrowing. Kernels are copied verbatim from
// src/models/gemma4/kernels/rocm/routed_f16.hip.cpp (device side).
#include "src/models/flint/kernels/rocm/moe/routed_compact.hpp"

#include <cstdint>

namespace gufo::models::flint::rocm {
namespace {

constexpr int kThreads = 256;
constexpr std::size_t kRoutedTileTokens = 16;
constexpr std::int32_t kDeadTile = 0x7FFF;

/// Compacts the routed assignments by expert with 16-row padded buckets.
/// One block: exclusive scan of the padded counts into pad_bounds[0..E].
__global__ void RoutedPadBoundsKernel(const std::uint32_t* __restrict__ counts,
                                      std::int32_t* __restrict__ pad_bounds,
                                      std::int32_t* __restrict__ cursors,
                                      std::int32_t* __restrict__ rows_token,
                                      std::int32_t* __restrict__ rows_slot,
                                      std::uint32_t n_experts) {
  __shared__ std::int32_t padded[1024];
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    padded[e] = static_cast<std::int32_t>((counts[e] + 15u) / 16u * 16u);
    cursors[e] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::int32_t running = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      pad_bounds[e] = running;
      running += padded[e];
    }
    pad_bounds[n_experts] = running;
  }
  __syncthreads();
  // Scatter overwrites every occupied row. Only the padding can retain a
  // previous batch's assignments; clear it while the bucket bounds are hot.
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    for (std::uint32_t row = counts[e];
         row < static_cast<std::uint32_t>(padded[e]); ++row) {
      const auto at = pad_bounds[e] + row;
      rows_token[at] = -1;
      rows_slot[at] = -1;
    }
  }
}

/// rows_token[c] / rows_slot[c] for every routed (token, slot); the order
/// inside a bucket is whatever the atomics produce, which changes nothing:
/// every output row is computed from its own inputs only.
__global__ void RoutedScatterKernel(const std::int32_t* __restrict__ ids,
                                    const std::int32_t* __restrict__ pad_bounds,
                                    std::int32_t* __restrict__ cursors,
                                    std::int32_t* __restrict__ rows_token,
                                    std::int32_t* __restrict__ rows_slot,
                                    std::uint32_t slots, std::uint32_t k) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }
  const std::int32_t e = ids[slot];
  if (e < 0) {
    return;
  }
  const std::int32_t c = pad_bounds[e] + atomicAdd(&cursors[e], 1);
  rows_token[c] = static_cast<std::int32_t>(slot / k);
  rows_slot[c] = static_cast<std::int32_t>(slot);
}

__global__ void NarrowHalfKernel(const float* x, __half* out,
                                 std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    out[i] = __float2half(x[i]);
  }
}

inline unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

// One block: thread e counts expert e's tiles of both heights, an ordered
// scan places them, and the remaining capacity entries become dead tiles.
__global__ void __launch_bounds__(256)
    BuildRoutedTilesKernel(const std::uint32_t* __restrict__ counts,
                           std::uint32_t experts, std::uint32_t rows_a,
                           std::uint32_t capacity_a,
                           std::int32_t* __restrict__ tiles_a,
                           std::uint32_t rows_b, std::uint32_t capacity_b,
                           std::int32_t* __restrict__ tiles_b) {
  __shared__ std::uint32_t scan_a[256];
  __shared__ std::uint32_t scan_b[256];
  const std::uint32_t e = threadIdx.x;
  const std::uint32_t padded = e < experts ? (counts[e] + 15U) / 16U * 16U : 0;
  const std::uint32_t n_a = (padded + rows_a - 1) / rows_a;
  const std::uint32_t n_b = (padded + rows_b - 1) / rows_b;
  scan_a[e] = n_a;
  scan_b[e] = n_b;
  __syncthreads();
  // Inclusive Hillis-Steele scan; integer sums, so order-free.
  for (std::uint32_t step = 1; step < 256; step <<= 1) {
    const std::uint32_t add_a = e >= step ? scan_a[e - step] : 0;
    const std::uint32_t add_b = e >= step ? scan_b[e - step] : 0;
    __syncthreads();
    scan_a[e] += add_a;
    scan_b[e] += add_b;
    __syncthreads();
  }
  const std::uint32_t first_a = scan_a[e] - n_a;
  const std::uint32_t first_b = scan_b[e] - n_b;
  for (std::uint32_t j = 0; j < n_a; ++j) {
    tiles_a[first_a + j] = static_cast<std::int32_t>(e | (j << 16));
  }
  for (std::uint32_t j = 0; j < n_b; ++j) {
    tiles_b[first_b + j] = static_cast<std::int32_t>(e | (j << 16));
  }
  constexpr std::int32_t kDead = kDeadTile << 16;
  for (std::uint32_t i = scan_a[255] + e; i < capacity_a; i += 256) {
    tiles_a[i] = kDead;
  }
  for (std::uint32_t i = scan_b[255] + e; i < capacity_b; i += 256) {
    tiles_b[i] = kDead;
  }
}

}  // namespace

void NarrowActivations(const float* x, __half* out, std::size_t count,
                       hipStream_t stream) {
  NarrowHalfKernel<<<Blocks(count), kThreads, 0, stream>>>(x, out, count);
}

std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts) {
  return slots + (n_experts * (kRoutedTileTokens - 1));
}

void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream) {
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  RoutedPadBoundsKernel<<<1, 1024, 0, stream>>>(counts, pad_bounds, cursors,
                                                rows_token, rows_slot,
                                                n_experts);
  RoutedScatterKernel<<<Blocks(slots), kThreads, 0, stream>>>(
      ids, pad_bounds, cursors, rows_token, rows_slot,
      static_cast<std::uint32_t>(slots), k);
}

void BuildRoutedTiles(const std::uint32_t* counts, std::uint32_t experts,
                      std::uint32_t rows_a, std::uint32_t capacity_a,
                      std::int32_t* tiles_a, std::uint32_t rows_b,
                      std::uint32_t capacity_b, std::int32_t* tiles_b,
                      hipStream_t stream) {
  BuildRoutedTilesKernel<<<1, 256, 0, stream>>>(counts, experts, rows_a,
                                                capacity_a, tiles_a, rows_b,
                                                capacity_b, tiles_b);
}

}  // namespace gufo::models::flint::rocm
