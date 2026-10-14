#ifndef GUFO_CORE_HIP_GEMM_DENSE_F16_GEMM_HPP_
#define GUFO_CORE_HIP_GEMM_DENSE_F16_GEMM_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Shared dense F16 WMMA GEMM for wide-batch prefill projections. The Q8_0
/// weights stay in their native encoding and are dequantized to F16 in LDS
/// (or, with `weights_f16`, are already binary16 and staged verbatim); the
/// activations are pre-narrowed F16 rows, so the matrix cores accumulate in
/// F32 with no per-block requantize. This is the wide-batch sibling of the
/// int8 mmq dense path and wins wherever the arithmetic intensity is low
/// enough that halved activation traffic and no FP32->Q8_1 requantize
/// outweigh int8's 2x tensor throughput. The kernel is a generic core (plain
/// transposed store); tile-plan selection per projection stays with the
/// model.
namespace gufo::core::hip::gemm {

/// Row-tile rasterization for wide batches (batch >= 1024). kAuto keeps the
/// generic shape heuristic; the two planned variants reuse each weight or
/// activation tile across a strip of row tiles so fewer fragments stay live:
/// kRowGroups8 stages a single K block per step, kStagedRowGroups8 two.
/// kStaged256 keeps the kAuto 4x2 waves and doubles the K-block stage,
/// kTile128 halves the row tile (more blocks per launch), and kStaged128 /
/// kTile64 halve the row tile with a two-block stage; the last two keep only
/// 64 accumulator registers (the 256-row tiles hold 128), which is what the
/// native-F16 weight route needs to stay under the VGPR cap without spilling.
/// All are measured selections for the model-private copies of the kernel.
enum class DenseF16Plan : std::uint8_t {
  kAuto,
  kRowGroups8,
  kStagedRowGroups8,
  kStaged256,
  kTile128,
  kStaged128,
  kTile64,
};

/// out[t][r] = sum_k W[r][k] * x[t][k] for t in [0, batch), r in [0, m), with
/// `w` a row-major [m x k] Q8_0 matrix (or binary16 when `weights_f16`), `x`
/// F16 [batch x k] and `out` F32 [batch x m]. Returns false (launching
/// nothing) for an empty shape or a `k` that is not a multiple of 32, so the
/// caller can fall back to another path. Asynchronous on `stream`.
bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream,
                  DenseF16Plan plan = DenseF16Plan::kAuto,
                  bool weights_f16 = false);

}  // namespace gufo::core::hip::gemm

#endif  // GUFO_CORE_HIP_GEMM_DENSE_F16_GEMM_HPP_
