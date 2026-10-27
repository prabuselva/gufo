#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_MOE_PREFILL_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_MOE_PREFILL_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::flint::rocm {

// Routed expert weight formats the 96-row prefill kernels decode natively.
// 32 bits wide: the value crosses from GCC-built host code into clang-built
// HIP code, and clang assumes a narrower argument arrives zero-extended where
// GCC leaves the upper bits undefined.
enum class ExpertFormat : std::uint32_t {
  kQ4_K,
  kQ5_K,
  kQ6_K,
  kQ8_0,
  kQ5_1,
  kF16,
  kQ4_0,
  kQ5_0,
};

// Routed prefill projection over binary16 activation rows `x` ([rows][k])
// in the RoutedCompact layout (16-padded expert buckets and a tile map of
// expert | tile << 16 entries built for `tile_rows` bucket rows per tile):
// row rows_out[i] of `out` (FP32) or `out_half` (binary16, saturated)
// receives W[expert] x[rows_in[i]]. Covers the formats the shared
// RoutedF16Gemm route lacks or runs slower (Q4_K, Q5_K, Q6_K, and Q4_0 /
// Q5_0 / Q5_1 / Q8_0 / binary16 with FP32 outputs; 96-row tiles); returns
// false for others. With `geglu` (a K-quant, Q4_0, Q8_0 or binary16 fused
// [gate | up] W, binary16 output) row r of `out_half` holds the m / 2 GeGLU
// values of the binary16 pair.
[[nodiscard]] bool LaunchRoutedHalfGemm(
    ExpertFormat format, const void* w, const void* x,
    const std::int32_t* tiles, std::uint32_t n_tiles, std::uint32_t tile_rows,
    const std::int32_t* pad_bounds, const std::int32_t* rows_in,
    const std::int32_t* rows_out, float* out, void* out_half, std::uint32_t m,
    std::uint32_t k, hipStream_t stream, bool geglu = false);

// Dense prefill projection over binary16 activation rows `x` ([rows][k])
// for Q4_K, Q5_K, Q6_K (the routed GEMM's arithmetic over an identity
// routing; k a multiple of 256) or Q4_0 / Q5_0 (the routed down kernel's
// layout). `out` (FP32) or `out_half` (binary16, saturated; K-quants only)
// is [rows][m]; `geglu` as in LaunchRoutedHalfGemm. Returns false for other
// formats or shapes.
[[nodiscard]] bool LaunchDenseHalfGemm(ExpertFormat format, const void* w,
                                       const void* x, float* out,
                                       void* out_half, std::uint32_t rows,
                                       std::uint32_t m, std::uint32_t k,
                                       hipStream_t stream, bool geglu = false);

}  // namespace gufo::models::flint::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_MOE_PREFILL_HPP_
