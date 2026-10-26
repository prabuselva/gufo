#ifndef GUFO_MODELS_FLINT_CPU_OPS_HPP_
#define GUFO_MODELS_FLINT_CPU_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/flint/weights.hpp"

/// Scalar CPU operators for the Flint Preview reference model: one row at a
/// time, every stored format decoded on the fly. Correctness oracle only;
/// nothing here is tuned and nothing is shared with another model.
namespace gufo::models::flint::cpu {

/// Decodes one row of `t` (expert `e`) into `out` (`t.cols` floats). Handles
/// the formats the weight loader accepts: F32, F16, BF16 and Q8_0.
/// Unsupported types decode to zero.
void DequantizeRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
                   float* out);

/// y[r] = sum_c W[e][r][c] * x[c] for every row of the matrix.
void MatVec(const TensorRef& t, std::uint64_t e, std::span<const float> x,
            std::span<float> y);

/// Dot product of one weight row with x.
float DotRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
             std::span<const float> x);

/// RMSNorm with the plain `w * x` scale applied after normalisation (the
/// LLM_NORM_RMS form; Flint norms are stored bare, not as 1 + w). When used
/// with the norm tensors narrowed to one attention head, this is also the
/// per-head `attn_q_norm`/`attn_k_norm` operator.
void RmsNorm(std::span<float> x, const float* w, float eps);

[[nodiscard]] float Sigmoid(float x) noexcept;
[[nodiscard]] float Silu(float x) noexcept;

/// NEOX-style rotary over the full head: pair (i, i + head_dim/2) rotates by
/// pos * theta^(-2i/head_dim) inside each of the `heads` heads. The caller
/// skips it entirely on layers where `rope_pattern` is false.
void Rope(float* x, std::uint32_t heads, std::uint32_t head_dim,
          std::uint32_t pos, float theta);

}  // namespace gufo::models::flint::cpu

#endif  // GUFO_MODELS_FLINT_CPU_OPS_HPP_
