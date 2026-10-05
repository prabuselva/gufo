#ifndef GUFO_MODELS_GEMMA4_CPU_OPS_HPP_
#define GUFO_MODELS_GEMMA4_CPU_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/gemma4/weights.hpp"

/// Scalar CPU operators for the Gemma-4-26B-A4B reference model: one row at a
/// time, every stored format decoded on the fly. Correctness oracle only;
/// nothing here is tuned and nothing is shared with another model.
namespace gufo::models::gemma4::cpu {

/// Decodes one row of `t` (expert `e`) into `out` (`t.cols` floats). Handles
/// the formats the weight loader accepts: F32, F16, BF16, Q8_0, Q4_K, Q5_K and
/// Q6_K. Unsupported types decode to zero.
void DequantizeRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
                   float* out);

/// y[r] = sum_c W[e][r][c] * x[c] for every row of the matrix.
void MatVec(const TensorRef& t, std::uint64_t e, std::span<const float> x,
            std::span<float> y);

/// Dot product of one weight row with x.
float DotRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
             std::span<const float> x);

/// RMSNorm in place: `x = w * x / sqrt(mean(x^2) + eps)`. The GGUF stores the
/// raw multiplier (not the `1 + w` form), and `w == nullptr` keeps the plain
/// scale, which is what the weightless V normalization and the router input
/// use.
void RmsNorm(std::span<float> x, const float* w, float eps);

/// In-place softmax over the whole span.
void Softmax(std::span<float> x);

/// GELU tanh approximation: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))).
[[nodiscard]] float Gelu(float x) noexcept;

/// NEOX rotary embedding on the first `rotary_dim` channels of each head:
/// pair (i, i + rotary_dim/2) rotates by `pos * theta^(-2i/rotary_dim) /
/// freq_factors[i]`. `freq_factors` (length rotary_dim/2) is the proportional
/// rope table of the full-attention layers; nullptr rotates every pair, which
/// is what the sliding-window layers do.
void RopeNeox(float* x, std::uint32_t heads, std::uint32_t head_dim,
              std::uint32_t rotary_dim, std::uint32_t pos, float theta,
              const float* freq_factors);

}  // namespace gufo::models::gemma4::cpu

#endif  // GUFO_MODELS_GEMMA4_CPU_OPS_HPP_