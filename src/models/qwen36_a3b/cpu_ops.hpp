#ifndef GUFO_MODELS_QWEN36_A3B_CPU_OPS_HPP_
#define GUFO_MODELS_QWEN36_A3B_CPU_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/qwen36_a3b/weights.hpp"

/// Scalar CPU operators for the Qwen3.6-35B-A3B reference model: one row at a
/// time, every stored format decoded on the fly. Correctness oracle only;
/// nothing here is tuned and nothing is shared with another model.
namespace gufo::models::qwen36_a3b::cpu {

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

/// RMSNorm over `x` in place with gamma `w` (nullptr keeps the raw scale). The
/// GGUF stores the `1 + w` form for the Qwen3.5 norms, so the plain `w * x`
/// scale is what the runtime applies.
void RmsNorm(std::span<float> x, const float* w, float eps);
/// x / sqrt(sum x^2 + eps), the DeltaNet query/key normalization.
void L2Norm(std::span<float> x, float eps);

[[nodiscard]] float Sigmoid(float x) noexcept;
[[nodiscard]] float Silu(float x) noexcept;
[[nodiscard]] float Softplus(float x) noexcept;

/// NEOX-style partial rotary embedding on the first `rotary_dim` elements of
/// each `head_dim` head: pair (i, i + rotary_dim/2) rotates by pos * theta_i.
/// Text-only mRoPE has equal positions on every axis, so it is exactly this
/// rotation over the leading `rotary_dim` channels.
void Rope(float* x, std::uint32_t heads, std::uint32_t head_dim,
          std::uint32_t rotary_dim, std::uint32_t pos, float theta);

}  // namespace gufo::models::qwen36_a3b::cpu

#endif  // GUFO_MODELS_QWEN36_A3B_CPU_OPS_HPP_