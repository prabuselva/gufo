#ifndef GUFO_MODELS_NEMOTRON_H_MOE_CPU_OPS_HPP_
#define GUFO_MODELS_NEMOTRON_H_MOE_CPU_OPS_HPP_

#include <cstdint>
#include <span>

#include "src/models/nemotron_h_moe/weights.hpp"

namespace gufo::models::nemotron_h_moe::cpu {

/// Dequantizes one row of `t` (stacked matrix `e`) into `out` (`t.cols`
/// floats). Covers every format the artifact stores: F32, Q8_0 (core
/// helper), plus the model-private IQ2_XXS and NVFP4 layouts.
void DequantizeRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
                   float* out);

/// y = W x for stacked matrix `e` (y.row i = W.row i dot x), double
/// accumulation over rows.
void MatVec(const TensorRef& t, std::uint64_t e, std::span<const float> x,
            std::span<float> y);

/// Single dot product of row `row` of stacked matrix `e` with `x`.
float DotRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
             std::span<const float> x);

/// RMSNorm over the whole span: x * rsqrt(mean(x^2) + eps) * w.
void RmsNorm(std::span<float> x, const float* w, float eps);

float Sigmoid(float x) noexcept;
float Silu(float x) noexcept;

/// Relu(x)^2, the FFN activation of every expert and dense branch.
float ReluSqr(float x) noexcept;

/// softplus with the upstream reference threshold at 20 (linear above).
float Softplus(float x) noexcept;

/// UE4M3 scale decode used by NVFP4 (unsigned, 4 exp bits bias 7, 3
/// mantissa bits), returned halved to match kFp4Values (2 * E2M1).
float Ue4m3ToFloat(std::uint8_t x) noexcept;

}  // namespace gufo::models::nemotron_h_moe::cpu

#endif  // GUFO_MODELS_NEMOTRON_H_MOE_CPU_OPS_HPP_
