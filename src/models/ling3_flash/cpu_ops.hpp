#ifndef GUFO_MODELS_LING3_FLASH_CPU_OPS_HPP_
#define GUFO_MODELS_LING3_FLASH_CPU_OPS_HPP_

#include <cstdint>
#include <span>

#include "src/models/ling3_flash/weights.hpp"

namespace gufo::models::ling3_flash::cpu {

/// Dequantizes one row of `t` (stacked matrix `e`) into `out` (`t.cols`
/// floats). Supports every format the reference artifact stores.
void DequantizeRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
                   float* out);

/// y = W x for one stacked matrix, double accumulation.
void MatVec(const TensorRef& t, std::uint64_t e, std::span<const float> x,
            std::span<float> y);

float DotRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
             std::span<const float> x);

/// RMSNorm over the whole span: x * rsqrt(mean(x^2) + eps) * w.
void RmsNorm(std::span<float> x, const float* w, float eps);

/// L2 normalization used by KDA: x / (sqrt(sum(x^2)) + eps).
void L2Norm(std::span<float> x, float eps);

float Sigmoid(float x) noexcept;
float Silu(float x) noexcept;

/// RoPE over the first `rotary_dim` entries of each head row, operating on
/// consecutive pairs (2i, 2i+1) with frequency theta^(-2i/rotary_dim), the
/// pairing llama.cpp selects for `bailingmoe3` (LLAMA_ROPE_TYPE_NORM).
void Rope(float* x, std::uint32_t heads, std::uint32_t head_dim,
          std::uint32_t rotary_dim, std::uint32_t pos, float theta);

}  // namespace gufo::models::ling3_flash::cpu

#endif  // GUFO_MODELS_LING3_FLASH_CPU_OPS_HPP_