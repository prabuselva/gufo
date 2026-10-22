#include "src/models/nemotron_h_moe/cpu_ops.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/nemotron_h_moe/cpu_tables.hpp"

namespace gufo::models::nemotron_h_moe::cpu {

using core::GgmlType;

namespace {

constexpr std::size_t kIq2XxsBlock = 256;  // elements per 66 B block
constexpr std::size_t kNvfp4Block = 64;    // elements per 36 B block
constexpr std::size_t kNvfp4Sub = 16;      // elements per UE4M3 scale

// IQ2_XXS block: fp16 scale + 32 uint16 codes (8 elements each, 2.0625
// bits per weight). Layout and decode transcribed from the upstream
// reference (block_iq2_xxs / dequantize_row_iq2_xxs).
void DequantizeIq2Xxs(const std::uint8_t* src, float* out, std::size_t k) {
  for (std::size_t b = 0; b < k / kIq2XxsBlock; ++b) {
    const std::uint8_t* blk = src + b * 66;
    std::uint16_t dh = 0;
    std::memcpy(&dh, blk, 2);
    const float d = gufo::quant::Fp16ToFloat(dh);
    // ggml's block field is uint16_t qs[QK_K/8], so `qs + 4 * ib32` there
    // advances 8 bytes per sub-block. Index in bytes to match exactly.
    const std::uint8_t* qs = blk + 2;
    for (std::size_t ib32 = 0; ib32 < kIq2XxsBlock / 32; ++ib32) {
      // ggml's `y` advances cumulatively (`++y` per element), so each ib32
      // appends its own 32 floats instead of rewriting the block's first 32.
      float* y = out + b * kIq2XxsBlock + ib32 * 32;
      std::uint32_t aux32[2] = {0, 0};
      std::memcpy(aux32, qs + 8 * ib32, sizeof(aux32));
      const std::uint8_t* aux8 = reinterpret_cast<const std::uint8_t*>(aux32);
      const float db = d * (0.5f + static_cast<float>(aux32[1] >> 28)) * 0.25f;
      for (std::size_t l = 0; l < 4; ++l) {
        const std::uint8_t* grid =
            reinterpret_cast<const std::uint8_t*>(kIq2XxsGrid + aux8[l]);
        const std::uint8_t signs = kSignsIq2Xs[(aux32[1] >> (7 * l)) & 127];
        for (std::size_t j = 0; j < 8; ++j) {
          y[l * 8 + j] = db * static_cast<float>(grid[j]) *
                         ((signs & kMaskIq2Xs[j]) != 0 ? -1.0f : 1.0f);
        }
      }
    }
  }
}

// NVFP4 block: 4 UE4M3 sub-scales + 32 B of packed E2M1 nibbles. Each byte
// carries the (j, j + 8) pair inside its 16-element sub-block, matching the
// upstream dequantize_row_nvfp4 layout.
void DequantizeNvfp4(const std::uint8_t* src, float* out, std::size_t k) {
  for (std::size_t b = 0; b < k / kNvfp4Block; ++b) {
    const std::uint8_t* blk = src + b * 36;
    for (std::size_t s = 0; s < kNvfp4Block / kNvfp4Sub; ++s) {
      const float d = Ue4m3ToFloat(blk[s]);
      const std::uint8_t* qs = blk + 4 + s * (kNvfp4Sub / 2);
      float* yb = out + b * kNvfp4Block + s * kNvfp4Sub;
      for (std::size_t j = 0; j < kNvfp4Sub / 2; ++j) {
        const std::int8_t v0 = kFp4Values[qs[j] & 0x0F];
        const std::int8_t v1 = kFp4Values[qs[j] >> 4];
        yb[j] = static_cast<float>(v0) * d;
        yb[j + kNvfp4Sub / 2] = static_cast<float>(v1) * d;
      }
    }
  }
}

}  // namespace

void DequantizeRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
                   float* out) {
  const std::uint8_t* src = t.Expert(e) + row * t.RowBytes();
  const std::size_t k = t.cols;
  switch (t.type) {
    case GgmlType::kF32:
      std::memcpy(out, src, k * sizeof(float));
      break;
    case GgmlType::kQ8_0:
      gufo::quant::DequantizeQ8_0(src, out, k);
      break;
    case GgmlType::kIQ2_XXS:
      DequantizeIq2Xxs(src, out, k);
      break;
    case GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(src, out, k);
      break;
    case GgmlType::kQ5_K:
      gufo::quant::DequantizeQ5_K(src, out, k);
      break;
    case GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(src, out, k);
      break;
    case GgmlType::kIQ4_XS:
      gufo::quant::DequantizeIQ4_XS(src, out, k);
      break;
    case GgmlType::kNVFP4:
      DequantizeNvfp4(src, out, k);
      break;
    default:
      std::memset(out, 0, k * sizeof(float));
      break;
  }
}

float DotRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
             std::span<const float> x) {
  std::vector<float> w(t.cols);
  DequantizeRow(t, e, row, w.data());
  double acc = 0.0;
  for (std::size_t i = 0; i < t.cols; ++i) {
    acc += static_cast<double>(w[i]) * x[i];
  }
  return static_cast<float>(acc);
}

void MatVec(const TensorRef& t, std::uint64_t e, std::span<const float> x,
            std::span<float> y) {
  const auto rows = static_cast<std::int64_t>(t.rows);
#pragma omp parallel
  {
    std::vector<float> w(t.cols);
#pragma omp for schedule(static)
    for (std::int64_t r = 0; r < rows; ++r) {
      DequantizeRow(t, e, static_cast<std::uint64_t>(r), w.data());
      double acc = 0.0;
      for (std::size_t i = 0; i < t.cols; ++i) {
        acc += static_cast<double>(w[i]) * x[i];
      }
      y[static_cast<std::size_t>(r)] = static_cast<float>(acc);
    }
  }
}
void RmsNorm(std::span<float> x, const float* w, float eps) {
  double sum_sq = 0.0;
  for (const float v : x) {
    sum_sq += static_cast<double>(v) * v;
  }
  const float scale = static_cast<float>(
      1.0 / std::sqrt(sum_sq / static_cast<double>(x.size()) + eps));
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = x[i] * scale * w[i];
  }
}

float Sigmoid(float x) noexcept {
  return 1.0f / (1.0f + std::exp(-x));
}

float Silu(float x) noexcept {
  return x * Sigmoid(x);
}

float ReluSqr(float x) noexcept {
  return x > 0.0f ? x * x : 0.0f;
}

float Softplus(float x) noexcept {
  // Upstream reference threshold: linear above 20 keeps dt finite.
  return x > 20.0f ? x : std::log1p(std::exp(x));
}

float Ue4m3ToFloat(std::uint8_t x) noexcept {
  if (x == 0 || x == 0x7F) {
    return 0.0f;
  }
  const int exp = (x >> 3) & 0xF;
  const int man = x & 0x7;
  float raw = 0.0f;
  if (exp == 0) {
    raw = std::ldexp(static_cast<float>(man), -9);
  } else {
    raw = std::ldexp(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
  }
  return raw * 0.5f;
}

}  // namespace gufo::models::nemotron_h_moe::cpu
