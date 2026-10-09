#ifndef GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::ling3_flash::rocm {

/// Fused single-token operators for the Ling 3.0 Flash decode graph. Every
/// entry mirrors the CPU oracle in `src/models/ling3_flash/reference.cpp`
/// (RMSNorm/L2Norm/RoPE formulas from `cpu_ops.cpp`); the GEMV tier in
/// `gemv.hpp` supplies the projections these operators sit between.

/// out = x * rsqrt(mean(x^2) + eps) * w. `w` may be null (ones).
void RmsNorm(const float* x, const float* w, float eps, float* out,
             std::uint32_t n, hipStream_t stream);

/// out = a + b.
void Add(const float* a, const float* b, float* out, std::uint32_t n,
         hipStream_t stream);

/// out[0:n] = a, out[n:2n] = b (the MTP [embedding; hidden] concat).
void Concat2(const float* a, const float* b, float* out, std::uint32_t n,
             hipStream_t stream);

/// x = sigmoid(x), in place.
void SigmoidInPlace(float* x, std::uint32_t n, hipStream_t stream);

/// RoPE over the first `rotary_dim` entries of each head row on consecutive
/// pairs (2i, 2i+1) with frequency theta^(-2i/rotary_dim), in place.
void Rope(float* x, std::uint32_t heads, std::uint32_t head_dim,
          std::uint32_t rotary_dim, std::uint32_t pos, float theta,
          hipStream_t stream);

/// Rope over rows of arbitrary stride: the rotary segment of row h starts at
/// `x + h * stride` (the MLA q tail, where the 64 rotary entries sit after the
/// 128 nope entries inside each 192-wide head row).
void RopeStrided(float* x, std::uint32_t heads, std::uint32_t stride,
                 std::uint32_t rotary_dim, std::uint32_t pos, float theta,
                 hipStream_t stream);

/// One causal depthwise conv tap step for q/k/v: `conv_w` is [inner][kern]
/// (channel-major taps), `history` is [kern-1][inner] oldest-first and is
/// shifted in place, `out = silu(conv(proj))`.
void ConvSilu(const float* conv_w, const float* proj, float* history,
              float* out, std::uint32_t inner, std::uint32_t kern,
              hipStream_t stream);

/// Per-head L2 normalization of q and k (head rows of length `d`); q is then
/// scaled by `q_scale`. In place.
void KdaNormHeads(float* q, float* k, std::uint32_t heads, std::uint32_t d,
                  float eps, float q_scale, hipStream_t stream);

/// Fused KDA delta-rule decode step plus the per-head output norm:
/// decay_j = exp(gate_lb * sigmoid(a[h] * f[h*d+j])); S_j *= decay_j;
/// u_j = S_j . k_h; delta_j = (v[h*d+j] - u_j) * sigmoid(beta[h]);
/// S_j += delta_j * k_h; o_j = S_j . q_h;
/// out[h*d:] = rmsnorm(o) * norm_w * sigmoid(g[h*d:]).
/// `state` is [heads][d][d] and is updated in place.
void KdaDelta(const float* q, const float* k, const float* v, const float* f,
              const float* beta, const float* a, const float* norm_w,
              const float* g, float* state, float* out, std::uint32_t heads,
              std::uint32_t d, float eps, float gate_lb, hipStream_t stream);

/// Sigmoid router with bias, group-limited top-k (ties to the smaller index,
/// matching ggml argsort_top_k): selection = sigmoid(logits) + bias, keep the
/// `groups_used` groups with the highest sum of their two best scores, take
/// the top `used` experts, weights = unbiased probs normalized by
/// max(sum, 6.103515625e-5) * weights_scale.
void RouterTopK(const float* logits, const float* bias, std::uint32_t n_exp,
                std::uint32_t groups, std::uint32_t groups_used,
                std::uint32_t used, float weights_scale, std::int32_t* ids,
                float* weights, hipStream_t stream);

/// out = clamp(silu(gate), -inf, limit) * clamp(up, -limit, limit) when
/// limit > 1e-6, else out = silu(gate) * up.
void SwigluClamped(const float* gate, const float* up, float* out,
                   std::uint32_t n, float limit, hipStream_t stream);

/// x[row*cols:] *= scales[row], in place (the MLA per-head output gate).
void RowScale(float* x, const float* scales, std::uint32_t rows,
              std::uint32_t cols, hipStream_t stream);

/// dst[i] = sum_r weights[r] * src[r*cols + i] (the MoE weighted expert-output
/// accumulation over the `used` selected experts).
void RowWeightedSum(const float* src, const float* weights, float* dst,
                    std::uint32_t rows, std::uint32_t cols, hipStream_t stream);

}  // namespace gufo::models::ling3_flash::rocm

#endif  // GUFO_MODELS_LING3_FLASH_KERNELS_ROCM_KERNELS_HPP_