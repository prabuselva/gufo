#ifndef GUFO_MODELS_FLINT_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_FLINT_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Flint-private fused operators for the ROCm trunk: RMSNorm, residuals,
/// SwiGLU, the per-head QK norm + gated NEOX rotary epilogue with the f16
/// KV-cache write, the sink/window attention (decode and prefill in one
/// kernel), the sigmoid top-6 router with the llama.cpp weight clamp, and the
/// weighted summed-expert epilogue. Every kernel mirrors
/// src/models/flint/reference.cpp within the parity budget.
namespace gufo::models::flint::rocm {

/// out[t][i] = x[t][i] * rsqrt(mean(x[t]^2) + eps) * gamma[i], per row.
void RmsNorm(const float* x, const float* gamma, float eps, float* out,
             std::uint32_t rows, std::uint32_t cols, hipStream_t stream);

/// x[t][i] += y[t][i].
void AddResidual(float* x, const float* y, std::uint32_t rows,
                 std::uint32_t cols, hipStream_t stream);

/// out[i] = silu(a[i]) * b[i] over rows*cols elements.
void SiluMul(const float* a, const float* b, float* out, std::size_t count,
             hipStream_t stream);

/// Per head over head_dim: RMS-normalize q in place (q_norm) and k, rotate
/// both by NEOX (pos, theta) when `rope` is set, then write the f16 k/v rows
/// into the cache planes at absolute positions [pos0, pos0 + batch).
/// q is [batch][n_head*head_dim], k/v [batch][n_kv*head_dim].
void QkEpilogue(float* q, float* k, const float* v, const float* q_gamma,
                const float* k_gamma, float eps, float theta, bool rope,
                std::uint32_t pos0, std::uint32_t batch, std::uint32_t n_head,
                std::uint32_t n_kv, std::uint32_t head_dim, __half* k_cache,
                __half* v_cache, std::uint32_t kv_row_elements,
                hipStream_t stream);

/// Sink + window causal attention against the f16 cache planes. For token t
/// at absolute position pos0 + t (batch is 1 on the decode path), keys
/// [max(0, pos+1-window) ... pos] are visible (window 0 selects all).
/// q [batch][n_head*hd], out [batch][n_head*hd]; k_cache/v_cache rows are
/// [n_kv*hd] at the absolute position. Sinks are extra per-head logits.
void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const float* sinks, float scale, std::uint32_t pos0,
               std::uint32_t batch, std::uint32_t window, std::uint32_t n_head,
               std::uint32_t n_kv, std::uint32_t head_dim,
               std::uint32_t kv_row_elements, float* out, hipStream_t stream);

/// probs = sigmoid(router[t]) then top-`used` by probability (ties keep the
/// lower index), weights clamped-sum renormalized. ids int32, weights f32,
/// both [batch][used]. `n_experts` must be 256.
void RouterTopK(const float* probs_logits, std::int32_t* ids, float* weights,
                std::uint32_t batch, std::uint32_t used, bool normalize,
                float weight_scale, hipStream_t stream);

/// x[t] += sum_s weights[t][s] * routed[t*used + s] (+ shexp[t] when set).
void SumExperts(float* x, const float* routed, const float* weights,
                const float* shared, std::uint32_t rows, std::uint32_t cols,
                std::uint32_t used, hipStream_t stream);

}  // namespace gufo::models::flint::rocm

#endif  // GUFO_MODELS_FLINT_KERNELS_ROCM_KERNELS_HPP_
