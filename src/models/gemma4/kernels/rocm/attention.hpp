#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::gemma4::rocm {

// Attention front-end and back-end for both layer classes: SWA layers
// (head_dim 256, 8 kv heads, window 1024, own V) and full layers
// (head_dim 512, 2 kv heads, V derived from K). The attention scale is 1.0
// (pinned by the oracle); softmax is max-subtracted.
//
// `inv_freq` is the per-class table [head_dim / 2] with the proportional
// rope_freqs factors already folded in (full layers) or nullptr-free plain
// theta^(-2i/d) values (SWA layers). Positions are contiguous starting at
// `start`. The KV caches are F16, position-major
// [position][kv_head][head_dim].

// Per-head RMSNorm(gamma) + NEOX rope of the Q projection.
// q/out: [tokens][heads][head_dim] F32.
void QknormRope(const float* q, const float* gamma, float* out,
                std::uint32_t tokens, std::uint32_t heads,
                std::uint32_t head_dim, const float* inv_freq,
                std::uint32_t start, hipStream_t stream);

// Per-head RMSNorm(k_gamma) + rope of K and weightless RMSNorm of V, written
// to the F16 caches at [start, start + tokens). `v == k` means V is the
// normed (pre-rope) K, as on full-attention layers.
void KvNormRopeWrite(const float* k, const float* v, const float* k_gamma,
                     __half* k_cache, __half* v_cache, std::uint32_t start,
                     std::uint32_t tokens, std::uint32_t kv_heads,
                     std::uint32_t head_dim, const float* inv_freq,
                     hipStream_t stream);

// Single-query GQA decode attention over cache positions [lo, n_keys).
// q: [heads][head_dim] F32, out: [heads][head_dim] F32.
// scratch: heads * splits * (head_dim + 2) floats (splits <= 32).
void AttentionDecode(const float* q, const __half* k_cache,
                     const __half* v_cache, float* out, float* scratch,
                     std::uint32_t n_keys, std::uint32_t lo,
                     std::uint32_t heads, std::uint32_t kv_heads,
                     std::uint32_t head_dim, hipStream_t stream);

// Causal windowed prefill attention over the F16 cache (the chunk's own K/V
// must already be written). q/out: [tokens][heads][head_dim] F32.
// `window == 0` is the exact dense pass.
void AttentionPrefill(const float* q, const __half* k_cache,
                      const __half* v_cache, float* out, std::uint32_t start,
                      std::uint32_t tokens, std::uint32_t heads,
                      std::uint32_t kv_heads, std::uint32_t head_dim,
                      std::uint32_t window, hipStream_t stream);

// Scalar tiled reference for the same contract. Kept as the numerical oracle
// for the WMMA fast path above; production uses AttentionPrefill.
void AttentionPrefillScalar(const float* q, const __half* k_cache,
                            const __half* v_cache, float* out,
                            std::uint32_t start, std::uint32_t tokens,
                            std::uint32_t heads, std::uint32_t kv_heads,
                            std::uint32_t head_dim, std::uint32_t window,
                            hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm